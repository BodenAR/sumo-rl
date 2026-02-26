import gymnasium as gym
import numpy as np
import warnings

def _to_array(obs):
    """Convert array or dict of arrays into 1D float32 vector."""
    if isinstance(obs, dict):
        parts = []
        for v in obs.values():
            parts.append(np.asarray(v, dtype=np.float32).ravel())
        return np.concatenate(parts) if parts else np.array([], dtype=np.float32)
    return np.asarray(obs, dtype=np.float32).ravel()

def _get_agent_ids(base_env):
    """
    Try multiple attribute names to retrieve traffic-signal agent IDs across sumo-rl versions.
    """
    for name in ("agents", "ts_ids", "tls_ids", "tlsIDs"):
        ids = getattr(base_env, name, None)
        if isinstance(ids, (list, tuple)) and all(isinstance(x, str) for x in ids):
            return list(ids)
    ts_dict = getattr(base_env, "traffic_signals", None)
    if isinstance(ts_dict, dict) and all(isinstance(k, str) for k in ts_dict.keys()):
        return list(ts_dict.keys())

    # Final fallback: infer from reset() if it returns a dict
    try:
        result = base_env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, _info = result
        else:
            obs = result
        if isinstance(obs, dict):
            return list(obs.keys())
    except Exception:
        pass

    raise AttributeError(
        "Could not determine agent IDs. Your sumo-rl version doesn’t expose "
        "any of: env.agents / env.ts_ids / env.tls_ids / env.traffic_signals."
    )

def _get_action_dims(base_env, agent_ids):
    """
    Return a list with action space size per agent:
    - If env.action_space(agent_id) exists → per-agent sizes.
    - Else if env.action_space is a Discrete → same size for all agents.
    - Else attempt from traffic_signal attributes; fallback to 2.
    """
    # Per-agent callable space?
    if hasattr(base_env, "action_space") and callable(base_env.action_space):
        try:
            dims = [int(base_env.action_space(a).n) for a in agent_ids]
            if all(d > 0 for d in dims):
                return dims
        except Exception:
            pass

    # Single global action space?
    global_as = getattr(base_env, "action_space", None)
    if global_as is not None and not callable(global_as) and hasattr(global_as, "n"):
        return [int(global_as.n)] * len(agent_ids)

    # Try traffic_signals dict
    ts_dict = getattr(base_env, "traffic_signals", None)
    if isinstance(ts_dict, dict):
        dims = []
        for a in agent_ids:
            ts = ts_dict.get(a)
            for attr in ("green_phases", "phases"):
                if hasattr(ts, attr):
                    try:
                        dims.append(len(getattr(ts, attr)))
                        break
                    except Exception:
                        pass
            else:
                dims.append(2)
        return dims

    warnings.warn(
        "Could not detect per-agent action sizes. Falling back to 2 actions per agent."
    )
    return [2] * len(agent_ids)

class SharedPolicyWrapper(gym.Env):
    """
    Parameter-sharing PPO wrapper compatible with both NEW (Gymnasium, 5-return)
    and OLD (Gym, 4-return) sumo-rl APIs.

    - Observation: concatenation of all agents' observations.
    - Action: MultiDiscrete, one discrete action per agent.
    - Reward: sum of per-agent rewards.
    - Done:
        * Gymnasium: terminated OR truncated (dicts)
        * Gym: single 'done' (bool or dict) → map to terminated, truncated=False
    """
    metadata = {"render_modes": []}

    def __init__(self, base_env):
        super().__init__()
        self.base_env = base_env

        # Discover agent IDs across API variants
        self.agents = _get_agent_ids(base_env)

        # Probe observation to size observation_space
        try:
            result = base_env.reset()
        except Exception as e:
            raise RuntimeError(f"SUMO env failed to reset. Underlying error: {e}")

        if isinstance(result, tuple) and len(result) == 2:
            sample_obs_dict, _info = result
        else:
            sample_obs_dict = result

        if not isinstance(sample_obs_dict, dict):
            sample_obs_dict = {a: sample_obs_dict for a in self.agents}

        flat_sample = self._flatten_obs(sample_obs_dict)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=flat_sample.shape, dtype=np.float32
        )

        # Build MultiDiscrete action space
        self.action_dims = _get_action_dims(base_env, self.agents)
        self.action_space = gym.spaces.MultiDiscrete(self.action_dims)

    def _flatten_obs(self, obs_dict):
        vectors = []
        for a in self.agents:  # keep agent order stable
            v = obs_dict.get(a)
            if v is None:
                v = np.zeros(1, dtype=np.float32)
            vectors.append(_to_array(v))
        return np.concatenate(vectors, dtype=np.float32)

    def reset(self, **kwargs):
        result = self.base_env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs_dict, base_info = result
        else:
            obs_dict, base_info = result, {}
        if not isinstance(obs_dict, dict):
            obs_dict = {a: obs_dict for a in self.agents}
        return self._flatten_obs(obs_dict), base_info

    def step(self, action_vec):
        # Vector -> dict
        actions = {agent: int(action_vec[i]) for i, agent in enumerate(self.agents)}

        result = self.base_env.step(actions)

        # Handle Gymnasium (5 returns) vs Gym (4 returns)
        if isinstance(result, tuple) and len(result) == 5:
            next_obs, rewards, terminated, truncated, info = result
            if not isinstance(next_obs, dict):
                next_obs = {a: next_obs for a in self.agents}
            if not isinstance(rewards, dict):
                rewards = {a: float(rewards) for a in self.agents}
            if not isinstance(terminated, dict):
                terminated = {a: bool(terminated) for a in self.agents}
            if not isinstance(truncated, dict):
                truncated = {a: bool(truncated) for a in self.agents}
            done_any = any(terminated.values()) or any(truncated.values())

        elif isinstance(result, tuple) and len(result) == 4:
            next_obs, rewards, done, info = result
            if not isinstance(next_obs, dict):
                next_obs = {a: next_obs for a in self.agents}
            if not isinstance(rewards, dict):
                rewards = {a: float(rewards) for a in self.agents}

            # Map old 'done' to Gymnasium-style terminated/truncated
            if isinstance(done, dict):
                terminated = {a: bool(done.get(a, False)) for a in self.agents}
                truncated = {a: False for a in self.agents}
                done_any = any(terminated.values())
            else:
                done_any = bool(done)
                terminated = {a: done_any for a in self.agents}
                truncated = {a: False for a in self.agents}
        else:
            raise RuntimeError(f"Unexpected number of items from base_env.step(): {type(result)} / len={len(result) if isinstance(result, tuple) else 'n/a'}")

        reward_scalar = float(sum(rewards.values()))
        next_state = self._flatten_obs(next_obs)

        # Return Gymnasium 5-tuple
        return next_state, reward_scalar, done_any, done_any, info