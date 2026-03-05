import numpy as np
from sumo_rl import SumoEnvironment
import gymnasium as gym
from gymnasium import spaces
import os
import csv
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig

class EpisodeRewardLogger(DefaultCallbacks):
    """
    Logs per-episode rewards for multi-agent envs.
    - Prints episode total reward (sum across agents)
    - Prints per-agent totals
    - Writes to CSV (optional)
    """
    def __init__(self, csv_path="training_log_ippo.csv"):
        super().__init__()
        self.csv_path = csv_path
        self._header_written = False

    def _ensure_header(self, agent_ids):
        if self._header_written:
            return
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["episode", "length", "total_reward"] + [f"{aid}_reward" for aid in agent_ids])
        self._header_written = True

    def on_episode_end(self, *, episode, env_runner=None, **kwargs):
        # episode.length is the number of env-steps in the episode
        length = episode.length

        # episode.total_reward is usually sum over all agents (works in most RLlib versions)
        total = float(getattr(episode, "total_reward", 0.0))

        # Per-agent totals: RLlib stores multi-agent rewards in episode.agent_rewards
        # Keys are typically (agent_id, policy_id) -> reward_sum
        per_agent = {}
        agent_rewards = getattr(episode, "agent_rewards", None)

        if isinstance(agent_rewards, dict):
            for key, r in agent_rewards.items():
                # key can be (agent_id, policy_id)
                if isinstance(key, tuple) and len(key) >= 1:
                    aid = key[0]
                else:
                    aid = str(key)
                per_agent[aid] = per_agent.get(aid, 0.0) + float(r)

        # Print
        # (If you have fixed agents, this prints a stable list each episode.)
        agent_ids = sorted(per_agent.keys())
        print(f"[EP] len={length} total_reward={total:.3f} " +
              " ".join([f"{aid}={per_agent[aid]:.3f}" for aid in agent_ids]))

        # CSV
        self._ensure_header(agent_ids)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([episode.episode_id, length, total] + [per_agent.get(aid, 0.0) for aid in agent_ids])

class RLLibSumoMAEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()  # IMPORTANT for RLlib new stack
        config = config or {}

        self.env = SumoEnvironment(
            net_file=config.get("net_file", "my_env.net.xml"),
            route_file=config.get("route_file", "routes_seed1.rou.xml"),
            use_gui=config.get("use_gui", False),
            num_seconds=config.get("num_seconds", 1000),
            delta_time=config.get("delta_time", 5),
            yellow_time=config.get("yellow_time", 3),
            min_green=config.get("min_green", 5),
            reward_fn=config.get("reward_fn", "diff-waiting-time"),
            sumo_seed=config.get("sumo_seed", "random"),
            enforce_max_green=config.get("enforce_max_green", True),
            max_green=config.get("max_green", 50),
            time_to_teleport=config.get("time_to_teleport", -1),
            max_depart_delay=config.get("max_depart_delay", 0),
        )

        obs = self.env.reset()
        self.agent_ids = list(obs.keys())

        # RLlib expects these MA base attributes in many places
        self.possible_agents = self.agent_ids
        self.agents = list(self.agent_ids)

        # In your sumo_rl version these are already Space objects
        per_obs_space = self.env.observation_space
        per_act_space = self.env.action_space

        # NEW STACK: expose MA spaces as Dict(agent_id -> space)
        self.observation_space = spaces.Dict({aid: per_obs_space for aid in self.agent_ids})
        self.action_space = spaces.Dict({aid: per_act_space for aid in self.agent_ids})

        # Keep these too (handy for policies dict)
        self._obs_space = per_obs_space
        self._act_space = per_act_space

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        self.agents = list(self.agent_ids)  # active agents this episode
        infos = {aid: {} for aid in obs.keys()}
        return obs, infos

    def step(self, action_dict):
        obs, rewards, dones, infos = self.env.step(action_dict)

        # --- Fix infos to only include agent IDs (RLlib requirement) ---
        # If SUMO-RL returns extra top-level info keys, move them to __common__
        common_info = {k: v for k, v in infos.items() if k not in self.agent_ids}
        agent_infos = {aid: infos.get(aid, {}) for aid in self.agent_ids}

        if common_info:
            agent_infos["__common__"] = common_info  # safe place for system metrics

        infos = agent_infos
        # -------------------------------------------------------------

        terminateds = {aid: bool(dones.get(aid, False)) for aid in self.agent_ids}
        terminateds["__all__"] = bool(dones.get("__all__", False))

        truncateds = {aid: False for aid in self.agent_ids}
        truncateds["__all__"] = False

        return obs, rewards, terminateds, truncateds, infos

    def close(self):
        self.env.close()

    def observation_space_sample(self):
        return self._obs_space

    def action_space_sample(self):
        return self._act_space


if __name__ == "__main__":
    # Build env once to get agent IDs and spaces
    tmp_env = None
    try:
        tmp_env = RLLibSumoMAEnv(
            {
                "net_file": "my_env.net.xml",
                "route_file": "routes_seed1.rou.xml",
            }
        )
        agent_ids = tmp_env.agent_ids
        obs_space = tmp_env._obs_space
        act_space = tmp_env._act_space
    finally:
        if tmp_env is not None:
            tmp_env.close()

    # IPPO = one PPO policy per agent_id
    policies = {
        aid: (None, obs_space, act_space, {})  # (policy_class, obs_space, act_space, config)
        for aid in agent_ids
    }

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return agent_id  # each agent uses its own policy

    config = (
        PPOConfig()
        .environment(
            env=RLLibSumoMAEnv,
            env_config={
                "net_file": "my_env.net.xml",
                "route_file": "routes_seed1.rou.xml",
                "use_gui": False,
                "num_seconds": 1000,
                "delta_time": 5,
                "yellow_time": 3,
                "min_green": 5,
                "reward_fn": "diff-waiting-time",
                "sumo_seed": "random",
                "enforce_max_green": True,
                "max_green": 50,
                "time_to_teleport": -1,
                "max_depart_delay": 0,
            },
        )
        .framework("torch")
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .api_stack(enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False)
        .env_runners(num_env_runners=0)
        .callbacks(lambda: EpisodeRewardLogger(csv_path="training_log_ippo.csv"))
        .training(
            gamma=0.99,
            train_batch_size=4096,
            minibatch_size=256,
            num_epochs=10,
            lr=3e-4,
        )
    )

    algo = config.build_algo()

    # Train
    for i in range(50):  # iterations; each iteration collects train_batch_size steps (across agents)
        result = algo.train()
        print(
            f"iter={i} "
            f"episode_reward_mean={result.get('episode_reward_mean')} "
            f"timesteps_total={result.get('timesteps_total')}"
        )

        if (i + 1) % 10 == 0:
            ckpt = algo.save()
            print("checkpoint saved at:", ckpt)

    final_ckpt = algo.save()
    print("Final checkpoint:", final_ckpt)
    algo.stop()