import numpy as np
from gymnasium import spaces
from sumo_rl import SumoEnvironment
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig


class RLLibSumoMAEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}

        self.env = SumoEnvironment(
            net_file=config.get("net_file", "my_env.net.xml"),
            route_file=config.get("route_file", "routes_seed1.rou.xml"),
            use_gui=config.get("use_gui", True),
            num_seconds=config.get("num_seconds", 1000),
            delta_time=config.get("delta_time", 5),
            yellow_time=config.get("yellow_time", 3),
            min_green=config.get("min_green", 5),
            reward_fn=config.get("reward_fn", "diff-waiting-time"),
            sumo_seed=config.get("sumo_seed", 1),
            enforce_max_green=config.get("enforce_max_green", True),
            max_green=config.get("max_green", 50),
            time_to_teleport=config.get("time_to_teleport", -1),
            max_depart_delay=config.get("max_depart_delay", 0),
        )

        obs = self.env.reset()
        self.agent_ids = list(obs.keys())
        self.possible_agents = self.agent_ids
        self.agents = list(self.agent_ids)

        per_obs_space = self.env.observation_space
        per_act_space = self.env.action_space

        self.observation_space = spaces.Dict(
            {aid: per_obs_space for aid in self.agent_ids}
        )
        self.action_space = spaces.Dict(
            {aid: per_act_space for aid in self.agent_ids}
        )

        self._obs_space = per_obs_space
        self._act_space = per_act_space

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        self.agents = list(self.agent_ids)
        infos = {aid: {} for aid in obs.keys()}
        return obs, infos

    def step(self, action_dict):
        obs, rewards, dones, infos = self.env.step(action_dict)

        common_info = {k: v for k, v in infos.items() if k not in self.agent_ids}
        agent_infos = {aid: infos.get(aid, {}) for aid in self.agent_ids}
        if common_info:
            agent_infos["__common__"] = common_info

        terminateds = {aid: bool(dones.get(aid, False)) for aid in self.agent_ids}
        terminateds["__all__"] = bool(dones.get("__all__", False))

        truncateds = {aid: False for aid in self.agent_ids}
        truncateds["__all__"] = False

        return obs, rewards, terminateds, truncateds, agent_infos

    def close(self):
        self.env.close()


def maybe_attack_observation(obs, agent_id, attack_cfg):
    attack_type = attack_cfg["type"]
    victim_agent = attack_cfg["victim_agent"]
    attack_prob = attack_cfg["attack_prob"]
    obs_low = attack_cfg["obs_low"]
    obs_high = attack_cfg["obs_high"]

    if attack_type != "obs_noise":
        return obs, False

    if agent_id != victim_agent:
        return obs, False

    if np.random.rand() > attack_prob:
        return obs, False

    sigma = attack_cfg["obs_noise_sigma"]
    attacked = obs + np.random.normal(0.0, sigma, size=obs.shape)
    attacked = np.clip(attacked, obs_low, obs_high).astype(obs.dtype, copy=False)
    return attacked, True


def maybe_attack_action(action, env, agent_id, attack_cfg):
    attack_type = attack_cfg["type"]
    victim_agent = attack_cfg["victim_agent"]
    attack_prob = attack_cfg["attack_prob"]

    if attack_type not in {"random_action", "opposite_action"}:
        return action, False

    if agent_id != victim_agent:
        return action, False

    if np.random.rand() > attack_prob:
        return action, False

    action_space = env.action_space.spaces[agent_id]

    if attack_type == "random_action":
        return action_space.sample(), True

    # "opposite_action" baseline:
    # choose any action different from the policy's chosen action.
    if hasattr(action_space, "n"):
        valid_actions = list(range(action_space.n))
        alternatives = [a for a in valid_actions if a != int(action)]
        if not alternatives:
            return action, False
        return int(np.random.choice(alternatives)), True

    return action, False


def main():
    # ------------------------------------------------------------------
    # EDIT THESE SETTINGS
    # ------------------------------------------------------------------
    checkpoint_path = "/home/vbodvik/N3Project/sumo-rl/my_env/ippo_checkpoint"

    env_config = {
        "net_file": "my_env.net.xml",
        "route_file": "routes_seed2.rou.xml",
        "use_gui": True,
        "num_seconds": 1000,
        "delta_time": 5,
        "yellow_time": 3,
        "min_green": 5,
        "reward_fn": "diff-waiting-time",
        "sumo_seed": 1,
        "enforce_max_green": True,
        "max_green": 50,
        "time_to_teleport": -1,
        "max_depart_delay": 0,
    }

    attack_cfg = {
        # type: "none", "obs_noise", "random_action", "opposite_action"
        "type": "obs_noise",
        "victim_agent": "J15",
        "attack_prob": 1.0,       # probability attack triggers at each victim step, def 1.0
        "obs_noise_sigma": 1.2,  # used only for obs_noise, 0.9 proved to worsen the results quite a bit
        "obs_low": 0.0,
        "obs_high": 1.0,
    }
    # ------------------------------------------------------------------

    tmp_env = None
    try:
        tmp_env = RLLibSumoMAEnv(env_config)
        agent_ids = tmp_env.agent_ids
        obs_space = tmp_env._obs_space
        act_space = tmp_env._act_space
    finally:
        if tmp_env is not None:
            tmp_env.close()

    policies = {
        aid: (None, obs_space, act_space, {})
        for aid in agent_ids
    }

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return agent_id

    config = (
        PPOConfig()
        .environment(env=RLLibSumoMAEnv, env_config=env_config)
        .framework("torch")
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .env_runners(num_env_runners=0)
    )

    algo = config.build_algo()
    algo.restore(checkpoint_path)

    env = RLLibSumoMAEnv(env_config)

    try:
        obs, infos = env.reset()
        terminateds = {"__all__": False}
        truncateds = {"__all__": False}

        total_rewards = {aid: 0.0 for aid in agent_ids}
        step_count = 0
        attacked_obs_count = 0
        attacked_action_count = 0

        while not terminateds["__all__"] and not truncateds["__all__"]:
            actions = {}

            for agent_id, agent_obs in obs.items():
                obs_for_policy, obs_attacked = maybe_attack_observation(
                    agent_obs, agent_id, attack_cfg
                )
                if obs_attacked:
                    attacked_obs_count += 1

                action = algo.compute_single_action(
                    observation=obs_for_policy,
                    policy_id=agent_id,
                    explore=False,
                )

                action, action_attacked = maybe_attack_action(
                    action, env, agent_id, attack_cfg
                )
                if action_attacked:
                    attacked_action_count += 1

                actions[agent_id] = action

            obs, rewards, terminateds, truncateds, infos = env.step(actions)

            for aid, r in rewards.items():
                total_rewards[aid] += float(r)

            step_count += 1

        print("\nEvaluation finished")
        print("Episode length:", step_count)
        print("Attack config:", attack_cfg)
        print("Attacked observations:", attacked_obs_count)
        print("Attacked actions:", attacked_action_count)
        print("Per-agent rewards:")
        for aid in agent_ids:
            print(f"  {aid}: {total_rewards[aid]:.6f}")
        print("Total reward:", sum(total_rewards.values()))

    finally:
        env.close()
        algo.stop()


if __name__ == "__main__":
    main()