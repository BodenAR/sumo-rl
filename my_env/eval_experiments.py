import os
import numpy as np
from train_ippo_rllib import RLLibSumoMAEnv, EpisodeRewardLogger
from ray.rllib.algorithms.ppo import PPOConfig
import csv

def load_policies(agent_ids, obs_space, act_space):
    return {
        aid: (None, obs_space, act_space, {})
        for aid in agent_ids
    }

def policy_mapping_fn(agent_id, *args, **kwargs):
    return agent_id  # same as in training

def evaluate_checkpoint(checkpoint_path, episodes=5):
    # --- Build temp env to recover agent IDs and spaces ---
    tmp_env = RLLibSumoMAEnv({
        "net_file": "my_env.net.xml",
        "route_file": "routes_seed1.rou.xml",
    })
    agent_ids = tmp_env.agent_ids
    obs_space = tmp_env._obs_space
    act_space = tmp_env._act_space
    tmp_env.close()

    # Recreate policies
    policies = load_policies(agent_ids, obs_space, act_space)

    # Rebuild PPO algorithm
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
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
    )

    algo = config.build_algo()
    algo.restore(checkpoint_path)

    print("Loaded checkpoint:", checkpoint_path)

    # --- Evaluation loop ---
    env = RLLibSumoMAEnv({
        "net_file": "my_env.net.xml",
        "route_file": "routes_seed2.rou.xml",
        "use_gui": True,   # You can turn GUI ON for visual debugging
    })

    for ep in range(episodes):
        obs, infos = env.reset()
        done = {aid: False for aid in agent_ids}
        done["__all__"] = False

        ep_return = {aid: 0.0 for aid in agent_ids}

        while not done["__all__"]:
            actions = {}

            for aid in agent_ids:
                if not done[aid]:

                    # ------ Modifiy obs_vec ------
                    # Multiplicative noice (to weak)
                    # eps = 1
                    # ones_part = np.ones(5)
                    # noise = 1 + np.random.uniform(-eps, eps, size=21-5)
                    # mult = np.append(ones_part, noise)
                    # obs_noise = np.clip(obs[aid] * mult, 0.0, 1.0)
                    # print(obs_noise, obs[aid])

                    # Clipped gausian noise
                    std = 0.2
                    ones_part = np.ones(5)
                    noise = np.multiply(np.random.normal(0, std, size=21-5),5)
                    add = np.append(ones_part, noise)
                    obs_noise = np.clip(obs[aid] + add, 0.0, 1.0)

                    # random_part_1 = np.random.randint(2, size=5)
                    # random_part_2 = np.random.rand(21-5)
                    # obs_noise = np.append(random_part_1, random_part_2)

                    actions[aid] = algo.compute_single_action(
                        obs_noise,
                        policy_id=aid,       # Important for IPPO!
                        explore=False        # Deterministic evaluation
                    )
            # ------ Modify act_vec ------
            # actions = {'J15': np.random.randint(4), 'J16': np.random.randint(4),'J17': np.random.randint(4),'J18': np.random.randint(4)}
            obs, rewards, terminated, truncated, infos = env.step(actions)

            for aid, r in rewards.items():
                ep_return[aid] += r

            done = {aid: terminated[aid] or truncated[aid] for aid in agent_ids}
            done["__all__"] = terminated["__all__"]

        print(f"\nEpisode {ep} returns:")
        for aid in agent_ids:
            print(f"  {aid}: {ep_return[aid]:.3f}")

    env.close()

if __name__ == "__main__":
    checkpoint_path = "/tmp/tmpyp8xjtdp"  # <--- REPLACE WITH YOUR PATH
    evaluate_checkpoint(checkpoint_path, episodes=3)