import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO


class SharedSumoMAEnv(gym.Env):
    """
    Single-policy cooperative wrapper:
    - action: one discrete action per intersection (MultiDiscrete)
    - obs: concatenation of all intersection observations
    - reward: sum of all intersection rewards (cooperative)
    Adds:
    - episode logging (print + CSV)
    """
    def __init__(self, log_csv_path="training_log.csv"):
        super().__init__()
        self.env = SumoEnvironment(
            net_file="my_env.net.xml",
            route_file="routes1.rou.xml",
            use_gui=False,
            num_seconds=1000,
            delta_time=5, # 1000/5 = RL steps 
            yellow_time=3, 
            min_green=5,
            reward_fn="diff-waiting-time",
            sumo_seed="random",

            enforce_max_green=True,
            max_green=50, # Avoid bad training where a light learns to just be constant green

            time_to_teleport=-1, # No teleporting when it gets too congested
            max_depart_delay=0, # ..
            #additional_sumo_cmd="--tripinfo-output tripinfo.xml",

        )
        obs = self.env.reset()
        self.agent_ids = list(obs.keys())
        self.n_agents = len(self.agent_ids)

        # Infer dimensions
        sample = np.array(obs[self.agent_ids[0]], dtype=np.float32).reshape(-1)
        self.per_dim = sample.shape[0]

        n_actions = self.env.action_space.n  # usually 2
        self.action_space = spaces.MultiDiscrete([n_actions] * self.n_agents)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_agents * self.per_dim,),
            dtype=np.float32
        )

        self._last_obs = obs

        # Episode logging
        self.episode_idx = 0
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.log_csv_path = log_csv_path

        # Write CSV header once (if file doesn't exist)
        try:
            with open(self.log_csv_path, "x") as f:
                f.write("episode,total_reward,steps\n")
        except FileExistsError:
            pass

    def _flat(self, obs_dict):
        return np.concatenate(
            [np.array(obs_dict[a], dtype=np.float32).reshape(-1) for a in self.agent_ids],
            axis=0
        )

    def reset(self, seed=None, options=None):
        # On reset, start a fresh SUMO episode
        obs = self.env.reset()
        self._last_obs = obs

        # Reset episode counters
        self.episode_reward = 0.0
        self.episode_steps = 0

        return self._flat(obs), {}

    def step(self, action):
        actions = {aid: int(action[i]) for i, aid in enumerate(self.agent_ids)}
        obs, rewards, dones, infos = self.env.step(actions)

        self._last_obs = obs

        # Cooperative reward
        reward = float(sum(rewards.values()))

        # Episode bookkeeping
        self.episode_reward += reward
        self.episode_steps += 1

        terminated = bool(dones.get("__all__", False))
        truncated = False

        # If episode ended, log it
        if terminated:
            self.episode_idx += 1
            print(
                f"Episode {self.episode_idx} finished: "
                f"steps={self.episode_steps}, total_reward={self.episode_reward:.3f}"
            )
            with open(self.log_csv_path, "a") as f:
                f.write(f"{self.episode_idx},{self.episode_reward},{self.episode_steps}\n")

        info = {
            "per_agent_rewards": rewards,
            "episode_total_reward": self.episode_reward,
            "episode_steps": self.episode_steps,
        }
        return self._flat(obs), reward, terminated, truncated, info

    def close(self):
        self.env.close()


if __name__ == "__main__":
    env = SharedSumoMAEnv(log_csv_path="training_log_v3.csv")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1, # basic logging/training infp
        n_steps=2048, # Collect this many env steps before doing PPO update (~10 episodes)
        batch_size=256, # 2048/256 = 8 mini batches
        gamma=0.99, # discount factor (0.99 good for long-term)
        tensorboard_log="./ppo_logs/",   # <--- TensorBoard logs
    )

    model.learn(total_timesteps=150_000) # divide by RL steps to get number of episodes
    model.save("shared_ppo_tls_v3")
    env.close()
    print("Saved: shared_ppo_tls_v3.zip")
    print("CSV log: training_log.csv")
    print("TensorBoard: run `tensorboard --logdir ppo_logs`")

