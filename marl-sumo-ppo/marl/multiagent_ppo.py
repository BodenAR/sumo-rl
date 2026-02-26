import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from sumo_rl import SumoEnvironment
from shared_policy_wrapper import SharedPolicyWrapper


def train_ps_ppo():

    env = SumoEnvironment(
        net_file="../net/network.net.xml",
        route_file="../net/routes.rou.xml",
        out_csv_name="../logs/marl_ppo",
        use_gui=False,
        num_seconds=1000,
        delta_time=5,
        min_green=5,
        yellow_time=3,
        reward_fn="diff-waiting-time",
        enforce_max_green=True,
        max_green=50,
        time_to_teleport=-1,
        max_depart_delay=0,
    )

    # Wrap in shared policy MARL wrapper
    env = SharedPolicyWrapper(env)


    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
        device="cpu",  # <— recommended for MLP
    )


    # Logging
    log_dir = "../logs/ppo_runs"
    os.makedirs(log_dir, exist_ok=True)
    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    print("🚦 Training shared-policy PPO for multi-agent traffic lights...")
    model.learn(total_timesteps=300_000)
    model.save("../logs/ppo_policy.zip")

    print("Done! Model saved.")


if __name__ == "__main__":
    train_ps_ppo()