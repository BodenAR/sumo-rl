import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_log.csv")
df.columns = df.columns.str.strip()

df["moving_avg"] = df["total_reward"].rolling(20).mean()

plt.plot(df["total_reward"], alpha=0.3, label="reward")
plt.plot(df["moving_avg"], label="moving average")
plt.legend()
plt.title("Training Reward Trend")

plt.savefig("reward_plot.png", dpi=150)
print("Saved plot to reward_plot.png")