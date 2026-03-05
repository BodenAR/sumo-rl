import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "training_log_ippo.csv"
OUT_DIR = "plots"
ROLLING_WINDOW = 20  # smoothing in episodes

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Ensure numeric columns
for col in ["length", "total_reward", "J15_reward", "J16_reward", "J17_reward", "J18_reward"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop bad rows (if any)
df = df.dropna(subset=["total_reward"]).reset_index(drop=True)

# Use row index as the true episode counter (since episode column is weird)
df["episode_num"] = df.index + 1

# Rolling mean for readability
df["total_smooth"] = df["total_reward"].rolling(ROLLING_WINDOW, min_periods=1).mean()

# ---------- Plot 1: Total reward ----------
plt.figure(figsize=(12, 6))
plt.plot(df["episode_num"], df["total_reward"], linewidth=1, alpha=0.35, label="Total reward (raw)")
plt.plot(df["episode_num"], df["total_smooth"], linewidth=2.5, label=f"Total reward (rolling mean, window={ROLLING_WINDOW})")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("IPPO Training: Total Episode Reward")
plt.grid(True, alpha=0.3)
plt.legend()
out1 = os.path.join(OUT_DIR, "ippo_total_reward.png")
plt.tight_layout()
plt.savefig(out1, dpi=200)
plt.show()
print(f"Saved: {out1}")

# ---------- Plot 2: Per-agent rewards (smoothed) ----------
plt.figure(figsize=(12, 6))
for col in ["J15_reward", "J16_reward", "J17_reward", "J18_reward"]:
    smooth = df[col].rolling(ROLLING_WINDOW, min_periods=1).mean()
    plt.plot(df["episode_num"], smooth, linewidth=2, label=f"{col} (smoothed)")

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("IPPO Training: Per-Agent Reward (Smoothed)")
plt.grid(True, alpha=0.3)
plt.legend()
out2 = os.path.join(OUT_DIR, "ippo_per_agent_reward.png")
plt.tight_layout()
plt.savefig(out2, dpi=200)
plt.show()
print(f"Saved: {out2}")

# ---------- Optional: Episode length ----------
# Useful to confirm episodes are consistent (yours are 200)
plt.figure(figsize=(12, 4))
plt.plot(df["episode_num"], df["length"], linewidth=1.5)
plt.xlabel("Episode")
plt.ylabel("Length (env steps)")
plt.title("Episode Length Over Time")
plt.grid(True, alpha=0.3)
out3 = os.path.join(OUT_DIR, "ippo_episode_length.png")
plt.tight_layout()
plt.savefig(out3, dpi=200)
plt.show()
print(f"Saved: {out3}")