import loco_mujoco
import gymnasium as gym

# 1. Initialize the Musculoskeletal Environment (Now called MyoSkeleton!)
env = gym.make("LocoMujoco", env_name="MyoSkeleton.walk")

# 2. Let's look at the brain of the agent!
print(f"Observation Space (What the agent sees): {env.observation_space.shape}")
print(f"Action Space (What the agent controls): {env.action_space.shape}")
print(f"Action Limits (The Clamp): {env.action_space.low[0]} to {env.action_space.high[0]}")