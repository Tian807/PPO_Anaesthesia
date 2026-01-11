from stable_baselines3 import PPO
from patient_env import AnesthesiaEnv

# 1. Setup Environment
env = AnesthesiaEnv()

# 2. Load the OLD model (the one you already have)
# Note: We need to pass the new env to it so it can keep interacting
model = PPO.load("ppo_anesthesia_agent", env=env)

print("Resuming training for another 100,000 steps...")

# 3. Train MORE (This adds to the previous knowledge)
model.learn(total_timesteps=100000)

# 4. Save the NEW, smarter model
model.save("ppo_anesthesia_agent_v2")
print("Done! Saved as v2.")