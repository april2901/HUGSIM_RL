from stable_baselines3 import PPO
from sim.utils.rl_attack_env import SimpleAttackEnv

if __name__ == "__main__":
    env = SimpleAttackEnv()
    model = PPO("MlpPolicy", env, verbose=1, device="cuda")
    model.learn(total_timesteps=30000)
    model.save("simple_attacker_ppo")
    print("Saved:", "simple_attacker_ppo.zip")