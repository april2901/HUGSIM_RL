import argparse
import os

from omegaconf import OmegaConf
from stable_baselines3 import PPO

from sim.utils.rl_attack_env import SimpleAttackEnv


def load_reward_config(path):
    if path is None:
        return {}, {}
    cfg = OmegaConf.load(path)
    return (
        OmegaConf.to_container(cfg.get("weights", {}), resolve=True),
        OmegaConf.to_container(cfg.get("params", {}), resolve=True),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reward_cfg",
        type=str,
        default="./configs/rl/reward.yaml",
        help="YAML with weights (approach/collision) and params (scales, bonuses).",
    )
    parser.add_argument("--timesteps", type=int, default=30000)
    parser.add_argument("--save_path", type=str, default="simple_attacker_ppo")
    args = parser.parse_args()

    reward_weights, reward_params = load_reward_config(args.reward_cfg)
    env = SimpleAttackEnv(
        reward_weights=reward_weights,
        reward_params=reward_params,
    )

    print(f"reward_cfg: {os.path.abspath(args.reward_cfg)}")
    print(f"weights: {env.reward_weights}")
    print(f"params: {env.cfg}")

    model = PPO("MlpPolicy", env, verbose=1, device="cuda")
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save_path)
    print("Saved:", f"{args.save_path}.zip")


if __name__ == "__main__":
    main()
