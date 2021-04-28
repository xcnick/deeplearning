import argparse
import torch

from omrl.utils.config import Config
from omrl.builder import build_agent, build_env
from omrl.run import get_video_to_watch_gym_render


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--config", help="train config file path")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    env = build_env(cfg.dict["env"])
    env_dims = {"state_dim": env.state_dim, "action_dim": env.action_dim}
    agent = build_agent(cfg.dict["agent"], default_args={**env_dims})
    get_video_to_watch_gym_render(agent, env)


if __name__ == "__main__":
    main()
