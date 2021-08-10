import argparse
import torch

from omrl.utils.config import Config
from omrl.builder import build_agent, build_env
from omrl.run import train_and_evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--config", help="train config file path")
    parser.add_argument('--train_url', help='the dir to save logs and models')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    env = build_env(cfg.dict["env"])
    env_dims = {"state_dim": env.state_dim, "action_dim": env.action_dim}
    cfg_args = cfg.dict["train_args"]
    if args.train_url is not None:
        # update configs according to CLI args if args.train_url is not None
        cfg_args["train_url"] = args.train_url

    agent = build_agent(cfg.dict["agent"], default_args={**cfg_args, **env_dims})
    train_and_evaluate(cfg_args, agent, env)


if __name__ == "__main__":
    main()
