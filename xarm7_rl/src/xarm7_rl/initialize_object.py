"""Initialize fixed object-feature statistics when reusing a throwing skill for RL."""
import argparse
from pathlib import Path

import torch

from .config import load_config
from .env import ThrowEnv


def calibrate_static_features(normalizer_state, observation, geometry_points):
    # Observation 32:42 is mass, CoM and inertia; the next 3*K are fixed
    # body-frame surface points. Never calibrate runtime state or test objects.
    end = 42 + 3 * geometry_points
    normalizer_state["mean"][32:end] = torch.as_tensor(observation[32:end], dtype=torch.float64)
    normalizer_state["var"][32:end] = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    env = ThrowEnv(config["env"])
    try:
        observation, _ = env.reset(seed=config["seed"])
    finally:
        env.close()
    calibrate_static_features(state["normalizer"], observation, config["env"]["geometry_points"])
    state["config"] = config
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)


if __name__ == "__main__":
    main()
