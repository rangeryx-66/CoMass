"""Bounded policy evaluation with actual first-floor-contact errors and optional video."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from .config import load_config
from .env import ThrowEnv
from .policy import ActorCritic, RunningMeanStd


def evaluate(checkpoint, config, episodes=20, output="evaluation", video=False, seed=1000):
    torch.set_num_threads(1)
    env = ThrowEnv(config["env"])
    device = torch.device(config["device"])
    model = ActorCritic(env.observation_space.shape[0], config["ppo"]["hidden"]).to(device)
    normalizer = RunningMeanStd(env.observation_space.shape[0]).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    normalizer.load_state_dict(state["normalizer"])
    model.eval()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for episode in range(episodes):
            obs, reset_info = env.reset(seed=seed + episode)
            writer = imageio.get_writer(out / f"episode_{episode:03d}.mp4", fps=round(1 / (env.config["timestep"] * env.config["substeps"]))) if video and episode == 0 else None
            trajectory = []
            total = 0.0
            try:
                while True:
                    with torch.no_grad():
                        tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                        action = model.act(normalizer(tensor), deterministic=True)[0][0].cpu().numpy()
                    obs, reward, terminated, truncated, info = env.step(action)
                    total += reward
                    trajectory.append(np.r_[env.data.time - env.start_time, env.data.qpos, env.data.qvel, action, env.data.ctrl])
                    if writer:
                        writer.append_data(env.render())
                    if terminated or truncated:
                        break
            finally:
                if writer:
                    writer.close()
            result = {"episode": episode, "seed": seed + episode, "return": total, **info, "target": info["target"].tolist(), "object": json.loads(reset_info["object_json"]), "release": env.release, "landing": env.landing}
            if not np.isfinite(result["landing_error"]):
                result["landing_error"] = None
            if result["release"] and result["landing"]:
                release = np.array(result["release"]["position"])
                landing = np.array(result["landing"]["position"])
                result["flight_horizontal_distance"] = float(np.linalg.norm(landing[:2] - release[:2]))
                result["flight_com_horizontal_distance"] = float(np.linalg.norm(
                    np.asarray(result["landing"]["com_position"])[:2] - np.asarray(result["release"]["com_position"])[:2]))
                result["flight_time"] = float(env.data.time - env.start_time - result["release"]["time"])
            results.append(result)
            np.savez_compressed(out / f"episode_{episode:03d}.npz", trajectory=np.asarray(trajectory), nq=env.model.nq, nv=env.model.nv, nu=env.model.nu)
            print(json.dumps(result), flush=True)
    finally:
        env.close()
    errors = [r["landing"]["error"] for r in results if r["landing"]]
    summary = {"episodes": episodes, "success_rate": float(np.mean([r["success"] for r in results])), "mean_landing_error_m": float(np.mean(errors)) if errors else None, "release_rate": float(np.mean([r["release"] is not None for r in results])), "initial_bilateral_rate": float(np.mean([r["initialization"]["bilateral_contact"] for r in results])), "results": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", default="evaluation")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    if args.config:
        config = load_config(args.config, args.set)
    else:
        config = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["config"]
        if args.set:
            parser.error("--set requires --config")
    summary = evaluate(args.checkpoint, config, args.episodes, args.output, args.video, args.seed)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))


if __name__ == "__main__":
    main()
