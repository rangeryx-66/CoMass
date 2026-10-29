"""Record fixed evaluation seeds, with readable metadata and optional slow replay."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import load_config
from .evaluate import evaluate


def font(size):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def caption(frame, result, dt, mode):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    spec = result["object"]
    error = result["landing_error"]
    target = ", ".join(f"{x:.2f}" for x in result["target"])
    com = ", ".join(f"{1000*x:.1f}" for x in spec["com"])
    draw.rectangle((0, 0, image.width, 103), fill=(12, 16, 22))
    draw.text((20, 10), f"xArm7 + native gripper | {spec['type']} {1000*spec['mass']:.1f} g | CoM [{com}] mm", font=font(23), fill="white")
    draw.text((20, 43), f"Target [{target}] m | {'HIT' if result['success'] else 'MISS'} | error {100*error:.2f} cm", font=font(23), fill="white")
    distance = result.get("flight_com_horizontal_distance")
    text = f"MuJoCo simulation | dt {1000*dt:g} ms | {mode} | success radius {100*result['success_radius']:g} cm"
    if distance is not None:
        text += f" | free flight {distance:.3f} m"
    draw.text((20, 77), text, font=font(19), fill=(225, 230, 235))
    return np.asarray(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", help="Choose the first example of each geometry from this evaluation")
    parser.add_argument("--per-geometry", type=int, default=1)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--timestep", type=float, default=.001)
    parser.add_argument("--slow-replay", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    config["device"] = "cpu"
    cfg = config["env"]
    period = cfg["timestep"] * cfg["substeps"]
    cfg.update(timestep=args.timestep, substeps=round(period / args.timestep), render_width=1280, render_height=720)
    cfg.setdefault("randomization", {}).pop("timesteps", None)
    # A fixed view keeps the arm, target and complete flight visible together.
    extent = max(1.4, cfg["target"].get("high", None)[0] if cfg["target"].get("high") else cfg["target"]["position"][0])
    cfg["render_camera"] = dict(lookat=[extent/2, 0, .45], distance=max(2.5, extent+1), azimuth=90, elevation=-23)
    seeds = [args.seed]
    if args.summary:
        results = json.loads(Path(args.summary).read_text())["results"]
        first = {}
        for result in results:
            selected = first.setdefault(result["object"]["type"], [])
            if len(selected) < args.per_geometry:
                selected.append(result["seed"])
        seeds = [seed for selected in first.values() for seed in selected]
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    manifest = []
    fps = round(1 / period)
    with imageio.get_writer(root / "showcase.mp4", fps=fps, macro_block_size=1) as writer:
        for seed in seeds:
            out = root / f"seed_{seed}"
            out.mkdir(exist_ok=True)
            with (out / "evaluation.jsonl").open("w") as log, redirect_stdout(log):
                result = evaluate(args.checkpoint, config, 1, out, video=True, seed=seed)["results"][0]
            manifest.append(result)
            result["success_radius"] = cfg["target"]["success_radius"]
            reader = imageio.get_reader(out / "episode_000.mp4")
            frames = [frame for frame in reader]
            reader.close()
            for frame in frames:
                writer.append_data(caption(frame, result, args.timestep, "1x real time"))
            if args.slow_replay:
                for frame in frames:
                    labeled = caption(frame, result, args.timestep, "0.25x replay")
                    for _ in range(4):
                        writer.append_data(labeled)
            for _ in range(fps):
                writer.append_data(caption(frames[-1], result, args.timestep, "paused at first floor contact"))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
