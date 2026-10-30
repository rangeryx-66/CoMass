"""Film the first held-out seed, and verify it reproduces the scored trajectories."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .evaluate import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, checkpoints, root = map(Path, (args.results, args.checkpoints, args.output))
    root.mkdir(parents=True, exist_ok=True)
    cases = [
        ("center_060", "center_060", "Guessed model: self-test", "Train 60 g / centered | actual 60 g / centered"),
        ("center_060", "corner_100", "Same policy: parameter mismatch", "Train 60 g / centered | actual 100 g / corner"),
        ("corner_100", "corner_100", "Correct model: retrained policy", "Train 100 g / corner | actual 100 g / corner"),
        ("center_100", "corner_100", "Robust counterexample", "Train 100 g / centered | actual 100 g / corner"),
    ]
    recordings, manifest = [], []
    fps = None
    for policy, world, label, description in cases:
        name = f"{policy}__on__{world}__dt_0.001"
        formal = source / name
        config = json.loads((formal / "config.json").read_text())
        seed = json.loads((formal / "summary.json").read_text())["results"][0]["seed"]
        config["env"].update(render_width=960, render_height=540, render_camera={
            "lookat": [.55, 0, .43], "distance": 2.15, "azimuth": 120, "elevation": -27})
        fps = round(1 / (config["env"]["timestep"] * config["env"]["substeps"]))
        out = root / name
        out.mkdir(exist_ok=True)
        (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        with (out / "evaluation.jsonl").open("w") as log, redirect_stdout(log):
            result = evaluate(checkpoints / f"{policy}.pt", config, 1, out, True, seed)["results"][0]
        with np.load(formal / "episode_000.npz") as expected, np.load(out / "episode_000.npz") as actual:
            np.testing.assert_array_equal(actual["trajectory"], expected["trajectory"])
        reader = imageio.get_reader(out / "episode_000.mp4")
        recordings.append([frame for frame in reader])
        reader.close()
        manifest.append({"label": label, "description": description, "policy": policy,
                         "physics": world, "trajectory_bitwise_matches_formal_test": True, **result})
    font_path = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    font = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default()
    length = max(map(len, recordings))
    with imageio.get_writer(root / "comparison.mp4", fps=fps, macro_block_size=1) as writer:
        for speed in (1, 4):
            for index in range(length + fps):
                canvas = Image.new("RGB", (1280, 960), (12, 16, 22))
                draw = ImageDraw.Draw(canvas)
                for number, (frames, result) in enumerate(zip(recordings, manifest)):
                    left, top = 640 * (number % 2), 480 * (number // 2)
                    canvas.paste(Image.fromarray(frames[min(index, len(frames)-1)]).resize((640, 360)), (left, top+120))
                    draw.text((left+12, top+5), result["label"], font=font, fill="white")
                    draw.text((left+12, top+31), result["description"], font=font, fill="white")
                    draw.text((left+12, top+57), f"{'HIT' if result['success'] else 'MISS'} | error {100*result['landing_error']:.2f} cm | radius 10 cm", font=font,
                              fill="#79e8a2" if result["success"] else "#ff9b85")
                    mode = "1x" if speed == 1 else "0.25x replay"
                    pause = " | paused at landing" if index >= len(frames) else ""
                    draw.text((left+12, top+83), f"Seed {result['seed']} | t={min(index,len(frames)-1)/fps:.2f}s | {mode}{pause}", font=font, fill="white")
                for _ in range(speed):
                    writer.append_data(np.asarray(canvas))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
