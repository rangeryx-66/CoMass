"""Same-scene recordings of task-only, cloned, and RL-refined throwing policies."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import load_config
from .evaluate import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="label:path")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    recordings, results = [], []
    for item in args.checkpoint:
        label, checkpoint = item.split(":", 1)
        out = root / label
        out.mkdir(exist_ok=True)
        with (out / "evaluation.jsonl").open("w") as log, redirect_stdout(log):
            result = evaluate(checkpoint, config, 1, out, True, 9000)["results"][0]
        results.append({"label": label, **result})
        reader = imageio.get_reader(out / "episode_000.mp4")
        recordings.append([frame for frame in reader])
        reader.close()
    columns = min(2,len(recordings))
    rows = (len(recordings)+columns-1)//columns
    font_path = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    font = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default()
    fps = round(1/(config["env"]["timestep"] * config["env"]["substeps"]))
    length = max(map(len, recordings))
    with imageio.get_writer(root / "comparison.mp4", fps=fps, macro_block_size=1) as writer:
        for speed in (1, 4):
            for frame_index in range(length):
                canvas = Image.new("RGB", (640*columns, 480*rows), (12,16,22))
                draw = ImageDraw.Draw(canvas)
                for column, (frames, result) in enumerate(zip(recordings, results)):
                    left = 640*(column%columns)
                    top = 480*(column//columns)
                    frame = Image.fromarray(frames[min(frame_index, len(frames)-1)]).resize((640,360))
                    canvas.paste(frame,(left,top+120))
                    draw.text((left+12,top+7),result["label"],font=font,fill="white")
                    draw.text((left+12,top+34),f"{'HIT' if result['success'] else 'MISS'} | error {100*result['landing_error']:.2f} cm",font=font,fill="white")
                    draw.text((left+12,top+61),f"Projected pose RMS {result['imitation_mean_error_degrees']:.1f} deg",font=font,fill="white")
                    mode = "1x" if speed==1 else "0.25x replay"
                    ended = " | paused at landing" if frame_index>=len(frames) else ""
                    draw.text((left+12,top+88),f"t={min(frame_index,len(frames)-1)/fps:.2f}s | {mode}{ended}",font=font,fill="white")
                for _ in range(speed):
                    writer.append_data(np.asarray(canvas))
    (root/"manifest.json").write_text(json.dumps(results,indent=2)+"\n")


if __name__=="__main__":
    main()
