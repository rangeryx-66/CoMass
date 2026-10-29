"""Paired deterministic evaluations on fixed seeds at the same control frequency."""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .config import load_config
from .evaluate import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, help="label:path (repeatable)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--timesteps", type=float, nargs="+", default=[.002, .001])
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    period = config["env"]["timestep"] * config["env"]["substeps"]
    rows = []
    for item in args.checkpoint:
        label, source = item.split(":", 1)
        # Freeze one atomic checkpoint version for both physical time steps.
        frozen = root / f"{label}.pt"
        shutil.copyfile(source, frozen)
        state = torch.load(frozen, map_location="cpu", weights_only=False)
        for dt in args.timesteps:
            cfg = deepcopy(config)
            cfg["device"] = "cpu"
            cfg["env"].update(timestep=dt, substeps=round(period / dt))
            cfg["env"].setdefault("randomization", {}).pop("timesteps", None)
            out = root / f"{label}_{dt:g}"
            out.mkdir(parents=True, exist_ok=True)
            with (out / "episodes.jsonl").open("w") as log, redirect_stdout(log):
                summary = evaluate(str(frozen), cfg, args.episodes, str(out), seed=args.seed)
            results = summary["results"]
            shapes = sorted({r["object"]["type"] for r in results})
            row = {"label": label, "checkpoint_steps": state["steps"], "timestep": dt,
                   "episodes": args.episodes, "seed": args.seed,
                   "success_radius": cfg["env"]["target"]["success_radius"],
                   "hits": sum(r["success"] for r in results),
                   **{k: v for k, v in summary.items() if k != "results"},
                   "median_landing_error_m": float(np.median([r["landing_error"] for r in results if r["landing_error"] is not None])),
                   "recontact_rate": float(np.mean([r["recontacts"] > 0 for r in results])),
                   "by_geometry": {shape: {"episodes": sum(r["object"]["type"] == shape for r in results),
                                             "hits": sum(r["success"] for r in results if r["object"]["type"] == shape)} for shape in shapes}}
            rows.append(row)
            (root / "table.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
