"""Archive the completed local run without discarding unsuccessful controls."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
from zipfile import ZipFile, ZIP_DEFLATED


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def evaluations(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for file in source.rglob("*"):
        if file.is_file() and file.suffix in (".json", ".jsonl", ".yaml"):
            copy(file, destination / file.relative_to(source))
    trajectories = sorted(source.rglob("episode_*.npz"))
    if trajectories:
        with ZipFile(destination / "trajectories.zip", "w", ZIP_DEFLATED) as archive:
            for file in trajectories:
                archive.write(file, str(file.relative_to(source)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/xarm7_identification")
    parser.add_argument("--output", default="xarm7_rl/verification/identification")
    args = parser.parse_args()
    source, root = Path(args.run), Path(args.output)
    copy(source / "protocol.json", root / "protocol.json")
    for case in ("center_060", "center_100", "corner_100"):
        for name in ("config.yaml", "metrics.jsonl"):
            copy(source / case / name, root / "training" / case / name)
        copy(source / f"{case}.log", root / "training" / case / "console.log")
    for label in ("validation_early", "validation_late", "selection_center_100"):
        evaluations(source / label, root / "validation" / label)
    evaluations(source / "test", root / "evaluation" / "main")
    evaluations(source / "check_half_ms", root / "evaluation" / "half_ms")
    for file in (source / "figures").iterdir():
        if file.is_file(): copy(file, root / "figures" / file.name)
    for name in ("comparison.mp4", "manifest.json", "preview_038.png"):
        copy(source / "video" / name, root / "video" / ("preview.png" if name.startswith("preview") else name))
    with ZipFile(root / "video" / "raw_recordings.zip", "w", ZIP_DEFLATED) as archive:
        for file in sorted((source / "video").glob("*__on__*/*")):
            if file.is_file(): archive.write(file, str(file.relative_to(source / "video")))
    for name in ("tests.log", "tests.xml"):
        copy(source / name, root / name)
    environment = {"python":sys.version, "platform":platform.platform(),
                   "packages":{name:importlib.metadata.version(name) for name in (
                       "torch", "mujoco", "numpy", "gymnasium", "trimesh", "scipy", "matplotlib", "imageio")}}
    files = {str(file.relative_to(root)):{"bytes":file.stat().st_size,
             "sha256":hashlib.sha256(file.read_bytes()).hexdigest()}
             for file in sorted(root.rglob("*")) if file.is_file() and file != root / "manifest.json"}
    (root / "manifest.json").write_text(json.dumps({"environment":environment, "files":files}, indent=2)+"\n")


if __name__ == "__main__":
    main()
