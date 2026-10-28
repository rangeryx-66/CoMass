"""Fetch only the upstream collision STL assets, verifying LFS SHA256 pointers."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "cf0d35c06e19002dc8fea059118b2dceea5a50e9"


def fetch(name):
    path = ROOT / "upstream/agilex_piper_mujoco/models/agilex_piper/assets" / name
    sha = json.loads((Path(__file__).with_name("asset_sha256.json")).read_text())[name]
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == sha:
        return name
    tmp = path.with_suffix(".download")
    url = f"https://media.githubusercontent.com/media/renesas-rdk/agilex_piper_mujoco/{COMMIT}/models/agilex_piper/assets/{name}"
    subprocess.run(["curl", "-fsSL", "--retry", "8", "--retry-all-errors", "--max-time", "300",
                    url, "-o", str(tmp)], check=True)
    if hashlib.sha256(tmp.read_bytes()).hexdigest() != sha:
        raise ValueError(f"LFS hash mismatch: {name}")
    tmp.replace(path)
    return name


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name in pool.map(fetch, ["base_link.stl"] + [f"link{i}.stl" for i in range(1, 9)]):
            print("verified", name, flush=True)
