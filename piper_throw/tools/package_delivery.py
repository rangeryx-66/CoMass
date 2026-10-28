"""Bundle pinned sources and compact evidence; retain full telemetry in results/."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT=Path(__file__).resolve().parents[2]
SELECTED={"best_throw","upward_box_best","hidden_best_contact","hidden_sensitive_best","openarm_original"}


def main():
    manifest={"full_raw_data_directory":str(ROOT/"results"),"archive_contains":"Pinned source, all non-pilot result metadata, summary CSV/figures/videos and selected complete raw trajectories. Full per-trial telemetry remains in results/.","trials":[]}
    for p in sorted((ROOT/"results").rglob("inputs.json")):
        if "pilot" in p.parts:continue
        result=p.with_name("result.json")
        if result.exists():
            r=json.loads(result.read_text())
            manifest["trials"].append(dict(directory=str(p.parent.relative_to(ROOT)),success=r.get("success"),release_mode=r.get("release_mode")))
    (ROOT/"results/DELIVERY_INDEX.json").write_text(json.dumps(manifest,indent=2))
    # Preserve commit identities when exporting Git worktrees without .git.
    for name,commit in {"openarm-control":"14ea59faad6706b9add5d7f8d386e43239a6360f","agilex_piper_mujoco":"cf0d35c06e19002dc8fea059118b2dceea5a50e9","pyAgxArm":"d90e234ba8e408cfa964b38aa9120b3131fc9f85"}.items():
        (ROOT/"piper_throw/upstream"/name/".upstream-commit").write_text(commit+"\n")
    destination=ROOT/"piper_throw_delivery.tar.gz"
    with tarfile.open(destination,"w:gz",compresslevel=6) as tar:
        for base in [ROOT/"piper_throw",ROOT/"configs",ROOT/"results"]:
            for p in sorted(base.rglob("*")):
                if not p.is_file() or any(part in {".git","__pycache__",".pytest_cache","pilot","active_structure_delivery"} for part in p.parts):continue
                rel=p.relative_to(ROOT)
                if any(part.startswith("r1a7") for part in rel.parts):continue
                if rel.parts[0]=="results":
                    selected=len(rel.parts)>1 and (rel.parts[1] in SELECTED or (rel.parts[1] in {"com_scan","upward_com_scan"} and "baseline" in rel.parts))
                    if p.name in {"trajectory.csv","replay.npz","scene.xml"} and not selected:continue
                    if p.suffix not in {".json",".jsonl",".csv",".png",".mp4",".xml",".log",".npz",".md"}:continue
                tar.add(p,arcname=str(rel),recursive=False)
    sha=hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix+".sha256").write_text(sha+"  "+destination.name+"\n")
    print(json.dumps(dict(archive=str(destination),bytes=destination.stat().st_size,sha256=sha,recorded_trials=len(manifest["trials"])),indent=2))


if __name__=="__main__":main()
