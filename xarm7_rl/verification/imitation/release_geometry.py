"""Reconstruct release landmarks by FK from recorded actual arm joint positions.

This is offline analysis only. No physics rollout or object state is modified.
Run from the CoMass root with PYTHONPATH=xarm7_rl/src.
"""
import json
from pathlib import Path

import mujoco
import numpy as np

from xarm7_rl.config import load_config
from xarm7_rl.env import ThrowEnv


root = Path(__file__).parent
env = ThrowEnv(load_config(root.parents[1] / "configs/default.yaml")["env"])
rows = []
try:
    env.reset(seed=0)
    for file in sorted((root / "test").glob("*/summary.json")):
        results = json.loads(file.read_text())["results"]
        landmarks = []
        for result in results:
            release = result["release"]
            if release is None:
                continue
            env.data.qpos[env.qadr] = release["arm_qpos"]
            mujoco.mj_kinematics(env.model, env.data)
            landmarks.append(env._arm_points())
        points = np.asarray(landmarks)
        rows.append({"case": file.parent.name, "released": len(points),
                     "wrist_above_elbow_fraction": float(np.mean(points[:, 2, 2] > points[:, 1, 2])),
                     "palm_above_elbow_fraction": float(np.mean(points[:, 3, 2] > points[:, 1, 2])),
                     "mean_palm_release_height_m": float(np.mean(points[:, 3, 2])),
                     "mean_valid_release_pose_error_degrees": float(np.mean([r["imitation_release_error_degrees"] for r in results if r["release"]]))})
finally:
    env.close()
(root / "release_geometry.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
