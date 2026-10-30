"""Fit xArm7 FK to observed 2-D bone directions; no 3-D recovery claim."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import least_squares

from .config import load_config
from .env import ThrowEnv
from .imitation import PosePrior, projected_directions, target_frame


def retarget(env, prior):
    previous = env.data.qpos[env.qadr].copy()
    solutions, errors = [], []
    forward, up = target_frame(env.target, env._arm_points()[0])
    lateral = np.cross(up, forward)
    for target in prior.directions:
        def residual(q):
            env.data.qpos[env.qadr] = q
            mujoco.mj_forward(env.model, env.data)
            points = env._arm_points()
            bones = np.diff(points, axis=0)
            bones /= np.linalg.norm(bones, axis=1, keepdims=True)
            return np.r_[(projected_directions(points, forward, up) - target).ravel(),
                         .03 * (q - previous), .05 * (bones @ lateral)]
        result = least_squares(residual, previous, bounds=(env.lower+1e-6, env.upper-1e-6),
                               max_nfev=160, ftol=1e-7, xtol=1e-7, gtol=1e-7)
        previous = result.x.copy()
        residual(previous)
        actual = projected_directions(env._arm_points(), forward, up)
        errors.append(float(np.rad2deg(np.sqrt(np.mean(np.arccos(np.sum(actual*target, axis=1).clip(-1, 1))**2)))))
        solutions.append(previous)
    return np.asarray(solutions), errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference", default="mit_overhand")
    parser.add_argument("--output", required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    config["env"]["render_width"], config["env"]["render_height"] = 1280, 720
    config["env"]["render_camera"] = dict(lookat=[.1, 0, .75], distance=2.6, azimuth=90, elevation=-8)
    env = ThrowEnv(config["env"])
    prior = PosePrior({"reference":args.reference})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    try:
        env.reset(seed=0)
        poses, errors = retarget(env, prior)
        np.savez_compressed(out/"retargeted.npz", qpos=poses, projected_directions=prior.directions,
                            projection_error_degrees=np.asarray(errors))
        (out/"summary.json").write_text(json.dumps({"source":prior.source,"mean_projection_error_degrees":float(np.mean(errors)),
             "max_projection_error_degrees":max(errors),"assumptions":"Monocular projected angles; weak planar/continuity regularization selects one IK solution. Not source 3-D joint telemetry or a dynamically feasible throw."},indent=2)+"\n")
        if args.video:
            env.model.geom_rgba[env.object_geom,3] = 0  # Hide the object in this FK-only pose preview.
            with imageio.get_writer(out/"reference.mp4",fps=25) as writer:
                for q in poses:
                    env.data.qpos[env.qadr] = q
                    mujoco.mj_forward(env.model,env.data)
                    frame=Image.fromarray(env.render());draw=ImageDraw.Draw(frame)
                    path="/System/Library/Fonts/Supplemental/Arial.ttf"
                    font=ImageFont.truetype(path,25) if Path(path).exists() else ImageFont.load_default()
                    draw.rectangle((0,0,1280,74),fill=(12,16,22))
                    draw.text((15,10),"xArm7 IK from MIT video projected keypoints",font=font,fill="white")
                    draw.text((15,42),"POSE PREVIEW ONLY | inferred depth | playback phase is arbitrary",font=font,fill="white")
                    writer.append_data(np.asarray(frame))
    finally:
        env.close()


if __name__=="__main__":
    main()
