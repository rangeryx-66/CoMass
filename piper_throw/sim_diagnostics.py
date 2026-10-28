"""Repeatable simulation-only diagnostics for controller/constraint mismatches.

Do not interpret these results as manufacturer limits or true hardware calibration.
No robot SDK, CAN interface, or robot command is imported.
"""
from pathlib import Path
import json
import numpy as np

from .simulator import run_throw


def diagnostic_params(scene, moving=False):
    ready = np.asarray(scene.get("ready_joints", [0, .8, -1.8, 0, .5, 0]), dtype=float)
    velocity = np.zeros(6)
    if moving:
        velocity[1] = .15
        velocity[2] = -.12
    return dict(family="basketball", release_joints=ready.tolist(),
                release_joint_velocity=velocity.tolist(),
                ramp_time=1.2, brake_time=1.2,
                open_lead=0., profile="smooth_acceleration")


def run_diagnostics(config, output_dir):
    """Run stationary and gentle motion with identical unrelaxed limits.

    The diagnostic cases may not enter a bucket. Hits are not their purpose.
    Each run stores a full trace and physical constraints remain enabled.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scene = dict(config.get("scene", {}))
    obj = dict(config.get("object", {}))
    # Disable *task-specific arc* requirements only; leave motor, tracking,
    # collision, slip and safety checks unchanged.
    scene["require_high_arc"] = False
    scene.pop("min_release_vertical_velocity", None)
    scene.pop("min_release_forward_velocity", None)
    scene["flight_settle_time"] = max(1.5, float(scene.get("flight_settle_time", 1.5)))
    summaries = []
    for moving in (False, True):
        for mode in ("ideal", "contact"):
            sc = dict(scene, release_mode=mode)
            name = ("gentle_swing" if moving else "static_hold") + "_" + mode
            result = run_throw(diagnostic_params(sc, moving), obj, sc,
                               output_dir=out / name)
            tracking = result["tracking"]
            row = dict(case=name, violation_flags=result["violations"],
                       slipped=result["slipped"],
                       pre_release_grasp_drift_m=result.get("grasp_translation_drift_max_m"),
                       max_acceleration_rad_s2=float(np.max(tracking["peak_acceleration"])),
                       max_tracking_error_rad=float(np.max(tracking["peak_position_error"])),
                       max_actuator_torque_Nm=float(np.max(tracking["peak_torque"])),
                       phase_peak_acceleration=tracking["phase_peak_acceleration"],
                       acceleration_exceedance_samples=tracking["acceleration_exceedance_samples"],
                       actuator_saturation_steps=tracking["actuator_saturation_steps"],
                       integration_steps=tracking["integration_steps"],
                       first_acceleration_exceedance=tracking["first_acceleration_exceedance"],
                       top_acceleration_exceedances=tracking["top_acceleration_exceedances"])
            summaries.append(row)
            (out / name / "diagnostic_summary.json").write_text(json.dumps(row, indent=2))
    manifest = dict(description="diagnostic only; no hardware claims",
                    scene_limits=dict(velocity_limits=scene.get("velocity_limits"),
                                      acceleration_limits=scene.get("acceleration_limits"),
                                      torque_limits=scene.get("torque_limits")),
                    results=summaries)
    (out / "diagnostics.json").write_text(json.dumps(manifest, indent=2))
    return manifest
