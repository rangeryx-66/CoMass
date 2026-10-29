"""Small real-MuJoCo controller and grasp experiments, without hardware access."""
from pathlib import Path
import csv
import json
import yaml
import numpy as np

from .simulator import run_throw


def diagnostic_params(scene, moving=False):
    ready = np.asarray(scene.get("ready_joints", [0, .8, -1.8, 0, .5, 0]), dtype=float)
    velocity = np.array([0., .15, -.12, 0., 0., 0.]) if moving else np.zeros(6)
    return dict(family="basketball", release_joints=ready.tolist(),
                release_joint_velocity=velocity.tolist(), ramp_time=1.2, brake_time=1.2,
                open_lead=0., profile="smooth_acceleration")


def run_diagnostics(config, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scene = dict(config["scene"], require_high_arc=False)
    scene.pop("min_release_vertical_velocity", None)
    scene.pop("min_release_forward_velocity", None)
    obj = config["object"]
    static = diagnostic_params(scene)
    single = diagnostic_params(scene, True)
    single["release_joint_velocity"][2] = 0.
    swing = diagnostic_params(scene, True)
    source = Path(__file__).resolve().parents[1] / "results/basketball_throw/published/diagnostic_action.yaml"
    full = yaml.safe_load(source.read_text())["params"]
    cases = [("A_static_empty", static, False), ("B_static_loaded", static, True),
             ("C_single_empty", single, False), ("D_single_loaded", single, True),
             ("E_swing_loaded", swing, True), ("F_throw_loaded", full, True)]
    trials = [(name+"_"+mode, params, dict(scene, release_mode=mode, grasp_object=attached))
              for name, params, attached in cases for mode in ("ideal", "contact")]
    for method in ("pd", "gravity"):
        trials.append(("E_"+method, swing, dict(scene, release_mode="contact",
                       bias_compensation=method=="gravity", acceleration_feedforward=False)))
    for repeat in (1, 2):
        trials.append((f"E_repeat_{repeat}", swing, dict(scene, release_mode="contact")))
    summaries = []
    for name, params, sc in trials:
        result = run_throw(params, obj, sc, output_dir=out / name)
        tr = result["tracking"]
        summaries.append(dict(case=name, violations=result["violations"], slipped=result["slipped"],
                              grasp_drift_m=result.get("grasp_translation_drift_max_m"),
                              peak_acceleration=max(tr["peak_acceleration"]),
                              peak_position_error=max(tr["peak_position_error"]),
                              peak_torque=max(tr["peak_torque"]),
                              phase_peak_acceleration=tr["phase_peak_acceleration"],
                              saturation_steps=tr["actuator_saturation_steps"],
                              release=result["release"] if sc.get("grasp_object", True) else None,
                              release_command_time=result["release_command_time"],
                              first_acceleration_exceedance=tr["first_acceleration_exceedance"]))
    manifest = dict(scene_limits={k: scene[k] for k in ("velocity_limits", "acceleration_limits", "torque_limits")},
                    randomization=False, results=summaries)
    (out / "diagnostics.json").write_text(json.dumps(manifest, indent=2)+"\n")
    with (out / "summary.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    plot_tracking(out)
    return manifest


def plot_tracking(output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(3,2,figsize=(11,8),sharex=True)
    for name,label in [("E_pd","PD"),("E_gravity","PD + bias"),
                       ("E_swing_loaded_contact","PD + dynamics")]:
        rows=np.genfromtxt(Path(output_dir)/name/"control.csv",delimiter=",",names=True)
        for j,ax in enumerate(axes.flat,1):
            ax.plot(rows["time"],rows[f"q_{j}"],label=label)
    for j,ax in enumerate(axes.flat,1):
        ax.plot(rows["time"],rows[f"target_q_{j}"],"k--",label="command",linewidth=1)
        ax.set_ylabel(f"J{j} (rad)");ax.grid(alpha=.25)
    axes[0,0].legend();axes[-1,0].set_xlabel("time (s)");axes[-1,1].set_xlabel("time (s)")
    fig.tight_layout();fig.savefig(Path(output_dir)/"tracking.png",dpi=160);plt.close(fig)
