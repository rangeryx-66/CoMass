"""Regression checks for a simulation-only, physically honest diagnosis path."""
import json
import numpy as np
from piper_throw import sim_diagnostics


def test_stationary_diagnostic_has_zero_swing_velocity():
    cfg=dict(ready_joints=[0,.8,-1.8,0,.5,0])
    p=sim_diagnostics.diagnostic_params(cfg,False)
    assert np.allclose(p["release_joint_velocity"],0)
    assert p["family"]=="basketball"


def test_diagnostic_does_not_relax_motor_limits(monkeypatch,tmp_path):
    calls=[]
    def fake_run(params,obj,scene,output_dir=None):
        calls.append(dict(params=params,scene=dict(scene),output_dir=output_dir))
        return dict(violations=["actual_acceleration_limit"],slipped=False,
                    grasp_translation_drift_max_m=0.,
                    tracking=dict(peak_acceleration=[6.,0,0,0,0,0],
                                  peak_velocity=[0]*6,peak_position_error=[0]*6,
                                  peak_torque=[5]*6,
                                  phase_peak_acceleration={"initial_settle":[6.,0,0,0,0,0]},
                                  acceleration_exceedance_samples=1,
                                  first_acceleration_exceedance={"time_s":.002,"joint":1},
                                  top_acceleration_exceedances=[{"time_s":.002,"joint":1}],
                                  actuator_saturation_steps=[0]*6,integration_steps=42))
    monkeypatch.setattr(sim_diagnostics,"run_throw",fake_run)
    scene=dict(require_high_arc=True,min_release_vertical_velocity=1.98,
               velocity_limits=[3]*6,acceleration_limits=[5]*6,
               torque_limits=[12,12,12,3,3,3])
    out=sim_diagnostics.run_diagnostics(dict(scene=scene,object=dict(shape="box")),tmp_path)
    assert len(calls)==4
    assert {(c["params"]["release_joint_velocity"][1],c["scene"]["release_mode"]) for c in calls}=={
        (0.0,"ideal"),(0.0,"contact"),(.15,"ideal"),(.15,"contact")}
    assert all(c["scene"]["acceleration_limits"]==[5]*6 for c in calls)
    assert all(c["scene"]["velocity_limits"]==[3]*6 for c in calls)
    assert all(c["scene"]["torque_limits"]==[12,12,12,3,3,3] for c in calls)
    assert all(c["scene"]["require_high_arc"] is False for c in calls)
    assert all("min_release_vertical_velocity" not in c["scene"] for c in calls)
    assert all(r["violation_flags"]==["actual_acceleration_limit"] for r in out["results"])
    assert len(json.loads((tmp_path/"diagnostics.json").read_text())["results"])==4
