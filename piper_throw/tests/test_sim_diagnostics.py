import json
from pathlib import Path
import numpy as np
import yaml
import mujoco
from piper_throw.sim_diagnostics import diagnostic_params
from piper_throw.simulator import run_throw


def test_real_static_jaw_transient_and_finite_opening(tmp_path):
    config=yaml.safe_load(Path('configs/basketball_throw.yaml').read_text())
    scene=dict(config['scene'],require_high_arc=False,release_mode='ideal',flight_settle_time=.3)
    scene.pop('min_release_vertical_velocity');scene.pop('min_release_forward_velocity')
    params=diagnostic_params(scene)
    step=run_throw(params,config['object'],dict(scene,gripper_motion_time=0.))
    smooth=run_throw(params,config['object'],scene,output_dir=tmp_path)
    assert step['tracking']['peak_acceleration'][3]>5
    assert 'actual_acceleration_limit' in step['violations']
    assert max(smooth['tracking']['peak_acceleration'])<1.3
    assert smooth['violations']==[]
    rows=np.genfromtxt(tmp_path/'control.csv',delimiter=',',names=True)
    assert np.allclose(rows['time']-rows['command_time'],.001)
    assert np.max(np.abs(rows['torque_2']))<=12
    assert np.allclose(np.diff(rows['qd_4'])/.001,rows['qdd_4'][1:])
    assert json.loads((tmp_path/'inputs.json').read_text())['scene']['acceleration_limits']==[5]*6
    assert smooth['tracking']['integration_steps']==len(smooth['trajectory'])-1
    assert rows['time'][-1]==smooth['trajectory'][-1,0]
    assert np.allclose(smooth['final_position'],smooth['trajectory'][-1,1:4])


def test_real_low_speed_contact_grasp_and_release(tmp_path):
    config=yaml.safe_load(Path('configs/basketball_throw.yaml').read_text())
    scene=dict(config['scene'],require_high_arc=False,release_mode='contact',flight_settle_time=.3)
    scene.pop('min_release_vertical_velocity');scene.pop('min_release_forward_velocity')
    result=run_throw(diagnostic_params(scene,True),config['object'],scene,output_dir=tmp_path)
    assert result['violations']==[] and not result['slipped']
    assert result['grasp_translation_drift_max_m']<.005
    assert result['release']['time']>=result['release_command_time']
    assert max(result['tracking']['peak_position_error'])<.003
    from piper_throw.scene.build import build_model
    model,_=build_model(config['object'],scene)
    data=mujoco.MjData(model)
    data.qpos[:]=np.load(tmp_path/'replay.npz')['qpos'][0]
    mujoco.mj_forward(model,data)
    obj=model.body('object').id
    assert all(c.dist>=-1e-8 for c in data.contact
               if obj in {model.geom_bodyid[c.geom1],model.geom_bodyid[c.geom2]})
    repeat=run_throw(diagnostic_params(scene,True),config['object'],scene)
    assert repeat['release']==result['release']
