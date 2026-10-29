from pathlib import Path
import numpy as np
import mujoco
import yaml
from piper_throw.range_throw import make_range_commands
from piper_throw.scene.build import build_model
from piper_throw.simulator import run_throw


def config():
    return yaml.safe_load(Path('configs/range_throw.yaml').read_text())


def stationary_params(scene):
    ready=scene['ready_joints']
    return dict(family='trebuchet',windup=ready,swing_end=ready,swing_velocity=[0]*6,
                brake_end=ready,prepare_s=1.,swing_s=1.,brake_s=1.,release_phase=.5)


def test_measurement_scene_preserves_table_floor_and_object():
    c=config();model,_=build_model(c['object'],c['scene']);data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    assert mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'bucket')==-1
    assert data.xpos[model.body('base_link').id,2]==.75
    gid=model.geom('table_top').id
    assert data.geom_xpos[gid,2]+model.geom_size[gid,2]==.75
    assert data.geom_xpos[model.geom('floor').id,2]==0
    assert model.body_mass[model.body('object').id]==.1
    assert np.allclose(model.geom_size[model.geom('object_geom').id],[.06,.04,.025])
    assert np.allclose(model.geom_friction[model.geom('object_geom').id],[.8,.01,.001])
    assert not model.eq_active0[model.equality('grasp_weld').id]


def test_six_axis_independent_paths_are_continuous():
    c=config();p=stationary_params(c['scene']);p.update(windup=[.2,1.,-1.8,.2,.8,-.2],
        swing_end=[.4,1.5,-1.2,.4,.5,.2],swing_velocity=[.1,.2,.3,.1,-.1,.2],
        brake_end=[.5,1.7,-1.,.3,.6,.4],brake_s=1.4)
    cmd=make_range_commands(p,c['scene']);dt=c['scene']['timestep']
    for boundary in [cmd['swing_start_time'],cmd['swing_end_time'],cmd['brake_end_time']]:
        k=round(boundary/dt)
        assert np.max(np.abs(cmd['q'][k+1]-cmd['q'][k-1]))<.002
        assert np.max(np.abs(cmd['qd'][k+1]-cmd['qd'][k-1]))<.001
        assert np.max(np.abs(cmd['qdd'][k+1]-cmd['qdd'][k-1]))<.2
    assert np.max(np.abs(cmd['q'][:,0]))>.3
    assert np.max(np.abs(cmd['q'][:,3]))>.3
    assert np.max(np.abs(cmd['q'][:,5]))>.3


def test_real_contact_first_floor_metric_excludes_later_motion_and_static_drop(tmp_path):
    c=config();p=stationary_params(c['scene'])
    result=run_throw(p,c['object'],c['scene'],output_dir=tmp_path)
    assert result['release']['time']>result['release_command_time']
    assert result['first_floor_contact'] is not None
    assert not result['controlled_downswing'] and not result['success']
    impact=result['first_floor_contact'];release=result['release']
    assert result['flight_time']==impact['time']-release['time']
    assert result['trajectory'][-1,0]>impact['time']
    assert not np.allclose(result['final_position'],impact['geometry_position'],atol=1e-4)
    assert result['range_m']==np.linalg.norm(np.array(impact['contact_position'])[:2]-np.array(release['position'])[:2])
    assert result['violations']==[]


def test_saved_six_axis_throw_has_real_unobstructed_gravity_flight(tmp_path):
    import json
    saved=json.loads(Path('data/range_20261029/selected_nominal/inputs.json').read_text())
    result=run_throw(saved['params'],saved['object'],saved['scene'],output_dir=tmp_path)
    assert result['success'] and result['range_m']>.8
    assert result['release']['time']>result['release_command_time']
    assert result['release']['end_effector_velocity'][2]<0
    assert result['first_flight_obstacle'] is None
    rows=np.genfromtxt(tmp_path/'trajectory.csv',delimiter=',',names=True)
    free=rows[(rows['time']>result['release']['time']+.02)&
              (rows['time']<result['first_floor_contact']['time']-.01)]
    assert len(free)>20
    assert np.allclose(np.diff(free['velocity_z'])/np.diff(free['time']),-9.81,atol=1e-6)
