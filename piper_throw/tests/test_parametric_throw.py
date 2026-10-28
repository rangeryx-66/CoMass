import json
from pathlib import Path
import numpy as np
import pytest
from piper_throw import throw, plan_throw


@pytest.fixture(scope='module')
def nominal(tmp_path_factory):
    output=tmp_path_factory.mktemp('target_throw')/'front'
    return throw(output_dir=output),output


def test_real_contact_and_shared_export(nominal):
    result,output=nominal
    assert result['success'] and not result['violations'] and not result['slipped']
    assert result['apex_height_above_release']>=result['parameters']['apex_height_m']
    assert result['descending_entry_after_apex']
    commands=np.load(output/'commands.npz')
    trial=Path(result['selected_trial'])
    replay=np.load(trial/'replay.npz')
    control=np.genfromtxt(trial/'control.csv',delimiter=',',names=True)
    assert np.array_equal(commands['joints_rad'],replay['commands'])
    assert np.allclose(commands['gripper_opening_m'][:-1],2*control['gripper_ctrl'])
    assert commands['joints_rad'].shape[1]==6
    assert np.max(commands['gripper_opening_m'])<=.07
    assert result['target_world_m']==pytest.approx([.7,0,.25])
    from piper_throw.simulator import run_throw
    inputs=json.loads((trial/'inputs.json').read_text())
    fine=run_throw(inputs['params'],inputs['object'],dict(inputs['scene'],timestep=.00025))
    assert fine['success'] and not fine['violations']


@pytest.mark.parametrize('angle',[-.35,.35])
def test_actual_left_and_right_targets(angle,tmp_path):
    result=throw({'target_position_m':[.7*np.cos(angle),.7*np.sin(angle),-.5]},output_dir=tmp_path/'trial')
    assert result['success'] and not result['violations'] and not result['slipped']
    assert np.sign(result['actual_direction_rad'])==np.sign(angle)
    assert abs(result['actual_direction_rad']-angle)<.2
    replay=np.load(Path(result['selected_trial'])/'replay.npz')
    assert np.ptp(replay['qpos'][:,0])>.3


def test_side_workspace_failure_is_reported(nominal,tmp_path):
    from piper_throw.simulator import run_throw
    result=throw({'target_position_m':[0,.7,-.5]},output_dir=tmp_path/'side')
    assert result['status']=='infeasible' and result['failure_reason']
    base,_=nominal
    inputs=json.loads((Path(base['selected_trial'])/'inputs.json').read_text())
    action=dict(inputs['params']);action['release_joints'][0]=np.pi/2
    scene=dict(inputs['scene'],throw_direction=[0,1],bucket_position=[0,.7,0])
    actual=run_throw(action,inputs['object'],scene)
    assert 'robot_scene_collision' in actual['violations'] and not actual['success']


def test_target_and_height_change_real_flight(nominal,tmp_path):
    base,_=nominal
    near=throw({'target_position_m':[.68,0,-.5]},output_dir=tmp_path/'near')
    low=throw({'apex_height_m':.006},output_dir=tmp_path/'low')
    assert near['success'] and low['success']
    assert base['crossing']['position'][0]-near['crossing']['position'][0]>.005
    assert abs(base['release']['linear_velocity'][0]-near['release']['linear_velocity'][0])>.01
    assert base['apex_height_above_release']-low['apex_height_above_release']>.002


def test_unreachable_target_is_an_experiment_result(tmp_path):
    result=throw({'target_position_m':[.8,0,-.5],'apex_height_m':.20},output_dir=tmp_path/'high')
    assert result['status']=='infeasible' and not result['success']
    assert json.loads((tmp_path/'high/summary.json').read_text())['failure_reason']
    with pytest.raises(ValueError,match='only target'):
        plan_throw({'distance_m':.8,'target_position_m':[.8,0,-.5]})
