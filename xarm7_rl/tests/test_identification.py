from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from xarm7_rl.ballast import ballasted_box, parallel_axis
from xarm7_rl.config import load_config
from xarm7_rl.env import ThrowEnv
from xarm7_rl.initialize_object import calibrate_static_features

ROOT = Path(__file__).parents[1]


def specs():
    arguments = ([.03,.025,.035],.001,.02,[.01,.01,.012],.08)
    return [ballasted_box(*arguments, p) for p in ([0,0,0],[.019,.014,.022])]


def matrix(i):
    xx,yy,zz,xy,xz,yz=i
    return np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])


def test_same_ballast_different_position_exact_parallel_axis():
    a,b=specs()
    assert a['mass']==b['mass']==pytest.approx(.1)
    assert b['com']==pytest.approx([.0152,.0112,.0176])
    expected=.02*.08/.1*parallel_axis([.019,.014,.022])
    assert np.allclose(matrix(b['inertia'])-matrix(a['inertia']),expected,rtol=1e-12,atol=1e-16)
    assert np.all(np.linalg.eigvalsh(matrix(b['inertia']))>0)


@pytest.mark.parametrize("estimated_com", [[0,0,0],[.004,-.003,.005]])
def test_estimate_does_not_change_physics_and_does_not_leak_true_com_velocity(estimated_com):
    a,b=specs()
    config=load_config(ROOT/'configs/default.yaml')['env']
    envs=[]
    try:
        for actual in (a,b):
            c=deepcopy(config);c['object'].update(actual)
            c['inertial_estimate']={k:a[k] for k in ('mass','com','inertia')}
            c['inertial_estimate']['com']=estimated_com
            e=ThrowEnv(c);e.reset(seed=0);envs.append(e)
        first,second=envs
        for e in envs:
            e.data.qpos[e.object_qadr:e.object_qadr+7]=[.8,.2,1.2,*([.5]*4)]
            e.data.qvel[:]=0
            e.data.qvel[e.object_vadr:e.object_vadr+6]=[1,2,3,.2,.4,.8]
        second.data.qpos[:]=first.data.qpos
        for e in envs:mujoco.mj_forward(e.model,e.data)
        assert np.array_equal(first._observation(),second._observation())
        assert np.allclose(second.model.body_ipos[second.object_body],b['com'])
        assert np.allclose(first.model.body_ipos[first.object_body],a['com'])
        assert not np.allclose(first.data.cvel[first.object_body,3:],second.data.cvel[second.object_body,3:])
        assert np.allclose(second._observation()[32:36],[.1,*estimated_com])
        twist=np.zeros(6)
        mujoco.mj_objectVelocity(second.model,second.data,mujoco.mjtObj.mjOBJ_XBODY,second.object_body,twist,0)
        expected_velocity=twist[3:]+np.cross(twist[:3],second.data.xmat[second.object_body].reshape(3,3)@estimated_com)
        assert np.allclose(second._observation()[23:26],expected_velocity)
    finally:
        for e in envs:e.close()


def test_initial_jitter_is_paired_and_each_case_starts_held():
    a,b=specs();config=load_config(ROOT/'configs/default.yaml')['env']
    envs=[]
    try:
        for obj in (a,b):
            c=deepcopy(config);c['object'].update(obj);c['initial_joint_noise']=.005
            e=ThrowEnv(c);_,i=e.reset(seed=40000);envs.append(e)
            assert i['initialization']['bilateral_contact']
            assert not i['initialization']['on_floor']
        assert np.array_equal(envs[0].arm_target,envs[1].arm_target)
        assert not np.array_equal(envs[0].arm_target,np.array(config['initial_joint_positions']))
    finally:
        for e in envs:e.close()


def test_training_initialization_changes_only_static_object_statistics():
    state={"mean":torch.arange(149,dtype=torch.float64),
           "var":torch.ones(149,dtype=torch.float64),"count":torch.tensor(1000.)}
    original={key:value.clone() for key,value in state.items()}
    obs=np.linspace(-1,1,149,dtype=np.float32)
    calibrate_static_features(state,obs,32)
    assert torch.equal(state['mean'][32:138],torch.as_tensor(obs[32:138],dtype=torch.float64))
    assert torch.count_nonzero(state['var'][32:138])==0
    for key in ['mean','var']:
        assert torch.equal(state[key][:32],original[key][:32])
        assert torch.equal(state[key][138:],original[key][138:])
    assert torch.equal(state['count'],original['count'])
