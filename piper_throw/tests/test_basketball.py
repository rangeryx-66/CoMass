import numpy as np
import mujoco
import pytest
from piper_throw.basketball_controller import ballistic_plan,make_arc_commands,solve_release_velocity
from piper_throw.scene.build import build_model
from piper_throw.simulator import run_throw


def test_ballistic_boundary_and_apex():
    p=np.array([.4,-.2,1.1]);target=np.array([2.,.3,.25]);r=ballistic_plan(p,target,rise=.25)
    v=np.array(r['velocity']);t=r['flight_time']
    assert np.allclose(p+v*t-[0,0,9.81*t*t/2],target)
    assert np.isclose(v[2]**2/(2*9.81),.25)
    assert r['time_to_apex']<t
    with pytest.raises(ValueError):ballistic_plan(p,target,rise=-1)


def test_bounded_jacobian_and_continuous_braking():
    J=np.c_[np.eye(3),np.zeros((3,3))]
    v,res=solve_release_velocity(J,[1,2,5],[3]*6)
    assert max(abs(v))<=3 and res>=2
    p=dict(family='basketball',release_joints=[0,1.5,-1.4,0,0,0],release_joint_velocity=[0,-.5,-1,0,-.2,0],ramp_time=1.,brake_time=1.)
    cmd=make_arc_commands(p,dict(timestep=.001,acceleration_limits=[5]*6,velocity_limits=[3]*6))
    assert np.max(abs(cmd['qdd']))<=5
    assert np.max(abs(np.diff(cmd['q'],axis=0)/.001-cmd['qd'][:-1]))<.003
    assert np.allclose(cmd['qd'][-1],0) and np.allclose(cmd['qdd'][-1],0)
    assert np.allclose(cmd['q'][0],[0,.8,-1.8,0,.5,0])


def test_table_keeps_object_collision_and_does_not_expand_joint_limits():
    scene=dict(table={'height':.75},base_position=[0,0,.75],position_limits=[[-2.6179,2.6179],[0,3.14],[-2.967,0],[-1.745,1.745],[-1.22,1.22],[-2.09439,2.09439]])
    m,_=build_model({'shape':'box'},scene)
    assert m.geom('table_top').size[2]>.0
    assert m.joint('joint3').range[0]==pytest.approx(-2.697)
    assert m.joint('joint6').range[1]==pytest.approx(2.09439)
    # Physical dropped box must rest on the tabletop, with the weld disabled.
    d=mujoco.MjData(m);q=m.joint('object_free').qposadr[0];d.qpos[q:q+3]=[-.5,0,1.1]
    d.eq_active[m.equality('grasp_weld').id]=0
    for _ in range(1000):mujoco.mj_step(m,d)
    mujoco.mj_forward(m,d)
    assert .75<d.xpos[m.body('object').id,2]<.82


def test_low_arc_is_rejected_using_measured_flight():
    r=run_throw(dict(speed_scale=1.5,release_phase=.52,swing_amplitude=1.1),{'shape':'box'},dict(require_high_arc=True,min_apex_rise=.20,brake_after_release=.2))
    assert not r['success'] and 'insufficient_apex_rise' in r['violations']
    assert r['apex_height_above_release']<.2
    assert abs(r['apex_height_above_release']-r['ballistic_rise_prediction'])<.002


def test_smooth_acceleration_profile_respects_the_same_budget():
    from piper_throw.basketball_controller import smooth_acceleration_progress
    samples=np.array([smooth_acceleration_progress(t) for t in np.linspace(0,1,1001)])
    assert np.all(np.diff(samples[:,1])>=-1e-12)
    assert samples[0,1]==0 and samples[-1,1]==1
    assert np.isclose(samples[-1,0],.5)
    assert np.isclose(samples[:,2].max(),1/.88)
    assert np.max(abs(np.gradient(samples[:,0],.001)-samples[:,1]))<1e-3


def test_ideal_release_box_lands_on_table_not_through_it():
    ready=[0,.3,-2,0,.35,0]
    p=dict(family='basketball',release_joints=ready,release_joint_velocity=[0]*6,ramp_time=1.,brake_time=1.)
    scene=dict(base_position=[0,0,.75],table={'height':.75},ready_joints=ready,flight_settle_time=2.)
    r=run_throw(p,{'shape':'box'},scene)
    assert r['final_position'][2]>.75
    assert any(x['geom']=='table_top' for x in r['airborne_environment_contacts'])


def test_controller_does_not_read_unknown_randomized_gain(monkeypatch):
    # Identical observed initial state must produce identical control, even
    # though the hidden plant gain differs. Subsequent feedback may differ.
    first_controls=[];original=mujoco.mj_step
    def record(model,data,*args,**kwargs):
        if data.time==0:first_controls.append(data.ctrl.copy())
        return original(model,data,*args,**kwargs)
    monkeypatch.setattr(mujoco,'mj_step',record)
    p=dict(family='basketball',release_joints=[0,.8,-1.8,0,.5,0],release_joint_velocity=[0]*6,ramp_time=.5,brake_time=.5)
    scene=dict(release_mode='contact',perturb_contact=True,actuator_gain_std=.1,release_delay_std=0,
               bounded_bias_compensation=True,acceleration_feedforward=True,flight_settle_time=.1)
    for seed in [10,11]:run_throw(p,{'shape':'box'},scene,seed=seed)
    assert len(first_controls)==2
    assert np.array_equal(first_controls[0],first_controls[1])
