import json
import numpy as np
import mujoco
import pytest
from piper_throw.scene.build import mass_properties, build_model
from piper_throw.simulator import run_throw, _velocity
from piper_throw.throw_controller import make_commands
from piper_throw.optimizer import public_observation
from piper_throw.hidden_mass import evaluator_objects
from piper_throw.real_piper import RealPiper

PARAMS=dict(speed_scale=1.1133685604602532,release_phase=.6218841100170127,
            swing_amplitude=1.216309762602005,direction_offset=-.0018431795969399056,
            wrist_pitch_offset=.19174609390924252)
SCENE=dict(flight_settle_time=4)


def test_inertia_physical_and_infeasible_offsets():
    for obj in list(evaluator_objects().values())+[{"com":[.04,.01,0]},{"com":[0,0,.01]}]:
        c,I=mass_properties(obj);eig=np.linalg.eigvalsh(I)
        assert np.all(eig>0) and eig[-1] <= eig[:2].sum()+1e-12
        assert np.allclose(c,obj["com"])
    with pytest.raises(ValueError,match="fit"):
        mass_properties({"shape":"box","com":[0,.04,0]})
    with pytest.raises(ValueError,match="fit"):
        mass_properties({"com":[0,.02,.02]})
    _,a=mass_properties({"shape":"box","com":[0,0,0]})
    _,b=mass_properties({"shape":"box","com":[.03,.015,0]})
    assert not np.allclose(a,b)


def test_hidden_boxes_keep_external_geometry_and_grasp():
    models=[build_model(c,SCENE)[0] for c in evaluator_objects().values()]
    for m in models[1:]:
        g=m.geom("object_geom");g0=models[0].geom("object_geom")
        assert np.array_equal(g.size,g0.size)
        assert np.array_equal(g.friction,g0.friction)
        assert np.isclose(m.body("object").mass,models[0].body("object").mass)


def test_com_velocity_differs_from_geometry_velocity():
    m,_=build_model({"com":[.02,.01,0]},SCENE);d=mujoco.MjData(m)
    d.qpos[8:11]=[0,0,1];d.qvel[8:]=[1,2,3,.4,.5,.6];mujoco.mj_forward(m,d)
    b=m.body("object").id;J=np.zeros((3,m.nv));Jr=J.copy()
    mujoco.mj_jacBodyCom(m,d,J,Jr,b)
    assert np.allclose(J@d.qvel,_velocity(m,d,b,True)[3:])
    mujoco.mj_jacBody(m,d,J,Jr,b)
    assert np.allclose(J@d.qvel,_velocity(m,d,b)[3:])
    assert not np.allclose(_velocity(m,d,b,True),_velocity(m,d,b))


def test_release_free_flight_obeys_gravity_and_angular_momentum():
    obj={"com":[.02,.008,.008]};r=run_throw(PARAMS,obj,SCENE)
    x=r["trajectory"];t0=r["release"]["time"]
    flight=x[(x[:,0]>t0+.03)&(x[:,0]<t0+.15)]
    # telemetry indices: t,geometry3,com3,quat4,omega3,velocity3,...
    dv=np.diff(flight[:,14:17],axis=0)/np.diff(flight[:,0])[:,None]
    assert np.allclose(dv,[0,0,-9.81],atol=1e-6)
    dc=np.diff(flight[:,4:7],axis=0)/np.diff(flight[:,0])[:,None]
    assert np.allclose(dc,(flight[1:,14:17]+flight[:-1,14:17])/2,atol=1e-6)
    _,I=mass_properties(obj);mom=[]
    for row in flight:
        R=np.zeros(9);mujoco.mju_quat2Mat(R,row[7:11]);R=R.reshape(3,3)
        mom.append(R@I@R.T@row[11:14])
    # MuJoCo's finite-step rotational integrator has bounded integration error.
    assert np.max(np.linalg.norm(np.asarray(mom)-mom[0],axis=1)) < 1e-5


def test_reset_determinism_and_frozen_command_identity(tmp_path):
    a=run_throw(PARAMS,{},SCENE,output_dir=tmp_path/"a")
    run_throw(PARAMS,{"com":[.04,0,0]},{**SCENE,"release_mode":"perturbed"},seed=8)
    b=run_throw(PARAMS,{},SCENE)
    assert np.array_equal(a["trajectory"],b["trajectory"])
    cmd=make_commands(PARAMS,SCENE)
    c=run_throw(PARAMS,{"com":[.04,0,0]},SCENE,frozen_commands=cmd)
    assert a["command_sha256"]==c["command_sha256"]
    assert np.array_equal(a["trajectory"][:,29:35],c["trajectory"][:,29:35])
    assert np.array_equal(a["trajectory"][0,1:4],c["trajectory"][0,1:4])
    assert a["success"] and a["stayed_in_bucket"] and not a["violations"]
    assert (tmp_path/"a/trajectory.csv").exists()


def test_safety_gate_and_disabled_hardware():
    r=run_throw({"speed_scale":8,"swing_amplitude":1.1},{},SCENE)
    assert not r["success"] and "command_velocity_limit" in r["violations"]
    with pytest.raises(RuntimeError,match="disabled"):RealPiper()
    with pytest.raises(ValueError):make_commands({"release_phase":1.1},SCENE)


def test_contact_release_is_not_weld_and_detaches():
    r=run_throw(dict(speed_scale=1.2,release_phase=.75,swing_amplitude=1.1),
                {"shape":"box"},{**SCENE,"release_mode":"contact"})
    assert r["release"] and not r["slipped"] and not r["post_release_robot_contact"]
    assert r["release"]["time"]>=r["release_command_time"]
    assert r["success"]


def test_public_observation_hides_truth():
    r=run_throw(PARAMS,{"shape":"box","com":[.03,.015,0]},SCENE)
    obs=public_observation(r)
    assert not any(k in json.dumps(obs) for k in ["com_","inertia","mass","quaternion","trajectory"])


def test_upward_release_gate_and_smooth_braking():
    rejected=run_throw(PARAMS,{},dict(**SCENE,min_release_vertical_velocity=.05))
    assert not rejected["success"] and "release_not_upward" in rejected["violations"]
    p=dict(speed_scale=1.45,release_phase=.4,swing_amplitude=1.1)
    r=run_throw(p,{"shape":"box"},{**SCENE,"release_mode":"contact","brake_after_release":.15,
                                      "min_release_vertical_velocity":.05,"min_release_forward_velocity":.1})
    assert r["success"] and r["release"]["linear_velocity"][2]>.05
    assert not r["post_release_robot_contact"] and not r["violations"]
