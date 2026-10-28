"""Fresh MuJoCo forward simulation for every trial, with auditable telemetry."""
from pathlib import Path
import csv
import json
import numpy as np
import mujoco
from .scene.build import build_model
from .throw_controller import make_commands


def _velocity(model, data, bid, com=False):
    v = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, v, 0)
    # mj_objectVelocity BODY uses the body's COM. Geometry-origin velocity is
    # v_com - omega x (world_com - world_origin).
    if not com:
        v[3:] -= np.cross(v[:3], data.xipos[bid]-data.xpos[bid])
    return v


def _extent(data, bid, shape):
    R = data.xmat[bid].reshape(3,3)
    if shape == "box":
        corners = np.array([[x,y,z] for x in [-.06,.06] for y in [-.04,.04] for z in [-.025,.025]])
        return data.xpos[bid] + corners @ R.T
    return data.xpos[bid] + np.array([[-.045,0,0],[.045,0,0]]) @ R.T


def run_throw(params, object_config=None, scene_config=None, *, output_dir=None, seed=0,
              frozen_commands=None):
    """Only physical integration determines releases/crossings/settled outcomes.

    object_config is privileged evaluator input. Agent-facing API is in tools_api.
    """
    object_config = dict(object_config or {})
    scene = dict(scene_config or {})
    release_mode = scene.get("release_mode", "ideal")
    if release_mode not in {"ideal", "contact", "perturbed"}:
        raise ValueError("unknown release mode")
    cmd = frozen_commands if frozen_commands is not None else make_commands(params, scene)
    model, xml = build_model(object_config, scene)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    ids = [model.joint(f"joint{i}").id for i in range(1,7)]
    qa, va = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
    acts = [model.actuator(f"joint{i}").id for i in range(1,7)]
    data.qpos[qa] = cmd["q"][0]
    # Contact model uses the actual jaw pads and finite-speed opening. Ideal
    # model omits object/finger contacts via collision masks, not object motion.
    data.qpos[6:8] = [.025,-.025]
    data.ctrl[model.actuator("gripper").id] = scene.get("gripper_closed", .015)
    mujoco.mj_forward(model, data)
    ee = model.body("link6").id; obj = model.body("object").id
    oq = model.joint("object_free").qposadr[0]
    ov = model.joint("object_free").dofadr[0]
    weld = model.equality("grasp_weld").id
    shape = object_config.get("shape", "capsule")
    grasp = np.array(scene.get("grasp_offset", [.025, 0, .105]), dtype=float)
    if cmd["parameters"]["family"] in {"planar", "translation"}:
        grasp[0] = 0
    data.qpos[oq:oq+3] = data.xpos[ee] + data.xmat[ee].reshape(3,3) @ grasp
    relative_quat = np.array([np.sqrt(.5),np.sqrt(.5),0,0]) if shape=="box" else np.array([1.,0,0,0])
    if "grasp_quaternion" in scene:
        relative_quat=np.asarray(scene["grasp_quaternion"],dtype=float)
        if relative_quat.shape!=(4,) or not np.isfinite(relative_quat).all() or np.linalg.norm(relative_quat)<1e-9:
            raise ValueError("invalid grasp quaternion (wxyz)")
        relative_quat=relative_quat/np.linalg.norm(relative_quat)
    mujoco.mju_mulQuat(data.qpos[oq+3:oq+7],data.xquat[ee],relative_quat)
    model.eq_data[weld, 3:6] = grasp
    model.eq_data[weld, 6:10] = relative_quat
    data.eq_active[weld] = release_mode != "contact"
    arm_bodies = {model.body(n).id for n in ["base_link"]+[f"link{i}" for i in range(1,9)]}
    geom = model.geom("object_geom").id
    if release_mode != "contact":
        # Robot geoms=bit1; environment=bit2; object=bit4. Ideal release bypasses
        # jaw contact only. This is explicitly not a model of finger opening.
        for gi in range(model.ngeom):
            if model.geom_bodyid[gi] in arm_bodies:
                model.geom_contype[gi] = 1; model.geom_conaffinity[gi] = 3
            elif gi != geom:
                model.geom_contype[gi] = 2; model.geom_conaffinity[gi] = 7
        model.geom_contype[geom] = 4; model.geom_conaffinity[geom] = 2
    nominal_gains=model.actuator_gainprm[acts,0].copy()
    nominal_velocity_gains=-model.actuator_biasprm[acts,2].copy()
    rng = np.random.default_rng(seed)
    delay, response = 0., 1.
    randomized = release_mode == "perturbed" or (release_mode == "contact" and scene.get("perturb_contact",False))
    if randomized:
        delay = float(rng.normal(0, scene.get("release_delay_std", .004)))
        response = float(np.clip(rng.normal(1, scene.get("actuator_gain_std", .025)), .8, 1.2))
        model.actuator_gainprm[acts,0] *= response
        model.actuator_biasprm[acts,1] *= response
        model.geom_friction[:,0] *= float(np.exp(rng.normal(0,.05)))
    release_at = max(0, cmd["release_time"]+delay)
    mujoco.mj_forward(model, data)
    limits = model.jnt_range[ids]
    vmax = np.asarray(scene.get("velocity_limits", [4,4,4,5,5,5]))
    amax = np.asarray(scene.get("acceleration_limits", [40,40,40,60,60,60]))
    violations = set()
    if ((cmd["q"] < limits[:,0]) | (cmd["q"] > limits[:,1])).any(): violations.add("command_joint_limit")
    if (np.abs(cmd["qd"]) > vmax).any(): violations.add("command_velocity_limit")
    if (np.abs(cmd["qdd"]) > amax).any(): violations.add("command_acceleration_limit")
    result = dict(success=False, release_mode=release_mode, seed=seed, randomization=bool(randomized),
                  parameters=cmd["parameters"], command_sha256=cmd["command_sha256"],
                  mass=object_config.get("mass",.1), com_local=object_config.get("com",[0,0,0]),
                  release=None, crossing=None, com_crossing=None, rim_contact=False, stayed_in_bucket=False,
                  perturbations=dict(release_delay_s=delay,actuator_gain_scale=response),
                  slipped=False, post_release_robot_contact=False, violations=[], failure_reason=None)
    bucket = np.asarray(scene.get("bucket_position", [.8,0,0]))
    rim = bucket[2] + scene.get("bucket_height", .22)
    radius = scene.get("bucket_radius", .14)
    old = None; rows = []; states = []; last_v = data.qvel[va].copy()
    released = False; separation_candidate = None; collision_pairs = set()
    final_containment = []; release_command_time = None; grasp_drift_max=0.
    arm_bodies = {model.body(n).id for n in ["base_link"]+[f"link{i}" for i in range(1,9)]}
    peak_velocity=np.zeros(6);peak_acceleration=np.zeros(6);peak_torque=np.zeros(6);peak_tracking=np.zeros(6)
    airborne_environment_contacts=[]
    for k, t in enumerate(cmd["time"]):
        data.ctrl[acts] = cmd["q"][k]
        if scene.get("bounded_bias_compensation",False):
            feed=data.qfrc_bias[va]+nominal_velocity_gains*cmd["qd"][k]
            if scene.get("acceleration_feedforward",False):
                desired_acc=np.zeros(model.nv);desired_acc[va]=cmd["qdd"][k]
                torque=np.zeros(model.nv);mujoco.mj_mulM(model,data,torque,desired_acc)
                feed+=torque[va]
                # Nominal measured total mass only. No hidden CoM/inertia read by the controller.
                if t<release_at:
                    J=np.zeros((3,model.nv));W=J.copy()
                    point=data.xpos[ee]+data.xmat[ee].reshape(3,3)@grasp
                    mujoco.mj_jac(model,data,J,W,point,ee);J=J[:,va]
                    feed+=scene.get("nominal_payload_mass",.1)*(J.T@(J@cmd["qdd"][k]+np.array([0.,0.,9.81])))
            data.ctrl[acts]+=feed/nominal_gains
        if t >= release_at:
            data.ctrl[model.actuator("gripper").id] = .035
            if release_command_time is None:
                release_command_time = float(data.time)
            if release_mode != "contact" and not released:
                data.eq_active[weld] = 0
                released = True
        touching = False
        for con in data.contact:
            g1,g2 = int(con.geom1),int(con.geom2)
            b1,b2 = int(model.geom_bodyid[g1]),int(model.geom_bodyid[g2])
            if obj in {b1,b2}:
                other = g2 if b1 == obj else g1
                name = model.geom(other).name or ""
                if released and (b2 if b1==obj else b1) not in arm_bodies and result["crossing"] is None:
                    airborne_environment_contacts.append(dict(time=float(data.time),geom=name))
                if scene.get("require_high_arc",False) and not released and (b2 if b1==obj else b1) not in arm_bodies:
                    violations.add("pre_release_object_scene_collision")
                if name.startswith("bucket_wall_") and data.xpos[obj,2] > rim-.04:
                    result["rim_contact"] = True
                if (b2 if b1==obj else b1) in arm_bodies:
                    touching = True
            if con.dist < -scene.get("collision_penetration_tolerance",.001) and b1 in arm_bodies and b2 in arm_bodies:
                pair = tuple(sorted([model.body(b1).name,model.body(b2).name]))
                collision_pairs.add(pair)
            if con.dist < -scene.get("collision_penetration_tolerance",.001) and ((b1 in arm_bodies and b2 not in arm_bodies|{obj}) or (b2 in arm_bodies and b1 not in arm_bodies|{obj})):
                violations.add("robot_scene_collision")
        if release_mode == "contact":
            relative = data.xmat[ee].reshape(3,3).T @ (data.xpos[obj]-data.xpos[ee])
            if t < release_at:
                grasp_drift_max=max(grasp_drift_max,float(np.linalg.norm(relative-grasp)))
                if np.linalg.norm(relative-grasp) > .02:result["slipped"] = True
            if t >= release_at and not released:
                if touching:
                    separation_candidate=None
                elif separation_candidate is None:
                    v0=_velocity(model,data,obj,True)
                    separation_candidate=dict(time=float(data.time),position=data.xpos[obj].tolist(),
                                              com_position=data.xipos[obj].tolist(),linear_velocity=v0[3:].tolist(),
                                              angular_velocity=v0[:3].tolist(),quaternion=data.xquat[obj].tolist())
                elif data.time-separation_candidate["time"] >= .02:
                    released=True;result["release"]=separation_candidate
            if released and touching and result["release"] and data.time-result["release"]["time"] > .02:
                result["post_release_robot_contact"] = True
        if released and result["release"] is None:
            v = _velocity(model,data,obj,True)
            result["release"] = dict(time=float(data.time), position=data.xpos[obj].tolist(),
                                     com_position=data.xipos[obj].tolist(), linear_velocity=v[3:].tolist(),
                                     angular_velocity=v[:3].tolist(), quaternion=data.xquat[obj].tolist())
        p, c = data.xpos[obj].copy(), data.xipos[obj].copy()
        v = _velocity(model,data,obj,True)
        if old is not None and released and result["com_crossing"] is None and old[1][2] > rim >= c[2]:
            fcom=(old[1][2]-rim)/(old[1][2]-c[2])
            cp=old[1]+fcom*(c-old[1])
            result["com_crossing"]=dict(time=float(old[2]+fcom*(data.time-old[2])),position=cp.tolist(),error_xy=(cp[:2]-bucket[:2]).tolist())
        if old is not None and released and result["crossing"] is None and old[0][2] > rim >= p[2]:
            f = (old[0][2]-rim)/(old[0][2]-p[2])
            pc = old[0] + f*(p-old[0]); cc = old[1]+f*(c-old[1])
            quat = data.xquat[obj].copy()  # nearest integration-step orientation
            extent = _extent(data,obj,shape)
            margin = 0 if shape=="box" else .025
            fits = bool(np.max(np.linalg.norm(extent[:,:2]-bucket[:2],axis=1))+margin < radius)
            result["crossing"] = dict(time=float(old[2]+f*(data.time-old[2])), position=pc.tolist(),
                                      com_position=cc.tolist(), quaternion=quat.tolist(),
                                      error_xy=(pc[:2]-bucket[:2]).tolist(), com_error_xy=(cc[:2]-bucket[:2]).tolist(),
                                      full_horizontal_envelope_inside=fits)
        extent = _extent(data,obj,shape)
        margin = 0 if shape=="box" else .025
        contained = (np.max(np.linalg.norm(extent[:,:2]-bucket[:2],axis=1))+margin < radius and
                     np.max(extent[:,2])+margin < rim and np.min(extent[:,2])-margin >= bucket[2]+.005)
        if t > cmd["time"][-1]-.3:
            final_containment.append(bool(contained and np.linalg.norm(v[3:])<.08 and np.linalg.norm(v[:3])<.8))
        rows.append(np.r_[data.time,p,c,data.xquat[obj],v,data.qpos[qa],data.qvel[va],cmd["q"][k],float(released)])
        states.append(data.qpos.copy())
        old = (p,c,float(data.time))
        mujoco.mj_step(model,data)
        mujoco.mj_forward(model,data)  # current-time xpos/xipos/contact/velocity telemetry
        if not np.isfinite(data.qpos).all() or data.time < t-.01:
            violations.add("numerical_instability"); break
        if ((data.qpos[qa]<limits[:,0]-scene.get("actual_position_tolerance",.01))|(data.qpos[qa]>limits[:,1]+scene.get("actual_position_tolerance",.01))).any(): violations.add("actual_joint_limit")
        if (np.abs(data.qvel[va]) > vmax*(1+scene.get("actual_velocity_tolerance",.05))).any(): violations.add("actual_velocity_limit")
        acceleration = (data.qvel[va]-last_v)/model.opt.timestep
        if (np.abs(acceleration) > amax*(1+scene.get("actual_acceleration_tolerance",.1))).any(): violations.add("actual_acceleration_limit")
        peak_velocity=np.maximum(peak_velocity,np.abs(data.qvel[va]))
        peak_acceleration=np.maximum(peak_acceleration,np.abs(acceleration))
        peak_torque=np.maximum(peak_torque,np.abs(data.actuator_force[acts]))
        peak_tracking=np.maximum(peak_tracking,np.abs(data.qpos[qa]-cmd["q"][min(k+1,len(cmd["q"])-1)]))
        last_v = data.qvel[va].copy()
    if np.max(peak_tracking)>scene.get("max_tracking_error",np.inf):violations.add("tracking_error_limit")
    if np.max(np.abs(data.qvel[va]))>scene.get("final_brake_velocity_limit",np.inf):violations.add("braking_not_settled")
    if collision_pairs: violations.add("robot_self_collision")
    if result["post_release_robot_contact"]: violations.add("post_release_robot_recontact")
    if result["release"]:
        rv=np.asarray(result["release"]["linear_velocity"])
        if rv[2]<scene.get("min_release_vertical_velocity",-np.inf):violations.add("release_not_upward")
        if rv[0]<scene.get("min_release_forward_velocity",-np.inf):violations.add("release_not_forward")
    result["tracking"] = dict(peak_velocity=peak_velocity.tolist(),peak_acceleration=peak_acceleration.tolist(),
                              peak_torque=peak_torque.tolist(),peak_position_error=peak_tracking.tolist(),
                              final_joint_velocity=data.qvel[va].tolist())
    result["airborne_environment_contacts"]=airborne_environment_contacts
    from .basketball_metrics import flight_metrics
    result.update(flight_metrics(result,np.asarray(rows),bucket,rim))
    result["airborne_environment_contacts"]=airborne_environment_contacts
    if scene.get("require_high_arc",False):
        if not result["release"]:violations.add("no_free_release")
        elif result.get("apex_height_above_release",0)<scene.get("min_apex_rise",.2):violations.add("insufficient_apex_rise")
        if not result.get("apex_between_release_and_bucket",False):violations.add("apex_not_between_release_and_bucket")
        if not result.get("descending_entry_after_apex",False):violations.add("no_entry_after_apex")
        if any(not c["geom"].startswith("bucket_") or c["time"]<=result.get("apex_time",np.inf) for c in airborne_environment_contacts):violations.add("contact_before_rim_crossing")
    result["collision_pairs"] = sorted(collision_pairs)
    result["violations"] = sorted(violations)
    result["stayed_in_bucket"] = bool(final_containment and all(final_containment))
    result["release_command_time"] = release_command_time
    result["grasp_translation_drift_max_m"] = grasp_drift_max if release_mode=="contact" else None
    result["gross_slip_threshold_m"] = .02
    result["final_position"] = data.xpos[obj].tolist()
    result["success"] = bool(released and result["crossing"] and result["stayed_in_bucket"] and not violations and not result["slipped"])
    if not result["success"]:
        result["failure_reason"] = (";".join(sorted(violations)) if violations else
                                    "pre_release_slip" if result["slipped"] else "invalid_release" if not released else
                                    "no_descending_crossing" if not result["crossing"] else "not_settled_inside_bucket")
    result["flight_time"] = (result["crossing"]["time"]-result["release"]["time"] if result["crossing"] and result["release"] else None)
    if output_dir:
        out = Path(output_dir); out.mkdir(parents=True,exist_ok=True)
        (out/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False))
        (out/"scene.xml").write_text(xml)
        (out/"inputs.json").write_text(json.dumps(dict(object=object_config,scene=scene,params=cmd["parameters"]),indent=2))
        header = ["time"] + [f"{prefix}_{axis}" for prefix in ["geometry","com"] for axis in "xyz"]
        header += ["qw","qx","qy","qz"] + [f"{p}_{a}" for p in ["omega","velocity"] for a in "xyz"]
        header += [f"{p}_{i}" for p in ["q","qd","command"] for i in range(1,7)] + ["released"]
        with (out/"trajectory.csv").open("w") as f:
            w=csv.writer(f);w.writerow(header);w.writerows(rows)
        np.savez_compressed(out/"replay.npz",qpos=states,time=np.asarray(rows)[:,0],commands=cmd["q"],
                            release_time=cmd["release_time"], parameters=json.dumps(cmd["parameters"]))
    result["trajectory"] = np.asarray(rows)
    return result
