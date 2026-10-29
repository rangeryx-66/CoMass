"""Ballistics -> bounded MuJoCo Jacobian -> acceleration/braking trajectory.

Only commands are planned here; object free velocities are never assigned.
"""
import hashlib
import numpy as np
from scipy.optimize import lsq_linear
import mujoco
from .quintic import quintic


def gripper_commands(times,release_time,scene):
    duration=scene.get('gripper_motion_time',0.)
    closed=scene.get('gripper_closed',.015)
    if not duration:
        return np.where(times<release_time,closed,.035)
    close=np.clip(times/duration,0,1);opening=np.clip((times-release_time)/duration,0,1)
    close=close*close*(3-2*close);opening=opening*opening*(3-2*opening)
    return np.where(times<release_time,.025+(closed-.025)*close,closed+(.035-closed)*opening)


def payload_support_fraction(times,release_time,scene):
    lead=scene.get('release_lead_s',0.) if scene.get('release_mode','ideal')=='contact' else 0.
    if not lead:
        return (times<release_time).astype(float)
    u=np.clip((times-release_time)/lead,0,1)
    return 1-u*u*(3-2*u)


def ballistic_plan(release_position, rim_center, *, rise=.25, gravity=9.81):
    p=np.asarray(release_position,dtype=float);target=np.asarray(rim_center,dtype=float)
    if p.shape!=(3,) or target.shape!=(3,) or not np.isfinite([*p,*target,rise,gravity]).all() or gravity<=0 or rise<=0:
        raise ValueError("invalid ballistic boundary conditions")
    rise=max(float(rise),float(target[2]-p[2])+.01)
    vz=np.sqrt(2*gravity*rise)
    t=(vz+np.sqrt(vz*vz+2*gravity*(p[2]-target[2])))/gravity
    v=np.r_[(target[:2]-p[:2])/t,vz]
    return dict(position=p.tolist(),velocity=v.tolist(),flight_time=float(t),
                time_to_apex=float(vz/gravity),rise=float(rise),
                apex=(p+v*vz/gravity-np.array([0,0,.5*vz*vz/gravity])).tolist(),
                horizontal_distance=float(np.linalg.norm(target[:2]-p[:2])))


def kinematic_release(model,data,q,scene):
    ids=[model.joint(f"joint{i}").id for i in range(1,7)]
    qa=model.jnt_qposadr[ids];va=model.jnt_dofadr[ids]
    data.qpos[qa]=q;mujoco.mj_forward(model,data)
    hand=model.body('link6').id
    point=data.xpos[hand]+data.xmat[hand].reshape(3,3)@np.asarray(scene.get('grasp_offset',[.025,0,.105]))
    J=np.zeros((3,model.nv));R=J.copy();mujoco.mj_jac(model,data,J,R,point,hand)
    return point.copy(),J[:,va].copy(),R[:,va].copy()


def solve_release_velocity(J,desired,vmax,angular_jacobian=None,angular_weight=.015):
    A=np.vstack([J,.03*np.eye(6)] if angular_jacobian is None else [J,angular_weight*angular_jacobian,.03*np.eye(6)])
    b=np.r_[desired,np.zeros(6)] if angular_jacobian is None else np.r_[desired,np.zeros(9)]
    r=lsq_linear(A,b,bounds=(-np.asarray(vmax),np.asarray(vmax)),tol=1e-10)
    return r.x,float(np.linalg.norm(J@r.x-desired))


def smooth_acceleration_progress(s,edge=.12):
    """Integral of a smooth, nearly constant acceleration pulse (C1 acceleration)."""
    s=float(np.clip(s,0,1));a=edge
    if not 0<a<.5:raise ValueError('invalid acceleration edge fraction')
    if s>a and s<1-a:
        return (s*s/2-a*s/2+.15*a*a)/(1-a),(s-a/2)/(1-a),1/(1-a)
    if s>=1-a:
        H,F,A=smooth_acceleration_progress(1-s,a)
        return s-.5+H,1-F,A
    u=s/a
    return a*a*(u**4/4-u**5/10)/(1-a),a*(u**3-.5*u**4)/(1-a),(3*u*u-2*u**3)/(1-a)


def make_arc_commands(params,scene):
    q=np.asarray(params['release_joints'],dtype=float);v=np.asarray(params['release_joint_velocity'],dtype=float)
    if q.shape!=(6,) or v.shape!=(6,) or not np.isfinite([*q,*v]).all():raise ValueError('invalid release joints')
    dt=float(scene.get('timestep',.002));amax=np.asarray(scene.get('acceleration_limits',[5]*6))
    ramp=float(params.get('ramp_time',max(.35,1.5*np.max(np.abs(v)/amax))))
    brake=float(params.get('brake_time',ramp))
    if min(ramp,brake)<=0:raise ValueError('invalid acceleration/braking time')
    settle=scene.get('settle_time',.6)
    ready=np.asarray(scene.get('ready_joints',[0,.8,-1.8,0,.5,0]),dtype=float)
    q0=q-.5*v*ramp
    delta=np.max(np.abs(q0-ready))
    prepare=max(1.,np.sqrt(8*delta/min(amax)),1.9*delta/min(scene.get('velocity_limits',[3]*6)))
    start=settle+prepare+scene.get('windup_settle_time',.3)
    nominal=start+ramp
    times=np.arange(int(np.ceil((nominal+max(brake,scene.get('flight_settle_time',4)))/dt))+1)*dt
    qs=[];vs=[];acs=[];q0=q-.5*v*ramp
    for t in times:
        if t<settle:x=ready;xd=np.zeros(6);xdd=xd
        elif t<settle+prepare:x,xd,xdd=quintic(t-settle,prepare,ready,np.zeros(6),q0,np.zeros(6))
        elif t<start:x=q0;xd=np.zeros(6);xdd=xd
        elif t<=nominal:
            s=(t-start)/ramp
            if params.get('profile')=='smooth_acceleration':
                H,F,A=smooth_acceleration_progress(s);x=q0+v*ramp*H;xd=v*F;xdd=v/ramp*A
            else:
                x=q0+v*ramp*(s**3-.5*s**4);xd=v*(3*s*s-2*s**3);xdd=v/ramp*(6*s-6*s*s)
        else:
            s=np.clip((t-nominal)/brake,0,1)
            if params.get('profile')=='smooth_acceleration':
                H,F,A=smooth_acceleration_progress(s);x=q+v*brake*(s-H);xd=v*(1-F);xdd=-v/brake*A
            else:
                x=q+v*brake*(s-s**3+.5*s**4);xd=v*(1-3*s*s+2*s**3);xdd=v/brake*(-6*s+6*s*s)
        qs.append(x);vs.append(xd);acs.append(xdd)
    qs=np.asarray(qs);release=nominal-float(params.get('open_lead',0))
    if release<start:raise ValueError('opening precedes swing')
    return dict(time=times,q=qs,qd=np.asarray(vs),qdd=np.asarray(acs),release_time=release,
                gripper=gripper_commands(times,release,scene),
                command_sha256=hashlib.sha256(qs.tobytes()+np.array([release]).tobytes()).hexdigest(),parameters=dict(params))
