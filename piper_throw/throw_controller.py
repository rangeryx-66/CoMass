"""Six-axis parameterized throws; no learning and no hardware dependencies."""
from dataclasses import dataclass, asdict
import hashlib
import numpy as np
from .quintic import quintic


@dataclass(frozen=True)
class ThrowParams:
    speed_scale: float = 1.0
    release_phase: float = .48
    swing_amplitude: float = 1.1
    direction_offset: float = 0.
    wrist_pitch_offset: float = 0.
    family: str = "eccentric"

    def to_dict(self):
        return asdict(self)


def make_commands(params, scene_config):
    if isinstance(params,dict) and params.get("family")=="trebuchet":
        from .range_throw import make_range_commands
        return make_range_commands(params,scene_config)
    if isinstance(params,dict) and params.get("family")=="basketball":
        from .basketball_controller import make_arc_commands
        return make_arc_commands(params,scene_config)
    p = params if isinstance(params, ThrowParams) else ThrowParams(**params)
    vals = [p.speed_scale, p.release_phase, p.swing_amplitude, p.direction_offset, p.wrist_pitch_offset]
    if not np.isfinite(vals).all() or p.speed_scale <= 0 or not 0 < p.release_phase < 1 or p.swing_amplitude <= 0:
        raise ValueError("invalid throw parameters")
    if p.family not in {"translation", "planar", "eccentric", "wrist"}:
        raise ValueError("unknown action family")
    dt = scene_config.get("timestep", .002)
    T = scene_config.get("swing_duration", .8) / p.speed_scale
    q0 = np.array(scene_config.get("initial_joints", [0, .30, -2., 0, .35, 0]), dtype=float)
    q0[0] += p.direction_offset
    q0[4] += p.wrist_pitch_offset
    qf = q0.copy(); qf[1] += p.swing_amplitude
    if p.family == "translation":
        qf[4] -= p.swing_amplitude  # approximate orientation-preserving planar family
    elif p.family == "wrist":
        qf[4] += .45 * p.swing_amplitude
    commands, velocities, accelerations = [], [], []
    settle = scene_config.get("settle_time", .3)
    horizon = settle + T + scene_config.get("flight_settle_time", 4.0)
    times = np.arange(int(np.ceil(horizon/dt)) + 1) * dt
    release_time=settle+p.release_phase*T
    brake=scene_config.get("brake_after_release",0.)
    qr,vr,_=quintic(p.release_phase*T,T,q0,np.zeros(6),qf,np.zeros(6))
    qstop=qr+.5*vr*brake
    for t in times:
        if brake and t>release_time:
            q,v,a=quintic(t-release_time,brake,qr,vr,qstop,np.zeros(6))
        else:
            q, v, a = quintic(t-settle, T, q0, np.zeros(6), qf, np.zeros(6))
        commands.append(q); velocities.append(v); accelerations.append(a)
    cmds = np.asarray(commands)
    return dict(time=times, q=cmds, qd=np.asarray(velocities), qdd=np.asarray(accelerations),
                release_time=settle+p.release_phase*T,
                command_sha256=hashlib.sha256(cmds.tobytes()+np.asarray([settle+p.release_phase*T]).tobytes()).hexdigest(),
                parameters=p.to_dict())


def jacobian_release_velocity(model, data, point_world, body_id, desired_velocity):
    """MuJoCo six-axis Jacobian solve, available for ballistic warm-start planning."""
    import mujoco
    J = np.zeros((3, model.nv)); Jr = np.zeros_like(J)
    mujoco.mj_jac(model, data, J, Jr, np.asarray(point_world), body_id)
    dofs = [model.joint(f"joint{i}").dofadr[0] for i in range(1,7)]
    A = J[:, dofs]
    return A.T @ np.linalg.solve(A@A.T + 1e-4*np.eye(3), desired_velocity)
