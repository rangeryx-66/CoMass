"""Run unmodified upstream ThrowController and preserve actual execution states."""
import json
import sys
from pathlib import Path
import numpy as np
import mujoco

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/"upstream/openarm-control"))
from openarm_control.config import THROW_SCENE
from openarm_control.throwing import ThrowController


def main():
    out=root.parent/"results/openarm_original";out.mkdir(parents=True,exist_ok=True)
    m=mujoco.MjModel.from_xml_path(THROW_SCENE);d=mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,m.key("ready").id);mujoco.mj_forward(m,d)
    tc=ThrowController(m,d,ball="ball_orange")
    assert tc.grasp_ball()
    bp=d.xpos[m.body("bin").id].copy();target=np.r_[bp[:2],.06]
    assert tc.plan_release(target) is not None,tc.reason
    states=[];times=[];original=mujoco.mj_step
    def record(model,data,*args,**kwargs):
        original(model,data,*args,**kwargs);states.append(data.qpos.copy());times.append(data.time)
    mujoco.mj_step=record
    try:landing=tc.execute()
    finally:mujoco.mj_step=original
    report=dict(commit="14ea59faad6706b9add5d7f8d386e43239a6360f",mujoco=mujoco.__version__,
                target=target.tolist(),landing=landing.tolist(),error_m=float(np.linalg.norm(landing[:2]-target[:2])),
                predicted_error_m=tc.pred_err,release_velocity=tc.release_vel.tolist(),release_step=tc.release_step,
                success=bool(np.linalg.norm(landing[:2]-target[:2])<.08 and landing[2]<.2),scene=str(THROW_SCENE))
    np.savez_compressed(out/"replay.npz",qpos=states,time=times)
    (out/"result.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=="__main__":main()
