"""Optional behavior-cloning warm start from retargeted poses executed in physics.

The scripted controller is used only to generate demonstrations. Deployment and
PPO use the usual eight-dimensional policy, with no reference-action blending.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator

from .config import load_config
from .env import ThrowEnv
from .imitation import PosePrior
from .policy import ActorCritic, RunningMeanStd
from .retarget import retarget


def collect(config, output, episodes=80, seed=11000, video=False):
    torch.set_num_threads(1)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    env = ThrowEnv(config["env"])
    prior = PosePrior(config["env"]["imitation"])
    # FK reference fitting never modifies a physical rollout's state.
    fit_config = deepcopy(config["env"])
    fit_config["target"].update(low=None, high=None, position=[1., 0, 0])
    fitting = ThrowEnv(fit_config)
    try:
        fitting.reset(seed=0)
        poses, errors = retarget(fitting, prior)
    finally:
        fitting.close()
    curve = PchipInterpolator(np.linspace(0,1,len(poses)), poses, axis=0)
    observations, actions, results = [], [], []
    rng = np.random.default_rng(seed)
    try:
        for episode in range(episodes):
            obs, reset = env.reset(seed=seed+episode)
            initial = env.data.qpos[env.qadr].copy()
            yaw = np.arctan2(env.target[1],env.target[0])
            prep, swing = rng.uniform(.65,1.0), rng.uniform(.28,.65)
            release_phase = rng.uniform(.55,.85)
            writer = imageio.get_writer(root/"teacher.mp4",fps=50) if video and episode==0 else None
            try:
                while True:
                    t = env.data.time-env.start_time
                    if t < prep:
                        a = np.clip(t/prep,0,1)
                        blend = a*a*(3-2*a)
                        desired = initial*(1-blend)+(poses[0]+np.r_[yaw,np.zeros(6)])*blend
                        opening = -1.
                    else:
                        phase = np.clip((t-prep)/swing,0,1)
                        desired = curve(phase)+np.r_[yaw,np.zeros(6)]
                        opening = 1. if phase>=release_phase else -1.
                    period = env.config["timestep"]*env.config["substeps"]
                    action = np.r_[(desired-env.arm_target)/((env.upper-env.lower)*env.config["action_rate"]*period),opening].clip(-1,1)
                    observations.append(obs.copy())
                    actions.append(action.copy())
                    obs, _, terminated, truncated, info = env.step(action)
                    if writer:
                        writer.append_data(env.render())
                    if terminated or truncated:
                        results.append({"seed":seed+episode,"success":info["success"],"outcome":info["outcome"],
                           "landing":env.landing,"release":env.release,"mean_style_error_degrees":info["imitation_mean_error_degrees"],
                           "initialization":reset["initialization"],"prep_seconds":prep,"swing_seconds":swing,
                           "teacher_release_phase":release_phase})
                        break
            finally:
                if writer:writer.close()
    finally:
        env.close()
    np.savez_compressed(root/"demonstrations.npz",observations=np.array(observations,dtype=np.float32),actions=np.array(actions,dtype=np.float32))
    (root/"collection.json").write_text(json.dumps({"results":results,"mean_fk_error_degrees":float(np.mean(errors)),
        "method":"Scripted joint-target tracking used only for demonstration collection, through normal ThrowEnv.step; no object force or state injection. No filtering for hit success."},indent=2)+"\n")
    print(json.dumps({"episodes":episodes,"transitions":len(actions),"hits":sum(r["success"] for r in results)}))


def clone(checkpoint, config, dataset, output, updates=2000):
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    state = torch.load(checkpoint,map_location="cpu",weights_only=False)
    model = ActorCritic(state["normalizer"]["mean"].numel(),config["ppo"]["hidden"])
    normalizer = RunningMeanStd(state["normalizer"]["mean"].numel())
    model.load_state_dict(state["model"])
    normalizer.load_state_dict(state["normalizer"])
    data=np.load(dataset)
    obs=torch.tensor(data["observations"],dtype=torch.float32)
    target=torch.tensor(data["actions"],dtype=torch.float32)
    # Freeze the existing normalizer and critic to preserve value calibration.
    obs=normalizer(obs)
    optimizer=torch.optim.Adam(model.actor.parameters(),lr=1e-4)
    log=[]
    for step in range(updates):
        indices=torch.randint(len(obs),(min(256,len(obs)),))
        predicted=model.actor(obs[indices]).tanh()
        loss=(predicted-target[indices]).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        if step%100==0:log.append({"update":step,"action_mse":float(loss.detach())})
    state["model"]=model.state_dict()
    state["config"]=config
    # Old PPO moments belong to the pre-BC actor. Start fresh moments for PPO.
    fresh=torch.optim.Adam(model.parameters(),lr=config["ppo"]["learning_rate"],eps=1e-5)
    state["optimizer"]=fresh.state_dict()
    state["best_success"],state["best_error"]=-1.,float("inf")
    state["bc"]={"updates":updates,"dataset":str(dataset),"transitions":len(obs),"log":log}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    torch.save(state,output)
    Path(str(output)+".json").write_text(json.dumps(state["bc"],indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="mode",required=True)
    for name in ("collect","clone"):
        p=sub.add_parser(name);p.add_argument("--config",required=True);p.add_argument("--output",required=True)
        if name=="collect":
            p.add_argument("--episodes",type=int,default=80);p.add_argument("--video",action="store_true")
        else:
            p.add_argument("--checkpoint",required=True);p.add_argument("--dataset",required=True);p.add_argument("--updates",type=int,default=2000)
    a=parser.parse_args();config=load_config(a.config)
    if a.mode=="collect":collect(config,a.output,a.episodes,video=a.video)
    else:clone(a.checkpoint,config,a.dataset,a.output,a.updates)


if __name__=="__main__":main()
