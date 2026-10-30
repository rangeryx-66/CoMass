"""Paired time-step experiments with shared actions and optional shared initial state."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .config import load_config
from .env import ThrowEnv
from .policy import ActorCritic, RunningMeanStd


class TracedEnv(ThrowEnv):
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.trace = []
        self.contact_names = []
        return obs, info

    def _physics_step(self):
        previous_velocity = self.data.cvel[self.object_body, 3:].copy()
        super()._physics_step()
        pads, hand, floor, robot = self._contacts()
        normal_force = 0.0
        contact = np.zeros(6)
        names = set()
        for index, item in enumerate(self.data.contact):
            if self.object_geom in item.geom and item.dist <= 0:
                if self.floor_geom in item.geom:
                    continue
                other = item.geom[1] if item.geom[0] == self.object_geom else item.geom[0]
                names.add(self.model.geom(other).name or self.model.body(self.model.geom_bodyid[other]).name)
                mujoco.mj_contactForce(self.model, self.data, index, contact)
                normal_force += max(0.0, float(contact[0]))
        impulse = self.mass * (self.data.cvel[self.object_body, 3:] - previous_velocity - self.model.opt.gravity * self.model.opt.timestep)
        self.contact_names.append(";".join(sorted(names)))
        self.trace.append(np.r_[
            self.data.time - self.start_time,
            self.data.xipos[self.object_body], self.data.cvel[self.object_body, 3:],
            self.data.xpos[self.object_body], self.data.xquat[self.object_body],
            self.data.cvel[self.object_body, :3], self.data.site_xpos[self.tcp],
            [float(hand), float(robot), float(bool(floor)), float(len(pads)), normal_force],
            self.data.qpos, self.data.qvel, self.data.ctrl,
            impulse,
            [self.data.solver_niter[0]],
        ])


def initial_snapshot(env):
    return {"qpos": env.data.qpos.copy(), "qvel": env.data.qvel.copy(),
            "act": env.data.act.copy(), "ctrl": env.data.ctrl.copy(),
            "arm_target": env.arm_target.copy(), "object_com": env.data.xipos[env.object_body].copy(),
            "object_velocity": env.data.cvel[env.object_body, 3:].copy()}


def apply_initial_snapshot(env, snapshot):
    for name in ("qpos", "qvel", "act", "ctrl"):
        getattr(env.data, name)[:] = snapshot[name]
    env.arm_target = snapshot["arm_target"].copy()
    env.data.qacc_warmstart[:] = 0
    mujoco.mj_forward(env.model, env.data)
    env.ever_contact = bool(env._contacts()[1])
    env.previous_prediction_score = env._prediction_score()


def flight_residual(trace):
    # Trace layout: time; CoM p,v; origin p; quaternion; omega; TCP; contact flags/force.
    t = trace[:, 0]
    first_floor = np.flatnonzero(trace[:, 22] > 0)
    stop = int(first_floor[0]) if len(first_floor) else len(trace)
    contacts = np.flatnonzero(trace[:stop, 21] > 0)
    start = int(contacts[-1]) + 1 if len(contacts) else 0
    if stop - start < 3:
        return None
    dt = t[start:stop] - t[start]
    prediction = trace[start, 1:4] + dt[:, None] * trace[start, 4:7]
    prediction[:, 2] -= .5 * 9.81 * dt ** 2
    error = np.linalg.norm(trace[start:stop, 1:4] - prediction, axis=1)
    return {"last_robot_contact_time": float(t[start - 1]) if start else None,
            "free_flight_seconds": float(dt[-1]), "max_ballistic_residual_m": float(error.max())}


def rollout(config, model, normalizer, seed, output, *, actions=None, snapshot=None):
    env = TracedEnv(config["env"])
    try:
        obs, reset = env.reset(seed=seed)
        if snapshot is not None:
            apply_initial_snapshot(env, snapshot)
            obs = env._observation()
        initial = initial_snapshot(env)
        sequence = []
        while True:
            if actions is None:
                with torch.no_grad():
                    tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    action = model.act(normalizer(tensor), deterministic=True)[0][0].numpy()
            else:
                action = actions[min(len(sequence), len(actions) - 1)]
            sequence.append(action.copy())
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        trace = np.asarray(env.trace)
        np.savez_compressed(str(output) + ".npz", trace=trace, actions=np.asarray(sequence),
                            contact_geometries=np.array(env.contact_names),
                            **{f"initial_{k}": v for k, v in initial.items()},
                            nq=env.model.nq, nv=env.model.nv, nu=env.model.nu)
        summary = {"timestep": config["env"]["timestep"], "substeps": config["env"]["substeps"],
                   "initialization": reset["initialization"], "initial_object_com": initial["object_com"].tolist(),
                   "initial_qpos": initial["qpos"].tolist(), "initial_qvel": initial["qvel"].tolist(),
                   "release": env.release, "landing": env.landing, "outcome": info["outcome"],
                   "ballistic": flight_residual(trace), "max_solver_iterations": int(trace[:, -1].max()),
                   "object_contact_normal_impulse": float(trace[:, 24].sum() * config["env"]["timestep"])}
        Path(str(output) + ".json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary, np.asarray(sequence), initial
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--actions", help="NPZ containing a frozen actions array, e.g. a legacy rollout")
    parser.add_argument("--timesteps", type=float, nargs="+", default=[.002, .001, .0005, .00025])
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_config(args.config)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    size = state["normalizer"]["mean"].numel()
    model = ActorCritic(size, config["ppo"]["hidden"])
    normalizer = RunningMeanStd(size)
    model.load_state_dict(state["model"])
    normalizer.load_state_dict(state["normalizer"])
    model.eval()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    results, sequences, snapshots = {}, {}, {}
    period = config["env"]["timestep"] * config["env"]["substeps"]
    for dt in args.timesteps:
        cfg = deepcopy(config)
        cfg["env"].update(timestep=dt, substeps=round(period / dt))
        name = f"closed_{dt:g}"
        summary, actions, initial = rollout(cfg, model, normalizer, args.seed, root / name)
        results[name], sequences[dt], snapshots[dt] = summary, actions, initial
    # Replay the same 2 ms actions, first with native initial states, then with
    # one common finest-step settled physical state. No policy is retrained here.
    for shared in (False, True):
        for dt in args.timesteps:
            cfg = deepcopy(config)
            cfg["env"].update(timestep=dt, substeps=round(period / dt))
            name = f"open_{'common' if shared else 'native'}_{dt:g}"
            replay_actions = np.load(args.actions)["actions"] if args.actions else sequences[.002]
            results[name] = rollout(cfg, model, normalizer, args.seed, root / name,
                                    actions=replay_actions, snapshot=snapshots[min(args.timesteps)] if shared else None)[0]
    (root / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    for key, result in results.items():
        print(json.dumps({"case": key, "release": result["release"], "landing": result["landing"], "ballistic": result["ballistic"]}), flush=True)


if __name__ == "__main__":
    main()
