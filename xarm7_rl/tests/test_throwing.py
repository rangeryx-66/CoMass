from copy import deepcopy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
import trimesh

from xarm7_rl.config import load_config
from xarm7_rl.env import ASSETS, ThrowEnv
from xarm7_rl.objects import object_geometry
from xarm7_rl.policy import ActorCritic
from xarm7_rl.train import compute_gae

ROOT = Path(__file__).parents[1]


@pytest.fixture
def config():
    return load_config(ROOT / "configs/default.yaml")


def test_asset_provenance():
    manifest = json.loads((ASSETS / "SOURCE.json").read_text())
    for entry in manifest["files"]:
        data = (ASSETS / entry["path"]).read_bytes()
        assert len(data) == entry["size"]
        assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == entry["sha"]


def test_real_native_grasp_and_release(config):
    env = ThrowEnv(config["env"])
    try:
        obs, info = env.reset(seed=10)
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(obs).all()
        assert info["initialization"]["bilateral_contact"]
        assert not info["initialization"]["on_floor"]
        initial = env.data.xpos[env.object_body].copy()
        for _ in range(20):
            _, _, term, trunc, info = env.step(np.r_[np.zeros(7), -1])
            assert not term and not trunc
        assert np.linalg.norm(env.data.xpos[env.object_body] - initial) < .02
        for _ in range(200):
            _, _, term, trunc, info = env.step(np.r_[np.zeros(7), 1])
            if term or trunc:
                break
        assert env.release is not None
        assert env.landing is not None
        assert env.release["time"] < env.data.time - env.start_time
        # Only native gripper linkage equalities; no object weld or object actuator.
        for equality in range(env.model.neq):
            assert env.model.eq_type[equality] != mujoco.mjtEq.mjEQ_WELD
        assert env.model.nu == 8
    finally:
        env.close()


def test_all_seven_joints_and_gripper_are_controllable(config):
    env = ThrowEnv(config["env"])
    try:
        env.reset(seed=0)
        before = env.arm_target.copy()
        env.step(np.r_[np.ones(7) * .1, .5])
        assert np.all(env.arm_target > before)
        assert env.data.ctrl[env.grip_act] == pytest.approx(63.75)
    finally:
        env.close()


@pytest.mark.parametrize("shape,size", [("box", [.02, .02, .025]), ("sphere", [.02]), ("cylinder", [.02, .025]), ("capsule", [.016, .02]), ("ellipsoid", [.02, .018, .025])])
def test_object_properties_reach_physics_and_observation(config, shape, size):
    cfg = deepcopy(config["env"])
    cfg["object"].update(type=shape, size=size, mass=.15, com=[.002, -.001, .001])
    cfg["target"]["position"] = [.8, .2, 0]
    env = ThrowEnv(cfg)
    try:
        obs, _ = env.reset(seed=2)
        assert env.model.body_mass[env.object_body] == pytest.approx(.15)
        assert np.allclose(env.model.body_ipos[env.object_body], [.002, -.001, .001])
        assert env.model.geom_type[env.object_geom] == getattr(mujoco.mjtGeom, "mjGEOM_" + shape.upper())
        assert np.allclose(obs[29:32], env.target - env.data.xpos[env.object_body])
        assert obs[32] == pytest.approx(.15)
        assert np.allclose(obs[33:36], [.002, -.001, .001])
        assert np.isfinite(obs).all()
    finally:
        env.close()


def test_custom_mesh_geometry(config, tmp_path):
    path = tmp_path / "object.stl"
    trimesh.creation.box(extents=[.04, .035, .05]).export(path)
    cfg = deepcopy(config["env"])
    cfg["object"].update(type="mesh", mesh=str(path), mass=.08, com=[.002, 0, 0])
    env = ThrowEnv(cfg)
    try:
        obs, _ = env.reset(seed=1)
        assert env.model.geom_type[env.object_geom] == mujoco.mjtGeom.mjGEOM_MESH
        assert np.isfinite(obs).all()
        assert obs[32] == pytest.approx(.08)
    finally:
        env.close()


def test_randomization_is_seeded_and_changes_mass_com_target(config):
    cfg = deepcopy(config["env"])
    cfg["randomization"].update(mass_range=[.06, .15], com_low=[-.002] * 3, com_high=[.002] * 3)
    cfg["target"].update(low=[.7, -.3, 0], high=[1.2, .3, 0])
    env = ThrowEnv(cfg)
    try:
        first, _ = env.reset(seed=123)
        repeated, _ = env.reset(seed=123)
        assert np.allclose(first, repeated)
        changed, _ = env.reset(seed=124)
        assert not np.allclose(first[29:36], changed[29:36])
    finally:
        env.close()


def test_timeout_is_not_a_hit(config):
    cfg = deepcopy(config["env"])
    cfg["episode_seconds"] = .02
    env = ThrowEnv(cfg)
    try:
        env.reset(seed=0)
        _, _, terminated, truncated, info = env.step(np.r_[np.zeros(7), -1])
        assert truncated and not terminated
        assert not info["success"]
    finally:
        env.close()


def test_gae_bootstraps_time_limits_but_not_true_terminal():
    reward = torch.tensor([[1., 1.]])
    value = torch.tensor([[2., 2.]])
    next_value = torch.tensor([[10., 10.]])
    advantage, returns = compute_gae(reward, value, next_value, torch.tensor([[True, False]]), torch.tensor([[False, True]]), .9, .95)
    assert torch.allclose(returns, torch.tensor([[1., 10.]]))


def test_tanh_policy_log_probability_and_gradients():
    torch.manual_seed(0)
    model = ActorCritic(149)
    obs = torch.randn(8, 149)
    action, latent, old_logprob, value = model.act(obs)
    new_logprob, entropy, new_value = model.evaluate(obs, latent)
    assert torch.allclose(old_logprob, new_logprob)
    assert (action.abs() <= 1).all()
    loss = -new_logprob.mean() + new_value.square().mean() - .001 * entropy.mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
