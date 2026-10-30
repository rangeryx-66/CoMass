from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from xarm7_rl.config import load_config
from xarm7_rl.env import ThrowEnv
from xarm7_rl.imitation import PosePrior


ROOT = Path(__file__).parents[1]


def test_projected_imitation_preserves_target_rotation_and_scale():
    prior = PosePrior({"reward_scale": 4})
    directions = prior.directions[28]
    lengths = np.array([.29, .35, .20])
    points = np.zeros((4, 3))
    points[1:, 0] = np.cumsum(directions[:, 0] * lengths)
    points[1:, 2] = np.cumsum(directions[:, 1] * lengths)
    target = np.array([2., 0, 0])
    assert prior.compare(points, target)["score"] == pytest.approx(1, abs=1e-12)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    offset = np.array([.1, .2, .3])
    moved = 2 * points @ rotation.T + offset
    assert prior.compare(moved, 2 * target @ rotation.T + offset)["score"] == pytest.approx(1, abs=1e-12)
    downward = np.array([[0, 0, 0], [0, 0, -.3], [0, 0, -.6], [0, 0, -.8]])
    assert prior.compare(downward, target)["score"] < .01


def test_imitation_reward_cannot_pay_for_parking_and_uses_physical_time():
    prior = PosePrior({"reward_scale": 4})
    assert prior.reward(1, .02) == 0
    assert prior.reward(.5, .02) < 0
    assert 10 * prior.reward(.5, .002) == pytest.approx(20 * prior.reward(.5, .001))
    terminal = PosePrior({"terminal_reward_scale":4,"release_reward_scale":4})
    assert terminal.terminal_reward(1,1) == 0
    assert terminal.terminal_reward(0,0) == -8


@pytest.mark.parametrize("weight", [0, 4])
def test_imitation_preserves_actions_physics_and_observations(weight):
    config = load_config(ROOT / "configs/default.yaml")["env"]
    styled = deepcopy(config)
    styled["imitation"] = {"reference": "mit_overhand", "reward_scale": weight,
                           "terminal_reward_scale":weight,"release_reward_scale":weight}
    baseline, observed = ThrowEnv(config), ThrowEnv(styled)
    saved = np.load(ROOT / "verification/default/trajectory.npz")
    offset = 1 + int(saved["nq"]) + int(saved["nv"])
    try:
        assert np.array_equal(baseline.reset(seed=1000)[0], observed.reset(seed=1000)[0])
        for action in saved["trajectory"][:, offset:offset+8]:
            a, b = baseline.step(action), observed.step(action)
            assert np.array_equal(a[0], b[0])
            assert a[2:4] == b[2:4]
            if weight == 0:
                assert a[1] == b[1]
            else:
                assert b[1] <= a[1]
            assert np.array_equal(baseline.data.qpos, observed.data.qpos)
            assert observed.action_space == baseline.action_space
            assert b[4]["imitation_reward"] <= 0
            if a[2] or a[3]:
                break
    finally:
        baseline.close()
        observed.close()
