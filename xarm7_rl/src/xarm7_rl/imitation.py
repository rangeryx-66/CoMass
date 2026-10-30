"""Optional, phase-free soft pose reward from annotated monocular video.

Only projected link directions are observed in the source. This is not a
reconstruction of the source robot's 3-D joint angles or physical motion speed.
"""
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator


MOTIONS = Path(__file__).parent / "assets" / "motions"


def target_frame(target, shoulder):
    forward = np.r_[np.asarray(target)[:2] - np.asarray(shoulder)[:2], 0.]
    length = np.linalg.norm(forward)
    forward = forward / length if length > 1e-9 else np.array([1., 0., 0.])
    return forward, np.array([0., 0., 1.])


def projected_directions(points, forward, up):
    bones = np.diff(np.asarray(points), axis=0)
    projection = np.stack([bones @ forward, bones @ up], axis=-1)
    return projection / np.maximum(np.linalg.norm(projection, axis=-1, keepdims=True), 1e-9)


class PosePrior:
    def __init__(self, config):
        name = config.get("reference", "mit_overhand")
        path = MOTIONS / f"{name}.json" if "/" not in name and not name.endswith(".json") else Path(name)
        self.source = json.loads(path.read_text())
        observations = np.asarray([r["points_pixels"] for r in self.source["keyframes"]], dtype=float)
        bones = np.diff(observations, axis=1)
        forward = np.asarray(self.source["image_forward"], dtype=float)
        up = np.asarray(self.source["image_up"], dtype=float)
        projected = np.stack([bones @ forward, bones @ up], axis=-1)
        angles = np.unwrap(np.arctan2(projected[..., 0], projected[..., 1]), axis=0)
        # The source has edited slow motion and freeze frames. Its order is useful;
        # its presentation timestamps must not be treated as physical velocities.
        phase = np.linspace(0, 1, len(observations))
        self.angles = PchipInterpolator(phase, angles, axis=0)(np.linspace(0, 1, config.get("samples", 64)))
        self.directions = np.stack([np.sin(self.angles), np.cos(self.angles)], axis=-1)
        self.weights = np.asarray(config.get("bone_weights", [1., 1., .5]), dtype=float)
        self.sigma = np.deg2rad(config.get("angular_sigma_degrees", 25.))
        if self.weights.shape != (3,) or np.any(self.weights < 0) or self.weights.sum() <= 0 or self.sigma <= 0:
            raise ValueError("Pose weights and angular bandwidth must be positive")
        self.weights /= self.weights.sum()
        self.scale = float(config.get("reward_scale", 0.))
        self.terminal_scale = float(config.get("terminal_reward_scale", 0.))
        self.release_scale = float(config.get("release_reward_scale", 0.))
        if min(self.scale, self.terminal_scale, self.release_scale) < 0:
            raise ValueError("Imitation reward scales must be nonnegative")

    def compare(self, points, target):
        forward, up = target_frame(target, points[0])
        actual = projected_directions(points, forward, up)
        cosine = np.einsum("tbi,bi->tb", self.directions, actual).clip(-1, 1)
        errors = np.arccos(cosine)
        squared = np.sum(errors ** 2 * self.weights, axis=1)
        index = int(squared.argmin())
        error = float(np.sqrt(squared[index]))
        return {"score": float(np.exp(-.5 * error ** 2 / self.sigma ** 2)),
                "error_degrees": float(np.rad2deg(error)), "reference_index": index}

    def reward(self, score, elapsed):
        # A bounded deviation cost avoids paying the policy to park in one pose.
        # Physical elapsed time keeps reward scale independent of integrator dt.
        return self.scale * elapsed * (score - 1.)

    def terminal_reward(self, mean_score, release_score):
        return self.terminal_scale * (mean_score - 1.) + self.release_scale * (release_score - 1.)
