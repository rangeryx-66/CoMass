"""A fixed xArm7 starts holding an object and learns to throw it at a target."""
from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from .config import merge_object
from .objects import SHAPES, object_geometry

ASSETS = Path(__file__).parent / "assets" / "ufactory_xarm7"


def numbers(values):
    return " ".join(f"{float(v):.12g}" for v in values)


class ThrowEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, config, render_mode=None):
        self.config = deepcopy(config)
        self.render_mode = render_mode
        self.renderer = None
        self.action_space = spaces.Box(-1, 1, (8,), np.float32)
        # q,dq:14; gripper:2; object pose/velocity:13; target:3; mass:1;
        # com:3; inertia:6; geometry:3*K; time/release/contact:3; previous action:8.
        dimension = 53 + 3 * config["geometry_points"]
        self.observation_space = spaces.Box(-np.inf, np.inf, (dimension,), np.float32)
        self.model = self.data = None
        self.active_object = None
        self.last_initialization = None

    def _sample_spec(self):
        cfg = self.config
        randomization = cfg.get("randomization", {})
        choices = randomization.get("objects", [])
        spec = merge_object(cfg["object"], choices[self.np_random.integers(len(choices))] if choices else {})
        if randomization.get("mass_range"):
            spec["mass"] = float(self.np_random.uniform(*randomization["mass_range"]))
        if randomization.get("com_low") is not None:
            spec["com"] = self.np_random.uniform(randomization["com_low"], randomization["com_high"]).tolist()
        if randomization.get("size_scale_range"):
            scale = self.np_random.uniform(*randomization["size_scale_range"])
            spec["size"] = (np.asarray(spec["size"]) * scale).tolist()
            if spec.get("mesh"):
                spec["mesh_scale"] = (np.asarray(spec.get("mesh_scale", [1, 1, 1])) * scale).tolist()
        return spec

    def _build(self, spec):
        mesh, self.mass, self.com, inertia, self.geometry = object_geometry(spec, self.config["geometry_points"])
        self.inertia = inertia[np.triu_indices(3)]
        root = ET.parse(ASSETS / "xarm7.xml").getroot()
        root.find("compiler").set("meshdir", str(ASSETS / "assets"))
        root.find("option").set("timestep", str(self.config["timestep"]))
        keyframe = root.find("keyframe")
        root.remove(keyframe)
        world = root.find("worldbody")
        base = world.find("body")
        base.set("pos", numbers(np.asarray([0, 0, .12]) + self.config["base_translation"]))
        ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 .1", pos=numbers([0, 0, self.target[2]]), rgba=".25 .28 .3 1", friction=".8 .01 .001")
        ET.SubElement(world, "light", pos="0 0 3", dir="0 0 -1")
        ET.SubElement(world, "site", name="target", pos=numbers(self.target + [0, 0, .002]), size=numbers([self.config["target"]["success_radius"], .002]), type="cylinder", rgba=".1 .9 .2 .8")
        body = ET.SubElement(world, "body", name="object", pos="0 0 2")
        ET.SubElement(body, "freejoint", name="object_free")
        full = [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]
        ET.SubElement(body, "inertial", pos=numbers(self.com), mass=str(self.mass), fullinertia=numbers(full))
        attrs = dict(name="object_geom", type=spec["type"], friction=numbers(spec.get("friction", [.8, .01, .001])), rgba=".85 .25 .08 1")
        if spec["type"] == "mesh":
            asset = root.find("asset")
            ET.SubElement(asset, "mesh", name="throw_object_mesh", file=str(Path(spec["mesh"]).resolve()), scale=numbers(spec.get("mesh_scale", [1, 1, 1])))
            attrs["mesh"] = "throw_object_mesh"
        else:
            attrs["size"] = numbers(spec["size"])
        ET.SubElement(body, "geom", **attrs)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.arm_joints = np.array([self.model.joint(f"joint{i}").id for i in range(1, 8)])
        self.qadr = self.model.jnt_qposadr[self.arm_joints]
        self.vadr = self.model.jnt_dofadr[self.arm_joints]
        self.arm_act = np.array([self.model.actuator(f"act{i}").id for i in range(1, 8)])
        self.grip_act = self.model.actuator("gripper").id
        self.grip_joint = self.model.joint("left_driver_joint").id
        self.object_body = self.model.body("object").id
        self.object_geom = self.model.geom("object_geom").id
        self.floor_geom = self.model.geom("floor").id
        self.object_qadr = self.model.joint("object_free").qposadr[0]
        self.object_vadr = self.model.joint("object_free").dofadr[0]
        self.gripper_body = self.model.body("xarm_gripper_base_link").id
        self.tcp = self.model.site("link_tcp").id
        self.pad_geoms = {self.model.geom(name).id for name in ("left_finger_pad_1", "left_finger_pad_2", "right_finger_pad_1", "right_finger_pad_2")}
        self.hand_bodies = set()
        for body_id in range(self.model.nbody):
            parent = body_id
            while parent:
                if parent == self.gripper_body:
                    self.hand_bodies.add(body_id)
                    break
                parent = self.model.body_parentid[parent]
        self.lower, self.upper = self.model.actuator_ctrlrange[self.arm_act].T
        self.mesh_extent = mesh.extents
        self.mesh_center = mesh.bounds.mean(axis=0)
        self.mesh_vertices = np.asarray(mesh.vertices)

    def _contacts(self):
        pads, hand, floor, robot = set(), False, [], False
        for contact in self.data.contact:
            if contact.dist > 0 or self.object_geom not in contact.geom:
                continue
            other = contact.geom[1] if contact.geom[0] == self.object_geom else contact.geom[0]
            if other in self.pad_geoms:
                pads.add("left" if "left" in self.model.geom(other).name else "right")
            body = self.model.geom_bodyid[other]
            hand |= body in self.hand_bodies
            if other == self.floor_geom:
                floor.append(contact.pos.copy())
            elif body != 0:
                robot = True
        return pads, hand, floor, robot

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        cfg = self.config
        target = cfg["target"]
        self.target = self.np_random.uniform(target["low"], target["high"]) if target.get("low") is not None else np.array(target["position"], dtype=float)
        spec = self._sample_spec()
        self.object_spec = spec
        signature = repr(spec)
        if self.model is None or signature != self.active_object or self.target[2] != self.model.geom_pos[self.floor_geom, 2]:
            if self.renderer:
                self.renderer.close()
                self.renderer = None
            self._build(spec)
            self.active_object = signature
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.model.site_pos[self.model.site("target").id] = self.target + [0, 0, .002]
        self.data.qpos[self.qadr] = cfg["initial_joint_positions"]
        self.arm_target = self.data.qpos[self.qadr].copy()
        self.data.ctrl[self.arm_act] = self.arm_target
        # Initialize the native four-bar opening around the configured object width.
        # This is a held-object initial condition, not a grasping policy or a weld.
        grip_joints = [self.model.joint(name).qposadr[0] for name in (
            "left_driver_joint", "left_finger_joint", "left_inner_knuckle_joint",
            "right_driver_joint", "right_finger_joint", "right_inner_knuckle_joint")]
        local_rotation = np.zeros(9)
        mujoco.mju_quat2Mat(local_rotation, np.array(cfg["grasp_quaternion"], dtype=float))
        vertices = self.mesh_vertices @ local_rotation.reshape(3, 3).T
        width = np.ptp(vertices[:, 1])
        lo, hi = 0.0, .85
        for _ in range(20):
            angle = (lo + hi) / 2
            self.data.qpos[grip_joints] = angle
            mujoco.mj_forward(self.model, self.data)
            rot = self.data.xmat[self.gripper_body].reshape(3, 3)
            left = self.data.geom_xpos[self.model.geom("left_finger_pad_2").id]
            right = self.data.geom_xpos[self.model.geom("right_finger_pad_2").id]
            opening = abs(np.dot(left - right, rot[:, 1])) - .004
            if opening > width + .0001:
                lo = angle
            else:
                hi = angle
        self.data.qpos[grip_joints] = lo
        mujoco.mj_forward(self.model, self.data)
        grip_rot = self.data.xmat[self.gripper_body].reshape(3, 3)
        position = self.data.xpos[self.gripper_body] + grip_rot @ (np.array(cfg["grasp_position"]) - local_rotation.reshape(3, 3) @ self.mesh_center)
        orientation = np.zeros(4)
        mujoco.mju_mulQuat(orientation, self.data.xquat[self.gripper_body], np.array(cfg["grasp_quaternion"], dtype=float))
        self.data.qpos[self.object_qadr:self.object_qadr + 7] = np.r_[position, orientation]
        self.data.ctrl[self.grip_act] = 255
        # Only reset initialization positions the object. No weld or object force is used.
        # Close under contact with the arm holding its configured start pose.
        for _ in range(round(cfg["settle_time"] / cfg["timestep"])):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        pads, hand, floor, _ = self._contacts()
        self.last_initialization = {"bilateral_contact": len(pads) == 2, "hand_contact": bool(hand), "on_floor": bool(floor)}
        self.start_time = self.data.time
        self.previous_action = np.zeros(8)
        self.released = False
        self.release_candidate = None
        self.release = None
        self.landing = None
        self.ever_contact = bool(hand)
        self.no_contact_steps = 0
        self.initial_object_position = self.data.xpos[self.object_body].copy()
        self.previous_prediction_score = self._prediction_score()
        return self._observation(), {"initialization": self.last_initialization, "object_json": json.dumps(spec), "target": self.target.copy()}

    def _object_state(self):
        # cvel is angular/linear velocity at the body's CoM in the world frame.
        omega = self.data.cvel[self.object_body, :3].copy()
        linear = self.data.cvel[self.object_body, 3:].copy()
        return self.data.xpos[self.object_body].copy(), self.data.xquat[self.object_body].copy(), linear, omega

    def _prediction_score(self):
        position = self.data.xipos[self.object_body]
        velocity = self.data.cvel[self.object_body, 3:]
        g = -self.model.opt.gravity[2]
        height = max(0, position[2] - self.target[2])
        flight = (velocity[2] + np.sqrt(velocity[2] ** 2 + 2 * g * height)) / g
        predicted = position[:2] + velocity[:2] * flight
        distance = np.linalg.norm(predicted - self.target[:2])
        scale = self.config["reward"]["distance_scale"]
        return float(np.exp(-distance ** 2 / (2 * scale ** 2)))

    def _observation(self):
        position, quaternion, velocity, omega = self._object_state()
        grip_q = self.model.jnt_qposadr[self.grip_joint]
        grip_v = self.model.jnt_dofadr[self.grip_joint]
        hand = self._contacts()[1]
        return np.concatenate([
            self.data.qpos[self.qadr], self.data.qvel[self.vadr],
            [self.data.qpos[grip_q], self.data.qvel[grip_v]],
            position, quaternion, velocity, omega,
            self.target - position, [self.mass], self.com, self.inertia,
            self.geometry.ravel(),
            [self.data.time - self.start_time, float(self.released), float(hand)],
            self.previous_action,
        ]).astype(np.float32)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1, 1)
        cfg = self.config
        duration = cfg["timestep"] * cfg["substeps"]
        self.arm_target = np.clip(self.arm_target + action[:7] * (self.upper - self.lower) * cfg["action_rate"] * duration, self.lower, self.upper)
        self.data.ctrl[self.arm_act] = self.arm_target
        # -1 closes, +1 opens; all continuous intermediate openings are available.
        self.data.ctrl[self.grip_act] = (1 - action[7]) * 127.5
        reward, terminated = 0.0, False
        outcome = "running"
        for _ in range(cfg["substeps"]):
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            _, hand, floor, _ = self._contacts()
            self.ever_contact |= bool(hand)
            if hand:
                self.no_contact_steps = 0
                self.release_candidate = None
            elif self.ever_contact and not self.released:
                if self.release_candidate is None:
                    pos, quat, vel, omega = self._object_state()
                    self.release_candidate = {"time": float(self.data.time - self.start_time), "position": pos.tolist(), "com_position": self.data.xipos[self.object_body].tolist(), "velocity": vel.tolist(), "angular_velocity": omega.tolist()}
                self.no_contact_steps += 1
                # Debounce one physics-step contact flicker; timestamp stays at first separation.
                if self.no_contact_steps * cfg["timestep"] >= cfg["release_confirm_seconds"] or floor:
                    self.released = True
                    self.release = self.release_candidate
                    reward += cfg["reward"]["release"]
            if floor:
                contact_position = np.mean(floor, axis=0)
                error = float(np.linalg.norm(contact_position[:2] - self.target[:2]))
                success = bool(self.released and error <= cfg["target"]["success_radius"])
                self.landing = {"position": contact_position.tolist(), "error": error, "success": success}
                reward += cfg["reward"]["landing"] * np.exp(-error ** 2 / (2 * cfg["reward"]["distance_scale"] ** 2)) if self.released else 0.0
                reward += cfg["reward"]["hit"] if success else 0.0
                terminated = True
                outcome = "hit" if success else "miss" if self.released else "dropped_without_release"
                break
        score = self._prediction_score()
        # Potential difference encourages useful release states without a fixed flight time.
        if not self.released:
            reward += cfg["reward"]["prediction_progress"] * (score - self.previous_prediction_score)
        self.previous_prediction_score = score
        reward -= cfg["reward"]["action_cost"] * np.square(action[:7]).sum()
        self.previous_action = action.copy()
        truncated = bool(not terminated and self.data.time - self.start_time >= cfg["episode_seconds"])
        if truncated:
            outcome = "timeout"
        info = {"outcome": outcome, "success": bool(self.landing and self.landing["success"]), "released": self.released, "landing_error": self.landing["error"] if self.landing else np.nan, "target": self.target.copy(), "initialization": self.last_initialization}
        return self._observation(), float(reward), terminated, truncated, info

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = (self.data.xpos[self.object_body] + np.r_[self.target[:2], .5]) / 2
        camera.distance = max(2.0, np.linalg.norm(self.target[:2]) + 1)
        camera.azimuth = 125
        camera.elevation = -20
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
