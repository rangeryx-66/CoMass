"""On-policy PPO rollout -> GAE -> clipped updates, following Catch It's training flow."""
import argparse
from collections import deque
from functools import partial
import json
from pathlib import Path
import random
import time

import gymnasium as gym
import numpy as np
import torch
import yaml

from .config import load_config
from .env import ThrowEnv
from .policy import ActorCritic, RunningMeanStd


def compute_gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    advantage = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for step in reversed(range(len(rewards))):
        # True terminal: no bootstrap. Time limit: bootstrap final observation,
        # but never propagate GAE into the next reset episode.
        delta = rewards[step] + gamma * (~terminated[step]).float() * next_values[step] - values[step]
        carry = delta + gamma * lam * (~(terminated[step] | truncated[step])).float() * carry
        advantage[step] = carry
    return advantage, advantage + values


class PPO:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        torch.set_num_threads(1)
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config["seed"])
        self.bc_obs = self.bc_actions = None
        if config["ppo"].get("bc_coef", 0) > 0:
            with np.load(config["ppo"]["bc_dataset"]) as dataset:
                observations, actions = dataset["observations"], dataset["actions"]
                if len(observations) == 0 or observations.shape[1] != 53+3*config["env"]["geometry_points"] or actions.shape != (len(observations), 8):
                    raise ValueError("Demonstration shapes must match policy observations and eight actions")
                self.bc_obs = torch.as_tensor(observations.copy(), dtype=torch.float32, device=self.device)
                self.bc_actions = torch.as_tensor(actions.copy(), dtype=torch.float32, device=self.device)
        creators = [partial(ThrowEnv, config["env"]) for _ in range(config["num_envs"])]
        self.env = gym.vector.AsyncVectorEnv(creators, context="spawn", autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
        self.model = ActorCritic(self.env.single_observation_space.shape[0], config["ppo"]["hidden"]).to(self.device)
        self.normalizer = RunningMeanStd(self.env.single_observation_space.shape[0]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["ppo"]["learning_rate"], eps=1e-5)
        self.steps = 0
        self.update = 0
        self.best_success = -1.0
        self.best_error = float("inf")
        self.start_time = time.monotonic()
        self.successes = deque(maxlen=200)
        self.errors = deque(maxlen=200)
        self.returns = deque(maxlen=200)
        self.release_flags = deque(maxlen=200)
        self.initialized = deque(maxlen=200)
        self.styles = deque(maxlen=200)
        self.style_errors = deque(maxlen=200)
        self.episode_returns = np.zeros(config["num_envs"])
        self.output = Path(config["output"])
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    def save(self, path):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({
            "model": self.model.state_dict(), "normalizer": self.normalizer.state_dict(),
            "optimizer": self.optimizer.state_dict(), "steps": self.steps, "update": self.update,
            "config": self.config, "best_success": self.best_success, "best_error": self.best_error,
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, temporary)
        temporary.replace(path)

    def restore(self, path, reset_best=False):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.normalizer.load_state_dict(state["normalizer"])
        self.optimizer.load_state_dict(state["optimizer"])
        for group in self.optimizer.param_groups:
            group["lr"] = self.config["ppo"]["learning_rate"]
        self.steps, self.update = state["steps"], state["update"]
        self.best_success, self.best_error = state["best_success"], state["best_error"]
        if reset_best:
            self.best_success, self.best_error = -1.0, float("inf")
        torch.set_rng_state(state["torch_rng"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng"]])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])

    def _episode_info(self, infos, done):
        final = infos.get("final_info", {})
        for index in np.flatnonzero(done):
            # Gymnasium SAME_STEP stores final_info as a batched dict.
            success = bool(final["success"][index])
            self.successes.append(success)
            self.returns.append(float(self.episode_returns[index]))
            released = bool(final["released"][index])
            self.release_flags.append(released)
            if np.isfinite(final["landing_error"][index]):
                self.errors.append(float(final["landing_error"][index]))
            init = final.get("initialization", {})
            self.initialized.append(bool(init["bilateral_contact"][index]))
            if "imitation_mean_score" in final:
                self.styles.append(float(final["imitation_mean_score"][index]))
                self.style_errors.append(float(final["imitation_mean_error_degrees"][index]))
            self.episode_returns[index] = 0

    def train(self):
        cfg = self.config["ppo"]
        run_start_steps = self.steps
        self.start_time = time.monotonic()
        obs, _ = self.env.reset(seed=self.config["seed"])
        horizon, actors = cfg["horizon"], self.config["num_envs"]
        try:
            while self.steps < cfg["total_steps"]:
                rollout = {k: [] for k in ("obs", "latent", "log_prob", "value", "next_value", "reward", "terminated", "truncated")}
                for _ in range(horizon):
                    raw = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                    self.normalizer.update(raw)
                    normalized = self.normalizer(raw)
                    with torch.no_grad():
                        action, latent, log_prob, value = self.model.act(normalized)
                    next_obs, reward, terminated, truncated, infos = self.env.step(action.cpu().numpy())
                    bootstrap_obs = np.array(next_obs, copy=True)
                    for index in np.flatnonzero(terminated | truncated):
                        bootstrap_obs[index] = infos["final_obs"][index]
                    with torch.no_grad():
                        next_value = self.model.value(self.normalizer(torch.as_tensor(bootstrap_obs, dtype=torch.float32, device=self.device)))
                    for key, val in [("obs", normalized), ("latent", latent), ("log_prob", log_prob), ("value", value), ("next_value", next_value)]:
                        rollout[key].append(val.detach())
                    for key, val in [("reward", reward), ("terminated", terminated), ("truncated", truncated)]:
                        rollout[key].append(torch.as_tensor(val, device=self.device))
                    self.episode_returns += reward
                    self._episode_info(infos, terminated | truncated)
                    obs = next_obs
                    self.steps += actors
                data = {key: torch.stack(value) for key, value in rollout.items()}
                advantage, returns = compute_gae(data["reward"].float(), data["value"], data["next_value"], data["terminated"], data["truncated"], cfg["gamma"], cfg["gae_lambda"])
                advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
                flat = {key: value.flatten(0, 1) for key, value in data.items()}
                advantage, returns = advantage.flatten(), returns.flatten()
                last_loss = last_kl = 0.0
                for epoch in range(cfg["epochs"]):
                    indices = torch.randperm(horizon * actors, device=self.device)
                    epoch_kls = []
                    for batch in indices.split(cfg["minibatch_size"]):
                        log_prob, entropy, value = self.model.evaluate(flat["obs"][batch], flat["latent"][batch])
                        log_ratio = log_prob - flat["log_prob"][batch]
                        ratio = log_ratio.exp()
                        actor_loss = torch.maximum(-advantage[batch] * ratio, -advantage[batch] * ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"])).mean()
                        value_loss = .5 * (value - returns[batch]).square().mean()
                        loss = actor_loss + cfg["value_coef"] * value_loss - cfg["entropy_coef"] * entropy.mean()
                        if self.bc_obs is not None:
                            demo = torch.randint(len(self.bc_obs), (128,), device=self.device)
                            imitation_action = self.model.actor(self.normalizer(self.bc_obs[demo])).tanh()
                            # Soft training regularization of arm style only. Release
                            # timing remains task-learned; no action blending at inference.
                            bc_loss = (imitation_action[:, :7] - self.bc_actions[demo, :7]).square().mean()
                            loss = loss + cfg["bc_coef"] * bc_loss
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg["grad_norm"])
                        self.optimizer.step()
                        with torch.no_grad():
                            epoch_kls.append(((ratio - 1) - log_ratio).mean().item())
                        last_loss = float(loss.detach())
                    last_kl = float(np.mean(epoch_kls))
                    # Same KL-based learning-rate control used in Catch It.
                    if cfg["target_kl"]:
                        lr = self.optimizer.param_groups[0]["lr"]
                        if last_kl > cfg["target_kl"] * 2:
                            lr = max(lr / 1.5, 1e-6)
                        elif last_kl < cfg["target_kl"] / 2:
                            lr = min(lr * 1.5, cfg["learning_rate"])
                        for group in self.optimizer.param_groups:
                            group["lr"] = lr
                self.update += 1
                success = float(np.mean(self.successes)) if self.successes else 0.0
                error = float(np.mean(self.errors)) if self.errors else None
                record = {"steps": self.steps, "update": self.update, "episodes_in_window": len(self.successes), "success_rate": success, "landing_error_m": error, "release_rate": float(np.mean(self.release_flags)) if self.release_flags else 0.0, "initial_bilateral_rate": float(np.mean(self.initialized)) if self.initialized else 0.0, "return": float(np.mean(self.returns)) if self.returns else 0.0, "loss": last_loss, "kl": last_kl, "fps": (self.steps - run_start_steps) / (time.monotonic() - self.start_time)}
                if self.styles:
                    record.update(imitation_mean_score=float(np.mean(self.styles)),
                                  imitation_mean_error_degrees=float(np.mean(self.style_errors)))
                if self.bc_obs is not None:
                    record["bc_arm_action_mse"] = float(bc_loss.detach())
                print(json.dumps(record), flush=True)
                with (self.output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                self.save(self.output / "last.pt")
                if self.successes and (success > self.best_success or (success == self.best_success and error is not None and error < self.best_error)):
                    self.best_success = success
                    self.best_error = error if error is not None else float("inf")
                    self.save(self.output / "best.pt")
                if self.update % cfg["save_every"] == 0:
                    self.save(self.output / f"step_{self.steps}.pt")
        except KeyboardInterrupt:
            self.save(self.output / "last.pt")
            raise
        finally:
            self.env.close(timeout=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--reset-best", action="store_true", help="Start checkpoint selection afresh after changing the task distribution")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    trainer = PPO(config)
    if args.resume:
        trainer.restore(args.resume, reset_best=args.reset_best)
    trainer.train()


if __name__ == "__main__":
    main()
