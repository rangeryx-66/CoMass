"""ELU actor/critic and diagonal Gaussian policy, adapted from Catch It's PPO models."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class RunningMeanStd(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size, dtype=torch.float64))
        self.register_buffer("var", torch.ones(size, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float64))

    @torch.no_grad()
    def update(self, values):
        values = values.double()
        count = values.shape[0]
        mean, var = values.mean(0), values.var(0, unbiased=False)
        delta = mean - self.mean
        total = self.count + count
        self.var.copy_((self.var * self.count + var * count + delta.square() * self.count * count / total) / total)
        self.mean.add_(delta * count / total)
        self.count.copy_(total)

    def forward(self, values):
        return ((values - self.mean.float()) / torch.sqrt(self.var.float() + 1e-6)).clamp(-10, 10)


def mlp(input_size, hidden, output_size, gain):
    layers = []
    for width in hidden:
        layer = nn.Linear(input_size, width)
        nn.init.orthogonal_(layer.weight, math.sqrt(2))
        nn.init.zeros_(layer.bias)
        layers.extend([layer, nn.ELU()])
        input_size = width
    layer = nn.Linear(input_size, output_size)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return nn.Sequential(*layers, layer)


class ActorCritic(nn.Module):
    def __init__(self, observation_size, hidden=(256, 128), action_size=8):
        super().__init__()
        self.actor = mlp(observation_size, hidden, action_size, .01)
        self.critic = mlp(observation_size, hidden, 1, 1)
        self.log_std = nn.Parameter(torch.full((action_size,), -.5))
        # Start mostly closed; opening remains learned and continuously controllable.
        with torch.no_grad():
            self.actor[-1].bias[-1] = -1

    def distribution(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.log_std.clamp(-5, 2).exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def act(self, obs, deterministic=False):
        distribution = self.distribution(obs)
        latent = distribution.mean if deterministic else distribution.sample()
        log_prob = self.log_prob(distribution, latent)
        return latent.tanh(), latent, log_prob, self.value(obs)

    @staticmethod
    def log_prob(distribution, latent):
        # Exact tanh Jacobian; PPO stores latent samples instead of clipped actions.
        correction = 2 * (math.log(2) - latent - F.softplus(-2 * latent))
        return (distribution.log_prob(latent) - correction).sum(-1)

    def evaluate(self, obs, latent):
        distribution = self.distribution(obs)
        return self.log_prob(distribution, latent), distribution.entropy().sum(-1), self.value(obs)
