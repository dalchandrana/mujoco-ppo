"""
networks.py — Actor and Critic neural networks for PPO on HalfCheetah-v5.

Implements FR-1 (Actor) and FR-2 (Critic) from the PRD.

Architecture (PRD Sections 4.2 and 4.3):
    Actor:  17 → 256 (Tanh) → 256 (Tanh) → 6 (mean)  +  learnable log_std
    Critic: 17 → 256 (Tanh) → 256 (Tanh) → 1 (scalar value)

Both networks use Tanh activations (not ReLU) — a deliberate choice for
continuous-control PPO, producing smoother outputs suitable for torque control.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Apply orthogonal initialisation to a linear layer.
    
    Args:
        layer: The linear layer to initialise.
        std: The gain for orthogonal init (default sqrt(2) for Tanh/ReLU).
        bias_const: Constant to initialise biases with.
        
    Returns:
        The initialised layer.
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorNetwork(nn.Module):
    """Policy network: maps state → Gaussian distribution over continuous actions.

    The actor outputs the *mean* of a 6-dimensional Gaussian.  The standard
    deviation is parameterised as a learnable log_std vector that is independent
    of the state — a common, simple choice for PPO (PRD Section 4.2).
    """

    def __init__(
        self,
        state_dim: int = 17,
        hidden_dim: int = 256,
        action_dim: int = 6,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )
        # State-independent log standard deviation (PRD Section 4.2)
        # Initialise to -0.5 (std ~ 0.6) instead of 0.0 (std 1.0) to prevent
        # excessively wild flailing early in training.
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return the mean of the action distribution for a given state.

        Args:
            state: Tensor of shape (state_dim,) or (batch, state_dim).

        Returns:
            Mean action tensor of shape (action_dim,) or (batch, action_dim).
        """
        return self.network(state)

    def get_distribution(self, state: torch.Tensor) -> Normal:
        """Build the Gaussian action distribution for a given state.

        Args:
            state: Tensor of shape (state_dim,) or (batch, state_dim).

        Returns:
            A Normal distribution over the action space.
        """
        mean = self.forward(state)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def get_action(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and return it with its log-probability.

        Used during rollout collection.  Analogous to CartPole's
        ``Categorical(probs).sample()`` but for continuous actions.

        Args:
            state: 1-D tensor of shape (state_dim,).

        Returns:
            action: Sampled action tensor of shape (action_dim,).
            log_prob: Scalar log-probability of the sampled action (summed
                      across the 6 independent dimensions).
        """
        dist = self.get_distribution(state)
        action = dist.sample()
        # Sum log-probs across action dims (independent Gaussians)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-evaluate log-probs and entropy for actions under the *current* policy.

        Used during the PPO update phase to compute the probability ratio.

        Args:
            states: Batch of states, shape (batch, state_dim).
            actions: Batch of actions, shape (batch, action_dim).

        Returns:
            log_probs: Log-probability of each action, shape (batch,).
            entropy: Entropy of the distribution, shape (batch,).
        """
        dist = self.get_distribution(states)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy


class CriticNetwork(nn.Module):
    """Value function: maps state → scalar estimate of expected future return.

    Separate network from the actor (PRD Section 4.3) — simpler to implement
    correctly and debug, at a negligible cost in parameter efficiency.
    """

    def __init__(
        self,
        state_dim: int = 17,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return the scalar value estimate for a given state.

        Args:
            state: Tensor of shape (state_dim,) or (batch, state_dim).

        Returns:
            Value estimate, shape (1,) or (batch, 1).
        """
        return self.network(state)
