"""
buffer.py — Rollout buffer with GAE advantage computation for PPO.

Implements:
    FR-3  Rollout buffer (states, actions, log_probs, rewards, values, dones)
    FR-4  GAE advantage computation with correct episode boundary handling
    FR-5  Advantage normalisation across the batch

The GAE computation follows the same backward-sweep pattern as CartPole's
reward-to-go, but blends one-step TD errors with longer-horizon returns
via the lambda parameter (PRD Section 3.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    """Fixed-size buffer that collects transitions across multiple episodes.

    Once full (``batch_size`` steps collected), GAE advantages and returns
    are computed before yielding minibatches for the PPO update.

    Attributes:
        batch_size: Number of environment steps to collect per batch.
        gamma: Discount factor for future rewards.
        gae_lambda: GAE smoothing parameter (0 = one-step TD, 1 = Monte Carlo).
    """

    batch_size: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Storage — populated during rollout collection
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    # Computed after the batch is full
    advantages: np.ndarray = field(default_factory=lambda: np.array([]))
    returns: np.ndarray = field(default_factory=lambda: np.array([]))

    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        """Record a single environment transition.

        Args:
            state: Observation at this timestep.
            action: Action taken by the actor.
            log_prob: Log-probability of the action under the collection policy.
            reward: Reward received after taking the action.
            value: Critic's value estimate for this state.
            done: Whether the episode ended after this step.
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    @property
    def size(self) -> int:
        """Number of transitions currently stored."""
        return len(self.states)

    @property
    def is_full(self) -> bool:
        """Whether the buffer has collected ``batch_size`` transitions."""
        return self.size >= self.batch_size

    def compute_gae(self, last_value: float = 0.0) -> None:
        """Compute GAE advantages and reward-to-go returns (FR-4).

        Works backward through the buffer — structurally identical to
        CartPole's reward-to-go computation but using TD errors blended
        by ``gae_lambda`` (PRD Section 3.3):

            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            A_t     = delta_t + (gamma * lambda) * A_{t+1}

        Episode boundaries (``done == True``) correctly zero out the
        bootstrapped value and the running advantage, preventing
        information from leaking across episodes within a single batch.

        Args:
            last_value: Critic's value estimate for the state *after* the
                        last transition in the buffer.  Zero if the last
                        step was terminal.
        """
        n = self.size
        self.advantages = np.zeros(n, dtype=np.float32)
        self.returns = np.zeros(n, dtype=np.float32)

        running_advantage = 0.0
        next_value = last_value

        for t in reversed(range(n)):
            # If the episode ended at step t, the next state has no value
            if self.dones[t]:
                next_value = 0.0
                running_advantage = 0.0

            delta = (
                self.rewards[t]
                + self.gamma * next_value
                - self.values[t]
            )
            running_advantage = (
                delta + self.gamma * self.gae_lambda * running_advantage
            )

            self.advantages[t] = running_advantage
            self.returns[t] = running_advantage + self.values[t]

            next_value = self.values[t]

    def get_batches(
        self, minibatch_size: int = 64
    ) -> Generator[dict[str, torch.Tensor], None, None]:
        """Yield shuffled minibatches with normalised advantages (FR-5, FR-8).

        Advantage normalisation (subtract mean, divide by std) is applied
        *once* across the whole batch before splitting into minibatches —
        this is the direct upgrade of CartPole's baseline subtraction
        (PRD Section 4.4, step 3).

        Args:
            minibatch_size: Number of transitions per minibatch.

        Yields:
            Dictionary with keys: states, actions, old_log_probs,
            advantages, returns — all as PyTorch tensors.
        """
        n = self.size

        # FR-5: normalise advantages across the whole batch
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std()
        if adv_std < 1e-8:
            normalised_adv = self.advantages - adv_mean
        else:
            normalised_adv = (self.advantages - adv_mean) / (adv_std + 1e-8)

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(np.array(self.log_probs))
        advantages = torch.FloatTensor(normalised_adv)
        returns = torch.FloatTensor(self.returns)

        # Shuffle indices and yield minibatches
        indices = np.arange(n)
        np.random.shuffle(indices)

        for start in range(0, n, minibatch_size):
            end = start + minibatch_size
            batch_idx = indices[start:end]

            yield {
                "states": states[batch_idx],
                "actions": actions[batch_idx],
                "old_log_probs": old_log_probs[batch_idx],
                "advantages": advantages[batch_idx],
                "returns": returns[batch_idx],
            }

    def clear(self) -> None:
        """Reset all storage for the next batch collection."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.advantages = np.array([])
        self.returns = np.array([])
