"""
normalisation.py — Running mean / std normalisation for MuJoCo observations.

MuJoCo's HalfCheetah-v5 observation space has 17 dimensions with wildly
different scales:

    - Joint positions / angles:     often in [-0.2, 0.2]
    - Joint angular velocities:     often in [-10, 10]
    - Root velocity (x-dir):        grows as the cheetah speeds up (0–10+)

Without normalisation, the neural network receives inputs where some
dimensions dominate the gradient signal simply because of their larger
magnitude — the network learns slowly because it's trying to accommodate
both scales simultaneously.

This module implements *online running normalisation* using Welford's
algorithm.  Every observation is whitened to approximately N(0, 1) before
being fed to the actor and critic.  This is the single most impactful
practical trick for PPO on MuJoCo environments — it can easily add
1,000+ points to the final return.

Usage:
    obs_normaliser = RunningMeanStd(shape=(17,))
    normalised_state = obs_normaliser.normalise(raw_state)
"""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Online running mean and variance estimator using Welford's algorithm.

    Tracks the count, mean, and variance of a stream of observations
    and normalises new observations to approximately zero mean and unit
    variance.  Thread-unsafe (single-env training only).

    Args:
        shape: Shape of a single observation (e.g. ``(17,)`` for HalfCheetah).
        epsilon: Small constant to prevent division by zero.
    """

    def __init__(self, shape: tuple[int, ...] = (17,), epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count: float = epsilon  # avoid division by zero on first call
        self.epsilon = epsilon

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a single observation.

        Uses Welford's online algorithm — numerically stable and O(1)
        per observation.

        Args:
            x: A single observation array of matching shape.
        """
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += (delta * delta2 - self.var) / self.count

    def update_batch(self, batch: np.ndarray) -> None:
        """Update running statistics with a batch of observations.

        More efficient than calling ``update`` in a loop — uses the
        parallel/batch variant of Welford's algorithm.

        Args:
            batch: Array of shape (batch_size, *shape).
        """
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalise(self, x: np.ndarray) -> np.ndarray:
        """Normalise an observation to approximately N(0, 1).

        Args:
            x: A single observation (or batch) to normalise.

        Returns:
            Normalised observation with same shape as input.
        """
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)

    def state_dict(self) -> dict:
        """Serialise statistics for checkpoint saving."""
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore statistics from a checkpoint."""
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]
