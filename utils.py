"""
utils.py — Plotting, logging, and utility helpers for HalfCheetah-PPO.

Implements FR-12 (reward curve + KL divergence plots) and supporting
logging infrastructure.  Follows the same style as the CartPole utils.py.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on headless / SSH sessions
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Plotting (FR-12)
# ---------------------------------------------------------------------------

def plot_training_rewards(
    rewards: List[float],
    save_path: str | Path = "experiments/reward_curve.png",
    window: int = 10,
) -> None:
    """Plot per-episode reward curve with a rolling-average overlay.

    Saved as a PNG for embedding in the README (PRD Section 8).

    Args:
        rewards: List of total episode returns, one per episode.
        save_path: Where to save the PNG.
        window: Window size for the rolling average smoothing line.
    """
    episodes = np.arange(1, len(rewards) + 1)
    rolling_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, alpha=0.3, color="#5e81ac", label="Episode return")
    if len(rewards) >= window:
        ax.plot(
            episodes[window - 1:],
            rolling_avg,
            color="#bf616a",
            linewidth=2,
            label=f"Rolling avg (window={window})",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Return")
    ax.set_title("PPO on HalfCheetah-v5 — Training Reward Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[utils] Reward curve saved → {save_path}")


def plot_kl_divergence(
    kl_values: List[float],
    save_path: str | Path = "experiments/kl_divergence.png",
) -> None:
    """Plot per-update average KL divergence over training.

    A sudden spike in KL typically precedes a reward collapse — this plot
    is the early-warning system described in PRD Section 10.

    Args:
        kl_values: List of average KL divergence, one per PPO update.
        save_path: Where to save the PNG.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(kl_values, color="#a3be8c", linewidth=1.5, label="Avg KL per update")
    ax.set_xlabel("Update")
    ax.set_ylabel("Average KL Divergence")
    ax.set_title("PPO on HalfCheetah-v5 — KL Divergence Over Training")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[utils] KL divergence plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Reward logging
# ---------------------------------------------------------------------------

def save_reward_log(
    rewards: List[float],
    save_path: str | Path = "experiments/reward_log.csv",
) -> None:
    """Save per-episode rewards to a CSV file for later analysis.

    Args:
        rewards: Total reward per episode.
        save_path: Destination CSV path.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("episode,reward\n")
        for i, r in enumerate(rewards, start=1):
            f.write(f"{i},{r:.2f}\n")
    print(f"[utils] Reward log saved → {save_path}")


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

def setup_logger(
    name: str = "halfcheetah_ppo",
    log_file: str | Path = "experiments/training.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a logger that writes to both console and a log file.

    Args:
        name: Logger name.
        log_file: Path to the log file.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
