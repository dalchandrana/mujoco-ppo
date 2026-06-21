#!/usr/bin/env python3
"""
evaluate.py — Deterministic evaluation and video recording for HalfCheetah-v5.

Implements:
    FR-11  Deterministic evaluation (Gaussian mean, no sampling) for 20 episodes
    FR-14  Rollout video recording of the trained policy

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/actor_baseline.pt --episodes 20
    python evaluate.py --record    # also record a rollout video
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from networks import ActorNetwork
from normalisation import RunningMeanStd
from reward_shaping import NaturalGaitWrapper


# ---------------------------------------------------------------------------
# Deterministic evaluation (FR-11)
# ---------------------------------------------------------------------------

def evaluate(
    actor: ActorNetwork,
    env: gym.Env,
    num_episodes: int = 20,
    obs_normaliser: RunningMeanStd | None = None,
) -> list[float]:
    """Run the trained policy deterministically for evaluation.

    Deterministic means using the Gaussian's mean directly — no sampling,
    no exploration.  This is the continuous-action analogue of CartPole's
    argmax evaluation (PRD FR-11).

    Args:
        actor: A trained ActorNetwork in eval mode.
        env: The HalfCheetah-v5 environment.
        num_episodes: How many episodes to run.
        obs_normaliser: Optional observation normaliser (must match training).

    Returns:
        List of total returns, one per episode.
    """
    actor.eval()
    episode_returns: list[float] = []

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_return = 0.0
        done = False

        while not done:
            with torch.no_grad():
                if obs_normaliser is not None:
                    norm_state = obs_normaliser.normalise(state).astype(np.float32)
                else:
                    norm_state = state
                state_tensor = torch.FloatTensor(norm_state)
                # Deterministic: use the mean, not a sample
                action_mean = actor(state_tensor)

            state, reward, terminated, truncated, _ = env.step(
                action_mean.numpy()
            )
            done = terminated or truncated
            total_return += reward

        episode_returns.append(total_return)
        print(f"  Episode {ep:>3d}: return = {total_return:.1f}")

    return episode_returns


# ---------------------------------------------------------------------------
# Video recording (FR-14)
# ---------------------------------------------------------------------------

def record_video(
    actor: ActorNetwork,
    env_id: str = "HalfCheetah-v5",
    video_dir: str | Path = "media",
    num_episodes: int = 1,
    obs_normaliser: RunningMeanStd | None = None,
) -> None:
    """Record a rollout video of the trained policy.

    Uses Gymnasium's RecordVideo wrapper to save a video to ``media/``.
    The video is the single most persuasive artefact for the README
    (PRD Section 8).

    Args:
        actor: A trained ActorNetwork in eval mode.
        env_id: Environment ID to record.
        video_dir: Directory to save the video files.
        num_episodes: Number of episodes to record.
        obs_normaliser: Optional observation normaliser (must match training).
    """
    actor.eval()
    video_path = Path(video_dir)
    video_path.mkdir(parents=True, exist_ok=True)

    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(video_path),
        name_prefix=f"{env_id.lower().replace('-', '_')}_ppo",
        episode_trigger=lambda ep_id: True,  # record all episodes
    )

    for _ep in range(num_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            with torch.no_grad():
                if obs_normaliser is not None:
                    norm_state = obs_normaliser.normalise(state).astype(np.float32)
                else:
                    norm_state = state
                state_tensor = torch.FloatTensor(norm_state)
                action_mean = actor(state_tensor)

            state, _reward, terminated, truncated, _ = env.step(
                action_mean.numpy()
            )
            done = terminated or truncated

    env.close()
    print(f"\n🎥 Rollout video(s) saved → {video_path}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    """Load checkpoint, evaluate, optionally record video."""
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        print("   Run  python train.py  first to train the policy.")
        return

    # Initialize environment first to get dimensions
    env = gym.make(args.env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Load the trained actor
    actor = ActorNetwork(state_dim=state_dim, action_dim=action_dim)
    actor.load_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=True)
    )
    print(f"✅ Loaded actor from {ckpt_path}")

    # Load observation normaliser if it exists
    norm_path = ckpt_path.parent / ckpt_path.name.replace("actor_", "obs_norm_").replace(".pt", ".npz")
    obs_normaliser = None
    if norm_path.exists():
        data = np.load(norm_path, allow_pickle=True)
        obs_normaliser = RunningMeanStd(shape=(state_dim,))
        obs_normaliser.load_state_dict({
            "mean": data["mean"],
            "var": data["var"],
            "count": float(data["count"]),
        })
        print(f"📊 Loaded obs normaliser from {norm_path}")
    else:
        print("ℹ️  No obs normaliser found — using raw observations")

    # --- Evaluate (FR-11) ---------------------------------------------------
    if args.reward_shaping:
        if "HalfCheetah" in args.env_id:
            env = NaturalGaitWrapper(env)
        else:
            print(f"⚠️  Reward shaping requested but is not supported for {args.env_id}. Disabling.")
    print(
        f"\n🔍 Evaluating over {args.episodes} episodes (deterministic) …\n"
    )
    returns = evaluate(actor, env, args.episodes, obs_normaliser)

    avg_return = np.mean(returns)
    std_return = np.std(returns)
    min_return = np.min(returns)
    max_return = np.max(returns)

    print(f"\n  Average return : {avg_return:.1f} ± {std_return:.1f}")
    print(f"  Min / Max      : {min_return:.1f} / {max_return:.1f}")
    print()

    target = 2500
    solved = avg_return >= target
    if solved:
        print(f"  🏆 TARGET MET!  Average {avg_return:.1f} ≥ {target}")
    else:
        print(f"  ⚠️  Below target — average {avg_return:.1f} < {target}")
        print("   Try training for more timesteps or tuning hyperparameters.")

    # --- Save evaluation log ------------------------------------------------
    log_path = Path("experiments/evaluation_log.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Episodes: {args.episodes}\n")
        f.write(f"Average return: {avg_return:.2f}\n")
        f.write(f"Std: {std_return:.2f}\n")
        f.write(f"Min: {min_return:.1f}\n")
        f.write(f"Max: {max_return:.1f}\n")
        f.write(f"Target met (≥{target}): {solved}\n\n")
        f.write("Per-episode returns:\n")
        for i, r in enumerate(returns, 1):
            f.write(f"  Episode {i:>3d}: {r:.1f}\n")
    print(f"\n  📝 Evaluation log saved → {log_path}")

    env.close()

    # --- Record video (FR-14) -----------------------------------------------
    if args.record:
        print("\n🎬 Recording rollout video …")
        record_video(actor, env_id=args.env_id, video_dir=args.video_dir, obs_normaliser=obs_normaliser)

    print(
        "\n🎯 Done!  Embed the reward curve, KL plot, and video in README.md\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO policy on HalfCheetah-v5"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/actor_baseline.pt",
        help="Path to saved actor weights",
    )
    parser.add_argument(
        "--env-id", type=str, default="HalfCheetah-v5",
        help="Environment ID to evaluate (default: HalfCheetah-v5)",
    )
    parser.add_argument(
        "--episodes", type=int, default=20,
        help="Number of evaluation episodes (default: 20)",
    )
    parser.add_argument(
        "--record", action="store_true",
        help="Record a rollout video to media/",
    )
    parser.add_argument(
        "--video-dir", type=str, default="media",
        help="Directory for video output (default: media/)",
    )
    parser.add_argument(
        "--reward-shaping", action="store_true",
        help="Enable natural-gait reward shaping wrapper (match training)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
