#!/usr/bin/env python3
"""
train.py — Main PPO training loop for HalfCheetah-v5.

Implements:
    FR-6   PPO clipped surrogate objective
    FR-7   Critic value loss (MSE)
    FR-8   Multi-epoch minibatched updates (10 epochs, minibatch=64)
    FR-9   Training for 500K–1M timesteps to reach target return
    FR-10  Per-episode return and per-update loss/KL logging
    FR-13  Actor + critic checkpoint saving

Usage:
    python train.py                          # baseline defaults
    python train.py --experiment baseline    # named experiment
    python train.py --experiment no_clip --clip-epsilon 1e6
    python train.py --experiment high_lr --lr 3e-3
    python train.py --experiment single_epoch --epochs 1
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from buffer import RolloutBuffer
from networks import ActorNetwork, CriticNetwork
from normalisation import RunningMeanStd
from reward_shaping import NaturalGaitWrapper
from utils import (
    plot_kl_divergence,
    plot_training_rewards,
    save_reward_log,
    setup_logger,
)


# ---------------------------------------------------------------------------
# Hyperparameter configuration (all defaults from PRD)
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """All PPO hyperparameters, with defaults sourced from the PRD.

    These can be overridden via CLI flags (see ``parse_args``).
    """

    total_timesteps: int = 1_000_000
    batch_size: int = 2048          # steps per rollout batch (PRD Section 4.4)
    minibatch_size: int = 64        # FR-8
    epochs: int = 10                # FR-8: multi-epoch reuse per batch
    gamma: float = 0.99             # discount factor
    gae_lambda: float = 0.95        # GAE lambda (PRD Section 3.3)
    clip_epsilon: float = 0.2       # PPO clip range (PRD Section 3.4)
    lr: float = 3e-4                # learning rate for Adam
    vf_coef: float = 0.5            # value loss coefficient
    max_grad_norm: float = 0.5      # gradient clipping norm
    seed: int = 42
    experiment: str = "baseline"    # experiment name for logging
    env_id: str = "HalfCheetah-v5"  # environment ID to train on
    obs_normalise: bool = True      # observation normalisation (stretch goal)
    lr_anneal: bool = True          # linear LR annealing to zero
    reward_shaping: bool = False    # enable natural-gait reward wrapper
    pitch_weight: float = 5.0       # pitch penalty weight
    height_weight: float = 1.0      # height bonus weight
    smoothness_weight: float = 0.1  # joint velocity smoothness penalty weight


# ---------------------------------------------------------------------------
# Gradient sanity check (PRD Section 10 — verify before long training)
# ---------------------------------------------------------------------------

def gradient_sanity_check(
    actor: ActorNetwork,
    critic: CriticNetwork,
    env: gym.Env,
    logger: object,
) -> None:
    """Run a few steps, compute loss, verify all gradients are non-zero.

    Catches silent bugs like detached tensors or frozen layers — same
    philosophy as CartPole's gradient check (PRD Section 10).
    """
    # Collect a few transitions
    state, _ = env.reset()
    states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

    for _ in range(64):
        state_t = torch.FloatTensor(state)
        with torch.no_grad():
            value = critic(state_t).item()
        action, log_prob = actor.get_action(state_t)

        next_state, reward, terminated, truncated, _ = env.step(
            action.numpy()
        )
        done = terminated or truncated

        states.append(state)
        actions.append(action.numpy())
        log_probs.append(log_prob.item())
        rewards.append(reward)
        values.append(value)
        dones.append(done)

        state = next_state if not done else env.reset()[0]

    # Compute a dummy loss and backprop
    states_t = torch.FloatTensor(np.array(states))
    actions_t = torch.FloatTensor(np.array(actions))
    returns_t = torch.FloatTensor(np.array(rewards))

    new_log_probs, _ = actor.evaluate_actions(states_t, actions_t)
    actor_loss = -(new_log_probs * returns_t).mean()
    actor_loss.backward()

    values_t = critic(states_t).squeeze()
    critic_loss = nn.MSELoss()(values_t, returns_t)
    critic_loss.backward()

    all_ok = True
    for name, param in list(actor.named_parameters()) + list(
        critic.named_parameters()
    ):
        if param.grad is None or torch.all(param.grad == 0):
            logger.warning(f"  ⚠️  {name}: gradient is zero or None!")
            all_ok = False
        else:
            logger.info(f"  ✓  {name}: grad norm = {param.grad.norm():.6f}")

    if all_ok:
        logger.info("  ✅ Gradient sanity check PASSED\n")
    else:
        logger.error("  ❌ Gradient sanity check FAILED\n")

    # Zero out grads after the check
    actor.zero_grad()
    critic.zero_grad()


# ---------------------------------------------------------------------------
# PPO update step (FR-6, FR-7, FR-8)
# ---------------------------------------------------------------------------

def ppo_update(
    actor: ActorNetwork,
    critic: CriticNetwork,
    actor_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    config: PPOConfig,
) -> dict[str, float]:
    """Perform multiple epochs of PPO updates on the collected batch.

    Implements:
        - PPO clipped surrogate loss (FR-6, PRD Section 3.4)
        - Value function loss as MSE (FR-7)
        - Multi-epoch minibatch updates (FR-8, PRD Section 3.5)
        - KL divergence tracking (FR-10)

    Args:
        actor: The actor network.
        critic: The critic network.
        actor_optimizer: Optimizer for the actor.
        critic_optimizer: Optimizer for the critic.
        buffer: Rollout buffer with computed GAE advantages and returns.
        config: Hyperparameter configuration.

    Returns:
        Dictionary of average metrics: policy_loss, value_loss, kl_divergence.
    """
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_kl = 0.0
    num_updates = 0

    for _epoch in range(config.epochs):
        for batch in buffer.get_batches(config.minibatch_size):
            states = batch["states"]
            actions = batch["actions"]
            old_log_probs = batch["old_log_probs"]
            advantages = batch["advantages"]
            returns = batch["returns"]

            # --- Advantage Normalisation (PRD Section 5.1) -----------------
            # Normalise at the minibatch level to keep gradients stable
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # --- PPO clipped surrogate loss (FR-6) -------------------------
            new_log_probs, entropy = actor.evaluate_actions(states, actions)

            # Probability ratio — old log-probs are detached by construction
            # (stored as floats during collection, no gradient graph)
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(
                ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon
            )

            # Take the minimum of clipped and unclipped — the PPO core idea
            surrogate_1 = ratio * advantages
            surrogate_2 = clipped_ratio * advantages
            policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

            # --- Value loss (FR-7) -----------------------------------------
            value_pred = critic(states).squeeze(-1)
            value_loss = nn.MSELoss()(value_pred, returns)

            # --- Actor update ----------------------------------------------
            actor_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
            actor_optimizer.step()

            # --- Critic update ---------------------------------------------
            critic_optimizer.zero_grad()
            (config.vf_coef * value_loss).backward()
            nn.utils.clip_grad_norm_(
                critic.parameters(), config.max_grad_norm
            )
            critic_optimizer.step()

            # --- KL divergence tracking (FR-10) ----------------------------
            with torch.no_grad():
                kl = (old_log_probs - new_log_probs).mean().item()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_kl += kl
            num_updates += 1

    return {
        "policy_loss": total_policy_loss / max(num_updates, 1),
        "value_loss": total_value_loss / max(num_updates, 1),
        "kl_divergence": total_kl / max(num_updates, 1),
    }


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollouts(
    env: gym.Env,
    actor: ActorNetwork,
    critic: CriticNetwork,
    buffer: RolloutBuffer,
    obs_normaliser: RunningMeanStd | None = None,
) -> list[float]:
    """Fill the rollout buffer by running the actor in the environment.

    Collects ``buffer.batch_size`` transitions, potentially spanning
    multiple episodes.  Records state, action, log-prob, reward, value,
    and done flag at each step (PRD Section 4.4, step 1).

    If ``obs_normaliser`` is provided, each raw observation is first
    used to update the running statistics, then normalised before being
    passed to the networks.  The *raw* state is stored in the buffer so
    that normalisation can be reapplied during the update phase with
    the latest statistics.

    Args:
        env: The HalfCheetah-v5 environment.
        actor: Current actor network (for sampling actions).
        critic: Current critic network (for value estimates).
        buffer: The rollout buffer to fill.
        obs_normaliser: Optional running mean/std normaliser.

    Returns:
        List of total episode returns completed during this rollout.
    """
    state, _ = env.reset()
    episode_return = 0.0
    completed_returns: list[float] = []

    for _step in range(buffer.batch_size):
        # Normalise the observation if enabled
        if obs_normaliser is not None:
            obs_normaliser.update(state)
            norm_state = obs_normaliser.normalise(state).astype(np.float32)
        else:
            norm_state = state

        state_tensor = torch.FloatTensor(norm_state)

        with torch.no_grad():
            action, log_prob = actor.get_action(state_tensor)
            value = critic(state_tensor).item()

        action_np = action.numpy()
        next_state, reward, terminated, truncated, _ = env.step(action_np)

        # --- Timeout Bootstrapping (Missing MuJoCo Trick) --------------
        # HalfCheetah episodes end purely due to a 1000-step time limit.
        # If we treat truncated as terminal, the value target becomes 0.
        # Instead, we bootstrap the expected future return from the critic.
        if truncated:
            with torch.no_grad():
                # Normalise true next_state to get correct value estimate
                if obs_normaliser is not None:
                    norm_next = obs_normaliser.normalise(next_state).astype(np.float32)
                else:
                    norm_next = next_state
                bootstrap_val = critic(torch.FloatTensor(norm_next)).item()
                reward += config.gamma * bootstrap_val

        # Since we folded the future value into the reward for truncated episodes,
        # we can safely tell GAE to break the boundary by setting done=True.
        done = terminated or truncated

        # Store the RAW state — we'll re-normalise during the update
        buffer.store(
            state=state,
            action=action_np,
            log_prob=log_prob.item(),
            reward=reward,
            value=value,
            done=done,
        )

        episode_return += reward

        if done:
            completed_returns.append(episode_return)
            episode_return = 0.0
            state, _ = env.reset()
        else:
            state = next_state

    # Bootstrap value for the last state if episode didn't end
    if obs_normaliser is not None:
        norm_last = obs_normaliser.normalise(state).astype(np.float32)
    else:
        norm_last = state
    with torch.no_grad():
        last_value = critic(torch.FloatTensor(norm_last)).item()
    buffer.compute_gae(last_value=last_value if not done else 0.0)

    return completed_returns


# ---------------------------------------------------------------------------
# Main training loop (FR-9)
# ---------------------------------------------------------------------------

def train(config: PPOConfig) -> None:
    """Full PPO training run on HalfCheetah-v5.

    Args:
        config: Hyperparameter configuration.
    """
    logger = setup_logger(
        log_file=f"experiments/{config.experiment}_training.log"
    )

    # --- Reproducibility (NFR: random seed settable and logged) -------------
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    logger.info(f"🔧 Experiment: {config.experiment}")
    logger.info(f"🔧 Config: {json.dumps(asdict(config), indent=2)}")

    # --- Environment --------------------------------------------------------
    env = gym.make(config.env_id)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # --- Reward Shaping (Natural Gait) -------------------------------------
    if config.reward_shaping:
        if "HalfCheetah" in config.env_id:
            env = NaturalGaitWrapper(
                env,
                pitch_weight=config.pitch_weight,
                height_weight=config.height_weight,
                smoothness_weight=config.smoothness_weight,
            )
            logger.info(
                f"🦎 Reward shaping: ENABLED "
                f"(pitch={config.pitch_weight}, height={config.height_weight}, "
                f"smooth={config.smoothness_weight})"
            )
        else:
            logger.warning(
                f"⚠️  Reward shaping requested but is not supported for {config.env_id}. Disabling."
            )

    # --- Observation normalisation (stretch goal, PRD Section 5.2) ----------
    obs_normaliser = RunningMeanStd(shape=(state_dim,)) if config.obs_normalise else None
    if config.obs_normalise:
        logger.info("📊 Observation normalisation: ENABLED")

    # --- Networks + Optimisers ----------------------------------------------
    actor = ActorNetwork(state_dim=state_dim, action_dim=action_dim)
    critic = CriticNetwork(state_dim=state_dim)
    actor_optimizer = optim.Adam(actor.parameters(), lr=config.lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=config.lr)

    # --- Gradient sanity check (PRD Section 10) -----------------------------
    logger.info("🔍 Running gradient sanity check …")
    gradient_sanity_check(actor, critic, env, logger)

    # Re-initialise after sanity check for a clean start
    actor = ActorNetwork(state_dim=state_dim, action_dim=action_dim)
    critic = CriticNetwork(state_dim=state_dim)
    actor_optimizer = optim.Adam(actor.parameters(), lr=config.lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=config.lr)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Compute total number of PPO updates for LR annealing
    total_updates = config.total_timesteps // config.batch_size
    if config.lr_anneal:
        logger.info(
            f"📉 LR annealing: ENABLED (linear decay over {total_updates} updates)"
        )

    # --- Training loop (FR-9) -----------------------------------------------
    buffer = RolloutBuffer(
        batch_size=config.batch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )

    all_episode_returns: list[float] = []
    all_kl_values: list[float] = []
    total_steps = 0
    update_count = 0
    start_time = time.time()

    logger.info(
        f"🚀 Starting PPO training — {config.total_timesteps:,} timesteps"
    )
    logger.info(
        f"   lr={config.lr}  ε={config.clip_epsilon}  γ={config.gamma}  "
        f"λ={config.gae_lambda}  epochs={config.epochs}  "
        f"batch={config.batch_size}  seed={config.seed}\n"
    )

    while total_steps < config.total_timesteps:
        # --- Linear LR annealing -------------------------------------------
        if config.lr_anneal:
            frac = 1.0 - update_count / total_updates
            current_lr = config.lr * max(frac, 0.0)
            for param_group in actor_optimizer.param_groups:
                param_group["lr"] = current_lr
            for param_group in critic_optimizer.param_groups:
                param_group["lr"] = current_lr

        # Step 1: Collect a batch of experience
        buffer.clear()
        episode_returns = collect_rollouts(
            env, actor, critic, buffer, obs_normaliser
        )
        total_steps += buffer.size

        # Step 2: Normalise states in the buffer before PPO update
        if obs_normaliser is not None:
            raw_states = np.array(buffer.states)
            norm_states = obs_normaliser.normalise(raw_states).astype(np.float32)
            buffer.states = list(norm_states)

        # Step 3–4: PPO update
        metrics = ppo_update(
            actor, critic, actor_optimizer, critic_optimizer, buffer, config
        )
        update_count += 1

        # Track metrics
        all_episode_returns.extend(episode_returns)
        all_kl_values.append(metrics["kl_divergence"])

        # Logging (FR-10)
        if episode_returns:
            avg_return = np.mean(episode_returns)
            elapsed = time.time() - start_time
            lr_now = current_lr if config.lr_anneal else config.lr
            logger.info(
                f"  Update {update_count:>4d} | "
                f"Steps {total_steps:>8,d} | "
                f"Ep return {avg_return:>8.1f} | "
                f"π loss {metrics['policy_loss']:>8.4f} | "
                f"V loss {metrics['value_loss']:>8.1f} | "
                f"KL {metrics['kl_divergence']:>8.5f} | "
                f"LR {lr_now:.6f} | "
                f"Time {elapsed:>6.1f}s"
            )

    elapsed = time.time() - start_time
    logger.info(
        f"\n✅ Training complete — {total_steps:,} steps in {elapsed:.1f}s"
    )

    if all_episode_returns:
        final_avg = np.mean(all_episode_returns[-20:])
        logger.info(f"   Final avg return (last 20 episodes): {final_avg:.1f}")

    # --- Save checkpoints (FR-13) -------------------------------------------
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    actor_path = ckpt_dir / f"actor_{config.experiment}.pt"
    critic_path = ckpt_dir / f"critic_{config.experiment}.pt"
    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    logger.info(f"💾 Actor saved  → {actor_path}")
    logger.info(f"💾 Critic saved → {critic_path}")

    # Save observation normaliser statistics (needed at evaluation time)
    if obs_normaliser is not None:
        norm_path = ckpt_dir / f"obs_norm_{config.experiment}.npz"
        stats = obs_normaliser.state_dict()
        np.savez(norm_path, mean=stats["mean"], var=stats["var"], count=stats["count"])
        logger.info(f"📊 Obs normaliser saved → {norm_path}")

    # --- Save experiment config ---------------------------------------------
    config_path = ckpt_dir / f"config_{config.experiment}.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    logger.info(f"📝 Config saved → {config_path}")

    # --- Plots + log (FR-10, FR-12) -----------------------------------------
    save_reward_log(
        all_episode_returns,
        save_path=f"experiments/{config.experiment}_reward_log.csv",
    )
    plot_training_rewards(
        all_episode_returns,
        save_path=f"experiments/{config.experiment}_reward_curve.png",
    )
    plot_kl_divergence(
        all_kl_values,
        save_path=f"experiments/{config.experiment}_kl_divergence.png",
    )

    env.close()
    logger.info(
        "\n🎯 Next step: run  python evaluate.py  to test the trained policy.\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments — all PRD hyperparameters exposed as flags."""
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on HalfCheetah-v5"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=1_000_000,
        help="Total env steps for training (default: 1,000,000)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048,
        help="Steps per rollout batch (default: 2048)",
    )
    parser.add_argument(
        "--minibatch-size", type=int, default=64,
        help="Minibatch size for PPO updates (default: 64)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Epochs per PPO update (default: 10)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--gae-lambda", type=float, default=0.95,
        help="GAE lambda (default: 0.95)",
    )
    parser.add_argument(
        "--clip-epsilon", type=float, default=0.2,
        help="PPO clip range epsilon (default: 0.2)",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate for Adam (default: 3e-4)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--experiment", type=str, default="baseline",
        help="Experiment name for file naming (default: baseline)",
    )
    parser.add_argument(
        "--env-id", type=str, default="HalfCheetah-v5",
        help="Environment ID to train on (default: HalfCheetah-v5)",
    )
    parser.add_argument(
        "--reward-shaping", action="store_true",
        help="Enable natural-gait reward shaping wrapper",
    )
    parser.add_argument(
        "--pitch-weight", type=float, default=5.0,
        help="Pitch penalty weight for reward shaping (default: 5.0)",
    )
    parser.add_argument(
        "--height-weight", type=float, default=1.0,
        help="Height bonus weight for reward shaping (default: 1.0)",
    )
    parser.add_argument(
        "--smoothness-weight", type=float, default=0.1,
        help="Joint velocity smoothness penalty weight (default: 0.1)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = PPOConfig(
        total_timesteps=args.total_timesteps,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        epochs=args.epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        lr=args.lr,
        seed=args.seed,
        experiment=args.experiment,
        env_id=args.env_id,
        reward_shaping=args.reward_shaping,
        pitch_weight=args.pitch_weight,
        height_weight=args.height_weight,
        smoothness_weight=args.smoothness_weight,
    )
    train(config)
