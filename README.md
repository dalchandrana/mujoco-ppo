# HalfCheetah-PPO

**From-scratch Proximal Policy Optimization (PPO) for continuous control** — training a simulated quadruped to run in MuJoCo.

This is the second project in **Phase 2, Lane B (Optimus)** of my reinforcement learning roadmap. It builds directly on [CartPole-REINFORCE](https://github.com/dalchandrana/cartpole-reinforce), upgrading from a discrete toy problem to continuous-action locomotion with the same algorithm family used in real legged-robotics research and RLHF fine-tuning of LLMs.

**What's new compared to CartPole:**
- **Continuous action space** — 6 real-valued joint torques parameterized by independent Gaussian distributions with a state-independent learnable standard deviation (`log_std`).
- **Actor-critic architecture** — a learned value function (critic) replaces the crude batch-average baseline.
- **PPO clipped objective** — the direct fix for the catastrophic policy collapses documented in my CartPole experiments.
- **Generalized Advantage Estimation (GAE)** — dramatically reduces the variance of the policy gradient by exponentially weighting n-step returns.
- **Orthogonal Weight Initialization** — implements the standard continuous-control PPO trick to prevent early vanishing/exploding gradients.
- **Multi-epoch batch reuse** — the same data trains for 10 epochs per batch, enabled by clipping's stability guarantee.

![HalfCheetah Natural Gait](halfcheetah_ppo_v3.mp4)

---

## Results

### Natural Gait Training (Reward Shaping)
The default HalfCheetah reward exploits the physics engine (running upside down or bouncing on the head). I implemented a custom `NaturalGaitWrapper` with biologically-motivated shaping terms:
1. **Pitch Penalty** (`pitch²`): prevents flipping and head-dragging (heavily weighted).
2. **Height Bonus**: rewards keeping the torso near the natural resting height of `0.5m`.
3. **Smoothness Penalty**: penalises high angular joint velocities to prevent spastic motions.

With these constraints, the agent learns a beautiful, stable, multi-legged running gait that closely resembles an actual animal, achieving a shaped return of ~4,110.

### Training Reward Curve (Natural Gait)
![Reward Curve](natural_gait_v3_reward_curve.png)

### KL Divergence Over Training
![KL Divergence](natural_gait_v3_kl_divergence.png)

### Experiment Comparison

| Experiment | Change Made | Final Avg Return | Notes |
|---|---|---|---|
| v4 Baseline | Legacy v4 Default PPO | **3,711.7** | Exploited the physics engine. |
| v5 Baseline | Default PPO on `HalfCheetah-v5` | **4,968.5** | High scores, but bizarre "scooting" locomotion. |
| **Natural Gait v3** | **Reward Shaping** | **4,110.4** | **Target met. Agent runs beautifully upright.** |
| No clipping (v4) | ε→very large (effectively removes PPO clip) | **-1148.2** | Catastrophic policy collapse; KL div shot >30.0 |
| Higher LR (v4) | lr=3e-3 instead of 3e-4 (10× increase) | **-115.7** | Plateaued but *did not permanently collapse* thanks to clipping |

---

## Why PPO? The Fix for CartPole's Collapse

In my [CartPole experiments](https://github.com/USERNAME/cartpole-reinforce), I observed that vanilla REINFORCE with a high learning rate caused **catastrophic, permanent policy collapse** — the loss hit exactly 0.0000 and never recovered. The policy made a single massive update that pushed it into a deterministic, degenerate state.

PPO's central innovation directly prevents this. It computes a **probability ratio** measuring how much the policy has changed since the data was collected:

```
ratio = exp(log_prob_new(a) - log_prob_old(a))
```

Then it **clips** this ratio to stay within [1 - ε, 1 + ε] (default ε = 0.2), meaning the policy is mechanically prevented from changing by more than 20% in a single update:

```
clipped_ratio = clip(ratio, 1 - ε, 1 + ε)
loss = -mean(min(ratio × advantage, clipped_ratio × advantage))
```

This single change — clip, then take the minimum — is the entire algorithmic innovation of PPO over vanilla policy gradient methods. Everything else (the critic, GAE, multi-epoch updates) is supporting infrastructure around this one idea.

---

## Reproduce These Results

### 1. Clone and install

```bash
git clone https://github.com/USERNAME/halfcheetah-ppo.git
cd halfcheetah-ppo
pip install -r requirements.txt
```

### 2. Train the natural-gait agent

```bash
python train.py --reward-shaping         # ~20-45 min on M4 CPU
```

### 3. Run ablation experiments

```bash
python train.py --experiment no_clip --clip-epsilon 1000000
python train.py --experiment high_lr --lr 3e-3
```

### 4. Evaluate the trained policy

```bash
python evaluate.py --checkpoint checkpoints/actor_natural_gait_v3.pt --reward-shaping --record
```

---

## Project Structure

```
halfcheetah-ppo/
├── networks.py        # ActorNetwork, CriticNetwork (Tanh, 256-unit hidden layers)
├── buffer.py          # RolloutBuffer — storage + GAE computation
├── train.py           # Main PPO training loop with CLI experiment support
├── evaluate.py        # Deterministic evaluation (20 episodes) + video recording
├── reward_shaping.py  # NaturalGaitWrapper for biologically-motivated locomotion
├── utils.py           # Plotting, logging helpers
├── experiments/       # Reward curves, KL plots, training logs
├── checkpoints/       # Saved actor + critic weights
├── media/             # Trained-agent rollout videos
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| Total timesteps | 1,000,000 | PRD Section 6 |
| Batch size | 2,048 steps | PRD Section 4.4 |
| Minibatch size | 64 | PRD FR-8 |
| Epochs per batch | 10 | PRD FR-8 |
| Discount (γ) | 0.99 | PRD Section 9 |
| GAE lambda (λ) | 0.95 | PRD Section 3.3 |
| Clip epsilon (ε) | 0.2 | PRD Section 3.4 |
| Learning rate | 3×10⁻⁴ | PRD Section 9 |
| Pitch Weight | 5.0 | Custom tuning |
| Height Weight | 1.0 | Custom tuning |

---

## What I'd Improve Next

- **Hopper-v5 transfer** — the same PPO implementation on a single-legged robot that can fall over, testing early-termination handling
- **Isaac Gym migration** — moving this same algorithm to GPU-accelerated parallel simulation for more complex robot bodies (Phase 3 of the Optimus roadmap)

---

## License

[MIT](LICENSE)
