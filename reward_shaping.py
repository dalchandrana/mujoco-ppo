"""
reward_shaping.py — Gymnasium RewardWrapper for natural cheetah locomotion.

The default HalfCheetah-v5 reward is:

    reward = forward_velocity − 0.1 × control_cost

This produces agents that exploit the physics engine by flipping upside down
or running on their front legs because there is no penalty for unnatural
posture.  This wrapper adds three biologically-motivated shaping terms to
encourage a natural, upright running gait.

Reference:
    - OpenAI, "Faulty Reward Functions in the Wild" (2016)
    - DeepMind locomotion reward shaping (body orientation + height bonus)
    - Georgia Tech symmetry-guided locomotion

Usage:
    env = gym.make("HalfCheetah-v5")
    env = NaturalGaitWrapper(env)  # default weights
    env = NaturalGaitWrapper(env, pitch_weight=2.0)  # heavier pitch penalty
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class NaturalGaitWrapper(gym.Wrapper):
    """Reshape HalfCheetah-v5 reward to encourage natural, upright running.

    Three shaping terms are added to the environment's original reward:

    1. **Pitch Penalty** — quadratic penalty on torso tilt angle, clamped
       to prevent value-function explosion when the agent is upside down.
       ``pitch_penalty = pitch_weight × min(pitch_angle², 4.0)``
       Default weight is 5.0 (aggressive — makes tilting very expensive).

    2. **Height Bonus** — linear bonus for keeping the torso near its
       natural resting height (~0.5 for HalfCheetah).  Clamped so it
       can never go negative.
       ``height_bonus = height_weight × clamp(z, 0, target) / target``

    3. **Smoothness Penalty** — penalises high angular velocities in the
       6 hinge joints.  Discourages spastic, jerky motions.
       ``smoothness_penalty = smoothness_weight × mean(joint_vel²)``

    Final reward:
        shaped = original − pitch_penalty + height_bonus − smoothness_penalty

    Args:
        env: The base HalfCheetah-v5 environment.
        pitch_weight: Weight for the pitch angle penalty (default: 5.0).
        height_weight: Weight for the height bonus (default: 1.0).
        smoothness_weight: Weight for the joint velocity penalty (default: 0.1).
        target_height: The ideal torso z-height (default: 0.5).
    """

    def __init__(
        self,
        env: gym.Env,
        pitch_weight: float = 5.0,
        height_weight: float = 1.0,
        smoothness_weight: float = 0.1,
        target_height: float = 0.5,
    ) -> None:
        assert "HalfCheetah" in env.unwrapped.spec.id, "NaturalGaitWrapper is specifically designed for HalfCheetah!"
        super().__init__(env)
        self.pitch_weight = pitch_weight
        self.height_weight = height_weight
        self.smoothness_weight = smoothness_weight
        self.target_height = target_height

    def step(self, action):
        """Take a step and reshape the reward."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Store the original (unshaped) reward for logging
        info["raw_reward"] = reward

        # --- Access the MuJoCo physics state directly ----------------------
        # qpos layout: [rootx, rootz, rooty_pitch, 6 × joint_angles]
        # qvel layout: [rootx_vel, rootz_vel, rooty_pitch_vel, 6 × joint_vels]
        data = self.env.unwrapped.data
        pitch_angle = data.qpos[2]       # torso pitch (radians)
        z_height = data.qpos[1]          # torso z-position
        joint_velocities = data.qvel[3:]  # 6 hinge joint angular velocities

        # --- Term 1: Pitch Penalty (anti-flip) -----------------------------
        # Clamped at 4.0 to prevent value-function explosion when upside down
        # but still very costly (5.0 × 4.0 = max 20.0 per step)
        pitch_penalty = self.pitch_weight * min(pitch_angle ** 2, 4.0)

        # --- Term 2: Height Bonus (stay upright) ---------------------------
        clamped_z = np.clip(z_height, 0.0, self.target_height)
        height_bonus = self.height_weight * (clamped_z / self.target_height)

        # --- Term 3: Smoothness Penalty (no jerking) -----------------------
        smoothness_penalty = self.smoothness_weight * np.mean(
            joint_velocities ** 2
        )

        # --- Combine -------------------------------------------------------
        shaped_reward = reward - pitch_penalty + height_bonus - smoothness_penalty

        # Store shaping breakdown in info for debugging
        info["pitch_penalty"] = pitch_penalty
        info["height_bonus"] = height_bonus
        info["smoothness_penalty"] = smoothness_penalty
        info["shaped_reward"] = shaped_reward

        return obs, shaped_reward, terminated, truncated, info

