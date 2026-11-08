"""Synthetic motions and tracking noise for trying and benchmarking the pipeline."""

from __future__ import annotations

import mujoco
import numpy as np
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings


def independent_hinges(model: mujoco.MjModel) -> np.ndarray:
    """Ids of the hinge joints that no joint coupling drives or locks."""
    c = JointCouplings.from_model(model)
    coupled = set(c.dep_dof.tolist()) | set(c.locked_dof.tolist())
    hinge = mujoco.mjtJoint.mjJNT_HINGE
    return np.array(
        [
            j
            for j in range(model.njnt)
            if model.jnt_type[j] == hinge and model.jnt_dofadr[j] not in coupled
        ],
        dtype=int,
    )


def reaching_motion(
    model: mujoco.MjModel,
    rate: float,
    duration: float,
    seed: int,
    *,
    start: np.ndarray | None = None,
) -> np.ndarray:
    """Random minimum-jerk reaches of every independent hinge.

    Each reach goes to a uniform posture in the middle 60 % of every joint's
    range in 0.5-0.9 s, then holds it for 0.3 s; coupled joints follow their
    couplings.

    Args:
        model: Fixed-base model.
        rate: Pose rate [Hz].
        duration: Length [s].
        seed: Seed of the reach goals and durations.
        start: ``(nq,)`` starting pose; default the first keyframe, else ``qpos0``.

    Returns:
        ``(round(duration * rate), nq)`` poses.

    Raises:
        ValueError: If the model has a free joint.
    """
    if (model.jnt_type == mujoco.mjtJoint.mjJNT_FREE).any():
        raise ValueError("Synthetic reaches need a fixed-base model")
    joints = independent_hinges(model)
    lo, hi = model.jnt_range[joints].T
    lo, hi = lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo)
    rng = np.random.default_rng(seed)
    if start is None:
        start = model.key_qpos[0] if model.nkey else model.qpos0
    n_total = round(duration * rate)
    cur, poses = np.clip(start[model.jnt_qposadr[joints]], lo, hi), []
    while sum(len(p) for p in poses) < n_total:
        goal, n = rng.uniform(lo, hi), int(rng.uniform(0.5, 0.9) * rate)
        s = np.arange(n) / n
        poses.append(cur + (10 * s**3 - 15 * s**4 + 6 * s**5)[:, None] * (goal - cur))
        poses.append(np.repeat(goal[None], int(0.3 * rate), 0))
        cur = goal
    qpos = np.tile(start, (n_total, 1))
    qpos[:, model.jnt_qposadr[joints]] = np.concatenate(poses)[:n_total]
    project_joint_couplings(JointCouplings.from_model(model), qpos)
    return qpos


def add_tracking_noise(
    model: mujoco.MjModel, qpos: np.ndarray, noise_deg: float, seed: int
) -> np.ndarray:
    """Add white Gaussian noise to every hinge angle, as a tracker would.

    Args:
        model: The model the poses are expressed in.
        qpos: ``(T, nq)`` poses.
        noise_deg: Standard deviation per hinge [deg].
        seed: Seed of the noise.

    Returns:
        The noisy copy.
    """
    hinges = model.jnt_qposadr[model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE]
    noisy = np.array(qpos, dtype=np.float64)
    rng = np.random.default_rng(seed)
    noisy[:, hinges] += np.radians(noise_deg) * rng.standard_normal(
        (len(noisy), len(hinges))
    )
    return noisy
