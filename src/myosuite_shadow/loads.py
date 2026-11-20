"""Generalized external loads for moving-reference-frame shadow sessions."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np


def _vec3(value: Sequence[float] | np.ndarray | None, name: str) -> np.ndarray:
    if value is None:
        return np.zeros(3, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 3-vector, got shape {arr.shape}")
    return arr


def inertial_external_qfrc(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    linear_accel: Sequence[float] | np.ndarray,
    *,
    angular_accel: Sequence[float] | np.ndarray | None = None,
    angular_vel: Sequence[float] | np.ndarray | None = None,
    qvel: np.ndarray | None = None,
    pivot: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
    data: mujoco.MjData | None = None,
) -> np.ndarray:
    """Project non-inertial D'Alembert wrenches into ``(nv,)`` generalized forces.

    Returns an ``external_qfrc`` vector suitable for :meth:`ShadowModel.push` and
    :meth:`LiveShadowModel.submit` when the participant's support frame accelerates
    or rotates in inertial space.

    Args:
        model: Compiled MuJoCo model.
        qpos: Joint positions ``(nq,)``.
        linear_accel: Support-frame linear acceleration ``(3,)`` [m/s^2].
        angular_accel: Support-frame angular acceleration ``(3,)`` [rad/s^2].
        angular_vel: Support-frame angular velocity ``(3,)`` [rad/s].
        qvel: Optional joint velocities ``(nv,)`` for Coriolis/gyroscopic terms.
        pivot: Rotation pivot in model world coordinates ``(3,)`` [m].
        data: Scratch ``MjData`` to reuse across frames (allocated if ``None``).

    Returns:
        Generalized external load ``(nv,)`` [N m or N].
    """
    q = np.asarray(qpos, dtype=np.float64)
    if q.shape != (model.nq,) or not np.all(np.isfinite(q)):
        raise ValueError(f"qpos must be finite with shape ({model.nq},), got {q.shape}")
    a0 = _vec3(linear_accel, "linear_accel")
    alpha = _vec3(angular_accel, "angular_accel")
    omega = _vec3(angular_vel, "angular_vel")
    p0 = _vec3(pivot, "pivot")

    scratch = mujoco.MjData(model) if data is None else data
    scratch.qpos[:] = q
    if qvel is None:
        scratch.qvel[:] = 0.0
    else:
        qv = np.asarray(qvel, dtype=np.float64)
        if qv.shape != (model.nv,) or not np.all(np.isfinite(qv)):
            raise ValueError(
                f"qvel must be finite with shape ({model.nv},), got {qv.shape}"
            )
        scratch.qvel[:] = qv

    if qvel is not None and np.any(omega != 0.0):
        mujoco.mj_forward(model, scratch)
    else:
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)

    qfrc = np.zeros(model.nv, dtype=np.float64)
    for b in range(1, model.nbody):
        if int(model.body_weldid[b]) == 0:
            continue
        mass = float(model.body_mass[b])
        inertia = np.asarray(model.body_inertia[b], dtype=np.float64)
        if mass <= 0.0 and not np.any(inertia > 0.0):
            continue

        xipos = np.asarray(scratch.xipos[b], dtype=np.float64)
        r_i = xipos - p0
        if qvel is not None and np.any(omega != 0.0):
            cvel = np.asarray(scratch.cvel[b], dtype=np.float64)
            omega_rel = cvel[:3]
            root_com = np.asarray(
                scratch.subtree_com[model.body_rootid[b]], dtype=np.float64
            )
            v_rel = cvel[3:] + np.cross(omega_rel, xipos - root_com)
        else:
            omega_rel = np.zeros(3, dtype=np.float64)
            v_rel = np.zeros(3, dtype=np.float64)

        force = -mass * (
            a0
            + np.cross(alpha, r_i)
            + np.cross(omega, np.cross(omega, r_i))
            + 2.0 * np.cross(omega, v_rel)
        )

        rot = np.asarray(scratch.ximat[b], dtype=np.float64).reshape(3, 3)

        def apply_inertia(vec: np.ndarray) -> np.ndarray:
            return rot @ (inertia * (rot.T @ vec))

        i_omega = apply_inertia(omega)
        i_omega_rel = apply_inertia(omega_rel)
        i_alpha_eff = apply_inertia(alpha + np.cross(omega, omega_rel))
        torque = (
            -i_alpha_eff
            - np.cross(omega, i_omega + i_omega_rel)
            - np.cross(omega_rel, i_omega)
        )
        mujoco.mj_applyFT(model, scratch, force, torque, xipos, b, qfrc)

    return qfrc
