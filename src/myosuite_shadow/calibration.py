"""Study-facing imports of MyoSuite's reusable participant calibration math.

Model calculations live in ``myosuite.biomechanics.calibration``. Session
records should store the returned calibration object's ``record()``.
"""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np
from myosuite.biomechanics.calibration import (
    FatigueFit,
    SegmentLengthMeasurement,
    StrengthCalibration,
    StrengthFit,
    check_segment_lengths,
    endurance_time,
    fit_fatigue_parameters,
    posture_at,
    scale_body_mass,
)
from myosuite.biomechanics.calibration import (
    fatigue_params as _fatigue_params,
)
from myosuite.biomechanics.calibration import (
    fit_strength_from_mvcs as _fit_strength_from_mvcs,
)
from myosuite.biomechanics.calibration import (
    max_joint_torque as _max_joint_torque,
)
from myosuite.biomechanics.calibration import (
    scale_strength as _scale_strength,
)
from myosuite.biomechanics.calibration import (
    strength_from_mvc as _strength_from_mvc,
)

from myosuite_shadow.pipeline import _check_installed_dgf


def max_joint_torque(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    joint: str,
    direction: float = 1.0,
    *,
    max_activation: np.ndarray | None = None,
) -> tuple[float, tuple[str, ...]]:
    """Predict isometric net MVC, rejecting zeroed CompliantTendons models."""
    _check_installed_dgf(model)
    return _max_joint_torque(
        model, qpos, joint, direction, max_activation=max_activation
    )


def strength_from_mvc(
    model: mujoco.MjModel, qpos: np.ndarray, joint: str, measured_torque: float
) -> StrengthCalibration:
    """Fit agonist strength, rejecting zeroed CompliantTendons models."""
    _check_installed_dgf(model)
    return _strength_from_mvc(model, qpos, joint, measured_torque)


def fit_strength_from_mvcs(
    model: mujoco.MjModel,
    postures: Sequence[np.ndarray],
    joint: str,
    measured_torques: Sequence[float],
    *,
    muscles: Sequence[str] | None = None,
) -> StrengthFit:
    """Fit shared group strength across MVC postures."""
    _check_installed_dgf(model)
    return _fit_strength_from_mvcs(
        model, postures, joint, measured_torques, muscles=muscles
    )


def scale_strength(
    model: mujoco.MjModel, scale: float, muscles: Sequence[str] | None = None
) -> mujoco.MjModel:
    """Scale selected muscles' peak forces, including De Groote-Fregly ``f0``."""
    _check_installed_dgf(model)
    out = _scale_strength(model, scale, muscles)
    if out.nuser_actuator >= 2:
        if muscles is None:
            ids = np.flatnonzero(out.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE)
        else:
            ids = np.array([out.actuator(name).id for name in muscles], dtype=int)
        dgf = ids[np.isin(out.actuator_user[ids, 0], (1.0, 2.0))]
        out.actuator_user[dgf, 1] *= scale
    return out


def fatigue_params(
    model: mujoco.MjModel, muscles: Sequence[str], fit: FatigueFit
) -> tuple[tuple[str, float, float, float], ...]:
    """``ShadowModelConfig.fatigue_params`` applying *fit* to *muscles*."""
    _check_installed_dgf(model)
    return _fatigue_params(model, muscles, fit)


__all__ = [
    "FatigueFit",
    "SegmentLengthMeasurement",
    "StrengthCalibration",
    "StrengthFit",
    "check_segment_lengths",
    "endurance_time",
    "fatigue_params",
    "fit_fatigue_parameters",
    "fit_strength_from_mvcs",
    "max_joint_torque",
    "posture_at",
    "scale_body_mass",
    "scale_strength",
    "strength_from_mvc",
]
