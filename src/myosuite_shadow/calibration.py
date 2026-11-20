"""Study-facing imports of MyoSuite's reusable participant calibration math.

Model calculations live in ``myosuite.biomechanics.calibration``. Session
records should store the returned calibration object's ``record()``.
"""

from myosuite.biomechanics.calibration import (
    FatigueFit,
    SegmentLengthMeasurement,
    StrengthCalibration,
    check_segment_lengths,
    endurance_time,
    fatigue_params,
    fit_fatigue_parameters,
    fit_strength_from_mvcs,
    max_joint_torque,
    posture_at,
    scale_body_mass,
    scale_strength,
    strength_from_mvc,
)

__all__ = [
    "FatigueFit",
    "SegmentLengthMeasurement",
    "StrengthCalibration",
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
