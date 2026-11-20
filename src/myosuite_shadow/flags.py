"""Quality flags of poses and estimates.

Input flags come from the tracker or :class:`~myosuite_shadow.ik.KeypointIK`; an
estimate carries the union of the flags of the poses in its kinematic fit plus
its own (gap, capacity, solver). Analyses should exclude or down-weight flagged
estimates; the flag column is in every log row.
"""

from __future__ import annotations

from enum import IntFlag


class Flag(IntFlag):
    """Bit flags; combine with ``|``."""

    NONE = 0
    LOW_CONFIDENCE = 1  # a keypoint below the tracker confidence threshold
    MISSING = 2  # a keypoint absent from the frame
    IK_RESIDUAL = 4  # IK could not reach the keypoints within tolerance
    JOINT_LIMIT = 8  # a joint at its range limit
    GLITCH = 16  # an implausible joint speed between frames
    AFTER_GAP = 32  # first estimate after a tracking gap or restart
    CAPACITY_EXCEEDED = 64  # reserve torque above the limit: muscles fall short
    NOT_SOLVED = 128  # the optimizer did not converge
    INFERRED = 256  # a required input joint is inferred rather than tracked


def describe(flags: int) -> list[str]:
    """Names of the set flags, e.g. ``["GLITCH", "AFTER_GAP"]``."""
    return [f.name for f in Flag if f and f in Flag(flags)]
