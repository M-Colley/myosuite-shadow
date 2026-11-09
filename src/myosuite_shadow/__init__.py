"""Live muscle estimates beside an experiment, built on MyoSuite.

The physics (live kinematics, static optimization with ground reactions, 3CC-r
fatigue) lives in MyoSuite. This package runs it on a live pose stream for user
studies: tracker IK with quality flags (:mod:`~myosuite_shadow.ik`), a real-time
runner (:mod:`~myosuite_shadow.live`), per-participant calibration
(:mod:`~myosuite_shadow.calibration`), crash-safe session logs with events and
LabStreamingLayer output (:mod:`~myosuite_shadow.log`, :mod:`~myosuite_shadow.lsl`),
and quality reports, sensitivity replays and validation statistics afterwards.
"""

from myosuite_shadow.flags import Flag
from myosuite_shadow.live import LiveShadowModel
from myosuite_shadow.log import Session, package_provenance, read_session
from myosuite_shadow.motions import add_tracking_noise, reaching_motion
from myosuite_shadow.pipeline import ShadowFrame, ShadowModel, ShadowModelConfig
from myosuite_shadow.report import format_report, session_report

__version__ = "0.2.0"

__all__ = [
    "Flag",
    "LiveShadowModel",
    "Session",
    "ShadowFrame",
    "ShadowModel",
    "ShadowModelConfig",
    "__version__",
    "add_tracking_noise",
    "format_report",
    "package_provenance",
    "reaching_motion",
    "read_session",
    "session_report",
]
