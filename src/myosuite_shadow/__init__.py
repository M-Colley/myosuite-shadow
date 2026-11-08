"""Live muscle estimates beside an experiment, built on MyoSuite.

The physics (live kinematics, static optimization with ground reactions, 3CC-r
fatigue) lives in MyoSuite; this package runs it on a live pose stream, logs
each session for reproducible analysis and replays motions and recorded
sessions.
"""

from myosuite_shadow.log import Session, package_provenance, read_session
from myosuite_shadow.motions import add_tracking_noise, reaching_motion
from myosuite_shadow.pipeline import ShadowFrame, ShadowModel, ShadowModelConfig

__version__ = "0.1.0"

__all__ = [
    "Session",
    "ShadowFrame",
    "ShadowModel",
    "ShadowModelConfig",
    "__version__",
    "add_tracking_noise",
    "package_provenance",
    "reaching_motion",
    "read_session",
]
