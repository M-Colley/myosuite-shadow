"""LabStreamingLayer outlets for estimates and experimenter markers (needs ``pylsl``).

LSL synchronises clocks across machines, and LabRecorder writes every stream
of a session (EMG, dynamometer, motion capture, these estimates) into one XDF
file on a common time base. Estimates are stamped with the capture time of the
frame they describe (else its receive time), converted to the LSL clock, so
they line up with EMG of the same instant despite the kinematics lag::

    shadow = ShadowModel(model, log_dir="logs/P07", sinks=(LslSink(model),))

Channels: ``a_<actuator>`` for every actuator, ``mf_<actuator>`` with fatigue,
then ``flags``.
"""

from __future__ import annotations

import json
from typing import Any

import mujoco
import numpy as np

from myosuite_shadow.clock import now
from myosuite_shadow.pipeline import ShadowFrame

try:
    import pylsl
except ImportError as err:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "myosuite_shadow.lsl needs pylsl: pip install 'myosuite-shadow[lsl]'"
    ) from err


class LslSink:
    """Publishes estimates and markers as two LSL streams."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        name: str = "myosuite-shadow",
        source_id: str = "myosuite-shadow",
        fatigue: bool = True,
    ) -> None:
        """Create the outlets.

        Args:
            model: The shadow model's model (for channel names).
            name: Stream name; markers go to ``<name>-markers``.
            source_id: Unique id, so recorders reconnect after a restart.
            fatigue: Include ``MF`` channels.
        """
        actuators = [model.actuator(i).name for i in range(model.nu)]
        self.channels = [f"a_{n}" for n in actuators]
        if fatigue:
            self.channels += [f"mf_{n}" for n in actuators]
        self.channels.append("flags")
        info = pylsl.StreamInfo(
            name,
            "MuscleEstimates",
            len(self.channels),
            pylsl.IRREGULAR_RATE,
            pylsl.cf_double64,
            source_id,
        )
        chans = info.desc().append_child("channels")
        for c in self.channels:
            chans.append_child("channel").append_child_value("label", c)
        markers = pylsl.StreamInfo(
            f"{name}-markers",
            "Markers",
            1,
            pylsl.IRREGULAR_RATE,
            pylsl.cf_string,
            f"{source_id}-markers",
        )
        self.outlet = pylsl.StreamOutlet(info)
        self.markers = pylsl.StreamOutlet(markers)
        self.fatigue = fatigue
        self.sync()

    def sync(self) -> None:
        """Re-measure the offset between the LSL clock and the wall clock."""
        self.offset = pylsl.local_clock() - now()

    def estimate(self, frame: ShadowFrame) -> None:
        """Push one estimate, stamped at its frame's time."""
        sample = list(frame.optimization.activation)
        if self.fatigue:
            mf = frame.fatigue
            sample += list(
                np.zeros_like(frame.optimization.activation) if mf is None else mf
            )
        sample.append(float(int(frame.flags)))
        stamp = frame.frame_unix + self.offset if np.isfinite(frame.frame_unix) else 0.0
        self.outlet.push_sample(sample, stamp)

    def event(self, unix: float, label: str, data: dict[str, Any]) -> None:
        """Push one marker (``label`` or ``label {json}``) at the event's time."""
        text = label if not data else f"{label} {json.dumps(data)}"
        self.markers.push_sample([text], unix + self.offset)

    def close(self) -> None:
        """Release the outlets."""
        del self.outlet, self.markers
