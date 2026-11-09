"""The LSL outlet carries every estimate and marker, stamped on the LSL clock."""

from __future__ import annotations

import json
import time

import mujoco
import numpy as np
import pytest

from myosuite_shadow import ShadowModel, ShadowModelConfig

try:  # pylsl raises RuntimeError, not ImportError, when liblsl cannot load
    import pylsl

    from myosuite_shadow import lsl
except (ImportError, RuntimeError) as err:
    pytest.skip(f"LabStreamingLayer unavailable: {err}", allow_module_level=True)


def _inlet(source_id: str) -> pylsl.StreamInlet:
    streams = pylsl.resolve_byprop("source_id", source_id, timeout=10)
    assert streams, f"stream {source_id} not found"
    inlet = pylsl.StreamInlet(streams[0], max_buflen=60)
    inlet.open_stream(timeout=10)
    return inlet


def test_estimates_and_markers_arrive_on_the_lsl_clock(elbow: mujoco.MjModel) -> None:
    sid = f"shadow-test-{time.time_ns()}"
    sink = lsl.LslSink(elbow, source_id=sid)
    data_in, marks_in = _inlet(sid), _inlet(f"{sid}-markers")
    assert data_in.info().channel_count() == 2 * elbow.nu + 1
    t0 = 5000.0  # capture times on the wall clock
    shadow = ShadowModel(elbow, ShadowModelConfig(window=5), sinks=(sink,))
    frames = [
        shadow.push(k / 20, np.array([1.2]), [-20.0], capture_unix=t0 + k / 20)
        for k in range(12)
    ]
    shadow.mark("set_start", set=2, unix=t0 + 0.3)
    # Without a capture time the frame is stamped on arrival: now, on the LSL clock.
    live = [shadow.push(1 + k / 20, np.array([1.2]), [-20.0]) for k in range(5)][-1]
    pushed_at = pylsl.local_clock()
    frames = [f for f in frames if f is not None]
    samples, stamps = [], []
    deadline = time.time() + 10
    while len(samples) < len(frames) + 1 and time.time() < deadline:
        chunk, ts = data_in.pull_chunk(timeout=1.0)
        samples += chunk
        stamps += ts
    marker, mstamp = marks_in.pull_sample(timeout=10)
    shadow.close()
    assert len(samples) == len(frames) + 1 and len(frames) == 8
    assert live is not None and abs(stamps[-1] - pushed_at) < 0.5
    samples, stamps = samples[:-1], stamps[:-1]
    last = np.array(samples[-1])
    np.testing.assert_allclose(last[: elbow.nu], frames[-1].optimization.activation)
    np.testing.assert_allclose(last[elbow.nu : 2 * elbow.nu], frames[-1].fatigue)
    # Stamps are the frames' capture times shifted by one constant clock offset.
    offset = np.array(stamps) - np.array([f.frame_unix for f in frames])
    assert np.ptp(offset) < 1e-6 and offset[0] == pytest.approx(sink.offset, abs=1e-6)
    label, data = marker[0].split(" ", 1)
    assert label == "set_start" and json.loads(data) == {"set": 2}
    assert mstamp == pytest.approx(t0 + 0.3 + sink.offset, abs=1e-6)
