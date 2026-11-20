"""The live runner never blocks, catches up, drops only poses and surfaces errors."""

from __future__ import annotations

import threading
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite_shadow import ShadowModel, ShadowModelConfig, read_session
from myosuite_shadow.live import LiveShadowModel


def test_a_burst_is_caught_up_and_fully_logged(
    elbow: mujoco.MjModel, tmp_path: Path
) -> None:
    frames = []
    shadow = ShadowModel(elbow, ShadowModelConfig(window=5), log_dir=tmp_path)
    with LiveShadowModel(shadow, on_frame=frames.append) as live:
        for k in range(300):  # far faster than the optimizer: a backlog builds
            live.submit(k / 90, np.array([1.2 + 0.2 * np.sin(k / 30)]), [-15.0])
    s = read_session(tmp_path)
    np.testing.assert_array_equal(s.input_time, np.arange(300) / 90)  # every pose
    states = 300 - 4
    assert len(s.time) + int(s.columns["skipped"].sum()) == states
    assert 1 <= len(frames) == len(s.time) and live.latest is frames[-1]
    assert s.time[-1] == pytest.approx(297 / 90)  # it ends caught up on the newest
    assert s.events[-1]["label"] == "live_summary"
    assert s.events[-1]["data"] == {"processed": 300, "dropped": 0}


def test_overflow_drops_the_oldest_poses_but_no_events(
    elbow: mujoco.MjModel, tmp_path: Path
) -> None:
    shadow = ShadowModel(elbow, ShadowModelConfig(window=5), log_dir=tmp_path)
    release, entered = threading.Event(), threading.Event()
    push = shadow.push

    def slow_push(*args, **kwargs):
        entered.set()
        release.wait(10)
        return push(*args, **kwargs)

    shadow.push = slow_push
    live = LiveShadowModel(shadow, max_queue=10)
    live.submit(0.0, [1.2])
    assert entered.wait(10)  # the worker holds pose 0
    for k in range(1, 31):
        live.submit(k / 90, [1.2])
        if k == 5:
            live.mark("probe", n=1)
    release.set()
    live.close()
    assert live.dropped == 20 and live.processed == 11
    s = read_session(tmp_path)
    np.testing.assert_array_equal(s.input_time, [0.0, *(np.arange(21, 31) / 90)])
    assert [e["label"] for e in s.events] == ["probe", "live_summary"]


def test_worker_errors_reach_the_caller(elbow: mujoco.MjModel) -> None:
    live = LiveShadowModel(ShadowModel(elbow))
    live.submit(0.0, [np.nan])
    live._thread.join(10)
    with pytest.raises(RuntimeError, match="worker failed") as err:
        live.submit(0.1, [1.2])
    assert isinstance(err.value.__cause__, ValueError)
    with pytest.raises(RuntimeError):
        live.close()


def test_a_busy_worker_optimizes_only_the_newest_pose(
    elbow: mujoco.MjModel, tmp_path: Path
) -> None:
    shadow = ShadowModel(elbow, ShadowModelConfig(window=5), log_dir=tmp_path)
    solve = shadow.optimizer.solve

    def slow_solve(*args, **kwargs):  # a 20 ms update, as in full-body walking
        threading.Event().wait(0.02)
        return solve(*args, **kwargs)

    shadow.optimizer.solve = slow_solve
    with LiveShadowModel(shadow) as live:
        for k in range(100):  # 100 poses at once: the worker falls behind
            live.submit(k / 90, [1.2])
    s = read_session(tmp_path)
    assert len(s.input_time) == 100 and live.dropped == 0
    assert len(s.time) < 20 and s.columns["skipped"].sum() > 75
    assert s.time[-1] == pytest.approx(97 / 90)


def test_timed_close_keeps_resources_open_until_worker_finishes(
    elbow: mujoco.MjModel, tmp_path: Path
) -> None:
    class Sink:
        closed = 0

        def estimate(self, frame):
            pass

        def event(self, unix, label, data):
            pass

        def close(self):
            self.closed += 1

    sink = Sink()
    shadow = ShadowModel(elbow, log_dir=tmp_path, sinks=(sink,))
    entered, release = threading.Event(), threading.Event()
    push = shadow.push

    def slow_push(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return push(*args, **kwargs)

    shadow.push = slow_push
    live = LiveShadowModel(shadow)
    live.submit(0.0, [1.2])
    assert entered.wait(10)
    try:
        with pytest.raises(TimeoutError):
            live.close(timeout=0.001)
        assert shadow.log is not None and sink.closed == 0
        assert live.health["status"] == "closing"
    finally:
        release.set()
        live.close()
    live.close()
    shadow.close()
    assert sink.closed == 1
    s = read_session(tmp_path)
    assert s.checkpoint["closed"]
    assert [e["label"] for e in s.events].count("live_summary") == 1


def test_health_retains_last_frame_when_stale_or_worker_fails(
    elbow: mujoco.MjModel, monkeypatch
) -> None:
    ready = threading.Event()
    live = LiveShadowModel(
        ShadowModel(elbow, ShadowModelConfig(window=5)), on_frame=lambda f: ready.set()
    )
    for k in range(5):
        live.submit(k / 20, [1.2])
    assert ready.wait(10)
    latest = live.latest
    monkeypatch.setattr("myosuite_shadow.live.now", lambda: latest.frame_unix + 2.0)
    assert live.health["status"] == "stale" and live.health["age_s"] == 2.0
    live.submit(0.3, [np.nan])
    live._thread.join(10)
    assert live.health["status"] == "failed"
    assert live.health["worker_error"] and live.latest is latest
    with pytest.raises(RuntimeError, match="worker failed"):
        live.drain(timeout=1)
    with pytest.raises(RuntimeError, match="worker failed"):
        live.close()


def test_drain_waits_for_events_resets_and_poses_without_stopping(
    elbow: mujoco.MjModel, tmp_path: Path
) -> None:
    shadow = ShadowModel(elbow, ShadowModelConfig(window=5), log_dir=tmp_path)
    entered, release = threading.Event(), threading.Event()
    mark = shadow.mark

    def slow_mark(*args, **kwargs):
        if args[0] == "busy":
            entered.set()
            assert release.wait(10)
        return mark(*args, **kwargs)

    shadow.mark = slow_mark
    live = LiveShadowModel(shadow, max_queue=2)
    live.mark("busy")
    assert entered.wait(10)
    live.reset_tracking()
    live.submit(0.0, [1.2])
    try:
        with pytest.raises(TimeoutError):
            live.drain(timeout=0.001)
    finally:
        release.set()
    live.drain(timeout=10)
    # More work can be submitted after a drain; every frame is optimized.
    for k in range(1, 15):
        live.submit(k / 20, [1.2])
        live.drain(timeout=10)
    live.close()
    session = read_session(tmp_path)
    assert len(session.input_time) == 15 and len(session.time) == 11
    assert live.dropped == 0 and not session.columns["skipped"].any()
    assert session.events[0]["label"] == "busy"
    assert any(op["kind"] == "reset" for op in session.operations)
