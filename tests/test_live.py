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
