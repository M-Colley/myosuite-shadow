"""Real-time runner: the shadow model on its own thread, never blocking the tracker.

:meth:`LiveShadowModel.submit` stamps and queues a pose and returns at once.
A worker thread feeds every pose to the kinematics but optimizes only when no
newer pose is waiting, so a slow frame (full-body walking with contacts takes
15 ms median, 56 ms p99) delays one estimate instead of the whole stream.
Fatigue still covers all elapsed time. If the queue overflows, the oldest poses
are dropped and counted; a long hole becomes a tracking gap::

    with LiveShadowModel(ShadowModel(model, log_dir="logs/P07")) as live:
        live.mark("set_start", set=1)
        for t, qpos in tracker:
            live.submit(t, qpos)
            frame = live.latest  # newest estimate, for display or feedback
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from myosuite_shadow.clock import now
from myosuite_shadow.flags import Flag
from myosuite_shadow.pipeline import ShadowFrame, ShadowModel


class LiveShadowModel:
    """Runs a :class:`ShadowModel` on a worker thread."""

    def __init__(
        self,
        shadow: ShadowModel,
        *,
        max_queue: int = 1024,
        on_frame: Callable[[ShadowFrame], None] | None = None,
        stale_after: float = 0.5,
    ) -> None:
        """Start the worker.

        Args:
            shadow: The pipeline; only the worker touches it from now on.
            max_queue: Poses held while the worker is busy; older ones drop.
            on_frame: Called on the worker thread with every estimate.
        """
        if max_queue < 1 or stale_after <= 0 or not np.isfinite(stale_after):
            raise ValueError("max_queue and stale_after must be positive")
        self.shadow, self.on_frame = shadow, on_frame
        self.stale_after = float(stale_after)
        self.latest: ShadowFrame | None = None
        self.dropped = 0
        self.processed = 0
        self._queue: deque[tuple[Any, ...]] = deque()
        self._max_queue, self._queued_poses = max_queue, 0
        self._cond = threading.Condition()
        self._stop = False
        self._error: BaseException | None = None
        self._closed = False
        self._busy = False
        self._close_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, name="shadow-model", daemon=True
        )
        self._thread.start()

    def _put(self, item: tuple[Any, ...]) -> None:
        with self._cond:
            if self._error is not None:
                raise RuntimeError("the shadow-model worker failed") from self._error
            if self._stop:
                raise RuntimeError("the live shadow model is closed")
            if item[0] == "pose":
                if self._queued_poses >= self._max_queue:  # drop the oldest pose
                    oldest = next(
                        i for i, q in enumerate(self._queue) if q[0] == "pose"
                    )
                    del self._queue[oldest]
                    self.dropped += 1
                else:
                    self._queued_poses += 1
            self._queue.append(item)
            self._cond.notify()

    def submit(
        self,
        time_s: float,
        qpos: np.ndarray,
        external_qfrc: np.ndarray | None = None,
        *,
        flags: int = Flag.NONE,
        capture_unix: float | None = None,
        evaluation_unix: float | None = None,
        receive_unix: float | None = None,
    ) -> None:
        """Queue a pose (see :meth:`ShadowModel.push`); returns immediately."""
        ext = None if external_qfrc is None else np.array(external_qfrc, dtype=float)
        item = ("pose", float(time_s), np.array(qpos, dtype=float), ext, int(flags))
        self._put(
            item
            + (
                capture_unix,
                now() if receive_unix is None else float(receive_unix),
                evaluation_unix,
            )
        )

    def mark(self, label: str, **data: Any) -> None:
        """Queue an experimenter event, stamped now."""
        self._put(("event", label, now(), data))

    def reset_tracking(self) -> None:
        """Queue a fit reset in order with poses (e.g. after tracker recentering)."""
        self._put(("reset",))

    def drain(self, timeout: float | None = None) -> None:
        """Wait for queued work without stopping the worker.

        Offline raw-frame reprocessing can call this after every submission
        to prevent scheduling-dependent skips or overflow. Live tracking
        normally leaves the worker asynchronous.

        Raises:
            TimeoutError: Work remains; the runner remains usable.
            RuntimeError: The worker failed, with the original error chained.
        """
        if threading.current_thread() is self._thread:
            raise RuntimeError("drain must be called outside the shadow-model worker")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._queue or self._busy:
                if self._error is not None:
                    raise RuntimeError(
                        "the shadow-model worker failed"
                    ) from self._error
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("the shadow-model worker has not drained")
                self._cond.wait(remaining)
            if self._error is not None:
                raise RuntimeError("the shadow-model worker failed") from self._error

    @property
    def health(self) -> dict[str, Any]:
        """Current display health, retaining the last frame after loss or failure.

        ``age_s`` is the age of the estimated frame, including fit delay, on
        the capture/evaluation/receive clock. A stale frame should not drive feedback.
        """
        with self._cond:
            age = (
                None
                if self.latest is None
                else max(0.0, now() - self.latest.frame_unix)
            )
            if self._error is not None:
                status = "failed"
            elif self._closed:
                status = "closed"
            elif self._stop:
                status = "closing"
            elif self.latest is None:
                status = "waiting"
            elif age is not None and age > self.stale_after:
                status = "stale"
            else:
                status = "running"
            return {
                "status": status,
                "age_s": age,
                "worker_error": None if self._error is None else str(self._error),
                "queued_poses": self._queued_poses,
                "processed": self.processed,
                "dropped": self.dropped,
            }

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stop:
                    self._cond.wait()
                if not self._queue:
                    return  # stopped and drained
                item = self._queue.popleft()
                self._busy = True
                if item[0] == "pose":
                    self._queued_poses -= 1
                behind = self._queued_poses > 0
            try:
                if item[0] == "reset":
                    self.shadow.reset_tracking()
                    continue
                if item[0] == "event":
                    _, label, unix, data = item
                    self.shadow.mark(label, unix=unix, **data)
                    continue
                _, t, qpos, ext, flags, capture, receive, evaluation = item
                frame = self.shadow.push(
                    t,
                    qpos,
                    ext,
                    flags=flags,
                    capture_unix=capture,
                    evaluation_unix=evaluation,
                    receive_unix=receive,
                    estimate=not behind,
                )
                self.processed += 1
                if frame is not None:
                    with self._cond:
                        self.latest = frame
                    if self.on_frame is not None:
                        self.on_frame(frame)
            except BaseException as err:  # surfaced on the caller's next call
                with self._cond:
                    self._error, self._stop = err, True
                    self._queue.clear()
                    self._queued_poses = 0
                return
            finally:
                with self._cond:
                    self._busy = False
                    self._cond.notify_all()

    def close(self, timeout: float | None = None) -> None:
        """Process the queued poses, log a summary event and close the pipeline.

        Raises:
            TimeoutError: The worker is still draining; resources remain open.
            RuntimeError: If the worker failed (the original error is chained).
        """
        if threading.current_thread() is self._thread:
            raise RuntimeError("close must be called outside the shadow-model worker")
        with self._close_lock:
            with self._cond:
                self._stop = True
                self._cond.notify()
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError(
                    "the worker is still running; call close again to finish"
                )
            if not self._closed:
                if self._error is None:
                    self.shadow.mark(
                        "live_summary", processed=self.processed, dropped=self.dropped
                    )
                self.shadow.close()
                self._closed = True
            if self._error is not None:
                raise RuntimeError("the shadow-model worker failed") from self._error

    def __enter__(self) -> LiveShadowModel:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
