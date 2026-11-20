"""Bounded Quest clock exchanges and automatic refresh outside acquisition."""

from __future__ import annotations

import json
import math
import secrets
import socket
import threading
import time
from collections.abc import Callable
from numbers import Integral, Real
from typing import Any

from myosuite_shadow.clock import now

from .frames import ClockMapping, fit_clock_mapping


class _Cancelled(Exception):
    pass


def validate_clock_sync_options(
    *,
    refresh_s: float | None = 20,
    port: int = 9764,
    session_epoch: float = 0,
    count: int = 30,
    interval_s: float = 0.2,
    timeout_s: float = 0.5,
    validity_s: float = 60,
    max_rtt_s: float = 0.05,
    max_uncertainty_s: float = 0.01,
) -> None:
    """Validate synchronization settings without network or worker side effects.

    ``refresh_s=None`` omits the periodic-refresh gate for one-shot exchanges.
    This checks scalar bounds, not whether a measured clock fit will succeed.
    """
    for name, value, minimum, maximum in (
        ("count", count, 3, None),
        ("port", port, 1, 65535),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value < minimum
            or (maximum is not None and value > maximum)
        ):
            raise ValueError(f"invalid clock synchronization {name}")
    values = {
        "interval_s": interval_s,
        "timeout_s": timeout_s,
        "validity_s": validity_s,
        "max_rtt_s": max_rtt_s,
        "max_uncertainty_s": max_uncertainty_s,
        "session_epoch": session_epoch,
    }
    if refresh_s is not None:
        values["refresh_s"] = refresh_s
    for name, value in values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or (name != "session_epoch" and value <= 0)
        ):
            raise ValueError(
                f"{name} must be finite and positive (epoch may be any sign)"
            )
    if refresh_s is not None and refresh_s >= validity_s:
        raise ValueError("clock refresh must be positive and shorter than validity")


def validate_clock_refresh(previous: ClockMapping, mapping: ClockMapping) -> None:
    """Require a refreshed mapping to preserve source, session and Unix axes."""
    if not isinstance(mapping, ClockMapping):
        raise TypeError("clock synchronization returned no ClockMapping")
    if mapping.clock_id != previous.clock_id:
        raise ValueError("source clock identity changed; start a new epoch")
    source = mapping.source_anchor_s
    old_session = (
        previous.session_anchor_s + (source - previous.source_anchor_s) * previous.rate
    )
    allowed = previous.uncertainty_s + mapping.uncertainty_s
    if abs(mapping.session_anchor_s - old_session) > allowed + 1e-6:
        raise ValueError("clock refresh changed the session time axis")
    if previous.unix_anchor is not None:
        if mapping.unix_anchor is None:
            raise ValueError("clock refresh lost its Unix reference")
        old_offset = previous.unix_anchor - previous.session_anchor_s
        new_offset = mapping.unix_anchor - mapping.session_anchor_s
        if abs(new_offset - old_offset) > allowed + 0.001:
            raise ValueError("clock refresh changed the Unix reference")


def synchronize_quest_clock(
    host: str,
    *,
    port: int = 9764,
    session_epoch: float = 0,
    count: int = 30,
    interval_s: float = 0.2,
    timeout_s: float = 0.5,
    validity_s: float = 60,
    max_rtt_s: float = 0.05,
    max_uncertainty_s: float = 0.01,
    stop_event: threading.Event | None = None,
) -> ClockMapping:
    """Measure an affine mapping with the native sender's timestamp service.

    Keep ``session_epoch`` constant for the whole study. The source clock ID
    comes from the replies; nonce/clock mismatch and uncertainty fail explicitly.
    """
    validate_clock_sync_options(
        refresh_s=None,
        port=port,
        session_epoch=session_epoch,
        count=count,
        interval_s=interval_s,
        timeout_s=timeout_s,
        validity_s=validity_s,
        max_rtt_s=max_rtt_s,
        max_uncertainty_s=max_uncertainty_s,
    )
    stopped = stop_event or threading.Event()
    samples = []
    clock_id = None
    unix_anchor = (time.perf_counter() - session_epoch, now())
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout_s)
        sock.connect((host, port))
        for i in range(count):
            if stopped.is_set():
                raise _Cancelled()
            nonce = secrets.randbits(52)
            sent = time.perf_counter() - session_epoch
            sock.send(f"shadow-clock-v1 {nonce}".encode("ascii"))
            payload = json.loads(sock.recv(2048))
            received = time.perf_counter() - session_epoch
            if payload.get("nonce") != nonce:
                raise ValueError("unexpected clock reply nonce")
            if clock_id is not None and payload["clock_id"] != clock_id:
                raise ValueError("sender clock changed during synchronization")
            clock_id = payload["clock_id"]
            samples.append((payload["source_time_s"], sent, received))
            if i < count - 1 and stopped.wait(interval_s):
                raise _Cancelled()
    return fit_clock_mapping(
        samples,
        clock_id=clock_id,
        host_unix_anchor=unix_anchor,
        validity_s=validity_s,
        max_rtt_s=max_rtt_s,
        max_uncertainty_s=max_uncertainty_s,
    )


class QuestClockSynchronizer:
    """Refresh a clock mapping on a worker and expose immutable frame snapshots.

    A refresh failure is surfaced on ``get_mapping``. Expiration remains enforced
    by ``ClockMapping.map``. Shutdown interrupts waits and joins the worker before
    releasing it; a blocked custom sync function produces an explicit timeout.
    """

    def __init__(
        self,
        host: str,
        *,
        initial: ClockMapping,
        refresh_s: float = 20,
        port: int = 9764,
        session_epoch: float = 0,
        count: int = 30,
        interval_s: float = 0.2,
        timeout_s: float = 0.5,
        validity_s: float = 60,
        max_rtt_s: float = 0.05,
        max_uncertainty_s: float = 0.01,
        sync_function: Callable[[], ClockMapping] | None = None,
    ) -> None:
        if refresh_s is None:
            raise ValueError("periodic clock refresh requires refresh_s")
        validate_clock_sync_options(
            refresh_s=refresh_s,
            port=port,
            session_epoch=session_epoch,
            count=count,
            interval_s=interval_s,
            timeout_s=timeout_s,
            validity_s=validity_s,
            max_rtt_s=max_rtt_s,
            max_uncertainty_s=max_uncertainty_s,
        )
        self._mapping = initial
        self._stop, self._refresh = threading.Event(), threading.Event()
        self._lock = threading.Lock()
        self._error: Exception | None = None
        self._last_refresh: float | None = None
        self._sync = sync_function or (
            lambda: synchronize_quest_clock(
                host,
                port=port,
                session_epoch=session_epoch,
                count=count,
                interval_s=interval_s,
                timeout_s=timeout_s,
                validity_s=validity_s,
                max_rtt_s=max_rtt_s,
                max_uncertainty_s=max_uncertainty_s,
                stop_event=self._stop,
            )
        )
        self._refresh_s = refresh_s
        self._thread = threading.Thread(
            target=self._run, name="quest-clock", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._refresh.wait(self._refresh_s)
            self._refresh.clear()
            if self._stop.is_set():
                return
            try:
                mapping = self._sync()
                with self._lock:
                    validate_clock_refresh(self._mapping, mapping)
                    self._mapping, self._last_refresh = mapping, now()
            except _Cancelled:
                return
            except Exception as error:
                with self._lock:
                    self._error = error
                return

    def get_mapping(self) -> ClockMapping:
        """Get one immutable packet-boundary snapshot or surface a refresh error."""
        with self._lock:
            if self._error is not None:
                raise RuntimeError("Quest clock refresh failed") from self._error
            if self._stop.is_set():
                raise RuntimeError("Quest clock synchronizer is closed")
            return self._mapping

    @property
    def health(self) -> dict[str, Any]:
        """Return clock identity, last refresh and worker/error state."""
        with self._lock:
            return {
                "clock_id": self._mapping.clock_id,
                "last_refresh_unix": self._last_refresh,
                "error": None if self._error is None else repr(self._error),
                "closed": self._stop.is_set(),
                "alive": self._thread.is_alive(),
            }

    def refresh(self) -> None:
        """Request an immediate background refresh without blocking acquisition."""
        self._refresh.set()

    def close(self, timeout: float = 2) -> None:
        """Cancel pending exchanges and join; signal a custom worker that hangs."""
        self._stop.set()
        self._refresh.set()
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("Quest clock worker did not stop before timeout")

    def __enter__(self) -> QuestClockSynchronizer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
