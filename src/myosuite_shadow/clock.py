"""High-resolution wall clock for latency and synchronisation.

On Windows with Python < 3.13, ``time.time()`` advances in 1-16 ms ticks.
:func:`now` anchors ``time.perf_counter()`` to a ``time.time()`` tick edge once
per process, so timestamps resolve microseconds and processes on one machine
agree to well under a millisecond. Clocks of different machines still need
NTP/PTP, or LabStreamingLayer's own clock synchronisation (:mod:`myosuite_shadow.lsl`).
"""

from __future__ import annotations

import time


def _anchor(timeout_s: float = 0.05) -> tuple[float, float]:
    """Pair the next ``time.time()`` tick with ``perf_counter``."""
    start = time.time()
    deadline = time.perf_counter() + timeout_s
    while True:
        wall, perf = time.time(), time.perf_counter()
        if wall != start or perf > deadline:
            return wall, perf


_WALL0, _PERF0 = _anchor()


def now() -> float:
    """Seconds since the Unix epoch, with ``perf_counter`` resolution."""
    return _WALL0 + (time.perf_counter() - _PERF0)
