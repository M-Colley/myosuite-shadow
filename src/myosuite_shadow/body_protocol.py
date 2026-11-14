"""Versioned live-body observations, measured clock exchanges and feedback."""

from __future__ import annotations

import json
import math
import secrets
from dataclasses import dataclass
from typing import Any

from myosuite_shadow.tracking import ClockMapping, TrackerFrame, fit_clock_mapping
from myosuite_shadow.tracking.quest_clock import validate_clock_refresh

SCHEMA = "myosuite-shadow-live/1"


def finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def message(value: str | bytes | dict) -> dict:
    """Decode one bounded, strict-JSON protocol message."""
    if isinstance(value, str | bytes):
        if len(value) > 2**20:
            raise ValueError("live body message exceeds one MiB")
        value = json.loads(value)
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("unsupported live-body schema")
    json.dumps(value, allow_nan=False)
    return value


@dataclass(frozen=True)
class BodySource:
    source_id: str
    person_id: str
    clock_id: str
    coordinate_frame: str = "quest_local"

    def __post_init__(self) -> None:
        for name in ("source_id", "person_id", "clock_id", "coordinate_frame"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"body {name} must be a nonempty string")

    def hello(self, value: dict) -> dict:
        value = message(value)
        if value.get("type") != "hello":
            raise ValueError("a body connection must begin with hello")
        for name in ("source_id", "person_id", "clock_id"):
            if value.get(name) != getattr(self, name):
                raise ValueError(f"unexpected body {name}")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("body hello metadata must be an object")
        return metadata

    def frame(
        self,
        value: dict,
        clock: ClockMapping,
        *,
        receive_unix: float,
        source_metadata: dict | None = None,
    ) -> TrackerFrame:
        value = message(value)
        if value.get("type") != "body_frame":
            raise ValueError("expected a body_frame")
        for name in ("source_id", "person_id", "clock_id", "coordinate_frame"):
            if value.get(name) != getattr(self, name):
                raise ValueError(f"unexpected body {name}")
        if clock.clock_id != self.clock_id:
            raise ValueError("body frame clock identity mismatch")
        positions = value.get("positions")
        if not isinstance(positions, dict) or len(positions) > 512:
            raise ValueError("body positions must be a bounded joint dictionary")
        for name in positions:
            if not isinstance(name, str) or not name or len(name) > 128:
                raise ValueError("invalid body landmark name")
        tracked = value.get("position_tracked")
        confidence = value.get("confidence")
        if not isinstance(tracked, dict) or set(tracked) != set(positions):
            raise ValueError("every body position requires an explicit tracked flag")
        if not isinstance(confidence, dict) or set(confidence) != set(positions):
            raise ValueError("every body position requires explicit confidence")
        source_time = finite(value.get("time_s"), "body time_s")
        session_time, mapped_unix = clock.map(source_time)
        fidelity = value.get("actual_fidelity", "unknown")
        if fidelity not in {"unknown", "low", "high"}:
            raise ValueError("unknown body actual_fidelity")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("body metadata must be an object")
        return TrackerFrame(
            time_s=session_time,
            source_id=self.source_id,
            person_id=self.person_id,
            sequence=value.get("sequence"),
            positions=positions,
            position_tracked=tracked,
            confidence=confidence,
            time_kind="estimate",
            evaluation_unix=mapped_unix,
            receive_unix=receive_unix,
            coordinate_frame=self.coordinate_frame,
            transform_revision=value.get("transform_revision"),
            skeleton_revision=value.get("skeleton_revision"),
            metadata={
                **metadata,
                "source_metadata": source_metadata or {},
                "clock_mapping": clock.record(),
                "actual_fidelity": fidelity,
                "pose_time_semantics": "client_observation_estimate",
                "capture_age_verified": False,
            },
        )


class BodyClock:
    """Fit the existing bounded clock mapping from nonce-matched ping replies."""

    def __init__(
        self,
        clock_id: str,
        unix_origin: float,
        *,
        max_uncertainty_s: float = 0.01,
        validity_s: float = 10.0,
    ) -> None:
        self.clock_id = clock_id
        self.unix_origin = finite(unix_origin, "unix origin")
        self.max_uncertainty = finite(max_uncertainty_s, "clock uncertainty")
        self.validity = finite(validity_s, "clock validity")
        if self.max_uncertainty <= 0 or self.validity <= 0:
            raise ValueError("clock uncertainty and validity must be positive")
        self.pending: dict[str, float] = {}
        self.exchanges: list[tuple[float, float, float]] = []
        self.mapping: ClockMapping | None = None
        self.error: str | None = None
        self._last_fit = -math.inf

    def probe(self, host_s: float) -> dict:
        host_s = finite(host_s, "clock send time")
        self.pending = {
            key: sent for key, sent in self.pending.items() if host_s - sent < 2
        }
        nonce = secrets.token_hex(12)
        self.pending[nonce] = host_s
        return {
            "type": "clock_probe",
            "schema": SCHEMA,
            "nonce": nonce,
            "clock_id": self.clock_id,
        }

    def reply(self, value: dict, host_s: float) -> None:
        value = message(value)
        host_s = finite(host_s, "clock receive time")
        if value.get("type") != "clock_reply" or value.get("clock_id") != self.clock_id:
            raise ValueError("clock reply identity mismatch")
        nonce = value.get("nonce")
        if not isinstance(nonce, str) or nonce not in self.pending:
            raise ValueError("unknown or repeated clock nonce")
        sent = self.pending.pop(nonce)
        source_s = finite(value.get("source_time_s"), "clock source time")
        if host_s < sent or host_s - sent >= 0.05:
            self.error = "clock round trip exceeds 50 ms"
            return
        self.exchanges.append((source_s, sent, host_s))
        self.exchanges = self.exchanges[-32:]
        if self.mapping is not None and host_s - self._last_fit < 2:
            return
        try:
            candidate = fit_clock_mapping(
                self.exchanges,
                clock_id=self.clock_id,
                host_unix_anchor=(0.0, self.unix_origin),
                max_uncertainty_s=self.max_uncertainty,
                validity_s=self.validity,
            )
            if self.mapping is not None:
                validate_clock_refresh(self.mapping, candidate)
        except ValueError as error:
            self.error = str(error)
        else:
            self.mapping, self._last_fit, self.error = candidate, host_s, None
