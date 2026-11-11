"""Validated acquisition frames, clock mappings and spatial registration."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

SCHEMA_VERSION = 1


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _vectors(values: dict, size: int, name: str) -> dict[str, np.ndarray]:
    result = {}
    for key, value in values.items():
        vector = np.array(value, dtype=float, copy=True)
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be nonempty joint names")
        if vector.shape != (size,) or not np.isfinite(vector).all():
            raise ValueError(f"{name}[{key}] must be a finite {size}D vector")
        if size == 4 and not np.isclose(np.linalg.norm(vector), 1, atol=1e-3):
            raise ValueError(f"{name}[{key}] must be a unit xyzw quaternion")
        vector.setflags(write=False)
        result[key] = vector
    return result


@dataclass(frozen=True)
class TrackerFrame:
    """One person's metric observations, before or after spatial registration.

    ``time_s`` is on the experiment's common monotonic time axis. Positions
    are metres; orientations are xyzw. Missing joints are omitted. Valid and
    tracked are separate: inferred/last-known values can be valid, untracked.
    ``capture_unix`` is populated only for measured capture timestamps with an
    explicit source-to-Unix clock mapping. Predictions never have capture time.
    """

    time_s: float
    source_id: str
    sequence: int
    person_id: str
    positions: dict[str, np.ndarray]
    confidence: dict[str, float] = field(default_factory=dict)
    orientations: dict[str, np.ndarray] = field(default_factory=dict)
    position_valid: dict[str, bool] = field(default_factory=dict)
    position_tracked: dict[str, bool] = field(default_factory=dict)
    orientation_valid: dict[str, bool] = field(default_factory=dict)
    orientation_tracked: dict[str, bool] = field(default_factory=dict)
    capture_unix: float | None = None
    evaluation_unix: float | None = None
    receive_unix: float | None = None
    prediction_s: float = 0.0
    time_kind: str = "estimate"
    coordinate_frame: str = "tracker_world"
    transform_revision: int = 0
    skeleton_revision: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != SCHEMA_VERSION
        ):
            raise ValueError(f"unsupported tracker schema {self.schema_version}")
        for name in ("source_id", "person_id", "coordinate_frame"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("sequence", "transform_revision", "skeleton_revision"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in (
            "time_s",
            "prediction_s",
            "capture_unix",
            "evaluation_unix",
            "receive_unix",
        ):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.prediction_s < 0 or self.time_kind not in {
            "capture",
            "estimate",
            "predicted",
        }:
            raise ValueError("invalid timestamp semantics")
        if self.prediction_s > 0 and self.time_kind != "predicted":
            raise ValueError("a prediction horizon requires time_kind='predicted'")
        if self.capture_unix is not None and (
            self.prediction_s > 0 or self.time_kind != "capture"
        ):
            raise ValueError("only actual capture time can populate capture_unix")
        positions = _vectors(self.positions, 3, "positions")
        orientations = _vectors(self.orientations, 4, "orientations")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "orientations", orientations)
        confidence = {
            n: _finite(self.confidence.get(n, 1), "confidence") for n in positions
        }
        if set(self.confidence) - positions.keys() or any(
            not 0 <= c <= 1 for c in confidence.values()
        ):
            raise ValueError("confidence must name positions and be within [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        for kind, vectors in (("position", positions), ("orientation", orientations)):
            for status in ("valid", "tracked"):
                name = f"{kind}_{status}"
                given = dict(getattr(self, name))
                if set(given) - vectors.keys() or any(
                    not isinstance(v, bool) for v in given.values()
                ):
                    raise ValueError(f"{name} must contain boolean joint flags")
                flags = {n: given.get(n, status == "valid") for n in vectors}
                object.__setattr__(self, name, flags)
            if any(
                getattr(self, f"{kind}_tracked")[n]
                and not getattr(self, f"{kind}_valid")[n]
                for n in vectors
            ):
                raise ValueError("a tracked joint must also be valid")
        object.__setattr__(
            self, "metadata", json.loads(json.dumps(self.metadata, allow_nan=False))
        )

    @property
    def inferred_joints(self) -> tuple[str, ...]:
        """Return valid positions that are not reported actively tracked."""
        return tuple(
            n
            for n in self.positions
            if self.position_valid[n] and not self.position_tracked[n]
        )

    def record(self) -> dict[str, Any]:
        """Return a JSON-compatible record preserving source and quality fields."""
        result = dataclasses.asdict(self)
        for key in ("positions", "orientations"):
            result[key] = {n: v.tolist() for n, v in result[key].items()}
        return result

    to_dict = record

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> TrackerFrame:
        """Decode a versioned canonical frame; reject unknown schema versions."""
        if "schema_version" not in record:
            raise ValueError("tracker records require schema_version")
        return cls(**record)


@dataclass(frozen=True)
class ClockMapping:
    """Measured affine mapping from a source clock's seconds to session time.

    No automatic receive-time anchoring is performed. For remote clocks,
    estimate offset/rate separately (e.g. repeated ping exchanges or LSL) and
    record uncertainty. XrTime must first use its documented time conversion.
    """

    source_anchor_s: float
    session_anchor_s: float
    clock_id: str
    unix_anchor: float | None = None
    rate: float = 1.0
    uncertainty_s: float = 0.0
    valid_until_source_s: float | None = None
    valid_from_source_s: float | None = None

    def __post_init__(self) -> None:
        for item in dataclasses.fields(self):
            if item.name != "clock_id" and getattr(self, item.name) is not None:
                object.__setattr__(
                    self, item.name, _finite(getattr(self, item.name), item.name)
                )
        if not self.clock_id or self.rate <= 0 or self.uncertainty_s < 0:
            raise ValueError("invalid clock mapping")
        if (
            self.valid_until_source_s is not None
            and self.valid_until_source_s < self.source_anchor_s
        ):
            raise ValueError("clock validity must extend beyond its anchor")
        if (
            self.valid_from_source_s is not None
            and self.valid_until_source_s is not None
            and self.valid_from_source_s > self.valid_until_source_s
        ):
            raise ValueError("clock validity interval is reversed")

    def map(self, source_s: float) -> tuple[float, float | None]:
        """Return session time and optional mapped Unix time."""
        source_s = _finite(source_s, "source time")
        if self.valid_from_source_s is not None and source_s < self.valid_from_source_s:
            raise ValueError("source time precedes clock calibration validity")
        if (
            self.valid_until_source_s is not None
            and source_s > self.valid_until_source_s
        ):
            raise ValueError("clock mapping expired; resynchronise the source")
        elapsed = (source_s - self.source_anchor_s) * self.rate
        unix = None if self.unix_anchor is None else self.unix_anchor + elapsed
        return self.session_anchor_s + elapsed, unix

    def record(self) -> dict[str, Any]:
        """Return calibration anchors, rate and uncertainty for the session log."""
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class Registration:
    """Rigid tracker-to-model registration for one source-space revision."""

    rotation: np.ndarray
    translation_m: np.ndarray
    source_frame: str = "tracker_world"
    revision: int = 0
    target_frame: str = "model_world"

    def __post_init__(self) -> None:
        r = np.array(self.rotation, dtype=float, copy=True)
        t = np.array(self.translation_m, dtype=float, copy=True)
        if (
            r.shape != (3, 3)
            or t.shape != (3,)
            or not np.isfinite(r).all()
            or not np.isfinite(t).all()
            or not np.allclose(r @ r.T, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-7)
        ):
            raise ValueError("registration requires a proper rotation and translation")
        if (
            not self.source_frame
            or not self.target_frame
            or isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 0
        ):
            raise ValueError("registration requires named frames and a revision")
        r.setflags(write=False)
        t.setflags(write=False)
        object.__setattr__(self, "rotation", r)
        object.__setattr__(self, "translation_m", t)

    def apply(self, frame: TrackerFrame) -> TrackerFrame:
        """Transform observations; refuse an unhandled recenter/frame change."""
        if (
            frame.coordinate_frame != self.source_frame
            or frame.transform_revision != self.revision
        ):
            raise ValueError("registration does not match source frame/revision")
        rot = Rotation.from_matrix(self.rotation)
        return dataclasses.replace(
            frame,
            positions={
                n: self.rotation @ p + self.translation_m
                for n, p in frame.positions.items()
            },
            orientations={
                n: (rot * Rotation.from_quat(q)).as_quat()
                for n, q in frame.orientations.items()
            },
            coordinate_frame=self.target_frame,
            metadata={**frame.metadata, "registration": self.record()},
        )

    def record(self) -> dict[str, Any]:
        """Return a JSON-compatible spatial calibration."""
        return {
            "rotation": self.rotation.tolist(),
            "translation_m": self.translation_m.tolist(),
            "source_frame": self.source_frame,
            "revision": self.revision,
            "target_frame": self.target_frame,
        }


def fit_registration(
    source: np.ndarray,
    target: np.ndarray,
    *,
    source_frame: str = "tracker_world",
    revision: int = 0,
    max_error_m: float = 0.02,
) -> Registration:
    """Fit rigid registration from >=3 noncollinear paired metric landmarks."""
    a, b = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if (
        a.ndim != 2
        or a.shape[1] != 3
        or a.shape != b.shape
        or len(a) < 3
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("registration needs paired finite Nx3 landmarks")
    ac, bc = a - a.mean(0), b - b.mean(0)
    if min(np.linalg.matrix_rank(ac), np.linalg.matrix_rank(bc)) < 2:
        raise ValueError("registration landmarks must not be collinear")
    u, _, vt = np.linalg.svd(ac.T @ bc)
    correction = np.diag([1, 1, np.linalg.det(vt.T @ u.T)])
    rotation = vt.T @ correction @ u.T
    translation = b.mean(0) - rotation @ a.mean(0)
    errors = np.linalg.norm(a @ rotation.T + translation - b, axis=1)
    if not np.isfinite(max_error_m) or max_error_m <= 0 or errors.max() > max_error_m:
        raise ValueError("registration residual exceeds max_error_m")
    return Registration(rotation, translation, source_frame, revision)


def fit_clock_mapping(
    exchanges: list[tuple[float, float, float]],
    *,
    clock_id: str,
    host_unix_anchor: tuple[float, float] | None = None,
    max_rtt_s: float = 0.05,
    max_uncertainty_s: float = 0.03,
    max_drift_ppm: float = 1000.0,
    validity_s: float = 60.0,
) -> ClockMapping:
    """Fit source/host clock pairs from timestamped ping replies.

    Each tuple is (source reply time, host send time, host receive time), all
    seconds. Host times must share the session axis. The network delay is
    bounded by RTT; asymmetry is retained as uncertainty, not assumed absent.
    At least three accepted exchanges spanning one second are needed to fit
    rate. Repeat the fit periodically before its validity expires.
    """
    data = np.asarray(exchanges, dtype=float)
    limits = [max_rtt_s, max_uncertainty_s, max_drift_ppm, validity_s]
    if (
        data.ndim != 2
        or data.shape[1] != 3
        or not np.isfinite(data).all()
        or not np.isfinite(limits).all()
        or min(limits) <= 0
    ):
        raise ValueError("clock fit needs finite exchanges and positive limits")
    rtt = data[:, 2] - data[:, 1]
    if np.any(rtt < 0):
        raise ValueError("clock reply precedes its request")
    keep = rtt <= max_rtt_s
    data, rtt = data[keep], rtt[keep]
    if len(data) < 3 or np.ptp(data[:, 0]) < 1.0:
        raise ValueError("need three low-RTT clock exchanges spanning one second")
    source_anchor = float(np.median(data[:, 0]))
    host_origin = float(np.median((data[:, 1] + data[:, 2]) / 2))
    midpoints = (data[:, 1] + data[:, 2]) / 2 - host_origin
    design = np.column_stack((data[:, 0] - source_anchor, np.ones(len(data))))
    weights = 1 / np.maximum(rtt, 1e-6)
    rate, anchor = np.linalg.lstsq(
        design * weights[:, None], midpoints * weights, rcond=None
    )[0]
    if max_drift_ppm >= 1e6:
        raise ValueError("max_drift_ppm must keep the clock rate positive")
    constraints = np.vstack((design, -design))
    bounds = np.concatenate((data[:, 2] - host_origin, -(data[:, 1] - host_origin)))
    drift = max_drift_ppm * 1e-6
    parameter_bounds = [(1 - drift, 1 + drift), (None, None)]

    def feasible(objective: np.ndarray) -> Any:
        return linprog(
            objective,
            A_ub=constraints,
            b_ub=bounds,
            bounds=parameter_bounds,
            method="highs",
        )

    fit = feasible(np.zeros(2))
    if not fit.success:
        raise ValueError("clock exchanges conflict with max_drift_ppm/RTT bounds")
    estimate = np.array([rate, anchor])
    if not 1 - drift <= rate <= 1 + drift or np.any(
        constraints @ estimate > bounds + 1e-9
    ):
        estimate = fit.x
        rate, anchor = estimate
    until = float(data[:, 0].max() + validity_s)
    uncertainty = 0.0
    # Affine clock discrepancy is largest at a validity-interval endpoint.
    # Bound slope/offset ambiguity, not just fitted calibration residuals.
    for endpoint in (float(data[:, 0].min()), until):
        objective = np.array([endpoint - source_anchor, 1.0])
        lower, upper = feasible(objective), feasible(-objective)
        if not lower.success or not upper.success:
            raise ValueError("could not bound clock calibration uncertainty")
        center = float(objective @ estimate)
        uncertainty = max(uncertainty, center - lower.fun, -upper.fun - center)
    uncertainty += 1e-7
    if uncertainty > max_uncertainty_s:
        raise ValueError("clock uncertainty exceeds max_uncertainty_s")
    anchor += host_origin
    unix = None
    if host_unix_anchor is not None:
        session_s, unix_s = host_unix_anchor
        unix = (
            _finite(unix_s, "unix anchor") + anchor - _finite(session_s, "host anchor")
        )
    return ClockMapping(
        source_anchor,
        float(anchor),
        clock_id,
        unix_anchor=unix,
        rate=float(rate),
        uncertainty_s=uncertainty,
        valid_until_source_s=until,
        valid_from_source_s=float(data[:, 0].min()),
    )
