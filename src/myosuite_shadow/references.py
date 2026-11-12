"""Explicit instrument adapters and alignment for independent study measurements."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from myosuite_shadow.flags import Flag
from myosuite_shadow.log import Session
from myosuite_shadow.tracking import ClockMapping
from myosuite_shadow.validation import (
    concordance_correlation,
    emg_envelope,
    emg_median_frequency,
    load_xdf,
    sample_at,
)


@dataclass(frozen=True)
class ReferenceStream:
    """Calibrated channels on the session clock, retaining their declared role."""

    time_s: np.ndarray
    values: np.ndarray
    channels: tuple[str, ...]
    units: tuple[str, ...]
    kind: str
    role: str
    provenance: dict[str, Any]
    break_before: np.ndarray | None = None


def load_reference(
    path: str | Path, spec: dict[str, Any], *, expected_sha256: str
) -> ReferenceStream:
    """Read CSV/XDF with explicit clock, units, instrument calibration and role.

    CSV channels name columns; XDF channels use integer indices and a unique
    stream name. Scale/offset are supplied instrument corrections, never fitted
    using validation outcomes. Blank CSV values remain missing observations.
    """
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    digest = digest.hexdigest()
    if digest != expected_sha256.lower():
        raise ValueError("reference file hash differs from the recording manifest")
    kind, role = spec["kind"], spec["role"]
    units = {"force": {"N"}, "torque": {"Nm"}, "emg": {"V", "uV"}}
    if kind not in units or role not in {"calibration", "validation"}:
        raise ValueError(
            "reference needs force/torque/emg and calibration/validation role"
        )
    if not isinstance(spec.get("instrument"), str) or not spec["instrument"]:
        raise ValueError("reference instrument/calibration identifier is required")
    channels = spec["channels"]
    if (
        not channels
        or any(not isinstance(c["name"], str) or not c["name"] for c in channels)
        or len({c["name"] for c in channels}) != len(channels)
    ):
        raise ValueError("reference channel names must be nonempty and unique")
    for channel in channels:
        if channel["unit"] not in units[kind]:
            raise ValueError(f"unsupported {kind} channel unit")
    if spec.get("format", "csv") == "csv":
        time_column = spec.get("time_column", "time")
        with path.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            required = {time_column, *(c["column"] for c in channels)}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("measurement CSV lacks declared time/channel columns")
            rows = list(reader)
        original_count = len(rows)
        role_column = spec.get("role_column", "role")
        if (
            "role_column" in spec
            and role_column is not None
            and role_column not in (reader.fieldnames or [])
        ):
            raise ValueError("measurement CSV lacks declared role column")
        selected = list(range(len(rows)))
        if role_column in (reader.fieldnames or []):
            if any(
                row[role_column] not in {"calibration", "validation"} for row in rows
            ):
                raise ValueError("measurement row roles must be calibration/validation")
            selected = [i for i, row in enumerate(rows) if row[role_column] == role]
            rows = [rows[i] for i in selected]
        breaks = np.r_[False, np.diff(selected) > 1] if rows else np.array([], bool)
        times = np.array([float(row[time_column]) for row in rows])
        values = np.array(
            [
                [
                    float(row[c["column"]]) if row[c["column"]] else np.nan
                    for c in channels
                ]
                for row in rows
            ],
            dtype=float,
        )
    elif spec["format"] == "xdf":
        streams = load_xdf(path)
        times, samples = streams[spec["stream"]]
        indices = [c["column"] for c in channels]
        if samples.ndim != 2 or any(
            isinstance(i, bool)
            or not isinstance(i, int)
            or not 0 <= i < samples.shape[1]
            for i in indices
        ):
            raise ValueError("XDF channels require valid numeric channel indices")
        values = np.asarray(samples[:, indices], dtype=float)
        breaks, original_count = np.zeros(len(times), bool), len(times)
    else:
        raise ValueError("reference format must be csv or xdf")
    times = np.asarray(times, dtype=float)
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("measurement times must be nonempty, finite and increasing")
    if values.shape != (len(times), len(channels)) or np.isinf(values).any():
        raise ValueError("measurement values must match channels; blanks may mark gaps")
    clock = ClockMapping(**spec["clock"])
    clock.map(times[0])
    clock.map(times[-1])
    mapped = clock.session_anchor_s + (times - clock.source_anchor_s) * clock.rate
    scale = np.array([c.get("scale", 1) for c in channels], float)
    offset = np.array([c.get("offset", 0) for c in channels], float)
    if not np.isfinite([scale, offset]).all() or np.any(scale == 0):
        raise ValueError("instrument scale/offset must be finite with nonzero scale")
    return ReferenceStream(
        mapped,
        values * scale + offset,
        tuple(c["name"] for c in channels),
        tuple(c["unit"] for c in channels),
        kind,
        role,
        {
            "path": str(path.resolve()),
            "sha256": digest,
            "instrument": spec["instrument"],
            "clock": clock.record(),
            "channels": channels,
            "excluded_role_rows": original_count - len(times),
        },
        breaks,
    )


def process_emg(
    stream: ReferenceStream,
    *,
    sampling_hz: float,
    max_gap_s: float,
) -> tuple[ReferenceStream, tuple[np.ndarray, np.ndarray]]:
    """Apply existing EMG envelope/MDF functions separately to intact epochs.

    Sampling rate is on the common session time axis. Missing values and timing
    discontinuities split epochs; short epochs retain missing processed values.
    No MVC normalization is inferred from the validation signal itself.
    """
    if stream.kind != "emg" or not np.isfinite(sampling_hz) or sampling_hz <= 0:
        raise ValueError("EMG processing requires its finite sampling rate")
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("EMG max_gap_s must be positive and finite")
    intact = np.flatnonzero(np.isfinite(stream.values).all(axis=1))
    boundaries = (
        np.flatnonzero(
            (np.diff(intact) != 1)
            | (
                np.abs(np.diff(stream.time_s[intact]) - 1 / sampling_hz)
                > 0.02 / sampling_hz
            )
            | (np.diff(stream.time_s[intact]) > max_gap_s)
            | (
                stream.break_before[intact[1:]]
                if stream.break_before is not None
                else False
            )
        )
        + 1
    )
    envelope = np.full_like(stream.values, np.nan)
    frequency_times, frequencies = [], []
    for indices in np.split(intact, boundaries):
        if len(indices) < max(32, int(0.5 * sampling_hz)):
            continue
        segment = stream.values[indices]
        envelope[indices] = emg_envelope(segment, sampling_hz)
        channels = [
            emg_median_frequency(segment[:, i], sampling_hz)
            for i in range(segment.shape[1])
        ]
        frequency_times.extend(stream.time_s[indices[0]] + channels[0][0])
        frequencies.extend(np.column_stack([item[1] for item in channels]))
    processed = ReferenceStream(
        stream.time_s,
        envelope,
        stream.channels,
        stream.units,
        "emg_envelope",
        stream.role,
        stream.provenance,
        stream.break_before,
    )
    return processed, (
        np.asarray(frequency_times),
        np.asarray(frequencies).reshape(-1, len(stream.channels)),
    )


def align_reference(
    stream: ReferenceStream, times: np.ndarray, *, max_gap_s: float
) -> np.ndarray:
    """Align measured channels without extrapolating or bridging reference gaps."""
    return sample_at(
        stream.time_s,
        stream.values,
        times,
        max_gap=max_gap_s,
        break_before=stream.break_before,
    )


def compare_reference(
    stream: ReferenceStream,
    session: Session,
    *,
    channel: str,
    prediction: str,
    max_gap_s: float,
    independent: bool,
) -> dict[str, Any]:
    """Compare held-out physical force/torque quantities using existing concordance.

    A measured force is never converted to joint torque without supplied geometry.
    EMG is a proxy signal and is exported/aligned separately, not labelled force
    agreement. Instrument axes/signs and independence are explicit declarations.
    """
    if stream.role != "validation" or independent is not True:
        raise ValueError("agreement requires explicitly independent validation data")
    if stream.kind not in {"force", "torque"}:
        raise ValueError("EMG proxies do not establish physical force agreement")
    unit = None
    if prediction.startswith("tau_"):
        dof = int(prediction[4:])
        for joint in session.record["model"]["joints"]:
            if joint["dofadr"] == dof and joint["type"] in (2, 3):
                unit = "N" if joint["type"] == 2 else "Nm"
        predicted = session.columns[prediction]
    elif prediction.startswith("contact:"):
        _, body, axis = prediction.split(":")
        predicted, unit = session.contact_force[body][:, "xyz".index(axis)], "N"
    else:
        raise ValueError("comparison needs a scalar tau_<dof> or contact:<body>:<axis>")
    index = stream.channels.index(channel)
    if unit is None or stream.units[index] != unit:
        raise ValueError("measurement/prediction units or joint type do not match")
    measured = align_reference(stream, session.time, max_gap_s=max_gap_s)[:, index]
    rejected = int(
        Flag.LOW_CONFIDENCE
        | Flag.MISSING
        | Flag.IK_RESIDUAL
        | Flag.GLITCH
        | Flag.NOT_SOLVED
        | Flag.INFERRED
    )
    keep = (
        np.isfinite(measured)
        & np.isfinite(predicted)
        & ((session.flags & rejected) == 0)
    )
    error = predicted[keep] - measured[keep]
    concordance = concordance_correlation(measured[keep], predicted[keep])
    return {
        "channel": channel,
        "prediction": prediction,
        "unit": unit,
        "paired": int(keep.sum()),
        "excluded": int((~keep).sum()),
        "bias": float(error.mean()) if len(error) else None,
        "rmse": float(np.sqrt(np.mean(error**2))) if len(error) else None,
        "concordance": float(concordance) if np.isfinite(concordance) else None,
        "time_s": session.time.tolist(),
        "measured": [float(v) if np.isfinite(v) else None for v in measured],
        "predicted": [float(v) if good else None for v, good in zip(predicted, keep)],
        "independence_declared": True,
        "provenance": stream.provenance,
    }
