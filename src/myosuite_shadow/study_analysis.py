"""Trial summaries with explicit coverage and semantic outcome contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from myosuite_shadow.flags import Flag
from myosuite_shadow.log import Session, package_provenance, read_session
from myosuite_shadow.study_protocol import TrialJournal

_FILES = (
    "record.json",
    "inputs.csv",
    "estimates.csv",
    "events.csv",
    "checkpoint.json",
    "operations.jsonl",
    "model.mjb",
    "tracking.jsonl",
    "tracking_quality.jsonl",
)


def _hashes(path: Path) -> dict[str, str]:
    result = {}
    for name in _FILES:
        file = path / name
        if file.exists():
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            result[name] = digest.hexdigest()
    return result


def _selector(session: Session, outcome: dict) -> tuple[np.ndarray, dict]:
    selector = outcome["selector"]
    if outcome["normalization"] != "none":
        raise ValueError("only normalization='none' is implemented")
    contract = {key: value for key, value in outcome.items() if key != "selector"}
    contract["analysis_recipe_sha256"] = hashlib.sha256(
        Path(__file__).read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()
    contract["model_implementation"] = {
        name: {key: info.get(key) for key in ("version", "commit", "dirty")}
        for name, info in session.record.get("packages", {}).items()
        if name in {"myosuite", "mujoco", "osqp"}
    }
    if selector.startswith(("tau_", "reserve_")):
        index = int(selector.rsplit("_", 1)[1])
        joint = next(
            (
                j
                for j in session.record["model"]["joints"]
                if j["dofadr"] == index and j["type"] in (2, 3)
            ),
            None,
        )
        if joint is None:
            raise ValueError(f"{selector} requires a scalar hinge/slide joint")
        if outcome["coordinate_frame"] != "model_joint":
            raise ValueError("joint outcomes require coordinate_frame='model_joint'")
        contract["model_coordinate"] = joint["name"]
        unit = "N" if joint["type"] == 2 else "Nm"
        prefix = "required" if selector.startswith("tau_") else "reserve"
        contract["quantity_class"] = prefix + (
            "_joint_force" if unit == "N" else "_joint_torque"
        )
        values = session.columns[selector]
    elif selector.startswith(("activation:", "fatigue:")):
        kind, muscle = selector.split(":", 1)
        index = session.actuators.index(muscle)
        if outcome["coordinate_frame"] != "model_actuator":
            raise ValueError(
                "muscle outcomes require coordinate_frame='model_actuator'"
            )
        contract["model_coordinate"] = muscle
        values = session.activation if kind == "activation" else session.fatigue
        if values is None:
            raise ValueError(f"{selector} was not recorded")
        values, unit = values[:, index], "1"
        contract["quantity_class"] = (
            "estimated_activation" if kind == "activation" else "latent_fatigue"
        )
    elif selector.startswith("contact:"):
        _, body, axis = selector.split(":")
        values, unit = session.contact_force[body][:, "xyz".index(axis)], "N"
        if outcome["coordinate_frame"] != "model_world":
            raise ValueError("contact forces use the model_world coordinate frame")
        contract["quantity_class"] = "estimated_contact_force"
        contract["model_coordinate"] = body + ":" + axis
    else:
        raise ValueError(f"unsupported outcome selector: {selector}")
    if outcome["unit"] != unit:
        raise ValueError(f"{selector} requires unit {unit}")
    return np.asarray(values, dtype=float), contract


def _within(times: np.ndarray, intervals: list[list[float]]) -> np.ndarray:
    mask = np.zeros(len(times), dtype=bool)
    for start, stop in intervals:
        mask |= (times >= start) & (times < stop)
    return mask


def _pairs(
    times: np.ndarray,
    eligible: np.ndarray,
    intervals: list[list[float]],
    max_gap: float,
    continuity: np.ndarray | None = None,
) -> np.ndarray:
    """Only adjacent, eligible samples wholly inside one active interval."""
    pairs = np.zeros(max(len(times) - 1, 0), dtype=bool)
    if not len(pairs):
        return pairs
    for start, stop in intervals:
        pairs |= (times[:-1] >= start) & (times[1:] < stop)
    dt = np.diff(times)
    pairs &= eligible[:-1] & eligible[1:] & (dt > 0) & (dt <= max_gap)
    return pairs if continuity is None else pairs & continuity


def _continuity(session: Session, rows: list[dict]) -> np.ndarray:
    times = session.time
    allowed = np.ones(max(len(times) - 1, 0), dtype=bool)
    rejected = np.sort([row["time_s"] for row in rows if not row["accepted"]])
    if len(rejected):
        allowed &= np.searchsorted(
            rejected, times[:-1], side="left"
        ) == np.searchsorted(rejected, times[1:], side="right")
    breaks = []
    for operation in getattr(session, "operations", None) or []:
        index = operation["input"]
        if operation["kind"] == "reset" and index < len(session.input_time):
            breaks.append(session.input_time[index])
    if breaks:
        epochs = np.searchsorted(np.sort(breaks), times, side="right")
        allowed &= epochs[:-1] == epochs[1:]
    allowed &= (session.flags[1:] & Flag.AFTER_GAP) == 0
    return allowed


def _tracking(path: Path) -> tuple[list[dict], int]:
    rows, corrupt = [], 0
    if not path.exists():
        return rows, corrupt
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
                stamp = row["time_s"]
                if (
                    isinstance(stamp, bool)
                    or not isinstance(stamp, int | float)
                    or not np.isfinite(stamp)
                    or not isinstance(row["accepted"], bool)
                ):
                    raise ValueError("invalid tracking decision")
                rows.append(row)
            except (ValueError, TypeError, KeyError):
                corrupt += 1
    return rows, corrupt


def study_report(journal_path: str | Path) -> dict[str, Any]:
    """Summarize trial intervals, retaining unknown and rejected evidence.

    Time means/RMS integrate piecewise linear values only between adjacent
    eligible samples. They never bridge pauses, invalid samples or gaps, or
    extrapolate to trial boundaries. Coverage includes intervals with no
    observations in its denominator. Peaks are absolute magnitudes.
    """
    journal = TrialJournal.open(journal_path)
    snapshot = journal.snapshot()
    if not snapshot["session_path"]:
        raise ValueError("study run is not bound to a recorded session")
    root = Path(snapshot["session_path"])
    before = _hashes(root)
    session = read_session(root)
    if session.record.get("metadata", {}).get("study") != snapshot["context"]:
        raise ValueError("session study context does not match the trial journal")
    times = session.time
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("estimate times must be finite and strictly increasing")
    definition = snapshot["definition"]
    policy = definition["quality"]
    rows, corrupt = _tracking(root / "tracking_quality.jsonl")
    tracked = (root / "tracking_quality.jsonl").exists()
    continuity = _continuity(session, rows)
    eligible = (session.status == "solved") & (
        (session.flags & policy["reject_flags"]) == 0
    )
    selected = [
        (outcome, *_selector(session, outcome)) for outcome in definition["outcomes"]
    ]
    for _, _, contract in selected:
        contract["quality_policy"] = policy
    trials = []
    closed = bool(session.checkpoint and session.checkpoint.get("closed"))
    for trial in snapshot["trials"]:
        intervals = trial["intervals"]
        duration = sum(stop - start for start, stop in intervals)
        observed = _within(times, intervals)
        pairs = _pairs(times, eligible, intervals, policy["max_gap_s"], continuity)
        supported = float(np.diff(times)[pairs].sum())
        coverage = supported / duration if duration > 0 else 0.0
        raw = [row for row in rows if any(a <= row["time_s"] < b for a, b in intervals)]
        rejected = sum(not row["accepted"] for row in raw)
        reasons: dict[str, int] = {}
        for row in raw:
            if not row["accepted"]:
                reason = str(row.get("reason") or "unspecified")
                reasons[reason] = reasons.get(reason, 0) + 1
        issues = []
        if trial["state"] != "completed":
            issues.append("trial_not_completed")
        if not duration:
            issues.append("no_active_duration")
        if coverage + 1e-12 < policy["min_coverage"]:
            issues.append("insufficient_estimate_coverage")
        if policy["require_tracking"] and (not tracked or not raw):
            issues.append("tracking_evidence_missing")
        if corrupt:
            issues.append("unassignable_tracking_rows")
        if raw and rejected / len(raw) > policy["max_rejected_fraction"]:
            issues.append("excessive_tracking_rejection")
        if not closed:
            issues.append("session_not_closed")
        status = "usable" if not issues else "needs_review"
        if not duration or not supported or trial["state"] == "aborted":
            status = "unusable"
        outcomes = {}
        for outcome, values, contract in selected:
            keep = _pairs(
                times,
                eligible & np.isfinite(values),
                intervals,
                policy["max_gap_s"],
                continuity,
            )
            dt = np.diff(times)[keep]
            seconds = float(dt.sum())
            available = seconds / duration if duration > 0 else 0.0
            value = None
            if seconds and available + 1e-12 >= policy["min_coverage"]:
                left, right = values[:-1][keep], values[1:][keep]
                if outcome["statistic"] == "time_mean":
                    value = float(np.sum(dt * (left + right) / 2) / seconds)
                elif outcome["statistic"] == "rms":
                    value = float(
                        np.sqrt(
                            np.sum(dt * (left**2 + left * right + right**2) / 3)
                            / seconds
                        )
                    )
                else:
                    value = float(np.max(np.abs(np.concatenate((left, right)))))
            outcomes[outcome["name"]] = {
                "value": value,
                "contract": contract,
                "coverage": available,
                "supported_duration_s": seconds,
                "exclusion": "insufficient_outcome_coverage" if value is None else None,
            }
        trials.append(
            {
                **trial,
                "condition_values": definition["conditions"][trial["condition"]],
                "quality": {
                    "scope": "acquisition_and_estimation",
                    "status": status,
                    "issues": issues,
                    "active_duration_s": duration,
                    "supported_duration_s": supported,
                    "coverage": coverage,
                    "estimates": int(observed.sum()),
                    "eligible_estimates": int((observed & eligible).sum()),
                    "raw_frames": len(raw) if tracked else None,
                    "rejected_frames": rejected if tracked else None,
                    "rejected_fraction": rejected / len(raw) if raw else None,
                    "rejection_reasons": reasons,
                    "inferred_frames": sum(
                        bool(row.get("inferred_landmarks")) for row in raw
                    ),
                    "flag_counts": {
                        str(int(flag)): int(
                            ((session.flags[observed] & flag) != 0).sum()
                        )
                        for flag in (1, 2, 4, 8, 16, 32, 64, 128, 256)
                    },
                },
                "outcomes": outcomes,
            }
        )
    if before != _hashes(root) or snapshot != journal.snapshot():
        raise ValueError(
            "study evidence changed during analysis; close capture before reporting"
        )
    return {
        "schema_version": 1,
        "context": snapshot["context"],
        "definition": definition,
        "state": snapshot["state"],
        "trials": trials,
        "provenance": {
            "session_files": before,
            "model_sha256": session.record["model_sha256"],
            "calibration": session.record.get("calibration", {}),
            "capture_packages": session.record.get("packages", {}),
            "analysis_packages": package_provenance(
                ("myosuite-shadow", "myosuite", "numpy", "scipy")
            ),
            "trial_events": snapshot["events"],
        },
        "interpretation": (
            "Model estimates and latent states; quality is not empirical accuracy. "
            "Trial boundaries use caller-supplied mapped session times."
        ),
    }
