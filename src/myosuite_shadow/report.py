"""Session quality report, for the experimenter right after a participant.

Summarises what decides whether a session is usable: tracking gaps and
quality flags, solver failures, frames where the muscles fall short
(reserves), skipped and dropped frames, latency, resumes and the provenance of
the software (a dirty checkout or a moving dependency is a warning)::

    myosuite-shadow report --session logs/P07
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from myosuite_shadow.flags import Flag
from myosuite_shadow.log import Session, read_session

# Warning thresholds (fraction of estimates, or seconds).
_MAX_UNSOLVED, _MAX_CAPACITY, _MAX_FLAGGED, _MAX_LATENCY = 0.01, 0.05, 0.10, 0.25


@dataclass(frozen=True)
class ReportThresholds:
    """Study-specific warning thresholds; these do not establish human validity."""

    unsolved_fraction: float = _MAX_UNSOLVED
    capacity_fraction: float = _MAX_CAPACITY
    flagged_fraction: float = _MAX_FLAGGED
    latency_s: float = _MAX_LATENCY
    reserve_rms_fraction: float = 0.05
    required_calibrations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("unsolved_fraction", "capacity_fraction", "flagged_fraction"):
            if (
                not np.isfinite(getattr(self, name))
                or not 0 <= getattr(self, name) <= 1
            ):
                raise ValueError(f"{name} must be within [0, 1]")
        if (
            not np.isfinite([self.latency_s, self.reserve_rms_fraction]).all()
            or min(self.latency_s, self.reserve_rms_fraction) < 0
        ):
            raise ValueError("latency and reserve thresholds must be nonnegative")


def _joint_quality(s: Session) -> list[dict[str, Any]]:
    out = []
    for joint in s.record["model"]["joints"]:
        kind, start = joint["type"], joint["dofadr"]
        count = 6 if kind == 0 else 3 if kind == 1 else 1
        for axis in range(count):
            reserve, torque = f"reserve_{start + axis}", f"tau_{start + axis}"
            if reserve not in s.columns or not len(s.time):
                continue
            r, t = s.columns[reserve], s.columns[torque]
            keep = np.isfinite(r) & np.isfinite(t)
            if not keep.any():
                continue
            rms_r = float(np.sqrt(np.mean(r[keep] ** 2)))
            rms_t = float(np.sqrt(np.mean(t[keep] ** 2)))
            unit = "N" if kind == 2 or kind == 0 and axis < 3 else "Nm"
            out.append(
                {
                    "joint": joint["name"],
                    "axis": axis,
                    "unit": unit,
                    "free_root": kind == 0,
                    "reserve_rms": rms_r,
                    "required_rms": rms_t,
                    "reserve_peak": float(np.abs(r[keep]).max()),
                    "reserve_fraction_rms": rms_r / rms_t if rms_t > 1e-9 else None,
                }
            )
    return out


def _tracking_quality(log_dir: str | Path) -> dict[str, Any] | None:
    path = Path(log_dir) / "tracking_quality.jsonl"
    if not path.exists():
        return None
    total, rejected, corrupt = 0, 0, 0
    reasons: dict[str, int] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
                total += 1
                if not row["accepted"]:
                    rejected += 1
                    reason = row["reason"]
                    reasons[reason] = reasons.get(reason, 0) + 1
            except (ValueError, KeyError, TypeError):
                corrupt += 1
    return {
        "frames": total,
        "rejected": rejected,
        "reasons": reasons,
        "unreadable_rows": corrupt,
    }


def _percentiles(x: np.ndarray) -> dict[str, float] | None:
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    return {
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(x.max()),
    }


def session_report(
    log_dir: str | Path,
    session: Session | None = None,
    *,
    thresholds: ReportThresholds | None = None,
) -> dict[str, Any]:
    """Quality summary of a session log.

    Args:
        log_dir: The session's directory.
        session: The session, if already read.

    Returns:
        A JSON-serialisable summary with a ``warnings`` list.
    """
    s = session or read_session(log_dir)
    cols, n = s.columns, len(s.time)
    flags = s.flags
    span = float(s.input_time[-1] - s.input_time[0]) if len(s.input_time) > 1 else 0.0
    flag_counts = {
        f.name: int(((flags & f) != 0).sum()) for f in Flag if f != Flag.NONE
    }
    input_flags = s.input_columns["flags"].astype(int)
    input_counts = {
        f.name: int(((input_flags & f) != 0).sum()) for f in Flag if f != Flag.NONE
    }
    gaps = cols["gap_s"][cols["gap_s"] > 0]
    live = [e["data"] for e in s.events if e["label"] == "live_summary"]
    end_to_end = cols["done_unix"] - cols["frame_unix"]
    report: dict[str, Any] = {
        "session": str(log_dir),
        "created_utc": s.record["created_utc"],
        "metadata": s.record.get("metadata", {}),
        "duration_s": span,
        "poses": len(s.input_time),
        "estimates": n,
        "estimate_rate_hz": (n - 1) / max(float(s.time[-1] - s.time[0]), 1e-9)
        if n > 1
        else 0.0,
        "delay_s": _percentiles(cols["latest_time"] - cols["time"]),
        "compute_ms": _percentiles(cols["compute_ms"]),
        "end_to_end_s": _percentiles(end_to_end),
        "unsolved": int((s.status != "solved").sum()),
        "estimate_flags": flag_counts,
        "input_flags": input_counts,
        "flagged_estimates": int((flags != 0).sum()),
        "gaps": {"count": len(gaps), "total_s": float(gaps.sum())},
        "skipped": int(cols["skipped"].sum()),
        "dropped": int(sum(d.get("dropped", 0) for d in live)),
        "max_reserve": float(np.nanmax(cols["max_abs_reserve"], initial=0.0)),
        "peak_fatigue": (
            {}
            if s.fatigue is None or not n
            else dict(
                sorted(
                    zip(s.actuators, s.fatigue.max(axis=0).round(4).tolist()),
                    key=lambda kv: -kv[1],
                )[:5]
            )
        ),
        "events": [e["label"] for e in s.events],
        "resumes": len(s.record.get("resumes", [])),
        "closed_cleanly": bool(s.checkpoint and s.checkpoint.get("closed")),
        "packages": {
            k: {key: v[key] for key in ("version", "commit", "dirty") if key in v}
            for k, v in s.record["packages"].items()
            if v.get("version") is not None
        },
        "joint_quality": _joint_quality(s),
        "tracking": _tracking_quality(log_dir),
        "exact_schedule": s.operations is not None
        and not s.record.get("schedule_reconstructed", False),
        "live_latency": not (
            s.record.get("metadata", {}).get("replay")
            or s.record.get("metadata", {}).get("acquisition_mode")
            == "offline_reprocess"
        ),
    }
    limits = thresholds or ReportThresholds()
    report["thresholds"] = asdict(limits)
    report["warnings"] = _warnings(report, s, limits)
    return report


def _warnings(r: dict[str, Any], s: Session, limits: ReportThresholds) -> list[str]:
    out = []
    n = max(r["estimates"], 1)
    if not r["estimates"]:
        out.append("no estimates")
    if r["unsolved"] / n > limits.unsolved_fraction:
        out.append(f"{r['unsolved']} estimates did not converge")
    cap = r["estimate_flags"]["CAPACITY_EXCEEDED"]
    if cap / n > limits.capacity_fraction:
        out.append(
            f"{cap / n:.0%} of estimates need reserves: load or strength "
            "miscalibrated, "
            "or tracking glitches"
        )
    if r["flagged_estimates"] / n > limits.flagged_fraction:
        out.append(f"{r['flagged_estimates'] / n:.0%} of estimates carry quality flags")
    if (
        r["live_latency"]
        and r["end_to_end_s"]
        and r["end_to_end_s"]["p95"] > limits.latency_s
    ):
        out.append(
            f"95th-percentile end-to-end latency {r['end_to_end_s']['p95']:.2f} s"
        )
    if r["dropped"]:
        out.append(f"{r['dropped']} poses dropped by the live runner")
    if r["resumes"]:
        out.append(f"resumed {r['resumes']} time(s) after a crash")
    if not r["closed_cleanly"]:
        out.append("not closed cleanly: resume it, or treat the end as lost")
    for name, pkg in s.record["packages"].items():
        if pkg.get("dirty"):
            out.append(f"{name} ran from a checkout with uncommitted changes")
        rev = pkg.get("requested_revision")
        if rev and pkg.get("commit") and rev != pkg["commit"]:
            out.append(
                f"{name} installed from the moving ref {rev!r}; "
                f"pin {pkg['commit'][:12]} "
                "(uv.lock) for the study"
            )
    if not s.record.get("calibration"):
        out.append(
            "uncalibrated model: absolute activations and fatigue rates are "
            "population defaults"
        )
    for key in limits.required_calibrations:
        if key not in s.record.get("calibration", {}):
            out.append(f"missing required calibration {key!r}")
    if not r["exact_schedule"]:
        out.append(
            "legacy log lacks the original optimization/reset schedule; "
            "replay is a recomputation"
        )
    for joint in r["joint_quality"]:
        ratio = joint["reserve_fraction_rms"]
        if (
            not joint["free_root"]
            and ratio is not None
            and ratio > limits.reserve_rms_fraction
        ):
            out.append(
                f"{joint['joint']} axis {joint['axis']}: reserve RMS is "
                f"{ratio:.0%} of required torque/force"
            )
    if r["estimate_flags"].get("INFERRED", 0):
        out.append(
            "estimates include inferred tracking landmarks; "
            "validate those tasks separately"
        )
    if r["tracking"] and r["tracking"]["unreadable_rows"]:
        out.append("raw tracking quality record has unreadable or torn rows")
    return out


def format_report(report: dict[str, Any]) -> str:
    """Markdown rendering of :func:`session_report`."""

    def pct(d: dict[str, float] | None, unit: str, scale: float = 1.0) -> str:
        if d is None:
            return "n/a"
        return (
            f"median {scale * d['median']:.1f} {unit}, p95 {scale * d['p95']:.1f}, "
            f"max {scale * d['max']:.1f}"
        )

    r = report
    lines = [
        f"# Session report: {r['session']}",
        "",
        f"- Duration {r['duration_s']:.1f} s, {r['poses']} poses, {r['estimates']} "
        f"estimates ({r['estimate_rate_hz']:.1f} Hz)",
        f"- Delay {pct(r['delay_s'], 'ms', 1e3)}",
        f"- Update {pct(r['compute_ms'], 'ms')}",
        f"- End to end (frame capture to estimate) {pct(r['end_to_end_s'], 'ms', 1e3)}",
        f"- Gaps: {r['gaps']['count']} ({r['gaps']['total_s']:.1f} s untracked); "
        f"skipped {r['skipped']}, dropped {r['dropped']}",
        f"- Not converged: {r['unsolved']}; "
        f"flagged estimates: {r['flagged_estimates']}",
        f"- Peak reserve {r['max_reserve']:.2f}; resumes {r['resumes']}; "
        f"closed cleanly: {'yes' if r['closed_cleanly'] else 'no'}",
        "",
        "| Flag | Estimates | Poses |",
        "|---|---|---|",
    ]
    for name, count in r["estimate_flags"].items():
        if count or r["input_flags"][name]:
            lines.append(f"| {name} | {count} | {r['input_flags'][name]} |")
    if r["peak_fatigue"]:
        lines += [
            "",
            "Peak MF: "
            + ", ".join(f"{k} {v:.2f}" for k, v in r["peak_fatigue"].items()),
        ]
    if r.get("tracking"):
        tracking = r["tracking"]
        lines += [
            "",
            f"Raw acquisition: {tracking['frames']} frames, "
            f"{tracking['rejected']} rejected before physics.",
        ]
    if not r.get("live_latency", True):
        lines += [
            "",
            "Offline processing: capture-to-completion includes recording age.",
        ]
    if r.get("joint_quality"):
        lines += [
            "",
            "| Joint | Axis | Reserve RMS | Required RMS | Unit | Root residual |",
            "|---|---|---|---|---|---|",
        ]
        for joint in r["joint_quality"]:
            lines.append(
                f"| {joint['joint']} | {joint['axis']} | "
                f"{joint['reserve_rms']:.3g} | {joint['required_rms']:.3g} | "
                f"{joint['unit']} | {joint['free_root']} |"
            )
    if r["events"]:
        lines += ["", "Events: " + ", ".join(r["events"])]
    lines += ["", "## Warnings", ""]
    lines += [f"- {w}" for w in r["warnings"]] or ["- none"]
    return "\n".join(lines) + "\n"
