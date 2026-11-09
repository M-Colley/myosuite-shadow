"""Session quality report, for the experimenter right after a participant.

Summarises what decides whether a session is usable: tracking gaps and
quality flags, solver failures, frames where the muscles fall short
(reserves), skipped and dropped frames, latency, resumes and the provenance of
the software (a dirty checkout or a moving dependency is a warning)::

    myosuite-shadow report --session logs/P07
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from myosuite_shadow.flags import Flag
from myosuite_shadow.log import Session, read_session

# Warning thresholds (fraction of estimates, or seconds).
_MAX_UNSOLVED, _MAX_CAPACITY, _MAX_FLAGGED, _MAX_LATENCY = 0.01, 0.05, 0.10, 0.25


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
    log_dir: str | Path, session: Session | None = None
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
    }
    report["warnings"] = _warnings(report, s)
    return report


def _warnings(r: dict[str, Any], s: Session) -> list[str]:
    out = []
    n = max(r["estimates"], 1)
    if not r["estimates"]:
        out.append("no estimates")
    if r["unsolved"] / n > _MAX_UNSOLVED:
        out.append(f"{r['unsolved']} estimates did not converge")
    cap = r["estimate_flags"]["CAPACITY_EXCEEDED"]
    if cap / n > _MAX_CAPACITY:
        out.append(
            f"{cap / n:.0%} of estimates need reserves: load or strength "
            "miscalibrated, "
            "or tracking glitches"
        )
    if r["flagged_estimates"] / n > _MAX_FLAGGED:
        out.append(f"{r['flagged_estimates'] / n:.0%} of estimates carry quality flags")
    if r["end_to_end_s"] and r["end_to_end_s"]["p95"] > _MAX_LATENCY:
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
    if r["events"]:
        lines += ["", "Events: " + ", ".join(r["events"])]
    lines += ["", "## Warnings", ""]
    lines += [f"- {w}" for w in r["warnings"]] or ["- none"]
    return "\n".join(lines) + "\n"
