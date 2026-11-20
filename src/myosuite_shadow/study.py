"""Evaluate independent MVC probes with recorded calibration and quality evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from myosuite_shadow.flags import Flag
from myosuite_shadow.log import read_session
from myosuite_shadow.validation import concordance_correlation, mvc_capacity


def validate_probes(session_dir: str | Path, probes: str | Path) -> dict[str, Any]:
    """Compare model MVC torque with independent, posture-matched measurements.

    Args:
        session_dir: Session including its calibrated model.
        probes: CSV containing time, joint, measured_torque, role and qpos_i.
            Only rows explicitly labelled ``role=validation`` are evaluated.

    Returns:
        Per-probe predictions, exclusions, joint-wise agreement and provenance.
    """
    session_dir = Path(session_dir)
    session = read_session(session_dir)
    model = session.load_model(session_dir)
    with open(probes, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"time", "joint", "measured_torque", "role"}
        required |= {f"qpos_{j}" for j in range(model.nq)}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"probe CSV requires columns {sorted(required)}")
        rows = list(reader)
    results = []
    for row in rows:
        role = row["role"].strip()
        if role not in {"calibration", "validation"}:
            raise ValueError("probe role must be calibration or validation")
        t, measured = float(row["time"]), float(row["measured_torque"])
        q = np.array([[float(row[f"qpos_{j}"]) for j in range(model.nq)]])
        if not np.isfinite([t, measured]).all() or measured == 0:
            raise ValueError("probe time and nonzero signed torque must be finite")
        reason = "used_for_calibration" if role == "calibration" else None
        prediction = float("nan")
        if reason is None:
            prediction = mvc_capacity(
                session,
                model,
                row["joint"],
                np.array([t]),
                q,
                direction=float(np.sign(measured)),
                reject_flags=int(
                    Flag.LOW_CONFIDENCE
                    | Flag.MISSING
                    | Flag.IK_RESIDUAL
                    | Flag.GLITCH
                    | Flag.NOT_SOLVED
                    | Flag.INFERRED
                ),
            )[0]
            if not np.isfinite(prediction):
                reason = "tracking_gap_or_quality"
        results.append(
            {
                "time": t,
                "joint": row["joint"],
                "role": role,
                "posture": q[0].tolist(),
                "unit": "N" if model.joint(row["joint"]).type[0] == 2 else "Nm",
                "measured_torque": measured,
                "predicted_torque": float(prediction)
                if np.isfinite(prediction)
                else None,
                "exclusion": reason,
            }
        )
    groups = sorted({(r["joint"], int(np.sign(r["measured_torque"]))) for r in results})
    metrics = []
    for joint, direction in groups:
        valid = [
            r
            for r in results
            if r["exclusion"] is None
            and r["joint"] == joint
            and int(np.sign(r["measured_torque"])) == direction
        ]
        if not valid:
            continue
        measured = np.array([r["measured_torque"] for r in valid])
        predicted = np.array([r["predicted_torque"] for r in valid])
        error = predicted - measured
        ccc = concordance_correlation(measured, predicted)
        sd = float(error.std(ddof=1)) if len(error) > 1 else None
        metrics.append(
            {
                "joint": joint,
                "unit": valid[0]["unit"],
                "direction": direction,
                "n": len(valid),
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt((error**2).mean())),
                "bias": float(error.mean()),
                "error_sd": sd,
                "limits_of_agreement": None
                if sd is None
                else [float(error.mean() - 1.96 * sd), float(error.mean() + 1.96 * sd)],
                "concordance": float(ccc) if np.isfinite(ccc) else None,
            }
        )
    return {
        "session": str(session_dir),
        "probes_file": str(probes),
        "probes_sha256": hashlib.sha256(Path(probes).read_bytes()).hexdigest(),
        "model_sha256": session.record["model_sha256"],
        "calibration": session.record.get("calibration", {}),
        "probes": results,
        "metrics": metrics,
        "included": sum(r["exclusion"] is None for r in results),
        "excluded": sum(r["exclusion"] is not None for r in results),
        "interpretation": (
            "Independent probe analysis; acceptance criteria belong to "
            "the study protocol."
        ),
        "agreement_method": (
            "Descriptive within-session limits; probes are repeated observations, "
            "not independent participants."
        ),
    }


def main() -> None:
    """Write an auditable validation report from a session and probe CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = validate_probes(args.session, args.probes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(
        f"{report['included']} independent probes; "
        f"{report['excluded']} excluded; {args.out}"
    )


if __name__ == "__main__":
    main()
