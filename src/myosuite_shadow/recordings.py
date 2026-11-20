"""Auditable external recordings and deterministic regression of recorded studies."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from myosuite_shadow.acquisition import TrackedSession
from myosuite_shadow.live import LiveShadowModel
from myosuite_shadow.log import model_sha256, package_provenance, read_session
from myosuite_shadow.pipeline import ShadowModel
from myosuite_shadow.references import compare_reference, load_reference, process_emg
from myosuite_shadow.study import validate_probes
from myosuite_shadow.tracking import Registration, read_frames
from myosuite_shadow.tracking_config import load_tracking_setup

_SESSION_FILES = (
    "record.json",
    "inputs.csv",
    "estimates.csv",
    "events.csv",
    "operations.jsonl",
    "checkpoint.json",
    "model.mjb",
)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(base: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def freeze_recording(draft: str | Path, destination: str | Path) -> dict[str, Any]:
    """Hash externally supplied artifacts; never copy or generate participant data.

    The operator declares real versus synthetic origin and actual source versions.
    Freezing establishes file identity, not scientific accuracy or data authenticity.
    """
    draft, destination = Path(draft).resolve(), Path(destination)
    manifest = _json(draft)
    for item in manifest["files"].values():
        path = _path(draft.parent, item["path"])
        item.update(path=str(path), sha256=_hash(path))
    setup = load_tracking_setup(manifest["files"]["config"]["path"])
    manifest["compiled_model_sha256"] = model_sha256(setup.model)
    baseline = manifest.get("baseline_session")
    if baseline is not None:
        path = _path(draft.parent, baseline["path"])
        read_session(path)
        baseline.update(
            path=str(path),
            sha256={
                name: _hash(path / name)
                for name in _SESSION_FILES
                if (path / name).exists()
            },
        )
    manifest["software_versions"] = package_provenance()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, allow_nan=False)
    return manifest


def audit_recording(manifest_path: str | Path) -> dict[str, Any]:
    """Verify frozen files and declared identity before opening a replay output."""
    manifest_path = Path(manifest_path).resolve()
    manifest = _json(manifest_path)
    if (
        not isinstance(manifest, dict)
        or isinstance(manifest.get("schema_version"), bool)
        or manifest.get("schema_version") != 1
        or manifest.get("origin") not in {"real", "synthetic"}
    ):
        raise ValueError(
            "recording requires schema_version=1 and explicit real/synthetic origin"
        )
    for key in ("recording_id", "source_id", "person_id"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError(f"recording {key} is required")
    if (
        not isinstance(manifest.get("source_versions"), dict)
        or not manifest["source_versions"]
    ):
        raise ValueError("actual source/device/model versions must be declared")
    files = manifest["files"]
    if not {"frames", "config", "model"}.issubset(files):
        raise ValueError("recording needs frames, config and model artifacts")
    checked = {}
    for name, item in files.items():
        if re.fullmatch(r"[A-Za-z0-9_-]+", name) is None:
            raise ValueError("artifact identifiers must be simple names, not paths")
        if item.get("role") not in {
            "recording",
            "configuration",
            "calibration",
            "validation",
            "baseline",
        }:
            raise ValueError(f"artifact role must be declared: {name}")
        path = _path(manifest_path.parent, item["path"])
        actual = _hash(path)
        if actual != item.get("sha256", "").lower():
            raise ValueError(f"recording artifact hash mismatch: {name}")
        checked[name] = {
            "path": str(path),
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    settings_path = Path(checked["config"]["path"])
    settings = _json(settings_path)
    if _path(settings_path.parent, settings["model"]) != Path(checked["model"]["path"]):
        raise ValueError("frozen model is not the tracking configuration's model")
    if model_sha256(load_tracking_setup(settings_path).model) != manifest.get(
        "compiled_model_sha256"
    ):
        raise ValueError("compiled model hash mismatch (including external assets)")
    count, missing_receive, revisions, clock_changes, prior_clock = 0, 0, set(), 0, None
    raw_fields = {
        e["raw_field"]: []
        for e in manifest.get("expected_quality", [])
        if "raw_field" in e
    }
    for frame in read_frames(checked["frames"]["path"]):
        if (frame.source_id, frame.person_id) != (
            manifest["source_id"],
            manifest["person_id"],
        ):
            raise ValueError(f"source/person identity mismatch at raw row {count}")
        for field, rows in raw_fields.items():
            if frame.metadata.get(field):
                rows.append(count)
        count += 1
        missing_receive += frame.receive_unix is None
        revisions.add((frame.transform_revision, frame.skeleton_revision))
        clock = frame.metadata.get("clock_mapping", frame.metadata.get("clock"))
        if count > 1 and clock != prior_clock:
            clock_changes += 1
        prior_clock = clock
    if not count:
        raise ValueError("recording has no raw frames")
    for expected in manifest.get("expected_quality", []) + manifest.get(
        "expected_events", []
    ):
        if not isinstance(expected.get("event"), str) or not expected["event"]:
            raise ValueError("quality expectations need a named observed event")
        lower, upper = expected.get("min_count", 1), expected.get("max_count", count)
        if (
            any(isinstance(v, bool) or not isinstance(v, int) for v in (lower, upper))
            or not 0 <= lower <= upper
        ):
            raise ValueError(
                "expected event counts must be nonnegative ordered integers"
            )
        if "label" in expected:
            continue
        if not {"accepted", "reason", "raw_field"}.intersection(expected):
            raise ValueError(
                "quality expectations need a decision or raw metadata selector"
            )
        start, end = expected.get("rows", [0, count])
        if (
            any(isinstance(v, bool) or not isinstance(v, int) for v in (start, end))
            or not 0 <= start < end <= count
        ):
            raise ValueError("quality row windows must be valid half-open intervals")
    if any("label" in e for e in manifest.get("expected_quality", [])):
        raise ValueError(
            "pipeline labels belong in expected_events, not expected_quality"
        )
    if any(
        not isinstance(e.get("label"), str) or not e["label"]
        for e in manifest.get("expected_events", [])
    ):
        raise ValueError("expected pipeline events need their recorded label")
    if set(manifest.get("tolerances", {})) - {
        "input_qpos",
        "input_qfrc",
        "activation",
        "fatigue",
        "torque",
        "reserve",
        "contact",
    }:
        raise ValueError(
            "quality flags and timestamps require exact regression equality"
        )
    for tolerance in manifest.get("tolerances", {}).values():
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("regression tolerances must be finite and nonnegative")
    timing = manifest.get("timing", "recorded_receive")
    if (
        timing not in {"recorded_receive", "offline"}
        or timing == "recorded_receive"
        and missing_receive
    ):
        raise ValueError(
            "freshness replay requires every recorded receive_unix; "
            "choose offline explicitly otherwise"
        )
    baseline = manifest.get("baseline_session")
    if baseline is not None:
        path = _path(manifest_path.parent, baseline["path"])
        if not {"record.json", "inputs.csv", "estimates.csv"}.issubset(
            baseline["sha256"]
        ):
            raise ValueError(
                "baseline session hashes must include record, inputs and estimates"
            )
        if any(
            (path / name).exists() and name not in baseline["sha256"]
            for name in _SESSION_FILES
        ):
            raise ValueError(
                "every baseline session file used in replay must be hashed"
            )
        for name, digest in baseline["sha256"].items():
            if name not in _SESSION_FILES or _hash(path / name) != digest:
                raise ValueError(f"baseline hash mismatch or unsupported file: {name}")
        read_session(path)
    for control in manifest.get("controls", []):
        row = control["before_row"]
        if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < count:
            raise ValueError("replay control requires a valid raw row index")
        if control["kind"] == "registration":
            Registration(**_json(Path(checked[control["file"]]["path"])))
        elif control["kind"] != "resume":
            raise ValueError(
                "only measured registration and explicit resume controls are supported"
            )
    for reference in manifest.get("references", []):
        if reference["file"] not in checked:
            raise ValueError("reference must select a frozen artifact")
        if reference["kind"] != "mvc_probes" and (
            reference.get("role") not in {"calibration", "validation"}
            or files[reference["file"]]["role"] != reference["role"]
        ):
            raise ValueError("reference role must match its declared artifact role")
        if reference["kind"] == "mvc_probes" and (
            reference.get("independent") is not True
            or files[reference["file"]]["role"] != "validation"
        ):
            raise ValueError(
                "MVC probes require declared independent validation artifacts"
            )
        if reference.get("person_id", manifest["person_id"]) != manifest["person_id"]:
            raise ValueError("reference and recording must identify the same person")
        if (
            "calibration_file" in reference
            and files[reference["calibration_file"]].get("role") != "calibration"
        ):
            raise ValueError(
                "instrument calibration must select a calibration artifact"
            )
    return {
        "recording_id": manifest["recording_id"],
        "origin_declared": manifest["origin"],
        "compiled_model_sha256": manifest["compiled_model_sha256"],
        "frames": count,
        "missing_receive_times": missing_receive,
        "revisions": [list(v) for v in sorted(revisions)],
        "clock_changes": clock_changes,
        "files": checked,
        "timing": timing,
        "raw_fields": raw_fields,
        "evidence": (
            "File/identity audit; this does not establish human measurement accuracy."
        ),
    }


def _checks(
    manifest: dict, decisions: list[dict], events: list[dict], raw_fields: dict
) -> list[dict]:
    checks = []
    for expected in manifest.get("expected_quality", []):
        start, end = expected.get("rows", [0, len(decisions)])
        selected = decisions[start:end]
        matches = sum(
            all(
                row.get(k) == expected[k]
                for k in ("accepted", "reason")
                if k in expected
            )
            and (
                "raw_field" not in expected
                or index in raw_fields[expected["raw_field"]]
            )
            for index, row in enumerate(selected, start)
        )
        checks.append(
            {
                "event": expected["event"],
                "observed": matches,
                "passed": expected.get("min_count", 1)
                <= matches
                <= expected.get("max_count", len(selected)),
            }
        )
    for expected in manifest.get("expected_events", []):
        matches = sum(row["label"] == expected["label"] for row in events)
        checks.append(
            {
                "event": expected["event"],
                "observed": matches,
                "passed": expected.get("min_count", 1)
                <= matches
                <= expected.get("max_count", len(events)),
            }
        )
    return checks


def _compare(a: Any, b: Any, atol: float) -> dict[str, Any]:
    if a is None or b is None:
        return {"passed": a is b, "max_abs_difference": None}
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return {
            "passed": False,
            "actual_shape": list(a.shape),
            "baseline_shape": list(b.shape),
        }
    error = np.abs(a - b)
    return {
        "passed": bool(np.allclose(a, b, rtol=0, atol=atol, equal_nan=True)),
        "max_abs_difference": float(error[np.isfinite(error)].max())
        if np.isfinite(error).any()
        else None,
    }


def run_recording_regression(
    manifest_path: str | Path, out_dir: str | Path
) -> dict[str, Any]:
    """Replay external raw data, check annotations/baselines and align references.

    Queue cadence is deterministic. When a baseline exists, physiology is also
    replayed under its original recorded skip/reset schedule using the new IK
    poses/loads, rather than pretending the live optimizer ran on every frame.
    """
    manifest_path, out_dir = Path(manifest_path).resolve(), Path(out_dir)
    audit, manifest = audit_recording(manifest_path), _json(manifest_path)
    files = {name: Path(item["path"]) for name, item in audit["files"].items()}
    setup = load_tracking_setup(files["config"], offline=audit["timing"] == "offline")
    baseline_spec = manifest.get("baseline_session")
    baseline = (
        read_session(_path(manifest_path.parent, baseline_spec["path"]))
        if baseline_spec
        else None
    )
    if baseline is not None and baseline.record["model_sha256"] != model_sha256(
        setup.model
    ):
        raise ValueError(
            "regression baseline and replay must use the same subject model"
        )
    load_mode = manifest.get("loads", "configuration")
    if load_mode not in {"configuration", "baseline_inputs"}:
        raise ValueError("loads must select configuration or baseline_inputs")
    loads = None
    if load_mode == "baseline_inputs":
        if (
            baseline is None
            or baseline.input_qfrc is None
            or not len(baseline.input_time)
        ):
            raise ValueError("baseline_inputs requires frozen baseline external loads")
        if setup.acquisition.held_mass_kg:
            raise ValueError("baseline loads cannot also include configured held mass")
        loads = dict(zip(baseline.input_time, baseline.input_qfrc))
    elif baseline is not None and baseline.input_qfrc is not None:
        if not setup.acquisition.held_mass_kg and np.any(baseline.input_qfrc):
            raise ValueError(
                "baseline contains external loads; explicitly select baseline_inputs"
            )
    out_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "audit": audit,
        "status": "running",
        "software_versions": package_provenance(),
        "frozen_software_versions": manifest.get("software_versions", {}),
        "source_versions_declared": manifest["source_versions"],
        "manifest_sha256": _hash(manifest_path),
        "mode": "offline_reprocess",
        "timing": audit["timing"],
        "freshness_limit": (
            "Receive timestamps are replayed; original CPU queuing delays "
            "are not recreated."
        ),
        "loads": load_mode,
        "load_provenance": {
            "baseline_inputs_sha256": baseline_spec["sha256"]["inputs.csv"],
            "time_bounds_s": [
                float(baseline.input_time[0]),
                float(baseline.input_time[-1]),
            ],
            "matching": "exact accepted input time; no interpolation or fitting",
            "independent_validation": False,
        }
        if loads is not None
        else {"configuration_sha256": manifest["files"]["config"]["sha256"]},
    }
    shadow, live, acquisition = None, None, None
    observed = [0.0]

    def external_loads(time_s, qpos):
        if time_s not in loads:
            raise ValueError("accepted pose has no exact frozen baseline load time")
        return loads[time_s].copy()

    def start(resume=False):
        nonlocal shadow, live, acquisition
        shadow, live, acquisition = None, None, None
        shadow = (
            ShadowModel.resume(out_dir / "session", setup.model)
            if resume
            else ShadowModel(
                setup.model,
                setup.shadow_config,
                log_dir=out_dir / "session",
                calibration=setup.settings.get("calibration", {}),
                metadata={
                    "recording_id": manifest["recording_id"],
                    "person_id": manifest["person_id"],
                    "origin_declared": manifest["origin"],
                    "acquisition_mode": "offline_reprocess",
                    "quality_clock": audit["timing"],
                    "external_load_provenance": report["load_provenance"],
                    "manifest_sha256": _hash(manifest_path),
                },
            )
        )
        if not resume and baseline is not None and shadow.fatigue is not None:
            initial = next(
                (
                    op["state"]
                    for op in baseline.operations or []
                    if op["kind"] == "initial_fatigue"
                ),
                None,
            )
            if initial is not None:
                shadow.fatigue.load_state_dict(initial)
        live = LiveShadowModel(shadow)
        setup.ik.reset()
        acquisition = TrackedSession(
            live,
            setup.ik,
            registration=setup.registration,
            config=setup.acquisition,
            clock=(lambda: observed[0])
            if audit["timing"] == "recorded_receive"
            else None,
            external_loads=external_loads if loads is not None else None,
        )
        if resume:
            live.reset_tracking()

    def stop():
        if acquisition is not None:
            acquisition.close()
        if live is not None:
            live.close()
        elif shadow is not None:
            shadow.close()

    try:
        start()
        for index, frame in enumerate(read_frames(files["frames"])):
            observed[0] = frame.receive_unix if frame.receive_unix is not None else 0
            for control in manifest.get("controls", []):
                if control["before_row"] != index:
                    continue
                if control["kind"] == "resume":
                    stop()
                    start(resume=True)
                else:
                    registration = Registration(**_json(files[control["file"]]))
                    setup = dataclasses.replace(setup, registration=registration)
                    acquisition.set_registration(registration)
            acquisition.submit(frame)
            live.drain()
        stop()
        actual = read_session(out_dir / "session")
        measurement_session, measurement_path = actual, out_dir / "session"
        decisions = [
            json.loads(line)
            for line in (out_dir / "session" / "tracking_quality.jsonl")
            .read_text()
            .splitlines()
        ]
        checks = _checks(manifest, decisions, actual.events, audit["raw_fields"])
        if "baseline_quality" in files:
            expected = [
                json.loads(line)
                for line in files["baseline_quality"].read_text().splitlines()
            ]
            keys = (
                "source_id",
                "person_id",
                "sequence",
                "time_s",
                "accepted",
                "reason",
                "flags",
            )
            checks.append(
                {
                    "event": "quality_baseline",
                    "passed": [{k: r[k] for k in keys} for r in decisions]
                    == [{k: r[k] for k in keys} for r in expected],
                }
            )
        comparisons = {}
        if baseline is not None:
            tolerance = manifest.get("tolerances", {})
            comparisons["config"] = {
                "passed": actual.record["config"] == baseline.record["config"]
            }
            comparisons["tracking_resets"] = _compare(
                [
                    op["input"]
                    for op in actual.operations or []
                    if op["kind"] == "reset"
                ],
                [
                    op["input"]
                    for op in baseline.operations or []
                    if op["kind"] == "reset"
                ],
                0,
            )
            for name, a, b, default in (
                ("input_time", actual.input_time, baseline.input_time, 0),
                ("input_qpos", actual.input_qpos, baseline.input_qpos, 1e-5),
                ("input_qfrc", actual.input_qfrc, baseline.input_qfrc, 1e-6),
                (
                    "input_flags",
                    actual.input_columns["flags"],
                    baseline.input_columns["flags"],
                    0,
                ),
            ):
                comparisons[name] = _compare(a, b, tolerance.get(name, default))
            if comparisons["input_time"]["passed"]:
                replay = dataclasses.replace(
                    baseline,
                    input_qpos=actual.input_qpos,
                    input_qfrc=actual.input_qfrc,
                    input_columns=actual.input_columns,
                )
                with ShadowModel(
                    setup.model,
                    setup.shadow_config,
                    log_dir=out_dir / "scheduled",
                    calibration=actual.record.get("calibration", {}),
                    metadata=actual.record.get("metadata", {})
                    | {"optimizer_schedule": "frozen_baseline"},
                ) as scheduled:
                    list(replay.replay(scheduled))
                evaluated = read_session(out_dir / "scheduled")
                measurement_session, measurement_path = evaluated, out_dir / "scheduled"
                for name, a, b, default in (
                    ("time", evaluated.time, baseline.time, 0),
                    ("flags", evaluated.flags, baseline.flags, 0),
                    ("activation", evaluated.activation, baseline.activation, 1e-6),
                    ("fatigue", evaluated.fatigue, baseline.fatigue, 1e-8),
                ):
                    comparisons[name] = _compare(a, b, tolerance.get(name, default))
                for name in baseline.columns:
                    if name.startswith(("tau_", "reserve_")):
                        kind = "torque" if name.startswith("tau_") else "reserve"
                        comparisons[name] = _compare(
                            evaluated.columns.get(name),
                            baseline.columns[name],
                            tolerance.get(kind, 1e-6),
                        )
                for body, force in baseline.contact_force.items():
                    comparisons[f"contact:{body}"] = _compare(
                        evaluated.contact_force.get(body),
                        force,
                        tolerance.get("contact", 1e-6),
                    )
        references = []
        for spec in manifest.get("references", []):
            if spec["kind"] == "mvc_probes":
                references.append(
                    validate_probes(measurement_path, files[spec["file"]])
                )
                continue
            reference = load_reference(
                files[spec["file"]],
                spec,
                expected_sha256=manifest["files"][spec["file"]]["sha256"],
            )
            if reference.kind == "emg":
                envelope, (times, frequency) = process_emg(
                    reference,
                    sampling_hz=spec["sampling_hz"],
                    max_gap_s=spec["max_gap_s"],
                )
                np.savez(
                    out_dir / f"{spec['file']}_emg.npz",
                    time=envelope.time_s,
                    envelope=envelope.values,
                    frequency_time=times,
                    median_frequency=frequency,
                )
                references.append(
                    {
                        "file": spec["file"],
                        "role": reference.role,
                        "kind": "emg_proxy",
                        "provenance": reference.provenance,
                        "finite_envelope_samples": int(
                            np.isfinite(envelope.values).sum()
                        ),
                        "amplitude_agreement": (
                            "not interpreted as force or model activation measurement"
                        ),
                    }
                )
            elif "comparison" in spec:
                references.append(
                    compare_reference(
                        reference,
                        measurement_session,
                        max_gap_s=spec["max_gap_s"],
                        **spec["comparison"],
                    )
                    if reference.role == "validation"
                    else {"file": spec["file"], "exclusion": "used_for_calibration"}
                )
            else:
                references.append(
                    {
                        "file": spec["file"],
                        "role": reference.role,
                        "provenance": reference.provenance,
                        "comparison": "not requested",
                    }
                )
        # An active writer must not silently change a supposedly frozen input
        # between the initial audit and its use in this run.
        if _hash(manifest_path) != report["manifest_sha256"]:
            raise ValueError("recording manifest changed during reprocessing")
        audit_recording(manifest_path)
        report.update(
            status="complete",
            checks=checks,
            comparisons=comparisons,
            references=references,
            passed=all(row["passed"] for row in checks + list(comparisons.values()))
            if checks or comparisons
            else None,
            scientific_validation="not established by software regression",
        )
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        stop()
        raise
    finally:
        (out_dir / "regression.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
    return report
