"""Read-only protocol and landmark-observability checks before acquisition."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from myosuite_shadow.clock import now
from myosuite_shadow.log import model_sha256, package_provenance
from myosuite_shadow.pipeline import ShadowModel
from myosuite_shadow.tracking import ClockMapping, synchronize_quest_clock
from myosuite_shadow.tracking.quest import QUEST_FULL_JOINTS, QUEST_UPPER_JOINTS
from myosuite_shadow.tracking.quest_clock import (
    validate_clock_refresh,
    validate_clock_sync_options,
)
from myosuite_shadow.tracking_config import load_tracking_setup


def _scope(options: dict, prefix: str, dofs: tuple) -> list[int] | None:
    indices, joints = options.get(f"{prefix}_dofs"), options.get(f"{prefix}_joints")
    if indices is not None and joints is not None:
        raise ValueError(f"choose {prefix}_dofs or {prefix}_joints")
    if indices is None and joints is None:
        return None
    selected = []
    values = joints if joints is not None else indices
    if not isinstance(values, list):
        raise ValueError(f"{prefix} scope must be a list")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | str):
            raise ValueError(f"{prefix} scope requires DOF indices or names")
        matches = [
            d.index
            for d in dofs
            if (joints is not None and value == d.joint)
            or (
                joints is None
                and not isinstance(value, bool)
                and (value == d.index or value == d.name)
            )
        ]
        if not matches or set(matches).intersection(selected):
            raise ValueError(f"unknown or repeated independent {prefix} DOF: {value}")
        selected.extend(matches)
    return selected


def _landmarks(model: mujoco.MjModel, keypoints: tuple) -> tuple:
    from myosuite.biomechanics.retarget import Landmark

    result = []
    for point in keypoints:
        if point.frame_type == "body":
            model.body(point.frame)
            result.append(Landmark(point.name, point.frame))
        elif point.frame_type == "site":
            site = model.site(point.frame)
            result.append(
                Landmark(
                    point.name,
                    model.body(int(model.site_bodyid[site.id])).name,
                    site=point.frame,
                )
            )
        elif point.frame_type == "geom":
            geom = model.geom(point.frame)
            result.append(
                Landmark(
                    point.name,
                    model.body(int(model.geom_bodyid[geom.id])).name,
                    pos=tuple(model.geom_pos[geom.id]),
                )
            )
        else:
            raise ValueError(f"unknown keypoint frame type: {point.frame_type}")
    return tuple(result)


def preflight(
    config_path: str | Path,
    *,
    live: bool = False,
    probe_clock: bool = False,
    include_matrices: bool = False,
) -> dict[str, Any]:
    """Check a model/configuration without opening a tracker or session writer.

    Optional clock probing sends the existing bounded timestamp exchanges and
    records the result in this report; it does not update the configuration.
    Observability is local to the supplied postures and declared DOF scales.
    """
    from myosuite.biomechanics.observability import (
        independent_dofs,
        landmark_observability,
    )
    from myosuite.physics.joint_coupling import (
        JointCouplings,
        project_joint_couplings,
    )

    setup = load_tracking_setup(config_path, offline=not live)
    settings, model = setup.settings, setup.model
    with ShadowModel(model, setup.shadow_config) as pipeline:
        resolved_shadow = dataclasses.asdict(pipeline.config)
    options = settings.get("preflight", {})
    if not isinstance(options, dict):
        raise ValueError("preflight settings must be an object")
    allowed = {
        "target_dofs",
        "target_joints",
        "fixed_dofs",
        "fixed_joints",
        "dof_scales",
        "translation_scale_m",
        "rotation_scale_rad",
        "landmark_scale_m",
        "rtol",
        "atol",
        "postures",
        "max_condition_number",
        "require_full_rank",
        "required_calibrations",
    }
    if set(options) - allowed:
        raise ValueError(
            f"unknown preflight settings: {sorted(set(options) - allowed)}"
        )
    checks = []

    def check(code: str, level: str, message: str) -> None:
        checks.append({"code": code, "level": level, "message": message})

    dofs = tuple(independent_dofs(model))
    target = _scope(options, "target", dofs)
    fixed = _scope(options, "fixed", dofs) or []
    if target is None:
        check(
            "scope_unspecified",
            "warning",
            "All independent DOFs are diagnostic; declare the study target scope.",
        )
    strict = options.get("require_full_rank", True)
    if not isinstance(strict, bool):
        raise ValueError("require_full_rank must be boolean")
    condition_limit = float(options.get("max_condition_number", 1e6))
    if not np.isfinite(condition_limit) or condition_limit < 1:
        raise ValueError("max_condition_number must be finite and at least one")
    scales = options.get("dof_scales")
    if scales is not None:
        if not isinstance(scales, dict):
            raise ValueError("dof_scales must map independent DOF indices to scales")
        scales = {int(index): float(value) for index, value in scales.items()}
    geometry_options = {
        key: options[key]
        for key in (
            "translation_scale_m",
            "rotation_scale_rad",
            "landmark_scale_m",
            "rtol",
            "atol",
        )
        if key in options
    }
    postures = options.get("postures")
    if postures is None:
        postures = [setup.ik.configuration.q.tolist()]
        check(
            "postures_unspecified",
            "warning",
            "Only the initial posture is checked; add measured task postures.",
        )
    if not isinstance(postures, list) or not postures:
        raise ValueError("preflight.postures must be a nonempty list of qpos vectors")
    required = set(setup.ik.config.required_keypoints) or {
        point.name for point in setup.ik.keypoints
    }
    quest = settings.get("quest")
    if live:
        if not isinstance(quest, dict):
            check(
                "live_source_missing",
                "error",
                "Live acquisition requires Quest settings.",
            )
            quest = None
        else:
            joint_set = quest.get("joint_set")
            if joint_set not in (None, "upper_body", "full_body"):
                raise ValueError("quest.joint_set must be upper_body or full_body")
            supported = (
                QUEST_UPPER_JOINTS if joint_set == "upper_body" else QUEST_FULL_JOINTS
            )
            unavailable = sorted(required - set(supported))
            if unavailable:
                check(
                    "quest_landmarks",
                    "error",
                    "Required landmarks are not in the declared Quest layout: "
                    f"{unavailable}",
                )
            if joint_set is None:
                check(
                    "quest_layout_unspecified",
                    "warning",
                    "Declare quest.joint_set and verify actual device capabilities.",
                )
            if (
                setup.registration is not None
                and setup.registration.source_frame
                not in {"quest_stage", "quest_local", "quest_local_floor"}
            ):
                check(
                    "quest_registration_source",
                    "error",
                    "Quest registration must use quest_stage, quest_local "
                    "or quest_local_floor.",
                )
            if any(
                not isinstance(quest.get(k), str) or not quest[k]
                for k in ("source_id", "person_id")
            ):
                check(
                    "quest_identity_unspecified",
                    "warning",
                    "Declare the expected Quest source_id and person_id.",
                )
            sync_options = quest.get("clock_sync", {})
            if (
                not isinstance(sync_options, dict)
                or sync_options.get("refresh_s", 20) is None
            ):
                raise ValueError(
                    "clock_sync must be an object with a positive refresh_s"
                )
            validate_clock_sync_options(**sync_options)
    landmarks = _landmarks(
        model, tuple(point for point in setup.ik.keypoints if point.name in required)
    )
    if setup.ik.config.min_keypoints > len(setup.ik.keypoints):
        check(
            "too_few_mapped_landmarks",
            "error",
            "The mapping cannot satisfy IK min_keypoints.",
        )
    geometry = []
    couplings = JointCouplings.from_model(model)
    for index, posture in enumerate(postures):
        qpos = np.array(posture, dtype=float)
        if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
            raise ValueError(f"posture {index} must have {model.nq} finite positions")
        project_joint_couplings(couplings, qpos)
        result = landmark_observability(
            model,
            qpos,
            landmarks,
            target_dofs=target,
            fixed_dofs=fixed,
            dof_scales=scales,
            **geometry_options,
        )
        record = result.record(include_matrices=include_matrices)
        mujoco.mj_normalizeQuat(model, qpos)
        geometry.append(
            {
                "posture_index": index,
                "requested_qpos": posture,
                "qpos": qpos.tolist(),
                **record,
            }
        )
        if result.rank < len(result.target_dofs):
            level = "error" if strict and target is not None else "warning"
            check(
                "unobservable_target",
                level,
                f"Posture {index}: rank {result.rank}/{len(result.target_dofs)}; "
                "landmarks do not distinguish all target motion from nuisance motion.",
            )
        elif not result.target_dofs:
            check(
                "empty_target_scope", "warning", "No independent target DOFs selected."
            )
        elif result.condition_number > condition_limit:
            check(
                "poor_conditioning",
                "warning",
                f"Posture {index}: scaled condition exceeds {condition_limit:g}.",
            )

    registration = setup.registration
    if registration is None:
        check(
            "registration_missing",
            "error" if live and "quest" in settings else "warning",
            "Provide measured registration or verify the source is model_world.",
        )
    elif registration.target_frame != "model_world":
        check("registration_target", "error", "Registration must target model_world.")
    if setup.acquisition.load_site is not None:
        model.site(setup.acquisition.load_site)
    if setup.acquisition.held_mass_kg and not np.linalg.norm(model.opt.gravity):
        check(
            "held_mass_zero_gravity",
            "warning",
            "The configured held mass produces zero force under model gravity.",
        )

    clock_record = None
    if live and quest is not None:
        mapping = ClockMapping(**quest["clock"])
        if not isinstance(mapping.clock_id, str):
            raise ValueError("Quest clock_id must be a nonempty string")
        if probe_clock:
            if not quest.get("clock_host"):
                raise ValueError("clock probing requires quest.clock_host")
            sync = dict(quest.get("clock_sync", {}))
            sync.pop("refresh_s", None)
            sync.setdefault(
                "max_uncertainty_s", setup.acquisition.max_clock_uncertainty_s
            )
            measured = synchronize_quest_clock(quest["clock_host"], **sync)
            validate_clock_refresh(mapping, measured)
            mapping = measured
        if mapping.uncertainty_s > setup.acquisition.max_clock_uncertainty_s:
            check(
                "clock_uncertainty", "error", "Clock uncertainty exceeds study policy."
            )
        if mapping.unix_anchor is None:
            check("clock_unmapped", "error", "The Quest clock needs a Unix reference.")
        else:
            source_now = (
                mapping.source_anchor_s + (now() - mapping.unix_anchor) / mapping.rate
            )
            try:
                mapping.map(source_now)
            except ValueError as error:
                check("clock_validity", "error", str(error))
        if mapping.valid_until_source_s is None:
            check("clock_expiry_unspecified", "warning", "Clock mapping has no expiry.")
        clock_record = mapping.record()
    elif probe_clock:
        raise ValueError("clock probing requires live Quest configuration")

    calibration = settings.get("calibration", {})
    required_calibrations = options.get("required_calibrations", [])
    if not isinstance(calibration, dict) or not isinstance(required_calibrations, list):
        raise ValueError(
            "calibration must be an object and required_calibrations a list"
        )
    if any(not isinstance(name, str) or not name for name in required_calibrations):
        raise ValueError("required_calibrations must contain nonempty names")
    for name in required_calibrations:
        if not calibration.get(name):
            check(
                "calibration_missing", "error", f"Required calibration missing: {name}"
            )
    if not required_calibrations:
        check(
            "calibration_policy_unspecified",
            "warning",
            "Declare the calibrations required by the study protocol.",
        )
    errors = sum(item["level"] == "error" for item in checks)
    warnings = sum(item["level"] == "warning" for item in checks)
    return {
        "schema_version": 1,
        "status": "blocked" if errors else "needs_attention" if warnings else "passed",
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "mode": "live_configuration" if live else "offline_configuration",
        "source_probe": "clock_only" if probe_clock else "not_requested",
        "config": str(setup.config_path),
        "config_sha256": hashlib.sha256(setup.config_path.read_bytes()).hexdigest(),
        "model": str(setup.model_path),
        "model_sha256": model_sha256(model),
        "packages": package_provenance(),
        "required_landmarks": sorted(required),
        "observability": geometry,
        "registration": None if registration is None else registration.record(),
        "clock_mapping": clock_record,
        "acquisition": dataclasses.asdict(setup.acquisition),
        "shadow_config": resolved_shadow,
        "declared_calibrations": sorted(calibration),
        "interpretation": (
            "Local geometry/protocol checks; participant accuracy needs "
            "independent measurements."
        ),
    }


def main(argv: list[str] | None = None) -> None:
    """Write a new JSON preflight report; blocked configurations exit nonzero."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--probe-clock", action="store_true")
    parser.add_argument("--include-matrices", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = preflight(
            args.config,
            live=args.live,
            probe_clock=args.probe_clock,
            include_matrices=args.include_matrices,
        )
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        parser.exit(2, f"preflight failed: {error}\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
    print(
        f"{report['status']}: {report['errors']} errors, {report['warnings']} warnings"
    )
    if report["errors"]:
        parser.exit(1)
