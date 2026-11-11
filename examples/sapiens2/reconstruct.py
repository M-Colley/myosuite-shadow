"""Convert calibrated official Sapiens2 exports into canonical metric JSONL."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import platform
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from myosuite.biomechanics.camera import PinholeCamera

from myosuite_shadow.log import package_provenance
from myosuite_shadow.tracking import (
    Pose2D,
    RawFrameWriter,
    TrackerFrame,
    read_sapiens2,
    reconstruct_multiview,
)
from myosuite_shadow.tracking.sapiens2 import SAPIENS2_REVISION


def _hash(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": size}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _provenance(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("inference", {})
    if not isinstance(value, dict):
        raise ValueError("inference provenance must be an object")
    checkpoint = value.get("checkpoint", {})
    if not isinstance(checkpoint, dict):
        raise ValueError("inference.checkpoint must be an object")
    for key in ("upstream_revision", "model"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"inference.{key} is required from the inference operator")
    for key in ("id", "revision", "sha256"):
        if not isinstance(checkpoint.get(key), str) or not checkpoint[key].strip():
            raise ValueError(f"inference.checkpoint.{key} is required")
    if re.fullmatch(r"[a-fA-F0-9]{64}", checkpoint["sha256"]) is None:
        raise ValueError("inference.checkpoint.sha256 must be the actual 64-digit hash")
    return value


def _views(
    iterator: Iterator[Pose2D],
    camera_id: str,
    provenance: dict[str, Any],
) -> Iterator[Pose2D]:
    previous = None
    for view in iterator:
        if previous is not None and view.time_s <= previous:
            raise ValueError(f"camera {camera_id} timestamps must strictly increase")
        previous = view.time_s
        metadata = dict(view.metadata)
        metadata.pop("sapiens2_revision", None)
        metadata["parser_reference_revision"] = SAPIENS2_REVISION
        metadata["inference_provenance"] = provenance
        yield dataclasses.replace(view, metadata=metadata)


def _groups(
    streams: dict[str, Iterator[Pose2D]],
    max_skew_s: float,
) -> Iterator[list[Pose2D]]:
    heads = {name: next(stream, None) for name, stream in streams.items()}
    while any(view is not None for view in heads.values()):
        anchor = min(view.time_s for view in heads.values() if view is not None)
        group = []
        for name in sorted(heads):
            view = heads[name]
            if view is not None and view.time_s - anchor <= max_skew_s:
                group.append(view)
                heads[name] = next(streams[name], None)
        yield group


def reconstruct(config_path: Path, out_dir: Path) -> dict[str, Any]:
    """Convert once without overwriting outputs; retain failed-run evidence too."""
    config_path = config_path.resolve()
    config = _json(config_path)
    if not isinstance(config, dict):
        raise ValueError("reconstruction config must be an object")
    provenance = _provenance(config)
    for key in ("person_id", "source_id"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(
                f"{key} must identify the participant and acquisition source"
            )
    camera_configs = config.get("cameras", [])
    if not isinstance(camera_configs, list) or len(camera_configs) < 2:
        raise ValueError("at least two calibrated cameras are required")
    options = dict(config.get("reconstruction", {}))
    allowed = {
        "max_skew_s",
        "min_confidence",
        "max_reprojection_px",
        "min_ray_angle_deg",
    }
    if set(options) - allowed:
        raise ValueError("unknown reconstruction options")
    max_skew = float(options.setdefault("max_skew_s", 0.01))
    if not math.isfinite(max_skew) or max_skew < 0:
        raise ValueError("max_skew_s must be finite and nonnegative")
    for name, default, lo, hi in (
        ("min_confidence", 0.5, 0, 1),
        ("min_ray_angle_deg", 1.0, 0, 90),
        ("max_reprojection_px", 5.0, 0, math.inf),
    ):
        value = float(options.setdefault(name, default))
        if (
            not math.isfinite(value)
            or value < lo
            or value > hi
            or (name != "min_confidence" and value == lo)
            or (name == "min_ray_angle_deg" and value == hi)
        ):
            raise ValueError(f"invalid reconstruction {name}")
        options[name] = value
    cameras, streams = {}, {}
    sources = [_hash(config_path)]
    for camera_config in camera_configs:
        name = camera_config["id"]
        if not isinstance(name, str) or not name or name in cameras:
            raise ValueError("camera IDs must be nonempty and unique")
        if camera_config.get("person_id", config["person_id"]) != config["person_id"]:
            raise ValueError("every camera must assign the same stable person_id")
        if camera_config.get("images_undistorted") is not True:
            raise ValueError(f"camera {name} must confirm images_undistorted=true")
        cameras[name] = PinholeCamera(**camera_config["calibration"])
        paths = {}
        for key in ("predictions", "timestamps", "instances", "capture_unix"):
            if key == "capture_unix" and key not in camera_config:
                continue
            path = Path(camera_config[key])
            paths[key] = path if path.is_absolute() else config_path.parent / path
            sources.append({**_hash(paths[key]), "camera_id": name, "role": key})
        stream = read_sapiens2(
            paths["predictions"],
            camera_id=name,
            person_id=config["person_id"],
            timestamps=_json(paths["timestamps"]),
            instance_indices=_json(paths["instances"]),
            capture_unix=_json(paths["capture_unix"])
            if "capture_unix" in paths
            else None,
        )
        streams[name] = _views(stream, name, provenance)
    out_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "config": config,
        "input_files": sources,
        "script": _hash(Path(__file__)),
        "inference_provenance": provenance,
        "inference_provenance_origin": "caller",
        "parser_reference_revision": SAPIENS2_REVISION,
        "packages": package_provenance(),
        "python": sys.version,
        "platform": platform.platform(),
        "calibrations": {name: camera.record() for name, camera in cameras.items()},
        "grouping": "oldest unconsumed time; at most one view/camera within max_skew_s",
        "frames": 0,
        "empty_frames": 0,
    }
    manifest_path = out_dir / "manifest.json"

    def save_manifest() -> None:
        temporary = out_dir / "manifest.tmp"
        temporary.write_text(
            json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
        )
        temporary.replace(manifest_path)

    save_manifest()
    try:
        with (
            RawFrameWriter(out_dir / "frames.jsonl") as writer,
            (out_dir / "quality.jsonl").open(
                "x", encoding="utf-8", newline="\n"
            ) as quality,
        ):
            for sequence, group in enumerate(_groups(streams, max_skew)):
                anchor = min(group, key=lambda view: (view.time_s, view.camera_id))
                reason = None
                if sum(bool(view.positions_px) for view in group) < 2:
                    reason = "fewer than two views with assigned participant landmarks"
                    frame = TrackerFrame(
                        anchor.time_s,
                        config["source_id"],
                        sequence,
                        config["person_id"],
                        {},
                        coordinate_frame=config.get("coordinate_frame", "camera_world"),
                        time_kind="capture",
                        capture_unix=anchor.capture_unix,
                        metadata={
                            "method": "calibrated_multiview",
                            "views": [view.record() for view in group],
                        },
                    )
                else:
                    try:
                        frame = reconstruct_multiview(
                            group,
                            cameras,
                            source_id=config["source_id"],
                            coordinate_frame=config.get(
                                "coordinate_frame", "camera_world"
                            ),
                            **options,
                        )
                    except ValueError as error:
                        reason = str(error)
                        frame = TrackerFrame(
                            anchor.time_s,
                            config["source_id"],
                            sequence,
                            config["person_id"],
                            {},
                            coordinate_frame=config.get(
                                "coordinate_frame", "camera_world"
                            ),
                            time_kind="capture",
                            capture_unix=anchor.capture_unix,
                            metadata={"views": [view.record() for view in group]},
                        )
                missing = sorted(set(cameras) - {view.camera_id for view in group})
                frame = dataclasses.replace(
                    frame,
                    sequence=sequence,
                    time_s=anchor.time_s,
                    capture_unix=anchor.capture_unix
                    if frame.capture_unix is not None
                    else None,
                    metadata={
                        **frame.metadata,
                        "inference_provenance": provenance,
                        "group_anchor_camera": anchor.camera_id,
                        "group_time_policy": "oldest_observation",
                        "missing_cameras": missing,
                        "empty_reason": reason,
                    },
                )
                writer.write(frame)
                quality.write(
                    json.dumps(
                        {
                            "sequence": sequence,
                            "time_s": frame.time_s,
                            "images": {
                                v.camera_id: v.metadata["image_name"] for v in group
                            },
                            "missing_cameras": missing,
                            "position_count": len(frame.positions),
                            "empty_reason": reason,
                            "landmark_rejections": frame.metadata.get(
                                "reconstruction_rejections", {}
                            ),
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                quality.flush()
                manifest["frames"] += 1
                manifest["empty_frames"] += not bool(frame.positions)
        for source in sources:
            if _hash(Path(source["path"]))["sha256"] != source["sha256"]:
                raise ValueError("an input changed during reconstruction")
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["outputs"] = [
            _hash(out_dir / name)
            for name in ("frames.jsonl", "quality.jsonl")
            if (out_dir / name).exists()
        ]
        save_manifest()
    return manifest


def main(argv: list[str] | None = None) -> None:
    """Read explicit camera/identity/provenance configuration and write new outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = reconstruct(args.config, args.out_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"reconstruction failed: {error}\n")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "frames": manifest["frames"],
                "empty_frames": manifest["empty_frames"],
                "frames_file": str((args.out_dir / "frames.jsonl").resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
