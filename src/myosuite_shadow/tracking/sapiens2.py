"""Read official Sapiens2 308-keypoint exports and reconstruct metric poses."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from myosuite.biomechanics.camera import PinholeCamera, lift_depth, triangulate_point

from .frames import TrackerFrame, _finite, _vectors

SAPIENS2_REVISION = "7e5bae88456ac418ff0e58e74106c9fe192055d4"
_BODY = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
)


def _hand_names(side: str) -> tuple[str, ...]:
    names = []
    for finger in (
        "thumb",
        "forefinger",
        "middle_finger",
        "ring_finger",
        "pinky_finger",
    ):
        names.extend(
            f"{side}_{finger}{suffix}" for suffix in ("4", "3", "2", "_third_joint")
        )
    return (*names, f"{side}_wrist")


SAPIENS2_JOINT_NAMES = (
    _BODY
    + _hand_names("right")
    + _hand_names("left")
    + (
        "left_olecranon",
        "right_olecranon",
        "left_cubital_fossa",
        "right_cubital_fossa",
        "left_acromion",
        "right_acromion",
        "neck",
    )
    + tuple(f"sapiens2_face_{index}" for index in range(70, 308))
)


@dataclass(frozen=True)
class Pose2D:
    """One explicitly identified person in an undistorted calibrated image.

    This is deliberately not a TrackerFrame: pixels cannot enter metric 3D IK.
    Image names must have measured timestamps and stable identity assignments.
    """

    time_s: float
    camera_id: str
    person_id: str
    sequence: int
    positions_px: dict[str, np.ndarray]
    confidence: dict[str, float]
    capture_unix: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_s", _finite(self.time_s, "time_s"))
        if not self.camera_id or not self.person_id:
            raise ValueError("camera_id and stable person_id are required")
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("sequence must be a nonnegative integer")
        points = _vectors(self.positions_px, 2, "positions_px")
        if set(self.confidence) != set(points):
            raise ValueError("each 2D point must have a confidence")
        scores = {n: _finite(c, "confidence") for n, c in self.confidence.items()}
        if any(not 0 <= c <= 1 for c in scores.values()):
            raise ValueError("confidence must be within [0, 1]")
        object.__setattr__(self, "positions_px", points)
        object.__setattr__(self, "confidence", scores)
        if self.capture_unix is not None:
            object.__setattr__(
                self, "capture_unix", _finite(self.capture_unix, "capture_unix")
            )

    def record(self) -> dict[str, Any]:
        """Return the observation and its original export record for auditing."""
        return {
            "time_s": self.time_s,
            "camera_id": self.camera_id,
            "person_id": self.person_id,
            "sequence": self.sequence,
            "positions_px": {n: p.tolist() for n, p in self.positions_px.items()},
            "confidence": self.confidence,
            "capture_unix": self.capture_unix,
            "metadata": self.metadata,
        }


def read_sapiens2(
    path: str | Path,
    *,
    camera_id: str,
    person_id: str,
    timestamps: Mapping[str, float],
    instance_indices: Mapping[str, int | None],
    capture_unix: Mapping[str, float] | None = None,
) -> Iterator[Pose2D]:
    """Read the official aggregate ``*_predictions.json`` format.

    ``timestamps`` maps each image name to the common session time. The
    reviewed ``instance_indices`` sidecar assigns that image's detection to
    ``person_id``; None explicitly marks an absent person. Detection order is
    not a stable identity, and inference can skip images, so neither index nor
    frame rate is silently used as a timestamp or identity.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("num_keypoints") != 308 or not isinstance(
        payload.get("frames"), list
    ):
        raise ValueError("expected official Sapiens2 308-keypoint aggregate export")
    image_size = np.asarray(payload.get("image_size"))
    if (
        image_size.shape != (2,)
        or not np.isfinite(image_size).all()
        or np.any(image_size <= 0)
    ):
        raise ValueError("image_size must be [height, width]")
    seen = set()
    for sequence, record in enumerate(payload["frames"]):
        image = record["image_name"]
        if image in seen:
            raise ValueError("duplicate Sapiens2 image_name")
        seen.add(image)
        if image not in timestamps or image not in instance_indices:
            raise ValueError(
                f"{image} needs an explicit timestamp and person assignment"
            )
        index = instance_indices[image]
        points, scores = {}, {}
        raw = None
        if index is not None:
            instances = record["instances"]
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(instances)
            ):
                raise ValueError(f"invalid identity assignment for {image}")
            raw = instances[index]
            if raw.get("person_id", person_id) != person_id:
                raise ValueError("Sapiens2 person identity changed")
            uv = np.asarray(raw["keypoints"], dtype=float)
            score = np.asarray(raw["keypoint_scores"], dtype=float)
            if (
                uv.shape != (308, 2)
                or score.shape != (308,)
                or not np.isfinite(uv).all()
                or not np.isfinite(score).all()
            ):
                raise ValueError("Sapiens2 keypoints/scores must be finite 308x2/308")
            for i, name in enumerate(SAPIENS2_JOINT_NAMES):
                if 0 <= uv[i, 0] < image_size[1] and 0 <= uv[i, 1] < image_size[0]:
                    points[name] = uv[i]
                    scores[name] = float(np.clip(score[i], 0, 1))
        yield Pose2D(
            timestamps[image],
            camera_id,
            person_id,
            sequence,
            points,
            scores,
            None if capture_unix is None else capture_unix[image],
            {
                "image_name": image,
                "image_size": image_size.tolist(),
                "instance_index": index,
                "raw_instance": raw,
                "sapiens2_revision": SAPIENS2_REVISION,
            },
        )


def reconstruct_multiview(
    views: Sequence[Pose2D],
    cameras: Mapping[str, PinholeCamera],
    *,
    source_id: str = "sapiens2",
    coordinate_frame: str = "camera_world",
    min_confidence: float = 0.5,
    max_skew_s: float = 0.01,
    max_reprojection_px: float = 5.0,
    min_ray_angle_deg: float = 1.0,
) -> TrackerFrame:
    """Reconstruct synchronized views of one assigned person using calibration.

    Individual failed landmarks are omitted with reasons retained in metadata.
    Camera synchronization, identity and metric calibration are mandatory.
    """
    if len(views) < 2 or len({v.camera_id for v in views}) != len(views):
        raise ValueError("need at least two distinct calibrated camera views")
    if len({v.person_id for v in views}) != 1:
        raise ValueError("camera views must identify the same person")
    if not 0 <= min_confidence <= 1 or not np.isfinite(max_skew_s) or max_skew_s < 0:
        raise ValueError("invalid multiview quality thresholds")
    if np.ptp([v.time_s for v in views]) > max_skew_s:
        raise ValueError("camera observations are not synchronized")
    if any(v.camera_id not in cameras for v in views):
        raise ValueError("every view needs a measured camera calibration")
    names = set().union(*(v.positions_px for v in views))
    positions, scores, errors, rejected = {}, {}, {}, {}
    for name in sorted(names):
        usable = [
            v
            for v in views
            if name in v.positions_px
            and v.confidence[name] >= min_confidence
            and v.confidence[name] > 0
        ]
        if len(usable) < 2:
            rejected[name] = "fewer than two confident observations"
            continue
        try:
            result = triangulate_point(
                [cameras[v.camera_id] for v in usable],
                np.array([v.positions_px[name] for v in usable]),
                weights=np.array([v.confidence[name] for v in usable]),
                max_reprojection_px=max_reprojection_px,
                min_ray_angle_deg=min_ray_angle_deg,
            )
        except ValueError as error:
            rejected[name] = str(error)
            continue
        positions[name] = result.position
        scores[name] = min(v.confidence[name] for v in usable)
        errors[name] = result.reprojection_errors.tolist()
    captures = [v.capture_unix for v in views]
    captured = None
    if all(t is not None for t in captures):
        if np.ptp(captures) > max_skew_s:
            raise ValueError("camera capture clocks disagree")
        captured = float(np.mean(captures))
    return TrackerFrame(
        float(np.mean([v.time_s for v in views])),
        source_id,
        max(v.sequence for v in views),
        views[0].person_id,
        positions,
        scores,
        position_tracked={n: True for n in positions},
        time_kind="capture",
        capture_unix=captured,
        coordinate_frame=coordinate_frame,
        metadata={
            "method": "calibrated_multiview",
            "views": [v.record() for v in views],
            "calibrations": {v.camera_id: cameras[v.camera_id].record() for v in views},
            "reprojection_px": errors,
            "reconstruction_rejections": rejected,
        },
    )


def reconstruct_depth(
    view: Pose2D,
    camera: PinholeCamera,
    depths_m: Mapping[str, float],
    *,
    source_id: str = "sapiens2",
    coordinate_frame: str = "camera_world",
    min_confidence: float = 0.5,
) -> TrackerFrame:
    """Lift landmarks with aligned, calibrated metric camera-Z depths.

    Relative monocular depth and uncalibrated pointmaps are not metric depth.
    Invalid/missing depth samples are omitted and retained as rejection reasons.
    """
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be within [0, 1]")
    positions, scores, rejected = {}, {}, {}
    for name, pixel in view.positions_px.items():
        if view.confidence[name] < min_confidence or name not in depths_m:
            rejected[name] = "missing depth or low confidence"
            continue
        try:
            positions[name] = lift_depth(camera, pixel, depths_m[name])
        except ValueError as error:
            rejected[name] = str(error)
            continue
        scores[name] = view.confidence[name]
    return TrackerFrame(
        view.time_s,
        source_id,
        view.sequence,
        view.person_id,
        positions,
        scores,
        position_tracked={n: True for n in positions},
        time_kind="capture",
        capture_unix=view.capture_unix,
        coordinate_frame=coordinate_frame,
        metadata={
            "method": "calibrated_metric_z_depth",
            "view": view.record(),
            "calibration": camera.record(),
            "depths_m": {
                n: float(d) if np.isfinite(d) else None for n, d in depths_m.items()
            },
            "reconstruction_rejections": rejected,
        },
    )
