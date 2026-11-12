"""Official Sapiens2 JSON converts into audited metric tracker input end to end."""

from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from myosuite.biomechanics.camera import PinholeCamera

from myosuite_shadow.tracking import Pose2D, read_frames

SCRIPT = (
    Path(__file__).resolve().parents[1] / "examples" / "sapiens2" / "reconstruct.py"
)


def _fixture(root: Path) -> tuple[Path, dict[str, np.ndarray]]:
    intrinsic = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    cameras = {
        "front": PinholeCamera(intrinsic, np.eye(3), np.zeros(3), "front-cal"),
        "side": PinholeCamera(intrinsic, np.eye(3), np.array([-0.8, 0, 0]), "side-cal"),
    }
    truth = {
        "left_shoulder": np.array([0.1, -0.2, 3.0]),
        "left_ankle": np.array([0.15, 0.6, 3.2]),
    }
    world = np.tile([0.0, 0, 3.0], (308, 1))
    world[5], world[13] = truth["left_shoulder"], truth["left_ankle"]
    config = {
        "person_id": "P07",
        "source_id": "sapiens2-study",
        "inference": {
            "upstream_revision": "actual-inference-revision",
            "model": "sapiens2_1b",
            "checkpoint": {
                "id": "facebook/study-checkpoint",
                "revision": "actual-weight-revision",
                "sha256": "a" * 64,
            },
        },
        "cameras": [],
        "reconstruction": {"max_skew_s": 0.01},
    }
    for name, camera in cameras.items():
        images = (
            ["f0.png", "f1.png", "f2.png"]
            if name == "front"
            else ["f0.png", "f1.png", "f3.png"]
        )
        times = [0, 1, 2] if name == "front" else [0.002, 1.002, 3]
        points = camera.project(world).tolist()
        instance = {
            "bbox": [0, 0, 640, 480],
            "keypoints": points,
            "keypoint_scores": [0.9] * 308,
        }
        payload = {
            "video": name,
            "image_size": [480, 640],
            "num_keypoints": 308,
            "frames": [
                {"image_name": image, "instances": [instance]} for image in images
            ],
        }
        (root / f"{name}-predictions.json").write_text(json.dumps(payload))
        (root / f"{name}-times.json").write_text(json.dumps(dict(zip(images, times))))
        assignments = {image: 0 for image in images}
        if name == "side":
            assignments["f1.png"] = None
        (root / f"{name}-instances.json").write_text(json.dumps(assignments))
        (root / f"{name}-capture.json").write_text(
            json.dumps({image: 1000 + t for image, t in zip(images, times)})
        )
        config["cameras"].append(
            {
                "id": name,
                "predictions": f"{name}-predictions.json",
                "timestamps": f"{name}-times.json",
                "instances": f"{name}-instances.json",
                "capture_unix": f"{name}-capture.json",
                "calibration": camera.record(),
                "images_undistorted": True,
            }
        )
    path = root / "reconstruct.json"
    path.write_text(json.dumps(config))
    return path, truth


def test_runnable_reconstruction_preserves_metric_pose_missing_views_and_provenance(
    tmp_path,
):
    config, truth = _fixture(tmp_path)
    output = tmp_path / "metric output"
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config),
            "--out-dir",
            str(output),
        ],
        text=True,
        capture_output=True,
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["status"] == "complete"
    frames = list(read_frames(output / "frames.jsonl"))
    assert [f.time_s for f in frames] == [0, 1, 2, 3]
    assert [len(f.positions) > 0 for f in frames] == [True, False, False, False]
    for name, position in truth.items():
        np.testing.assert_allclose(frames[0].positions[name], position, atol=1e-12)
    assert frames[0].capture_unix == 1000
    assert frames[0].coordinate_frame == "camera_world"
    consumed = [
        (v["camera_id"], v["metadata"]["image_name"])
        for frame in frames
        for v in frame.metadata["views"]
    ]
    assert len(consumed) == len(set(consumed)) == 6
    assert (
        frames[0].metadata["inference_provenance"]["checkpoint"]["revision"]
        == "actual-weight-revision"
    )
    for view in frames[0].metadata["views"]:
        assert "sapiens2_revision" not in view["metadata"]
        assert (
            view["metadata"]["parser_reference_revision"] != "actual-inference-revision"
        )
    quality = [
        json.loads(line) for line in (output / "quality.jsonl").read_text().splitlines()
    ]
    assert quality[1]["empty_reason"]
    assert quality[2]["missing_cameras"] == ["side"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["frames"] == 4 and manifest["empty_frames"] == 3
    assert len(manifest["input_files"]) == 9
    for item in manifest["input_files"]:
        assert (
            hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
            == item["sha256"]
        )
    assert manifest["inference_provenance_origin"] == "caller"


def test_reconstruction_requires_actual_checkpoint_provenance_and_never_overwrites(
    tmp_path,
):
    config, _ = _fixture(tmp_path)
    api = runpy.run_path(str(SCRIPT))
    values = json.loads(config.read_text())
    del values["inference"]["checkpoint"]
    config.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="checkpoint"):
        api["reconstruct"](config, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
    config, _ = _fixture(tmp_path)
    output = tmp_path / "output"
    api["reconstruct"](config, output)
    before = (output / "frames.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        api["reconstruct"](config, output)
    assert (output / "frames.jsonl").read_bytes() == before


def test_reconstruction_rejects_person_switch_and_marks_partial_run_failed(tmp_path):
    config, _ = _fixture(tmp_path)
    path = tmp_path / "side-predictions.json"
    values = json.loads(path.read_text())
    values["frames"][0]["instances"][0]["person_id"] = "P08"
    path.write_text(json.dumps(values))
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="identity"):
        runpy.run_path(str(SCRIPT))["reconstruct"](config, output)
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"


def test_grouping_consumes_fast_camera_once_and_anchors_times_monotonically():
    def view(camera, time_s, sequence):
        return Pose2D(time_s, camera, "P07", sequence, {}, {})

    groups = list(
        runpy.run_path(str(SCRIPT))["_groups"](
            {
                "A": iter([view("A", 0, 0), view("A", 0.001, 1)]),
                "B": iter([view("B", 0.009, 0)]),
            },
            0.01,
        )
    )
    assert [[v.camera_id for v in group] for group in groups] == [["A", "B"], ["A"]]
    assert [min(v.time_s for v in group) for group in groups] == [0, 0.001]
