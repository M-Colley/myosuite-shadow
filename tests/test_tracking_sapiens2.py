"""Official Sapiens2 export parsing, explicit identities and calibrated 3D input."""

import json
from pathlib import Path

import numpy as np
import pytest
from myosuite.biomechanics.camera import PinholeCamera

from myosuite_shadow.tracking import (
    SAPIENS2_JOINT_NAMES,
    Pose2D,
    TrackerFrame,
    read_sapiens2,
    reconstruct_depth,
    reconstruct_multiview,
)


def export(tmp_path: Path) -> Path:
    instance = {
        "bbox": [0, 0, 640, 480],
        "keypoints": [[320, 240]] * 308,
        "keypoint_scores": [0.9] * 308,
    }
    path = tmp_path / "video_predictions.json"
    path.write_text(
        json.dumps(
            {
                "video": "video",
                "image_size": [480, 640],
                "num_keypoints": 308,
                "kpt_thr_used": 0.3,
                "frames": [
                    {"image_name": "00012.png", "instances": [instance, instance]}
                ],
            }
        )
    )
    return path


def test_reads_actual_308_export_without_assuming_coco_or_stable_detection_index(
    tmp_path,
):
    path = export(tmp_path)
    kwargs = dict(
        camera_id="A",
        person_id="P07",
        timestamps={"00012.png": 1.25},
        instance_indices={"00012.png": 1},
    )
    result = list(read_sapiens2(path, **kwargs))[0]
    assert result.time_s == 1.25 and result.person_id == "P07"
    assert len(result.positions_px) == 308
    assert SAPIENS2_JOINT_NAMES[9] == "left_hip"
    assert SAPIENS2_JOINT_NAMES[41] == "right_wrist"
    assert SAPIENS2_JOINT_NAMES[62] == "left_wrist"
    assert result.metadata["instance_index"] == 1
    with pytest.raises(ValueError, match="timestamp and person"):
        list(read_sapiens2(path, **(kwargs | {"instance_indices": {}})))
    absent = list(
        read_sapiens2(path, **(kwargs | {"instance_indices": {"00012.png": None}}))
    )[0]
    assert absent.positions_px == {}
    with pytest.raises(ValueError, match="3D"):
        TrackerFrame(1, "camera", 0, "P07", result.positions_px)


def test_calibrated_multiview_and_metric_depth_reconstruct_actual_scale():
    k = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    cameras = {
        "A": PinholeCamera(k, np.eye(3), np.zeros(3), "calA"),
        "B": PinholeCamera(k, np.eye(3), np.array([-0.8, 0, 0]), "calB"),
    }
    truth = {
        "left_shoulder": np.array([0.1, -0.2, 3]),
        "left_ankle": np.array([0.15, 0.6, 3.2]),
    }
    views = [
        Pose2D(
            1,
            name,
            "P07",
            12,
            {n: c.project(p) for n, p in truth.items()},
            {n: 0.9 for n in truth},
            capture_unix=1001,
        )
        for name, c in cameras.items()
    ]
    result = reconstruct_multiview(views, cameras)
    for name, value in truth.items():
        np.testing.assert_allclose(result.positions[name], value, atol=1e-12)
    assert result.capture_unix == 1001
    depth = reconstruct_depth(
        views[0], cameras["A"], {n: p[2] for n, p in truth.items()}
    )
    np.testing.assert_allclose(depth.positions["left_ankle"], truth["left_ankle"])
    bad_depth = reconstruct_depth(views[0], cameras["A"], {"left_ankle": -1})
    assert bad_depth.positions == {}
    assert "left_ankle" in bad_depth.metadata["reconstruction_rejections"]
    swapped = Pose2D(1, "B", "P08", 12, {}, {})
    with pytest.raises(ValueError, match="same person"):
        reconstruct_multiview([views[0], swapped], cameras)
    late = Pose2D(1.1, "B", "P07", 12, {}, {})
    with pytest.raises(ValueError, match="synchronized"):
        reconstruct_multiview([views[0], late], cameras)
