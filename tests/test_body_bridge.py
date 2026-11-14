"""The real wire boundary: clocks, metric provenance and fail-closed feedback."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from myosuite_shadow.body_cli import register, synthetic_stream
from myosuite_shadow.body_protocol import SCHEMA, BodyClock, BodySource, message
from myosuite_shadow.body_server import BodyServer
from myosuite_shadow.tracking import ClockMapping
from myosuite_shadow.tracking_config import load_tracking_setup


def packet():
    return {
        "type": "body_frame",
        "schema": SCHEMA,
        "source_id": "test",
        "person_id": "P01",
        "clock_id": "testclock",
        "coordinate_frame": "model_world",
        "time_s": 11.0,
        "sequence": 1,
        "positions": {"wrist": [1.0, 2.0, 3.0]},
        "confidence": {"wrist": 0.9},
        "position_tracked": {"wrist": False},
        "transform_revision": 0,
        "skeleton_revision": 0,
    }


def test_body_observation_time_is_not_capture_time():
    source = BodySource("test", "P01", "testclock", "model_world")
    mapping = ClockMapping(10.0, 0.0, "testclock", unix_anchor=100.0)
    frame = source.frame(packet(), mapping, receive_unix=101.01)
    assert frame.time_s == 1.0
    assert frame.evaluation_unix == 101.0
    assert frame.capture_unix is None
    assert frame.inferred_joints == ("wrist",)
    assert frame.metadata["capture_age_verified"] is False
    assert frame.metadata["pose_time_semantics"] == "client_observation_estimate"


@pytest.mark.parametrize(
    "field,value",
    [
        ("position_tracked", {}),
        ("confidence", {}),
        ("sequence", True),
        ("time_s", float("nan")),
        ("person_id", "P02"),
        ("coordinate_frame", "pixels"),
        ("actual_fidelity", "madeup"),
        ("positions", {"wrist": [0.0, 1.0]}),
    ],
)
def test_body_packet_requires_explicit_metric_evidence(field, value):
    raw = {**packet(), field: value}
    with pytest.raises((ValueError, TypeError)):
        BodySource("test", "P01", "testclock", "model_world").frame(
            raw,
            ClockMapping(10.0, 0.0, "testclock", unix_anchor=100.0),
            receive_unix=101.0,
        )


def test_clock_fits_only_matched_low_rtt_replies():
    clock = BodyClock("clock", 1000.0, max_uncertainty_s=0.02)
    for host in [0.0, 0.5, 1.0, 1.5]:
        probe = clock.probe(host)
        reply = {
            "type": "clock_reply",
            "schema": SCHEMA,
            "clock_id": "clock",
            "nonce": probe["nonce"],
            "source_time_s": host + 10.001,
        }
        clock.reply(reply, host + 0.002)
        with pytest.raises(ValueError, match="nonce"):
            clock.reply(reply, host + 0.003)
    assert clock.mapping is not None
    session, unix = clock.mapping.map(12.0)
    assert session == pytest.approx(2.0, abs=0.003)
    assert unix == pytest.approx(1002.0, abs=0.003)
    probe = clock.probe(2.0)
    clock.reply(
        {
            "type": "clock_reply",
            "schema": SCHEMA,
            "clock_id": "clock",
            "nonce": probe["nonce"],
            "source_time_s": 12.0,
        },
        2.1,
    )
    assert clock.error == "clock round trip exceeds 50 ms"
    assert len(clock.exchanges) == 4


def test_message_rejects_unknown_schema_and_nonfinite():
    for raw in ['{"type":"hello"}', [], {"schema": SCHEMA, "number": float("inf")}]:
        with pytest.raises(ValueError):
            message(raw)


def setup_elbow(tmp_path, elbow):
    model_path = tmp_path / "elbow.mjb"
    mujoco.mj_saveModel(elbow, str(model_path), None)
    path = tmp_path / "body.json"
    path.write_text(
        json.dumps(
            {
                "model": str(model_path),
                "initial_qpos": [1.2],
                "keypoints": [
                    {"name": "wrist", "frame": "wrist", "frame_type": "site"},
                    {"name": "elbow", "frame": "r_ulna_radius_hand"},
                ],
                "ik": {"required_keypoints": ["wrist", "elbow"]},
                "shadow": {"window": 5, "lag": 1},
                "body_source": {
                    "source_id": "synthetic-model",
                    "person_id": "P01",
                    "clock_id": "synthetic_monotonic",
                    "coordinate_frame": "model_world",
                },
                "synthetic": True,
            }
        ),
        encoding="utf-8",
    )
    return load_tracking_setup(path)


def regions(elbow):
    return (
        SimpleNamespace(
            id="arm",
            label="Arm muscles",
            anchor="elbow",
            required_landmarks=("wrist", "elbow"),
            actuators=tuple(elbow.actuator(i).name for i in range(elbow.nu)),
        ),
    )


def test_live_body_loopback_reaches_estimates_and_stales(tmp_path, elbow):
    async def run():
        setup = setup_elbow(tmp_path, elbow)
        server = BodyServer(setup, log_dir=tmp_path / "run", regions=regions(elbow))
        try:
            await server.start(port=0)
            producer = asyncio.create_task(
                synthetic_stream(setup.config_path, server.uri, 3.5)
            )
            states = []
            while not producer.done():
                states.append(server.state())
                await asyncio.sleep(0.04)
            await producer
            fresh = [s for s in states if s["quality"]["status"] == "running"]
            assert fresh, [
                (s["quality"]["status"], s["quality"]["reason"]) for s in states[-10:]
            ]
            assert all(s["regions"][0]["activation"] is not None for s in fresh)
            assert all(s["quality"]["capture_age_verified"] is False for s in states)
            assert server.state()["quality"]["status"] == "stale"
            assert server.state()["regions"][0]["activation"] is None
        finally:
            await server.close()
        raw = [
            json.loads(line)
            for line in (tmp_path / "run/tracking.jsonl").read_text().splitlines()
        ]
        assert raw and raw[-1]["metadata"]["source_metadata"]["synthetic"]
        assert (tmp_path / "run/body_wire.jsonl").is_file()

    asyncio.run(run())


def test_region_masks_reject_inferred_and_bad_results(tmp_path, elbow):
    setup = setup_elbow(tmp_path, elbow)
    server = BodyServer(setup, log_dir=tmp_path / "unused", regions=regions(elbow))
    server._active = True
    server._reason = None
    now = time.time()
    frame = BodySource("test", "P01", "testclock", "model_world").frame(
        {
            **packet(),
            "positions": {"wrist": [1, 2, 3], "elbow": [0, 0, 0]},
            "confidence": {"wrist": 1.0, "elbow": 1.0},
            "position_tracked": {"wrist": True, "elbow": False},
        },
        ClockMapping(11.0, 0.0, "testclock", unix_anchor=now),
        receive_unix=now,
    )
    server._history.append(frame)
    estimate = SimpleNamespace(
        frame_unix=now,
        state=SimpleNamespace(time=0.0),
        flags=0,
        optimization=SimpleNamespace(activation=np.full(elbow.nu, 0.2)),
        fatigue=np.full(elbow.nu, 0.1),
    )
    server._live = SimpleNamespace(latest=estimate)
    state = server.state()
    assert state["regions"][0]["status"] == "inferred"
    assert state["regions"][0]["activation"] is None
    server._history.append(
        dataclasses.replace(frame, position_tracked={"wrist": True, "elbow": True})
    )
    assert server.state()["regions"][0]["activation"] == 0.2
    estimate.optimization.activation[0] = float("nan")
    assert server.state()["quality"]["status"] == "failed"
    assert server.state()["regions"][0]["activation"] is None
    estimate.optimization.activation[0] = 0.2
    server.strict_pose_timing = True
    assert server.state()["quality"]["reason"] == "native_pose_capture_time_unverified"


def test_registration_preserves_explicit_pairs_and_refuses_overwrite(tmp_path, elbow):
    setup = setup_elbow(tmp_path, elbow)
    points = {"a": [0, 0, 0], "b": [1, 0, 0], "c": [0, 1, 0]}
    pairs = tmp_path / "pairs.json"
    pairs.write_text(
        json.dumps(
            {
                "source_frame": "model_world",
                "revision": 0,
                "source": points,
                "target": points,
            }
        )
    )
    output = tmp_path / "registered.json"
    register(setup.config_path, pairs, output, 0.02)
    result = load_tracking_setup(output)
    assert result.registration is not None
    assert result.settings["registration_evidence"]["pairs"]["target"] == points
    with pytest.raises(ValueError, match="already exists"):
        register(setup.config_path, pairs, output, 0.02)
