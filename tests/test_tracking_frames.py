"""Clock/registration semantics and acquisition records withstand bad inputs."""

from __future__ import annotations

import dataclasses
import json
import socket
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from myosuite_shadow.tracking import (
    ClockMapping,
    RawFrameWriter,
    Registration,
    TrackerFrame,
    UdpSource,
    fit_registration,
    read_frames,
)


def frame(**kwargs) -> TrackerFrame:
    values = dict(
        time_s=1.0,
        source_id="test",
        sequence=3,
        person_id="P07",
        positions={"wrist": [0, 1, 2]},
        orientations={"wrist": [0, 0, 0, 1]},
    )
    return TrackerFrame(**(values | kwargs))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"positions": {"wrist": [1, 2]}},
        {"positions": {"wrist": [1, 2, np.nan]}},
        {"confidence": {"wrist": np.nan}},
        {"confidence": {"wrist": -1}},
        {"time_s": np.inf},
        {"sequence": -1},
        {"schema_version": 2},
        {"orientations": {"wrist": [0, 0, 0, 0]}},
        {"position_valid": {"wrist": False}, "position_tracked": {"wrist": True}},
        {"capture_unix": 123.0, "time_kind": "predicted", "prediction_s": 0.02},
    ],
)
def test_canonical_frame_rejects_invalid_or_nonmetric_data(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        frame(**kwargs)


def test_registration_recovers_landmarks_orientation_and_enforces_recenter() -> None:
    source = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    r = Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_matrix()
    t = np.array([0.3, 0.2, -0.1])
    registration = fit_registration(source, source @ r.T + t)
    result = registration.apply(frame())
    np.testing.assert_allclose(result.positions["wrist"], r @ [0, 1, 2] + t)
    np.testing.assert_allclose(
        Rotation.from_quat(result.orientations["wrist"]).as_matrix(), r
    )
    assert result.coordinate_frame == "model_world"
    with pytest.raises(ValueError, match="revision"):
        registration.apply(dataclasses.replace(frame(), transform_revision=1))
    with pytest.raises(ValueError, match="collinear"):
        fit_registration(np.zeros((3, 3)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="proper rotation"):
        Registration(np.diag([1, 1, -1]), np.zeros(3))


def test_clock_is_explicit_rate_corrected_and_bounded() -> None:
    clock = ClockMapping(
        100,
        0,
        "device_clock",
        unix_anchor=1000,
        rate=1.0001,
        uncertainty_s=0.002,
        valid_until_source_s=150,
    )
    assert clock.map(110) == pytest.approx((10.001, 1010.001))
    with pytest.raises(ValueError, match="expired"):
        clock.map(151)
    assert ClockMapping(100, 0, "xr").map(101) == (1, None)


def test_raw_frame_roundtrip_and_exclusive_creation(tmp_path: Path) -> None:
    path = tmp_path / "tracking.jsonl"
    value = frame(metadata={"raw_packet": {"sdk_version": "future"}})
    with RawFrameWriter(path) as output:
        output.write(value)
        assert path.read_text()
    with pytest.raises(FileExistsError):
        RawFrameWriter(path)
    with RawFrameWriter(path, append=True) as output:
        output.write(value)
    records = list(read_frames(path))
    assert len(records) == 2 and records[0].record() == value.record()
    with path.open("a") as output:
        output.write('{"invalid":true}\n')
    with pytest.raises(ValueError, match="line 3"):
        list(read_frames(path))


def test_udp_decodes_real_datagram_and_rejects_oversized_packets() -> None:
    class Decoder:
        def decode(self, packet, *, receive_unix):
            return TrackerFrame.from_dict(packet)

    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with UdpSource(Decoder(), port=0, timeout_s=0.2) as source:
            sender.sendto(json.dumps(frame().record()).encode(), source.address)
            assert next(source).person_id == "P07"
            with pytest.raises(TimeoutError):
                next(source)
        with UdpSource(Decoder(), port=0, max_packet_bytes=10) as source:
            sender.sendto(b"x" * 11, source.address)
            with pytest.raises(ValueError, match="exceeded"):
                next(source)
    finally:
        sender.close()
