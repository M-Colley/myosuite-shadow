"""Crash append preserves torn bytes and leaves complete tracking evidence usable."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import pytest

from myosuite_shadow.acquisition import AcquisitionConfig, TrackedSession
from myosuite_shadow.ik import IKConfig
from myosuite_shadow.tracking.frames import TrackerFrame
from myosuite_shadow.tracking.io import RawFrameWriter, read_frames, repair_jsonl_tail


def _frame(sequence: int) -> TrackerFrame:
    return TrackerFrame(
        time_s=sequence / 20,
        source_id="tracker",
        sequence=sequence,
        person_id="P07",
        positions={"wrist": [0.0, 0.0, 0.0]},
        position_tracked={"wrist": True},
    )


@pytest.mark.parametrize("tail_size", [0, 200_000])
def test_raw_append_preserves_torn_utf8_and_complete_frames(
    tmp_path: Path, tail_size: int
) -> None:
    path = tmp_path / "tracking.jsonl"
    with RawFrameWriter(path) as writer:
        writer.write(_frame(0))
    complete = path.read_bytes()
    torn = b'{"person":"' + b"x" * tail_size + b"\xe2\x82"
    with path.open("ab") as stream:
        stream.write(torn)
    with RawFrameWriter(path, append=True) as resumed:
        assert resumed.recovery_path.read_bytes() == torn
        assert path.read_bytes() == complete
        resumed.write(_frame(1))
    assert [frame.sequence for frame in read_frames(path)] == [0, 1]
    backup = resumed.recovery_path
    with RawFrameWriter(path, append=True) as repeated:
        assert repeated.recovery_path is None
    assert list(tmp_path.glob("*.bin")) == [backup]


def test_entire_torn_first_record_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "tracking.jsonl"
    torn = b'{"source":"\xf0\x9f'
    path.write_bytes(torn)
    recovered = repair_jsonl_tail(path)
    assert recovered.read_bytes() == torn and path.read_bytes() == b""
    with RawFrameWriter(path, append=True) as writer:
        writer.write(_frame(0))
    assert [frame.sequence for frame in read_frames(path)] == [0]


def test_complete_and_absent_files_are_unchanged(tmp_path: Path) -> None:
    absent = tmp_path / "absent.jsonl"
    assert repair_jsonl_tail(absent) is None and not absent.exists()
    for name, content in (
        ("empty", b""),
        ("complete", '{"person":"\u00e9"}\n'.encode()),
    ):
        path = tmp_path / f"{name}.jsonl"
        path.write_bytes(content)
        before = path.stat().st_mtime_ns
        assert repair_jsonl_tail(path) is None
        assert path.read_bytes() == content and path.stat().st_mtime_ns == before
    assert not list(tmp_path.glob("*.bin"))


def _boundary(model: mujoco.MjModel):
    events = []
    live = SimpleNamespace(
        shadow=SimpleNamespace(model=model, log=None),
        mark=lambda label, **data: events.append((label, data)),
    )
    ik = SimpleNamespace(model=model, config=IKConfig(), keypoints=[])
    return live, ik, events


def test_acquisition_repairs_both_records_and_logs_backup_evidence(
    elbow: mujoco.MjModel, tmp_path: Path
) -> None:
    raw = tmp_path / "tracking.jsonl"
    quality = tmp_path / "tracking_quality.jsonl"
    with RawFrameWriter(raw) as writer:
        writer.write(_frame(0))
    quality.write_text('{"accepted":true}\n', encoding="utf-8")
    tails = {raw: b'{"source":"\xe2', quality: b'{"reason":"\xf0\x9f'}
    for path, tail in tails.items():
        with path.open("ab") as stream:
            stream.write(tail)
    live, ik, events = _boundary(elbow)
    with TrackedSession(live, ik, raw_dir=tmp_path) as session:
        session._raw.write(_frame(1))
        session._write(0, {"accepted": False, "reason": "test"})
    assert [frame.sequence for frame in read_frames(raw)] == [0, 1]
    assert [
        json.loads(line)["accepted"] for line in quality.read_text().splitlines()
    ] == [True, False]
    recovered = next(
        data["files"] for label, data in events if label == "raw_recovered"
    )
    assert len(recovered) == 2
    for record in recovered:
        assert Path(record["backup"]).read_bytes() == tails[Path(record["file"])]
        assert record["bytes"] == len(tails[Path(record["file"])])


def test_preflight_errors_do_not_repair_or_create_files(
    elbow: mujoco.MjModel, tmp_path: Path
) -> None:
    raw = tmp_path / "tracking.jsonl"
    raw.write_bytes(b'{"torn":"\xe2')
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    live, ik, _ = _boundary(elbow)
    with pytest.raises(KeyError):
        TrackedSession(
            live,
            ik,
            config=AcquisitionConfig(held_mass_kg=2, load_site="absent_site"),
            raw_dir=tmp_path,
        )
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        RawFrameWriter(raw)
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}
