"""Resume commits atomically across process death, checkpoints and replay failures."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile

import numpy as np
import pytest
from myosuite.biomechanics.static_optimization import StaticOptimizer

import myosuite_shadow.log as session_log
from myosuite_shadow import ShadowModel, ShadowModelConfig, read_session
from myosuite_shadow.log import (
    CHECKPOINT,
    EVENTS,
    INPUTS,
    RECORD,
    RECOVERY,
    SessionWriter,
)


def _crashed(model, path):
    shadow = ShadowModel(
        model, ShadowModelConfig(window=5, checkpoint_every=1000), log_dir=path
    )
    for index in range(12):
        shadow.push(index / 20, [1.2], [-15.0])
    shadow.log.close_files()


def test_replay_exception_preserves_all_inputs_for_second_attempt(
    elbow, tmp_path, monkeypatch
):
    _crashed(elbow, tmp_path)
    expected = read_session(tmp_path)
    original = StaticOptimizer.solve

    def unavailable(*args, **kwargs):
        raise RuntimeError("solver temporarily unavailable")

    monkeypatch.setattr(StaticOptimizer, "solve", unavailable)
    with pytest.raises(RuntimeError, match="temporarily"):
        ShadowModel.resume(tmp_path)
    assert (tmp_path / RECOVERY).exists()
    monkeypatch.setattr(StaticOptimizer, "solve", original)
    with ShadowModel.resume(tmp_path):
        pass
    actual = read_session(tmp_path)
    np.testing.assert_array_equal(actual.input_time, expected.input_time)
    np.testing.assert_allclose(actual.activation, expected.activation, atol=1e-6)
    assert len(actual.input_time) == 12 and len(actual.record["resumes"]) == 1
    assert not (tmp_path / RECOVERY).exists()


def test_boolean_checkpoint_offset_is_rejected_before_journal_creation(elbow, tmp_path):
    _crashed(elbow, tmp_path)
    path = tmp_path / CHECKPOINT
    checkpoint = json.loads(path.read_text())
    checkpoint["offsets"][EVENTS] = True
    path.write_text(json.dumps(checkpoint))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="checkpoint file offset"):
        ShadowModel.resume(tmp_path)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    assert not (tmp_path / RECOVERY).exists()


def test_reopen_record_failure_closes_unreturned_writer_handles(
    elbow, tmp_path, monkeypatch
):
    _crashed(elbow, tmp_path)
    closed = []
    write_json, close_files = session_log._write_json, SessionWriter.close_files

    def unavailable(path, *args, **kwargs):
        if path.name == RECORD:
            raise OSError("record metadata write failed")
        return write_json(path, *args, **kwargs)

    def tracked_close(writer):
        closed.append(writer)
        close_files(writer)

    monkeypatch.setattr(session_log, "_write_json", unavailable)
    monkeypatch.setattr(SessionWriter, "close_files", tracked_close)
    with pytest.raises(OSError, match="metadata") as retained_error:
        ShadowModel.resume(tmp_path)
    assert retained_error.traceback and closed
    assert all(stream.closed for stream in closed[0]._files.values())
    monkeypatch.setattr(session_log, "_write_json", write_json)
    with ShadowModel.resume(tmp_path):
        pass
    assert len(read_session(tmp_path).input_time) == 12


def test_process_death_after_replay_checkpoint_restores_original_pending_inputs(
    elbow, tmp_path
):
    _crashed(elbow, tmp_path)
    expected = read_session(tmp_path)
    script = """
import os, sys
from myosuite_shadow import ShadowModel
original = ShadowModel.push
count = 0
def interrupted(self, *args, **kwargs):
    global count
    result = original(self, *args, **kwargs)
    count += 1
    if count == 6:
        self.log.write_checkpoint(self._state())
        os._exit(77)
    return result
ShadowModel.push = interrupted
ShadowModel.resume(sys.argv[1])
"""
    child = subprocess.run([sys.executable, "-c", script, str(tmp_path)], timeout=30)
    assert child.returncode == 77
    assert json.loads((tmp_path / CHECKPOINT).read_text())["state"]["last_pose"] == 0.25
    # An interrupted rollback may also leave metadata only partly restored.
    (tmp_path / RECORD).write_text("{", encoding="utf-8")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="retry.*resume"):
        read_session(tmp_path)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with ShadowModel.resume(tmp_path) as restored:
        restored.push(0.6, [1.2], [-15.0])
    actual = read_session(tmp_path)
    np.testing.assert_array_equal(actual.input_time[:12], expected.input_time)
    np.testing.assert_allclose(actual.activation[:8], expected.activation, atol=1e-6)
    assert len(actual.input_time) == 13
    assert not (tmp_path / RECOVERY).exists()


@pytest.mark.parametrize("damage", ["checksum", "paths", "bounds"])
def test_invalid_recovery_archive_cannot_modify_session(elbow, tmp_path, damage):
    _crashed(elbow, tmp_path)
    writer, _, _, _ = SessionWriter.reopen(tmp_path, elbow.nv)
    writer.close_files()
    path = tmp_path / RECOVERY
    with zipfile.ZipFile(path) as archive:
        data = {name: archive.read(name) for name in archive.namelist()}
    if damage == "checksum":
        data[INPUTS] += b"modified"
        manifest = json.loads(data["manifest.json"])
        manifest["files"][INPUTS]["size"] += len(b"modified")
        data["manifest.json"] = json.dumps(manifest).encode()
    elif damage == "paths":
        data["../unexpected.csv"] = b"invalid path"
    else:
        manifest = json.loads(data["manifest.json"])
        manifest["files"][INPUTS]["offset"] = -1
        data["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in data.items():
            archive.writestr(name, content)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="recovery"):
        ShadowModel.resume(tmp_path)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
