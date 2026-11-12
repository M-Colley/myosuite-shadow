"""Session logs: estimates, the raw pose stream, events, checkpoints and provenance.

A log directory holds

* ``record.json``: creation time, the model's SHA-256 and joint layout, the
  versions and sources (git commit for VCS and editable installs) of the
  packages, the settings, the calibration, the caller's metadata and every resume;
* ``estimates.csv``: one row per estimate (estimated frame time, newest pose
  time, wall-clock times, update cost, solver status, quality flags, untracked
  gap integrated before it, skipped frames, peak reserve and root residual,
  activations, ``MF``, chosen bodies' contact forces);
* ``inputs.csv``: every pose pushed (time, receive, capture and evaluation wall-clock
  times, flags, ``qpos``, known external generalized force), so a session can be
  replayed offline with other settings;
* ``operations.jsonl``: input optimization decisions, explicit tracking resets
  and the initial fatigue state, preserving the live processing schedule;
* ``events.csv``: experimenter markers (wall-clock and session time, label, JSON);
* ``checkpoint.json``: the pipeline state and file offsets, rewritten atomically
  every ``checkpoint_every`` seconds, from which a crashed session resumes;
* ``model.mjb``: the compiled model (optional).

Rows are flushed as they are written; checkpoints also ``fsync``. Floats are
written with Python's shortest round-trip repr, so a replay of ``inputs.csv``
sees bit-identical poses.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import time
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse
from urllib.request import url2pathname

import mujoco
import numpy as np

if TYPE_CHECKING:
    from myosuite_shadow.pipeline import ShadowFrame, ShadowModel

RECORD, ESTIMATES, INPUTS, EVENTS, CHECKPOINT, MODEL = (
    "record.json",
    "estimates.csv",
    "inputs.csv",
    "events.csv",
    "checkpoint.json",
    "model.mjb",
)
OPERATIONS = "operations.jsonl"
RECOVERY = ".resume-recovery.zip"
_RECOVERY_FILES = (RECORD, CHECKPOINT, INPUTS, ESTIMATES, EVENTS, OPERATIONS)
_PACKAGES = (
    "myosuite-shadow",
    "myosuite",
    "myo-sim",
    "mujoco",
    "osqp",
    "numpy",
    "mink",
    "pylsl",
)
ESTIMATE_COLUMNS = [
    "time",
    "latest_time",
    "frame_unix",
    "done_unix",
    "compute_ms",
    "status",
    "flags",
    "gap_s",
    "skipped",
    "max_abs_reserve",
    "max_abs_residual",
]
INPUT_COLUMNS = ["time", "receive_unix", "capture_unix", "flags"]
EVENT_COLUMNS = ["unix", "session_time", "label", "data"]


def model_sha256(model: mujoco.MjModel) -> str:
    """SHA-256 of the compiled model's binary (MJB) form."""
    buffer = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def _git_state(path: Path) -> dict[str, Any]:
    """Commit and dirty flag of a local checkout (empty if *path* is not one)."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {}  # No git, or not a checkout: the version alone identifies it.
    return {"commit": commit, "dirty": bool(dirty)}


def package_provenance(names: tuple[str, ...] = _PACKAGES) -> dict[str, Any]:
    """Version and source of each installed package.

    Args:
        names: Distribution names.

    Returns:
        Per package ``version`` (``None`` if not installed) and, for direct
        installs (PEP 610), the source ``url`` with its ``commit``; editable
        installs from a git checkout add the checkout's commit and dirty flag.
    """
    out: dict[str, Any] = {}
    for name in names:
        try:
            dist = importlib_metadata.distribution(name)
        except importlib_metadata.PackageNotFoundError:
            out[name] = {"version": None}
            continue
        entry: dict[str, Any] = {"version": dist.version}
        direct = dist.read_text("direct_url.json")
        if direct:
            info = json.loads(direct)
            entry["url"] = info.get("url")
            if "vcs_info" in info:
                entry["commit"] = info["vcs_info"].get("commit_id")
                entry["requested_revision"] = info["vcs_info"].get("requested_revision")
            elif info.get("dir_info", {}).get("editable"):
                path = Path(url2pathname(urlparse(entry["url"]).path))
                entry.update(_git_state(path))
        out[name] = entry
    return out


def _model_layout(model: mujoco.MjModel) -> dict[str, Any]:
    joints = [
        {
            "name": model.joint(j).name,
            "type": int(model.jnt_type[j]),
            "qposadr": int(model.jnt_qposadr[j]),
            "dofadr": int(model.jnt_dofadr[j]),
        }
        for j in range(model.njnt)
    ]
    return {
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "mass": float(model.body_mass[1:].sum()),
        "joints": joints,
        "actuators": [model.actuator(i).name for i in range(model.nu)],
    }


def _csv_line(row: list[Any]) -> bytes:
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerow(row)
    return buf.getvalue().encode("utf-8")


def _num(value: float | None) -> float | str:
    return "" if value is None else float(value)


def _complete_rows(path: Path, offset: int) -> list[list[str]]:
    """Whole CSV rows after byte *offset* (a torn last line is dropped)."""
    data = path.read_bytes()[offset:]
    data = data[: data.rfind(b"\n") + 1]
    return list(csv.reader(io.StringIO(data.decode("utf-8"))))


def _truncate(path: Path, offset: int) -> None:
    with open(path, "r+b") as f:
        f.truncate(offset)


class SessionWriter:
    """Writes one session's record, estimates, inputs, events and checkpoints."""

    def __init__(
        self,
        log_dir: Path,
        model: mujoco.MjModel,
        *,
        config: dict[str, Any],
        fatigue: bool,
        contact_bodies: tuple[str, ...],
        metadata: dict[str, Any],
        calibration: dict[str, Any],
        save_model: bool,
    ) -> None:
        """Write the record (and model) and start the CSV files.

        Args:
            log_dir: Output directory, created if needed; must not hold a session.
            model: The model the poses are expressed in.
            config: JSON-serialisable settings of the pipeline.
            fatigue: Whether ``MF`` columns are written.
            contact_bodies: Bodies whose contact force columns are written.
            metadata: Caller fields stored under ``metadata``.
            calibration: Calibration record stored under ``calibration``.
            save_model: Also store ``model.mjb``.

        Raises:
            FileExistsError: If *log_dir* already holds a session.
        """
        log_dir.mkdir(parents=True, exist_ok=True)
        if (log_dir / RECORD).exists():
            raise FileExistsError(f"{log_dir} already holds a session; use resume")
        self.log_dir = log_dir
        record = {
            "schedule_version": 1,
            "frame_unix_semantics": (
                "measured capture_unix; otherwise tracked/predicted "
                "evaluation_unix; otherwise receive_unix"
            ),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_provenance(),
            "model_sha256": model_sha256(model),
            "model": _model_layout(model),
            "config": config,
            "calibration": calibration,
            "metadata": metadata,
            "resumes": [],
            "estimates": (
                "minimum-effort static optimization; model estimates, not measurements"
            ),
        }
        _write_json(log_dir / RECORD, record)
        if save_model:
            mujoco.mj_saveModel(model, str(log_dir / MODEL), None)
        names = record["model"]["actuators"]
        header = ESTIMATE_COLUMNS + [f"a_{n}" for n in names]
        if fatigue:
            header += [f"mf_{n}" for n in names]
        header += [f"contact_{b}_{x}" for b in contact_bodies for x in "xyz"]
        header += [f"reserve_{i}" for i in range(model.nv)]
        header += [f"tau_{i}" for i in range(model.nv)]
        inputs = INPUT_COLUMNS + [f"qpos_{i}" for i in range(model.nq)]
        inputs += [f"qfrc_ext_{i}" for i in range(model.nv)]
        inputs += ["evaluation_unix"]
        self._files = {
            ESTIMATES: open(log_dir / ESTIMATES, "wb"),  # noqa: SIM115
            INPUTS: open(log_dir / INPUTS, "wb"),  # noqa: SIM115
            EVENTS: open(log_dir / EVENTS, "wb"),  # noqa: SIM115
            OPERATIONS: open(log_dir / OPERATIONS, "wb"),  # noqa: SIM115
        }
        self.input_count = 0
        self._log_joint_details = True
        self._log_evaluation_time = True
        for name, cols in (
            (ESTIMATES, header),
            (INPUTS, inputs),
            (EVENTS, EVENT_COLUMNS),
        ):
            self._write(name, cols)
        self.nv = model.nv

    @classmethod
    def reopen(
        cls, log_dir: Path, nv: int, *, fatigue_count: int | None = None
    ) -> tuple[
        SessionWriter, dict[str, Any], list[list[str]], list[dict[str, Any]] | None
    ]:
        """Cut a crashed session back to its checkpoint and append from there.

        Inputs and estimates after the checkpoint are removed (the inputs are
        returned for replay); events are kept, minus a torn last line.

        Args:
            log_dir: The session's directory.
            nv: Degrees of freedom of the session's model.

        Returns:
            Writer, checkpoint, pending input rows and their replay operations.
        """
        cls.restore_recovery(log_dir)
        checkpoint = json.loads((log_dir / CHECKPOINT).read_text(encoding="utf-8"))
        record = json.loads((log_dir / RECORD).read_text(encoding="utf-8"))
        offsets = checkpoint["offsets"]
        for name, offset in offsets.items():
            path = log_dir / name
            if (
                not isinstance(offset, int)
                or isinstance(offset, bool)
                or offset < 0
                or not path.exists()
                or offset > path.stat().st_size
            ):
                raise ValueError(f"invalid checkpoint file offset: {name}")
        pending = _complete_rows(log_dir / INPUTS, offsets[INPUTS])
        operations = None
        has_schedule = OPERATIONS in offsets and (log_dir / OPERATIONS).exists()
        if record.get("schedule_version") and not has_schedule:
            raise ValueError("session replay schedule is missing")
        if has_schedule:
            all_input_rows = _complete_rows(log_dir / INPUTS, 0)
            all_ops = (log_dir / OPERATIONS).read_bytes()
            validate_operations(
                _read_operations_bytes(all_ops),
                len(all_input_rows) - 1,
                fatigue_count=fatigue_count,
            )
            raw_ops = all_ops[offsets[OPERATIONS] :]
            operations = _read_operations_bytes(raw_ops)
            restored_count = len(all_input_rows) - 1 - len(pending)
            operations = [
                dict(op, input=op["input"] - restored_count) for op in operations
            ]
            validate_operations(operations, len(pending))
        _prepare_recovery(log_dir, offsets)
        if has_schedule:
            _truncate(log_dir / OPERATIONS, offsets[OPERATIONS])
        _truncate(log_dir / INPUTS, offsets[INPUTS])
        _truncate(log_dir / ESTIMATES, offsets[ESTIMATES])
        events = (log_dir / EVENTS).read_bytes()
        _truncate(log_dir / EVENTS, events.rfind(b"\n") + 1)
        self = cls.__new__(cls)
        self.log_dir, self.nv = log_dir, nv
        input_header, input_rows = _read_csv(log_dir / INPUTS)
        self.input_count = len(input_rows)
        self._log_evaluation_time = "evaluation_unix" in input_header
        self._log_joint_details = "reserve_0" in _read_csv(log_dir / ESTIMATES)[0]
        self._files = {}
        try:
            for name in (ESTIMATES, INPUTS, EVENTS):
                self._files[name] = open(log_dir / name, "ab")  # noqa: SIM115
            # A legacy resume starts a schedule covering its existing inputs too.
            if not has_schedule:
                with open(log_dir / OPERATIONS, "wb") as f:
                    for i in range(self.input_count):
                        f.write(
                            (
                                json.dumps(
                                    {"kind": "pose", "input": i, "estimate": True}
                                )
                                + "\n"
                            ).encode()
                        )
            self._files[OPERATIONS] = open(log_dir / OPERATIONS, "ab")
            if not has_schedule:
                record["schedule_reconstructed"] = True
            record["resumes"].append(
                {
                    "resumed_utc": datetime.now(timezone.utc).isoformat(),
                    "checkpoint_time": checkpoint["state"]["last_pose"],
                    "replayed_inputs": len(pending),
                    "packages": package_provenance(),
                }
            )
            _write_json(log_dir / RECORD, record, durable=True)
        except BaseException:
            self.close_files()
            raise
        return self, checkpoint, pending, operations

    @staticmethod
    def read_record(log_dir: Path) -> dict[str, Any]:
        """Read original metadata for preflight without applying pending recovery."""
        if not (log_dir / RECOVERY).exists():
            return json.loads((log_dir / RECORD).read_text(encoding="utf-8"))
        with zipfile.ZipFile(log_dir / RECOVERY) as archive:
            files = _recovery_manifest(archive)
            item, data = files[RECORD], archive.read(RECORD)
            if (
                not isinstance(item, dict)
                or item.get("offset") != 0
                or item.get("size") != len(data)
                or item.get("sha256") != hashlib.sha256(data).hexdigest()
            ):
                raise ValueError("invalid resume recovery record checksum/metadata")
            return json.loads(data)

    @staticmethod
    def restore_recovery(log_dir: Path) -> None:
        """Restore an interrupted resume transaction, retaining its durable journal."""
        path = log_dir / RECOVERY
        if not path.exists():
            return
        with zipfile.ZipFile(path) as archive:
            files = _recovery_manifest(archive)
            for name, item in files.items():
                if item is None:
                    if name != OPERATIONS:
                        raise ValueError("required recovery file is missing")
                    continue
                if not isinstance(item, dict):
                    raise ValueError("invalid resume recovery file metadata")
                offset, size = item.get("offset"), item.get("size")
                if (
                    not isinstance(offset, int)
                    or isinstance(offset, bool)
                    or not isinstance(size, int)
                    or isinstance(size, bool)
                    or not 0 <= offset <= size
                    or name in (RECORD, CHECKPOINT)
                    and offset != 0
                    or archive.getinfo(name).file_size != size - offset
                ):
                    raise ValueError("invalid resume recovery file bounds")
                if offset and (
                    not (log_dir / name).exists()
                    or (log_dir / name).stat().st_size < offset
                ):
                    raise ValueError("recovery checkpoint prefix is missing")
                digest = hashlib.sha256()
                with archive.open(name) as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != item.get("sha256"):
                    raise ValueError("resume recovery checksum mismatch")
            checkpoint = json.loads(archive.read(CHECKPOINT))
            for name in (INPUTS, ESTIMATES, EVENTS, OPERATIONS):
                if files[name] is not None and files[name]["offset"] != checkpoint[
                    "offsets"
                ].get(name, 0):
                    raise ValueError(
                        "recovery file offset differs from original checkpoint"
                    )
            # Validate every backup before modifying any file; keep checkpoint last.
            for name in (*[n for n in _RECOVERY_FILES if n != CHECKPOINT], CHECKPOINT):
                item = files[name]
                target = log_dir / name
                if item is None:
                    target.unlink(missing_ok=True)
                    continue
                with target.open("r+b" if target.exists() else "w+b") as output:
                    output.truncate(item["offset"])
                    output.seek(item["offset"])
                    with archive.open(name) as source:
                        shutil.copyfileobj(source, output)
                    output.flush()
                    os.fsync(output.fileno())

    def commit_recovery(self, state: dict[str, Any]) -> None:
        """Durably commit the fully replayed state before retiring its journal."""
        self.write_checkpoint(state)
        (self.log_dir / RECOVERY).unlink()
        _sync_directory(self.log_dir)

    def close_files(self) -> None:
        """Release handles without committing a partially replayed state."""
        for stream in self._files.values():
            if not stream.closed:
                stream.close()

    def _write(self, name: str, row: list[Any]) -> None:
        f = self._files[name]
        f.write(_csv_line(row))
        f.flush()

    def write_input(
        self,
        time_s: float,
        receive_unix: float,
        capture_unix: float | None,
        flags: int,
        qpos: np.ndarray,
        external_qfrc: np.ndarray | None,
        *,
        estimate: bool = True,
        evaluation_unix: float | None = None,
    ) -> None:
        """Append one pushed pose; no external load is written as zeros."""
        ext = np.zeros(self.nv) if external_qfrc is None else external_qfrc
        row = [float(time_s), float(receive_unix), _num(capture_unix), int(flags)]
        row += np.asarray(qpos, dtype=np.float64).tolist()
        row += np.asarray(ext, dtype=np.float64).tolist()
        if self._log_evaluation_time:
            row += [_num(evaluation_unix)]
        self.write_operation(
            "pose", estimate=bool(estimate), evaluation_unix=evaluation_unix
        )
        self._write(INPUTS, row)
        self.input_count += 1

    def write_operation(self, kind: str, **data: Any) -> None:
        """Write a replay decision before its input or state change is applied."""
        f = self._files[OPERATIONS]
        f.write(
            (
                json.dumps({"kind": kind, "input": self.input_count, **data}) + "\n"
            ).encode("utf-8")
        )
        f.flush()

    def write_estimate(
        self,
        frame: ShadowFrame,
        done_unix: float,
        reserve: float,
        residual: float,
        contact_force: np.ndarray,
    ) -> None:
        """Append one estimate with the chosen bodies' ``(n, 3)`` contact forces."""
        res = frame.optimization
        row = [
            frame.state.time,
            frame.latest_time,
            frame.frame_unix,
            done_unix,
            frame.compute_ms,
            str(res.status),
            int(frame.flags),
            frame.gap,
            frame.skipped,
            reserve,
            residual,
        ]
        row += res.activation.tolist()
        row += [] if frame.fatigue is None else frame.fatigue.tolist()
        row += contact_force.ravel().tolist()
        if self._log_joint_details:
            row += res.reserve.tolist()
            row += res.joint_torque.tolist()
        self._write(ESTIMATES, row)

    def write_event(
        self,
        unix: float,
        session_time: float | None,
        label: str,
        data: dict[str, Any],
    ) -> None:
        """Append one experimenter event."""
        self._write(EVENTS, [unix, _num(session_time), label, json.dumps(data)])

    def write_final_event(
        self, unix: float, session_time: float | None, label: str, data: dict[str, Any]
    ) -> None:
        """Retry one final event without retaining a partial or duplicate CSV row."""
        if getattr(self, "_final_event_written", False):
            return
        stream = self._files[EVENTS]
        if not hasattr(self, "_final_event_offset"):
            self._final_event_offset = stream.tell()
        stream.flush()
        stream.truncate(self._final_event_offset)
        stream.seek(self._final_event_offset)
        self.write_event(unix, session_time, label, data)
        self._final_event_written = True

    def write_checkpoint(self, state: dict[str, Any], *, closed: bool = False) -> None:
        """Make the data durable, then atomically replace the checkpoint."""
        offsets = {}
        for name, f in self._files.items():
            f.flush()
            os.fsync(f.fileno())
            offsets[name] = f.tell()
        _write_json(
            self.log_dir / CHECKPOINT,
            {"offsets": offsets, "closed": closed, "state": state},
            durable=True,
        )

    def close(self, state: dict[str, Any]) -> None:
        """Write a final checkpoint and close the files."""
        if not getattr(self, "_final_checkpoint_written", False):
            self.write_checkpoint(state, closed=True)
            self._final_checkpoint_written = True
        self.close_files()


def _sync_directory(path: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _recovery_manifest(archive: zipfile.ZipFile) -> dict[str, Any]:
    manifest = json.loads(archive.read("manifest.json"))
    if not isinstance(manifest, dict):
        raise ValueError("invalid resume recovery manifest")
    files, version = manifest.get("files", {}), manifest.get("version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != 1
        or not isinstance(files, dict)
        or set(files) != set(_RECOVERY_FILES)
    ):
        raise ValueError("invalid resume recovery manifest")
    expected = {"manifest.json"} | {n for n, item in files.items() if item is not None}
    if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected):
        raise ValueError("invalid resume recovery archive paths")
    return files


def _prepare_recovery(log_dir: Path, offsets: dict[str, int]) -> None:
    journal = log_dir / RECOVERY
    if journal.exists():
        return
    temporary = journal.with_suffix(".zip.tmp")
    files = {}
    with temporary.open("w+b") as output:
        with zipfile.ZipFile(output, "w") as archive:
            for name in _RECOVERY_FILES:
                path = log_dir / name
                if not path.exists():
                    if name != OPERATIONS:
                        raise ValueError(f"required session file is missing: {name}")
                    files[name] = None
                    continue
                offset = offsets.get(name, 0) if name not in (RECORD, CHECKPOINT) else 0
                digest, count = hashlib.sha256(), 0
                with (
                    path.open("rb") as source,
                    archive.open(name, "w", force_zip64=True) as backup,
                ):
                    source.seek(offset)
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        backup.write(chunk)
                        digest.update(chunk)
                        count += len(chunk)
                files[name] = {
                    "offset": offset,
                    "size": offset + count,
                    "sha256": digest.hexdigest(),
                }
            archive.writestr(
                "manifest.json", json.dumps({"version": 1, "files": files})
            )
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, journal)
    _sync_directory(log_dir)


def _write_json(path: Path, value: dict[str, Any], *, durable: bool = False) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
        if durable:
            f.flush()
            os.fsync(f.fileno())
    os.replace(tmp, path)
    if durable:
        _sync_directory(path.parent)


@dataclass(frozen=True)
class Session:
    """A session log read back.

    Attributes:
        record: The provenance record (``record.json``).
        columns: Every ``estimates.csv`` column by name (numbers as float
            arrays, ``status`` as strings).
        activation: ``(n, nu)`` activations.
        fatigue: ``(n, nu)`` ``MF``, or ``None`` without fatigue.
        contact_force: Per logged body, ``(n, 3)`` world-frame contact force [N].
        input_time: ``(m,)`` timestamps of every pushed pose [s].
        input_qpos: ``(m, nq)`` pushed poses.
        input_qfrc: ``(m, nv)`` known external generalized forces.
        input_columns: ``receive_unix``, ``capture_unix``, ``evaluation_unix``
            (NaN if unknown) and ``flags`` of every pose.
        events: Experimenter events as dicts (``unix``, ``session_time``,
            ``label``, ``data``).
        checkpoint: The last checkpoint (``closed`` is true after a clean close).
    """

    record: dict[str, Any]
    columns: dict[str, np.ndarray]
    activation: np.ndarray
    fatigue: np.ndarray | None
    contact_force: dict[str, np.ndarray]
    input_time: np.ndarray
    input_qpos: np.ndarray
    input_qfrc: np.ndarray
    input_columns: dict[str, np.ndarray]
    events: list[dict[str, Any]] = field(default_factory=list)
    checkpoint: dict[str, Any] | None = None
    operations: list[dict[str, Any]] | None = None

    def replay(
        self,
        shadow: ShadowModel,
        *,
        qpos: np.ndarray | None = None,
        realtime: bool = False,
    ) -> Iterator[ShadowFrame]:
        """Replay accepted inputs, optimization decisions and explicit resets.

        Logs from 0.2 without an operation schedule use the original all-input
        replay. Such logs cannot reconstruct an unrecorded live skip or reset.
        """
        poses = self.input_qpos if qpos is None else np.asarray(qpos)
        if poses.shape != self.input_qpos.shape:
            raise ValueError("replay qpos must match the recorded input shape")
        inputs = [
            (
                float(t),
                q,
                ext,
                int(flags),
                _optional_time(capture),
                float(receive),
                _optional_time(evaluation),
            )
            for t, q, ext, flags, capture, receive, evaluation in zip(
                self.input_time,
                poses,
                self.input_qfrc,
                self.input_columns["flags"],
                self.input_columns["capture_unix"],
                self.input_columns["receive_unix"],
                self.input_columns.get(
                    "evaluation_unix", np.full(len(self.input_time), np.nan)
                ),
            )
        ]
        yield from replay_operations(shadow, inputs, self.operations, realtime=realtime)

    @property
    def time(self) -> np.ndarray:
        """``(n,)`` timestamps of the estimated frames [s]."""
        return self.columns["time"]

    @property
    def flags(self) -> np.ndarray:
        """``(n,)`` quality flags of the estimates (:class:`~myosuite_shadow.Flag`)."""
        return self.columns["flags"].astype(int)

    @property
    def status(self) -> np.ndarray:
        """``(n,)`` solver status strings."""
        return self.columns["status"]

    @property
    def actuators(self) -> list[str]:
        """Actuator names in column order."""
        return list(self.record["model"]["actuators"])

    def load_model(self, log_dir: str | Path) -> mujoco.MjModel:
        """The stored model, checked against the record's hash.

        Args:
            log_dir: The session's directory.

        Returns:
            The compiled model.

        Raises:
            FileNotFoundError: If the session has no ``model.mjb``.
            ValueError: If the model's hash differs from the record's.
        """
        model = mujoco.MjModel.from_binary_path(str(Path(log_dir) / MODEL))
        if model_sha256(model) != self.record["model_sha256"]:
            raise ValueError("model.mjb does not match the record's model_sha256")
        return model


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r]
    return rows[0], rows[1:]


def _float(values: list[str]) -> np.ndarray:
    return np.array([float(v) if v != "" else np.nan for v in values], dtype=float)


def _optional_time(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _read_operations_bytes(data: bytes) -> list[dict[str, Any]]:
    """Only newline-terminated records survive a process crash."""
    data = data[: data.rfind(b"\n") + 1]
    try:
        return [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid replay schedule record") from error


def validate_operations(
    operations: list[dict[str, Any]],
    input_count: int,
    *,
    fatigue_count: int | None = None,
) -> None:
    """Require ordered, complete replay decisions before applying any of them.

    One final pose decision may precede a torn input row. Resets after the last
    complete pose are valid. A missing decision for a complete input is corrupt.
    """
    cursor, initial_seen = 0, False
    for number, op in enumerate(operations):
        if not isinstance(op, dict) or op.get("kind") not in {
            "pose",
            "reset",
            "initial_fatigue",
        }:
            raise ValueError(f"unknown replay operation at record {number + 1}")
        index = op.get("input")
        if isinstance(index, bool) or not isinstance(index, int) or index != cursor:
            raise ValueError(
                f"replay operation index/order is invalid at record {number + 1}"
            )
        if op["kind"] == "pose":
            if not isinstance(op.get("estimate"), bool):
                raise ValueError("replay pose decisions require a boolean estimate")
            evaluation = op.get("evaluation_unix")
            if evaluation is not None and (
                isinstance(evaluation, bool)
                or not isinstance(evaluation, float | int)
                or not np.isfinite(evaluation)
            ):
                raise ValueError("replay evaluation timestamp must be finite")
            if cursor == input_count:
                if number != len(operations) - 1:
                    raise ValueError(
                        "a replay decision without a complete input must be last"
                    )
                break
            cursor += 1
        elif op["kind"] == "initial_fatigue":
            if cursor != 0 or initial_seen:
                raise ValueError(
                    "initial fatigue must be recorded once before the first input"
                )
            state = op.get("state")
            if not isinstance(state, dict) or set(state) != {"MA", "MR", "MF"}:
                raise ValueError("initial fatigue requires MA, MR and MF compartments")
            try:
                compartments = np.asarray(
                    [state[k] for k in ("MA", "MR", "MF")], dtype=float
                )
            except (TypeError, ValueError) as error:
                raise ValueError("invalid initial fatigue compartments") from error
            if compartments.ndim != 2 or not np.isfinite(compartments).all():
                raise ValueError("invalid initial fatigue compartments")
            if fatigue_count is not None and compartments.shape[1] != fatigue_count:
                raise ValueError("initial fatigue does not match the model's muscles")
            if (
                np.any(compartments < 0)
                or np.any(compartments > 1)
                or not np.allclose(compartments.sum(axis=0), 1)
            ):
                raise ValueError("initial fatigue compartments must sum to one")
            initial_seen = True
    if cursor != input_count:
        raise ValueError(
            "replay schedule is missing pose decisions for complete inputs"
        )


def replay_operations(
    shadow: ShadowModel,
    inputs: Sequence[tuple],
    operations: list[dict[str, Any]] | None,
    *,
    realtime: bool = False,
) -> Iterator[ShadowFrame]:
    """Apply one recorded schedule; used for replay, resume and sensitivity."""
    if operations is None:
        operations = [
            {"kind": "pose", "input": i, "estimate": True} for i in range(len(inputs))
        ]
    validate_operations(
        operations,
        len(inputs),
        fatigue_count=None
        if shadow.fatigue is None
        else len(shadow.fatigue.muscle_ids),
    )
    start = time.perf_counter()
    first = inputs[0][0] if inputs else 0.0
    for op in operations:
        index = op["input"]
        kind = op["kind"]
        if kind == "reset":
            shadow.reset_tracking()
        elif kind == "initial_fatigue":
            if shadow.fatigue is not None:
                shadow.fatigue.load_state_dict(op["state"])
        elif kind == "pose" and index < len(inputs):
            t, q, ext, flags, capture, receive, *rest = inputs[index]
            evaluation = op.get("evaluation_unix", rest[0] if rest else None)
            if realtime:
                time.sleep(max(0.0, t - first - (time.perf_counter() - start)))
            frame = shadow.push(
                t,
                q,
                ext,
                flags=flags,
                capture_unix=capture,
                evaluation_unix=evaluation,
                receive_unix=receive,
                estimate=op["estimate"],
            )
            if frame is not None:
                yield frame


def read_session(log_dir: str | Path) -> Session:
    """Read a session log written by :class:`~myosuite_shadow.ShadowModel`.

    Args:
        log_dir: The session's directory.

    Returns:
        The record, estimates, inputs and events as arrays.
    """
    log_dir = Path(log_dir)
    if (log_dir / RECOVERY).exists():
        raise ValueError(
            "unfinished session recovery; retry ShadowModel.resume before reading"
        )
    record = json.loads((log_dir / RECORD).read_text(encoding="utf-8"))
    names, nq, nu = (
        record["model"]["actuators"],
        record["model"]["nq"],
        record["model"]["nu"],
    )
    header, rows = _read_csv(log_dir / ESTIMATES)
    cols = list(zip(*rows)) if rows else [()] * len(header)
    by_name = dict(zip(header, cols))
    columns = {
        c: (np.asarray(by_name[c], dtype=str) if c == "status" else _float(by_name[c]))
        for c in ESTIMATE_COLUMNS
    }
    columns.update(
        {
            c: _float(by_name[c])
            for c in header
            if c.startswith("reserve_") or c.startswith("tau_")
        }
    )

    def block(prefix: str) -> np.ndarray | None:
        keys = [f"{prefix}{n}" for n in names]
        if keys[0] not in by_name:
            return None
        return np.array([_float(by_name[k]) for k in keys]).T.reshape(-1, nu)

    contact = {}
    for c in header:
        if c.startswith("contact_") and c.endswith("_x"):
            body = c[len("contact_") : -2]
            contact[body] = np.array(
                [_float(by_name[f"contact_{body}_{x}"]) for x in "xyz"]
            ).T.reshape(-1, 3)

    in_header, in_rows = _read_csv(log_dir / INPUTS)
    k = len(INPUT_COLUMNS)
    data = np.array(
        [[float(v) if v != "" else np.nan for v in r] for r in in_rows], dtype=float
    ).reshape(-1, len(in_header))
    input_columns = {c: data[:, i] for i, c in enumerate(INPUT_COLUMNS)}
    input_columns["evaluation_unix"] = (
        data[:, in_header.index("evaluation_unix")]
        if "evaluation_unix" in in_header
        else np.full(len(data), np.nan)
    )
    events = []
    if (log_dir / EVENTS).exists():
        _, ev_rows = _read_csv(log_dir / EVENTS)
        events = [
            {
                "unix": float(r[0]),
                "session_time": float(r[1]) if r[1] != "" else None,
                "label": r[2],
                "data": json.loads(r[3]),
            }
            for r in ev_rows
        ]
    checkpoint = None
    if (log_dir / CHECKPOINT).exists():
        checkpoint = json.loads((log_dir / CHECKPOINT).read_text(encoding="utf-8"))
    operations = None
    if (log_dir / OPERATIONS).exists():
        operations = _read_operations_bytes((log_dir / OPERATIONS).read_bytes())
        validate_operations(operations, len(data))
    elif record.get("schedule_version"):
        raise ValueError("session replay schedule is missing")
    return Session(
        record=record,
        columns=columns,
        activation=block("a_"),
        fatigue=block("mf_"),
        contact_force=contact,
        input_time=data[:, 0],
        input_qpos=data[:, k : k + nq],
        input_qfrc=data[:, k + nq : k + nq + record["model"]["nv"]],
        input_columns=input_columns,
        events=events,
        checkpoint=checkpoint,
        operations=operations,
    )
