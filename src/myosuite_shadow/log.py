"""Session logs: estimates, the raw pose stream, events, checkpoints and provenance.

A log directory holds

* ``record.json``: creation time, the model's SHA-256 and joint layout, the
  versions and sources (git commit for VCS and editable installs) of the
  packages, the settings, the calibration, the caller's metadata and every resume;
* ``estimates.csv``: one row per estimate (estimated frame time, newest pose
  time, wall-clock times, update cost, solver status, quality flags, untracked
  gap integrated before it, skipped frames, peak reserve and root residual,
  activations, ``MF``, chosen bodies' contact forces);
* ``inputs.csv``: every pose pushed (time, receive and capture wall-clock
  times, flags, ``qpos``, known external generalized force), so a session can be
  replayed offline with other settings;
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
import subprocess
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
    from myosuite_shadow.pipeline import ShadowFrame

RECORD, ESTIMATES, INPUTS, EVENTS, CHECKPOINT, MODEL = (
    "record.json",
    "estimates.csv",
    "inputs.csv",
    "events.csv",
    "checkpoint.json",
    "model.mjb",
)
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
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
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
        inputs = INPUT_COLUMNS + [f"qpos_{i}" for i in range(model.nq)]
        inputs += [f"qfrc_ext_{i}" for i in range(model.nv)]
        self._files = {
            ESTIMATES: open(log_dir / ESTIMATES, "wb"),  # noqa: SIM115
            INPUTS: open(log_dir / INPUTS, "wb"),  # noqa: SIM115
            EVENTS: open(log_dir / EVENTS, "wb"),  # noqa: SIM115
        }
        for name, cols in (
            (ESTIMATES, header),
            (INPUTS, inputs),
            (EVENTS, EVENT_COLUMNS),
        ):
            self._write(name, cols)
        self.nv = model.nv

    @classmethod
    def reopen(
        cls, log_dir: Path, nv: int
    ) -> tuple[SessionWriter, dict[str, Any], list[list[str]]]:
        """Cut a crashed session back to its checkpoint and append from there.

        Inputs and estimates after the checkpoint are removed (the inputs are
        returned for replay); events are kept, minus a torn last line.

        Args:
            log_dir: The session's directory.
            nv: Degrees of freedom of the session's model.

        Returns:
            The writer, the checkpoint, and the input rows after it.
        """
        checkpoint = json.loads((log_dir / CHECKPOINT).read_text(encoding="utf-8"))
        offsets = checkpoint["offsets"]
        pending = _complete_rows(log_dir / INPUTS, offsets[INPUTS])
        _truncate(log_dir / INPUTS, offsets[INPUTS])
        _truncate(log_dir / ESTIMATES, offsets[ESTIMATES])
        events = (log_dir / EVENTS).read_bytes()
        _truncate(log_dir / EVENTS, events.rfind(b"\n") + 1)
        self = cls.__new__(cls)
        self.log_dir, self.nv = log_dir, nv
        self._files = {
            name: open(log_dir / name, "ab")  # noqa: SIM115
            for name in (ESTIMATES, INPUTS, EVENTS)
        }
        record = json.loads((log_dir / RECORD).read_text(encoding="utf-8"))
        record["resumes"].append(
            {
                "resumed_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint_time": checkpoint["state"]["last_pose"],
                "replayed_inputs": len(pending),
                "packages": package_provenance(),
            }
        )
        _write_json(log_dir / RECORD, record)
        return self, checkpoint, pending

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
    ) -> None:
        """Append one pushed pose; no external load is written as zeros."""
        ext = np.zeros(self.nv) if external_qfrc is None else external_qfrc
        row = [float(time_s), float(receive_unix), _num(capture_unix), int(flags)]
        row += np.asarray(qpos, dtype=np.float64).tolist()
        row += np.asarray(ext, dtype=np.float64).tolist()
        self._write(INPUTS, row)

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
        self.write_checkpoint(state, closed=True)
        for f in self._files.values():
            f.close()


def _write_json(path: Path, value: dict[str, Any], *, durable: bool = False) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
        if durable:
            f.flush()
            os.fsync(f.fileno())
    os.replace(tmp, path)


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
        input_columns: ``receive_unix``, ``capture_unix`` (NaN if unknown) and
            ``flags`` of every pose.
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


def read_session(log_dir: str | Path) -> Session:
    """Read a session log written by :class:`~myosuite_shadow.ShadowModel`.

    Args:
        log_dir: The session's directory.

    Returns:
        The record, estimates, inputs and events as arrays.
    """
    log_dir = Path(log_dir)
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
    return Session(
        record=record,
        columns=columns,
        activation=block("a_"),
        fatigue=block("mf_"),
        contact_force=contact,
        input_time=data[:, 0],
        input_qpos=data[:, k : k + nq],
        input_qfrc=data[:, k + nq :],
        input_columns={c: data[:, i] for i, c in enumerate(INPUT_COLUMNS)},
        events=events,
        checkpoint=checkpoint,
    )
