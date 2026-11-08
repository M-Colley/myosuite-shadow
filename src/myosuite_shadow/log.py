"""Session logs: per-frame estimates, the raw pose stream and a provenance record.

A log directory holds

* ``record.json``: creation time, the model's SHA-256 and joint layout, the
  versions and sources (git commit for VCS and editable installs) of the
  packages, the settings and the caller's metadata;
* ``estimates.csv``: one row per estimate (estimated and newest pose time,
  update cost, solver status, peak reserve, activations, ``MF``, chosen
  bodies' contact forces);
* ``inputs.csv``: every pose pushed (time, ``qpos``, known external
  generalized force), so a session can be replayed offline with other settings;
* ``model.mjb``: the compiled model (optional).

Rows are flushed as they are written, so a crash loses at most one frame.
Floats are written with Python's shortest round-trip repr, so a replay of
``inputs.csv`` sees bit-identical poses.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
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

RECORD, ESTIMATES, INPUTS, MODEL = (
    "record.json",
    "estimates.csv",
    "inputs.csv",
    "model.mjb",
)
_PACKAGES = ("myosuite-shadow", "myosuite", "myo-sim", "mujoco", "osqp", "numpy")
_ESTIMATE_COLUMNS = ["time", "latest_time", "compute_ms", "status", "max_abs_reserve"]


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
                entry.update(
                    _git_state(Path(url2pathname(urlparse(entry["url"]).path)))
                )
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
    actuators = [model.actuator(i).name for i in range(model.nu)]
    return {
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "joints": joints,
        "actuators": actuators,
    }


class SessionWriter:
    """Writes one session's record, estimates and inputs."""

    def __init__(
        self,
        log_dir: Path,
        model: mujoco.MjModel,
        *,
        config: dict[str, Any],
        fatigue: bool,
        contact_bodies: tuple[str, ...],
        metadata: dict[str, Any],
        save_model: bool,
    ) -> None:
        """Write the record (and model) and open both CSV files.

        Args:
            log_dir: Output directory, created if needed.
            model: The model the poses are expressed in.
            config: JSON-serialisable settings of the pipeline.
            fatigue: Whether ``MF`` columns are written.
            contact_bodies: Bodies whose contact force columns are written.
            metadata: Caller fields stored under ``metadata``.
            save_model: Also store ``model.mjb``.
        """
        log_dir.mkdir(parents=True, exist_ok=True)
        self.nq, self.nv = model.nq, model.nv
        record = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_provenance(),
            "model_sha256": model_sha256(model),
            "model": _model_layout(model),
            "config": config,
            "metadata": metadata,
            "estimates": (
                "minimum-effort static optimization; model estimates, not measurements"
            ),
        }
        (log_dir / RECORD).write_text(json.dumps(record, indent=2), encoding="utf-8")
        if save_model:
            mujoco.mj_saveModel(model, str(log_dir / MODEL), None)

        names = record["model"]["actuators"]
        header = _ESTIMATE_COLUMNS + [f"a_{n}" for n in names]
        if fatigue:
            header += [f"mf_{n}" for n in names]
        header += [f"contact_{b}_{x}" for b in contact_bodies for x in "xyz"]
        self._estimates = open(log_dir / ESTIMATES, "w", newline="", encoding="utf-8")  # noqa: SIM115
        self._estimate_writer = csv.writer(self._estimates)
        self._estimate_writer.writerow(header)
        self._inputs = open(log_dir / INPUTS, "w", newline="", encoding="utf-8")  # noqa: SIM115
        self._input_writer = csv.writer(self._inputs)
        self._input_writer.writerow(
            ["time"]
            + [f"qpos_{i}" for i in range(self.nq)]
            + [f"qfrc_ext_{i}" for i in range(self.nv)]
        )

    def write_input(
        self, time_s: float, qpos: np.ndarray, external_qfrc: np.ndarray | None
    ) -> None:
        """Append one pushed pose; no external load is written as zeros."""
        ext = np.zeros(self.nv) if external_qfrc is None else external_qfrc
        row = [float(time_s)]
        row += np.asarray(qpos, dtype=np.float64).tolist()
        row += np.asarray(ext, dtype=np.float64).tolist()
        self._input_writer.writerow(row)
        self._inputs.flush()

    def write_estimate(self, frame: ShadowFrame, contact_force: np.ndarray) -> None:
        """Append one estimate with the chosen bodies' ``(n, 3)`` contact forces."""
        res = frame.optimization
        row = [frame.state.time, frame.latest_time, frame.compute_ms, str(res.status)]
        row += [float(np.abs(res.reserve).max())] + res.activation.tolist()
        row += [] if frame.fatigue is None else frame.fatigue.tolist()
        row += contact_force.ravel().tolist()
        self._estimate_writer.writerow(row)
        self._estimates.flush()

    def close(self) -> None:
        """Close both CSV files."""
        self._estimates.close()
        self._inputs.close()


@dataclass(frozen=True)
class Session:
    """A session log read back.

    Attributes:
        record: The provenance record (``record.json``).
        time: ``(n,)`` timestamps of the estimated frames [s].
        latest_time: ``(n,)`` newest pose time at each estimate [s].
        compute_ms: ``(n,)`` update cost [ms].
        status: ``(n,)`` solver status strings.
        max_abs_reserve: ``(n,)`` largest reserve generalized force.
        activation: ``(n, nu)`` activations.
        fatigue: ``(n, nu)`` ``MF``, or ``None`` without fatigue.
        contact_force: Per logged body, ``(n, 3)`` world-frame contact force [N].
        input_time: ``(m,)`` timestamps of every pushed pose [s].
        input_qpos: ``(m, nq)`` pushed poses.
        input_qfrc: ``(m, nv)`` known external generalized forces.
    """

    record: dict[str, Any]
    time: np.ndarray
    latest_time: np.ndarray
    compute_ms: np.ndarray
    status: np.ndarray
    max_abs_reserve: np.ndarray
    activation: np.ndarray
    fatigue: np.ndarray | None
    contact_force: dict[str, np.ndarray]
    input_time: np.ndarray
    input_qpos: np.ndarray
    input_qfrc: np.ndarray

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
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def read_session(log_dir: str | Path) -> Session:
    """Read a session log written by :class:`~myosuite_shadow.ShadowModel`.

    Args:
        log_dir: The session's directory.

    Returns:
        The record, estimates and inputs as arrays.
    """
    log_dir = Path(log_dir)
    record = json.loads((log_dir / RECORD).read_text(encoding="utf-8"))
    nu = record["model"]["nu"]
    header, rows = _read_csv(log_dir / ESTIMATES)
    cols = list(zip(*rows)) if rows else [()] * len(header)

    def numeric(i: int) -> np.ndarray:
        return np.asarray(cols[i], dtype=np.float64)

    def block(first: int, n: int) -> np.ndarray:
        return np.array([numeric(i) for i in range(first, first + n)]).T.reshape(-1, n)

    k = len(_ESTIMATE_COLUMNS)
    activation = block(k, nu)
    k += nu
    fatigue = None
    if k < len(header) and header[k].startswith("mf_"):
        fatigue, k = block(k, nu), k + nu
    contact = {
        header[i][len("contact_") : -2]: block(i, 3) for i in range(k, len(header), 3)
    }
    in_header, in_rows = _read_csv(log_dir / INPUTS)
    inputs = np.array(in_rows, dtype=np.float64).reshape(-1, len(in_header))
    nq = record["model"]["nq"]
    return Session(
        record=record,
        time=numeric(0),
        latest_time=numeric(1),
        compute_ms=numeric(2),
        status=np.asarray(cols[3], dtype=str),
        max_abs_reserve=numeric(4),
        activation=activation,
        fatigue=fatigue,
        contact_force=contact,
        input_time=inputs[:, 0],
        input_qpos=inputs[:, 1 : 1 + nq],
        input_qfrc=inputs[:, 1 + nq :],
    )
