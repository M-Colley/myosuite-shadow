"""Flushable raw-frame records and bounded UDP acquisition."""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

from myosuite_shadow.clock import now

from .frames import TrackerFrame


def repair_jsonl_tail(path: str | Path) -> Path | None:
    """Preserve and remove a non-newline-terminated tail before resuming append.

    Writers commit newline-terminated records. Scan backwards in 64 KiB chunks,
    copy exact incomplete bytes to a unique durable sidecar, then truncate only
    that tail. Complete and absent files are unchanged. Call with no active writer.
    """
    path = Path(path)
    if not path.exists():
        return None
    chunk_size = 64 * 1024
    with path.open("rb") as source:
        original = os.fstat(source.fileno())
        size = original.st_size
        if not size:
            return None
        source.seek(size - 1)
        if source.read(1) == b"\n":
            return None
        cursor, boundary = size, 0
        while cursor:
            start = max(0, cursor - chunk_size)
            source.seek(start)
            chunk = source.read(cursor - start)
            last_newline = chunk.rfind(b"\n")
            if last_newline >= 0:
                boundary = start + last_newline + 1
                break
            cursor = start
        recovered = path.with_name(f"{path.name}.recovered-{uuid4().hex}.bin")
        source.seek(boundary)
        remaining = size - boundary
        with recovered.open("xb") as backup:
            while remaining:
                chunk = source.read(min(chunk_size, remaining))
                if not chunk:
                    raise OSError(
                        "JSONL file changed while preserving its incomplete tail"
                    )
                backup.write(chunk)
                remaining -= len(chunk)
            backup.flush()
            os.fsync(backup.fileno())
        with path.open("r+b") as target:
            current = os.fstat(target.fileno())
            if (current.st_size, current.st_mtime_ns, current.st_ino) != (
                size,
                original.st_mtime_ns,
                original.st_ino,
            ):
                raise OSError(
                    "JSONL file changed during tail recovery; original retained"
                )
            target.truncate(boundary)
            target.flush()
            os.fsync(target.fileno())
    return recovered


class RawFrameWriter:
    """Append every decoded frame, including rejected frames, before IK."""

    def __init__(self, path: str | Path, *, append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.recovery_path = repair_jsonl_tail(self.path) if append else None
        self._file = self.path.open(
            "a" if append else "x", encoding="utf-8", newline="\n"
        )

    def write(self, frame: TrackerFrame) -> None:
        """Write and flush a versioned frame, including its original packet."""
        self._file.write(json.dumps(frame.record(), allow_nan=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the raw acquisition record."""
        self._file.close()

    def __enter__(self) -> RawFrameWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_frames(path: str | Path) -> Iterator[TrackerFrame]:
    """Read canonical JSONL, reporting corrupt records with line numbers."""
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                yield TrackerFrame.from_dict(json.loads(line))
            except (TypeError, ValueError, KeyError) as error:
                raise ValueError(
                    f"invalid tracker record at line {line_number}"
                ) from error


class UdpSource:
    """Receive one JSON packet per datagram; malformed packets fail explicitly.

    Bind to loopback by default; choose a LAN interface explicitly for a Quest.
    Sequence gaps are retained for the study runner to audit. A timeout raises
    ``TimeoutError`` rather than silently displaying an old sample forever.
    """

    def __init__(
        self,
        decoder: Any,
        *,
        host: str = "127.0.0.1",
        port: int = 9763,
        timeout_s: float = 2.0,
        max_packet_bytes: int = 60_000,
        on_packet: Callable[[dict[str, Any], float], None] | None = None,
    ) -> None:
        if not 0 < timeout_s or not 1 <= max_packet_bytes <= 65_507:
            raise ValueError("invalid UDP timeout or maximum packet size")
        self.decoder, self.on_packet = decoder, on_packet
        self.max_packet_bytes = max_packet_bytes
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(timeout_s)
        self.socket.bind((host, port))

    @property
    def address(self) -> tuple[str, int]:
        """Return the bound host/port, including the actual ephemeral port."""
        return self.socket.getsockname()

    def __iter__(self) -> UdpSource:
        return self

    def __next__(self) -> TrackerFrame:
        data, _ = self.socket.recvfrom(self.max_packet_bytes + 1)
        received = now()
        if len(data) > self.max_packet_bytes:
            raise ValueError("UDP packet exceeded max_packet_bytes")
        packet = json.loads(data.decode("utf-8"))
        if not isinstance(packet, dict):
            raise ValueError("UDP packet must be a JSON object")
        if self.on_packet is not None:
            self.on_packet(packet, received)
        return self.decoder.decode(packet, receive_unix=received)

    def close(self) -> None:
        """Release the receiving socket."""
        self.socket.close()

    def __enter__(self) -> UdpSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
