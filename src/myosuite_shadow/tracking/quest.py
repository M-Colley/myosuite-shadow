"""Decode versioned packets from the native Quest/OpenXR example sender."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .frames import ClockMapping, TrackerFrame

_CORE = (
    "root",
    "hips",
    "spine_lower",
    "spine_middle",
    "spine_upper",
    "chest",
    "neck",
    "head",
    "left_collar",
    "left_scapula",
    "left_shoulder",
    "left_elbow",
    "left_wrist_twist",
    "right_collar",
    "right_scapula",
    "right_shoulder",
    "right_elbow",
    "right_wrist_twist",
)


def _hand(side: str) -> tuple[str, ...]:
    names = [f"{side}_palm", f"{side}_wrist"]
    for finger in ("thumb", "index", "middle", "ring", "little"):
        bones = (
            ("metacarpal", "proximal", "distal", "tip")
            if finger == "thumb"
            else ("metacarpal", "proximal", "intermediate", "distal", "tip")
        )
        names.extend(f"{side}_{finger}_{bone}" for bone in bones)
    return tuple(names)


QUEST_UPPER_JOINTS = _CORE + _hand("left") + _hand("right")
QUEST_FULL_JOINTS = QUEST_UPPER_JOINTS + tuple(
    f"{side}_{joint}"
    for side in ("left", "right")
    for joint in (
        "hip",
        "knee",
        "ankle_twist",
        "ankle",
        "subtalar",
        "transverse",
        "foot_ball",
    )
)


class QuestDecoder:
    """Validate OpenXR body packets while preserving valid versus tracked status.

    Device/SDK versions are recorded, not used as capability guesses. The
    sender must negotiate extensions and report the joint set and actual
    fidelity. A supplied clock mapping must match the packet's clock identity.
    """

    def __init__(
        self,
        clock: ClockMapping,
        *,
        expected_person_id: str | None = None,
        expected_source_id: str | None = None,
        clock_provider: Callable[[], ClockMapping] | None = None,
    ) -> None:
        self.clock = clock
        self.expected_person_id = expected_person_id
        self.expected_source_id = expected_source_id
        self.clock_provider = clock_provider

    def decode(
        self,
        packet: dict[str, Any],
        *,
        receive_unix: float | None = None,
    ) -> TrackerFrame:
        """Return one frame; reject ambiguous clocks, versions and joint layouts."""
        clock = self.clock_provider() if self.clock_provider is not None else self.clock
        if (
            not isinstance(packet, dict)
            or isinstance(packet.get("schema_version"), bool)
            or packet.get("schema_version") != 1
            or packet.get("kind") != "quest_body"
        ):
            raise ValueError("unsupported Quest packet kind/schema_version")
        if packet.get("clock_id") != clock.clock_id:
            raise ValueError("Quest packet clock does not match clock mapping")
        for key, expected in (
            ("person_id", self.expected_person_id),
            ("source_id", self.expected_source_id),
        ):
            if expected is not None and packet.get(key) != expected:
                raise ValueError(f"unexpected Quest {key}; refusing identity switch")
        joint_set = packet.get("joint_set")
        if joint_set not in ("upper_body", "full_body"):
            raise ValueError("unknown Quest joint set")
        capabilities = packet.get("capabilities", {})
        if not isinstance(capabilities, dict):
            raise ValueError("capabilities must be an object")
        if capabilities.get("body_tracking") is not True or (
            joint_set == "full_body" and capabilities.get("full_body") is not True
        ):
            raise ValueError("joint set is unsupported by reported capabilities")
        names = QUEST_FULL_JOINTS if joint_set == "full_body" else QUEST_UPPER_JOINTS
        joints = packet.get("joints")
        if not isinstance(joints, list) or len(joints) != len(names):
            raise ValueError(f"{joint_set} requires exactly {len(names)} joints")
        if any(not isinstance(joint, dict) for joint in joints):
            raise ValueError("each joint must be an object")
        ids = [joint.get("id") for joint in joints]
        if any(isinstance(i, bool) or not isinstance(i, int) for i in ids) or set(
            ids
        ) != set(range(len(names))):
            raise ValueError("joint IDs must uniquely cover the reported joint set")
        active = packet.get("is_active")
        if not isinstance(active, bool):
            raise ValueError("is_active must be boolean")
        fidelity = packet.get("actual_fidelity", "unknown")
        if fidelity not in ("unknown", "low", "high"):
            raise ValueError("unknown actual_fidelity")
        if packet.get("base_space") not in ("stage", "local", "local_floor"):
            raise ValueError("base_space must be a world-locked OpenXR reference space")
        body_confidence = float(packet.get("confidence", 0))
        if not np.isfinite(body_confidence) or not 0 <= body_confidence <= 1:
            raise ValueError("body confidence must be within [0, 1]")
        positions, orientations, tracked, orientation_tracked = {}, {}, {}, {}
        for joint in joints:
            name = names[joint["id"]]
            flags = joint.get("location_flags")
            if isinstance(flags, bool) or not isinstance(flags, int) or flags < 0:
                raise ValueError("location_flags must be a nonnegative integer")
            if (flags & 8 and not flags & 2) or (flags & 4 and not flags & 1):
                raise ValueError("OpenXR tracked flags require matching valid flags")
            if active and flags & 2:
                positions[name] = joint["position"]
                tracked[name] = bool(flags & 8)
            if active and flags & 1:
                orientations[name] = joint["orientation"]
                orientation_tracked[name] = bool(flags & 4)
        source_s = float(packet["time_s"])
        session_s, unix = clock.map(source_s)
        time_kind = packet.get("time_kind", "estimate")
        prediction_s = float(packet.get("prediction_s", 0))
        capture_unix = unix if time_kind == "capture" and prediction_s == 0 else None
        return TrackerFrame(
            time_s=session_s,
            source_id=packet["source_id"],
            sequence=packet["sequence"],
            person_id=packet["person_id"],
            positions=positions,
            confidence={n: body_confidence for n in positions},
            orientations=orientations,
            position_tracked=tracked,
            orientation_tracked=orientation_tracked,
            capture_unix=capture_unix,
            evaluation_unix=unix,
            receive_unix=receive_unix,
            prediction_s=prediction_s,
            time_kind=time_kind,
            coordinate_frame=f"quest_{packet['base_space']}",
            transform_revision=packet["space_revision"],
            skeleton_revision=packet["skeleton_revision"],
            metadata={
                "raw_packet": packet,
                "clock_mapping": clock.record(),
                "actual_fidelity": fidelity,
                "joint_set": joint_set,
                "is_active": active,
                "capabilities": capabilities,
            },
        )
