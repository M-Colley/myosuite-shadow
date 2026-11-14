"""Reference body correspondences, descriptive regions and explicit DOF scope.

Reference anatomy needs participant geometry/landmark review and measured
tracker-to-model registration. This profile does not establish observability.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from myosuite.biomechanics.observability import independent_dofs

from myosuite_shadow.ik import Keypoint


@dataclass(frozen=True)
class BodyRegion:
    """Descriptive muscle group, canonical display anchor and prerequisites."""

    id: str
    label: str
    anchor: str
    actuators: tuple[str, ...]
    required_landmarks: tuple[str, ...]


@dataclass(frozen=True)
class BodyProfile:
    """Compiled anatomy and declared independent tangent indices.

    Finger/toe DOFs remain nuisance motion rather than being assumed fixed.
    Muscles reaching those bodies are excluded from reported region summaries.
    Ground reactions require floor-aligned registration and valid foot contacts.
    """

    name: str
    model: mujoco.MjModel
    keypoints: tuple[Keypoint, ...]
    regions: tuple[BodyRegion, ...]
    target_dofs: tuple[int, ...]
    excluded_actuators: tuple[str, ...] = ()


def _subtree(model: mujoco.MjModel, name: str) -> set[int]:
    root = model.body(name).id
    result = {root}
    for index in range(root + 1, model.nbody):
        if int(model.body_parentid[index]) in result:
            result.add(index)
    return result


def _keypoints() -> tuple[Keypoint, ...]:
    points = [
        Keypoint("hips", "pelvis"),
        Keypoint("chest", "torso"),
        Keypoint("head", "head"),
    ]
    correspondences = (
        ("shoulder", "humerus"),
        ("elbow", "ulna"),
        ("wrist", "lunate"),
        ("index_metacarpal", "secondmc"),
        ("little_metacarpal", "fifthmc"),
        ("hip", "femur"),
        ("knee", "tibia"),
        ("ankle", "talus"),
        ("subtalar", "calcn"),
        ("foot_ball", "toes"),
    )
    for side, suffix in (("left", "l"), ("right", "r")):
        points.extend(
            Keypoint(f"{side}_{source}", f"{body}_{suffix}")
            for source, body in correspondences
        )
    return tuple(points)


def profile_for_model(model: mujoco.MjModel, name: str = "full_body") -> BodyProfile:
    """Describe compatible anatomy without modifying calibrated parameters.

    Body origins are joint-centre proxies requiring anatomical review.
    All mapped observations require a full-body source joint layout.
    """
    if name != "full_body":
        raise ValueError("available body profile: full_body")
    if not isinstance(model, mujoco.MjModel):
        raise TypeError("body profiles require a compiled MuJoCo model")
    points = _keypoints()
    expected = {p.frame for p in points} | {"clavicle_l", "clavicle_r"}
    missing = sorted(
        body
        for body in expected
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) < 0
    )
    if missing:
        raise ValueError(f"full_body model lacks expected body frames: {missing}")
    if np.count_nonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE) != 1:
        raise ValueError("full_body profile requires exactly one free root")
    branches = {
        "left_arm": _subtree(model, "clavicle_l"),
        "right_arm": _subtree(model, "clavicle_r"),
        "left_leg": _subtree(model, "femur_l"),
        "right_leg": _subtree(model, "femur_r"),
    }
    unobserved = set()
    for suffix in ("l", "r"):
        wrist = model.body(f"lunate_{suffix}").id
        unobserved |= _subtree(model, f"lunate_{suffix}") - {wrist}
        unobserved |= _subtree(model, f"toes_{suffix}")
    groups: dict[str, list[str]] = {key: [] for key in ("trunk", *branches)}
    excluded = []
    muscles = np.flatnonzero(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)
    for index in muscles:
        muscle = model.actuator(int(index)).name
        if not muscle or model.actuator_trntype[index] != mujoco.mjtTrn.mjTRN_TENDON:
            raise ValueError("full_body muscles require named tendon transmissions")
        tendon = int(model.actuator_trnid[index, 0])
        start, count = int(model.tendon_adr[tendon]), int(model.tendon_num[tendon])
        attachments = {
            int(model.site_bodyid[model.wrap_objid[wrap]])
            for wrap in range(start, start + count)
            if model.wrap_type[wrap] == mujoco.mjtWrap.mjWRAP_SITE
        }
        membership = [key for key, bodies in branches.items() if attachments & bodies]
        if len(membership) > 1:
            raise ValueError(f"muscle crosses multiple profile limbs: {muscle}")
        if attachments & unobserved:
            excluded.append(muscle)
            continue
        groups[membership[0] if membership else "trunk"].append(muscle)
    if any(not names for names in groups.values()):
        raise ValueError("full_body requires trunk and bilateral arm/leg muscles")
    regions = [
        BodyRegion(
            "trunk",
            "Trunk",
            "chest",
            tuple(groups["trunk"]),
            (
                "hips",
                "chest",
                "head",
                "left_shoulder",
                "right_shoulder",
                "left_hip",
                "right_hip",
            ),
        )
    ]
    for side in ("left", "right"):
        for limb, label, anchor, names in (
            (
                "arm",
                "arm and shoulder",
                "shoulder",
                ("shoulder", "elbow", "wrist", "index_metacarpal", "little_metacarpal"),
            ),
            ("leg", "leg", "hip", ("hip", "knee", "ankle", "subtalar", "foot_ball")),
        ):
            region = f"{side}_{limb}"
            regions.append(
                BodyRegion(
                    region,
                    f"{side.title()} {label}",
                    f"{side}_{anchor}",
                    tuple(groups[region]),
                    ("hips", "chest", *(f"{side}_{landmark}" for landmark in names)),
                )
            )
    targets = tuple(
        dof.index
        for dof in independent_dofs(model)
        if int(model.jnt_bodyid[model.dof_jntid[dof.index]]) not in unobserved
    )
    return BodyProfile(name, model, points, tuple(regions), targets, tuple(excluded))


def load_body_profile(name: str = "full_body") -> BodyProfile:
    """Build fresh reference anatomy from installed assets without downloads."""
    if name != "full_body":
        raise ValueError("available body profile: full_body")
    import myo_sim

    return profile_for_model(myo_sim.load_model("myofullbody"), name)
