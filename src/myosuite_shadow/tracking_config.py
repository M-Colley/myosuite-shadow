"""Load one tracking setup for acquisition, preflight and recorded regressions."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco

from myosuite_shadow.acquisition import AcquisitionConfig
from myosuite_shadow.ik import IKConfig, Keypoint, KeypointIK
from myosuite_shadow.pipeline import ShadowModelConfig
from myosuite_shadow.tracking import Registration


@dataclass(frozen=True)
class TrackingSetup:
    """Validated model and protocol settings; opens no source or session writer."""

    config_path: Path
    model_path: Path
    settings: dict[str, Any]
    model: mujoco.MjModel
    ik: KeypointIK
    registration: Registration | None
    acquisition: AcquisitionConfig
    shadow_config: ShadowModelConfig


def load_tracking_setup(
    config_path: str | Path, *, offline: bool = False
) -> TrackingSetup:
    """Resolve model paths relative to a JSON configuration and build its IK.

    Offline reprocessing disables wall-clock freshness while retaining the
    original source clocks. Spatial registration and all other gates remain.
    """
    config_path = Path(config_path).resolve()
    settings = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(settings, dict):
        raise ValueError("tracking configuration must be a JSON object")
    model_path = Path(settings["model"])
    if not model_path.is_absolute():
        model_path = config_path.parent / model_path
    model = (
        mujoco.MjModel.from_binary_path(str(model_path))
        if model_path.suffix.lower() == ".mjb"
        else mujoco.MjModel.from_xml_path(str(model_path))
    )
    ik_values = dict(settings.get("ik", {}))
    if "required_keypoints" in ik_values:
        ik_values["required_keypoints"] = tuple(ik_values["required_keypoints"])
    ik = KeypointIK(
        model, [Keypoint(**p) for p in settings["keypoints"]], IKConfig(**ik_values)
    )
    registration = (
        Registration(**settings["registration"]) if "registration" in settings else None
    )
    acquisition = AcquisitionConfig(**settings.get("acquisition", {}))
    if offline:
        acquisition = dataclasses.replace(acquisition, check_wallclock=False)
    return TrackingSetup(
        config_path,
        model_path.resolve(),
        settings,
        model,
        ik,
        registration,
        acquisition,
        ShadowModelConfig.from_dict(settings.get("shadow", {})),
    )
