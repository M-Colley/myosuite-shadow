"""Shared models."""

from __future__ import annotations

from collections.abc import Iterator

import mujoco
import pytest
from myosuite import make_env


@pytest.fixture(scope="session")
def elbow() -> Iterator[mujoco.MjModel]:
    """MyoSuite's 1-dof elbow with six muscles."""
    env = make_env("myoElbowPose1D6MRandom-v0")
    yield env.unwrapped.model
    env.close()


@pytest.fixture(scope="session")
def arm() -> Iterator[mujoco.MjModel]:
    """myoArm: 50 muscles and joint couplings, fixed base."""
    env = make_env("myoArmReachRandom-v0")
    yield env.unwrapped.model
    env.close()
