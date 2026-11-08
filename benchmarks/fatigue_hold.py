"""Holding a load at the elbow for two minutes, with and without the fatigue cap.

MyoSuite's 1-dof elbow (six muscles) holds the forearm at 69 degrees against a
hand load, estimated at 20 Hz. With the cap, each muscle is limited to
``1 - MF`` of its force, so load shifts to fresher synergists and, once they
saturate, to the reserve: in model terms the load can no longer be held::

    python benchmarks/fatigue_hold.py
"""

from __future__ import annotations

import numpy as np
from myosuite import make_env

from myosuite_shadow import ShadowModel, ShadowModelConfig

RATE, SECONDS, ANGLE = 20.0, 120.0, 1.2


def hold(model, load: float, cap: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Activations, MF and elbow reserve of every estimate."""
    shadow = ShadowModel(
        model, ShadowModelConfig(window=5, fatigue_limits_activation=cap)
    )
    frames = [
        shadow.push(k / RATE, np.array([ANGLE]), np.array([load]))
        for k in range(int(SECONDS * RATE))
    ]
    frames = [f for f in frames if f is not None]
    return (
        np.array([f.optimization.activation for f in frames]),
        np.array([f.fatigue for f in frames]),
        np.array([f.optimization.reserve[0] for f in frames]),
    )


def main() -> None:
    """Print activations at the start and after two minutes as a Markdown table."""
    env = make_env("myoElbowPose1D6MRandom-v0")
    model = env.unwrapped.model
    shown = ("BIClong", "BICshort", "BRA")
    ids = [model.actuator(n).id for n in shown]
    print(
        "| Load [N m] | Cap | "
        + " | ".join(f"{n} start / 120 s" for n in shown)
        + " | Peak MF | Max excess over 1 - MF | Reserve at 120 s [N m] |"
    )
    print("|---|---|" + "---|" * len(shown) + "---|---|---|")
    for load in (10.0, 20.0, 40.0):
        for cap in (True, False):
            act, mf, reserve = hold(model, -load, cap)
            cells = " | ".join(f"{act[0, i]:.2f} / {act[-1, i]:.2f}" for i in ids)
            excess = max(float((act[1:] - (1.0 - mf[:-1])).max()), 0.0)
            print(
                f"| {load:g} | {'yes' if cap else 'no'} | {cells} | "
                f"{mf[-1].max():.2f} | {excess:.2f} | {reserve[-1]:.1f} |"
            )
    env.close()


if __name__ == "__main__":
    main()
