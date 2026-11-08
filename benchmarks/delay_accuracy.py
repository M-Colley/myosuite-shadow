"""Delay against accuracy of live muscle estimates for seated reaching on myoArm.

Random minimum-jerk reaches are generated at 1.8 kHz; their noise-free
kinematics and static optimization are the reference. The shadow model sees
the 90 Hz poses with Gaussian tracking noise on every hinge and estimates each
frame ``lag`` poses behind the newest. Reported: RMS error of the activations
(all muscles and frames) and of the shoulder-elevation and elbow torques, and
the median update cost. ``qacc = 0`` is the quasi-static alternative to waiting
for accelerations::

    python benchmarks/delay_accuracy.py
"""

from __future__ import annotations

import argparse

import numpy as np
from myosuite import make_env
from myosuite.physics.inverse_dynamics import clip_kinematics
from myosuite.physics.static_optimization import StaticOptimizer, static_optimization
from myosuite.physics.streaming_kinematics import StreamingKinematics

from myosuite_shadow import (
    ShadowModel,
    ShadowModelConfig,
    add_tracking_noise,
    reaching_motion,
)

CASES = ((7, 0), (11, 0), (7, 3), (11, 5), (15, 7), (21, 10), (31, 15))
EDGE = 30  # frames dropped at either end (window fill, clip boundary)


def main() -> None:
    """Run every window/lag case at two noise levels and print Markdown tables."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rate", type=float, default=90.0, help="tracking rate [Hz]")
    p.add_argument("--duration", type=float, default=10.0, help="motion length [s]")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = make_env("myoArmReachRandom-v0")
    model = env.unwrapped.model
    up = 20
    fine = reaching_motion(
        model, args.rate * up, args.duration, args.seed, start=model.qpos0
    )
    truth = clip_kinematics(model, fine, 1.0 / (args.rate * up))
    idx = np.arange(0, len(fine), up)
    times = idx / (args.rate * up)
    ref = static_optimization(model, truth.qpos[idx], truth.qvel[idx], truth.qacc[idx])
    dof = {model.joint(model.dof_jntid[i]).name: i for i in range(model.nv)}
    key = [dof["shoulder_elv_r"], dof["elbow_flexion_r"]]
    peak = np.abs(ref.joint_torque[:, key]).max(axis=0)
    print(
        f"myoArm, {len(idx)} frames at {args.rate:g} Hz, peak torque shoulder "
        f"elevation {peak[0]:.1f} N m, elbow {peak[1]:.1f} N m\n"
    )

    def errors(k: np.ndarray, act: np.ndarray, tau: np.ndarray) -> tuple:
        keep = (k >= EDGE) & (k < len(idx) - EDGE)
        a = np.sqrt(np.mean((act[keep] - ref.activation[k[keep]]) ** 2))
        t = np.sqrt(np.mean((tau[keep] - ref.joint_torque[k[keep]]) ** 2, axis=0))
        return a, t[key[0]], t[key[1]]

    for noise in (0.5, 1.0):
        noisy = add_tracking_noise(model, fine[idx], noise, args.seed + 1)
        print(f"Tracking noise {noise} deg per hinge\n")
        print(
            "| Window | Lag | Delay [ms] | Activation RMS | Shoulder elev. torque "
            "RMS [N m] | Elbow torque RMS [N m] | Update [ms] |"
        )
        print("|---|---|---|---|---|---|---|")
        for window, lag in CASES:
            shadow = ShadowModel(
                model, ShadowModelConfig(window=window, lag=lag, fatigue=False)
            )
            frames = [shadow.push(t, q) for t, q in zip(times, noisy)]
            frames = [f for f in frames if f is not None]
            k = np.arange(len(frames)) + window - 1 - lag
            act = np.array([f.optimization.activation for f in frames])
            tau = np.array([f.optimization.joint_torque for f in frames])
            a, ts, te = errors(k, act, tau)
            ms = np.median([f.compute_ms for f in frames])
            print(
                f"| {window} | {lag} | {1e3 * lag / args.rate:.0f} | {a:.3f} | "
                f"{ts:.3f} | {te:.3f} | {ms:.2f} |"
            )
        # Quasi-static: no delay, accelerations (and optionally velocities) dropped.
        stream, so = StreamingKinematics(model, window=7, lag=0), StaticOptimizer(model)
        act, tau = [], []
        for t, q in zip(times, noisy):
            s = stream.push(t, q)
            if s is not None:
                r = so.solve(s.qpos, s.qvel, np.zeros(model.nv))
                act.append(r.activation)
                tau.append(r.joint_torque)
        k = np.arange(len(act)) + 6
        a, ts, te = errors(k, np.array(act), np.array(tau))
        print(f"| 7, `qacc = 0` | 0 | 0 | {a:.3f} | {ts:.3f} | {te:.3f} | |\n")
    env.close()


if __name__ == "__main__":
    main()
