"""Command line: ``myosuite-shadow replay | report | sensitivity``.

``report --session DIR`` prints the session's quality report (and ``--json``
writes it); ``sensitivity --session DIR --out DIR`` replays the session under
calibration errors (see :mod:`myosuite_shadow.sensitivity`).

``replay``

Streams poses one at a time through :class:`~myosuite_shadow.ShadowModel`, as
a tracker would, and prints the delay, update cost, solver status count and
peak ``MF``. The poses come from one of

* random minimum-jerk reaches on a fixed-base model (default; no data needed),
* ``--clip``: a motion NPZ (qpos, joint names, frequency),
* ``--session``: a recorded session's ``inputs.csv``, replayed with its own
  settings and model unless overridden, e.g. with a longer window::

    myosuite-shadow replay --log-dir logs/demo
    myosuite-shadow replay --env myoMimicFullbody-v0 --clip walk.npz --contacts
    myosuite-shadow replay --session logs/P07 --window 21 --log-dir logs/P07_w21
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np

from myosuite_shadow.log import MODEL, read_session
from myosuite_shadow.motions import add_tracking_noise, reaching_motion
from myosuite_shadow.pipeline import ShadowFrame, ShadowModel, ShadowModelConfig
from myosuite_shadow.report import format_report, session_report
from myosuite_shadow.sensitivity import DEFAULT_VARIANTS, Variant, run_sensitivity

_DEFAULT_ENV = "myoArmReachRandom-v0"


def _load_model(path: Path) -> mujoco.MjModel:
    if path.suffix.lower() == ".mjb":
        return mujoco.MjModel.from_binary_path(str(path))
    return mujoco.MjModel.from_xml_path(str(path))


def _env_model(env_id: str) -> mujoco.MjModel:
    from myosuite import make_env

    env = make_env(env_id)
    model = env.unwrapped.model
    env.close()
    return model


def _replay_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("replay", help="stream a motion through the shadow model")
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--env", help=f"registered env whose model to use ({_DEFAULT_ENV})"
    )
    src.add_argument("--model", type=Path, help="MJCF (.xml) or MJB model file")
    motion = p.add_mutually_exclusive_group()
    motion.add_argument("--clip", type=Path, help="motion NPZ (qpos, joint names, Hz)")
    motion.add_argument("--session", type=Path, help="recorded session to replay")
    p.add_argument("--rate", type=float, default=90.0, help="reach pose rate [Hz]")
    p.add_argument("--duration", type=float, default=10.0, help="reach length [s]")
    p.add_argument(
        "--noise-deg",
        type=float,
        help="tracking noise on hinges [deg] (0.5; 0 for sessions)",
    )
    p.add_argument("--window", type=int, help="poses per kinematic fit (15)")
    p.add_argument("--lag", type=int, help="frames of delay (window // 2)")
    p.add_argument(
        "--contacts",
        action=argparse.BooleanOptionalAction,
        help="estimate ground reactions",
    )
    p.add_argument(
        "--fatigue", action=argparse.BooleanOptionalAction, help="track 3CC-r fatigue"
    )
    p.add_argument(
        "--realtime", action="store_true", help="pace poses at their timestamps"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", type=Path, help="write a session log here")


def _replay(args: argparse.Namespace) -> list[ShadowFrame]:
    session = read_session(args.session) if args.session is not None else None
    if args.model is not None:
        model = _load_model(args.model)
    elif args.env is not None or session is None:
        model = _env_model(args.env or _DEFAULT_ENV)
    else:
        model = session.load_model(args.session)

    external, flags = None, None
    if session is not None:
        times, qpos = session.input_time, session.input_qpos
        external = session.input_qfrc
        flags = session.input_columns["flags"].astype(int)
        base = ShadowModelConfig.from_dict(session.record["config"])
        noise = args.noise_deg or 0.0
    elif args.clip is not None:
        from myosuite.core.trajectory_io import (
            expand_motion_clip_to_model,
            load_motion_clip,
        )

        clip = expand_motion_clip_to_model(
            load_motion_clip(args.clip, model.nq, model.nv), model
        )
        rate = clip.frequency_hz or args.rate
        qpos, times = clip.qpos, np.arange(len(clip.qpos)) / rate
        base = ShadowModelConfig()
        noise = 0.5 if args.noise_deg is None else args.noise_deg
    else:
        qpos = reaching_motion(model, args.rate, args.duration, args.seed)
        times = np.arange(len(qpos)) / args.rate
        base = ShadowModelConfig()
        noise = 0.5 if args.noise_deg is None else args.noise_deg
    if noise > 0:
        qpos = add_tracking_noise(model, qpos, noise, args.seed)

    overrides = {
        k: getattr(args, k)
        for k in ("window", "lag", "fatigue")
        if getattr(args, k) is not None
    }
    if args.contacts is not None:
        overrides["optimization"] = dataclasses.replace(
            base.optimization, contact_forces=args.contacts
        )
    config = dataclasses.replace(base, **overrides)

    metadata = {
        "replay": {
            "env": args.env,
            "model": None if args.model is None else str(args.model),
            "clip": None if args.clip is None else str(args.clip),
            "session": None if args.session is None else str(args.session),
            "noise_deg": noise,
            "seed": args.seed,
        }
    }
    frames, wall0 = [], time.perf_counter()
    with ShadowModel(model, config, log_dir=args.log_dir, metadata=metadata) as shadow:
        for k, (t, q) in enumerate(zip(times, qpos)):
            if args.realtime:
                time.sleep(max(0.0, t - times[0] - (time.perf_counter() - wall0)))
            frame = shadow.push(
                float(t),
                q,
                None if external is None else external[k],
                flags=0 if flags is None else int(flags[k]),
            )
            if frame is not None:
                frames.append(frame)
    return frames


def _summary(frames: list[ShadowFrame], log_dir: Path | None) -> str:
    if not frames:
        return "no estimates: the motion is shorter than one kinematic window"
    ms = np.array([f.compute_ms for f in frames])
    solved = sum(f.optimization.status == "solved" for f in frames)
    rate = (len(frames) - 1) / max(frames[-1].state.time - frames[0].state.time, 1e-9)
    mf = max(
        (float(f.fatigue.max()) for f in frames if f.fatigue is not None), default=0.0
    )
    line = (
        f"{len(frames)} frames at {rate:.3g} Hz, delay {1e3 * frames[0].delay:.0f} ms, "
        f"update {np.median(ms):.2f} ms (p99 {np.percentile(ms, 99):.2f}), "
        f"solved {solved}/{len(frames)}, peak MF {mf:.3f}"
    )
    return line + ("" if log_dir is None else f"; log in {log_dir}")


def _analysis_parsers(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("report", help="quality report of a recorded session")
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--json", type=Path, help="also write the report as JSON")
    p = sub.add_parser("sensitivity", help="replay a session under calibration errors")
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument(
        "--window", type=int, action="append", default=[], help="also try this window"
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point of ``myosuite-shadow``."""
    parser = argparse.ArgumentParser(
        prog="myosuite-shadow", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _replay_parser(sub)
    _analysis_parsers(sub)
    args = parser.parse_args(argv)
    if args.command == "report":
        rep = session_report(args.session)
        if args.json is not None:
            args.json.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(format_report(rep), end="")
        return
    if args.command == "sensitivity":
        variants = (
            *DEFAULT_VARIANTS,
            *(Variant(f"window {w}", window=w) for w in args.window),
        )
        res = run_sensitivity(args.session, variants, out_dir=args.out)
        for v, mf in zip(res.variants, res.fatigue):
            print(f"{v.name:>20}: peak MF {mf.max():.3f}, final MF {mf[-1].max():.3f}")
        print(f"summary.csv and bands.csv in {args.out}")
        return
    if args.session is not None and args.model is None and args.env is None:
        if not (args.session / MODEL).exists():
            parser.error(f"{args.session} has no {MODEL}; pass --model or --env")
    frames = _replay(args)
    print(_summary(frames, args.log_dir))


if __name__ == "__main__":
    main()
