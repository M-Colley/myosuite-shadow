# myosuite-shadow

Live muscle activation and fatigue estimates beside an experiment, built on
[MyoSuite](https://github.com/MyoHub/myosuite).

A shadow model follows a participant's tracked poses with a musculoskeletal
model. Each pose goes through live joint kinematics, static optimization
(optionally with ground reactions) and the 3CC-r fatigue model. A fatigued
muscle is capped at `1 - MF` of its force, so the load moves to fresher
synergists and then to the reserves. Every pose, estimate and experimenter
event is logged, and a crashed session can be resumed.

Status: alpha. The estimates come from a model under a minimum-effort
assumption. They are not measurements: read [docs/limitations.md](docs/limitations.md)
before using them in a study.

## A study session

```python
import numpy as np
from myosuite_shadow import LiveShadowModel, ShadowModel, ShadowModelConfig
from myosuite_shadow.calibration import (fatigue_params, fit_fatigue_parameters,
                                         posture_at, scale_body_mass, scale_strength,
                                         strength_from_mvc)
from myosuite_shadow.ik import Keypoint, KeypointIK
from myosuite_shadow.lsl import LslSink

# 1. Calibrate: MVC at 90 deg elbow flexion, body mass, endurance holds at 30 and 50 % MVC.
qpos = posture_at(model, {"elbow_flexion_r": np.radians(90)})
strength = strength_from_mvc(model, qpos, "elbow_flexion_r", measured_torque=62.0)
# Scale the muscles the MVC measured; it says nothing about the others.
model = scale_body_mass(scale_strength(model, strength.scale, strength.agonists), 71.5)
fit = fit_fatigue_parameters([0.3, 0.5], [205.0, 74.0], F=0.01086, R=0.00225, r=4.93)
flexors = ["BIClong", "BICshort", "BRA", "BRD"]
config = ShadowModelConfig(fatigue_params=fatigue_params(model, flexors, fit), gap_load="rest")

# 2. Run: tracker keypoints -> IK -> shadow model on its own thread, streamed to LSL.
# Pronation needs points on both sides of the hand (see myosuite_shadow.ik).
points = {"shoulder": "humerus_r", "elbow": "ulna_r", "wrist": "lunate_r",
          "index_mcp": "secondmc_r", "little_mcp": "fifthmc_r", "middle_tip": "distph3_r"}
ik = KeypointIK(model, [Keypoint(name, body) for name, body in points.items()])
shadow = ShadowModel(model, config, log_dir="logs/P07", sinks=(LslSink(model),),
                     metadata={"participant": "P07", "condition": "B"},
                     calibration={"strength": strength.record(), "fatigue": fit.record()})
with LiveShadowModel(shadow) as live:
    live.mark("set_start", set=1, load_kg=2.0)
    for t, points, confidence, captured in tracker:
        pose = ik.solve(t, points, confidence)
        if pose.usable:  # unusable frames become a tracking gap
            live.submit(t, pose.qpos, flags=pose.flags, capture_unix=captured)
        if live.latest is not None:
            show(live.latest.fatigue)
```

```bash
# 3. Check the session, then bound its estimates under calibration errors.
myosuite-shadow report --session logs/P07
myosuite-shadow sensitivity --session logs/P07 --out logs/P07/sensitivity
```

After a crash, `ShadowModel.resume("logs/P07")` continues the session from
its last checkpoint. The time it was down counts as a tracking gap.

## What it does

| Need | Here |
|---|---|
| Fatigue that recovers in breaks | Untracked time longer than `max_gap` is integrated as rest (`gap_load="rest"`) or as the last load held (`"hold"`); the first estimate after it is flagged |
| Per-participant calibration | `strength_from_mvc` and `scale_strength` (one MVC scales every muscle's peak force; exact, since MuJoCo muscle forces are proportional to it); `scale_body_mass`; `endurance_time` and `fit_fatigue_parameters` fit 3CC-r F (and R) to sustained holds |
| Tracker input | `KeypointIK` (mink): keypoints onto bodies, sites or geoms, joint ranges as limits, joint couplings as hard constraints; 1-1.5 ms per myoArm frame, 3-5 ms with 3 mm keypoint noise |
| Quality flags | Per pose and estimate: `LOW_CONFIDENCE`, `MISSING`, `IK_RESIDUAL`, `JOINT_LIMIT`, `GLITCH`, `AFTER_GAP`, `CAPACITY_EXCEEDED`, `NOT_SOLVED` (`myosuite_shadow.Flag`) |
| Time sync | Receive and capture wall-clock times per pose; frame and completion times per estimate; `mark()` events; `LslSink` puts estimates and markers on the LSL clock, stamped at their frame's capture time, for LabRecorder |
| Real time | `LiveShadowModel` never blocks the tracker. Behind schedule it optimizes only the newest pose; on overflow it drops only poses and counts them. `estimate_every` decimates optimizations |
| Crash safety | Rows flushed as written; `checkpoint.json` (fsynced, atomic) every `checkpoint_every` s; `ShadowModel.resume` |
| Reproducibility | `uv.lock` pins every dependency, including the MyoSuite commit. The record stores package commits (and dirty checkouts), the model hash, settings, calibration and resumes. Replaying `inputs.csv` is bit-exact |
| After the session | `report` (gaps, flags, reserves, latency, drops, provenance warnings); `sensitivity` (uncertainty bands from strength x0.8/x1.25, F and R x0.5/x2, other windows); `myosuite_shadow.validation` (XDF import, EMG envelope and median frequency, residual capacity at MVC probes, Lin's concordance, repeated-measures correlation) |

## What lives here and what lives in MyoSuite

MyoSuite has anything that computes something from a model and a motion,
offline or live. This repository has what runs a session: the streaming
pipeline, calibration, IO, logs and the analysis around a study. They can
change from one study to the next without touching MyoSuite's API or parity
tests.

| MyoSuite (`myosuite.physics`, `myosuite.core`) | myosuite-shadow |
|---|---|
| Inverse dynamics of clips ([#635](https://github.com/MyoHub/myosuite/pull/635)) | `ShadowModel`, `LiveShadowModel` |
| Static optimization ([#636](https://github.com/MyoHub/myosuite/pull/636)) and ground reactions ([#639](https://github.com/MyoHub/myosuite/pull/639)) | Calibration, tracker IK, quality flags |
| Live joint kinematics, `StreamingKinematics` ([#637](https://github.com/MyoHub/myosuite/pull/637)) | Session logs, events, checkpoints, LSL |
| Per-frame activation bounds, `solve(max_activation=)` ([#640](https://github.com/MyoHub/myosuite/pull/640)) | `replay`, `report`, `sensitivity`, validation statistics |
| 3CC-r fatigue, `CumulativeFatigue`; computed muscle control ([#644](https://github.com/MyoHub/myosuite/pull/644)) | Benchmarks |

## Install

Python 3.10 or newer; CI tests 3.12 and 3.15 on Linux and Windows. Extras: `ik` (mink), `lsl` (pylsl), `validation` (pyxdf,
pingouin), `all`.

```bash
pip install "myosuite-shadow[all] @ git+https://github.com/M-Colley/myosuite-shadow"
```

**For a study, freeze the environment.** The MyoSuite physics stack is still
under review (#635 to #640), so `pyproject.toml` follows its branch. Install
from the lockfile instead, so nothing moves between participants:

```bash
git clone https://github.com/M-Colley/myosuite-shadow && cd myosuite-shadow
uv sync --frozen --extra all
```

`myosuite-shadow report` warns when a session ran from a moving ref or a
checkout with uncommitted changes.

On Python 3.15, `h5py` (a MyoSuite dependency) has no wheels yet and builds
against the HDF5 library (`HDF5_DIR`); see `.github/workflows/ci.yml`.

For development: `uv venv -p 3.12 && uv pip install -e ".[test]"`, then
`uv run --no-sync pytest`.

## Replay

`myosuite-shadow replay` streams poses one at a time, as a tracker would. It
prints the delay, the update cost, the solver status count and the peak `MF`.

```bash
# Random minimum-jerk reaches on myoArm at 90 Hz with 0.5 deg tracking noise; no data needed
myosuite-shadow replay --log-dir logs/demo
# A subject model (MJCF or MJB) instead of a registered env
myosuite-shadow replay --model subject.xml --log-dir logs/subject
# A motion clip (NPZ with qpos, joint names, frequency); --contacts for standing and walking
myosuite-shadow replay --env myoMimicFullbody-v0 --clip walk.npz --contacts --log-dir logs/walk
# A recorded session with its own model, poses, loads and settings, here with a longer window
myosuite-shadow replay --session logs/P07 --window 21 --log-dir logs/P07_w21
```

## Session logs

| File | Content |
|---|---|
| `record.json` | Creation time, Python and platform, versions and sources of the packages (git commit and dirty flag for VCS and editable installs), the model's SHA-256, mass and joint layout, the settings, the calibration, your `metadata`, and every resume |
| `estimates.csv` | Per estimate: `time` (estimated frame), `latest_time` (newest pose), `frame_unix`, `done_unix`, `compute_ms`, `status`, `flags`, `gap_s` (untracked time integrated before it), `skipped`, `max_abs_reserve` (joints), `max_abs_residual` (free root), `a_<actuator>`, `mf_<actuator>`, `contact_<body>_{x,y,z}` [N] |
| `inputs.csv` | Every pose: `time`, `receive_unix`, `capture_unix`, `flags`, `qpos_i`, known external `qfrc_ext_i` |
| `events.csv` | `unix`, `session_time`, `label`, `data` (JSON) |
| `checkpoint.json` | Pipeline state and file offsets for `resume`; `closed` after a clean close |
| `model.mjb` | The compiled model (`save_model=False` skips it) |

`read_session(log_dir)` returns everything as arrays. `Session.load_model`
checks the stored model against the record's hash.

## Benchmarks

Both scripts run in under a minute on one CPU core. The numbers below come
from Windows, Python 3.10, MuJoCo 3.17, OSQP 1.1.3.

### Delay against accuracy

`python benchmarks/delay_accuracy.py`: myoArm minimum-jerk reaches (0.5 to
0.9 s, 0.3 s holds) tracked at 90 Hz. The reference is static optimization on
the noise-free kinematics of the 1.8 kHz motion. Activation RMS over all 50
muscles and frames:

| Delay | Window | 0.5 deg noise | 1.0 deg noise |
|---|---|---|---|
| 0 ms | 7 | 0.40 | 0.47 |
| 0 ms | 11 | 0.32 | 0.38 |
| 33 ms | 7 | 0.25 | 0.33 |
| 56 ms | 11 | 0.12 | 0.19 |
| 78 ms | 15 (default) | 0.074 | 0.12 |
| 111 ms | 21 | 0.050 | 0.073 |
| 167 ms | 31 | 0.055 | 0.061 |
| 0 ms, `qacc = 0` | 7 | 0.16 | 0.19 |

Noise in the accelerations dominates the error, so use the longest delay the
study can tolerate. If there can be no delay at all, dropping the
accelerations (quasi-static) works better than a short fit. One update costs
about 1 ms on myoArm. The script also prints the torque errors.

### Fatigue

`python benchmarks/fatigue_hold.py`: MyoSuite's elbow (six muscles) holds the
forearm at 69 deg for two minutes, estimated at 20 Hz:

| Load | Cap | Long biceps, start / 120 s | Brachialis, start / 120 s | Peak MF | Max excess over `1 - MF` | Reserve at 120 s |
|---|---|---|---|---|---|---|
| 20 N m | yes | 0.51 / 0.31 | 0.33 / 0.42 | 0.69 | 0.00 | 1.8 N m |
| 20 N m | no | 0.51 / 0.51 | 0.33 / 0.33 | 0.69 | 0.19 | 0 |
| 40 N m | yes | 0.92 / 0.28 | 0.60 / 0.29 | 0.72 | 0.00 | 27.4 N m |

With the cap, the fatigued long head drops to what it has left and the
brachialis and short head take over. At 40 N m every flexor saturates and the
reserve grows: in model terms the load cannot be held any longer. Without the
cap, muscles exceed their remaining capacity by up to 0.19 (0.64 at 40 N m).

### Full body

On a retargeted KIT walk (MyoFullBody at 100 Hz, with ground reactions; the
clip is not included here) an update takes 15 ms (median), with a p99 of 56 ms.
With `LiveShadowModel` the kinematics see every pose and the optimization
should run at about 60 Hz on the newest one (from the update time; not yet
measured live). Ground reactions are only
estimated against world-fixed bodies (floor, chair, table).

## Citing

See [`CITATION.cff`](CITATION.cff). Please also cite MyoSuite (Caggiano et al.,
L4DC 2022), the 3CC-r fatigue model (Looft, Herkert and Frey-Law, J. Biomech.
2018), static optimization (Crowninshield and Brand, J. Biomech. 1981;
Anderson and Pandy, J. Biomech. 2001), and mink and LabStreamingLayer if you
use the IK or LSL parts.

## License

Apache 2.0, like MyoSuite.
