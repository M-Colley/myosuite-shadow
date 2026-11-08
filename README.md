# myosuite-shadow

Live muscle activation and fatigue estimates beside an experiment, built on
[MyoSuite](https://github.com/MyoHub/myosuite).

A shadow model follows a participant's tracked poses with a musculoskeletal
model. Each pose goes through live joint kinematics, static optimization
(optionally with ground reactions) and the 3CC-r fatigue model, and every
estimate is logged for later analysis. A fatigued muscle is capped at
`1 - MF` of its force, so the load moves to fresher synergists and then to the
reserves.

```python
from myosuite_shadow import ShadowModel, ShadowModelConfig

with ShadowModel(model, ShadowModelConfig(window=15), log_dir="logs/P07",
                 metadata={"participant": "P07", "condition": "B"}) as shadow:
    for t, qpos in tracker:  # IK of each tracked pose onto the subject model
        frame = shadow.push(t, qpos)
        if frame is not None:  # estimate of the pose `lag` frames back
            frame.optimization.activation, frame.fatigue, frame.delay
```

Status: alpha. The estimates come from a model under a minimum-effort
assumption. They are not measurements.

## What lives here and what lives in MyoSuite

MyoSuite has anything that computes something from a model and a motion,
offline or live. This repository has what runs a session: the streaming
pipeline, the logs and the replay tools. They can change from one study to the
next without touching MyoSuite's API or parity tests.

| MyoSuite (`myosuite.physics`, `myosuite.core`) | myosuite-shadow |
|---|---|
| Inverse dynamics of clips ([#635](https://github.com/MyoHub/myosuite/pull/635)) | `ShadowModel`: kinematics, static optimization and fatigue chained on a live stream, with tracking gaps and known external loads |
| Static optimization ([#636](https://github.com/MyoHub/myosuite/pull/636)) | Session logs with provenance (`record.json`, `estimates.csv`, `inputs.csv`, `model.mjb`) and `read_session` |
| Live joint kinematics, `StreamingKinematics` ([#637](https://github.com/MyoHub/myosuite/pull/637)) | `myosuite-shadow replay` for synthetic reaches, motion clips and recorded sessions |
| Ground reactions in static optimization ([#639](https://github.com/MyoHub/myosuite/pull/639)) | Synthetic reaches and tracking noise |
| Per-frame activation bounds, `solve(max_activation=)` ([#640](https://github.com/MyoHub/myosuite/pull/640)) | Benchmarks: delay against accuracy, fatigue hold |
| 3CC-r fatigue, `CumulativeFatigue` | |
| Computed muscle control ([#644](https://github.com/MyoHub/myosuite/pull/644)) | |

## Install

Python 3.10 or newer. The MyoSuite physics stack is still under review
(#635 to #640), so the dependency points at that branch until it is released:

```bash
pip install "myosuite-shadow @ git+https://github.com/M-Colley/myosuite-shadow"
```

For development:

```bash
git clone https://github.com/M-Colley/myosuite-shadow && cd myosuite-shadow
uv venv -p 3.10 && uv pip install -e ".[test]"
uv run --no-sync pytest
```

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

With `log_dir`, a session writes:

| File | Content |
|---|---|
| `record.json` | Creation time, Python and platform, versions and sources of `myosuite-shadow`, `myosuite`, `myo-sim`, `mujoco`, `osqp` and `numpy` (git commit for VCS and editable installs), the model's SHA-256 and joint layout, the settings and your `metadata` |
| `estimates.csv` | One row per estimate: estimated frame time, newest pose time, update cost [ms], solver status, peak reserve, activation `a_<actuator>`, fatigue `mf_<actuator>`, and `contact_<body>_{x,y,z}` [N] for `log_contact_bodies` |
| `inputs.csv` | Every pose pushed (`time`, `qpos_i`, known external `qfrc_ext_i`) |
| `model.mjb` | The compiled model (`save_model=False` skips it) |

Rows are flushed as they are written. Floats use Python's shortest round-trip
representation, so replaying `inputs.csv` with the recorded settings reproduces
the estimates bit for bit (tested). `read_session(log_dir)` returns everything
as arrays, and `Session.load_model` checks the stored model against the
record's hash.

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
That is about 60 Hz live. Ground reactions are only estimated against
world-fixed bodies (floor, chair, table).

## Citing

See [`CITATION.cff`](CITATION.cff). Please also cite MyoSuite (Caggiano et al.,
L4DC 2022), the 3CC-r fatigue model (Looft, Herkert and Frey-Law, J. Biomech.
2018) and static optimization (Crowninshield and Brand, J. Biomech. 1981;
Anderson and Pandy, J. Biomech. 2001).

## License

Apache 2.0, like MyoSuite.
