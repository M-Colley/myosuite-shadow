# Live full-body shadow and headset feedback

The headset streams canonical metric body landmarks to shadow and receives
region-level model estimates. The applications run in separate environments:
myosuite-headset owns capture, display and A/B/C feedback; shadow owns registration,
IK, quality gates, physics, fatigue and replay. The existing elbow mode remains.

Install with `uv sync --frozen --extra test --python 3.12`, or install the
`ik,bridge` extras. Keep the headset's own dependency environment.

## Exercise the real protocol using synthetic model landmarks

```powershell
myosuite-shadow-body init --source synthetic --out-dir studies/body-demo
myosuite-shadow-body serve --config studies/body-demo/body.json --log-dir logs/body-demo
```

In the headset application's environment:

```powershell
ff-run serve-shadow --shadow-uri ws://127.0.0.1:9766 --condition C
```

Then send model forward-kinematic landmarks to the gateway's printed URL:

```powershell
myosuite-shadow-body synthetic --config studies/body-demo/body.json --uri ws://127.0.0.1:8765 --seconds 15
```

Adjust the gateway port to its printed URL. This exercises measured clock fits,
IK and model estimation. It is synthetic evidence, not a participant recording
or a hardware latency measurement. A reference full-body model may have
unobservable coordinates: the bridge reports this and withholds physiology.

## Configure a participant

```powershell
myosuite-shadow-body init --source unity --person-id P01 --out-dir studies/P01
```

The `full_body` profile includes both arms and legs. Inspect the landmark mapping,
model anatomy and region definitions. A compatible calibrated model can replace
the generated model file; each run retains its hash. Unsupported anatomy fails
profile validation.

Match model sites to actual tracker joint definitions, scale for the participant
and declare target DOFs, nuisance-coordinate assumptions and measured task
postures in `preflight`. Sparse positions cannot identify all coordinates in a
detailed musculoskeletal model. The bridge saves the geometric preflight report
and suppresses physiological values when it is blocked.

Rigid registration requires at least three noncollinear, explicitly paired
source landmarks and corresponding model landmarks, in metres, from an explicitly
known posture. It does not calibrate segment lengths, strength or fatigue.
Create a JSON object with `source_frame`, integer `revision`, and matching
`source` / `target` dictionaries of named three-element position arrays.
Use measured values; do not invent an identity transform.

```powershell
myosuite-shadow-body register --config studies/P01/body.json --pairs studies/P01/pairs.json --out studies/P01/registered.json
myosuite-shadow-body serve --config studies/P01/registered.json --log-dir logs/P01-session1
```

Source/person/clock/reference frame must match `body_source`. Unity defaults:
`quest-unity`, `P01`, `unity_realtime`, `quest_local_floor`. WebXR defaults use
`quest-webxr` and `webxr_performance`; configure its actual negotiated reference
space, `quest_local_floor` or `quest_local`. A recenter or skeleton revision
invalidates registration. Renew it and restart with a new run directory.
A restarted source clock/sequence also requires a new run. Reconnecting does not
reset fatigue or participant identity.

Missing joints remain absent. Inferred required landmarks are rejected by
default. If a study explicitly enables `acquisition.allow_inferred`, inferred
regions remain marked and their physiological values stay unavailable for
actionable feedback. WebXR cannot attest native optical tracking per joint.

## Interpretation and timing

Regions contain an anchor, tracking status, optional activation and optional
3CC-r fatigued fraction MF. Aggregation is the maximum over the listed muscles.
MF is not measured strength remaining. Contacts and external loads are configured
assumptions, not measurements. The generated template estimates contacts and
holds the last load across tracking gaps; review these choices for each task.
The full-body policy rejects unexplained root force above 50 N or root moment
above 10 N m. These example study thresholds need task validation. Unsupported
solves use the configured gap-load assumption and retain a quality event.
Cumulative history flags and assumed-load seconds remain visible after recovery.

Muscles affected by omitted finger/toe motion are excluded from region summaries
(82 of 416 in the reference). Remaining estimates are conditional on the
regularized nuisance posture and configured load model.

`running` means computational freshness and configured gates passed; it does
not validate participant accuracy. Stale, failed, rejected and uncertain results
carry null physiological values. Tracking status is matched to the raw frame at
the estimate's delayed timestamp.

Nonce-matched low-RTT clock exchanges span at least one second, bound uncertainty
and refresh periodically. Freshness uses **client observation time** mapped by the
latest valid clock fit. Kinematics keeps the first fit's source-to-session axis
for the connection so routine clock refreshes cannot introduce artificial motion
or interrupt the derivative window. Use `evaluation_unix` to align these
observations with other recordings; the raw record retains both clock mappings.
Native pose capture age remains unverified, even if an SDK exposes a predicted
display time. `--strict-pose-timing` suppresses physiological feedback for these
sources. Hardware accuracy and complete capture-to-feedback latency need a real,
independently timed recording.

The gateway enforces A/B/C conditions on visual output and LLM input: A exposes
movement/task/quality, B adds activation, C adds fatigue. Slow LLM generation
must not block poses. Use the demo provider for integration checks.

## Records and wire protocol

Schema `myosuite-shadow-live/1` begins with `hello`, relays
`clock_probe` / `clock_reply` and accepts `body_frame` observations.
Positions are right-handed metres; every position has explicit confidence and
tracked flags. Packets include a source-clock timestamp, sequence, coordinate
frame and reference/skeleton revisions. Outputs identify session, model hash
and profile. One configured producer is accepted; pose queues are bounded.

Each fresh run directory contains `body_wire.jsonl`, `body_preflight.json`,
canonical `tracking.jsonl`, `tracking_quality.jsonl`, saved model, resolved
settings, estimates, events and checkpoints. The wire record includes packets
replaced by the latest-pose queue. Existing recording/reprocessing and study
evidence tools consume the canonical tracking and session records.

Repeat the separate-process A/B/C regression from the shadow environment:

```powershell
python examples/live_body_roundtrip.py --headset-python <headset-python-executable> --headset-repo <headset-checkout> --out-dir <new-evidence-directory>
```

It retains exact prompts/results and verifies condition fields on outputs and
prompts using the deterministic demo provider. `roundtrip.json` reports p95
model compute time and mapped observation-to-estimate time. Model compute time
excludes IK and queueing. Observation-to-estimate time includes transport, IK
and the derivative window; it ends before feedback generation or display.
