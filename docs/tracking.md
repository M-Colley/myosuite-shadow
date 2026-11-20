# Quest and Sapiens2 acquisition

See [Running a participant session](study_session.md) for the subject model,
study quality policy, tracking command and session recovery.

The acquisition path is `source -> TrackerFrame -> Registration -> study quality
gate -> KeypointIK -> ShadowModel`. Raw frames are recorded before IK; rejected
frames remain available for audit and reprocessing. `TrackerFrame` accepts finite
3D positions in metres, xyzw unit quaternions and explicit source/person identity.
It keeps valid/inferred/tracked states separate. Source confidence is a quality
score, not a calibrated probability of biomechanical accuracy.

## Quest: use the body API skeleton

`examples/quest/QuestBodySender.{h,cpp}` is a native component for an existing
OpenXR application. It reads `xrLocateBodyJointsFB` directly, before avatar
retargeting, animation blending or locomotion synthesis. It sends one JSON body
packet per update and answers timestamp pings. No GPU model or credentials are
needed in the Python receiver.

Enable `XR_FB_body_tracking` and body tracking permission in the Android manifest:

```xml
<uses-feature android:name="com.oculus.software.body_tracking" />
<uses-permission android:name="com.oculus.permission.BODY_TRACKING" />
```

Enumerate extensions before creating the instance. Enable
`XR_KHR_convert_timespec_time` on Android (`XR_KHR_win32_convert_performance_counter_time`
for Windows/Link). Enable `XR_META_body_tracking_full_body` and
`XR_META_body_tracking_fidelity` only when advertised. Pass those **enabled**
extension booleans to the sender. Add the example's CMake subdirectory to your
native app, link `shadow_quest_sender`, and instantiate after creating your session
and a STAGE space:

```cpp
QuestBodySender sender(instance, systemId, session, stageSpace,
                       fullBodyExtensionEnabled, fidelityExtensionEnabled,
                       "192.168.1.20", "P07", "Quest 3", "<your SDK version>");
// Each app update:
sender.Send();
// In the existing xrPollEvent loop, forward reference-space-change events:
sender.ReferenceSpaceChanged(referenceSpaceChangeEvent);
```

The sender queries full-body capability and selects 70 upper-body or 84 full-body
joints. Quest 2/Pro/3/3S and newer SDK versions use that capability negotiation;
the decoder never guesses support from the device's marketing name. Actual
fidelity is recorded separately from requested High fidelity. A low-fidelity
fallback must be considered by the study acceptance policy.

Joint names are in `QUEST_UPPER_JOINTS` / `QUEST_FULL_JOINTS`. Upper-arm joint origins
map to `left_shoulder` / `right_shoulder`, lower-arm origins to `*_elbow`, and hand
wrist origins to `*_wrist`; clavicle/shoulder bones map to `*_collar`. Model sites
still need anatomical landmark calibration. Finger names retain OpenXR segment
names (`right_index_proximal`, etc.). Positions are OpenXR right-handed metres,
not Unity-transformed coordinates. The sender's `stageSpace` must actually be
STAGE; use a matching registration, not an arbitrary axis swap guessed from a rig.

Full-body API output includes Generative Legs. A valid flag alone does not mean a
joint was observed. Keep estimated legs out of claims about measured leg loading
unless validated for the protocol. A Quest-only study may instead limit its
analysis to sufficiently observed upper-body landmarks. Default strict tracking
policies can reject inferred joints; relax them only through an explicit study
policy and preserve the source flags.

## Clock calibration and freshness

OpenXR `XrTime` is not Unix time. The native sender converts it using the OpenXR
time interoperability functions into Android CLOCK_MONOTONIC seconds (or Windows
performance-counter seconds). Its ordinary poses are estimates at the requested
time, not sensor capture measurements. `evaluation_unix` can locate those poses
on a wall-clock timeline; `capture_unix` remains unset. Positive prediction
horizons are labelled `predicted`.

Start the app, then measure the source-to-host mapping:

```bash
python examples/quest/receive.py sync --host 192.168.1.30 --out clock.json
python examples/quest/receive.py receive --clock clock.json --person P07 \
  --bind 0.0.0.0 --clock-host 192.168.1.30 --out tracking.jsonl
```

The clock service listens on UDP 9764; body packets go to UDP 9763. `sync` performs
30 ping exchanges across approximately six seconds, filters excessive RTT, fits
offset/rate, bounds uncertainty including asymmetric delay, and writes a mapping
that expires after 60 seconds. Source reply times and host send/receive times are
passed to `fit_clock_mapping`; no offset is invented from one packet's arrival.
The example fails explicitly on an expired map or missing stream. With
`--clock-host`, `QuestClockSynchronizer` refreshes on its worker every 20 seconds;
clock probes do not block acquisition. `QuestDecoder(clock,
clock_provider=client.get_mapping)` snapshots the immutable mapping once at each
packet boundary, and each frame records that mapping. Refresh errors, source
clock-identity changes and session-axis jumps fail explicitly. Without
`--clock-host`, the receiver is limited to the original map's validity interval.
The study tracking CLI uses the same worker when these fields are added to the
`quest` object alongside its measured `clock` mapping:

```json
{"clock_host":"192.168.1.30",
 "clock_sync":{"port":9764,"session_epoch":0,"refresh_s":20,
               "validity_s":60,"count":30,"interval_s":0.2,
               "timeout_s":0.5,"max_rtt_s":0.05,"max_uncertainty_s":0.01}}
```
Keep the same host `--session-epoch` for all refreshes (default 0 means raw host
`perf_counter` time). On source reboot, start a new source/clock epoch and
re-register rather than reusing the previous calibration.

`max_rtt_s`, `max_uncertainty_s`, `max_drift_ppm` and mapping validity are protocol
settings. The clock fit bounds all affine rate/offset mappings consistent with
the accepted send/receive intervals, including extrapolation until expiry. The
rate is assumed stable during that interval. Short calibration spans and long
validity periods can exceed the uncertainty budget even with small RTT. Use a
longer probe span, shorter validity/refresh interval or a better clock transport
when the fit rejects the budget; do not discard its uncertainty. Long or
asymmetric network delays produce uncertainty, not stronger
claims of alignment. For validation against EMG/dynamometry, check a common
recorded synchronization event as well.

## Spatial registration and recentering

Fit `fit_registration(source_landmarks, model_landmarks,
source_frame="quest_stage", revision=0)` using at least three noncollinear paired
landmarks in metres. It rejects excessive residual and reflections; it does not
fit participant segment lengths or muscle strength. Save `registration.record()`
with the session and verify the model reproduces calibration poses.

The sender forwards STAGE reference-space changes as `space_revision` changes at
the effective event time. `Registration.apply` rejects a different revision.
Supply a newly measured registration or a transform derived from the OpenXR
space-change event, and reset the kinematic differentiation window. Do the same
when skeleton proportions change. Recenter discontinuities must not be
interpreted as joint acceleration.

## Sapiens2: pixels need calibrated reconstruction

The adapter reads the official 308-keypoint aggregate export at upstream revision
`7e5bae88456ac418ff0e58e74106c9fe192055d4`:

```json
{"video":"trial", "image_size":[1080,1920], "num_keypoints":308,
 "frames":[{"image_name":"000012.png", "instances":[
   {"bbox":[0,0,1000,1000], "keypoints":[[123,456]],
    "keypoint_scores":[0.9]}]}]}
```

The example is abbreviated: each instance must contain exactly 308 pixel pairs
and 308 scores. The layout differs from COCO (hips are 9/10; wrists are 41/62).
The first 70 musculoskeletal landmarks retain upstream names; remaining detailed
face landmarks retain stable indexed names. Original scores and detection data
are preserved. Scores are clipped to [0,1] for the common confidence interface.

Run Sapiens2 in its own GPU environment following the upstream instructions.
The shadow package does not vendor its models or change its license. Import its
output with measured image timestamps and stable person assignments:

```python
from myosuite_shadow.tracking import read_sapiens2, reconstruct_multiview

views = read_sapiens2("trial_predictions.json", camera_id="front", person_id="P07",
                     timestamps={"000012.png": 1.25},
                     instance_indices={"000012.png": 0})
```

Both sidecars must cover every exported image. Inference can skip images;
enumerating successful outputs at a nominal frame rate gives incorrect times.
The detector's instance order is not a person identity. Obtain assignments from
a reviewed external identity tracker or a controlled single-participant capture;
explicitly assign `None` when that participant is absent. Ambiguous detections
must be reviewed rather than switching automatically to instance zero.

`read_sapiens2` yields `Pose2D`, which cannot enter 3D IK. Use either:

- `reconstruct_multiview(views_at_one_time, cameras)`: at least two synchronized,
  calibrated cameras in the same metric world. Identity mismatches, excessive
  skew, insufficient parallax, behind-camera points and reprojection errors are
  rejected. Individual landmark failures are retained in reconstruction metadata.
- `reconstruct_depth(view, camera, depths_m)`: registered metric **camera-Z depth**
  at each landmark. Relative monocular depth, ray distance and an uncalibrated
  pointmap are different quantities and must not be passed as metric Z depth.

Reusable `PinholeCamera`, `triangulate_point` and `lift_depth` live in MyoSuite's
`myosuite.biomechanics.camera`. Pixels must be undistorted with the matching
intrinsics; crops/resizing require updated intrinsics. Calibration translations
are metres. Register the resulting `camera_world` observations to the subject
model before IK. Both reconstruction paths are source-level pose estimates and
need validation with the actual camera placement, tasks and participants.

### Convert exports into study input

The runnable converter is `examples/sapiens2/reconstruct.py`. Copy
[config.example.json](../examples/sapiens2/config.example.json) beside your exports
and sidecars. Its numerical camera matrices are illustrative: replace them with
your measured calibration. Paths resolve relative to that configuration file.
Confirm that inference used undistorted images and record the actual inference
code commit, model identity, checkpoint revision and checkpoint SHA-256. The
parser's reference revision is recorded separately; it does not identify the
weights that produced your observations.

Each camera has a `timestamps` JSON object mapping image names to measured common
session seconds, e.g. `{"000012.png":1.25}`. Its `instances` JSON object maps image
names to the reviewed detection index for the same participant, e.g.
`{"000012.png":0,"000013.png":null}`. `null` explicitly records an absent
participant. Optional `capture_unix` sidecars have the same image keys; omit that
camera field when actual capture wall-clock times are unavailable.

```bash
python examples/sapiens2/reconstruct.py --config cameras.json --out-dir camera_metric
myosuite-shadow-track --config camera_study.json --frames camera_metric/frames.jsonl --log-dir logs/P07_camera
```

The converter consumes each image observation once. It groups at most one view
per camera within `max_skew_s` of the oldest unconsumed timestamp and uses that
oldest time as the group anchor. Original per-view times and capture timestamps
remain in the raw records. Missing/absent views produce explicit empty 3D frames
when fewer than two useful views remain; the study gate then records their
rejection. Failed landmarks are omitted with reconstruction reasons.

The new output directory contains `frames.jsonl`, `quality.jsonl` and
`manifest.json`. The manifest saves full configuration/calibration, source-file
hashes, package/source provenance, declared inference/checkpoint metadata, output
hashes, counts and completion status. Use completed conversions for analysis;
partial failed runs retain their evidence with `status="failed"`. Existing
output directories are never overwritten. Register the resulting `camera_world`
positions to your participant model following [the study workflow](study_session.md).

## Verification and hardware status

Synthetic metric reconstruction, malformed packets, clock uncertainty, source
versions, identity changes, raw records and real loopback UDP packets are tested.
The native sender compiles against official Khronos OpenXR 1.1.65 headers with
MSVC; its shared extension signatures are checked without SDK stubs. An Android
NDK build, on-device permissions, Quest quality/latency, recenter events and
Sapiens2 GPU inference still require hardware testing. No participant recording
or scientific validation is implied by those software tests.

Sources: [Meta body tracking](https://developers.meta.com/vr/documentation/unity/move-body-tracking/),
[OpenXR body locations](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrBodyJointLocationsFB.html),
[OpenXR time](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrTime.html),
[Sapiens2 pose guide](https://github.com/facebookresearch/sapiens2/blob/7e5bae88456ac418ff0e58e74106c9fe192055d4/docs/POSE.md),
[official exporter](https://github.com/facebookresearch/sapiens2/blob/7e5bae88456ac418ff0e58e74106c9fe192055d4/sapiens/pose/tools/vis/vis_pose.py).
