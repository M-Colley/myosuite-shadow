# Run and learn across studies

Shadow can act as a persistent companion to an experiment: record its protocol,
trial boundaries and condition assignments alongside continuous acquisition,
produce quality-aware trial results, and retain those results in a local catalog.
The catalog supports descriptive condition comparisons and checks whether a
declared dataset is eligible for a later learning experiment. It does not train
or silently replace a participant model.

Start with the [study session](study_session.md) and [preflight](preflight.md)
workflows. A study definition adds consistent identities and analysis contracts;
it does not make unsupported motions observable. Hardware versions, inferred
legs, camera calibration, unknown loads and missing reference measurements still
limit which outputs a particular study can support. A study can declare no model
outcomes and still retain its trial journal and operational quality evidence.

## Freeze a protocol and run identity

Copy [the elbow example](../examples/studies/elbow_protocol.json), then review its
conditions, trial order, outcome selectors and task-specific quality thresholds.
`tau_0` is a model-specific scalar DOF, not a universal elbow identifier. Verify its
joint, sign and frame against the prepared model. The example is a configuration
template, not participant data or a validated exertion protocol.

```powershell
myosuite-shadow-studies init study.json runs/P01_A.sqlite --participant P01 --run-id P01_A --session-id day1
myosuite-shadow-studies status runs/P01_A.sqlite
myosuite-shadow-track --config tracking.json --study-run runs/P01_A.sqlite --quest-port 9763 --host 0.0.0.0 --log-dir logs/P01_A
```

`init` exclusively creates a durable SQLite journal containing the resolved
definition and its hash. Study, participant, run and session identities are
explicit. Run IDs are unique within a study; session IDs identify captures within
a participant. Use pseudonymous IDs. If participants are intentionally linked
across studies, supply a controlled `--participant-link-key` consistently; no identity is
inferred from names, images or body measurements. A missing link is unresolved
cross-study independence, not proof of a new participant.

Acquisition binds the journal to its intended session and writes the same frozen
context in `record.json`. It runs the existing preflight before opening capture;
blocked configurations fail before binding or logging. The definition can require
`model_sha256`, `tracking_config_sha256` and named `required_calibrations` under
`requirements`. These are exact identities/presence checks, not calibration
accuracy measurements. The full preflight result remains in session metadata.
Warnings stay visible. Offline canonical frames use the same `--study-run`
option with `--frames`; the source-specific capture path remains unchanged.

Conditions are explicit metadata, including physical units. They do not apply a
force: the experiment's actual load must still reach the existing acquisition or
external-load provider. Freeze the resolved trial order for each run, including
the output of any separately reviewed randomization/counterbalancing procedure.
This first schema executes that explicit order; it does not invent randomization.

## Record trial transitions from the experiment

Use the SDK in the task application, mapping its timestamps to the same session
axis used by tracking before calling it:

```python
from myosuite_shadow.study_protocol import TrialJournal

run = TrialJournal.open("runs/P01_A.sqlite")
run.start("unloaded_1", time_s=0.0)
run.pause(time_s=12.0)
run.resume(time_s=15.0)
run.end(time_s=25.0)
run.start("loaded_1", time_s=30.0)
run.end(time_s=55.0)
```

Numbers above illustrate session seconds; they are not collection instructions.
`start`, `pause`, `resume`, `end` and `abort` record atomic transitions. Wrong
order, repeated completion and backwards/nonfinite times fail without appending
events. Every event retains its occurrence time separately from receipt Unix
time. Reopening a journal reconstructs its state; incomplete active segments
remain incomplete, and their endpoints are not fabricated after a crash.

For annotations or explicit external control, the same operations are available:

```powershell
myosuite-shadow-studies trial runs/P01_A.sqlite start --trial-id unloaded_1 --time 0
myosuite-shadow-studies trial runs/P01_A.sqlite end --time 25
```

The CLI does not measure an event's time for you. Journal pause excludes that
interval from trial outcomes; acquisition and physiological state continue to
follow the actual recorded motion/load and elapsed time. Journal recovery does
not replace [acquisition recovery](study_session.md#monitor-and-recover).

## Get automatic trial results

After capture closes:

```powershell
myosuite-shadow-studies report runs/P01_A.sqlite reports/P01_A.json
myosuite-shadow-studies catalog-add studies.sqlite reports/P01_A.json
myosuite-shadow-studies catalog-list studies.sqlite
myosuite-shadow-studies compare studies.sqlite required_elbow_torque unloaded loaded --study-id elbow_hold_example
```

Reporting preserves condition values, trial state, active duration, supported
duration, raw/rejected frames, rejection reasons, inference exposure, estimate
flags and per-outcome coverage. Missing tracking evidence, unreadable decisions,
open sessions and insufficient coverage cannot silently become usable trials.
The denominator includes active intervals with no observations. A completed
process or trial is distinct from usable evidence.

Means and RMS integrate piecewise linear signals between adjacent eligible
estimates. They never extrapolate to trial boundaries or bridge pauses, flagged
samples or gaps exceeding `max_gap_s`. `peak` means maximum absolute magnitude
over supported segments. Each result records its statistic, units, semantic
target, coordinate frame, normalization and method version. Scalar hinge/slide
required torque/force and reserves, estimated activation, latent fatigue and
world-frame estimated contacts remain separate quantity classes. Supported raw
coordinate frames are `model_joint`, `model_actuator` and `model_world`, respectively.
Only `normalization: "none"` is implemented; no transform is guessed. Resolved model
coordinate names, analysis source hashes and implementation versions also enter
compatibility checks. Different naming conventions need a reviewed mapping.
EMG and instrument agreement retain their [existing reference workflow](independent_references.md).

Quality status describes acquisition and estimation coverage. Each declared
outcome separately reports its finite coverage and exclusion; a usable capture
can lack a particular outcome. Catalog comparisons exclude such missing outcomes.

The report hashes its session evidence, records capture/analysis versions and
the exact trial events, and rejects files or journals that change during its
read. Reports are written to new files; catalog insertion is immutable and
idempotent. A conflicting report for an existing run is rejected. Reprocessing
is a separate analysis version and must not duplicate the same session/trials
as additional people.

The CLI exits 1 for ineligible learning data or a comparison without usable pairs,
and 2 for invalid inputs. Successful report creation is not scientific acceptance.

## Compare people and decide what can teach a future model

Comparison reduces eligible repeated trials within each participant and study
to condition means and paired differences (`condition_b - condition_a`). It
averages paired study contrasts once per linked person before comparing people.
Conditions observed only in different studies cannot fabricate a pair. People,
trials and exclusions stay distinct. No frames are treated as independent
participants, and no automatic significance or causal claim is produced.
Pooling requires matching semantic outcome contracts and condition definitions;
matching names or units alone are insufficient. Multiple-study comparisons
require explicit participant linkage, so aliases cannot quietly inflate the
participant count. Differences in source fidelity, model construction or
methodology still require scientific review of the declared contract.

`myosuite-shadow-studies learning-check learning_manifest.json` checks a declared
artifact DAG, input/fit ancestry, target kinds and frozen participant-grouped
train/tune/test assignments. Hash aliases and derived artifacts cannot evade
split membership. Estimates/latent states cannot be relabelled as empirical
force or fatigue truth. Reusing a measurement as an input or fitting parameter
prevents its use as an independent validation target. EMG remains a named proxy.
Shared software/configuration nodes are infrastructure, while their participant
data ancestry is still checked. EMG aliases cannot silently become torque labels.
Missing ancestry or participant linkage remains unresolved. This checks the
declared lineage, not authenticity or completeness of an operator's declarations.

Run an incoming study against a frozen model before using eligible data for a
new candidate. Keep the new candidate, dataset/split hashes, transformations and
held-out evaluation as a separate version; promote it only after independent
evaluation and review. Never rewrite previous study conclusions or tune on the
evaluation set. The existing [recording regression](recording_regression.md)
provides a reproducible failure corpus as real recordings become available.

The next platform increments are task-specific capability profiles, richer
timed-event adapters, operational dashboards and an offline candidate evaluation
runner. This foundation supplies their durable identities and evidence boundary.
The repository fixtures remain synthetic; real participant accuracy still needs
actual recordings and independent measurements.
