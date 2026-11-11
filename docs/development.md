# Installing and developing study support

The default dependency is the published MyoSuite source commit
[`751057ba63cb09c4843c1973eaa81e658b27f50a`](https://github.com/MyoHub/myosuite/commit/751057ba63cb09c4843c1973eaa81e658b27f50a),
on `codex/study-shadow-support`. It provides the required
`myosuite.biomechanics.calibration`, `external_loads`, `streaming_kinematics`
and `camera` interfaces. This source pin is not an official MyoSuite release.
The shadow package remains alpha; software checks do not establish participant
accuracy.

## Frozen installation and local checks

A sibling model checkout is not required. From `myosuite-shadow`:

```bash
uv sync --frozen --extra test --python 3.12
uv lock --check
uv run --no-sync python -m pytest
uv run --no-sync python -m ruff check src tests benchmarks
uv run --no-sync python -m ruff format --check src tests
```

For study use without test tools, select `--extra all` instead. Standard package
metadata also supports `python -m pip install ".[all]"`: it resolves the same
MyoSuite Git commit, but other dependencies are not frozen by that pip command.

Archive the shadow revision, lockfile, prepared model, acquisition configuration
and protocol. Session provenance records dependency/source commits and dirty
state. `myosuite-shadow` runs replay/report tools, `myosuite-shadow-track` runs
acquisition and `myosuite-shadow-validate` evaluates independent probes.

## Optional editable MyoSuite development

Keep model math in MyoSuite and acquisition, study protocols and logs in shadow.
An optional sibling checkout can use this layout:

```text
parent/
  myosuite-shadow/
  myosuite-study-support/
```

Start that checkout from the pinned commit, then override only the installed
model package after synchronizing the frozen environment:

```bash
git clone https://github.com/MyoHub/myosuite.git ../myosuite-study-support
git -C ../myosuite-study-support checkout 751057ba63cb09c4843c1973eaa81e658b27f50a
uv pip install --no-deps --editable ../myosuite-study-support
uv run --no-sync python -m pytest
uv run --no-sync python -m pytest ../myosuite-study-support/myosuite/tests/test_subject_calibration.py ../myosuite-study-support/myosuite/tests/test_streaming_kinematics.py ../myosuite-study-support/myosuite/tests/test_camera_reconstruction.py
```

Use `uv run --no-sync` while editing. The next `uv sync` restores the published
pin; the editable override does not change the committed dependency or lock.
`--no-deps` assumes the already frozen dependencies remain compatible. Review
dependency changes explicitly.

Before a study uses edited model code, commit/archive its revision and local
changes, then update the published pin and lock together. Editable source files
are not frozen by `uv.lock`. Avoid silently mixing participants across model
revisions.

## CI and dependency updates

CI uses Ubuntu 24.04 and Windows 2025 with Python 3.12 and 3.15. The runner images
are pinned because their Conda installations supply HDF5 1.14 for source-built
Python 3.15 dependencies. Linux exposes that prefix through `LD_LIBRARY_PATH`;
Windows registers its DLL directory before importing h5py. The required-interface
step imports both h5py and pylsl, so missing native libraries cannot silently
reduce the all-extras test coverage. HDF5 provisioning does not depend on APT
mirrors, and the Linux setup step has a bounded duration.

The Checks workflow has defaults for the published repository and exact commit;
no Actions variables are required for the normal pinned build. It installs with
`uv sync --frozen --extra test`, verifies the installed MyoSuite commit, and
checks out the same core commit separately to supply its regression tests.
It tests required interfaces, shadow and the targeted model regressions.

Optional `MYOSUITE_REPOSITORY`/`MYOSUITE_REF` Actions variables or
`workflow_dispatch` inputs can select another prerequisite for an update, but
they must match the source pin and lock. Changing only the workflow reference
does not replace the package installed from the lock; the revision check catches
a mismatch. Use a full 40-character commit SHA.

To update MyoSuite, publish and review the core commit, update shadow's Git
dependency and CI default together, regenerate `uv.lock`, and verify the frozen
installation. Once the interfaces ship in an official release, replace the Git
pin with that release and regenerate the lock. Local results below do not imply
that remote CI has passed.

## Local verification baseline

Shadow's final local suite passed **158 tests**, with Ruff lint/format and
`uv lock --check` passing. The publication audit found no credentials, generated
binaries, participant recordings or model weights among the candidate shadow
changes. Full and targeted MyoSuite pre-commit hooks passed before the core commit
was pushed.

The combined MyoSuite checks on Windows, Python 3.12.7, MuJoCo 3.17.0 and NumPy
2.5.4 produced **344 passed, 4 expected failures and 5 failures**, including the
26 study-support tests. The five failures are four parity mismatches and a
registry check requiring unavailable KIT clips 971/972. All five were reproduced
against the unchanged original `dev` checkout at
`87f8ac4adb3498cefbbbb6b0fcabb76d16475d4f`, with tracked files clean and imports
verified to come from that original checkout. No study-support regression was
observed in those comparisons; the baseline/platform/data failures remain
unresolved. Frozen baselines and tolerances were retained.

The four parity node IDs are in `myosuite/tests/test_parity.py`:

| Test | First mismatch | Difference and tolerance |
|---|---|---|
| `test_parity[myoChallengeBaodingP2-v1]` | Step 34 reward | `1.0183083e-6`, tolerance `1e-6` |
| `test_parity[myoChallengeRelocateP1-v0]` | Step 4 observations | Maximum violating difference `0.00034794`, tolerance `0.0001` |
| `test_parity[myoHandReorient8-v0]` | Step 42 observations | Maximum violating difference `2.19520569`, tolerance `0.01` |
| `test_parity[myoLegRoughTerrainWalk-v0]` | Step 1 observations | Maximum violating difference `0.02777147`, tolerance `0.001` |

The missing dataset is an environment/data prerequisite, not a passing registry
check. Keep these limitations alongside local results and check the actual
remote workflow results separately.
