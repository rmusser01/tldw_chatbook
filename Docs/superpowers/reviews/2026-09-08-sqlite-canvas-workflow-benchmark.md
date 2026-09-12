# Paired actual Canvas workflow measurement — 2026-09-08

Status: measurement complete and independently approved for SQLite closeout.
Canvas V2 remains disabled; this is not V2 admission or merge qualification.

ADR required: no new ADR. Existing [ADR125](../../../backlog/decisions/125-lock-safe-private-sqlite-validation.md)
and the approved [SQLite design](../specs/2026-09-07-sqlite-lock-safe-private-validation-design.md)
apply unchanged. Task16 completes the actual-child timing omitted from Task7;
it does not repeat the completed app/TTS/helper benchmarks or qualify V2.

## Compared inputs

- Current product: `210831e1928177d73d278b6adfad1112a240ce81`.
- Baseline product: `9bc73ffb35ccd6eb24629bfa8021b28063dc9112`.
- Both are private Git archives under `/private/tmp/task16-canvas-benchmark.MirPEY`.
  Root verified all 13,367 current and 13,326 baseline extracted regular files.
- The baseline has exactly two current test-harness overlays:
  `Tests/Canvas/browser/test_canvas_served_flow.py` and
  `Tests/Canvas/browser/canvas_live_chatbook_child.py`. Both arms use identical
  corrected readiness/action machinery. No baseline product fix is transplanted.
- Canvas/served static assets, root/Canvas fixtures and the common live harness
  match between revisions. Each invocation uses its archive cwd and explicitly
  selects its own `packages/tldw_profile_core/src` via PYTHONPATH.
- Shared interpreter: Python3.12.11 arm64, SHA256
  `c413c0042baa93e9c328cecea9289c06da442b4a55aa13db540437c6c8f28835`.
  Installed Chromium headless-shell1234 arm64, SHA256
  `7687bff7cb2db075f250e6d5848bbc8838cac3802ac3952a899c574f8eccab45`,
  is selected explicitly through TLDW_CANVAS_CHROMIUM_EXECUTABLE in both arms.
  No install/download, user-profile access or test-state isolation change.

## Method and behavioral gate

Exact existing node, with unchanged assertions and deadlines:

```text
Tests/Canvas/browser/test_canvas_served_flow.py::test_actual_chatbook_console_finalizes_canvas_create_and_update[normal-True]
```

Five independent invocations per arm, serial, alternating pair order:
current1/baseline1, baseline2/current2, current3/baseline3,
baseline4/current4, current5/baseline5. Each has fresh basetemp/cache and an
independent repository-owned configuration/data sandbox. Random IDs, TLS keys
and access tokens are fresh; deterministic provider prompts/content and workflow
are identical. No other test or benchmark was run concurrently by this task.

Built-in pytest JUnit `junit_duration_report=call` measures the complete test
body: served stack/browser setup, real TldwCli child, Canvas create/update,
Mermaid rendering, pin/reopen, revoke/reconnect, provider-free saved-conversation
restore, selected revision/content/persisted-row checks, and test-body cleanup.
It excludes collection and external fixture setup/teardown. All phases must pass
for a sample to count; this is not isolated SQL time or first-paint latency.

Every invocation exited0, contained exactly the named case, and had zero JUnit
failure/error/skip outcomes. Original cleanup assertions verify owned paths,
runner sites and completed child processes. Root also checked that owned
test_data and generated TLS material were absent after every run.
Fixture lifecycle snapshots are saved before the next sample can overwrite them;
their last events precede final cleanup and may show a null child return code.
They are not independent end-of-process receipts. One optional root `ps` check
was denied by the sandbox; no process-absence claim is inferred from that denial.

## Results

All ten invocations passed. Each emitted one inherited RequestsDependencyWarning.

| Pair | First arm | Current call seconds | Baseline call seconds |
| --- | --- | ---: | ---: |
| 1 | Current | 36.834 | 35.390 |
| 2 | Baseline | 36.975 | 35.673 |
| 3 | Current | 36.702 | 33.845 |
| 4 | Baseline | 35.257 | 33.528 |
| 5 | Current | 38.056 | 34.186 |
| Median | — | 36.834 | 34.186 |
| Maximum | — | 38.056 | 35.673 |

Median delta: **+2.648s (+7.75%)**. Maximum delta: **+2.383s (+6.68%)**.
The current full-workflow cost was higher in this five-pair local measurement.
This small sample is descriptive, not a statistical guarantee or attribution
of the entire delta to SQLite. JUnit seconds are recorded to three decimals.
No new performance threshold or waiver is introduced; existing unchanged
startup/import/UI budgets retain their separately verified results.

## Failed attempt and retained evidence

The initial series stopped after one current-arm invocation failed before
Chromium launch: root omitted the browser-path override needed by the private
archive/isolated home. That exit1 result (1 failed, 1 warning, 6.73s overall)
remains preserved at `evidence/current-1`. Its raw0.483s call duration is not
a workflow sample. The user explicitly approved a fresh series with the existing
browser path set in both arms. Series2 had no retries, failures or discarded runs.

Raw evidence is retained at
`/private/tmp/task16-canvas-benchmark.MirPEY/evidence/series-2`:

- Each arm/sample directory contains exact `invocation.json`, complete emitted
  `pytest.log`, `junit.xml`, `verification.json`, copied native diagnostics
  and remaining test state.
- `series.json` preserves ordered source/cwd/command/controller timestamps,
  exits and per-file hashes; SHA256
  `2620087b0f84db6f9113f0c08dcda3eb858e69ff207e1c0a5cbcb18dc7ea987d`.
  Controller times include launch/approval delay and are not used for timing.
- `summary.json` records all raw durations and independently recomputed
  median/max/deltas. Root rehashed all recorded output/JUnit/diagnostic files
  and revalidated exact case identities directly from disk before aggregation.
- Full archive/overlay hashes, preparation and first-failure records remain in
  `task-16-report.md` in the retained SQLite SDD directory. No earlier evidence
  was erased; temporary archive copies are not new source-of-truth checkouts.

The independent fix-only reviewer verified the source/harness/archive identities,
all ten exact JUnit cases and artifact hashes, serial order, recomputed statistics,
preserved failure and cleanup limits. Verdicts: finding ADDRESSED, spec compliance
PASS, task quality PASS, SQLite closeout READY; no new Critical/Important/Minor
findings. Complete review is retained as task-16-closeout-rereview.md in this
plan's SDD directory. No test or broad correction review was repeated.

The required missing paired workflow measurement now supports the
[acceptance closeout](2026-09-08-sqlite-acceptance-closeout.md). Windows/Linux,
other native builds and full candidate/admitted V2 qualification remain distinct.
No product/test source, policy, dependency, PR, push, rebase or merge changed.
