# SQLite final local qualification — 2026-09-08

Status: partial; qualification is in progress, not Canvas V2 admission.

Product/test checkpoint: `55f74aa0097fb5f657460f1240b1836731f21266` in the existing
`canvas-v1` worktree. Immutable comparison baseline:
`9bc73ffb35ccd6eb24629bfa8021b28063dc9112`. No branch switching, shared dependency
change, host cleanup, full repository sweep or external Git/PR action.

ADR required: no new ADR. Qualification implements
[ADR-125](../../../backlog/decisions/125-lock-safe-private-sqlite-validation.md)
and preserves [ADR-097](../../../backlog/decisions/097-boot-budget-ratchets.md).
The bounded test-only Task 11 changes no runtime or authority contract.

## Reviewed teardown correction

Task 11 commit `55f74aa009` changes only
`Tests/Chat/test_console_trace_compaction.py::test_maintenance_setup_failure_closes_handle_and_releases_exclusion`.
All operations after successful database construction are protected by `try`,
with unconditional `database.close_connection()` in `finally`. All existing
production-behavior assertions, timeout and exception matchers remain intact.

The collected, temporary control invokes the actual test, forces its late count
assertion and measures the real registry with the captured original accessor.
Before the fix it observes one retained connection; afterward it observes zero.
The probe has its own fallback cleanup and is preserved as ignored evidence,
not a permanent test-of-test framework.

- Functional baseline: 1 passed, 1 inherited warning, 1.46s.
- Failure-path RED: 1 failed, 1 warning, 2.58s; forced assertion `999 != 1`,
  followed by actual registry `1 != 0`.
- Failure-path GREEN: 1 passed, 1 warning, 2.19s.
- Complete compaction/admission selection: 20 passed, 1 warning, 16.24s.
- Root committed probe: 1 passed, 1 warning, 1.66s; original target independently
  1 passed, 1 warning, 0.93s.
- Changed-file Ruff and changed-function formatting pass. Aggregate checks
  retain one inherited admission import-order diagnostic and two files with
  pre-existing formatter suggestions. Whitespace checks pass.
- Independent Task 11 review approves both spec compliance and task quality,
  with no Critical/Important findings. Inherited warning/static noise remains
  disclosed; the earlier teardown Minor is resolved.

The root's first combined probe/ordinary-target invocation errored before
collection because explicit `-p Tests.conftest` collided with automatic
conftest registration. Separate invocations above are the valid evidence; no
application or fixture change was made to repair the invocation.

Exact commands, outputs, probe and immutable review package remain under
`.superpowers/sdd/2026-09-07-sqlite-lock-safe-private-validation-implementation/`
in `task-11-{brief,report}.md`, `task-11-probe.py`, static logs and
`review-f2faa5d65c..55f74aa009.diff`.

## Remaining qualification execution

The explicit 37-file manifest unions the original Task 7 selection, changed tests
through Task 10 and adjacent Canvas/admission/provenance coverage. Its command
and paths are preserved as `task-7-final-command.txt` and
`task-7-final-selection.txt` in the same SDD directory. Only the eleven
previously diagnosed semaphore-blocked parameter IDs are explicitly deselected.
The selection is targeted, not the full repository suite.

Separate checks cover bundle UI consumers, the local SQLite parameter whose ID
`live` is accidentally caught by the global skip keyword, and the five existing
actual Canvas-child nodes. `--run-live` is limited to that exact local unit test;
its network guard remains active. No source override or V2 admission switch is
used for these pytest checks.

### Affected-selection result

The exact 37-file command completed with **1844 passed, 5 skipped, 11
deselected, 9 warnings in 562.77s**, exit 0. Owned loopback permission was
granted; no source override was used. Full output: `task-7-final-selection.log`.
This includes installed-helper packaging, actual app retained-owner exit,
database privacy/lock/lifecycle, import provenance and all 26 repaired baseline
cases. It is one targeted run, not a full repository result.

Unchanged startup guards report boot 625/660, UI 963/972, preload 499/500 modules;
364325/378740 total LOC and 110163/123319 largest-route LOC. No budget or snapshot
was changed.

The five skips are explicitly unqualified in this run:

- Python 3.11 unavailable for historical-verdict re-derivation (the supported
  product floor remains Python >=3.12).
- Two Windows-only private SQLite cases.
- A TTS draft-options case disabled by its existing slice contract.
- The local schema parameter named `live`, scheduled separately below.

Nine warnings: one inherited Requests version mismatch; one fork-from-threaded
process deprecation in the intentional fork-refusal test; three intentional
startup-headroom notices; joblib semaphore errno 28 with serial fallback during
the real-transformers import check; and three inherited invalid-escape syntax
warnings exposed by source inventory. The joblib fallback does not qualify
parallel operation or resolve the host semaphore blocker. No dependency or
warning suppression was introduced.

### Separate consumer checks

```sh
../../.venv/bin/python -m pytest Tests/UI/test_stts_profile_library.py -k bundle -q -ra --tb=short --show-capture=no
../../.venv/bin/python -m pytest 'Tests/TTS/test_profile_schema.py::test_profile_row_validation_rejects_oversized_raw_options_before_parsing[live]' --run-live -q -ra --tb=short --show-capture=no
```

Bundle selection: **59 passed, 104 deselected, 1 warning, 21.22s**. Local schema
parameter: **1 passed, 1 warning, 1.56s**. Both exit 0, with only the inherited
Requests warning. Logs: `task-7-final-bundle.log` and
`task-7-final-local-live-id.log`. The second result resolves only that skipped
local case; it does not relabel the original 37-file run or other skipped gates.

### Actual Canvas children

The existing exact five-node selection ran with owned browser/loopback permission,
no `PYTHONPATH` override and existing candidate fixtures only:

```sh
../../.venv/bin/python -m pytest \
  'Tests/Canvas/browser/test_canvas_served_flow.py::test_actual_child_control_refusal_keeps_terminal_usable[snapshot]' \
  'Tests/Canvas/browser/test_canvas_served_flow.py::test_actual_chatbook_console_finalizes_canvas_create_and_update[normal-True]' \
  'Tests/Canvas/browser/test_canvas_served_flow.py::test_actual_chatbook_console_finalizes_canvas_create_and_update[read-publication-False]' \
  Tests/Canvas/browser/test_canvas_served_flow.py::test_canonical_adversarial_corpus_stays_in_served_product_route \
  Tests/Canvas/browser/test_canvas_zero_egress.py::test_canonical_adversarial_corpus_stays_in_native_product_route \
  -q -ra --tb=short --show-capture=no
```

Result: **4 passed, 1 failed, 1 warning in 172.20s**, exit 1. Full output:
`task-7-final-canvas-children.log`. The read-publication parameter failed at
`test_canvas_served_flow.py:1029`: after reconnect, saved-conversation load and
the second F12 action, `canvas-live-card-pressed` did not exist. Earlier creation,
update, interleaved-read refusal and initial pinning checks had completed. The
other four selected workflows passed; this is not five-node qualification.

The bounded independent read-only diagnosis found the replacement child alive
and connected, but no restored selection bound before teardown. F10's load
acknowledgement proves successful logical hydration without provider calls, not
that the target card is mounted. The F12 fixture uses an unguarded `next(...)`
over mounted cards; the normal UI sync can coalesce and return before later
paint. A harness readiness/action-delivery race is plausible; stuck deferred sync
or a session guard remains unexcluded. No SQLite crash or root cause is proven.

Before one exact unchanged-node rerun, the original source-free lifecycle was
copied to `task-7-final-first-failure-lifecycle.json` (29978 bytes). The rerun
failed **earlier**, at line 838's existing five-second `first-byte` body-class
expectation: body remained `-loaded`. Result: **1 failed, 1 warning in 10.01s**,
exit 1; log `task-7-final-canvas-rerun.log`, separately preserved lifecycle
`task-7-final-rerun-lifecycle.json`. It did not reach the recovery boundary and
therefore neither reproduces nor clears the first failure. No further reruns.

Read-only findings and next diagnostic are in `task-7-final-canvas-diagnosis.md`.
The fixture normally removed its own temporary data at teardown. No source,
assertion, timeout or admission behavior was changed.

Recommended next step is a bounded diagnostic spike: preserve source-free
startup/lifecycle timing plus stage booleans/card counts at F10 completion,
F12 entry and production-open entry. Do not retry, wait longer, catch/suppress
the failure, or change product behavior. The brainstorming approval gate applies
before adding this new instrumentation; no spike implementation is authorized
or begun at this checkpoint.

## Checkout timing comparison

Both runs used the same preserved `task-7-benchmark.py` driver, interpreter and
five-sample workloads. The driver asserts owned HOME/XDG/config and offline,
null-keyring setup before product imports in both parent and import subprocesses.
No concurrent tests or benchmarks ran. Each arm explicitly includes its own
checkout-local `tldw_profile_core`; these are **checkout measurements**, not
installed-package qualification. Installed-helper evidence is in the targeted
pytest run above.

Baseline was a fresh `git archive` of the exact immutable commit, extracted to
`/tmp/tldw-task7-final-baseline.9GenCx`. The driver ran with these argument pairs:

```text
current: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/canvas-v1 /tmp/tldw-task7-final-metrics.Ei6ZS4/current
baseline: /tmp/tldw-task7-final-baseline.9GenCx /tmp/tldw-task7-final-metrics.Ei6ZS4/baseline
```

Both completed exit 0. Logs: `task-7-final-current-benchmark.log` and
`task-7-final-baseline-benchmark.log`. All archives/scratch evidence are preserved.

| Five-sample timing | Current median / max (ms) | Baseline median / max (ms) |
| --- | --- | --- |
| App import | 1120.993 / 1286.426 | 1361.905 / 1580.152 |
| Actual app UI readiness | 8814.697 / 9019.774 | 8525.682 / 9435.812 |
| Threaded TTS repository open | 192.666 / 194.587 | 13.214 / 13.737 |
| Helper prepare launch/batch | 54.002 / 55.745 | API absent |
| Fixed TTS child readiness | 56.398 / 57.122 | API absent |
| Proof recheck (50 samples) | 0.291 / 0.772 | API absent |

Observed median UI readiness is 289.015ms (3.4%) higher. Repository open adds
179.452ms (14.6x); this is a material cost of the corrected path, not a claim of
unchanged performance. Existing guardrails remain unchanged and pass in the
targeted tests; no new latency threshold was invented from these samples.

Sampled high-water observations: repository open current 2 helpers/+9 FDs versus
baseline 0 helpers/+11 FDs; isolated helper workload current 1 helper/+3 FDs. Sampling
at 0.5ms can miss transients: these are not exact kernel peaks or standalone leak
proof. The missing baseline helper API is reported unavailable, not comparable.

Both app-harness logs contain the same error categories: unmount sysctl EPERM,
missing sidebar timer, footer setup after the screen stack is gone, and TTS/STTS
initialization errors. The inherited Requests warning appears in both; the fresh
archive also exposes two invalid-escape warnings. Successful metric collection
does not make these pristine logs or a clean end-to-end app-lifecycle gate. No
extra permission, harness change or unrelated repair was used to hide this noise.

## Final static checks

Explicit scope: the 53 Python files returned by
`git diff --name-only 9bc73ffb35ccd6eb24629bfa8021b28063dc9112..55f74aa0097fb5f657460f1240b1836731f21266 -- '*.py'`.
Ruff check reports **1373 diagnostics**, exit 1; format check reports **12 files
would be reformatted, 41 already formatted**, exit 1. Full outputs are
`task-7-final-ruff.log` and `task-7-final-format.log`. No fixes were applied.

Whole-correction and working-diff whitespace checks pass. Earlier task-scoped
BASE attribution remains preserved, including Task 10's added-range/import-block
audit and Task 11's exact function check. This aggregate count does not itself
prove every diagnostic inherited or waive the nonzero static gate. No unrelated
formatting or warning suppression was introduced.

## Qualification limits

The eleven spawned repository cases remain unqualified until a meaningful host
state change and successful isolated stdlib allocation control. The prior
[host diagnosis](2026-09-08-semaphore-allocation-diagnosis.md) is not permission
to unlink semaphores, terminate processes or restart the Mac. No new state change
has been reported at this checkpoint.

Windows and other skipped platform/optional gates must be reported separately,
not treated as successful local evidence. TASK-31942 remains In Progress and
Canvas V2 remains disabled while required gates are open.

## Subsequent approved diagnostic spike

The approved test-only stage-marker run reproduced the recovery failure and
established premature synthetic card action: no cards were mounted at F10
completion or F12 entry 46ms later. The separate startup failure remains open.
The diagnostic was removed from live test files, with exact baseline identity
verified; no fix or green qualification was claimed. See the
[stage-marker evidence](2026-09-08-canvas-card-readiness-spike.md) for the one-run
result, in-run comparison, process-exit evidence and inference limits.

## Subsequent bounded harness correction

The user-approved mounted-current-card correction in `72af5b63bd` now passes
the original exact actual-browser node (1passed1warning49.52s), preserving
production dispatch, pin/metadata/provider-count assertions and outer waits.
Root committed unit selection10passed1warning3.10s; scoped review approved.
See the [correction evidence](2026-09-08-canvas-card-readiness-fix.md) for exact
commands and limits. This does not erase the earlier failures, explain the
separate startup miss or close the other qualification gaps recorded above.

## Subsequent approved startup test-contract correction

Task13 (`235641b380`) retains both first-output and Composer readiness conditions
within one shared45s deadline, replacing the implicit5s first-output sub-limit
with explicit user approval. No retry/reset, later assertion/timeout or production
change. Root original actual-browser case passes once:1passed1warning46.86s,
exit0; root committed focused tests5passed1warning2.37s, exit0. Scoped spec and
quality review approved, no Critical/Important findings. Ruff/new-file format/
whitespace pass; existing whole-file formatting debt and Requests warning remain.
See [deadline correction evidence](2026-09-08-canvas-startup-deadline-fix.md).
Historical timing causality is not claimed, prior failures are preserved, and
the remaining host/platform/optional/static gates and V2-disabled status stand.

## Subsequent static attribution

The [paired static comparison](2026-09-08-sqlite-static-attribution.md) now
attributes the saved nonzero results without rewriting code: 1360 of 1373
diagnostics match unchanged baseline spans; thirteen changed/moved spans are
documented separately. All 307 formatter edit groups in the twelve recorded
dirty files match baseline edit content and adjacent lines. Explicit py311/py312
controls distinguish six language-floor-sensitive recommendations. This does
not make the static gate green or clear modified import blocks by assumption.
No pytest, browser, benchmark, host-control rerun or admission change occurred.
