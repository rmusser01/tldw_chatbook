# Console qualification ownership checkpoint — 2026-10-04

Base: rename repair `2c949c7d72`, on merged PR #3009 (`7d155170dc`).
Branch: `codex/switcher-workstream-burndown`; private synthetic profiles only.
These are SQLite / Textual compositor checks, not native release acceptance.

## Corrections

- Saved sidebar hydration uses Textual's watcher-free `set_reactive`; loading
  a file is not a user edit. The old assignment admitted a debounce timer during
  construction without an app context. Two genuine restored-state REDs failed
  on that timer. Mounted restoration, later user toggle, debounce and immediate
  quit-flush still pass; the existing persistence policy is unchanged.
- A module-opt-in fixture captures original constructor resources before tests
  replace public attributes. After its actual harness/workers stop, it joins the
  runtime and retires collections, evaluation, workspace, subscriptions and
  instance-lock owners. Failed disposal retains that undrained owner's resources,
  attempts independent owners and raises an aggregate error. The exception-path
  regression failed before this protection, then passed.
- Controller identity tests restore sentinel replacements before runtime
  disposal. The canonical-default test seeds its previous model explicitly;
  it still uses the production shared config cache rather than a cleared fake.
  Its preceding-test/target pair reproduced the old failure, then both passed.
- A changed-profile rename modal asserts no rename worker is admitted. Calling
  Textual `wait_for_complete([])` instead waits for **all** workers, including
  intentionally cancelled profile refreshes; that invalid wait caused a wider
  batch-only failure, not a committed rename in the wrong profile.
- Two finite workspace scope callbacks use the already-installed
  `run_owned_db_call` boundary. Real WorkspaceDB read and write REDs each left an
  extra registered worker cache; the helper retires that newly opened handle on
  its original thread, preserving caller-owned caches, memory/custom owners and
  cancellation. No global pooling or GC policy was changed.
- Allocation stacks then identified two more scope reads in the shared RAG
  resolver (cached display and fresh authorization). Both real WorkspaceDB REDs
  left an extra registered executor handle. They now use the same existing
  helper; their queries, cache flags, parsing and fail-closed behavior are
  unchanged. Independent source review found no actionable issue.
- The named-workspace case opened a cache in the pending membership projection
  retry. An intermediate store-only wrapper passed its success/failure REDs and
  removed that mounted leak, but the broader census exposed pretransaction
  validation and direct service projection in temporary-save/fork flows.
  Allocation stacks identified the service's exact `get_workspace` validation
  and `link_membership` projection callbacks. Final ownership is at those two
  existing service calls; the redundant store-only wrapper was removed. Real
  valid/missing validation and direct projection success/failure REDs each
  retained a handle. Failure injection happens after acquiring the real registry
  handle. Caller caches, exceptions, membership identity and retry bits remain
  unchanged. No transaction, registry pooling or global lifetime policy changed.
- The expanded census found temporary Chat database handles outside the original
  constructor-profile filter. Seven save/fork tests now explicitly register their
  exact local database with the opt-in fixture. Final same-file quiescence follows
  the fixture's successful runtime disposal. A first local-quiescence attempt
  closed all registered handles, but the fixture's second disposal reopened a
  handle through `detach_view` / attention / local-marks refresh. The disposed
  latch is set before those steps and is not proof of completion. The final fix
  does not change production disposal or automatically own replacement resources.
- The rename profile-departure test separately closed its secondary profile
  while the harness was still live. It explicitly registers that exact database
  with the existing activation fixture for quiescence after successful runtime
  disposal. The fixture keeps its three-tuple API and uses per-node test state;
  no production owner field or implicit replacement capture is introduced.
- Source review found the observer's suffix gate missed `.sqlite3` and its
  sidecars. Three contract tests genuinely failed before the matcher correction
  and now pass. The observer remains read-only and macOS-only.

## Final-code outcomes

- Final expanded strict combined gate: **155 passed in 479.69s**, no warnings
  or teardown errors, process exit **0**. Every teardown has zero retained
  database files under constructor profiles **and** its explicit temporary root.
  This is the final fixture code, including secondary-profile registration.
- Expanded activation/reuse gate: **71 passed in 197.45s**, no warnings,
  strict exit **0**, zero database files at all 71 teardowns. Production code is
  unchanged since that run; the subsequent test-only secondary-profile extension
  is covered by the final 155-case gate and the 14-case rename gate below.
- Combined affected Console/store/rename/sidebar/controller/ownership gate:
  **152 passed in 308.02s**, no warnings or teardown errors.
- Exact activation, cancellation, cold/warm ownership and History reuse gate:
  **71 passed in 134.49s**, no warnings or teardown errors.
- Both read-only censuses observed zero retained **constructor-profile** database
  files after every one of their 152 and 71 teardowns. Their original filter
  covers this process's private constructor/config profiles, not every
  `tmp_path` file. A subsequent expanded census additionally checks its own fresh
  `--basetemp`; do not extrapolate that coverage from the earlier two runs.
- All eleven final derived-artifact guards pass. Clean changed paths pass Ruff
  and formatting. Production diagnostic comparison against immutable
  `2c949c7d72`: the four-source group is 490/490, and the added persistence path
  is 70/70, with zero introduced findings. The store's intermediate ownership
  edit is fully removed. The [formatter baseline](formatter-baseline.json)
  pins inherited service/store debt to that immutable source; verification
  passes without reformatting unrelated code.
- Independent source review found no actionable issue in the final service
  read/link ownership consolidation. It is not native or participant evidence.

## Earlier staged outcomes

- Expanded temporary-root census: **61 passed in 100.50s**, but retained
  32 Chat SQLite descriptors, 28 WAL and seven SHM descriptors. This is a
  resource failure despite successful bodies. A strict one-case reproduction
  passed its body but exited **1**, with five SQLite/four WAL/one SHM handles.
- The first local-quiescence attempt passed three focused cases in 16.64s, but
  the expanded strict combined batch still exited **1**: **152 bodies passed in
  256.27s**, retaining eight SQLite/seven WAL/seven SHM descriptors. Exact tracing
  confirmed zero handles immediately after local quiescence, followed by the
  fixture's later disposal reacquisition. These are not retirement passes.
- Final fixture ownership and SQLite3 observer focused gate: **six passed in
  26.17s**, strict process exit **0**, zero retained database files after every
  teardown. This focused result does not substitute for the combined gate.
- Expanded activation/reuse gate: **71 passed in 197.45s**, no warnings,
  strict process exit **0** and zero database files after all 71 teardowns.
  This checks both constructor profiles and the run's explicit temporary root.
- The subsequent expanded 155-case batch passed all bodies in 402.25s with no
  warnings, but strict process exit remained **1** for one
  `departed-profile.sqlite` file starting at the profile-departure rename case.
  The seven temporary save/fork databases no longer accumulated. This retained
  secondary owner prompted the final activation-fixture registration above;
  this batch is not a terminal retirement pass.
- Final secondary-profile correction: all **14 rename workflows passed in
  86.92s**, no warnings, strict process exit **0** and no retained database files
  at any teardown. Independent source review found no actionable issue in the
  explicit owner registration or disposal-success ordering.

- Ownership / scope suite: **21 passed in 7.00s**, no warnings.
- Reviewed disposal/sentinel/profile guards: **10 passed in 19.40s**, no warnings.
- Actual restoration, user-edit/debounce/quit and wiring-order gate:
  **11 passed in 70.85s**, no warnings.
- Full affected store/rename/persistence/sidebar/controller/ownership batch:
  **144 passed in 289.93s**, **no warnings or teardown errors**. This is targeted,
  not a repository-wide sweep.
- All eleven artifact guards pass; new ownership tests join the PR census
  (140 files, floor 138). Changed clean paths format clean. Normalized lint
  comparison for six inherited-debt paths: **274 base / 273 feature, zero new
  findings**; one import-order finding removed. Independent source review found
  no remaining issue in the bounded corrections.

## Intermediate resource limit — retained historical failure

The initial combined attempt had **140 passed, one invalid profile-worker wait
failure, four sentinel teardown errors and a +352 descriptor warning**. Those
outcomes remain recorded; they are not retroactively passed.

A read-only macOS descriptor census checks private files after every teardown,
without collecting garbage, closing observed handles or changing thresholds.
Before owner retirement, the fourteen sidebar cases accumulated collections,
workspace and evaluation files. After the opt-in fix, the same fourteen cases
left zero such files after each case.

The final 144-case batch leaves **zero collections/evaluation files after every
case**, but still accumulates **75 workspace SQLite descriptors, 25 shared-memory
descriptors and 38 WAL descriptors**. This is below the existing warning limit,
not successful terminal resource qualification. Native thread affinity prevents
closing these worker caches from the teardown thread; no unsafe cross-thread
close was applied. This was the checkpoint **before** the shared RAG and
projection corrections above, not a resource pass. The broader native resource
DoD and AC #17 remain open pending the final combined census and installed
qualification. The final-code constructor-profile results above supersede these
intermediate counts, not the native/external acceptance gaps.

The subsequent 39-case mounted batch passed without warnings but still retained
three workspace SQLite descriptors plus one WAL/SHM pair beginning at the named
workspace test. Its allocation stack led to the projection correction. The
focused ownership/filtered-rename/promotion run after that correction was
**41 passed, one failed** in 25.74s, with no warning: every teardown retained
only the profile admission/lease files, zero workspace/collections/evaluation
files. The atomic-promotion failure was independently reproduced on the frozen
baseline (**30 passed, the same one failed** in 17.18s); it asserts that a sparse
context-policy write must not occur postcommit. No change to that unrelated
promotion behavior or its test is included here.

The broad shared-RAG check was **96 passed, four failed, one +513 descriptor
warning** in 35.36s. Frozen pre-change RAG tests reproduce the same four
`raw_source_selection_changed` fixture setup failures (**73 passed, four failed,
one +448 warning**, 26.65s). These are not newly failing production authority
assertions, but the warning counts differ and that broader RAG fixture ownership
has **not** been qualified. No warning filter, GC trigger or fixture marker was
added to disguise this result. Source-only review and the real ownership
regressions validate the narrow change, not a clean whole-RAG-suite claim.

An attempted cross-thread fixture assertion failed at SQLite's same-thread
guard, before checking retirement; it is **not** a valid resource RED. Two early
test invocations omitted a unique `--basetemp` and pytest emitted cleanup errors
for pre-existing pytest temporary directories. Later runs use fresh owned roots;
no user directory cleanup or warning filter was introduced. A first lint path
normalization compared different prefixes and was corrected before the claim
above; baseline and feature findings are matched by the six unique basenames,
code and message, excluding shifted redefinition line numbers.

## Reproduction and external gaps

Use Python >=3.12 and a new private temporary pytest root:

```sh
TLDW_TEST_REQUIRE_FILE_RETIREMENT=1 PYTHONPATH=Docs/QA/task-31245 \
python -m pytest Tests/Chat/test_console_title_publication.py \
  Tests/UI/test_console_rename_consistency.py \
  Tests/UI/test_console_conversation_persistence.py \
  Tests/UI/test_console_tray_read_aware_recompose.py \
  Tests/UI/test_console_workspace_tray_recompose_guard.py \
  Tests/UI/test_console_session_controller.py \
  Tests/UI/test_console_controller_wiring.py \
  Tests/UI/test_console_fixture_ownership.py \
  -p descriptor_census_probe -p no:cacheprovider \
  --basetemp /path/to/new-owned-temporary-root --tb=short
PYTHON=/path/to/python3.12 bash scripts/preflight.sh
```

Local raw logs: `/tmp/switcher-qualification-owned.log`,
`/tmp/switcher-qualification-owned-final.log`,
`/tmp/qualification-workspace-scope-red.log`,
`/tmp/qualification-workspace-scope-green.log`,
`/tmp/qualification-disposal-error-red.log`,
`/tmp/qualification-preflight-final.log`. Temporary paths alone are not portable
evidence; the source regressions and bounded outcomes above are the receipt.

Later raw logs: `/tmp/qualification-session-scope-red.log`,
`/tmp/qualification-session-scope-green.log`,
`/tmp/qualification-rag-baseline.log`, `/tmp/qualification-projection-red.log`,
`/tmp/qualification-projection-green.log`,
`/tmp/qualification-promotion-baseline.log`,
`/tmp/qualification-preflight-final-projection.log`.

Final-code logs: `/tmp/switcher-qualification-release-checkpoint.log`,
`/tmp/switcher-activation-current-checkpoint.log`,
`/tmp/qualification-direct-projection-red.log`,
`/tmp/qualification-direct-projection-green.log`,
`/tmp/qualification-validation-red.log`,
`/tmp/qualification-validation-green.log`,
`/tmp/qualification-preflight-current-final.log`.

Expanded strict-gate logs: `/tmp/switcher-final-strict-retirement.log`
(152-body pass, process failure), `/tmp/switcher-temp-db-diagnostic.log`
(one-body pass, process failure with reacquisition trace),
`/tmp/switcher-probe-suffix-red.log` (three genuine suffix REDs),
`/tmp/switcher-temp-owned-green.log` (six strict GREENs).
`/tmp/switcher-final-owned-retirement.log` (155-body pass, secondary-profile
resource failure), `/tmp/switcher-activation-strict-retirement.log`
(71 strict passes with expanded coverage).
`/tmp/switcher-profile-terminal-green.log` (14 strict rename passes).
`/tmp/switcher-terminal-final-retirement.log` (final 155 strict passes).

An allocation observer initially assumed WorkspaceDB had a
`quiesce_connections` method and failed before exercising the case. That
diagnostic run is invalid, not a regression RED; the corrected observer uses
the actual same-thread `close` method without changing its result or lifetime.

The [native checklist](native-qualification-checklist.md) is HOLD until a clean
frozen head and synthetic launch packet are available. Native macOS keyboard,
Windows Terminal, three first-time participants and the full latency matrix
remain unwaived. ADR-198's accepted boot freeze is a partial GC improvement;
TASK-33545 retains the under-50-ms tour goal. No semantics implementation or new
PR is represented by this checkpoint.

The [fixture rebuild proposal](fixture-rebuild-proposal.md) is a proposed
source-bound replacement for the missing ignored corpus/launcher, not a new
measurement or implemented launcher. No old corpus digest or timing is reused
as current-head evidence.
