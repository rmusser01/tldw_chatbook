# Explicit Console rename publication — 2026-10-04

Frozen base: `7d155170dc95557736a239a1ce7427981f4d50ec` (merged PR #3009).
Task-owned branch: `codex/switcher-workstream-burndown`. These are isolated
real-SQLite and Textual compositor tests, not native-terminal evidence.

## Contract and scope

Saved tab/F2/palette and rail/tree actions share the incumbent workspace rename
worker. It reserves every exact open alias before its optimistic durable write,
then publishes to those retained instances. New hydration/rebinding into the
transition is refused; rebinding away and same-binding preparation cleanup stay
safe. Cancellation drains the already-running SQLite write before publishing
its committed title into the captured store. Profile/lifecycle changes prohibit
painting or confirmation in the new or retired view.

Success awaits tab/header publication and both mounted sidebar projections,
including filtered named-workspace rows. The receipt checks completed compose
state and actual DOM signatures, and retires when its tray detaches. An inactive
transcript gets the renamed header when selected; renaming never activates it.
Unbound/automatic titles retain their incumbent behavior. Library currently has
no title-edit control; this patch does not create one. ADR-085 owns the narrow
amendment; no schema, dependency, event bus or GC policy changed.

## Evidence

- Actual mounted rail-menu active/inactive and bound tab-modal tests on the
  frozen pre-fix source: **4 failed**, all at the saved/live title divergence.
  The saved write succeeded; the runtime stayed old. One old-path unawaited
  refresh warning also occurred. No setup failure is counted as this RED.
- Targeted store, rename, persistence and shared recomposition guard batch:
  **46 passed in 235.70s**, **one aggregate descriptor warning** (+297).
  Fourteen rename cases cover the actual menu, selected/unselected tab, profile
  departure during a held real write, cancellation, changed-profile modal,
  durable refusal/exception, filtered tree titles and retired publication owner.
- Final added inactive-header checks plus existing Ctrl+K rename dispatch:
  **3 passed in 35.57s**, no warnings.
- Earlier bounded core: **31 passed in 92.03s**, no warnings; shared guards
  separately: **14 passed in 44.53s**, no warnings. The combined warning is not
  hidden by those smaller runs. Eight legacy store rename cases and two named
  preparation cases also passed without warnings.
- Wider controller comparison: feature batch **121 passed, 2 failed,
  33 warnings**. Frozen-base controller batch **90 passed, the same 2 failed,
  28 warnings**. Both isolated base cases passed individually: the failures
  depend on batch lifetime/configuration. They and the timer/descriptor warnings
  remain for qualification/resource remediation, not a clean-suite claim.
- Changed tests pass Ruff. Six baseline-clean Python paths format clean;
  [immutable formatter baseline](formatter-baseline.json) verifies that the
  other three source paths add no formatting debt. Normalized source Ruff
  comparison: **431 base / 431 feature, zero additions or removals**.
- All eleven derived-artifact checks pass. Diagnostic-inventory change is one
  owner digest for reindented existing logging: **53 calls before and after**,
  no added/removed messages or sinks. Whitespace clean.
- The rename UI module is added to the PR fast-lane census. No full sweep ran.

Later qualification fixes retire finite workspace callback caches and explicit
test-owned databases after the last successful runtime disposal. The final
combined targeted gate is **155 passed, no warnings**, with strict process exit
0 and zero database files after every teardown across both private constructor
profiles and the run's temporary root. All **14 rename workflows** separately
pass the same strict gate. The earlier failures/warnings above remain historical;
see the [ownership checkpoint](../task-31245/ownership-qualification-2026-10-04.md)
for valid REDs, unsuccessful teardown attempts, inherited broader failures and
the standalone activation gate. Native acceptance remains separate.

Reproduce the affected batch with the repository Python >=3.12 environment:

```sh
python -m pytest Tests/Chat/test_console_title_publication.py \
  Tests/UI/test_console_rename_consistency.py \
  Tests/UI/test_console_conversation_persistence.py \
  Tests/UI/test_console_tray_read_aware_recompose.py \
  Tests/UI/test_console_workspace_tray_recompose_guard.py \
  -p no:cacheprovider --tb=short
python scripts/terminal_qualification/format_ratchet.py verify \
  --baseline Docs/QA/task-33620.9/formatter-baseline.json
PYTHON=/path/to/python3.12 bash scripts/preflight.sh
```

## Limits and unsuccessful attempts

No native control, Windows host, first-time participant or full latency matrix
was exercised here. Existing character-navigation acceptance remains open.
Independent source review's admission, cancellation, rebinding and receipt
findings were corrected and re-reviewed with no remaining production blocker.

An early success assertion included the glyph-only menu button; it was not a
title assertion and was corrected. A later inactive-menu assertion assumed
runtime metadata survives a saved-row rebuild; it now admits only the exact
runtime or exact persisted conversation, never a matching title. Genuine stale
and empty-DOM failures remain in the evidence history. A rail profile test
initially pushed its modal through the unmounted owner rather than the mounted
harness, and an early tree test queried flat buttons instead of tree nodes.
Those fixture errors are not product RED evidence. A too-broad legacy cancel
selection hung; only its verified task-owned pytest PID was terminated, and no
result count is claimed. No thresholds or warning filters were raised.

Raw local logs remain in `/tmp/rename-real-menu-red.log`,
`/tmp/rename-core-verified.log`, `/tmp/rename-inactive-final.log`,
`/tmp/rename-base-controller-batch.log`, `/tmp/rename-integration.log` and
`/tmp/rename-preflight-verified.log`. These temporary paths are not portable
evidence; the source-bound commands and bounded outcomes above are the durable
receipt. Frozen baseline diagnostic copy is `/tmp/rename-frozen-base-PbBiHI`.

## Publication review follow-up — 2026-10-05

Source checkpoint: `c7e5558f25036996e85fb570419f4a31a529df5c`, on dev
`74557e202ac38c6d29510d0940a062ca7cc7f38b`. Independent read-only review
found two missed rename boundaries. A pending preparation or optimistic send
could restore its old title after a committed rename; tracing successful sends
found the same overwrite through staged identity publication. Saved sends do
not auto-title, so these three existing store paths now preserve the current
title for an unchanged saved binding. Genuine scratch/rebound rollback and
first-save naming remain intact; no new title ledger, cache, or dependency was
added. Control-only saved-tab input could also sanitize to blank after its
initial validation and erase the durable title before publication failed. The
shared workspace entry point now rejects sanitized, stripped blank input before
worker admission.

Strict regression RED: **6 failed, 5 passed in 8.79s**, with stale-title
restoration and real SQLite blank-title writes; a separate successful-publication
RED had **2 failures**, including replay after first save. GREEN: **13 passed in
8.36s**, no pytest warnings, with zero database files at every recorded teardown.
The controls cover cancellation before/during/after publication, preparation
and optimistic rollback, saved/scratch/rebound identities, first-save naming,
repeat publication, and actual mounted saved-tab/shared validation. Independent
follow-up review found no remaining scoped code/test blockers for draft
publication. This is not native or release qualification.

Raw receipts: `/tmp/task33620-rename-review-1aQNn4/{red,identity-red,green}.log`.
Both changed tests are Ruff-clean and all four Python paths format-clean.
Normalized production Ruff remains **186 baseline / 186 feature** with zero
additions or removals. Broader affected checks are recorded below when complete.

The initial seven-file affected selection stopped with two hook-admission
fixture failures and **171 passing cases in 246.20s**. Both failures reproduced
in isolation before changed title paths. The automatic-preparation module now
uses the existing `bootstrap_profile` contract for canonical private config
reads; hook admission remains real. Those two cases and the held-provider
control passed (**3 in 1.07s**). The next run exposed two stale test barriers
and stopped with **230 passing cases in 242.14s**: postaccept stubs intercepted
the installed observation-only READY assessment. Eight stubs now skip only
that invocation, retaining every original ACCEPTED/identity/cancellation
assertion. All **14 owner controls passed in 3.85s**, but the strict process
**failed** on retained `tldw_chatbook_workspaces.db` files. No warning filter,
forced collection, blanket close, or production preflight change was added.
Independent review found no blocker in either bounded fixture correction.
The interrupted attempts and strict resource failure remain unsuccessful
evidence, not a qualified batch.

Raw receipts additionally include `affected.log`, `refusal-isolated.log`,
`refusal-profile-green.log`, `affected-final.log`, and `barrier-green.log` in
the same private root. The preparation-file Ruff ratchet remains three
inherited diagnostics with zero additions or removals; it was not bulk-formatted.

The complete original seven-file selection at
`cc3e3137e45ee8e2d36ffa81d55bb244a2b97337` finished **340 passed, 4 failed,
1 warning in 493.80s**, process exit **1**. Its existing 1,000-send retention
case passed in 334.19s; neither count nor timeout was reduced. Two shutdown
cases fail retained-controller and held-COMMITTING assertions; two postdispatch
fixtures fail issued-generation retirement and unknown-delivery assertions.
All four reproduce in an isolated selection with their parameter controls:
**4 failed, 10 passed, 1 warning in 8.33s**. The warning is a diagnostic
ContextVar token reset during abandoned-task finalization, not suppressed.
Source/ownership review is ongoing; unchanged controller and durable-round-one
files are not a substitute for frozen-base execution or a reason to waive them.

Read-only census records all 344 teardowns. Retained database descriptors begin
at a real durable queued-recovery fixture and reach **155**; both test-owned
SQLite databases and the private workspace database remain at final teardown.
This is failed retirement, not a constructor cache allowance or terminal pass.
Raw receipts: `affected-complete.log` and `four-failures-isolated.log` in the
same private root. Draft publication must retain these failures and the separate
native/Windows/participant/latency HOLD gates.

Follow-up review separated three fixture-boundary defects from a genuine
emergency-ownership bug. The live-loop test now releases its paired store root
after every shutdown assertion; the store's handoff callback legitimately owns
the controller. The closed-loop setup waits boundedly for actual history entry
or submit completion, instead of assuming 20 zero-delay ticks complete real
off-thread admission. Failed setup cancels and awaits its tasks before closing
the owned loop. Two postdispatch injections now invoke the real
`before_provider_dispatch` checkpoint callback; every original generation,
unknown-delivery and exact-owner assertion remains.

That corrected readiness revealed TASK-34412: permanent emergency detachment
removed a closed task from the submit ledger but retained it in maintenance
admission, whose coroutine finalizer can never run on that closed loop. The
production fix is one exact-task removal inside the existing proven-closed
branch. Strengthened actual closed-submit and mixed closed/live controls fail
before it (**2 failed, 1 warning in 0.98s**) and verify that live maintenance
ownership is preserved afterward. Independent review finds no scoped blocker;
no blanket clear, callback detachment, task-state mutation, diagnostic filter,
or global runtime/GC policy change.

Affected shutdown/postcommit/maintenance controls: **25 passed, 2 warnings in
9.68s**, strict process exit **1**. Both warnings retain cross-context token
reset failures from deliberately abandoned coroutines. All 25 teardown census
observations remain; the 12 file-backed durable controls retain up to **48**
SQLite descriptors. This is body GREEN, not strict retirement or qualification
GREEN. Earlier correction attempts at 13 pass/1 failure/1 warning are retained
as `four-fixture-corrections*.log`; final raw receipts are
`maintenance-ledger-{red,green}.log`. Both changed tests format clean; five
inherited test lint diagnostics and 182 production diagnostics are unchanged.
Normalized controller formatting debt exactly matches dev; no bulk format ran.
The complete seven-file batch has not been rerun after this follow-up.

### Exact durable-send fixture ownership — TASK-34403

The twelve durable generation/postcommit controls reproduce accumulation on
`cb2d88ce236c12c58bd9b8c94af49c5197048ea4`: all twelve bodies pass in 8.34s,
but strict retirement exits **1**, retaining up to **48 SQLite descriptors**.
Their imported helpers create real database/controller owners without teardown.

An opt-in fixture wraps only this importing module's controller and ready-store
helper bindings, captures their exact returned owners, and registers the direct
evidence owner explicitly. It awaits existing controller shutdown before the
existing same-file database quiescence boundary. A failed drain keeps that
owner's files, continues independent owners, and rethrows; no production,
constructor, shared-profile, diagnostic or GC policy changes.

The unchanged twelve controls then pass in **8.16s**, no warnings, strict exit
**0**, with zero DB files after every teardown. Independent scoped review finds
no actionable issue. Raw receipts: `durable-owners-{red,green}.log` in the same
private evidence root. The full affected module then passes **25 tests in
336.58s**, no warnings, strict exit **0**, with zero DB files at every teardown;
its unchanged 1,000-send case takes 323.71s. The original combined 25 shutdown,
postcommit and maintenance controls also pass in **9.31s**, strict exit **0**,
with zero DB files, but retain **two ContextVar warnings** from intentional
closed-loop abandonment. Both results are separate: ordinary module retirement
is green; emergency diagnostic cleanliness is not.

All eleven artifact guards pass. This changed test remains format clean with
its two inherited Ruff findings and no added diagnostics. Receipts:
`durable-module-green.log`, `durable-maintenance-green.log`, and
`durable-owners-preflight.log`. The complete seven-file resource batch and
native/scale qualifications remain unverified; no waiver follows from this
narrow fixture repair.

### Queued-recovery owner adoption — TASK-34403

The three real-SQLite queued recovery cases on published
`be1ba5b4742c4f7980c2c3e32992a6c3400fbb68` pass in **4.09s**, without
warnings, but strict retirement exits **1** with **12 SQLite descriptors**.
The files are their own `reclaim.sqlite` and `reclaim-one.sqlite`, not the
separate workspace-profile cache observed in the earlier wider batch.

The existing exact-owner finalizer now lives in an explicitly requested shared
Chat fixture. Durable helper wrapping stays local to its importing module;
the three queued cases register only their own database/controller pair. A
real-DB failure control proves an unsuccessful shutdown retains that owner's
file and rethrows, still retires an independent healthy owner, and leaves a
foreign database open. No global constructor interception or production
shutdown/cache/GC/diagnostic policy was added.

The three recovery cases plus the durable module's non-retention controls and
new failure control pass **28 tests in 17.50s**, without warnings, strict exit
**0**, zero DB files at every teardown. The unchanged 1,000-send case is
explicitly deselected here; its prior complete-module receipt above remains
historical evidence, not a current shared-fixture full-module claim.
Independent scoped review has no actionable finding. All eleven artifact
guards pass; all three changed test paths format clean, with the same six
inherited Ruff diagnostics and no additions. Raw receipts are
`/tmp/switcher-queued-retirement-Kk49kF/{red,green,preflight}.log`.
The complete seven-file resource batch, emergency warnings and all native,
Windows, participant and scale gaps remain open.

### Empty frozen-authority admission — TASK-34410

At published `a9e65dcabc518474b38a8d2e8bc7b10375e9bb7f`, the original
direct/agent shutdown controls pass **2 tests in 1.14s**, without warnings,
but strict retirement exits **1**. A read-only allocation trace ties the
agent's three retained workspace SQLite/WAL/SHM descriptors to
`_run_agent_reply` → `frozen_workspace_roots` → the default registry factory.
The captured binding authority is empty: the helper opens a database solely
to return no roots. This is unnecessary admission, not evidence that the
process-wide registry must be closed by test teardown.

Four real-SQLite regressions fail before repair (**4 failed in 1.05s**, no
warnings): empty tuples and iterators both create the default database or
reopen a supplied handle. Their exact test-owned cleanup leaves zero DB files.
The three-line production guard materializes the captured iterable inside the
existing fail-closed boundary and returns before a registry read when empty.
Nonempty authority retains every live binding/root/locator/identity check;
no cache lifetime, constructor, foreign-owner, warning or GC policy changes.

The four new regressions, existing nonempty authority/retarget controls,
original preparation/shutdown/cancellation controls and queued recovery cases
pass **22 tests in 7.77s**, without warnings, strict exit **0**, zero DB files
at every teardown. Independent scoped review finds no actionable issue. All
eleven artifact guards pass. The new test is Ruff/format clean; the source's
eight inherited Ruff findings and normalized formatting debt are unchanged.
No complete seven-file or 1,000-send rerun is claimed by this bounded result.

Raw receipts: `/tmp/switcher-workspace-root-m7n8QI/{red,unit-red,green,preflight}.log`
and `static.json`; these temporary paths are not portable evidence. Runnable
empty/nonempty authority controls:

```sh
python -m pytest Tests/Tools/test_frozen_workspace_root_read_admission.py \
  Tests/Chat/test_console_turn_execution_context.py::test_frozen_workspace_binding_maximum_excludes_later_roots \
  -p no:cacheprovider --tb=short
```

All earlier failed receipts remain above. Emergency ContextVar warnings, the
complete affected resource batch, native/Windows/participant/scale qualification
and current-head Qodo review remain open; exhausted Qodo credits are not a
review waiver.

### Broader resource recheck and emergency fixture context — TASK-34412

The original seven-file selection plus four empty-authority controls at
`1b79a6ac0c8918c990c6e4e69904d94fdfba3bf7` finishes **349 passed, 2 warnings
in 678.53s**, strict exit **1**. Its unchanged 1,000-send test passes in
456.20s. All 349 teardown observations are retained: database retention starts
at the acceptance module's conversation-failure case and reaches **55 SQLite
descriptors** from `acceptance.sqlite` and four accepted-cancellation databases.
The earlier workspace-profile and queued/durable fixture files no longer
appear; this is still a failed broader resource gate, not qualification.

The deliberate closed-loop fixture produces a fresh warning-as-error RED:
**1 failed in 1.17s**, with both ContextVar resets running outside their
token-owning Task context. Running collection in the exact public Task context
alone fails (**1 in 0.59s**): Python's custom exception handler re-enters that
already-active context and loses the required pending-task diagnostic. That
unsuccessful experiment was not installed or counted as GREEN. A forwarding
probe of the installed default handler passes (**1 in 0.81s**, strict exit 0).

The bounded test correction retains every original COMMITTING, sidecar,
ledger, provider and weakref assertion. Only this deliberately abandoned
Task receives a captured public Context for its existing final collection.
The actual default handler is observed through exact first-line/count equality:
one `Task was destroyed but it is pending!`, no preceding asyncio errors.
Both scoped bindings must restore, every captured warning fails, and unraisable
exceptions are errors. No production reset, coroutine close, Task state,
collector setting, foreign-owner or emergency-policy change. Python's
[Task context](https://docs.python.org/3.12/library/asyncio-task.html#asyncio.create_task)
and [exception-handler contract](https://docs.python.org/3.12/library/asyncio-eventloop.html#asyncio.loop.set_exception_handler)
describe the context boundary; the failed probe preserves its concrete limit.

Original live/closed controls: **15 passed in 4.98s**, no warnings, strict
exit 0, zero DB files at all teardowns. Complete preparation module plus durable
non-retention controls: **114 passed, 1 deselected in 76.12s**, no warnings,
strict exit 0, zero DB files at all 114 teardowns. The 1,000-send case is not
duplicated in this follow-up. Independent scoped review finds no actionable
issue. All eleven artifact guards pass. Changed test format clean; its three
inherited Ruff findings unchanged.

Raw receipts: `/tmp/switcher-emergency-context-ciQIs5/{affected,red,probe,default-probe,green,covering}.log`.
These are temporary logs, not portable evidence. The 55 acceptance-fixture
descriptors, full corrected resource rerun and native/Windows/participant/scale
and Qodo gates remain open. Controlled test-owned abandonment is not proof of
warning-free production emergency teardown or terminal abandoned Tasks.

### Acceptance-fixture owner adoption — TASK-34403

The remaining acceptance module reproduces in isolation: **19 passed in
12.85s**, no warnings, strict exit **1**, **55 SQLite descriptors**. Thirteen
helper-returned databases and four directly built accepted-cancellation owners
explain the files; the two publication controls already close their own caller
handles. The real `_ready_store` implementation and every sibling imported
binding remain unchanged.

The module now wraps only its own helper binding and explicitly registers the
four direct database/controller pairs before starting their submits. It reuses
the installed opt-in shared awaited-shutdown/quiescence fixture. Original
rollback, policy, attachment, cancellation and publication assertions remain;
the existing failed-drain/healthy/foreign-owner control remains in coverage.
No constructor interception, production/shared-profile/cache/GC change, foreign
close or warning suppression.

Acceptance, complete preparation and durable non-retention controls pass
**133 tests, 1 deselected in 63.31s**, without warnings, strict exit **0**,
zero DB files at all 133 teardowns. Independent scoped review finds no actionable
issue. Changed test format clean; its three inherited Ruff findings are unchanged.
Raw receipts: `/tmp/switcher-emergency-context-ciQIs5/acceptance-{red,green}.log`.
All eleven artifact guards pass; receipt:
`/tmp/switcher-emergency-context-ciQIs5/acceptance-preflight.log`.
The prior complete 349-case failure and unchanged 1,000-send result remain
historical; this follow-up does not rerun that long body or claim a fresh full
corrected batch. Native/Windows/participant/scale and current-head Qodo gates
remain open, separately from these repaired test-owner paths.

### Published-head broader resource checkpoint — `51860b4558`

The eight affected files, excluding only the unchanged long 1,000-send body,
pass **348 tests, 1 deselected in 196.56s**, without warnings. The strict
resource process still exits **1**: four replacement-profile descriptors first
appear after `test_cancelled_rename_settles_committed_title_before_releasing_worker[profile]`
and persist through the last teardown. All 348 observations are retained;
327 contain `departed-profile.sqlite`/WAL/SHM files. None contain the previously
repaired acceptance, queued-recovery, durable or workspace-profile files.
This is body GREEN, not a passed broader resource gate.

Bounded diagnostic repeats do not repair that failure. The isolated two
cancellation controls pass in 9.35s with strict retirement. One forwarding-only
33-control trace passes the bodies in 82.64s but exits **1**, retaining four
`second-profile.sqlite` descriptors across ten teardowns instead. Three further
forwarding traces pass 33 controls each (98.57s, 91.15s and 71.09s), strict
exit 0. They observe registry and physical files retiring, without evidenced
late reacquisition; the extra observer can perturb allocation/scheduling.

A final native SQLite audit observes all 22 target-file opens using weak
references, delegates the existing close helper unchanged, and leaves original
pytest capture enabled. It passes 33 controls in 79.71s, no warnings, strict
exit 0 and all 33 DB censuses empty. All three target retirements have registry
zero and no physical files. Its retained-file branch never runs, so it does
not explain the failing runs. No collector, extra close, warning filter,
production or fixture change follows from these clean diagnostics.

The second-profile modal fixture still closes its own database before the
activation fixture's runtime disposal, unlike the registered departed-profile
owner. Independent review identifies that ordering gap but does not establish
it as the cause of either intermittent physical-retirement failure. A repair
requires a failing exact-owner proof; the existing failures remain unwaived.

Raw logs under `/tmp/switcher-emergency-context-ciQIs5`:
`affected-corrected.log`, `departed-profile-red.log`,
`departed-profile-trace.log`, `profile-owner-trace.log`,
`profile-session-trace.log`, `profile-quiet-trace.log`,
`profile-native-origin.log`. The isolated log's historical “red” filename does
not change its passing outcome. SHA256: broader failed log
`3ff0725a21c0db876b3966f39e2d390102692b29a6beaec341241443e1434d54`;
native-origin log `e151abfe442e8282a7e6619485aef5436bd6bc6c6799808661427f9ea18cfe9d`;
temporary native-origin script `784c9107b7a0a26bf670af54350f56ac55ceb7a374c28cb81afe5ced83e00869`.
Temporary logs/scripts are retained locally, not portable archived evidence.

### Earlier published-head CI failure and matching dev baseline

Perf Guard run [37371292704](https://github.com/rmusser01/tldw_chatbook/actions/runs/37371292704)
at `51860b4558` passes latency and boot-budget
steps but fails the tested-evidence storage census: credential polling bills
**10.125 os.open/tick**, above the unchanged **6.75 × 1.05** ceiling. Exact
dev base `74557e202a`, run
[37299377360](https://github.com/rmusser01/tldw_chatbook/actions/runs/37299377360),
fails the same variant and value.
This matching failure predates the PR; it is not a passed CI gate or proof of
the exact failing caller. The earlier PR2953 native-pause attribution lesson
describes this failure class, not a stack captured in these Linux runs.
No budget, native admission, monitor or production policy was changed, and no
blind workflow retry was requested. Artifact run **37371292729** remains
pending at this checkpoint. Qodo remains credit-blocked, with no actual review;
CodeRabbit's draft-skip status is not a substitute.

### Exact finite-operation replacement retirement — TASK-34411

A deterministic control parks the installed character scope-metadata read
inside `operation_owned_connection` before SQL. Exact-file quiescence retires
the worker's original borrower; after quiescence resumes acquisition, the read
opens a different native handle. The old entry-time `borrowed` boolean skips
retirement even after physical callback completion. This establishes a concrete
shared-helper defect, **not** the cause of the uncaptured intermittent profile
failure recorded above.

The corrected repository regression records **7 failed / 7 passed in 1.99s**:
warm character metadata and success/error replacements for all three installed
raw core owners fail native closed-handle assertions; cold metadata and
unchanged active transactions pass. The error crosses the actual ownership
guard's `finally`. The preceding draft test also recorded 7 failures / 7 passes,
but its nested error catch did not establish error propagation through the outer
guard; that receipt is retained and superseded by the corrected control.

The fix captures the exact original native object on both supported routes and
retires a different current handle after operation completion. Existing
owner-thread close/refusal, memory/custom routing and async cancellation behavior
are unchanged. No new lifetime, native admission, collector or cleanup policy is
introduced; existing ADR-126 applies, with no new ADR required.

Targeted verification: **136 passed in 64.73s, no pytest warnings, strict exit 0**.
All 136 parent-process post-teardown observations contain zero test database
files. Existing subprocess Library controls independently assert native closure
and preserved borrowers; the parent census is not a child descriptor census.
The new test is Ruff/formatter clean; the edited helper region formats clean,
with the same six inherited whole-file Ruff diagnostics and unrelated formatter
changes as its base. The existing formatter ratchet, whitespace check and all
eleven derived-artifact guards pass. Independent scoped source/test review found
no actionable findings. The complete nine-file affected batch records **363
passed in 574.06s, no pytest warnings, strict exit 0**; all 363 post-teardown
observations contain zero test database files. The unchanged 1,000-send control
passes in 387.96s with its original count, assertions and timeout. The
failure-conditioned retention probe reports `null`: this passing run does not
capture or attribute the old intermittent failure.

The new regression is included in the existing admission-sensitive PR Fast Lane
invocation, separate from the sandboxed cohort as required by its
`bootstrap_profile` marker. No job, dependency or timeout is changed.
The local combined invocation records **280 passed, 2 existing xfails, 6
warnings in 379.97s**, but the extra strict parent resource observer returns
**exit 1**. Its first database retention appears at the existing manually
mounted startup-app test, after the new 14 native-retirement cases recorded no
database files. Later observations retain Library, Workspace, Evals and
Subscriptions handles; the session descriptor warning reports growth of 598.
The other five warnings are four source-parser escape warnings and a synchronous
Notes test with an asyncio marker. Pytest also warns during cleanup of older
default-root garbage directories. No unrelated directory was removed and no
warning, threshold or ownership assertion was suppressed.

This invocation had no explicit `--basetemp`: the observer covers named private
profile roots, not every pytest temporary path. It does not prove full cohort
retirement, application-lifetime qualification or a new cause for the earlier
intermittent failure. The failed receipt is retained at
`/tmp/pr3024-replacement-ci-cohort.log`, SHA256
`cd6dd40fd8e6d421edee7b7223b5128ba1b47c5e5e3e393132aec6ed99f6fa9e`.
All eleven artifact guards pass with this CI selection; incremental independent
review found no actionable source or workflow finding. Resource qualification
remains HOLD.

Local receipts: `/tmp/pr3024-replacement-red-corrected.log` (SHA256
`02116e1e9b7916828390adc2780aabc8ca7c203b465ae6c4f0f110e3ad0eea20`)
and `/tmp/pr3024-replacement-green.log` (SHA256
`6a92becff8a480faf9f01f4aaddd00977ef9674291a28f0023cf3b47a38f3bb7`).
The complete affected receipt `/tmp/pr3024-replacement-affected.log` has SHA256
`56d8b421bd063d11f4d9e8b72df233ae1992937f6abcdf0c87d706f17072eb9e`.
Temporary receipts are retained locally, not portable archived evidence.
Native, Windows, participant, latency and external-review gates remain HOLD.

The earlier exact published head `7e7fb51ee0cbfe2eb01a162b20f38ec574fd0e34`
has a successful [Perf Guard run 37377248523](https://github.com/rmusser01/tldw_chatbook/actions/runs/37377248523),
with both tested and untested credential-poll variants at **6.75 os.open/tick**
under the unchanged ceiling. That is evidence only for the named older head,
not a demonstrated cause or repair of the older 10.125 result, nor qualification
for subsequent helper or CI-selection patches.
Qodo is credit-blocked; CodeRabbit's explicit full-review request terminates with
no files to review because organization path filters exclude all 72 changed
files. Neither skipped review is a clean external review. Review-configuration
direction remains pending; no organization settings or PR-local override was
changed.

### Latest-dev integration and selected rename waits — 2026-10-05

Rebased onto dev `54ac2758af81730b3e8b9effdabde2cbf898275c` without conflicts;
all 42 replayed patches are identical by range-diff. Upstream changes hosted
provider normalization and model discovery, not the finite ownership helper or
its regression. All five affected provider contract files pass **263 tests in
6.94s**, with no pytest warnings or live API calls.

The initial native-retirement/rename smoke records **29 passed, 1 failed in
126.99s**, no pytest warnings and zero database files at all 30 teardowns. The
exception-tab refusal completes its actual write/error feedback, then passes an
empty selected worker list to Textual's `wait_for_complete`. Installed Textual
uses `(workers or self)`, so this admits unrelated cancelled background work.
The strengthened real-worker control reproduces **1 failure in 10.10s**, with
zero database files. No production rename failure is established by that wait.

Three test waits now use stdlib `asyncio.gather` over only the rename workers:
empty selections finish immediately, while selected-worker failures still
propagate. Refusal controls require actual error feedback and worker settlement,
cancel a distinct unrelated worker and verify its native `WorkerCancelled`
behavior separately. Saved/live title and toast assertions, blank-input controls
and deliberately cancelled rename controls remain unchanged. **30 affected
tests pass in 129.73s**, no pytest warnings, strict exit 0 and zero database files
at all 30 teardowns. Changed tests are Ruff/formatter clean; independent scoped
review found no actionable findings. Existing ADR-085 applies; no new ADR or
production policy change.

After the import-order-only formatting correction, all four deterministic
refusal cases also pass **4 tests in 20.37s**, no pytest warnings, strict exit 0
and zero database files at each teardown. Receipt
`/tmp/pr3024-rename-empty-wait-final.log`, SHA256
`a4195643845a82a3d2cd3bcf8598569f1097a3c1daae55744190e0a31e1bda1c`.

The initial latest-dev artifact pass attempt fails only the duplicate task-ID
guard: dev's Vercel vision TASK-34402 was created before our Console emergency
task. The younger Console record is now TASK-34412, with add-commit/creation
provenance and all current inbound references updated. The older upstream file
is byte-identical to dev; its scope and status are untouched. The failed artifact
receipt `/tmp/pr3024-rebase-54ac-preflight.log` is retained, SHA256
`4e06f11384d1bfe5d4287027c9375dadfbe61bef9a8afdb7bd931d16eef342f1`.

After task-ID reconciliation, all eleven derived-artifact guards pass; whitespace
is clean. Receipt `/tmp/pr3024-rebase-54ac-corrected-preflight.log`, SHA256
`4652125092d82f67edcfc15a51476afdd002998ec9f2cebc1ed69627a585eeb9`.
This clears scoped draft publication only, not the qualification HOLD.

Separately, the first retaining startup-app case runs without the new regression:
its body passes in **12.40s**, but strict exit 1 still observes Library, Workspace,
Evals and Chacha database handles under an explicit fresh temporary root. This
confirms retention independently of executing the new regression, not its full
cause or a repair. Application-owner qualification remains HOLD.

Receipts: `/tmp/pr3024-rebase-54ac-provider.log` SHA256
`110e416606b62f92d959d1c64e3a855b63b697f57ba97d4cfedd4bb1948022b3`;
initial smoke `/tmp/pr3024-rebase-54ac-retirement.log`
`8be912b7f0acef6b005d6fd44f4f6ce6dae86f136693467a3ddb6ee899003752`;
deterministic RED `/tmp/pr3024-rename-empty-wait-red.log`
`405baa26f1a32a08c9129db9ea223b8971b0d8e2fb663f1da55d876d42ab4b56`;
GREEN `/tmp/pr3024-rename-empty-wait-green.log`
`7605cb8d0f5c59f8e38d43115edaba53de32cf48e4078541a308f39085f2a508`;
isolated startup `/tmp/pr3024-rebase-54ac-startup.log`
`2fe6d9b55f9b366c917c49644722cbeb83d34eba79fbf19488f41526453fcab9`.
These local receipts are not portable qualification archives. Native, Windows,
actual participant, measured latency, application-owner and external-review
gates remain HOLD.

### Current-head CI inactive-header settlement — 2026-10-05

Published `d60adb9ff616609a14028990e01a9acfb2e682d9` UI Fast Lane shard 3
records **1 failed, 552 passed, 1 warning in 846.71s**. Only the final
`tab-inactive` header assertion fails: immediately after generic activation,
the header still reads `Conversation | Other chat`. All preceding saved/live
title, switcher and confirmation-order assertions pass. Generic activation
requests the incumbent broad UI sync; an already-running pass coalesces that
request and returns before header publication. This differs from the strict
Character exact-ready boundary, which already awaits its renderer.

The unchanged isolated case passes in 7.25s, with unrelated pytest cleanup
warnings for older temporary garbage directories. A test-only hold before the
real sync's tab/header publication deterministically reproduces the same
premature assertion: **1 failed in 9.57s**, no pytest warnings and zero database
files at teardown. This is a scheduling/observer RED, not a production rename
fault. Both inactive variants now release and join that exact real sync before
the unchanged header assertion; no production activation, confirmation ordering,
warning filter, GC/lifetime policy or existing timing limit changes.

All four active/inactive rail/tab cases pass **4 in 21.97s**. The complete
affected file passes **16 in 67.91s**. Both have no pytest warnings, strict
process exit 0 and zero database files at every observed teardown. Independent
scoped review finds no actionable issue. Task status and broader qualification
HOLD remain unchanged. These narrow green results do not replace current-head
CI or the failed full latency/native/application-owner gates.

The formatter-final four cases separately pass **4 in 27.87s**, with no pytest
warnings, strict exit 0 and zero database files. Changed-test Ruff/format,
whitespace and all eleven publication artifact guards pass. Production,
package, scripts and workflows are unchanged by this checkpoint.

The CI warning is separate: the unchanged first-run cancellation test harness
records `run_worker` but discards its navigation coroutine. Its test file is
byte-identical to dev. It remains recorded, not suppressed or classified as
warning-free production shutdown. The required Derived Artifacts job fails its
UI-lane dependency even though all eleven source-reproduction guards pass;
the same-head Perf Guard succeeds with its separately recorded headroom warnings.

Local receipts and SHA256s:

- CI shard log `/tmp/pr3024-d60adb9-ui-shard3.log`:
  `76064240352c57e9ff72c2e9161067e0556a756c7f64f9fdd41500799e4fde38`.
- Deterministic RED `/tmp/pr3024-rename-header-red.log`:
  `a04d5d61e062df02e130509cbec9c5d3c80265bc02002b7893a7912145c89d35`.
- Four-case GREEN `/tmp/pr3024-rename-header-green.log`:
  `cb6cea54c64e11c1b114864fb90192bcfb95c670f5b66ebc5c37a2e3979ba6fb`.
- Complete-file GREEN `/tmp/pr3024-rename-header-module.log`:
  `7528bbf8536373225e0548c35e1d13de361a1e8b04488268efc1929b45568dd0`.
- Formatter-final GREEN `/tmp/pr3024-rename-header-final.log`:
  `88b01bed20c46950d38383edb07bd7611b45d3f33d34a3a3a21a9e919b65bc91`.
- Publication guards `/tmp/pr3024-rename-header-preflight.log`:
  `4652125092d82f67edcfc15a51476afdd002998ec9f2cebc1ed69627a585eeb9`.

These temporary logs are not portable qualification archives. Existing
ADR085/120 apply; this corrects a test settlement boundary, not an architectural
decision. Reproduce the affected behavior with
`python -m pytest Tests/UI/test_console_rename_consistency.py -p no:cacheprovider --tb=short`.

### Runtime test-owner adoption — 2026-10-05

At published `3e179b0f3a7929de114590c4481d985cfd059066`, the unchanged
manual-startup case passes in 6.95s but strict retirement exits 1 with private
Library, Workspace, Evals and Chat files. Its factory alias bypasses the existing
opt-in `owned_console_apps` capture; the attached Chat owner is also unregistered.
Using the actual capture binding and registering the exact attached database
clears that case: **1 passed in 7.10s**, no warnings, strict exit 0.

The first complete runtime/fixture control batch records **97 passed, 1 xfailed,
4 inherited AST SyntaxWarnings in 146.92s**, strict exit 1. Separate hand-built
hydration runtimes retain their receipt database. Their unchanged focused
three-case run passes in 9.18s but exits 1 with `agent_runs.db` files. The two
runtime tests now release held callbacks, dispose their exact runtime, then use
the existing close helper. Injected native-start rigs register their exact runs
and Chat owners for final retirement after successful app-runtime disposal.
Their existing controller shutdown remains; a mount failure before registration
retains the original post-shutdown closes. All 461 existing assertions remain
AST-identical. Shared fixture and production code are unchanged.

Independent review caught the unregistered mount-failure path. One real-SQLite
control exercises all three actual workflows and captures native handles:
without fallback cleanup **3 fail in 6.04s** at the closed-handle expectation.
Final failure controls, both hydration cases, startup/synchronous lifecycle and
all existing failed-drain/healthy/foreign-owner controls pass **24 in 19.11s**,
no warnings, strict exit 0, zero DB files at every teardown.

The intermediate complete two-file recheck remains **97 passed, 1 xfailed,
4 inherited AST SyntaxWarnings in 154.23s**, strict exit 1. Receipt/constructor
files are absent, but `runs.db` files first appear at the caret-only native-start
control and reach **15 descriptors** (nine database, three WAL, three SHM).
AgentRunsDB's close is current-thread only; neither that helper nor cancelled
hydration-task completion proves physical executor completion or all-thread
retirement. The final 24-case result does not waive this wider failed gate or
replace a full corrected rerun. No warning filters, forced GC, timeout/count
changes or new database-lifetime policy were added. Native, Windows, participant,
measured-latency, application-retirement and actual external-review HOLDs remain.

Local receipts (temporary, not portable qualification archives):
`/tmp/pr3024-runtime-owner-{red,green,suite,final,covering}.log`,
`/tmp/pr3024-runtime-receipt-red.log`, `/tmp/pr3024-runtime-rig-red.log` and
`/tmp/pr3024-runtime-mount-red.log`. The isolated rig's historical “red” filename
records a passing 1-case/strict-exit-0 run, not a product RED. Complete failed
recheck SHA256: `0fc6273b6d4d3e454afb87364b45dddd8ff7e8bfde18b740e5374a6058a41b48`;
mount-failure RED: `b22e47254f8b81f7654b377f141edebe1906a8da1555122030231d096410570b`.
Ruff remains 12 inherited findings with no additions; normalized formatter debt
remains 56 units with identical digest and no changed-line overlap. No new ADR:
this adopts the incumbent test-owner contract, not a production lifetime policy.
Independent final review finds no remaining scoped issue for partial draft
publication. All eleven artifact guards and whitespace checks pass. Covering
receipt SHA256: `63de612dbc535fbdd89d1429c3703a071ed432d1a6dd98b4e4149b6e36309c2a`;
artifact receipt `/tmp/pr3024-runtime-owner-preflight.log`:
`4652125092d82f67edcfc15a51476afdd002998ec9f2cebc1ed69627a585eeb9`.

### Finite run-log reader ownership — 2026-10-06

On published `c97e5aa57311ecf820cc78ddf50f16e19f2085c0`, a forwarding-only
native lease trace ran four real mounted start/acceptance workflows: **4 passed
in 29.64s**, no warnings, strict retirement exit **1**. The last two retain
one/two `runs.db` leases on exited executor threads with zero active operations
and no close-failure flag. Both allocation stacks lead through the installed
`_probe_console_agent_run_log` → `run_log_available` →
`_owning_run_id_for_log` → `get_run_metadata`. The observer delegates the actual
registration and stores only identifiers/path/stack strings; it never closes
handles or retains native objects. It is diagnostic, not latency qualification.
Trace `/tmp/pr3024-runs-origin.log` SHA256:
`98af1733c7e46a90c6c3de3f51b1779724fb1019fdbf993238a85c1f414b8912`;
temporary plugin `/tmp/pr3024_runs_origin_probe.py` SHA256:
`940a0e6835b07eca77f6f01baa86495d4ed85e8cc63eb0ce4f859d3367639128`.

Reuse the existing finite-operation guard around that one shared metadata
lookup, matching the neighboring resolver. No general connection-lifetime,
constructor/shared-cache, scratch-authority, cancellation, GC or budget change.
Real SQLite controls for all three log readers produce a valid **6 cold failures,
6 warm passes in 2.34s** before repair, then **26 passed in 3.88s**, no warnings,
strict exit **0** with zero database files, including incumbent replacement and
borrowed-transaction controls. Native closure is asserted before test cleanup.
The new regression joins the existing admission-sensitive CI invocation only.

The complete corrected runtime/fixture batch, without the allocation observer,
passes **100 tests, 1 existing xfail, 4 inherited AST SyntaxWarnings in 183.34s**,
strict exit **0**, zero database files at all **101** teardown checkpoints.
The existing log contract independently passes **34 in 8.19s**, no warnings.
The broader mixed rail run is **38 failed, 46 passed in 14.29s**; inspected
failures occur in unchanged configuration admission (`raw_source_selection_changed`)
before the repaired lookup. Its failure is retained, not counted as GREEN.
The first contract invocation without an explicit temporary root passed 34 bodies
but warned while pytest cleaned older unrelated garbage; no cleanup or warning
suppression was performed. The final contract receipt uses a fresh owned root.

Independent scoped source/test review finds no actionable issue for partial draft
publication. New test Ruff/format clean; production Ruff remains 29 identical
inherited findings and normalized formatter debt 160/160 with no changed-hunk
overlap (digest `e2d2644aa46035bb63c789d1be26f3f6c7c788be8c59ff75b80843ecf28a729a`).
Existing ADR126 applies; no new ADR. Previous failed receipts remain historical.
Native/Windows/participant/full-latency/application-retirement and actual
external-review HOLDs are not waived; tasks remain In Progress.

Temporary local receipts (not portable qualification archives):
`/tmp/pr3024-run-log-{red,green,contract,mounted,behavior-fresh,static}.log`.
RED SHA256 `1c1b2ebdbeff111a4c88699cdad1005415db3e441d6b4bb79bd6f46409720e90`;
GREEN `b97fc2ef8057f4bce63af8b1dbd8eaef11ed66f5998a0fcd457e844d757ed9f9`;
complete mounted `9ea582af17cfa162ec1aca591de72461f0a89eeee448371baf973bce181b09f4`;
contract `1b91b524d7f3c8c0ad3c4af0f490eca444b383956e5df5f8895464ee5776cb2d`;
failed mixed run `36a05f91cd50bf0368cc4ea287f53b756c9dd34be5f9a1ba0f2c313d78cedf95`.
All eleven source-reproduction guards pass; final task-file guards and whitespace
also pass after documentation edits. The two affected CI contract modules pass
**23 in 1.02s**, no warnings. Artifact receipt `/tmp/pr3024-run-log-preflight.log`
SHA256 `4652125092d82f67edcfc15a51476afdd002998ec9f2cebc1ed69627a585eeb9`;
CI contract `/tmp/pr3024-run-log-ci-contract.log`
`6523a39bdb6a8e82597177020e2b065f8e5264f1287b4e87d15939d1e9d37228`.

### Close fixture authority and interrupted initialization — 2026-10-06

Published source `ed4b0541987b8ef52b9a02c432aa55bbe904ee64` fails UI shard 2
at the prepared surviving-child authority check. A real Console refresh replaces
the fixture's controller-only namespace bridge with the installed runtime bridge.
The fixture now installs a complete incumbent `ConsoleAgentBridge` through that
runtime, and explicitly exercises the real refresh before its original checks.
All **208 original assertions are AST-identical**; timeouts, native clicks,
declined decisions, cancellation and fleet checks are unchanged. Existing opt-in
fixture capture owns constructor databases and explicitly registered Chat/runs
files, retiring them after successful runtime disposal.

The first behavioral correction still fails strict resource retirement. Native
allocation tracing identifies two finite request reads: requester attribution
opens the first runs handle before the later source observer could borrow it.
Both now use the existing `operation_owned_connection` at their respective
boundaries, outside locks and human waits. Real SQLite RED has **four cold
failures/four warm passes**; final eight controls also prove remembered-grant
success, unknown-requester denial on actual SQL failure, copied parent/child
identity and preservation of a warm transaction. No permission or cache policy
changes.

That correction removes runs-file retention but still leaves Chat files. Actual
quiescence succeeds with zero registered handles, with no later registration;
the forwarding-only getter trace then captures the real missing cleanup path:
`RuntimeError(database_maintenance_in_progress)` during an initialization PRAGMA,
with the native handle **open, unpublished and unregistered**. The initializer's
existing cleanup caught only SQLite/path exceptions. Extend that same cold-owner
cleanup to escaping exceptions, keeping existing SQLite/path conversion,
rethrowing other original exceptions and preserving failed-close evidence.
No registry, admission, borrower, transaction, GC or shutdown policy changes.

The deterministic real acquisition/quiescence race fails before repair at the
native-closed expectation, then the complete quiescence file passes **10 in
4.07s**, no pytest warnings. The uninstrumented complete Close file plus eight
request controls and ten quiescence cases passes **21 in 127.36s**, no pytest
warnings, strict exit **0**, zero database files in each of the three private
Close children and every outer teardown. Bootstrap admission/lease files are
still present in the outer census; zero database files is not zero resources.
Final ownership/quiescence/CI contracts pass **43 in 7.22s**, no warnings.
The earlier affected request-authority batch passes **201 in 64.14s**, no warnings.

The wider native-owner suite remains **1 failed, 59 passed in 18.97s**: its
ChaChaNotes case consumes the transaction's already-closed cursor after scope
exit. Temporarily reversing only this initializer patch leaves the database file
byte-identical to published HEAD and reproduces the identical failure (**1 in
1.25s**). The patch was restored. This separate existing contract finding is
retained, not silently fixed, excluded, or counted as GREEN.

Independent scoped review finds no actionable issue. Incremental Ruff adds no
diagnostics (controller 182/182, interrupt host 50/50, Chat DB 603/603,
Close 6/5, quiescence 5/5); normalized formatter debt is unchanged. New request
test is Ruff/format clean; whitespace passes. Initial failed static annotations
and diagnostic observer/startup attempts remain in their temporary receipts.
Both regressions join the existing admission-sensitive CI invocation only.
All eleven publication artifact guards pass after the documentation update:
`/tmp/pr3024-close-create-final-preflight.log`, SHA256
`4652125092d82f67edcfc15a51476afdd002998ec9f2cebc1ed69627a585eeb9`.
Existing ADR126 applies; no new ADR. Native/Windows/actual participants,
full measured latency, application retirement and actual external review remain
HOLDs; tasks and unchecked acceptance criteria are unchanged.

Temporary local receipts, not portable qualification archives:

- Getter trace `/tmp/pr3024-close-getter-nVOyTA/pytest.log`, SHA256
  `798cc6699f95176fad6a809244fc5d4d18b861c9dc57ce4356bbda5f3d6718f5`;
  its private child `0c70968602d295f8074a427783ea0dc02672429f45151cf69a00b2d4d8982fce`.
  Forwarding plugin `/tmp/pr3024_close_getter_failures.py`:
  `97457958326cdfedc9d591ead7d1e9ba8505dc2a33dd0060d690282cbc7bc746`.
- Deterministic race RED `/tmp/pr3024-init-race-red-xYvHTC/pytest.log`:
  `e9e2e73075b65394b128334368f278b9b964e6ae5a15c575852beee3e170fcac`;
  ten-case GREEN `/tmp/pr3024-init-race-green-GLo7x7/pytest.log`:
  `fe0f5d7111421e7ea4e6e18a72be3511d05897fa8e1aa0d5fa3a91cc5b9bef74`.
- Complete strict GREEN `/tmp/pr3024-close-init-green-KMHIPz/pytest.log`:
  `afbd5b1941fef20715bd7fe9af3de2de0c459c542b676aa53085f5d8dbcb627b`;
  pending/fleet private child:
  `2ab3aa381c6d0a545c139a5baaf0b26ec091816b25bab4b0e5cb6062cec8c8a4`.
- Final controls `/tmp/pr3024-final-owner-ci-controls-JhQRSs/pytest.log`:
  `0a05b05edbb49655cd3f01abc587ddb89e951966db1780b0ec133ad31ad47e51`;
  static `/tmp/pr3024-close-create-static-verified.log`:
  `f395fcdfa18968ca29b62f32a5690742a75b6bfac749efd682b2b13e34a14da3`.
- Separate failed owner suite `/tmp/pr3024-core-owner-controls-C98UB8/pytest.log`:
  `a1cbe07ae7a029e0aceaf7c9fd20119e155132cde46399d18b84f85952590a9b`;
  exact pre-fix comparison `/tmp/pr3024-core-cursor-baseline-1N5LtW/pytest.log`:
  `b28f45e37fe9bf655d71713ee0a50b3e5f86920c0e10a6778098f2abf183045c`.

Reproduce the strict corrected gate from the repository root:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD/Docs/QA/task-31245:$PWD" \
PYTEST_PLUGINS=descriptor_census_probe TLDW_TEST_REQUIRE_FILE_RETIREMENT=1 \
python -m pytest Tests/UI/test_console_session_tab_close.py \
  Tests/Chat/test_console_chat_create_connection_ownership.py \
  Tests/DB/test_chachanotes_connection_quiescence.py \
  -p pytest_asyncio.plugin -p pytest_timeout -q --basetemp=/fresh/owned/root/pytest
```

### Latest-dev integration of the Close repair — 2026-10-06

Rebased all 47 patches from `3877daa109f8dfee80162ef74eba1ab30782e369` onto
dev `f99ad86ebd9189fd477992c8c6c3976bc2297ef1`; tested combined head
`754a926df6e6147a1f79264d47dfbb5fdaf574e3`. Range-diff retains 43 identical
patches; four differ only in additive UI-census floor/context. Keep upstream's
three UI modules plus all PR additions: 151 unique paths, floor 150. Both
upstream CI invocations and every PR selector are preserved. No production
conflict; the Chat DB and three directly affected test files are byte-identical
before/after replay. Upstream receipt visibility, first-reply/window and recovery
changes remain intact. No generated-artifact shortcut or protection bypass.

Fresh combined-source evidence: complete strict Close/request/quiescence **21
passed in 123.56s**, no warnings, strict exit 0, zero database files in all
three private children and outer teardowns; request-authority/ownership/CI
**201 passed in 73.42s**, no warnings. All eleven artifact guards and whitespace
pass. Incremental static remains unchanged against the rebased repair parent;
controller formatting debt is now 35/35 due to preserved upstream code, not
the previous-base 23/23. All 208 original Close assertions remain identical.
These are integration tests, not current-head native/scale qualification.
The documentation-only receipt commit following this tested head changes no
production, tests, packages, scripts or workflow inputs. Broader HOLDs and the
separately reproduced existing cursor-lifetime failure remain unwaived.

Local receipts (temporary, not portable qualification archives):

- `/tmp/pr3024-rebase-f99-close-NWEllU/pytest.log`:
  `3051605c7f4b700cbc44f6227eb39eb76dbf5622d01c16ed2c76880cc9eaedb3`.
- `/tmp/pr3024-rebase-f99-authority-XfAKf7/pytest.log`:
  `e77aba2b55c1be0c1f8375f7b723a600f734f2e4ec57be8f967ffac61c50f1e7`.
- `/tmp/pr3024-rebase-f99-preflight.log`:
  `6776090263b415414701e2389873ac4018e9b00e8b4f05be348daf172f58321a`.
- `/tmp/pr3024-rebase-f99-static.log`:
  `7847ec4f7932c691131e51fb8302f6e2a602c7b8eaa28fe87fb5c03aef93c49a`.
- `/tmp/pr3024-rebase-f99-range-diff.log`:
  `468b7e5ae610fec2c618a0f6b0edf240c8345fc6089b1c4310b45fb0391d9981`.
