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

That corrected readiness revealed TASK-34402: permanent emergency detachment
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

### Broader resource recheck and emergency fixture context — TASK-34402

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
