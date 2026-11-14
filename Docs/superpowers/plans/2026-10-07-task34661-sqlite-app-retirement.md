# App-scoped SQLite retirement implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline. Steps use checkbox (`- [ ]`) syntax. Do not control other sessions or dispatch implementers.

**Goal:** Retire eligible app-created native SQLite handles without revoking shared borrowers or hiding uncertain physical work.
**Architecture:** Keep the existing composition/shutdown root and source-owner close APIs. Remember Chat's pre-existing handle identity without acquiring one, drain existing producer owners, then retire only eligible current-thread caches after Textual teardown.
**Tech Stack:** Python >=3.12, installed Textual 8.x, real SQLite, existing pytest/private-profile bootstrap.
**Spec:** `Docs/superpowers/specs/2026-10-07-task34661-sqlite-app-retirement-design.md` (user approved).

## Global constraints

- No global owner sweep, registry, quiescence, GC/cache policy or dependency.
- No foreign-thread close, deadline/FD ceiling change or warning suppression.
- Preserve existing primary shutdown errors and failed-close native/cache evidence.
- Only targeted verification; no full sweep without opt-in.
- Raw aggregate logs remain local; publish audited outcomes/hashes only.
- Work only in the existing `codex/sqlite-interrupt-ownership` worktree. No push, merge or other-session controls.

ADR required: yes
ADR path: `backlog/decisions/225-app-scoped-sqlite-retirement.md`
Reason: accepted application-versus-process SQLite ownership and shutdown ordering; ADR036/094/126 remain governing inputs.

## Task 1: Native retirement and borrower/failure preservation

**Files:** Modify `tldw_chatbook/app.py` constructor and `tldw_chatbook/app_lifecycle.py` final shutdown; test `Tests/App/test_app_sqlite_retirement.py` and new `Tests/App/test_app_sqlite_retirement_contract.py`. The evaluation close boundary also needs a bounded correction in `tldw_chatbook/Evals/eval_orchestrator.py`: fresh native REDs show its old task census misses admitted preparation and its implicit close rolls back a transaction left by a cancellation finalizer. Keep its existing cancel/gather/source-close flow, fence intake, include already tracked preparation and refuse the live tail transaction.
**Interfaces:** Consume the existing source caches/close methods, `_core_cached_connection`, `_close_settled_core_cache`, native admission evidence and `EvaluationOrchestrator.aclose()`. Produce `LifecycleMixin._retire_app_sqlite_handles(self) -> None`, awaited only after successful producer and Textual shutdown. Constructor records a pre-existing shared Chat borrower, without a getter call.

- [ ] Retain fresh actual-app cold RED and write real native contract controls before production edits. In the contract fixture, build the five real owners and invoke the production lifecycle method on their actual field graph. Assert native SQL, not mock calls:

```python
await LifecycleMixin._retire_app_sqlite_handles(app)
with pytest.raises(sqlite3.ProgrammingError):
    captured_connection.execute("SELECT 1")
```

Warm Chat and active borrower controls instead require the exact captured handle's `SELECT 1` to succeed. A native-close error must retain its source reference and be observable; successful close is repeat-safe.

- [ ] Run `python -m pytest Tests/App/test_app_sqlite_retirement_contract.py Tests/App/test_app_sqlite_retirement.py -q -p no:cacheprovider --basetemp PRIVATE_PATH`. Expected: cold retirement fails on live native handles; new method controls fail because retirement is not implemented. Preserve setup errors separately.
- [ ] Implement the smallest fixed-owner retirement method. Inspect native/cache identity without opening a handle; skip custom/memory owners and unchanged live Chat borrowers. Refuse active raw/managed/tracked-cursor work or foreign/uncertain ownership. Reuse existing source close APIs; no raw native close or global sweep. Drain eligible evaluations with existing `aclose()`. Keep failed caches and report incomplete retirement. Follow this control structure:

```python
if owner_error is None and shutdown_error is None:
    await self._retire_app_sqlite_handles()
```

Catch retirement failure at the existing root error boundary, preserve primary shutdown/cancellation semantics and do not falsely retire when producer teardown already failed.

- [ ] Run the same two targeted modules. Expected: native cold closure plus warm/replacement/raw/managed/tracked-cursor/native-failure/foreign-thread/evaluation-task controls pass.
- [ ] Add root integration controls preserving a primary producer failure and real native references; run them RED against omission/wrong ordering, then GREEN.
- [ ] Review native identity checks against source close semantics and all production callers. Keep compatibility for partial app fixtures that have no eligible native owner. No unrelated lint cleanup.

## Task 2: Physical-boundary evidence and affected regressions

**Files:** Extend real-app/private-child checks in `Tests/App/test_app_sqlite_retirement.py`; update `Docs/QA/task-34661/README.md`, the Backlog task and incident lesson.
**Interfaces:** Consume Task 1's native retirement and existing selected private-profile process. No new production API.

- [ ] Exercise a real warm Chat borrower across actual app construction/shutdown, then an app-opened replacement after prior native retirement. Assert exact original native identity remains usable for the warm case and the replacement actually closes.
- [ ] Run repeated actual app sessions under isolated loops, then verify native SQL/descriptor state after physical executor join and before test-owned cleanup. Test-owned cleanup cannot be credited as production retirement. Darwin uses native `F_GETPATH`, not a readlink-only zero census.
- [ ] Run affected selections: `Tests/DB/test_operation_owned_connection_retirement.py`, `Tests/App/test_app_lifecycle_events.py`, `Tests/App/test_app_shutdown.py`, `Tests/Backup_Recovery/test_core_participant_lifetimes.py`, `Tests/Backup_Recovery/test_runtime_producer_settlement.py`, and `Tests/Chat/test_hooks_v2_teardown.py`. Use existing bootstrap/private-profile modes where required. Expected: affected cases pass; retain any genuine failure/warning rather than widening bounds.
- [ ] Run scoped Ruff/format, existing app-size/extracted-patch guards, both Backlog guards and whitespace checks. Compare inherited diagnostics rather than declaring large inherited files lint-clean.
- [ ] Record exact source/log hashes and audited outcomes. Keep uncertain resource owners and parent qualification requirements unwaived.
- [ ] Perform final branch review and proportionate verification before local source commit. No push or merge. If fresh-context review is not performed, state that limitation before integration; never infer review from tests alone.

## Execution record

Starting commit: `54954c0af5` on the isolated branch. Existing cursor/Interrupt repairs and their retained receipts are not to be reimplemented. This plan executes inline under the user's approved contract; no new execution-choice prompt is needed.

### Bounded implementation and remaining ownership boundary — 2026-10-07

Task 1 is implemented locally and its final native selection passed 32 tests
in 94.76s. The source close APIs retain raw/managed transactions, tracked
cursors, native-close failures and primary producer/Textual errors. Warm Chat
identity survives; app-opened replacements close. A separate thread's shared
Chat operation does not revoke the app thread's independent handle or become
authority to close the foreign handle. A new real-thread RED reproduced the
overly broad refusal before that distinction was corrected.

Implementation file addendum: the fixed five-owner function lives in
`tldw_chatbook/app_sqlite_retirement.py`; the root delegates to it. The existing
`_exception_frames` function and `_PUMP_DISPATCH_FRAMES` constant moved verbatim
to existing `app_keep_alive.py` and are re-exported by `app_lifecycle.py`.
AST identity and runtime binding controls preserve their original behavior.
This keeps app.py at 5712 lines and app_lifecycle.py at 2134, without raising
either existing ceiling. No new lifecycle framework or generic owner registry.

Task 2 is **not complete**. Repeated actual apps prove the five constructor
handles close after physical Runner/default-executor join, before test cleanup.
However, native database descriptors intermittently grow outside those five:
a retained failure records counts 0, 3, 6 and registered `db.agent_runs` handles.
Other bounded diagnostic runs pass; they do not supersede the failing plateau
evidence. Runtime AgentRuns UI/worker ownership requires a separate explicit
scope decision; no AgentRuns production code has been changed or foreign handle
closed. Creation-stack observation is not unique-source attribution.

The affected aggregate remains non-green: 183 passed / four failed. The two
real-app mount setup failures and one separate runtime-binding import failure
all pass when the original nodes use the existing bootstrap-profile fixture
(three passed in 5.91s). Two original MCP maintenance cases reproduce admission
timeouts without loading any modified production module; their code/dependency
inputs remain byte-identical to the base. This is not a waiver or green aggregate.

Scoped static checks retain the exact base diagnostic multisets for inherited
files; five new/clean paths pass Ruff/format. The existing eleven artifact guards
passed. Audited outcomes and hashes are in `Docs/QA/task-34661/README.md`; full
new raw logs remain local. All test/diagnostic sessions are finished. No full
sweep, source commit, push, merge or other-session control was performed.
The pending descriptor/AgentRuns ownership decision prevents final completion;
no fresh-context review or parent qualification is claimed.

### Task 3: Approved AgentRuns ownership extension — 2026-10-07

The user approved the pending AgentRuns extension. Keep Tasks 1/2 evidence and
the non-green aggregate intact; execute inline without other-session controls.
ADR required: yes; ADR path: `backlog/decisions/225-app-scoped-sqlite-retirement.md`.
Reason: applies the same accepted ownership contract to the remaining app store.

- [x] Trace every caller of the actual surviving native owner's acquisition and
  disposal boundaries. Distinguish recreation after disposal from an unclosed
  existing cache; creation stacks are not unique-source attribution.
- [x] Add the smallest real SQLite causal regression and run RED before changing
  production. Preserve native references so GC cannot masquerade as retirement.
- [x] Reuse existing source close/operation boundaries for the minimum root fix.
  Preserve warm borrowers, transactions/cursors and failed-close cache evidence;
  no foreign close, second registry or generic lifetime framework.
- [x] Verify the causal GREEN, affected receipt/runtime ownership controls and
  repeated actual app native/FD measurements after physical executor join and
  before cleanup. Run proportional static/artifact/Backlog guards.
- [x] Append audited outcomes and hashes without copying raw aggregate dumps.
  Keep the separate MCP failures and broader qualification gaps unwaived.

### AgentRuns execution record — 2026-10-07

Task 3's local source repair and native verification are complete. A real-native
late app accessor RED admitted a second AgentRuns owner; retaining the existing
disposed runtime fence passes the same case. A retained creation stack also
reached late bridge badge counts through Console attach reconciliation. Both
count APIs reproduced a cold native leak (two failed / two warm controls passed)
and now reuse `operation_owned_connection`; all four controls pass. This is a
supported causal gap, not unique attribution of every historical FD failure.

The existing settled-source helper replaces unchecked runtime close for exact
file-backed AgentRuns only. Three native REDs exposed false-success disposal of
raw/managed/failed-close cases; final nine native controls pass in 2.05s, including
an actual foreign managed borrower and explicit retry after its owner settles.
Custom/memory compatibility, cancellation precedence and sibling cleanup remain.
No AgentRuns schema, global registry, foreign close or budget change.

Actual cold/warm/replacement and repeated app checks pass (four in 47.72s).
The strong-reference observer finds no retained native AgentRuns handles and no
database descriptors at each of three physical-join boundaries, before cleanup.
An independent uninstrumented three-session plateau passes in 15.78s. Existing
five-store controls and the cursor/Interrupt receipts remain unchanged.

The original affected selection remains 61 passed / five failed / one warning,
not a green aggregate. Three failures directly refuse config source selection;
two turn controls fail on timeout/refusal in that ordinary fixture mode. Seven
original controls pass in existing bootstrap-profile mode; Home still refuses
its own profile redirection. Exact incoming source export reproduces Home's
same failure plus the unchanged scheduler-failure coroutine warning (one failed,
one passed, one warning). The earlier MCP maintenance failures remain unwaived.

Affected worker, diagnostic and UI-census guards pass; unchanged size checks
pass. The original extracted binding control passes in existing bootstrap mode
after its ordinary import/source refusal. Scoped formatting and clean test lint
pass; inherited Ruff code/message multisets remain runtime30/bridge28/lifecycle70.
Inline source review is complete; fresh-context review and integration are not
claimed. Raw logs remain local with audited receipt hashes in QA. No other
sessions, source commit, push, PR or merge were performed.

### Task 3 follow-up: physical run-worker ownership — 2026-10-07

The original mounted Interrupt regression now passes its body but fails final
runtime disposal with `sqlite_retirement_incomplete:AgentRuns` (10.40s; observed
again in 10.15s). This supersedes the earlier bounded completion wording: the
no-work app checks do not cover actual run-worker ownership. Native call-through
census identifies a live foreign `asyncio_0` handle opened by `run_reply`'s roster
read; its managed operations and raw transaction are settled, but its own worker
never closed the cache. Runtime disposal correctly refuses foreign closure.

Ruling: reuse `operation_owned_connection` at the existing reply wrapper and
shared roster-read boundary — the physical reply worker must own all its newly
opened AgentRuns work; both preview callers also use the roster read and have no
outer AgentRuns ownership guard. Cost if wrong: premature close of an active or
borrowed handle, pinned by actual native warm/error/cancel controls. No foreign
close, broader controller guard, extra registry or weakened disposal check.

- [x] Add real worker-native REDs for cold/warm reply completion, error/cancel
  paths and the sibling preview roster path before changing production.
- [x] Apply the existing ownership helper at those traced shared boundaries.
- [x] Run the same GREEN controls, original mounted Interrupt whole-process
  regression and proportional affected native/FD/static guards.
- [x] Append current results/hashes; keep Task 3 incomplete until those checks
  pass. Earlier receipts and separate config/MCP/coroutine gaps stay retained.

The first-acquisition aggregate census further distinguishes physical work from
a settled cache: the original mounted node still fails after its eight preceding
cases (eight passed/one failed in 32.61s). Its foreign reply worker is *active* in
the existing storage/agent admission scope at runtime close. The reply guard now
owns that handle, but disposal's census omitted the controller's already tracked
`_maintenance_background` tasks after the caller cancelled.

Ruling: include that exact controller task census in the runtime's existing
bounded drain, without cancelling the physical-owner task on budget expiry.
Keep the same original deadline and strict incomplete-native result if work does
not settle. Cost if wrong: shutdown may wrongly certify a logical cancellation
or revoke a physical owner's task; real settled/zero-budget controls pin both.
No new worker collection or unbounded join. The prior getter observer recorded
latest access frames, not first acquisition; retain that limitation rather than
claiming its roster stack uniquely identifies every open handle.

### Final bounded worker execution record — 2026-10-07

The four follow-up steps are complete locally. Cold worker/preview REDs fail
four cases with four warm passes; the shared ownership guards pass all eight.
A whole-module failure after a single mounted pass exposed a second gap: the
controller retained physical tasks after their cancelled callers disappeared.
The existing task set is now included in the unchanged runtime drain; real
settled/zero-budget controls preserve both completion and strict uncertainty.
The first foreign-SQL oracle failure is retained; corrected native-property
checks do not perform SQL from a foreign thread.

Original Interrupt module: 13 passed with native census and 13 passed
uninstrumented. Final memory compatibility RED established that the strict
close-error branch must stay exact file-backed only; the final current-source
native/compatibility/disposal selection passes 41 tests in 7.88s. Final three-app
physical-join child reports zero retained AgentRuns native handles and database
descriptors at all three boundaries before cleanup (one parent, 16.35s).

Scoped static succeeds with inherited diagnostics unchanged and changed ranges
formatted. All six extracted-patch controls and changed app/lifecycle caps pass,
but the broader architecture selection retains nine incoming size failures:
49 passed/nine failed in 12.66s. Their sources and ratchet test are byte-identical
to incoming base. No ceiling was raised or failure waived. Earlier config/MCP
and fake-worker coroutine failures remain retained. Current hashes/outcomes are
in Docs/QA/task-34661/README.md; full dumps remain local.

This supersedes historical pending native/worker wording only. Inline self-review
is complete; fresh-context review, static-gate disposition and integration are
not. TASK34661 remains In Progress with bounded AC verified; parents remain
unwaived. No source commit, PR, push, merge or other-session control.

## Task 4: Broader qualification record-read ownership

**Base:** ecf4b215973b1b96a85ac45e44fa4146ab5ccd42.
**Files:** Workspace registry record reader, existing native operation test and
original SessionEnd lifecycle fixture teardown; existing Backlog/QA records.
ADR required: no new ADR.
ADR path: `backlog/decisions/225-app-scoped-sqlite-retirement.md`.
Reason: apply the approved finite-reader guard and exact fixture-owner cleanup;
no new ownership, source, admission or shutdown policy.

- [x] Retain the broader 19-failure aggregate, isolated raw 55-pass control and
  forwarding acquisition/post-fixture census before implementation.
- [x] Extend the existing catalog native test to record lookup: expected RED is
  cold success/error closure failure; warm transactional borrowers must pass.
- [x] Reuse the existing owning-thread guard at `get_workspace`; preserve SQL,
  record construction, validation and error translation. Settle each manually
  built runtime before retiring its explicit runs and exact fixture Chat owner.
- [x] Verify the original complete SessionEnd/raw/native-pause sequence and
  affected Workspace service/reuse/size/static/artifact guards. Expected: all
  original assertions pass and post-fixture lease census is empty.
- [x] Commit locally, rerun source-bound wider timings, record every failed
  budget and external limitation, then perform one fresh whole-branch review.

Ruling: fixture Chat quiescence applies only after its actual controller settles
and only to its unique test-owned file; it is not production/global quiescence.
Cost if wrong: unfinished fixture work would refuse retirement rather than be
hidden; original bounded quiescence and strict raw-drain assertions stay active.
Ruling: retain all local evidence instead of deleting the plan workspace at
finish, as the user's explicit evidence-preservation requirement takes priority.
Source/evidence parent ecf4b2 completes all 60 searches/eight exact activations
but fails nine unchanged 50 ms UI interval gates. No budget or GC-policy waiver.

### Task 4 verified producer/fixture addendum

Ruling: reuse the existing cancellation-settled default-worker helper for the
actual Files callback and include only its existing group in root view drain.
Read-only original app trace pins an active Workspace operation at logical view
cancellation; cost if wrong is waiting for an unfinished owner rather than false
retirement. Existing runtime/process bounds and strict incomplete result remain.
Ruling: reuse existing opt-in exact app fixture owners after actual disposal,
and create the fresh navigation view only after real startup. Cost if wrong is
an unchanged native/stale-view refusal; no global sweep or weakened assertions.

Verified: native record22; SessionEnd/raw155/all leases zero; registry115/all
leases zero; app/Interrupt21 including physical-join child (two sandbox cache
warnings retained); affected Workspace142/all lease/native-FD censuses zero;
final mounted rail1. All52 size controls and eleven artifact guards pass;
inherited Ruff multisets unchanged. The initial sandbox-unavailable Mermaid
input and later legitimate successful check both remain. Audited receipts:
Docs/QA/task-34661/broader-qualification-2026-10-08.md. Source checkpoint,
fresh source-bound timing and whole-branch review are the remaining Task4 gates.

Source committed locally as dc03d6123f; committed causal/native/size gate88 passes.
Fresh exact clean-source scale Keyword passes all300 (P95110.965750ms, max loop
14.987083ms). Tiny smoke300 succeeds. Installed UI completes60 searches/eight
activations but fails seven50ms intervals, max113.955042ms; no cap/GC waiver.
Startup11 passes with three expected census warnings. Detailed audited results
and hashes append to the broader QA receipt. Whole-branch fresh-context review
is next; no push, integration or broader qualification completion is inferred.

Fresh final review of61e2869..c376905 covers every authored production section,
related callers/tests/fixtures and ownership contracts: no Critical or confirmed
Important production finding. One Minor historical extraction-checker final-HEAD
replay limitation is deferred. Actual native/Windows/human, whole-app aggregate
FD and full50ms UI gaps remain unqualified; no synthetic replacement or waiver.
Task4 bounded source/verification/review work is complete; integration and parent
qualification are not. Keep the worktree/branch/evidence; no push or full sweep.
