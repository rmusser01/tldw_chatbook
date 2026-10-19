# Agent-assisted local archive recovery qualification

Date: 2026-09-27. Status: **automated qualification recorded; ordinary Capture-Off live recovery qualified**. The [final native continuation](2026-09-27-agent-archive-recovery-live.md#approved-follow-up-implementation-and-native-qualification) closes the earlier missing conversational selection/Open evidence. Earlier then-open live checkpoints below are historical; Capture-On and recorded aggregate resource limits remain unqualified.

Governance: [ADR-166](../../../../backlog/decisions/166-agent-assisted-archive-recovery.md),
[approved specification](../../specs/2026-09-17-agent-archive-recovery-design.md),
[implementation plan](../../plans/2026-09-27-agent-archive-recovery.md).
No new ADR is required: this qualifies and repairs the approved contracts under
ADR-166, preserving storage and authority boundaries and the existing design tokens.

## Environment and reproducibility

Feature revision before Task7: `d978edb6786562f73d2ce111f3bca5bea787431b`,
branch `codex/agent-archive-recovery`. Task7 changes were present uncommitted during
the original 22-file selection. Its hashes identify that phase; later affected
corrections, the resource probe, and the final review fixes have separate evidence
below. The report-containing commit is not a claim that the original run covered
all later bytes.
Interpreter: `/tmp/archive-recovery-venv/bin/python` (Python 3.12.11).
Working directory: `/Users/macbook-dev/.codex/worktrees/agent-archive-recovery/tldw_chatbook`.
Pytest fixtures use disposable configuration/data and real SQLite. Only the
external model and explicit race/failure/clock inputs are faked. Native evidence
uses the actual Console producer, runtime, catalog, AgentService and mounted
Textual event dispatch. No full test sweep was run.

The complete scoped selection is:

```sh
/tmp/archive-recovery-venv/bin/python -m pytest \
  Tests/DB/test_archive_chronology.py \
  Tests/DB/test_archive_recovery_query.py \
  Tests/Agents/test_archive_recovery_tools.py \
  Tests/Chat/test_archive_recovery_search.py \
  Tests/Chat/test_archive_recovery_result_store.py \
  Tests/Chat/test_archive_recovery_operations.py \
  Tests/Chat/test_archive_recovery_coordinator.py \
  Tests/Chat/test_console_archive_recovery_dispatch.py \
  Tests/UI/test_console_archive_recovery_cards.py \
  Tests/UI/test_console_archive_recovery_flow.py \
  Tests/Chat/test_conversation_archive_actions.py \
  Tests/DB/test_conversation_archive.py \
  Tests/UI/test_console_conversation_archive_flow.py \
  Tests/Workspaces/test_workspace_registry_service.py \
  Tests/Agents/test_library_name_reservation.py \
  Tests/UI/test_console_runtime_ownership.py \
  Tests/Chat/test_console_runtime_shutdown.py \
  Tests/UI/test_console_archive_recovery_boundaries.py \
  Tests/UI/test_design_token_governance.py \
  Tests/UI/test_css_bundle_sync_guard.py \
  Tests/UI/test_workspace_archive_review_regressions.py \
  Tests/UI/test_console_archive_cancellation.py \
  -q -p no:cacheprovider --basetemp=/tmp/task7-qualified \
  --show-capture=no --tb=short
```

The actual invocation used `/tmp/task7-qualified.py`, a temporary wrapper calling
`pytest.main` with exactly these arguments and printing loaded module paths after
the run. It placed the feature working directory first on `sys.path`; the final
log verifies package, ChatScreen, controller and recovery-runtime imports from
this worktree. Tested Task7 file SHA-256 values:
`8b26794b5767e7b9717e90b13a9ef0348c27d9abb5b02831d020cd744292c071`
for the UI flow test and
`861c86f4d19ba25530c4f2e8fd2eddacf74c8ce1b54481822d41331710942214`
for the recovery card source. Log: `/tmp/task7-qualified.log`.

Result: **486 passed, 12 failed, 0 skipped, 7 warnings in 1231.20 s**.
Exit status 1. All feature files passed. The twelve existing regressions remain
failed and are classified below. Four imported module paths printed at teardown
confirm this worktree's package, ChatScreen, controller and recovery runtime.

Six warnings are existing invalid-escape `SyntaxWarning` messages in the runtime
construction inspection. The seventh is a session-level file-descriptor growth
diagnostic: **12 → 741 (+729), limit 200**. It appeared at the last cancellation
node but is not attributed to that node or classified as baseline. This is an
unqualified resource concern; no broad repair or full rerun was performed.

## New qualification and corrections

`Tests/UI/test_console_archive_recovery_flow.py` adds eleven native cases:
three eligible lifecycle combinations with RAG-only/no index and active/active
Default exclusion; Monday-chat/Friday-workspace, unknown-only and proven-match
with another unknown event; system-only and tool-only needle exclusion; stale
rehoming; cancellation after an actual committed workspace write; and a local
turn followed by an external destination. Existing button/agent compact/wide
flows now also assert the original active leaf and the model invocation delta.

The destination case uses an actual prior on-device native turn and observes
the real session destination state at the next native provider invocation. It
proves the external disclosure contains the previous local identity, actual
resolved destination and exact attempt/message owner; it clears after completion.
This positive evidence is separate from older failing disclosure regressions.

The chronology tests initially failed because the native card displayed only
matching dates and omitted the unknown-range possibility label. The authorized
correction in `archive_recovery_cards.py` shows both archive
dates applicable to the captured lifecycle state, independently lists the date-match basis, and says “Archive time unknown;
may match this date range.” only when no known event proves a match. A proven
chat date with an unknown workspace date remains a proven match. Dates belonging to a resource captured as active are not displayed. No storage/API/permission change was made.

Iteration evidence: `/tmp/task7-new1.log` contains **8 passed, 2 failed, 14
deselected** (108.03 s); the two failures exposed the copy gaps. The first affected
check `/tmp/task7-copy-green.log` contained **1 passed, 1 failed, 27 deselected**
(19.92 s); remaining failure was the new test comparing canonical timestamps
without their normalized microseconds. The test now compares parsed instants.
`/tmp/task7-copy-final.log`: **8 passed** (27.72 s), all three chronology cases
and all five mounted card cases. The final integration result is recorded above.


### Post-selection focused corrections

Final spec review identified missing visible row numbers and inactive-branch
match labels. Native Enter checks also proved that the transcript consumed Enter
before plain result Restore and receipt Open buttons could activate. The focused
RED command selected `test_native_rows_number_immutable_page_and_label_inactive_match`
and the two `[enter-restore]` / `[enter-open]` keyboard cases in the UI flow file:
`-q -p no:cacheprovider --basetemp=/tmp/task7-row-keyboard-red --show-capture=no --tb=short`.
Result: **3 failed in 48.36 s** (`/tmp/task7-row-keyboard-red.log`).

The authorized narrow correction prefixes each row with its immutable page
ordinal and labels an inactive-branch text match without promising a branch
jump. `RecoveryActionButton` owns local Enter/Space activation at result,
manual-Library fallback and receipt sites. The exact-emission DecisionButton
inherits this handler while retaining its captured binding/Pressed override.
Native disabled/active readiness guards, source identity and authority remain
unchanged; no global transcript binding or stylesheet changed.

A separate lifecycle probe proved the explicit new recovery fixture left a real
worker-owned ChaChaNotesDB handle open after its current-thread-only teardown:
`test_explicit_recovery_fixture_closes_real_worker_connections`,
`--basetemp=/tmp/task7-fixture-red` → **1 failed in 3.78 s**.
The fixture now quiesces its same-file registry after work settles. The same
pattern in the new explicit runtime thread-owner search test also uses
quiescence. The pure synchronous search fixture was inspected and left unchanged.
No cached-config or legacy fixture/production pool was changed.
`/tmp/task7-cleanup-green.log`: real file/memory owner combinations and retained
worker-handle closure → **5 passed in 10.76 s**. This proves those handles close;
it does not explain the full +729 growth or establish a process-wide plateau.
The incident is recorded in `backlog/docs/lessons-testing-evidence.md`.

Affected command after correction:

```sh
/tmp/archive-recovery-venv/bin/python -m pytest \
  Tests/UI/test_console_archive_recovery_flow.py::test_native_rows_number_immutable_page_and_label_inactive_match \
  Tests/UI/test_console_archive_recovery_flow.py::test_focused_native_recovery_actions_activate_from_keyboard \
  Tests/UI/test_console_archive_recovery_flow.py::test_native_date_search_labels_monday_chat_friday_workspace_and_unknown \
  Tests/UI/test_console_archive_recovery_flow.py::test_production_recovery_keeps_exact_history_and_active_draft \
  Tests/UI/test_console_archive_recovery_cards.py \
  Tests/Chat/test_archive_recovery_operations.py \
  Tests/Chat/test_archive_recovery_coordinator.py::test_actual_write_cancellation_is_joined_before_release \
  Tests/Chat/test_console_archive_recovery_dispatch.py::test_real_console_native_dispatch_waits_for_exact_confirmation \
  -q -p no:cacheprovider --basetemp=/tmp/task7-affected-final \
  --show-capture=no --tb=short
```

**54 passed, 4 failed in 257.74 s**, `/tmp/task7-affected-final.log`.
All four failures were the new keyboard guard harness: Textual explicitly blurs
controls when disabled, so the positive test must refocus after enabling; its
navigation spy must also filter unrelated Notify/Callback events. The test now
checks disabled refusal, refocuses, then checks real Enter/Space action and
original identity/draft/no-send behavior. No production retry masked a failure.
First keyboard-only correction: **2 passed, 2 failed in 42.45 s**,
`/tmp/task7-keyboard-final.log`. Positive Restore/Open activation worked, but
keypresses after disabled blur reached the composer and changed the draft.
The negative guard now uses an actual click on the disabled control, then
reenables/refocuses for the positive keyboard event. No draft or invocation
counter is reset to hide that harness failure.

Final affected command:

```sh
/tmp/archive-recovery-venv/bin/python -m pytest Tests/UI/test_console_archive_recovery_flow.py::test_focused_native_recovery_actions_activate_from_keyboard -q -p no:cacheprovider --basetemp=/tmp/task7-keyboard-green --show-capture=no --tb=short
```

**4 passed in 35.50 s**, no warnings, `/tmp/task7-keyboard-green.log`.
Together the post-correction selections qualify **62 distinct affected nodes**:
the 58-node correction selection with its four keyboard nodes corrected, plus
four file/memory thread-owner nodes. The retained-handle closure node overlaps
these selections. Intermediate failures above remain recorded; neither that
union nor absence of a smaller-run warning establishes a whole-process FD plateau.

These affected phases are separate from the 22-file selection and its original
byte hashes. That selection was not repeated. The durable compact/wide exports
below now come from passing core cases in the affected correction run.

Manual restoration leaves the source session/draft and persisted message IDs
unchanged. Explicit Open is triggered through its actual receipt button event,
then the original-ID resume consumer verifies original persisted conversation,
active leaf and both active/inactive message branches. Manual Restore/Open does
not increase model calls. An agent awaiting its own confirmation has exactly
one ordinary post-tool continuation; this is distinct from a new user send.

## Specification review

The table names concrete covering files; all test paths are relative to the
repository. Prior migration-specific evidence remains in Task1's report:
`test_conversation_archive.py` and `test_workspace_db.py` use real fresh/upgraded
SQLite and forced migration rollback. Task1's scoped selection passed 164 tests.
This does not invent a new Task7 run of files outside the command above.

| Specification section | Implementation and concrete evidence |
|---|---|
| 1 Purpose and local inventory | `archive_recovery/search_service.py`; native lifecycle/RAG matrix and button/agent flows; soft deletion/Default/missing owners in `Tests/Chat/test_archive_recovery_search.py::test_mixed_lifecycle_default_missing_workspace_and_deleted`. |
| 2 Architecture and alternatives | Authenticated provider in `Agents/archive_recovery_tool_provider.py`, catalog seam and runtime injections; `Tests/Agents/test_archive_recovery_tools.py` plus actual AgentService dispatch file. Shared Library/MCP descriptor contracts stay separate. |
| 3 Authority/data exposure | Native Allowed/Blocked/Direct/RAG/kill/persona/queued/wake cases in `Tests/Chat/test_console_archive_recovery_dispatch.py`; positive native destination transition in the UI flow file; no fake actor permission. Manual refresh isolation and inert imported JSON use mounted controls. |
| 4 Search/chronology | Search/request validation and SQL query files: timezone boundaries, either-event Monday/Friday oracle, undated candidates, query budget/locks, filter-before-limit, role predicates/snippets, inactive message provenance, Unicode byte-fitting/cursors. New native date/role cases cross provider→storage→card. |
| 5 Stable results/presentation | `result_store.py` and its tests: immutable source-bound references, ten-page/thirty-minute expiry, keyset/filter/session binding. Native imported/manual refresh/pending return tests and card queued-event tests preserve app provenance. |
| 6 Exact confirmation/restore | `operations.py`/`coordinator.py` tests: four scope states, transaction CAS, rename uniqueness, version/owner drift, two-store partial outcomes. Actual mounted Confirm and Restore as events; display copy correction above. |
| 7 Ownership/results/navigation | Coordinator cancellation/dedup/receipt eviction/teardown tests, actual runtime close/disposal dispatch tests, native cancellation after committed workspace write and explicit Preview/Open original branch/draft assertions. |
| 8 Persistence | ChaChaNotes migration v74 and Workspace v9; real chronology, sync/log/index and revision/clock tests; prior Task1 migration rollback/fresh/upgraded evidence. Writer audit below. No event history/backfill/sync payload added. |
| 9 Integration boundaries | Focused query/operation/runtime/widget modules; actual view-hook inventory and runtime ownership tests; archive-boundary import guards. No new source imports or boundary in Task7. |
| 10 Verification/docs | Scoped command above, Python static/link/whitespace checks below, design-token and generated-CSS guards, three updated user guides. Automated compact/wide SVG/compositor artifacts are separate from missing live-model evidence. |
| 11 ADR/handoff | Existing ADR-166 linked here and in all three guides. Task33020 remains In Progress because live AC2 is not satisfied. |

All ten section-10 verification requirements map to these rows: migrations (8),
four lifecycle states (1/6), bounded chronology (4), authority/modes (3), both
entry points (6), stale identity (5/6), cancellation/replay (7), preserved
identity/drafts (7), mounted/token behavior (9/10), excluded-role/import provenance
(3/4/5). Baseline failures below remain failures; coverage is not an all-green
regression claim.

## Archive writer audit

| Writer or caller | Review and evidence |
|---|---|
| `DB/ChaChaNotes_DB.py::set_conversations_archived` | Immediate transaction conditions version and prior archive flag; sets observed UTC archive time, clears on restore, advances existing version; no-op stays unchanged. Chronology and conversation-archive real-store tests cover sync retention and branch/ID preservation. |
| `Chat/conversation_archive_actions.py::change_conversation_archive` and local service delegation | Existing shared reservation/service seam reaches that writer; archive-actions regressions verify partial/busy/version behavior. No second writer added. |
| `Workspaces/registry_service.py::archive_workspace` | Immediate transaction captures/checks prior archive state/revision; conditional update sets time/increments revision. Registry/chronology tests cover repeated/backward clocks and concurrent episodes. |
| `Workspaces/registry_service.py::unarchive_workspace` | Same-transaction state/revision/name and active-name uniqueness check, clears time/increments revision. Operation/registry tests verify drift and duplicate-name rollback. |
| Registry Default repair and creation | Creation is active with null time/revision zero. Actual Default repair conditionally clears archive time and increments revision for a real transition; Task1 real-store/registry tests cover repair. |
| Console `workspace.py` archive, restore, Restore as and Undo | Selected archive revision and restore name reach registry transaction; Undo reuses the same restore path. `test_workspace_archive_review_regressions.py` covers changed episodes and confirmation revision drift through actual callers. |
| Settings `settings_screen.py` archive/Undo/immediate Restore | Archive and Undo supply captured revision/name; immediate Restore uses transaction-captured current state. Same regression file tests caller races/storage failures/cancellation. |
| Library/Console exact resume `UI/Console_Modules/archive.py` | Explicit workspace lifecycle path delegates to registry; chat restore delegates to shared actions. Recovery operation adds captured revision/name; no identity or membership rewrite. Existing resume failures listed below are not hidden; new native Open qualifies current producer/branch/draft behavior. |
| Agent `archive_recovery/operations.py` | Reobserves target, workspace-first conditional write then versioned chat write, truthful final observation and partial cancellation. No compensation, copy or implicit navigation. Real-store coordinator plus mounted post-commit race evidence. |
| Sync/export and migrations | New fields remain local: explicit shared payload/export selections exclude them. Migration recreates current sync triggers; no historical migration rewritten or dates backfilled. Task1 real SQLite sync/log tests and field-selection audit. |

## Bounded retained-owner resource qualification (fix round 1)

The review requested exact-owner evidence beyond an isolated passing run. One
retained mounted native Console, controller, recovery runtime and real SQLite
chat/workspace/run stores execute five serial search → native row Restore →
native Confirm → actual `chat_and_workspace` write cycles, then source close
and runtime disposal. Re-archiving uses the same target/stores; no fresh database
or app is created per cycle. Only the external model is the existing test gateway.
Production `storage_call`, native dispatch and storage methods remain intact;
wrappers record real connection return values without replacing them.

The test owns a real **two-worker** default executor, restored after cleanup.
This controlled topology permits a blocking native agent runner and marshalled
storage to overlap. A barrier dispatches inspection/close to both actual creating
workers, so thread-affine workspace/run handles are never queried or closed from
the wrong thread. The observer deliberately retains strong connection references;
closure assertions qualify explicit close behavior, **not** natural reference
release or GC behavior. Chat/workspace stores and the run store are externally
owned by the test fixture/helper; their lifetime is distinct from recovery runtime
shutdown. No production lifecycle defect or generalized pool fix is asserted.

Final exact command (same interpreter/worktree, isolated profile):

```sh
/tmp/archive-recovery-venv/bin/python -m pytest \
  Tests/UI/test_console_archive_recovery_flow.py::test_retained_native_recovery_resource_owners_plateau_and_close \
  -q -s -p no:cacheprovider --basetemp=/tmp/task7-resource-final \
  --show-capture=no --tb=short
```

**1 passed in 15.26 s**, `/tmp/task7-resource-final.log`. No warning threshold,
warning filter or GC setting was changed; every measurement/assertion occurs
inside the test, before its autouse fixture GC teardown. OS total uses the
sentinel's `/dev/fd` listing; `lsof` assigns numeric descriptors to the exact
three disposable database files and WAL/SHM sidecars. An available `lsof` is a
prerequisite (the node explicitly skips if unavailable). The committed
[content-free resource samples](artifacts/2026-09-27-agent-archive-recovery/resource-owners.json)
contain counts only, not conversation bodies, secrets or unrelated file paths.

| Measured phase | Process FDs | Live chat/workspace/run handles | Exact file FDs chat/workspace/run |
|---|---:|---|---|
| Stores created, before app | 18 | 1 / 1 / 0 | 3 / 3 / 0 |
| Each settled cycle 1–5 | **47 / 47 / 47 / 47 / 47** | **3 / 3 / 2** each | **7 / 7 / 5** each |
| Source closed | 47 | 3 / 3 / 2 | 7 / 7 / 5 |
| Actual runtime/helper disposed | 46 | 3 / 3 / 1 | 7 / 7 / 4 |
| Existing current-thread store close + chat quiescence | 38 | 0 / 2 / 1 | 0 / 6 / 4 |
| Explicit store close on both creating workers | **28** | **0 / 0 / 0** | **0 / 0 / 0** |

The final three settled samples must match exactly in process FD count, exact
file ownership, live handles and chat registry count. Actual two worker threads
plus main own chat/workspace handles; main plus one worker own run handles in
this execution. Scheduler-dependent warmup can allocate the second run worker:
a preceding passing probe held 49 FDs and 3/3/3 handles, also plateaued across
all five cycles, and closed every named handle/FD. The final command used the
hash below and supplies the table's exact numbers.

This attributes **ten observed descriptors** after current-thread teardown to
these retained workspace/run owners, and proves their explicit owner-thread
cleanup returns all three named files to zero open descriptors/handles. Other
app/process resources account for the remaining 28 total descriptors; they are
outside this bounded ownership claim. This is representative retained-owner
plateau and teardown evidence, **not** attribution of the original 498-node
**+729** growth, a repeated-app/many-worker stress result, or proof of a natural
GC lifetime plateau. The original warning remains unqualified independently
of this evidence and the unavailable live-model AC2.

Tested revision: fix base `30cf4e90ec19106a226e356697a67a29238564a9` plus
uncommitted UI test SHA-256
`4eb3fb3c9727e7f77a5d15416ee61259ab8e486a53382d56544dddf7460df2d2`.
No production source changed. The first probe used an invalid metadata accessor
and attempted a thread-affine close; the next one-worker topology deadlocked the
blocking agent/storage handoff and was stopped. An empty per-session search set
was initially mistaken for pending work; that assertion now checks actual tasks.
The next probe reached plateau and exposed current-thread-only fixture closure;
explicit creating-worker cleanup now qualifies that ownership. Final static
Ruff check/format and local document/whitespace checks are recorded in the fix
report. No broad selection, baseline export or live endpoint was rerun.

## Baseline regressions

| Current failing node(s) | Count | Diagnostic and prior evidence |
|---|---:|---|
| `Tests/UI/test_console_conversation_archive_flow.py::test_restore_resume_then_send_retains_original_history_and_unrelated_draft[False/True]` | 2 | Missing retired `_submit_console_native_draft` at line 323. Task6 reported both identical verified pre-task export failures; its report preserves the classification. |
| `Tests/Agents/test_library_name_reservation.py::test_library_reservation_is_one_derived_immutable_union` | 1 | Existing expectation 24 descriptors versus actual 21. Individually reproduced on Task5's verified `b990` export (`/tmp/task5-baseline.log`). |
| `Tests/UI/test_console_runtime_ownership.py::test_runtime_owned_custody_tracks_only_lifetime_handles` | 1 | Expected field inventory omits existing `archive_conversation_id`; Task5 exact-node export evidence. |
| `...::test_runtime_tombstones_before_shutdown_and_disposes_via_to_thread` | 1 | Existing extra to-thread event/order; Task5 exact-node export evidence. |
| `...::test_persistent_attach_sync_failure_has_bounded_backoff_and_resume_retry` | 1 | Existing test SimpleNamespace lacks `chat_controller`; Task5 exact-node export evidence. |
| `...::test_reconciled_view_keeps_each_live_poll_reason_and_one_timer[wake]` | 1 | Existing zero versus one poll assertion; Task5 exact-node export evidence. |
| `...::test_opening_console_during_a_headless_delivery_arms_the_poll` | 1 | Retired `delivering_session_id` fixture call; Task5 exact-node export evidence. |
| `Tests/UI/test_console_archive_cancellation.py::test_cancelled_archive_preflight_restores_only_own_keyboard_draft[visible/background/closed/closed-visible]` | 4 | All fail at shared line 48 accessing retired `_console_inflight_send_stashes`, before task creation or parameter-dependent owner/archive behavior. Task4's exported `4aad926884` proof (`/tmp/task4-baseline-ui.log`) explicitly reproduces **visible only**. The other three have the same current shared pre-action defect, not individual export evidence. |

All eight non-stash nodes have prior per-node baseline classifications. The
shared-stash limitation above is deliberately narrower than claiming four
individual export reproductions. None of these files was repaired or its
failed result counted as passing.

Previous exact-node export comparisons establish baseline provenance, not
positive requirement evidence. Task5 exported `b990` and verified imported module
paths for its failures; Task6 verified its pre-task export for the two retired
submit nodes. Task4 exported `4aad926884` and verified ChatScreen path for the
**visible** retired-stash variant only. The other three currently fail at the
same shared pre-action fixture line; they have no individual exported proof. These older fields are not restored to make fixtures
pass. New native tests use current runtime custody and `_admit_console_turn_to_runtime`.

## Static and artifact evidence

Scoped static commands (same interpreter/cwd):

```sh
/tmp/archive-recovery-venv/bin/python -m ruff check --no-cache Tests/UI/test_console_archive_recovery_flow.py tldw_chatbook/Widgets/Chat_Widgets/archive_recovery_cards.py
/tmp/archive-recovery-venv/bin/python -m ruff format --check Tests/UI/test_console_archive_recovery_flow.py tldw_chatbook/Widgets/Chat_Widgets/archive_recovery_cards.py
git diff --check
/tmp/archive-recovery-venv/bin/python /tmp/task7-doc-check.py
```

Ruff check passes; formatter reports two already formatted files; whitespace
check passes. Logs: `/tmp/task7-ruff-check.log`, `/tmp/task7-ruff-format.log`,
`/tmp/task7-whitespace.log`, `/tmp/task7-doc-check.log`. The document check
validates local Markdown targets and heading fragments in the three guides
and this report, comparing diagnostics with HEAD. Zero introduced diagnostics.
The agent-tools guide already has three repeated links to the missing
`../index.md#console-agent-runs-are-screen-scoped` heading; they remain baseline
limitations. No broad document repair or network link probing was attempted.
The final selection includes all design-token and generated-CSS bundle guards;
Task7 changes no stylesheet or token value. Final scoped Python checks cover
the UI flow, operations and dispatch tests plus both recovery UI modules: Ruff
check passes and formatter reports five formatted files. Logs:
`/tmp/task7-ruff-check-final.log`, `/tmp/task7-ruff-format-final.log`; final
whitespace log: `/tmp/task7-whitespace-final.log`.

Actual Textual whole-screen SVG exports and compositor text from the post-correction
mounted automated run are retained below. Assertions inspect the composed
scope, effects and receipt; Pilot performs actual Restore/Confirm clicks and
Tab focus. These synthetic-data artifacts have a fake external model and are
not live-model or raster-render evidence. Trailing blank cells in text snapshots
and one blank indented SVG line per export are trimmed for repository whitespace
hygiene; rendered content and all other source output are preserved.

| Entry / viewport | Results | Confirmation | Receipt |
|---|---|---|---|
| button 100×30 | [results SVG](artifacts/2026-09-27-agent-archive-recovery/results-button-100.svg) / [compositor text](artifacts/2026-09-27-agent-archive-recovery/results-button-100.txt) | [confirmation SVG](artifacts/2026-09-27-agent-archive-recovery/confirmation-button-100.svg) / [compositor text](artifacts/2026-09-27-agent-archive-recovery/confirmation-button-100.txt) | [receipt SVG](artifacts/2026-09-27-agent-archive-recovery/receipt-button-100.svg) / [compositor text](artifacts/2026-09-27-agent-archive-recovery/receipt-button-100.txt) |
| button 160×44 | [results SVG](artifacts/2026-09-27-agent-archive-recovery/results-button-160.svg) / [compositor text](artifacts/2026-09-27-agent-archive-recovery/results-button-160.txt) | [confirmation SVG](artifacts/2026-09-27-agent-archive-recovery/confirmation-button-160.svg) / [compositor text](artifacts/2026-09-27-agent-archive-recovery/confirmation-button-160.txt) | [receipt SVG](artifacts/2026-09-27-agent-archive-recovery/receipt-button-160.svg) / [compositor text](artifacts/2026-09-27-agent-archive-recovery/receipt-button-160.txt) |
| agent 100×30 | [results SVG](artifacts/2026-09-27-agent-archive-recovery/results-agent-100.svg) / [compositor text](artifacts/2026-09-27-agent-archive-recovery/results-agent-100.txt) | [confirmation SVG](artifacts/2026-09-27-agent-archive-recovery/confirmation-agent-100.svg) / [compositor text](artifacts/2026-09-27-agent-archive-recovery/confirmation-agent-100.txt) | [receipt SVG](artifacts/2026-09-27-agent-archive-recovery/receipt-agent-100.svg) / [compositor text](artifacts/2026-09-27-agent-archive-recovery/receipt-agent-100.txt) |
| agent 160×44 | [results SVG](artifacts/2026-09-27-agent-archive-recovery/results-agent-160.svg) / [compositor text](artifacts/2026-09-27-agent-archive-recovery/results-agent-160.txt) | [confirmation SVG](artifacts/2026-09-27-agent-archive-recovery/confirmation-agent-160.svg) / [compositor text](artifacts/2026-09-27-agent-archive-recovery/confirmation-agent-160.txt) | [receipt SVG](artifacts/2026-09-27-agent-archive-recovery/receipt-agent-160.svg) / [compositor text](artifacts/2026-09-27-agent-archive-recovery/receipt-agent-160.txt) |


## Prior live qualification gap

This historical server-unavailable checkpoint is superseded by the [native local-model continuation](2026-09-27-agent-archive-recovery-live.md). AC2 remains open for the unresolved live paths documented there.

The prior authorized GET checks of `http://127.0.0.1:9099/health` and
`http://127.0.0.1:9099/v1/models` both failed with `ConnectionRefusedError`; the
observation was recorded at **2026-09-27 18:56:55 UTC**. The configured local
provider/model were `llama_cpp`/`local-model`. The user was asked for a reachable
tool-capable model/server startup, but no answer is available. Task7 did not
repeat the unchanged probe, switch endpoints, inspect secrets or download models.

No live-model restoration or original-chat opening is claimed. Pilot/fake-model
SVGs and real SQLite assertions establish automated evidence only. Live AC2
remains unchecked and Task33020 remains In Progress.

When an authorized model becomes reachable, create a disposable configuration
with explicit `[paths] data_dir` and `TLDW_CONFIG_PATH`, verify actual storage
paths, then use two named workspaces and three distinct saved chats to exercise
chat-only, workspace-only and both-archived cases from a different Chat with an
unsent draft. Ask for recent archives, restore through a result button and a
conversational follow-up, explicitly Open, and repeat RAG-only with no index.
Record actual screenshots, original chat/leaf/message IDs, draft and workspace
effects and model invocation counts. Do not promote automated artifacts to that
live evidence.

## Earlier affected-correction byte identity (before the resource probe)

The first selection hashes above identify its earlier tested bytes. This table
records the subsequent keyboard/chronology/cleanup correction phase, not final
branch bytes. The later resource probe used UI test SHA-256
`4eb3fb3c9727e7f77a5d15416ee61259ab8e486a53382d56544dddf7460df2d2`
as recorded with its command above. The final review wave below changes that
file again; none of these phases implies a repeat of the original 22-file run.

| File | SHA-256 |
|---|---|
| `Tests/UI/test_console_archive_recovery_flow.py` | `45c423bc3025f08d39c86625aec8219eb46dc12e12c77f5385d7c6f8f6912a14` |
| `Tests/Chat/test_archive_recovery_operations.py` | `f728f79bb00d6d47eac43c0094d554c3a6223e1d38bfe5430b65e54be4480e06` |
| `Tests/Chat/test_console_archive_recovery_dispatch.py` | `8d09a6cb5d784816b3ddd9c48e7542913ba4dd98cf9ad41a7fb9e460fcbcb32c` |
| `tldw_chatbook/Widgets/Chat_Widgets/archive_recovery_cards.py` | `f7cf7b70f7d6b9fec93c338f3f3dd5ed9359e42da4deccd9b8916927efd5d3ee` |
| `tldw_chatbook/UI/Console_Modules/archive_recovery.py` | `728ca923b9dcc656793efb93d195a7ee9ebf5626411e62e1151cb71a1bed581d` |


## Final whole-branch review fix wave (I1–I3, M1–M3)

Fix base: `2792b4b0cd5ee6eb55c9c0dc2f9bfe826fd54e2f`. This is one authorized
review-fix wave, not a repeat of the original 22-file selection. ADR required:
no new ADR; direct repair of [ADR-166](../../../../backlog/decisions/166-agent-assisted-archive-recovery.md)
and approved spec §7. Task33020 stays In Progress and live AC2 stays unchecked.

- **I1:** `ArchiveTarget.title` is explicitly presentation-only (`compare=False`).
  Conversation ID/version/archive state/ownership and the entire workspace
  ID/state/revision/name/chronology snapshot still participate in equality;
  transaction CAS is unchanged. Search and byte fitting only change display text.
  After equality succeeds, preflight replaces the display snapshot with validated
  current metadata for confirmation. Real SQLite long/empty/padded/UTF-8 fitted
  rows restore; actual versioned title mutations still refuse. The native label
  is sanitized and capped at 80 characters with a visible ellipsis; the full
  validated title stays in the confirmation record, independent of display.
- **I2:** receipt Refresh catches only expected `ArchiveRecoveryError`, retaining
  the receipt and warning only if its source/render generation is still current.
  Cancellation and unexpected errors keep their existing semantics. Actual
  mounted unavailable-workspace, expired-budget and pending source-switch cases
  assert no app crash, late paint, writes or new model send.
- **I3:** the app-owned recovery runtime retains asynchronous completion delivery
  independently of disposable view hooks. A view acknowledges an actual current
  receipt projection; otherwise an admitted operation notifies through the app
  with sanitized status and separate chat/workspace changed outcomes. Close and
  shutdown join delivery before releasing runtime session state. Coordinator
  settlement remains one-shot; remount and repeated decisions do not notify again.
  Real-write barriers test background, actual detach and actual Console owner
  close; the latter reports a committed workspace with unchanged archived chat.
- **M1:** an outer `finally` covers executor installation and partial setup, with
  nested fixture/creator-thread handle cleanup and unconditional prior-executor
  restoration/shutdown. The two narrow injected faults pass. The existing five
  cycles still explicitly close exact handles on their creators before GC.
- **M2:** the table above is labeled as the earlier affected-correction phase;
  the later resource-test hash and this final wave are separate. The historical
  22-file 486/12 result does not cover these later bytes.
- **M3:** removed the extra migration EOF blank line. Local `git diff --check`
  passes. The controller owns the final whole-branch range check after this commit.
  No behavior test was run for the whitespace-only SQL change.

The tracked Task3/Task5 reports now identify themselves as historical snapshots
and link later QA/fix disposition. No plan or Backlog task file was edited by the
fix implementer, and no ledger/history or worktree was removed.

### Exact focused commands and intermediate results

Every command below ran in the worktree/interpreter identified above, with real
SQLite and the disposable test-profile bootstrap. RED production bytes were
reviewed HEAD `2792b4b0cd`; test bodies evolved through the explicitly listed
harness corrections. Raw logs remain at the exact `/tmp` paths in the commands;
counts, diagnostics, final tests and artifacts are preserved here durably.

```sh
/tmp/archive-recovery-venv/bin/python -m pytest -p no:cacheprovider --basetemp=/tmp/archive-final-fix-red Tests/Chat/test_archive_recovery_coordinator.py::test_display_title_fitting_preserves_exact_recovery_identity Tests/UI/test_console_archive_recovery_flow.py::test_receipt_refresh_refusal_keeps_native_source_safe Tests/UI/test_console_archive_recovery_flow.py::test_admitted_restore_delivers_app_completion_when_source_cannot_receive -q --tb=short > /tmp/archive-final-fix-red.log 2>&1
```

11 failed, 3 passed in 73.86 s. Three unchanged titles stale; three Refresh crashes; background/detached completion missing. Two byte cases initially selected an omitted row; closed case initially waited on UI idle through the barrier.

```sh
/tmp/archive-recovery-venv/bin/python -m pytest -p no:cacheprovider --basetemp=/tmp/archive-final-fix-red2 Tests/Chat/test_archive_recovery_coordinator.py::test_display_title_fitting_preserves_exact_recovery_identity -k byte_fitted Tests/UI/test_console_archive_recovery_flow.py::test_admitted_restore_delivers_app_completion_when_source_cannot_receive --override-ini='addopts=' -q --tb=short --show-capture=no > /tmp/archive-final-fix-red2.log 2>&1
```

1 failed, 1 passed, 9 deselected in 6.22 s. Corrected real emitted byte-fitted candidate is stale unchanged; actual versioned title mutation is refused.

```sh
/tmp/archive-recovery-venv/bin/python -m pytest -p no:cacheprovider --basetemp=/tmp/archive-final-fix-red-close 'Tests/UI/test_console_archive_recovery_flow.py::test_admitted_restore_delivers_app_completion_when_source_cannot_receive[closed]' -q --tb=short --show-capture=no > /tmp/archive-final-fix-red-close.log 2>&1
```

1 failed in 9.57 s. Corrected actual owner close commits workspace only and emits no notification.

```sh
/tmp/archive-recovery-venv/bin/python -m pytest -p no:cacheprovider --basetemp=/tmp/archive-final-fix-red-cleanup Tests/UI/test_console_archive_recovery_flow.py::test_resource_probe_restores_executor_after_partial_failure -q --tb=short --show-capture=no > /tmp/archive-final-fix-red-cleanup.log 2>&1
```

2 failed in 5.17 s. Both setup and cleanup faults leave the replacement executor installed; regression finally restores it to protect subsequent tests.

```sh
/tmp/archive-recovery-venv/bin/python -m pytest -p no:cacheprovider --basetemp=/tmp/archive-final-fix-red-title-ui Tests/UI/test_console_archive_recovery_flow.py::test_long_current_title_has_bounded_usable_native_confirmation -q --tb=short --show-capture=no > /tmp/archive-final-fix-red-title-ui.log 2>&1
```

1 failed in 29.15 s. Mounted compact long-title search never reaches confirmation.

```sh
/tmp/archive-recovery-venv/bin/python -m pytest -p no:cacheprovider --basetemp=/tmp/archive-final-fix-green1 Tests/Chat/test_archive_recovery_coordinator.py::test_display_title_fitting_preserves_exact_recovery_identity Tests/UI/test_console_archive_recovery_flow.py::test_receipt_refresh_refusal_keeps_native_source_safe Tests/UI/test_console_archive_recovery_flow.py::test_admitted_restore_delivers_app_completion_when_source_cannot_receive Tests/UI/test_console_archive_recovery_flow.py::test_resource_probe_restores_executor_after_partial_failure Tests/UI/test_console_archive_recovery_flow.py::test_long_current_title_has_bounded_usable_native_confirmation -q --tb=short --show-capture=no > /tmp/archive-final-fix-green1.log 2>&1
```

14 passed, 3 failed in 60.56 s. All three completion cases actually receive truthful app notifications; the test wrongly counted Textual Notify events as navigation.

```sh
/tmp/archive-recovery-venv/bin/python -m pytest -p no:cacheprovider --basetemp=/tmp/archive-final-fix-green2 Tests/UI/test_console_archive_recovery_flow.py::test_admitted_restore_delivers_app_completion_when_source_cannot_receive Tests/UI/test_console_archive_recovery_flow.py::test_long_current_title_has_bounded_usable_native_confirmation -q --tb=short --show-capture=no > /tmp/archive-final-fix-green2.log 2>&1
```

1 collection error in 0.34 s from a test-edit indentation slip; no behavioral result. Repaired before the next invocation.

```sh
/tmp/archive-recovery-venv/bin/python -m pytest -p no:cacheprovider --basetemp=/tmp/archive-final-fix-green3 Tests/UI/test_console_archive_recovery_flow.py::test_admitted_restore_delivers_app_completion_when_source_cannot_receive Tests/UI/test_console_archive_recovery_flow.py::test_long_current_title_has_bounded_usable_native_confirmation -q --tb=short --show-capture=no > /tmp/archive-final-fix-green3.log 2>&1
```

3 passed, 1 failed in 30.76 s. Completion including remount/replay passes; whole-card top capture did not simultaneously show Cancel. Focus/activation is checked by the real keyboard route next.

```sh
/tmp/archive-recovery-venv/bin/python -m pytest -p no:cacheprovider --basetemp=/tmp/archive-final-fix-affected Tests/Chat/test_archive_recovery_coordinator.py Tests/Chat/test_archive_recovery_operations.py Tests/Chat/test_console_archive_recovery_dispatch.py Tests/UI/test_console_archive_recovery_flow.py::test_pending_source_remount_expiry_and_detach_fail_closed Tests/UI/test_console_archive_recovery_flow.py::test_native_controls_show_true_stale_and_partial_storage_outcomes Tests/UI/test_console_archive_recovery_flow.py::test_restore_as_requires_fresh_final_name_confirmation Tests/UI/test_console_archive_recovery_flow.py::test_retained_native_recovery_resource_owners_plateau_and_close -q --tb=short --show-capture=no > /tmp/archive-final-fix-affected.log 2>&1
```

80 passed in 136.79 s. Affected coordinator, real storage, native authority/dispatch, receipt/rename and modified resource-probe regressions. No warnings/skips reported.

```sh
/tmp/archive-recovery-venv/bin/python -m pytest -p no:cacheprovider --basetemp=/tmp/archive-final-fix-title-final Tests/UI/test_console_archive_recovery_flow.py::test_long_current_title_has_bounded_usable_native_confirmation -q --tb=short --show-capture=no > /tmp/archive-final-fix-title-final.log 2>&1
```

1 passed in 12.15 s. Focused long-title controls activate; screenshot inspection then exposed the clipped ellipsis at a 160-character cap.

```sh
/tmp/archive-recovery-venv/bin/python -m pytest -p no:cacheprovider --basetemp=/tmp/archive-final-fix-title-visible-red Tests/UI/test_console_archive_recovery_flow.py::test_long_current_title_has_bounded_usable_native_confirmation -q --tb=short --show-capture=no > /tmp/archive-final-fix-title-visible-red.log 2>&1
```

1 failed in 15.51 s. New compositor assertion proves the actual truncation marker is clipped.

```sh
/tmp/archive-recovery-venv/bin/python -m pytest -p no:cacheprovider --basetemp=/tmp/archive-final-fix-final Tests/UI/test_console_archive_recovery_flow.py::test_long_current_title_has_bounded_usable_native_confirmation 'Tests/UI/test_console_archive_recovery_flow.py::test_admitted_restore_delivers_app_completion_when_source_cannot_receive[closed]' Tests/UI/test_console_archive_recovery_cards.py -q --tb=short --show-capture=no > /tmp/archive-final-fix-final.log 2>&1
```

7 passed in 32.48 s. Final 80-character display cap shows the ellipsis; Tab reaches Cancel and Shift+Tab/Enter restores. Actual close also asserts no late receipt paint. All five directly affected card cases pass.

### Final bytes and coverage provenance

The final production/test/SQL SHA-256 values are:

| File | SHA-256 |
|---|---|
| `tldw_chatbook/Chat/archive_recovery/contracts.py` | `301c7a67b76c77f15fd67af448fbcb0167d6b2ff533f2efb732eb6cf7135a32b` |
| `tldw_chatbook/Chat/archive_recovery/coordinator.py` | `5641afc868cdd83521fe9a7086be556e313064dc2f1b37bf79e39e1e7738b5c9` |
| `tldw_chatbook/Chat/archive_recovery/runtime.py` | `e1db22ac012fd64492737ea92eeeb4ddf433489ed38c5041edcbe36f6bfe16a6` |
| `tldw_chatbook/UI/Console_Modules/archive_recovery.py` | `1d868915018f4dbb728e7c27dce5ee61854267dd336c769a460438110bfb7ae7` |
| `tldw_chatbook/Widgets/Chat_Widgets/archive_recovery_cards.py` | `8b24fed8d2b1fa36a0770adcfa7064f257f7e91a3dc0704890faf92767e3dec7` |
| `Tests/Chat/test_archive_recovery_coordinator.py` | `fe00f3ea483da6966f5d36e66fed0cefd779023817677eaf5dc01322685c0b10` |
| `Tests/UI/test_console_archive_recovery_flow.py` | `16dc37f87233cafbb4668ba48bbe15ac8f9f181cf930228e09597ce172d16f5b` |
| `tldw_chatbook/DB/migrations/chachanotes_v73_to_v74_archive_chronology.sql` | `2f57280ed1f423ce846bae0b46e973c1359d1e50471526fb89c2b8174f0eb0f2` |

The 80-node affected run preceded the final display-cap reduction and the final
long-title/closed-source assertions in the UI test file. Its card-source SHA-256
was `a5062a7025e4989ab11b61f8a3e02198f10955a1b6680c1b17876a5b841d8645`
(the 160-character phase), not the final card hash above. It covers the unchanged
coordinator/operations/runtime and the selected existing UI/resource test bodies.
The final seven-node run covers the final card bytes, long-title keyboard flow
and added no-late-paint assertion. The earlier 14-pass run covers the unchanged
new Refresh and cleanup-fault bodies; the three notification cases passed after
correcting the navigation event type assertion. No single earlier run is claimed
to have loaded the final whole UI test file, and no broad selection was repeated.

Scoped static commands:

```sh
/tmp/archive-recovery-venv/bin/python -m ruff check --no-cache tldw_chatbook/Chat/archive_recovery/contracts.py tldw_chatbook/Chat/archive_recovery/coordinator.py tldw_chatbook/Chat/archive_recovery/runtime.py tldw_chatbook/UI/Console_Modules/archive_recovery.py tldw_chatbook/Widgets/Chat_Widgets/archive_recovery_cards.py Tests/Chat/test_archive_recovery_coordinator.py Tests/UI/test_console_archive_recovery_flow.py
/tmp/archive-recovery-venv/bin/python -m ruff format --check --no-cache tldw_chatbook/Chat/archive_recovery/contracts.py tldw_chatbook/Chat/archive_recovery/coordinator.py tldw_chatbook/Chat/archive_recovery/runtime.py tldw_chatbook/UI/Console_Modules/archive_recovery.py tldw_chatbook/Widgets/Chat_Widgets/archive_recovery_cards.py Tests/Chat/test_archive_recovery_coordinator.py Tests/UI/test_console_archive_recovery_flow.py
git diff --check
```

Final results: **All checks passed; 7 files already formatted; no whitespace
errors**. No changed huge legacy Python file needs a baseline lint waiver.
Intermediate static attempts: Ruff first could not write `.ruff_cache` under the
sandbox; the corrected `--no-cache` invocation found one import-order issue,
which was fixed. A subsequent test-edit indentation slip was caught by Ruff and
pytest collection (listed above), then corrected. The first staged whitespace
check also found trailing spaces in the new exported SVG/compositor files (they
were untracked during the earlier working-diff check); those were trimmed and
the staged check passed before amending the same fix commit. Git emitted existing
worktree gc/unreachable-object maintenance warnings; no repository cleanup was
attempted. These intermediate failures are not green results.

### Final-wave artifacts and bounded resource evidence

[Focused long-title confirmation SVG](artifacts/2026-09-27-agent-archive-recovery/confirmation-long-title-100.svg)
and [compositor text](artifacts/2026-09-27-agent-archive-recovery/confirmation-long-title-100.txt)
show the exact native focused confirmation at 100×30 with its ellipsis, Cancel,
Restore as and unrelated draft. The test uses actual Tab/Shift+Tab/Enter dispatch.
The SVG was parsed and rasterized for inspection; exported SVG/compositor line-end
whitespace was trimmed for packaging without changing rendered content; Cairo initially lacked its
shared-library search path, then worked with
`DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`. Font fallback glyphs in the local
raster are not attributed to product code. This is mounted synthetic-model
qualification, not live-model evidence.

[Final-wave resource samples](artifacts/2026-09-27-agent-archive-recovery/resource-owners-final-fix.json)
are copied byte-for-byte from the affected run's test output. All five settled
samples have process FD55, live handles3/3/3 and named file FDs7/7/7. Explicit
creator-thread close reaches handles0/0/0 and named FDs0/0/0, with34 total process
FDs remaining. This process had already run other affected nodes; those remaining
FDs are outside the named-store attribution. The old resource artifact/table
remains the earlier FD47/handles3/3/2 sample. Neither sample explains historical
+729 growth, tests natural GC/reference release, qualifies repeated new apps or
many workers, or closes the missing real-model walkthrough.

Unchanged readiness limits: original selection **486 passed, 12 failed, 7
warnings**, including **+729 FDs UNATTRIBUTED**; existing invalid-escape/link
warnings remain baseline diagnostics, not repaired here. Authorized localhost9099
was not probed again; no endpoint change, inference, secret read or model download
occurred. No push, PR, merge, full sweep or worktree cleanup was performed.
