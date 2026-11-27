---
id: TASK-41950
title: "A retained Change Review initialization job's opaque claim refuses every later workspace activity until restart"
status: Done
assignee: []
created_date: '2026-11-19 07:30'
updated_date: '2026-11-25 19:36'
labels:
- follow-up
- workspaces
priority: medium
references:
- tldw_chatbook/Workspaces/change_review_consent.py
- tldw_chatbook/Workspaces/worktree_activity.py
dependencies:
- TASK-41635
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** When a Change Review init job runs under its own owner and its git cleanup is retained (unproven), it keeps an opaque activity claim, and an opaque claim conflicts with every other. That likely refuses later Console sends ("Workspace activity is busy") as well as later inits until restart. Releasing the tool slot frees only the slot. TASK-40716 decided claims stay held until restart; this asks whether that should hold for an init job's claim.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner's decision is recorded (ADR-134 or ADR-285)
- [x] #2 A test pins the chosen behaviour for a later Send while an init job's cleanup is retained
- [x] #3 A Send refused by a retained initialization claim tells the user that a restart clears it
- [x] #4 A tool the model calls whose widening is refused by a claim held until restart tells the model that a restart clears it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Owner decision 2026-11-25: a retained init job keeps its opaque claim until restart (TASK-40716); Release frees only the tool slot. ADR required: yes, a dated ADR-134 amendment next to TASK-41635's; ADR-285 cross-reference (its Task 8D2A section specifies the initializer).

1. Establish what happens first: does the retained job keep its claim, and is a later Console Send refused, and with what copy? Stop and report if not.
2. Test over the real consent service, issuer, `_InitializationSource`, capacity, activity and a real `ConsoleChatController.submit_draft`: a Send before the job is admitted (control); the job retains "native Git" and holds; a later Send is refused; Release returns the slot and the Send is still refused. Negative controls by temporary mutation.
3. ADR-134 amendment and ADR-285 cross-reference.
4. Second owner decision (2026-11-25, after step 1 showed every chat is refused): keep the claim, but a Send refused by a retained cleanup's claim names the restart (AC #3). At `OrdinaryTurn.admit`, tell a retained owner's conflicting claim apart (the check `_InitializationSource.admit` makes) and raise `CleanupUnprovenRefused`; the Console's Send maps it to restart copy. Ordinary busy conflicts keep the old copy. Red first; a control for the old copy; cover TASK-40716 retained tool operations if the same check does.
5. Review round 2: key the restart copy on the claim never retiring (a member whose drain or cleanup is unproven) through one `WorktreeActivity` predicate used by admit, widen and the initializer; give the model-facing widen refusal a restart variant (AC #4); pin the footprint filter and a sub-agent member.
6. Review round 3: drop the "exactly" claim and file the uncovered non-durable paths (TASK-42605); pin the `_InitializationSource.admit` switch with a marked-unproven case; document both `CleanupUnprovenRefused` meanings; drop the retained-record lookup from the predicate; give the model text a next step; name the resume tool; delete `holds_conflicting_claim`.
7. Review round 4: replace the handler comments with a structural guard (end the run only on a refusal that carries its record), pin it per handler with a negative control, name both raisers of the record-carrying meaning, add the legacy-hook path to TASK-42605, and say in the User Guide that no held slot may be listed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two owner decisions, 2026-11-25. First: a retained init job keeps its opaque claim until restart, and Release frees only the slot. Second, made after that showed every chat is refused: keep the claim, but say that a restart clears it. ADR-134 records both, and ADR-285 cross-references them. A Send, or a tool's widening, refused by a claim held until restart now names the restart.

**What happens (established first, on origin/dev 2959b48314).**
- An init job admitted outside an ordinary turn takes its own owner and an opaque activity claim (`_InitializationSource.admit`).
- When its initializer retains "native Git" and holds, as `run_native_git` does without a cleanup proof, the custody worker never reaches `_SnapshotTicket.returned`, so the claim is never settled.
- A later Console Send reaches `OrdinaryTurn.admit`. Its `prelaunch` conflicts with the opaque claim and raised `CapacityRefused("worktree_activity_busy")`, which `submit_draft` showed as "Workspace activity is busy; retry after it finishes." That copy never said a restart clears it.

Refinements to the task's premise:
- **Every chat is refused, not only the job's workspace.** There is one `WorktreeActivity` per Console runtime, and opaque conflicts with every footprint. After Change Review is enabled, a Send in the job's own workspace captures an opaque footprint itself: an uncommitted probe printed `opaque=True`, and `opaque=False` for the Send made before it was enabled. A Send in another workspace does not, so it is refused only because the init claim is opaque. A mutation with a non-opaque init claim let that Send through.
- **Nothing in the running process retires the claim.** `retain_unproven` makes the owner's drain unproven, and `_retire_if_settled` needs a proven drain. Even `UnprovenRetention.prove()` leaves it held; the test now asserts this.
- **Platform.** An ordinary turn takes a claim only with native process groups (macOS, Linux), so on Windows this admission refusal does not occur.

**Message change (AC #3, AC #4).**
- **One predicate.** `WorktreeActivity.restart_refusal(footprint, *, excluding=None)`. It returns a `CleanupUnprovenRefused("workspace activity")` (no retained record) when a claim that conflicts with `footprint` has a member whose owner's cleanup is unproven (new public `ExecutionOwner.cleanup_unproven`), else `None`.
  - Such a member's drain can never be proven, so the claim never retires. A member drained unproven always has the flag set (its drain outcome is `not _cleanup_unproven`, set once at prune), so the flag alone decides.
  - The owners' flags are read after the activity lock is released, so no lock nesting is added.
  - Round 3 dropped the `capacity` parameter and the retained-record lookup. No consumer reads `.retention` or `.label` from these refusals, and the record found usually belonged to another chat, where `explain_pre_run_stop` would have offered Release for it.
- **Not covered.** Known paths that hold a claim until restart without an unproven owner cleanup include (round 4 dropped round 3's "two paths"):
  - a Change Review snapshot whose write failed outside a workflow wave (`_SnapshotTicket.returned` settles `durable=False`, `Workspaces/change_review_ownership.py:300-303`; the producer is the UNRESOLVED handoff in `Workspaces/change_review_finalization.py:1136-1141`);
  - an ordinary turn whose run-row check failed, or whose started row captured no terminal (`OrdinaryTurn.host_returned`, `Agents/ordinary_tree.py:893-929`); that registry member is never settled at all;
  - a legacy hook event outside an ordinary tree whose required write failed (`Agents/legacy_hook_ownership.py:238-255`, `:321`/`:332` on dev b534feabd3).

  Those claims still get the busy copy, because the registry cannot tell "not yet durable" from "never". They are filed as TASK-42605, with the queued-snapshot gap (`_SnapshotSource._admit_owned`'s `_snapshot_prelaunch` refusal, `Workspaces/change_review_ownership.py:216-218`).
- **`CleanupUnprovenRefused` has two meanings.**
  - With a retained record, raised by `reserve_tool` and `run_on_custody_thread`: the owner's own cleanup is unproven and the run or turn ends.
  - Without one, raised by the predicate: another owner's claim is held and the run goes on.

  Its docstring names both raisers. Round 4 replaced round 3's comment convention ("handlers must only wrap `reserve_tool`") with a structural guard. It was wrong because `run_on_custody_thread` raises the record-carrying meaning too, and `explain_pre_run_stop` ends a turn on it.
  - The three run-ending handlers match `isinstance(exc, CleanupUnprovenRefused) and exc.retention is not None`: `_call_with_timeout`'s reservation (`agent_service.py`) and `invoke_tool`'s definitive and no-timeout paths (`agent_request_execution.py`).
  - Every record-carrying raiser sets `retention`: `reserve_tool` from `op.retention`, and `run_on_custody_thread` from the operation's retention. Their behaviour is unchanged.
  - A record-less refusal falls through to each handler's existing ordinary-refusal branch: "tool call refused: …" (no termination outside a goal run), `TOOL_ADMISSION_REFUSED`, the same as any other `CapacityRefused` today.
  - `agent_service.py` stays at exactly 7,698 lines, its ratchet row. `explain_pre_run_stop` already requires a retention.
- **Send.** `OrdinaryTurn.admit` raises the refusal, else the busy `CapacityRefused`. `submit_draft` picks its copy with `activity_refusal_copy` (`Chat/console_ordinary_tree.py`, typed, with a docstring): the restart copy for this refusal, the busy copy for every other `CapacityRefused`. The controller edit is net zero lines, so its exact size-ratchet row (28,777) is unchanged.
- **Widen.** `OrdinaryTurn.widen` raises the refusal for the opaque footprint, excluding the turn's own claim, else the busy refusal; its docstring now names both. `widen_refusal` gains "<tool> is unavailable: workspace activity is held by a cleanup that could not be proven; only restarting Chatbook clears it. Continue without it."
  - The spawn admission in `agent_run_tools` routes `CleanupUnprovenRefused` to `widen_refusal`; before, it would have read "runtime sub-agent limit reached".
  - A `send_to_agent` resume now names `send_to_agent`, not `spawn_subagent`. This holds for the busy text too: `_launch_fleet_child` passes `SEND_TO_AGENT_TOOL_NAME` to `_reserve_child_execution` when `resumed_managed` is set.
  - The skill and code-review paths already pass any `CapacityRefused` to `widen_refusal`.
- **Initialization.** `_InitializationSource.admit` uses the same predicate, which replaces its own loop over retained records.
- **Coverage.** The predicate covers:
  - every retained record's owner: an init job, an agent turn's own tool (TASK-40716), a sub-agent that joined the turn's claim;
  - an owner marked unproven with no record (`mark_cleanup_unproven`, for example `agent_worktree_git`, or a workflow child with an unknown launch outcome).

  The copy stays the restart copy after `prove()` removes a record, because the claim is still held.
- **Wording.** I wrote "held by a cleanup that could not be proven" instead of the owner's example "stuck behind an unfinished cleanup". The cleanup may well have finished; what is missing is the proof of it. The new wording matches the app's other unproven-cleanup copy.
- **No logging changed.**
- `holds_conflicting_claim` is deleted: only tests used it. The init test file has a local `_member_of_claim` helper instead.

**Tests.**
- **Main test.** `test_a_retained_cleanups_claim_refuses_later_console_sends_until_restart` (`Tests/Workspaces/test_native_change_review_initialization.py`), parametrized `initialization`, `tool_operation`, `child_member` and `marked_unproven`. It runs on the real consent service, issuer, `_InitializationSource`, `_SnapshotTicket`, capacity, activity and `ConsoleChatController.submit_draft`; the Console rig is the new `_console_rig` helper. Steps:
  - Controls first. A live opaque claim, with an unrelated retained "Raw CLI" record whose owner holds no claim, gets the old busy copy. Once that is cleared, a Send in the job's workspace and one in another workspace are both admitted.
  - Then the hold, one per case:
    - the init job runs under its own owner and retains "native Git";
    - an agent turn's opaque claim retains a tool operation;
    - a sub-agent joined that claim (`attach_child`) and retains one;
    - the turn's owner is marked unproven with no record.
  - Sends in both workspaces get the restart copy.
  - With a record: Release goes through the production `release_held_tool_slot(controller, owner.execution_id)`. Tool workers go from 1 to 0, `holds_conflicting_claim` is still true, and Sends in both workspaces still get the restart copy. Then `prove()` runs, the custody worker is joined, and no record remains.
  - In every case: the owner finishes, its drain is `unproven`, the claim is still held, and a Send still gets the restart copy.
  - Teardown joins every custody worker, not only `hold()`'s exit. The neighbouring TASK-41635 test `test_retained_cleanups_hold_later_roots_until_restart_and_say_so` now does the same.
- **Footprint control.** `test_a_retained_claim_that_a_send_does_not_need_keeps_the_busy_copy`. A retained cleanup holds a bounded claim on one folder, and a live claim covers the folder that holds each turn's scratch space (a light Send's whole footprint). The Send gets the busy copy, then is admitted once the live claim is gone.
- **Run-ending guard.** `test_only_a_refusal_carrying_its_record_ends_the_run` (`Tests/Agents/test_execution_capacity.py`), parametrized over the timeout-wrapped, definitive-after-start and no-timeout dispatch paths × record or no record. A refusal with its record ends the run (`CLEANUP_UNPROVEN`); a record-less one is an ordinary refusal and nothing runs. It reuses `_unwrapped_dispatch` from `test_unproven_tool_cleanup_run.py`, which is left unedited because it carries a known dev red the light tier would inherit.
- **Initializer with no record.** `test_a_claim_whose_owner_is_marked_unproven_holds_roots_until_restart`. An opaque claim's owner is marked unproven with no retained record. The refused root reads `RESTART_REQUIRED` with `_RESTART_REASON`, and retry returns 0. This pins the `_InitializationSource.admit` switch, which round 2 left unpinned.
- **Resume naming.** `test_a_refused_child_launch_names_the_tool_the_model_called` (`Tests/Chat/test_console_ordinary_tree_widening.py`), parametrized resumed × held. It calls `AgentRunTools._launch_fleet_child` over a fake service whose tree refuses the child. A resume names `send_to_agent`, a spawn names `spawn_subagent`, and each gets the busy or the restart text.
- **Widen.** `test_spawn_refused_by_a_retained_cleanups_claim_names_the_restart` (`Tests/Chat/test_console_ordinary_tree_widening.py`), parametrized `retained` True/False. Another owner holds a bounded claim on a different folder, so the chat's own Send is admitted. The model's spawn then widens and is refused. With the owner's tool retained, the model sees the restart text and not the busy text; with a live owner, it sees the old busy text. The existing `test_spawn_conflicting_with_another_running_turn_is_refused_not_awaited` also still sees it.

**Red first and negative controls.** Each mutation was temporary and restored; files were compared with saved copies. All were run against the round-3 code and tests:
- R, admit never raises the restart refusal: all four main cases red at the first refused Send.
- N2, admit always raises it: all four red at the ordinary-busy control, and the footprint control red.
- FP, admit's predicate is given an opaque footprint instead of the Send's: only the footprint control red.
- P, only owners whose drain already finished unproven count: all four main cases red at the first held Send, as are the marked-unproven init test and the retained widen case.
- N3, the Console ignores the refusal type: all four red at the first refused Send.
- W1, widen never raises the restart refusal: the retained widen case red.
- W2, spawn admission does not route `CleanupUnprovenRefused`: the retained widen case red, and both held resume-naming cases red.
- W3, `widen_refusal` has no restart variant: same as W2.
- I1, `_InitializationSource.admit` back to the round-1 loop over retained records (the review's finding 2): only `test_a_claim_whose_owner_is_marked_unproven_holds_roots_until_restart` red.
- NAME, a resume names `spawn_subagent` again: both resumed cases red.
- A, the init job's claim released on retention: `initialization` red at the first refused Send. The mutation also breaks the TASK-41635 retry control.
- C, Release also frees the owner's claim: `initialization`, `tool_operation` and `child_member` red at the held-claim check after Release.
- Round 4, the run-ending guard, with each handler's `and exc.retention is not None` removed in turn: `agent_service.py` turns only `[False-timeout_wrapped]` red; `agent_request_execution.py`'s definitive-path handler turns only `[False-definitive_after_start]` red; its no-timeout handler turns only `[False-no_timeout]` red.

**Round-4 evidence (2026-11-25, load average 8-10). Every file whole, at the round-4 head:**

| File | Result |
|---|---|
| `test_execution_capacity.py` (new guard test) | 30 passed, serial and `-n 4` |
| `test_native_change_review_initialization.py` | 33 passed, serial and `-n 4` |
| `test_console_ordinary_tree_widening.py` | 26 passed, serial and `-n 4` |
| `test_native_change_review_git_custody.py` | 15 passed, serial and `-n 4` |
| `test_console_unproven_cleanup_release.py` (`-n 4`) | 16 passed |
| `test_console_pre_run_cleanup_stop.py` (`-n 4`) | 4 passed |
| `test_worktree_activity.py` (`-n 4`) | 51 passed |
| `test_fleet_runtime_admission.py` (`-n 4`) | 6 passed |
| `test_fleet_continuation.py` (`-n 4`) | 49 passed |
| `test_fleet_runtime.py` (`-n 4`) | 146 passed |
| base set | 52 passed |
| two `test_chat_functions` nodes | 128 passed |
| `test_unproven_tool_cleanup_run.py` | 10 passed + the known dev red |

- `Tests/Architecture` serially in full, after the last code edit: 1507 passed, 5 skipped.
- `agent_service.py` is 7,698 lines, its exact ratchet row.
- `./scripts/preflight.sh` under a scratch HOME passed. Ruff F, `ruff_gate` and `check_markup_interpolation` passed.

**Round-3 evidence (2026-11-25, base origin/dev 2959b48314; machine load average 18-90 from other sessions). Every file whole, serially and at `-n 4`, at the round-3 head:**

| File | Serial | `-n 4` |
|---|---|---|
| `test_native_change_review_initialization.py` (dev: 27) | 33 passed | 33 passed |
| `test_console_ordinary_tree_widening.py`, the only file asserting the widen text | 26 passed | 26 passed |
| `test_worktree_activity.py` | 51 passed | 51 passed |
| `test_execution_capacity.py` | 24 passed | 24 passed |
| `test_console_unproven_cleanup_release.py` | 16 passed | 16 passed |
| `test_console_held_tool_slots.py` | 6 passed | 6 passed |
| `test_console_pre_run_cleanup_stop.py` | 4 passed | 4 passed |
| `test_fleet_runtime_admission.py` | 6 passed | 6 passed |
| `test_fleet_continuation.py` | 49 passed | 49 passed |
| `test_fleet_runtime.py` | 146 passed | 146 passed |
| `test_console_ordinary_tree.py` | 75 passed | 75 passed |
| `test_console_reference_custody.py` | not run | 44 passed |
| `Tests/Agents/test_unproven_tool_cleanup_run.py` | 10 passed, 1 failed | 10 passed, 1 failed |
| `test_native_change_review_git_custody.py` | see below | see below |

- The failure in `test_unproven_tool_cleanup_run.py` is `test_host_return_is_visible_only_with_its_decided_durability`, a known red on dev. It fails identically there: `'OrdinaryTurn' object has no attribute 'runtime'`, from `host_returned`, which this branch does not touch.
- `test_native_change_review_git_custody.py` is load-flaky on dev too. At load 84-90 head failed 2 serially and 8 plus 1 error at `-n 4`, all `native initializer did not settle` waits and the snapshot tests' equivalents. Alternating base/head rounds under the same load:
  - round 1 at load 25-60: base failed 7 at `-n 4` and 5 serially; head failed 9 and 1;
  - round 2 at load 16-24: both passed 15 at `-n 4` and serially.

  The flake reproduces on dev at the same load, so the branch adds no red. Rounds 1 and 2 at load 30-48 passed 15 on both sides.
- `test_console_default_send_capture.py` (round 1; untouched since): `default` and `full_tools` pass. `writable` is a known red on dev: it failed 2 of 2 on base and on head in alternating rounds.
- Light-tier selection (`select_changes`): the files above plus the base set (`test_console_approval_scope_journeys`, `test_approval_timeout_policy`, `test_smoke`: 52 passed) and two `test_chat_functions.py` nodes (128 passed).
- `Tests/Architecture` serially: dev 1507 passed, 5 skipped.
  - The first round-3 run had 1506 passed and 1 failed, `test_module_size_ratchet.py[agent_service.py]`: my comment line put that module one line over its exact row.
  - I folded the comment into the existing line. A rerun of `test_module_size_ratchet.py`, `test_persistent_diagnostic_inventory.py`, `test_agents_chat_boundary.py` and `test_code_identity_ratchet.py` then passed: 315 passed, 1 skipped.
  - The controller's exact row (28,777) is unchanged.
- `./scripts/preflight.sh` under a scratch HOME: all derived-artifact checks passed, including no duplicate task ids across 7,130 task files with TASK-42605 present.
- `ruff check tldw_chatbook --select F`, `Tests/lint/ruff_gate.py` and `scripts/check_markup_interpolation.py` pass. The new lines are `ruff format` clean; the touched files have older unformatted lines.
- TASK-42605's id was swept first: the highest task id cited in any ref's blobs (100,829 distinct), in any ref's file names, or in any worktree's backlog was 42604, apart from known non-id matches above it.

Files: `tldw_chatbook/Workspaces/worktree_activity.py`, `tldw_chatbook/Agents/execution_capacity.py`, `tldw_chatbook/Agents/ordinary_tree.py`, `tldw_chatbook/Agents/agent_run_tools.py`, `tldw_chatbook/Agents/agent_service.py`, `tldw_chatbook/Agents/agent_request_execution.py` (comments), `tldw_chatbook/Workspaces/change_review_ownership.py`, `tldw_chatbook/Chat/console_ordinary_tree.py`, `tldw_chatbook/Chat/console_chat_controller.py`, `Tests/Workspaces/test_native_change_review_initialization.py`, `Tests/Chat/test_console_ordinary_tree_widening.py`, `backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md`, `backlog/decisions/285-native-phased-agent-workflows.md`, `Docs/User_Guide/console/agent-runs-and-tools.md` (held-slots paragraph: the refusal and its copy), `backlog/tasks/task-42605 - …` (follow-up).
<!-- SECTION:NOTES:END -->
