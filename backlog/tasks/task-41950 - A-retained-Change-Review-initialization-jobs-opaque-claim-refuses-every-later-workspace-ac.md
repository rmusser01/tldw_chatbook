---
id: TASK-41950
title: "A retained Change Review initialization job's opaque claim refuses every later workspace activity until restart"
status: Done
assignee: []
created_date: '2026-11-19 07:30'
updated_date: '2026-11-25 17:01'
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
- **One predicate.** `WorktreeActivity.restart_refusal(footprint, capacity, *, excluding=None)`. When a claim that conflicts with `footprint` has a member whose drain was recorded unproven, or whose owner's cleanup is unproven (new public `ExecutionOwner.cleanup_unproven`), it returns a `CleanupUnprovenRefused`; otherwise `None`. That is exactly the condition under which `_retire_if_settled` never retires the claim.
  - The refusal carries a retained record of such a member when one exists, else `retention=None` with the label "workspace activity". No caller of these refusals dereferences `retention`: the Console copy, `widen_refusal` and the consent service's `_fail` check only the type. `explain_pre_run_stop` already skips a `None` retention.
  - Owners' flags are read after the activity lock is released, so no lock nesting is added.
- **Send.** `OrdinaryTurn.admit` raises the refusal, else the busy `CapacityRefused`. `submit_draft` picks its copy with `activity_refusal_copy` (`Chat/console_ordinary_tree.py`, typed, with a docstring): the restart copy for this refusal, the busy copy for every other `CapacityRefused`. The controller edit is net zero lines, so its exact size-ratchet row (28,777) is unchanged.
- **Widen.** `OrdinaryTurn.widen` raises the refusal for the opaque footprint, excluding the turn's own claim, else the busy refusal. `widen_refusal` gains "<tool> is unavailable: workspace activity is held by a cleanup that could not be proven; restart Chatbook to clear it." The spawn admission in `agent_run_tools` routes `CleanupUnprovenRefused` to `widen_refusal`; before, it would have read "runtime sub-agent limit reached". The skill and code-review paths already pass any `CapacityRefused` to it.
- **Initialization.** `_InitializationSource.admit` uses the same predicate, which replaces its own loop over retained records.
- **Coverage.** The predicate covers every retained record's owner (an init job, an agent turn's own tool from TASK-40716, a sub-agent that joined the turn's claim). It also covers an owner marked unproven with no record (`mark_cleanup_unproven`, for example `agent_worktree_git`, a workflow child with an unknown launch outcome). The copy stays the restart copy after `prove()` removes a record, because the claim is still held.
- **Wording.** I wrote "held by a cleanup that could not be proven" instead of the owner's example "stuck behind an unfinished cleanup". The cleanup may well have finished; what is missing is the proof of it. The new wording matches the app's other unproven-cleanup copy.
- **No logging changed.**
- `holds_conflicting_claim` stays: no production code calls it any more, but the test's held-claim assertions do.

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
- **Widen.** `test_spawn_refused_by_a_retained_cleanups_claim_names_the_restart` (`Tests/Chat/test_console_ordinary_tree_widening.py`), parametrized `retained` True/False. Another owner holds a bounded claim on a different folder, so the chat's own Send is admitted. The model's spawn then widens and is refused. With the owner's tool retained, the model sees the restart text and not the busy text; with a live owner, it sees the old busy text. The existing `test_spawn_conflicting_with_another_running_turn_is_refused_not_awaited` also still sees it.

**Red first and negative controls.** Each mutation was temporary and restored; files were compared to saved copies. All were run against the final tests:
- R, admit never raises the restart refusal: all four main cases red at the first refused Send.
- N2, admit always raises it: all four red at the ordinary-busy control, and the footprint control red.
- FP, admit's predicate is given an opaque footprint instead of the Send's: only the footprint control red. This is the footprint filter the review found unpinned.
- P, the predicate counts only retained records (the round-1 rule):
  - `marked_unproven` red at the first refused Send;
  - the other three red at the Send after `prove()`, where the copy flipped back to "retry".
- N3, the Console ignores the refusal type: all four red at the first refused Send.
- W1, widen never raises the restart refusal: the `retained` widen case red.
- W2, spawn admission does not route `CleanupUnprovenRefused` to `widen_refusal`: red.
- W3, `widen_refusal` has no restart variant: red.
- A, the init job's claim released on retention: `initialization` red at the first refused Send. The others pass; the mutation touches only init jobs.
- C, Release also frees the owner's claim:
  - `initialization` and `tool_operation` red at the held-claim check after Release;
  - `child_member` passes, because this throwaway mutation finds the claim through a closure variable named `claim_id`, which a child's drain callback names `member_id`.

**Evidence (2026-11-25, base origin/dev 2959b48314; machine load average 16-80 from other sessions). Every file whole, serially and at `-n 4`, at this round's head:**

| File | Serial | `-n 4` |
|---|---|---|
| `test_native_change_review_initialization.py` (dev: 27) | 32 passed | 32 passed |
| `test_console_ordinary_tree_widening.py`, the only file asserting the widen text | 22 passed | 22 passed |
| `test_console_ordinary_tree.py` | 75 passed | 75 passed |
| `test_console_unproven_cleanup_release.py` | 16 passed | 16 passed |
| `test_console_reference_custody.py` | 44 passed | 44 passed |
| `test_console_pre_run_cleanup_stop.py` | 4 passed | 4 passed |
| `test_console_held_tool_slots.py` | not run | 6 passed |
| `Tests/Agents/test_unproven_tool_cleanup_run.py` | 10 passed, 1 failed | 10 passed, 1 failed |
| `test_native_change_review_git_custody.py` | 15 passed | 3 timeouts at load 80; then 15 passed, twice |

Notes on the reds:
- The failure in `test_unproven_tool_cleanup_run.py` is `test_host_return_is_visible_only_with_its_decided_durability`, a known red on dev. It fails identically there: `'OrdinaryTurn' object has no attribute 'runtime'`, from `host_returned`, which this branch does not touch.
- The `git_custody` `-n 4` timeouts were `native initializer did not settle`, a 5 s teardown wait for the real-Git init queue. Two alternating base/head `-n 4` rounds then passed 15 each. Round 1 saw the same flake at load 48 and the same A/B result.

Other checks:
- `test_console_default_send_capture.py` (round 1; untouched since): `default` and `full_tools` pass. `writable` is a known red on dev: it failed 2 of 2 on base and on head in alternating rounds, because the first writable Send never reaches the provider.
- Light-tier selection (`select_changes`): the files above plus the base set (`test_console_approval_scope_journeys`, `test_approval_timeout_policy`, `test_smoke`: 52 passed) and two `test_chat_functions.py` nodes (128 passed).
- `Tests/Architecture` serially: dev 1507 passed, 5 skipped; this head 1507 passed, 5 skipped. The controller's exact size-ratchet row (28,777) is unchanged.
- `./scripts/preflight.sh` under a scratch HOME: all derived-artifact checks passed.
- `ruff check tldw_chatbook --select F`, `Tests/lint/ruff_gate.py` and `scripts/check_markup_interpolation.py` pass. The new lines are `ruff format` clean; the touched files have older unformatted lines.

Files: `tldw_chatbook/Workspaces/worktree_activity.py`, `tldw_chatbook/Agents/execution_capacity.py`, `tldw_chatbook/Agents/ordinary_tree.py`, `tldw_chatbook/Agents/agent_run_tools.py`, `tldw_chatbook/Workspaces/change_review_ownership.py`, `tldw_chatbook/Chat/console_ordinary_tree.py`, `tldw_chatbook/Chat/console_chat_controller.py`, `Tests/Workspaces/test_native_change_review_initialization.py`, `Tests/Chat/test_console_ordinary_tree_widening.py`, `backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md`, `backlog/decisions/285-native-phased-agent-workflows.md`, `Docs/User_Guide/console/agent-runs-and-tools.md` (held-slots paragraph: the refusal and its copy).
<!-- SECTION:NOTES:END -->
