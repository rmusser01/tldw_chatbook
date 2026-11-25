---
id: TASK-41950
title: "A retained Change Review initialization job's opaque claim refuses every later workspace activity until restart"
status: Done
assignee: []
created_date: '2026-11-19 07:30'
updated_date: '2026-11-25 15:19'
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
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Owner decision 2026-11-25: a retained init job keeps its opaque claim until restart (TASK-40716); Release frees only the tool slot. ADR required: yes, a dated ADR-134 amendment next to TASK-41635's; ADR-285 cross-reference (its Task 8D2A section specifies the initializer).

1. Establish what happens first: does the retained job keep its claim, and is a later Console Send refused, and with what copy? Stop and report if not.
2. Test over the real consent service, issuer, `_InitializationSource`, capacity, activity and a real `ConsoleChatController.submit_draft`: a Send before the job is admitted (control); the job retains "native Git" and holds; a later Send is refused; Release returns the slot and the Send is still refused. Negative controls by temporary mutation.
3. ADR-134 amendment and ADR-285 cross-reference.
4. Second owner decision (2026-11-25, after step 1 showed every chat is refused): keep the claim, but a Send refused by a retained cleanup's claim names the restart (AC #3). At `OrdinaryTurn.admit`, tell a retained owner's conflicting claim apart (the check `_InitializationSource.admit` makes) and raise `CleanupUnprovenRefused`; the Console's Send maps it to restart copy. Ordinary busy conflicts keep the old copy. Red first; a control for the old copy; cover TASK-40716 retained tool operations if the same check does.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two owner decisions, 2026-11-25. First: a retained init job keeps its opaque claim until restart, and Release frees only the slot. Second, made after that showed every chat is refused: keep the claim, but say that a restart clears it. ADR-134 records both, ADR-285 cross-references them, and a Send refused by a retained cleanup's claim now names the restart.

**What happens (established first, on origin/dev 2959b48314).**
- An init job admitted outside an ordinary turn takes its own owner and an opaque activity claim (`_InitializationSource.admit`).
- When its initializer retains "native Git" and holds, as `run_native_git` does without a cleanup proof, the custody worker never reaches `_SnapshotTicket.returned`, so the claim is never settled.
- A later Console Send reaches `OrdinaryTurn.admit`. Its `prelaunch` conflicts with the opaque claim and raised `CapacityRefused("worktree_activity_busy")`, which `submit_draft` showed as "Workspace activity is busy; retry after it finishes." That copy never said a restart clears it.

Refinements to the task's premise:
- **Every chat is refused, not only the job's workspace.** There is one `WorktreeActivity` per Console runtime, and opaque conflicts with every footprint. After Change Review is enabled, a Send in the job's own workspace captures an opaque footprint itself: an uncommitted probe printed `opaque=True`, and `opaque=False` for the Send made before it was enabled. A Send in another workspace does not, so it is refused only because the init claim is opaque. A mutation with a non-opaque init claim let that Send through.
- **Nothing in the running process retires the claim.** `retain_unproven` makes the owner's drain unproven, and `_retire_if_settled` needs a proven drain. Even `UnprovenRetention.prove()` leaves it held; the test now asserts this.
- **Platform.** An ordinary turn takes a claim only with native process groups (macOS, Linux), so on Windows this admission refusal does not occur.

**Message change (AC #3).**
- `OrdinaryTurn.admit` (`Agents/ordinary_tree.py`): when `prelaunch` refuses and a retained record's owner is a member of a conflicting claim, it raises `CleanupUnprovenRefused(label, retention)` instead of the busy `CapacityRefused`. This is the same `capacity.retained()` + `holds_conflicting_claim` check that `_InitializationSource.admit` makes.
- `ConsoleChatController.submit_draft` picks its copy with the new `activity_refusal_copy(refused)` (`Chat/console_ordinary_tree.py`). That refusal gets "Workspace activity is held by a cleanup that could not be proven; restart Chatbook to clear it."; every other `CapacityRefused` keeps "Workspace activity is busy; retry after it finishes." The helper lives outside the controller so the controller's exact size-ratchet row (28,777) stays unchanged: the controller edit is net zero lines.
- I wrote "held by a cleanup that could not be proven" instead of the owner's example "stuck behind an unfinished cleanup". The cleanup may well have finished; what is missing is the proof of it. The new wording matches the app's other unproven-cleanup copy.
- **TASK-40716 is covered too.** The check keys on any retained record, not on init jobs. A retained tool operation on an agent turn's owner, or on a sub-agent that joined the turn's claim (`_attach_child` adds it as a member), keeps the turn's claim until restart in the same way, and the test covers that case.
- **Not covered.** An owner marked unproven without a retention record keeps the busy copy.
- **No logging changed.**

**Test.** `test_a_retained_cleanups_claim_refuses_later_console_sends_until_restart` (`Tests/Workspaces/test_native_change_review_initialization.py`), parametrized `initialization` / `tool_operation`. It runs on the real consent service, issuer, `_InitializationSource`, `_SnapshotTicket`, capacity, activity and `ConsoleChatController.submit_draft`.
- Controls first. A live opaque claim, with an unrelated retained "Raw CLI" record whose owner holds no claim, gets the old busy copy. Once that is cleared, a Send in the job's workspace and one in another workspace are both admitted.
- Then the hold: the init job runs under its own owner and retains "native Git", or an agent turn's owner holds an opaque claim and retains a tool operation.
- Sends in both workspaces get the restart copy.
- Release goes through the production seam `release_held_tool_slot(controller, owner.execution_id)`. Tool workers go from 1 to 0, the claim is still held (`holds_conflicting_claim`), and further Sends in both workspaces still get the restart copy.
- After `prove()`, the custody worker is joined, the owner's drain is `unproven`, and the claim is still held.
- Teardown joins every custody worker after `prove()`, not only `hold()`'s exit, so `_complete` and `ticket.returned` finish before the fixture tears down. The neighbouring TASK-41635 test `test_retained_cleanups_hold_later_roots_until_restart_and_say_so` had the same pattern and now joins its custody workers too.

**Red first and negative controls** (each mutation temporary and restored, all run against the final test):
- No fix (the three source files reverted): both cases red at the first refused Send (old busy copy instead of restart copy).
- N2, restart copy whenever any retained record exists (membership check replaced by `True`): both red at the ordinary-busy control.
- N3, the Console ignores the refusal type (`activity_refusal_copy` always returns the busy copy): both red at the first refused Send.
- A, the job's claim released when the pool thread sees the retention: `initialization` red at the first refused Send. `tool_operation` passes, which is expected: the mutation touches only init jobs.
- C, Release also frees the owner's claim: both red at the held-claim check after Release.

**Evidence (2026-11-25, base origin/dev 2959b48314; machine load average 12-57 from other sessions).**
- `Tests/Workspaces/test_native_change_review_initialization.py` whole: base 27 passed serially and at `-n 4`; this head 29 passed serially and at `-n 4`. In the first round, one head run at load about 57 had 6 serial and 4 `-n 4` `wait(3)` timeouts in older init tests, some ordered before the new test. Two alternating base/head serial rounds at the same load then all passed.
- `Tests/Workspaces/test_native_change_review_git_custody.py` (imports that file): 15 passed serially. One `-n 4` run at load 48 had 2 `native initializer did not settle` timeouts (a 5 s teardown wait for the real-Git init queue). Two alternating base/head `-n 4` rounds then passed 15 each.
- Files asserting the old busy text, each whole, serially and at `-n 4`:
  - `test_console_ordinary_tree_widening.py`: 20 passed both ways.
  - `test_console_reference_custody.py`: 44 passed both ways.
  - `test_console_default_send_capture.py`: `default` and `full_tools` passed; `writable` failed both ways. **That red predates this branch:** base dev failed it 2 of 2, and head 2 of 2, in alternating rounds, always at the same point, where the first writable chat's Send never reaches the provider. The busy assertions it carries (`full_tools`) pass.
- Light-tier base set: `test_console_approval_scope_journeys`, `test_approval_timeout_policy` and `test_smoke`, 52 passed. The two selected `test_chat_functions.py` nodes: 128 passed.
- Retention-related files at `-n 4`: `test_console_unproven_cleanup_release`, `test_console_held_tool_slots`, `test_console_pre_run_cleanup_stop`, `test_unproven_tool_cleanup_run` and `test_console_ordinary_tree`. Result: 111 passed, 1 failed. The failure is `test_unproven_tool_cleanup_run.py::test_host_return_is_visible_only_with_its_decided_durability` (`'OrdinaryTurn' object has no attribute 'runtime'`, from `host_returned`'s `self.runtime._wake_first_send()`). **It predates this branch:** it fails identically on base dev, and this branch does not touch `host_returned`.
- `Tests/Architecture` serially: base 1507 passed, 5 skipped; this head 1507 passed, 5 skipped. The controller's exact size-ratchet row (28,777) is unchanged.
- `./scripts/preflight.sh` under a scratch HOME: all derived-artifact checks passed. `ruff check tldw_chatbook --select F` and `Tests/lint/ruff_gate.py` pass. The new lines are `ruff format` clean; the touched files have older unformatted lines.

Files: `tldw_chatbook/Agents/ordinary_tree.py`, `tldw_chatbook/Chat/console_ordinary_tree.py`, `tldw_chatbook/Chat/console_chat_controller.py`, `Tests/Workspaces/test_native_change_review_initialization.py`, `backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md`, `backlog/decisions/285-native-phased-agent-workflows.md`, `Docs/User_Guide/console/agent-runs-and-tools.md` (held-slots paragraph: the refusal and its copy).
<!-- SECTION:NOTES:END -->
