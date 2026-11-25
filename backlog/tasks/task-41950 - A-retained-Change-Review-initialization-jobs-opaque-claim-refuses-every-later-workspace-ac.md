---
id: TASK-41950
title: "A retained Change Review initialization job's opaque claim refuses every later workspace activity until restart"
status: Done
assignee: []
created_date: '2026-11-19 07:30'
updated_date: '2026-11-25 14:03'
labels:
- follow-up
- workspaces
- needs-decision
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
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Owner decision 2026-11-25: a retained init job keeps its opaque claim until restart (TASK-40716); Release frees only the tool slot. ADR required: yes, a dated ADR-134 amendment next to TASK-41635's; ADR-285 cross-reference (its Task 8D2A section specifies the initializer).

1. Establish what happens first: does the retained job keep its claim, and is a later Console Send refused, and with what copy? Stop and report if not.
2. Test over the real consent service, issuer, `_InitializationSource`, capacity, activity and a real `ConsoleChatController.submit_draft`: a Send before the job is admitted (control); the job retains "native Git" and holds; a later Send is refused; Release returns the slot and the Send is still refused. Negative controls by temporary mutation.
3. ADR-134 amendment and ADR-285 cross-reference.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Owner decision 2026-11-25: a retained init job keeps its opaque claim until restart; Release frees only the slot. No product code changed; ADR-134 records the decision and a new test pins it.

**What happens (established first, on origin/dev 2959b48314).** An init job admitted outside an ordinary turn takes its own owner and an opaque activity claim (`_InitializationSource.admit`). When its initializer retains "native Git" and holds, as `run_native_git` does without a cleanup proof, the custody worker never reaches `_SnapshotTicket.returned`, so the claim is never settled. A later Console Send reaches `OrdinaryTurn.admit`; its `prelaunch` conflicts with the opaque claim and raises `CapacityRefused("worktree_activity_busy")`, which `submit_draft` shows as **"Workspace activity is busy; retry after it finishes."** That copy does not say a restart clears it. The task's premise holds, with two refinements:
- The refusal covers every chat, not only the job's workspace: one `WorktreeActivity` per Console runtime, and opaque conflicts with every footprint. After Change Review is enabled, a Send in the job's own workspace captures an opaque footprint itself (a probe, not committed, printed `opaque=True`; the pre-enable Send printed `opaque=False`). A Send in another workspace does not (mutation B below admits it), so it is refused only because the init claim is opaque.
- Nothing in the running process retires the claim. `retain_unproven` makes the owner's drain unproven and `_retire_if_settled` needs a proven drain. A first test draft called `UnprovenRetention.prove()` and expected a Send to be admitted afterwards: it was still refused, with no executions left.
- An ordinary turn takes a claim only with native process groups (macOS, Linux), so on Windows this admission refusal does not occur.

**Decision recorded.** ADR-134 "Retained Change Review initialization claim amendment (TASK-41950, 2026-11-25)", after the TASK-41635 section. ADR-285's "Consent initializer ownership (Task 8D2A)" section, which specifies the initializer's opaque activity, gets a one-paragraph cross-reference.

**Test.** `test_a_retained_initialization_claim_refuses_later_console_sends_until_restart` (`Tests/Workspaces/test_native_change_review_initialization.py`). Real consent service, issuer, `_InitializationSource`, `_SnapshotTicket`, capacity, activity, a real `ConsoleChatController` and `submit_draft`. A Send before the job is admitted (control). The job runs under its own owner; its initializer retains "native Git" and holds. Then: a Send in the job's workspace and a Send in another workspace are both refused with the busy copy; Release (`abandon`) drops tool workers 1 to 0 and a further Send is still refused; no provider call after the control.

**Negative controls** (temporary mutations, each restored, run against the final test):
- A, claim released when the pool thread sees the retention (`_worker_loop` drops the claim on refusal): red at the job-workspace Send, `(True, 'reply') == (False, busy)`.
- B, a non-opaque init footprint: red at the other-workspace Send.
- C, Release also frees the claim (`abandon` drops the owner's claim): red at the Send after Release.

**Evidence (2026-11-25, base origin/dev 2959b48314, machine load average 35-57 from other sessions).**
- `Tests/Workspaces/test_native_change_review_initialization.py` whole: base 27 passed serially and at `-n 4`; head 28 passed serially and at `-n 4`.
- A first head run under load average about 57 had 6 serial and 4 `-n 4` failures, all `wait(3)` timeouts in older init tests, some ordered before the new test. Two alternating base/head serial rounds at the same load then all passed (base 27, head 28, twice), as did the final runs above. TASK-41635 recorded the same init waits failing on dev under load.
- Light-tier selection (`scripts/ci/tier_policy.select_changes`): that file, `Tests/Workspaces/test_native_change_review_git_custody.py` (imports it; 15 passed serially and at `-n 4`), and the base set: `test_console_approval_scope_journeys` 15, `test_approval_timeout_policy` 21, `test_smoke` 16, all passed.
- `ruff check tldw_chatbook --select F` and `Tests/lint/ruff_gate.py` pass; the new lines are `ruff format` clean (the file has older unformatted lines). `./scripts/preflight.sh` run at this head.

Not changed (scope): the busy copy does not tell the user that only a restart clears it.

Files: `Tests/Workspaces/test_native_change_review_initialization.py`, `backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md`, `backlog/decisions/285-native-phased-agent-workflows.md`.
<!-- SECTION:NOTES:END -->
