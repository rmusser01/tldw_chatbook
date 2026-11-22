---
id: TASK-42426
title: "Console runtime dispose refuses sqlite_retirement_incomplete:AgentRuns in two stock skill-trust cases and fails the recovered-suites lane"
status: To Do
assignee: []
created_date: 2026-11-22 22:10
updated_date: 2026-11-22 22:44
labels:
- follow-up
- ci
- console
- agents
- flaky
priority: high
references:
- Tests/Performance/test_console_skill_trust_stock_owned.py
- tldw_chatbook/Chat/console_runtime.py
- tldw_chatbook/DB/base_db.py
- tldw_chatbook/DB/AgentRuns_DB.py
- tldw_chatbook/Chat/console_activity_receipts.py
- backlog/decisions/225-app-scoped-sqlite-retirement.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 21 of the follow-up burn-down (2026-11-22), in the CI runs the round's filing read. Checked on the round-21 stack 77f124a740 (dev ddd3c02052 plus seven ports); paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** In a scan of the 140 Derived Artifacts PR runs created from 2026-11-20 00:00Z to about 21:50Z on 2026-11-22 (every attempt; times are job start times), the required "PR lane (recovered suites)" failed in 7 jobs, from 2026-11-21 02:22Z to 2026-11-22 19:58Z, on two cases of `Tests/Performance/test_console_skill_trust_stock_owned.py::test_stock_console_skill_setup_has_original_owned_lifetime`:
- `[runtime_dispose]`: runs 44953045328, 44953086559 attempt 2, 44974782181, 45055944482 attempt 2;
- `[runtime_replacement]`: runs 44819090982 attempt 2, 44974742439 and 45048643145 attempt 2.

TASK-40324 (Done) fixed a different case of the same test, `[app_closing]`, which hung to its 240 s bound.

**The failure.** Each case's body runs in a child process (`<stock-skill-control>`). In the six failure logs read (all but 44819090982), the child fails in `held_control` while awaiting `runtime.dispose()` with a 10 s bound (`asyncio.wait_for(runtime.dispose(), timeout=10)` at test line 549 for `runtime_replacement`; the shielded control task at line 553 for `runtime_dispose`), and the error is `tldw_chatbook.DB.base_db.SqliteRetirementIncomplete: sqlite_retirement_incomplete:AgentRuns`, raised by `ConsoleRuntime._dispose_owned` (`Chat/console_runtime.py:6910-6911` at this base) when `retire_agent_runs_db` (`:6841`) could not retire the AgentRuns connection. So on a loaded runner the dispose finds the AgentRuns database still held by someone else. A user closing the app or replacing the Console runtime at that moment would get the same refusal.

TASK-34661 (In Progress) owns retiring application-owned SQLite handles after producer shutdown and records the same `sqlite_retirement_incomplete:AgentRuns` refusal at app shutdown (2026-10-08 notes: Fleet Wake recovery still in native admission). It may be the same holder; this task is the CI-red symptom in these two cases. ADR-225 (Accepted) keeps foreign-thread ownership and uncertain settlement, and busy handles "must not be counted as closed"; TASK-34661 rules that a final runtime refusal is correct for a foreign run-worker holder (no foreign close, larger grace or suppression). So the fix is in who owns and settles the holder, never in forcing or hiding the refusal. Coordinate with TASK-34661 before changing the retirement path.

Both cases pass locally at the base.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The holder of the AgentRuns connection when these disposes refuse is identified and recorded in the notes
- [ ] #2 If the holder found in AC#1 has an owner that can settle it, it is settled before the dispose, so retirement succeeds, shown with a test that fails before the change (a temporary delay in the holder, reverted). If the refusal is correct for that holder (genuinely foreign or unsettled, as ADR-225 requires), the test settles the holder itself or asserts the refusal; the notes say which
- [ ] #3 Both cases pass, as AC#2 leaves them, in repeated runs of the file under parallel load (at least 20 runs, or a soak recorded in the notes)
<!-- AC:END -->
