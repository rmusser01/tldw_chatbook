---
id: TASK-42426
title: "Console runtime dispose refuses sqlite_retirement_incomplete:AgentRuns in two stock skill-trust cases and fails the recovered-suites lane"
status: Done
assignee:
- '@claude'
created_date: 2026-11-22 22:10
updated_date: 2026-11-23 03:55
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
- [x] #1 The holder of the AgentRuns connection when these disposes refuse is identified and recorded in the notes
- [x] #2 If the holder found in AC#1 has an owner that can settle it, it is settled before the dispose, so retirement succeeds, shown with a test that fails before the change (a temporary delay in the holder, reverted). If the refusal is correct for that holder (genuinely foreign or unsettled, as ADR-225 requires), the test settles the holder itself or asserts the refusal; the notes say which
- [x] #3 Both cases pass, as AC#2 leaves them, in repeated runs of the file under parallel load (at least 20 runs, or a soak recorded in the notes)
<!-- AC:END -->

## Implementation Plan

1. Read the CI failure logs (read-only) for the seven failed jobs: which call refuses, and what the child printed.
2. Add a temporary diagnostic (reverted before commit) at the AgentRuns idle close that, on refusal, prints which guard refused (managed use, acquisition, counted core operation, open transaction) and the holding thread's owner and current stack. Reproduce under parallel load, and if needed with a temporary forced delay in the suspected holder.
3. Record the holder in the notes, then follow AC#2's branch: if the holder has an owner that can settle it before dispose, make that owner settle it (test red with a temporary delay in the holder, green after); if the refusal is correct for that holder under ADR-225, make the test settle the holder itself or assert the refusal. No foreign close, larger grace or suppression.
4. Soak both cases (at least 20 runs of the file's two cases under parallel load), then run the recovered-suites step once with its CI flags.
5. Preflight, ruff F, size ratchets for touched modules; notes, ACs, Done.

## Implementation Notes

**Holder (AC#1).** The holder is the Console view's sub-agent badge count. The view (ChatScreen) runs it as a Textual worker in group `console-subagent-counts`: `ConsoleAgentController._load_subagent_counts` → `run_owned_db_call` → `AgentRunsDB.count_subagents_by_conversation`, on an executor thread. The rail and inspector builds schedule it (`_with_console_conversation_browser_state` → `_console_subagent_counts_for_rows`) whenever its 2 s refresh window has passed. While it runs, it holds a counted core operation on `agent_runs.db` and, while it opens its handle through the private-sqlite helper process, a registry acquisition. `close_if_idle` refuses while either is held, so `retire_agent_runs_db` returns False and `_dispose_owned` raises `sqlite_retirement_incomplete:AgentRuns`.
- CI: the four failure logs read again (44974782181, 44953045328, 44974742439 and 45048643145) all raise from `_dispose_owned` at the test's `runtime.dispose()` awaits (script lines 539 and 535). The product prints no holder.
- Local (reverted diagnostics: a refusal snapshot in `retire_agent_runs_db` of the registry guards, counted storage operations, thread stacks and task awaits; a timeline of every counted AgentRuns operation; and the scheduling stack of every `console-sync`/`console-subagent-counts` worker). At base, 72 of 72 runs passed at load 36-48 on 18 CPUs. In 11 of 36 `runtime_dispose` runs and 6 of 36 `runtime_replacement` runs, a count read ran inside the dispose window and ended 68-867 ms before the retire. Locally the reads take 60-120 ms; on a slower runner they are still in flight at the retire.
- Forced delay (temporary sleep in `count_subagents_by_conversation`, reverted): every refusal snapshot showed one counted operation on `db.agent_runs` on an executor thread. That thread's stack was `run_owned_db_call.invoke` → `count_subagents_by_conversation`, and its awaiting task was the `_load_subagent_counts` worker.

**AC#2: the refusal is correct for this holder, so the test settles the holder itself.**
- The read belongs to the view, not the runtime. ADR-225 counts a busy handle as not closed, and its extension makes late badge reads finite, operation-owned reads. TASK-34661 rules a final runtime refusal correct for a foreign worker holder. So the runtime cannot settle the read, and no foreign close, grace or suppression was added.
- The app's only dispose route (`_shutdown_console_runtime` → `settle_view_and_dispose`) drains the view's Console workers, this group included, before `dispose_console_runtime`. These two cases instead call `runtime.dispose()` directly with the view live.
- The script now waits, without cancelling, for the view's in-flight AgentRuns reads (`console-subagent-counts` and `console-agent-history`) through the existing `capture_console_view_workers` selection:
  - `runtime_dispose` waits while its dispose is parked on the held setup read, before `release`;
  - `runtime_replacement` waits just before its dispose.
- Waiting rather than cancelling matters. A cancelled count keeps its old refresh time, so the next render would start a new read at once.
- All existing assertions are unchanged.

**Red/green.** Both runs used a temporary 3 s delay in every count read after the first, with 3 parallel loops × 2 rounds at load 41-44.
- Without the test change: 11 of 12 cases failed, each with `SqliteRetirementIncomplete: sqlite_retirement_incomplete:AgentRuns`. Every refusal snapshot named the `_load_subagent_counts` worker as the holder.
- With the change: 12 of 12 passed. The delayed read ended 6-81 ms before the retire in every run, so the settle waited for it.
- A 1.5 s delay first failed 2 of 2. It then passed 6 of 6 unfixed, because the delay moved the next read away from the dispose. With the fix, the 1.5 s delay passed 42 of 42 (3 loops × 7 rounds).

**Soak (AC#3).** This ran on the final code, with diagnostics reverted: 4 parallel loops × 6 rounds of both cases (`-n 2`), at load 39-47 on 18 CPUs. Result: 48 of 48 passed, 24 per case. Base also passes locally, so the delayed red/green above is the evidence for the CI race.

**Checks.**
- `ruff check --select F` on the test file passes, and `scripts/preflight.sh` passes.
- No size ratchet lists the test file.
- The recovered-suites step (CI flags `-p no:xdist --storage-lease-guard --timeout=180`): 2004 passed, 2 failed, 1 skipped in 24:41 (load 34-46). Every one of this file's 14 lane cases passed. Neither failure is in this file, and both are outside this diff:
  - `Tests/Library/test_library_ingest_runner.py::test_real_pool_worker_exit_is_reported_for_owning_generation` fails at base with the same flags.
  - `Tests/Performance/test_console_managed_received_acceptance.py::test_received_managed_demand_uses_one_catalog_and_real_cold_initializer` failed with `managed_received_prerequisite_timeout` after a `phase=trace_reservation` ValueError. That is TASK-42412's open flake. It is a separate child script that never imports this file. On this branch it failed 2 more times in 3 serial runs (load about 29) and passed 8 of 8 run 4 at a time; at base it passed 8 of 8.
  - Also outside the lane: this file's `send_enter` and `send_button` cases fail at base with the same command (`'types.SimpleNamespace' object has no attribute 'session'`, then `Cold Send did not reach original configuration SQL`). This change does not touch their path.

**For TASK-34661 (not verified).** On the app's real quit path, `settle_view_and_dispose` cancels the count worker. The cancelled worker keeps its old refresh time, so a render during the dispose could start a fresh count read that races the retire. Quit would then log `SQLite retirement incomplete at quit: AgentRuns`. This change does not touch that product path, and it was not reproduced here.

**Files.** `Tests/Performance/test_console_skill_trust_stock_owned.py`; a lesson in `backlog/docs/lessons-backlog-hygiene.md`: in zsh, `"$F[case]"` subscripts the variable, so a soak's node ids came out empty and pytest collected the whole tree.
