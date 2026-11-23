---
id: TASK-42424
title: "Closing a background Console tab with a pending decision also answers the viewed tab's decision in the session-close journeys test"
status: Done
assignee:
  - '@claude'
created_date: 2026-11-22 22:10
updated_date: 2026-11-23 02:27
labels:
- follow-up
- ci
- console
- tests
- flaky
priority: high
references:
- Tests/UI/test_console_session_tab_close.py
- tldw_chatbook/Chat/console_runtime.py
- tldw_chatbook/Chat/console_chat_controller.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 21 of the follow-up burn-down (2026-11-22), in the CI runs the round's filing read. Checked on the round-21 stack 77f124a740 (dev ddd3c02052 plus seven ports); paths are under `tldw_chatbook/` unless they start with `Tests/` or `scripts/`.

**Why.** The required "UI Fast Lane (26)" fails on `Tests/UI/test_console_session_tab_close.py::test_session_close_pending_journeys` with `closing the background tab answered the viewed tab`. In a scan of the 140 Derived Artifacts PR runs created from 2026-11-20 00:00Z to about 21:50Z on 2026-11-22 (every attempt; times are job start times), it failed 25 jobs, from 2026-11-20 09:06Z to 2026-11-22 20:01Z, every one with this message (all 25 logs read), for example runs 44948612804, 44952102973, 44953045328, 44974735998, 44974790480, 44978286545, and 45016281720 attempts 2 and 3. It also fails locally at the base: 2 of 3 runs at load 6-8 (one in a `-n 4` run, one of two serial runs), with the same message at line 1129.

**The failure.** The scenario (`test_console_session_tab_close.py:1074-1131`) arms a pending decision in the viewed chat (`sibling`, session `keeper`) and another in a background chat (`doomed`), then closes the background tab. The close must answer only the doomed session's decision. At line 1129, `assert not sibling.done()` fails: the viewed chat's decision was answered too. If that is not a test artefact, closing one chat resolves another chat's approval or question, which a user would see as an answer they never gave.

TASK-35587 (In Progress since 2026-10-19) is about the same test failing at app shutdown with a `CancelledError`, a different signature. TASK-42342 (round 21) changed how a session close drains the controller's preparation reads. This stack carries that change and the test still fails here, but re-check on the dev that merges it before deciding the cause.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Why the viewed chat's pending decision is answered when the background tab closes is recorded: a cross-session product defect, or a test artefact, with the evidence
- [x] #2 A defect is fixed so that closing a chat answers only that chat's pending decisions, with a test shown red first; an artefact is fixed in the test
- [x] #3 The test passes in repeated runs of its file under parallel load (at least 20 runs, or a soak recorded in the notes)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce `test_session_close_pending_journeys` on base 2f7790873a (serial and `-n 4`, several runs) and record the rate and load.
2. Read the CI failure logs (read-only) and trace the background-close path: the confirmed close, the pending-decision cancel sweep, and the TASK-42342 voice/preparation drain. Find what can resolve the viewed session's question round.
3. Make the race deterministic with a temporary forced delay, then revert it.
4. Fix it where it lives: product code if closing one chat can answer another chat's decision, the test if it is an artefact. Keep the journey's assertions.
5. Add a deterministic regression test (red at base, green after) in a file a PR lane already runs.
6. Check sibling close paths (approval, chat-create, worktree merge, skill install/script) for the same mechanism.
7. Soak: the journey 20 times at `-n 4` with its file, and 5 times serially with `-p no:xdist --timeout=180`; record counts and load.
8. Preflight, ruff F, size ratchets for touched modules; notes, ACs, Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Verdict: a test artefact, not a cross-session defect.** The close path answers only the closing session. `_cancel_pending_decisions_for_session`, `set_answerable_decision`, `_discard_approval_rows_for_closing_session` and `cancel_all_subagents` are all keyed by the closing session or its conversation, and nothing in the close sweeps the viewed session's question. The viewed tab's question is instead failed closed by its own arm-time liveness check, TASK-41637's no-running-app guard in `InterruptRoundHost.run_round`: `app_gone = not app.is_running and _approval_view_is_detached()`.

**Mechanism.**
- Console harnesses point `controller.app` at a never-run `TldwCli`, so `is_running` is False on every arm. The trace printed `running=False` for every question arm it logged. Liveness therefore rests on `has_answerable_view()` alone, and that is False whenever a modal is on top (`view.app.screen is view`).
- The question's worker thread can reach the check after the Close dialog has opened. The round is then treated as having no running app and settles `cancelled` after one 1 s poll.
- CI shows the same result: run 44978286545 has `sibling` finished with `{'answered': False, 'reason': 'cancelled'}`.
- In the `chat_create` pass, the test never waited for the viewed tab's question to arm. Locally the gap from that check to the Close click was 0.7-1.8 s, and 0.27-0.48 s in the `question` pass.
- This began with TASK-41637's review (merged into dev 2026-11-19 20:48Z). The review dropped the `app.call_from_thread.__self__ is app` condition, which had kept these harness rounds live: the test routes that marshal to the host. The first of the 25 CI failures is 2026-11-20 09:06Z.
- In the shipped app, `controller.app` is the running `TldwCli`, so `is_running` is True and a modal can never fail a round closed.

**Evidence.**
- Base 2f7790873a without a delay: 7/7 passed (3 serial and 4 file runs at `-n 4`, load 28-42), so it does not reproduce naturally here.
- Temporary forced delay: hold the viewed session's question check (bounded at 5 s) until the view is covered. At base the journey failed 2/2 with `closing the background tab answered the viewed tab`, the trace showed `app_gone=True running=False detached=True`, and the result was `{'answered': False, 'reason': 'cancelled'}`. With the fix it passed 2/2 under the same delay, with every check `app_gone=False`. All instrumentation was reverted.

**Fix (test).** Before opening the Close dialog, `_verify_background_pending_close_releases_round_without_an_active_turn` now waits for the viewed tab's question card (`#chat-question-card`, displayed, payload `session_id == keeper`) and asserts the question is still live. The card mounts only after the liveness check passes, so the check can no longer run under the dialog. Every existing assertion is unchanged.

**Pin for the shipped app.** `Tests/Chat/test_console_runtime_lifetime.py::test_running_app_round_stays_pending_while_no_view_can_answer` covers a running App whose own screen covers the Console view: the round stays pending until it is answered. Under a temporary mutation that makes `app_gone` ignore `is_running`, it failed (`a running App's round settled unanswered`, 1 failed / 4 passed). After reverting, it passes.

**Siblings.**
- The first pending scenario arms the doomed tab's rounds. A late check there settles them with the result the close gives anyway, so no assertion flips. No other journey in the file keeps a round live across a modal.
- No other test file combines the re-pointed `call_from_thread` harness with live rounds and a confirmation modal.
- Other pending kinds: the close sweep is session-scoped for approval, question, skill install, skill script and chat create. Worktree merge is never swept and listens to its own run's cancel signal.

**Soak (fixed tree).**
- The whole file 20 times at `-n 4`: 20/20 runs, 80/80 tests passed, load 27-58, 56-137 s per run.
- The journey alone 5 times serially with `-p no:xdist --timeout=180`: 5/5 passed, load 35-59, 63-131 s.

**Checks.**
- `Tests/Chat/test_console_runtime_lifetime.py` at `-n 4`: 42 passed.
- `Tests/Architecture/test_module_size_ratchet.py` serially: 209 passed. No product module was touched.
- `ruff --select F` on both test files: clean.
- `preflight.sh`: all checks passed.

No user-visible behaviour changed, so no User Guide edit. There are no new census entries.

**Files.** `Tests/UI/test_console_session_tab_close.py`, `Tests/Chat/test_console_runtime_lifetime.py`, and `backlog/docs/lessons-testing-evidence.md` (a new entry: a green mutation run cannot clear a guard condition that only matters in a race).
<!-- SECTION:NOTES:END -->
