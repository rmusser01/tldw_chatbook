---
id: TASK-38967
title: "Console readiness warm-up hands back a display proof about to expire, so the FULL sync it prepared refuses and replays 0.2 s later"
status: To Do
created_date: 2026-11-06 01:21
labels:
- follow-up
- console
- performance
- tests
priority: low
references:
- tldw_chatbook/UI/Console_Modules/console_spend_projection.py
- tldw_chatbook/UI/Console_Modules/config_sync.py
- tldw_chatbook/UI/Screens/chat_screen.py
- Tests/UI/test_console_pending_facts_cold_readiness.py
- Tests/UI/test_console_pending_facts_direct_control_cold_readiness.py
- Tests/UI/test_first_persistence_getter_attribution.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. No branch due to merge before this one changes the code below.

**Mechanism** (files under `tldw_chatbook/UI/Console_Modules/`):
- `ConsoleReadinessConfigProjection.warm()` (`console_spend_projection.py:496-511`) returns True whenever the proof is younger than `max_age`, which is 1.0 s (`:510`).
- `run()` re-reads only when the proof is at least `life` old (`:425-435`). `life` is 1.0 s, or 0.5 s while the run-poll timer exists (`REFRESH_AHEAD_SHARE`, `:200`). So with no run poll, a 0.99 s proof is handed back as warm.
- `_checked_display_status` then refuses the proof once it is `min(1.0, max_age)` old or more (`:865`). `_run_checked_display_sync` (`:1109-1123`) calls `defer`, which is `_request_console_control_bar_sync(delayed=True)` (`config_sync.py:236`) and waits 0.2 s (`config_sync.py:218`).
- In production:
  - a FULL sync awaits `prepare_console_presentation_sync` (`chat_screen.py:17853`), which calls `warm()` (`config_sync.py:141`);
  - then `project_console_tick` returns False at `chat_screen.py:17869` ("Replay owns it");
  - the whole sync replays 0.2 s later.

**Measured.** `warm` and `_checked_display_status` were wrapped in the three files below and each file was run 5 times, at load 9-11:

| What | Count |
|---|---|
| `warm()` calls, all returning True (mostly production FULL preparation) | 191 |
| … handing back a proof at least 0.5 s old | 72 |
| … handing back a proof at least 0.9 s old | 15 |
| … of those 15, followed straight away by an expiry refusal | 11 |
| Refusals in total | 140 |
| … of which were expiry refusals (the rest are the tests' deliberate cold reads) | 27 |

The 11 proofs were 0.928-0.999 s old when warmed and 1.006-1.311 s old when refused. TASK-38745 (Done) recorded the same thing independently and fixed only its own test, with a bounded retry; the product is unchanged. TASK-38696 (Done) is the refusal backoff, which is a different issue.

**Tests carrying the race.** Each warms, runs one `_sync_console_rail_and_controls()`, waits one `pilot.pause()`, and never checks that the sync rendered:
- `Tests/UI/test_console_pending_facts_cold_readiness.py:564-566`
- `Tests/UI/test_console_pending_facts_direct_control_cold_readiness.py:565-567`
- `Tests/UI/test_first_persistence_getter_attribution.py:93-95`

None of them is on a PR lane. Each passed 6 of 6 here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Measured first: over a stated workload, the number of FULL syncs whose warm-up returned True but whose render refused on proof age is counted before and after the change; if a refresh-ahead margin does not reduce it, the notes record the numbers and the product stays unchanged
- [ ] #2 If changed, warm-up re-reads a proof within a margin of max_age before returning True, so the render that follows it does not refuse on age; a deterministic test with the proof aged to max_age minus a small epsilon is red on dev
- [ ] #3 The margin adds no config read to an idle Console, with re-reads per idle minute measured before and after
- [ ] #4 The three tests assert that their sync rendered, or retry as TASK-38745's _prepared does, and pass 10 of 10 -n 4 runs with the load recorded
<!-- AC:END -->
