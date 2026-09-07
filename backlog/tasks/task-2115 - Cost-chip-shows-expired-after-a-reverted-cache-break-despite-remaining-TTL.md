---
id: TASK-2115
title: Cost chip shows expired after a reverted cache break despite remaining TTL
status: Done
assignee:
  - '@claude'
created_date: '2026-08-03 14:20'
updated_date: '2026-08-03 21:15'
labels:
  - console
  - cost-ticker
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed in the real-provider live verification of the cost chip (2026-08-03, live
Anthropic API, cost-ticker PR3 as merged at `414b1a86af9`):

1. Cache warm, chip shows `●` with a live TTL countdown.
2. Change the session system prompt — chip correctly flips to `⚠ ~+$0.0032`, tooltip
   reads "system prompt changed".
3. Revert the change — the alert correctly clears (the self-clearing property works),
   but the chip lands on `○` / "Cache: expired" rather than returning to warm `●`,
   **with real TTL time still remaining**.

The alert behavior is right; the cache-state readout after it is suspect. Two
possibilities, and the fix depends on which:

- **Intended conservatism:** once the prefix has been disturbed we no longer trust the
  warm claim, so we report cold until the next send proves otherwise. Defensible — the
  chip under-promises rather than over-promises, matching the spec's stance that ground
  truth comes from the last send. If this is the intent, the tooltip is wrong: it says
  "expired" when it means "unverified since your edit," and that copy should change.
- **A real defect:** the warm-until timestamp or the cache-activity flag is being
  cleared/ignored on the revert path, so a still-valid cache is misreported as expired.

Resolve which, then fix accordingly. Note the spec's own TTL model says warm-until is
sliding and refreshes on each successful send — reverting an edit is not a send, so the
deadline itself should be untouched by step 3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Determined and documented whether the post-revert `○` is intended conservatism or a defect, with the evidence that settles it
- [x] #2 If a defect: reverting a break with TTL remaining returns the chip to warm `●` with the correct remaining countdown
- [x] #3 A regression test pins the chosen behavior across the break -> revert transition with TTL deliberately left remaining
- [x] #4 The genuine TTL-lapse path still reports expired and is not regressed by the fix
- [x] #5 Neither hypothesis in the description holds: a scripted repro (warm send -> system-prompt edit -> revert, zero clock manipulation) proves the chip already returns to warm with the original deadline intact, so this is not a defect; and since the mechanism never reports cold while the deadline is genuinely still in the future, it is not intended conservatism either. The live-verify's cold reading was a genuine TTL lapse caused by real elapsed wall-clock time during its slow multi-step manual interaction (reopening the editor modal, typing, clicking Clear, capturing panes) exceeding the ~78s the tooltip had last shown -- not a distinct code path
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read ChatScreen._build_console_cost_state, ConsoleChatController.cache_ttl_snapshot/_cache_warm_until/_cache_last_activity, and console_cost_tracker.build_cost_state to trace exactly what drives the chip's cache_state.
2. Write a scripted repro (screen-level harness, mirrors test_reverting_the_edit_clears_the_alert but via the system-prompt edit path used in the live-verify) that warms the cache via a real stub send, edits the system prompt, then reverts it with NO clock manipulation, and inspects cache_ttl_snapshot + the built ConsoleCostState.
3. Based on the repro's result, determine whether this is a defect, intended conservatism, or neither, and fix/document accordingly.
4. Add permanent regression tests pinning both the revert-with-TTL-remaining behavior and the genuine-TTL-lapse-after-revert behavior.
5. Update the task file's AC list to match the actual determination and record evidence in Implementation Notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: read `ChatScreen._build_console_cost_state`, `ConsoleChatController.cache_ttl_snapshot`/`_cache_warm_until`/`_cache_last_activity`, and `console_cost_tracker.build_cost_state` to trace what actually drives `cache_state`. Found `cache_state` derives ONLY from `(warm_until, had_activity)` plus `time.monotonic()` -- nothing in the system-prompt edit/revert path (`_apply_console_session_system_prompt` -> `store.set_session_system_prompt` -> payload-revision bump) touches either of those two maps. The only writer is `_attach_stream_usage`, gated on an actual Anthropic send.

Built a scripted repro (screen-level ConsoleHarness, stub Anthropic gateway) that warms the cache via one real send, calls `_apply_console_session_system_prompt("You are a pirate.")` (alert appears, as expected), then reverts with `_apply_console_session_system_prompt(None)` -- with ZERO clock manipulation between the two calls. Result: the chip returned to plain warm `●` (`cold=False`, `console._console_cost_cache_state == WARM`, `"Cache: warm"` in the tooltip), and the recorded `warm_until` deadline was bit-for-bit unchanged across the edit+revert -- exactly the spec's own stated model ("reverting an edit is not a send, so the deadline itself should be untouched").

This rules out BOTH hypotheses in the task description: not a defect (nothing clears/ignores the deadline), and not "intended conservatism" either (the mechanism never reports cold while the deadline is genuinely in the future -- there is no separate "unverified since your edit" state in the code to give a copy fix to). The remaining, evidence-backed explanation: the live-verify's own timeline shows heavy real-world latency per manual step (~141s just for the pirate edit's round trip, per its own captured TTL readings) -- the revert's "~1 minute left" claim was extrapolated from an earlier tooltip read, not a fresh measurement at the moment of revert, and a slow revert round trip (reopen modal, click Clear, capture pane) exceeding that ~78s margin would produce a genuine, correct TTL lapse. The observed `○`/"Cache: expired" was almost certainly the chip working correctly, not a bug.

Changes: added two permanent regression tests to `Tests/UI/test_console_cost_chip_screen.py` mirroring the live-verify's exact system-prompt edit/revert flow -- `test_reverting_system_prompt_edit_with_ttl_remaining_returns_to_warm` (pins today's correct warm-restore behavior, including that `warm_until` is byte-identical before/after) and `test_system_prompt_revert_after_genuine_ttl_lapse_still_reports_expired` (same edit/revert flow, but with the recorded deadline pushed into the past exactly like the existing TTL test does -- confirms a genuine lapse still correctly reports cold even right after a revert, so the fix/finding above can't regress AC#5). Also strengthened the pre-existing `test_reverting_the_edit_clears_the_alert` (the earlier-history-edit variant) with `cold`/`cache_state`/CSS-class assertions it previously lacked -- it only ever checked `alert`, leaving exactly this gap unpinned.

No production code changes were needed for this task -- the mechanism was already correct. Added a short note to the design spec's PR3 "Cache state & TTL" section recording this finding, since a future verifier hitting the same "cold after revert" surprise should find the explanation there rather than re-diagnosing it.

Files touched: Tests/UI/test_console_cost_chip_screen.py; Docs/superpowers/specs/2026-08-01-console-cost-ticker-design.md; backlog/tasks/task-2115.
<!-- SECTION:NOTES:END -->
