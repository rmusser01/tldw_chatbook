---
id: TASK-42136
title: "Four unbounded entered.wait() calls in test_briefing_cast.py turn an early failure into a 300 s timeout"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 20:06
updated_date: 2026-11-21 18:40
labels:
- follow-up
- tests
- subscriptions
priority: low
references:
- Tests/Subscriptions/test_briefing_cast.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the docs stream (TASK-42027, round 18). Four tests in `Tests/Subscriptions/test_briefing_cast.py` start a generation whose `_slow_chat` sets `entered` and then waits on `release`, and then `await entered.wait()` with no bound (lines 865, 929, 1072 and 1472). If the generation fails before it calls the chat function, `entered` is never set, and the test waits for pytest-timeout's 300 s (`timeout = 300` in `pyproject.toml`) instead of failing with the generation's error.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the four waits is bounded, and when the generation task ends before `entered` is set the test fails at once with that task's error
- [x] #2 A negative control (a temporary raise before the chat call) makes each test fail within seconds with the raised error, not at the 300 s timeout; the notes record it
- [x] #3 The file's tests pass
<!-- AC:END -->

## Implementation Plan

1. Add one helper in `Tests/Subscriptions/test_briefing_cast.py` that waits for `entered` OR the generation task, bounded: if the task ends first it re-raises the task's error; if neither happens in time it fails.
2. Use it at the four `await entered.wait()` sites.
3. Negative control: a temporary raise before the chat call in `briefing_cast`; each of the four tests must fail within seconds with that error. Revert, run the file.

## Implementation Notes

- **Fix.** New helper `_wait_for_chat(entered, cast, timeout=30.0)` in `Tests/Subscriptions/test_briefing_cast.py` waits on `entered` and the cast task together (`asyncio.wait(..., FIRST_COMPLETED)`). If the cast ends first it calls `cast.result()`, which re-raises the cast's error; a cast that returns a row without calling chat fails with that row. If neither happens in 30 s it cancels the cast and fails. The four `await entered.wait()` sites call it.
- **Negative control (AC#2).** A temporary `raise RuntimeError("TASK-42136 negative control")` before the `try:` that wraps `_invoke_chat` in `generate_script` and `generate_script_from_text`, since reverted. Plain, these tests die locally earlier, on the brief's `RecoveryRequired: raw_source_selection_changed` trap, so the control ran with the four under a temporary `@private_profile_test` (also removed):
  - old waits, `--timeout=30`: 4 failed, each at the 30 s timeout (`Failed: Timeout`), 128 s for the four;
  - new helper: 4 failed in 9.7 s, each 2.1-2.4 s (a child pytest process), every one with `RuntimeError: TASK-42136 negative control`.
  - Without the private profile the new helper names the real cause at once: `RecoveryRequired: raw_source_selection_changed` from `cast.result()`, 0.3-1.2 s each.
- **The file (AC#3).** `-n 4` with the four under the temporary private profile: 17 failed / 50 passed; the four pass. The 17 are red at base with the same command (base copy of the file: 21 failed / 46 passed in 611 s, the 17 plus the four at 300 s each), 16 of them `RecoveryRequired` and one a regex mismatch. Plain at the end (no marker): 21 failed / 46 passed in 18 s (load 45), the same 21 nodes, where base took 611 s.
- Files: `Tests/Subscriptions/test_briefing_cast.py`.
