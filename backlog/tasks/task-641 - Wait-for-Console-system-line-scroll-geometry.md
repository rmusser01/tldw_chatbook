---
id: TASK-641
title: Wait for Console system-line scroll geometry
status: Done
assignee: []
created_date: '2026-07-26 00:42'
updated_date: '2026-07-26 00:44'
labels:
  - tests
  - textual
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Eliminate the full-suite-only Console system-prompt rail click failure by synchronizing the test with settled Textual scroll geometry before exercising the public click contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The system-prompt rail click test waits until the target is fully inside both the rail viewport and active screen before clicking.
- [x] #2 The test continues to use Pilot.click and preserves the modal-depth and empty-editor assertions.
- [x] #3 Production code is unchanged.
- [x] #4 The exact test passes in five independent processes and the full Console system-prompt test module passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the full-suite OutOfBounds failure and verify the isolated/full-order distinction.
2. Add a bounded geometry waiter using Textual Region containment, then replace the fixed post-scroll delay.
3. Run repeated exact-test, module, formatting, compile, and diff checks.
4. Re-run the full fail-fast suite before relying on the repair.

ADR required: no
ADR path: N/A
Reason: This is a test-only synchronization repair that preserves application behavior and architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Summary: Replaced the Console system-prompt rail click test's fixed post-scroll delay with a bounded wait for actual clickable geometry.

RED evidence and root cause:
- A permitted full `pytest -q -x` run reached 8,113 passed and 198 skipped before `Pilot.click` raised `OutOfBounds`; the target was still at y=78 on a 70-row screen after `scroll_to_widget(..., animate=False)` and a fixed 0.1-second pause.
- The exact test passed in isolation before the repair, confirming a load/full-order synchronization defect rather than a reproducible application modal failure.
- Repository history contained no later reviewed correction for this test.

Implementation:
- Added a test-local, monotonic-deadline geometry waiter. It requires the target to have nonzero geometry and be fully contained by both the rail body content region and active screen region.
- Kept `Pilot.click`, modal stack-depth, and empty-editor assertions unchanged.
- No production code changed.

Verification:
- Exact regression test in five fresh independent pytest processes: 5/5 passed.
- Full `Tests/UI/test_console_system_prompt.py`: 23 passed.
- Ruff format check: already formatted.
- Ruff check: all checks passed.
- `py_compile`: passed.
- `git diff --check`: passed.
- Self-review: the waiter observes the state needed by `Pilot.click`, has a bounded failure with region diagnostics, and does not catch or suppress `OutOfBounds`.

ADR required: no
ADR path: N/A
Reason: Test-only synchronization repair; application behavior and architecture are unchanged.

Files modified:
- Tests/UI/test_console_system_prompt.py
- backlog/tasks/task-641 - Wait-for-Console-system-line-scroll-geometry.md
<!-- SECTION:NOTES:END -->
