---
id: TASK-42460
title: "The Trace long-inspector test reads the overflow cue after one pause and sometimes finds it empty, on no PR lane"
status: To Do
assignee: []
created_date: 2026-11-23 06:38
updated_date: 2026-11-23 07:09
labels:
- follow-up
- trace
- tests
- flaky
priority: low
references:
- Tests/UI/test_trace_responsive.py
- tldw_chatbook/UI/Screens/trajectory_screen.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), in TASK-42414's covering runs. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches); paths are under `tldw_chatbook/` unless they start with `Tests/` or `scripts/`.

**The flake.** `Tests/UI/test_trace_responsive.py::test_long_inspector_is_focusable_scrolls_to_painted_bottom_and_full_pane` (line 198) failed once at load 47 with `assert '▼ more — scroll' in ''` at line 225, then passed 4 of 4 serially. In 32 repeats at `-n 4` it failed 1 and 0 times with dev's Trace screen file, and 2 and 0 times with TASK-42414's, so it flakes at dev too. The test presses Enter, waits one `pilot.pause()` (line 216), and reads `#trajectory-inspector-overflow`. The assert just before it, `inspector.max_scroll_y > 0` (line 224), passed, so the overflow was there and only the cue text was missing. The cue is written by `_schedule_inspector_cue` (`UI/Screens/trajectory_screen.py:1675`), which runs a second pass 10 ms later against the inspector's layout. Whether the read lands before that pass, or the cue pass runs before layout is final, is not known.

**CI.** The file is in no PR lane: it is not in `scripts/ui_pr_gate_census.txt` and no `derived-artifacts.yml` step names it, so CI has never run it on a pull request. TASK-42414's scan of 84 failed Derived Artifacts runs (2026-11-20 03:35Z to 2026-11-22 18:09Z) found no Trace node failing other than its own. Low priority for that reason; once the file is laned, this becomes a lane flake.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The race is named with evidence: a temporary delay (reverted) in the cue's pass or the inspector's layout makes the failure at line 225 reproducible
- [ ] #2 A product race is fixed in the screen, so the cue is right whenever the inspector overflows; a test-timing race is fixed in the test, which waits for the cue with a load-tolerant deadline. The test keeps its assertions
- [ ] #3 The test passes in repeated runs of its file under parallel load (at least 32 runs at `-n 4`, or a soak recorded in the notes)
- [ ] #4 The file runs in a PR lane, with its serial time under the lane's flags recorded in the notes
<!-- AC:END -->
