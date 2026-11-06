---
id: TASK-39062
title: "Library Artifacts: in the empty narrow stage, Tab from the list walks four reader controls painted at zero or negative width"
status: To Do
created_date: 2026-11-06 12:31
labels:
- follow-up
- library
- focus
- accessibility
priority: low
references:
- tldw_chatbook/Widgets/Library/library_artifacts_reader_shell.py
- tldw_chatbook/Widgets/adaptive_pane_shell.py
- tldw_chatbook/UI/Library_Modules/library_artifacts_controller.py
- Tests/UI/test_library_artifacts_focus.py
- scripts/ui_pr_gate_census.txt
dependencies:
- TASK-38981
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on the Library branch that lands TASK-38981 (the state after it merges). TASK-38981's notes (re-review fixes, r10l-rev2) record it as a pre-existing follow-up. Line numbers are from the branch.

**What TASK-38981 changed.**
- TASK-38981 hides a zero-width work pane in every Library destination's narrow stage.
- Artifacts opts out (`LibraryArtifactsReaderShell.hide_zero_width_work = False`, `library_artifacts_reader_shell.py:30`), because F6 or Tab into its reader reveals it at full width (`reveal_focused_reader`, `:78`).
- With no artifact selected there is nothing to reveal, so the reader's controls stay in the focus chain off screen.

**Probe.** A mounted Library with no reports, `artifacts-reports` selected, Tab from `#library-artifacts-list`. The result is identical on dev and after TASK-38981:

| Size | Layout | Tab walk (control, visible cells) |
|---|---|---|
| 60x24 | `items_open` True, list 50, `reader_width` 0 | items-grip 5, **`library-artifacts-back` -1, `-preview` -11, `-details` -21, `-body` -1**, library-grip 5, search 50, all 15 |
| 50x25 | list 40, `reader_width` 0 | the same four reader stops, at -1 / -11 / -21 / -1 |

- The top bar's `nav-*` buttons also sit at negative cells in the chain, but they scroll into view when focused (8-15 cells each in the 80-column walk), so they are not part of this.
- At 80x25 with nothing selected, the list is closed and the empty reader takes the 70-cell stage. That is not examined here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With no artifact selected at 60x24 and 50x25, every Tab stop reached from the Artifacts list has visible cells; a mounted test is red before the change (Tab reaches `library-artifacts-back` at -1 cells)
- [ ] #2 With an artifact open and the list holding the stage, `test_items_opened_beside_an_open_artifact_take_the_stage` and `test_narrow_f6_reveals_reader_and_returns_through_items_grip` stay green
- [ ] #3 The new test runs on a PR lane (a census tail entry) and passes 10 of 10 at `-n 4`, with the load recorded
<!-- AC:END -->
