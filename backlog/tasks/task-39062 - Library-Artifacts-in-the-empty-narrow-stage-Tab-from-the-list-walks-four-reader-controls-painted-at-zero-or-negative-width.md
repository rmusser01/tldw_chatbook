---
id: TASK-39062
title: "Library Artifacts: in the empty narrow stage, Tab from the list walks four reader controls painted at zero or negative width"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 12:31
updated_date: 2026-11-06 16:13
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
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on the round-10 Library branch that lands TASK-38981 (not yet a final pushed head). TASK-38981's notes (re-review fixes, r10l-rev2) record it as a pre-existing follow-up. The branch's code is named by symbol, since its lines are still moving.

**What TASK-38981 changed.**
- TASK-38981 hides a zero-width work pane in every Library destination's narrow stage.
- Artifacts opts out (`LibraryArtifactsReaderShell.hide_zero_width_work = False`, in `library_artifacts_reader_shell.py`), because F6 or Tab into its reader reveals it at full width (`LibraryArtifactsReaderShell.reveal_focused_reader`).
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
- [x] #1 With no artifact selected at 60x24 and 50x25, every Tab stop reached from the Artifacts list has visible cells; a mounted test is red before the change (Tab reaches `library-artifacts-back` at -1 cells)
- [x] #2 With an artifact open and the list holding the stage, `test_items_opened_beside_an_open_artifact_take_the_stage` and `test_narrow_f6_reveals_reader_and_returns_through_items_grip` stay green
- [x] #3 The new test runs on a PR lane (a census tail entry) and passes 10 of 10 at `-n 4`, with the load recorded
<!-- AC:END -->

## Implementation Plan

1. Write a mounted test: no reports, `artifacts-reports` selected, at 60x24 and 50x25, Tab from `#library-artifacts-list` through the chain and assert each stop has visible cells inside its compositor clip. Show it red on the base (Back at -1 cells).
2. Fix in `LibraryArtifactsReaderShell`: hide the zero-width work pane while no artifact is selected (nothing for a reader control to reveal), and keep it displayed when one is, so Tab or F6 into the reader still reveals it. A selection change re-applies the layout, so the pane comes back as soon as an artifact is selected.
3. AC#2: `test_items_opened_beside_an_open_artifact_take_the_stage` and `test_narrow_f6_reveals_reader_and_returns_through_items_grip` stay green, with the rest of the Artifacts focus file.
4. Census tail entries for the new nodes; 10 of 10 at `-n 4` with the load.

## Implementation Notes

- **Fix.** `LibraryArtifactsReaderShell.hide_zero_width_work` is now a property instead of `False`. It returns `controller.selected is None`.
  - With an artifact selected, the zero-width reader stays displayed, so Tab or F6 onto any reader control still reveals it at full width (`reveal_focused_reader`, TASK-38981).
  - With nothing selected, `focus_reader` has nothing to reveal, so the reader hides the way every other destination's does, and its controls leave the focus chain.
  - The shell's `sync()` runs after every selection change. It now re-applies `_applied_layout` through `sync_layout`, so the reader is shown again as soon as an artifact is selected. `sync_layout` changes nothing in place when its layout is unchanged; it moves focus only out of a pane it hides.
- **Test (AC#1).** `test_empty_narrow_stage_tab_never_rests_on_an_unseen_reader_control[60x24, 50x25]` mounts Library with no reports. `artifact_library` gained a `reports=` count; at 0 it stops waiting once the page has loaded. The test Tabs one whole lap from `#library-artifacts-list` back to it and measures each stop's cells inside its compositor clip.
  - **Red** before the fix, at both sizes: `[('library-artifacts-items-grip', 5), ('library-artifacts-back', -1), ('library-artifacts-preview', -11), ('library-artifacts-details', -21), ('library-artifacts-body', -1), ...]`.
  - **Green** after it.
- **AC#2.** The whole Artifacts focus and canvas files pass: 28 passed at `-n 4`, load 43 on 18 CPUs. That includes `test_items_opened_beside_an_open_artifact_take_the_stage[3 sizes]`, `test_narrow_f6_reveals_reader_and_returns_through_items_grip[2 sizes]` and `test_narrow_empty_items_skips_zero_width_reader`.
- **AC#3.** Both nodes are appended at the census tail, and the floor is the measured count (1753). They took about 11 s each at `-n 4` under load 43. A temporary 10-repeat parametrize (since removed) gave 20 of 20 at `-n 4` in 100.6 s, at load 19-40.
- **User Guide.** `Docs/User_Guide/artifacts.md` ("Browse and read") says that Tab and F6 bring the reader forward on a narrow terminal, and that with nothing selected Tab skips the reader.
- Files: `tldw_chatbook/Widgets/Library/library_artifacts_reader_shell.py`, `Tests/UI/test_library_artifacts_focus.py`, `Tests/UI/test_library_artifacts_canvas.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `Docs/User_Guide/artifacts.md`.
