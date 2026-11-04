---
id: TASK-36513
title: File Notes linked-folder root row is elided to 11 cells at 60 columns
status: Done
assignee:
- '@claude'
created_date: 2026-10-23 23:37
updated_date: 2026-11-03 23:36
labels:
- follow-up
- library
- ux
priority: low
references:
- Tests/UI/test_library_file_notes_workspace.py
dependencies:
- TASK-36383
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36383's review (2026-10-23). At 60x20, the Library File Notes root row gives the linked folder's status only 11 cells, so the user reads "Chec...otes" or "Link...otes" and cannot tell the state. The always-displayed "Review recovered pairing…" button (b5251e9a6e) takes 27 of the row's cells. With that button hidden, the status fits ("Linked · Local folder: notes", 38 cells).

QA logged the button as N-37/S-20. TASK-34000.30 AC#10 sets the button's visibility rule only for the no-folder state, so the linked row stays squeezed.

Because of this defect:
- `test_file_notes_linked_root_leads_with_friendly_folder_identity[size2]` (the 60x20 case) is red on dev.
- `test_initial_root_scan_projects_checking_authority_while_actions_are_gated` checks the summary and the tooltip instead of the painted row.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 60x20 with a folder linked, the root row paints the state word (Checking, Linked, Offline or Warning) in full, and the folder name in full or middle-elided
- [x] #2 The recovered-pairing review stays reachable whenever there is a recovered pairing to review
- [x] #3 At 60x20, the row's Details and Change… controls stay on the row
- [x] #4 test_file_notes_linked_root_leads_with_friendly_folder_identity passes at every size it is parameterized over
- [x] #5 test_initial_root_scan_projects_checking_authority_while_actions_are_gated checks the painted root row again
<!-- AC:END -->

## Implementation Plan

ADR required: no
ADR path: N/A
Reason: Visibility rule for one existing control and the folder row's fit; no architectural change.

1. Measured on eba3ad14b5 (probe under `_CssTrueWorkspaceHarness`): at 60x20 the row's Details and Change… take 11 cells each and "Review recovered pairing…" 27, leaving the status 11 ("Link...otes"); at 40x20 it leaves 1 (".") and the review button runs past the right edge.
2. Show **Review pairing…** only while a folder is linked, no folder change owns the row, and the folder's scan or reconcile reported the recovery-activation warning (the service's signal that a restored pairing awaits review; the review itself needs a linked folder). Shorten the label so a Warning row with the button still has room for the folder name at 60 columns.
3. Fit the linked summary by parts: when "<state> · Local folder: <name>" does not fit, keep the state words whole, drop "Local folder:", and middle-elide the folder name; when even that leaves no room for a name, show the state words alone.
4. Tests: the 60x20 case of the friendly-identity test goes green; the checking-authority test reads the painted row again; add a Warning-with-pairing case at 60x20 (state word whole, folder name middle-elided, Details/Change… and the review button on the row) and a no-pairing case asserting the button is hidden. Show each red before the fix.
5. Run the Backup_Recovery notes pairing UI tests (they press the button), the touched test file's File Notes nodes, size ratchets, markup check, preflight; update the File Notes user guide.

## Implementation Notes

**Cause.** Measured on eba3ad14b5 under the production stylesheet: Details and Change… take 11 cells each and the always-displayed "Review recovered pairing…" took 27, leaving a linked folder's status 11 cells at 60x20 ("Link...otes") and 1 cell at 40x20 ("."), where the button also ran 10 cells past the right edge.

**Fix (`library_file_notes_workspace.py`).**
- The pairing review composes hidden. `_update_root_surface` shows it only while a folder is linked, no folder change owns the row, and the runtime warning carries the service's activation warning: the `replica_warning` every scan and reconcile reports while `require_pairing` refuses, i.e. while a restored pairing awaits review. The constant is now public, `ACTIVATION_WARNING` in `Notes/file_notes_service.py` (it was `_ACTIVATION_WARNING`, with no other users). With no folder the review cannot run ("Choose a local folder before reviewing pairing."), so it is hidden there too. The real restored-pairing UI tests (`Tests/Backup_Recovery/test_notes_recovery_controls.py`, which press the button) pass 10/10.
- Label "Review recovered pairing…" -> "Review pairing…" (27 -> 17 cells). With the long label, a Warning row at 60 columns had room for the state word only (negative control: painted 'Warning').
- `_fit_root_status` fits the linked line by parts (`_root_status_parts`): the state words stay whole, "Local folder:" goes first, then the folder name's middle, and under 5 spare cells the state words stand alone. A folder-change reason or wait line still middle-elides whole. Negative control with the part-wise fit off: 'Warning ·...rch Notes'.

**Tests (`Tests/UI/test_library_file_notes_workspace.py`).**
- New `test_file_notes_root_row_keeps_state_and_folder_at_60_columns`: Linked, Offline, Warning, and Warning with a pending pairing at 60x20; the painted row starts with the whole state word, the folder name is whole or middle-elided, the review shows only with the pairing, and Details, Change… (and the review) end within 60 columns. Red before the fix: `('Linked', False, 'Link...otes')`.
- `test_initial_root_scan_projects_checking_authority_while_actions_are_gated` reads the painted row again through `_settled_root_row` (waits for the deferred fit). Red before: 'Chec...otes'.
- `test_file_notes_linked_root_leads_with_friendly_folder_identity[size2]`: red before ('Link...otes'), green after; all three sizes pass.
- The 40x20 Offline squeeze in TASK-36528 had the same cause: with the button hidden that case stops failing on 'Offline' and fails only on its contrast assertion.

**Also.** File Notes user guide: the Review pairing… row and the narrow-fit rule. Module size ratchet re-pinned 9541 -> 9568 with a dated comment. The no-folder half of TASK-34000.30 AC#10 for this button now holds as a side effect; that task's other criteria are untouched. `Tests/UI/test_library_file_notes_git.py::test_40x20_prepare_scrolls_actions_below_a_long_linked_root` (TASK-38693, red on dev) still fails, now because the 40-column row reads "Linked · notes" rather than "."; its `"..."` expectation predates the part-wise fit.

**Side effect, measured.** `Tests/UI/test_library_notes_riders_r_file_notes.py::test_the_slow_wait_row_keeps_every_control_on_pane_at_60_columns[linked]` and `[unlinked]` (task-32180) were red on eba3ad14b5 in the same whole-file run: "At index 0 diff: 'Review recovered pairing…' != 'Cancel'". The button no longer shows during a folder change, and both pass at ab866f9619.

**Review round (fe0d40d6d9, Minor).** At 40x20 with a pairing pending, Details, Change… and "Review pairing…" took 39 cells and the status painted "." — the same symptom at a narrower width. The button is now "Review…" with the tooltip "Review the recovered folder pairing" and `min-width: 0`, since Button's 16-cell floor would otherwise apply to the short label. That leaves the status 9 cells at 40 columns, which hold the state word: when even "Offline · Warning" does not fit, Offline stays. New test `test_file_notes_root_row_keeps_state_word_beside_pairing_review_at_40_columns` covers Warning, and Offline, each with a pending pairing; it was red at fe0d40d6d9 (painted '.'). Negative control with the Offline-only fallback removed: 'Off...ing'. The user guide's narrow-fit copy now says "down to 40 columns".

**Files.** `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`, `tldw_chatbook/Notes/file_notes_service.py`, `Tests/UI/test_library_file_notes_workspace.py`, `Tests/Architecture/test_module_size_ratchet.py`, `Docs/User_Guide/library/file-notes.md`.
