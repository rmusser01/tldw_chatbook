---
id: TASK-32623
title: >-
  Library Notes: typographic and copy inconsistencies across the editor, Folder
  files and the chooser
status: Done
assignee:
  - '@robert'
created_date: '2026-09-15 06:43'
updated_date: '2026-09-15 18:45'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor B D15 and assessor A heuristic 2, every persona. A cleanup pass, filed as one task because each item is a one-line fix in a different file.

- 'Unavailable - server sync-folder capability not installed' uses a hyphen where the rest of the screen uses an em dash (B cap 29).
- Word counts disagree between two panes describing the same note: the editor footer reads 5,453 words and the Info pane reads 5454 (B caps 11 and 16). Residual of task-32538, which fixed a much larger error (404 for 5,407).
- The Folder-files footer reads with a leading space, a double space, and an instruction about a future state: 'typing in field | esc notes |  after esc: / focus search' (B cap 34).
- The Folder-files empty state offers 'Use file_notes' as a button label -- an internal config key shown to a first-timer (A cap 26). This is the single item that cost heuristic 2 its fourth point.
- The 'ctrl+end end of note' footer chip stays visible while focus is on a button, where the key does nothing (A section 11).
- The Synced placement badge is repeated on all 54 rows of a folder whose own row already says Sync managed (A section 11).
- Mode-row buttons shift horizontally when Discard new note appears and disappears, moving click targets under the cursor (A caps 04 to 05).

Cause PROVEN by capture for every item.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One dash convention across the surface
- [x] #2 Two panes describing the same note report the same word count
- [x] #3 Footer strings carry no stray whitespace and describe the current state, not a future one
- [x] #4 No button label is an internal config key
- [x] #5 A footer chip is shown only where its key does something, and controls do not move under the pointer as state changes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace each of the seven items to its exact source string/mechanism.
2. Item 1 (hyphen vs em dash): fix the one literal default string in library_notes_lasting_sync_state.py; sweep for other quoters (test, guide).
3. Item 2 (word-count mismatch): trace both display sites to confirm they share one computed int (they do, via LibraryNotePresentationState); the only real defect is Info's copy lacking the thousands-separator formatting the editor footer already has -- fix the format, not any timing/race.
4. Item 3 (leading/double space + "future state" footer text): root-cause to ShortcutAction.render()'s unconditional f"{key} {label}" -- an empty key produces a leading space, doubled wherever the joiner (" | ") lands next to it. Fix once in the shared render(), which reaches every "" -key chip across the app, not just Library. Judge the "after esc: ..." construct itself as intentional existing design (extensively reasoned in nearby comments across three prior tasks), not a defect this task should relitigate.
5. Item 4 ("Use file_notes" button label): trace _folder_label()/_configured_sync_folder() -- confirm the label is always the user's own configured folder's basename (never a hardcoded config-section/key string) and sweep the touched files for any other config-key-as-label instance. Record as INFERRED if it cannot be verified live.
6. Item 5 (dead ctrl+end chip off the body): gate the chip on self.focused being the note body specifically, mirroring the existing _library_focus_enter_label id-check idiom so it survives the module's SimpleNamespace-fake tests.
7. Item 6 (repeated Synced-placement badge): drop the note row's redundant status_text in the managed+active case only (the folder's own row already carries it); keep semantic_status and the needs-attention case unchanged.
8. Item 7 (mode-row shift): trace to #library-note-task-actions's width:auto shrinking when Discard new note's `display` toggles off, which moves its width:auto sibling #library-note-mode-controls under a shared width:auto parent. Reserve the row's widest width via CSS min-width (edited in the true source, components/_agentic_terminal.tcss, rebuilt with build_css.py) rather than changing the button's display/visible semantics, to avoid the ~17-reference blast radius across test_library_shell.py's existing `.display` assertions/waits for this exact button.
9. Add or update one test per item; prove each red against a reverted copy of its fix before restoring.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Seven one-line-per-file findings, each root-caused separately.

1 (hyphen/em dash): LastingSyncSetup.server_disabled_reason's default
("Unavailable - server sync-folder capability not installed") was the one
hyphen holdout; the screen's own _disabled_action_label() grammar already
uses an em dash for this exact sentence's grammar elsewhere. Fixed the
literal, plus the two tests and the one guide line quoting it verbatim.

2 (word-count mismatch, "5,453" vs "5454"): both displays are built from
the SAME int inside one LibraryNotesController._library_note_presentation_state()
call (word_count=word_count is fed to both the chrome-strip and the meta
line in the same method) -- they cannot diverge in VALUE within one
coherent render. The only real, permanent defect was FORMAT: the editor
footer's library_note_chrome_facts() already used f"{word_count:,}", Info's
word_copy did not. Fixed Info to match. (The specific "5453 vs 5454"
reading in the critique's capture is most plausibly two non-simultaneous
screenshots with a keystroke landing between them, not a same-render bug --
INFERRED, not reproduced live.)

3 (leading/double space + "future state" wording): root-caused to
ShortcutAction.render() in UI/Navigation/shortcut_context.py, which
unconditionally returned f"{key} {label}" -- a "" key (Library's "typing in
field" state word, and its "after esc: ..." swallowed-keys summary) always
carried a literal leading space, doubled wherever the " | " joiner landed
next to it. This is the SAME render path every screen's footer goes
through (AppFooterStatus._render_actions and ShortcutContext.render()
both call it), so the one-line fix reaches Folder-files' identical footer
for free -- it is the same "mode of Notes" screen and mechanism, not a
second copy to patch. The "future state" half of the finding ("after esc:
...") is judged intentional, well-reasoned existing design (task-31223/
32346's extensive comments explain exactly why swallowed keys are
advertised this way) rather than a defect; not touched.

4 ("Use file_notes" button label): traced _configured_sync_folder() (reads
[file_notes] root / [notes] sync_directory from config) and _folder_label()
(returns path.name) -- the button always shows the user's OWN configured
folder's basename, never a hardcoded config-section/key string. Swept the
touched files (editor, Folder files, chooser) for any other
config-key-as-label pattern; found none. Not a defect: the cited capture
is a scratch/test fixture folder that happened to be named literally
"file_notes" on disk. **REPRODUCED, not INFERRED (review round 1, F4)**:
the settling experiment already existed in the tree --
Tests/UI/test_library_notes_riders_r_file_notes.py::
test_use_folder_offers_the_modern_file_notes_root configures
[file_notes] root / [notes] sync_directory at two OTHER folder names
("modern-vault", "legacy-sync") and asserts the button label follows suit
("Use modern-vault", "Use legacy-sync") -- confirming the label is always
the configured folder's own basename, not a hardcoded key. No code
change; no new test needed, the existing one already proves it.

5 (dead ctrl+end chip): LIBRARY_NOTES_EDITOR_SHORTCUTS's ctrl+end entry
was static regardless of which control inside the editor region actually
held focus -- ctrl+end is a TextArea-class binding (task-32247), dead the
instant Tab (task-32246) moves focus to a toolbar Button. Gated it on
self.focused being #library-note-body, via the same
getattr(focused, "id", "") idiom _library_focus_enter_label already uses
(needed for the pinned SimpleNamespace-fake unit test, which has no
query_one).

6 (repeated Synced-placement badge): _note_row() (library_notes_tree_state.py)
gave the SAME "⇄ Synced placement" text to every note under a managed+
active folder, redundant with that folder's own "⇄ Sync managed" row.
Dropped the note row's status_text (keeping semantic_status="connected",
so any status-keyed styling is untouched) in that one case; the
needs-attention case is real per-note information and keeps its text.

7 (mode-row shift): #library-note-primary-actions is width:auto, sized as
the sum of its two Horizontal children (#library-note-mode-controls,
#library-note-task-actions); Discard new note's Python-level `display`
toggle shrank task-actions' content width, which shrank primary-actions'
auto width, which (primary-actions is effectively right-anchored inside
its row by its Static sibling's width:1fr) moved mode-controls' whole box
-- and Edit/Preview/Info inside it -- left by exactly Discard's width
whenever it appeared. Verified empirically (region.x probes) before and
after. Fixed via CSS: #library-note-task-actions now carries a measured
min-width (61 cells, the row's own widest state with all three buttons
shown) at wide terminals, with a compact-mode override restoring 0 (compact
already didn't shift -- measured, unaffected). Edited the true source
(css/components/_agentic_terminal.tcss, per its own "GENERATED FILE" header
on the derived screen_agentic_library.tcss) and reran build_css.py;
tldw_chatbook/css/check_bundle_sync.py confirms all five generated
stylesheets still reproduce from source. Chose CSS min-width over switching
the button's display to Textual's `visible` (which also reserves space)
specifically to avoid ~17 existing `.display` assertions/waits on this one
button across Tests/UI/test_library_shell.py -- zero Python behaviour
changed for this item.

Tests: one new/updated test per item (see file list). Each new pin proven
red first: shortcut_context.py's render() reverted to the old
f"{key} {label}" (test_app_footer_shortcut_context.py's two new tests
red), library_notes_lasting_sync_state.py's string reverted to a hyphen
(existing tests red -- string literal only, no new test needed), library_
notes_controller.py's word_copy format reverted (new data-truth test red),
library_screen.py's ctrl+end focus gate removed (new editor-keys test
red), library_notes_tree_state.py's status_text reverted (new tree-state
test red), and the CSS min-width rule reverted at HEAD via `git show
HEAD:<path> > <path>` -- never `git checkout --` on the tracked
worktree file -- (new reader test red, at 235 columns specifically: 170,
this suite's usual LIBRARY_TEST_SIZE, does not reproduce the shift).

Full-suite check: Tests/Library/test_library_notes_lasting_sync_state.py +
test_library_notes_tree_state.py + Tests/UI/test_library_honesty_
accessibility.py + test_library_notes_reader.py + test_library_notes_w4_
data_truth.py + test_library_notes_wave_editor_keys.py + Tests/Widgets/
Library/test_library_notes_add_from_files_canvas.py + Tests/UI/test_app_
footer_shortcut_context.py = 4 pre-existing baseline reds (verified
identical on origin/dev: test_library_disabled_contrast_rules_live_in_
source_and_bundle, test_media_canvas_actions_share_one_toolbar_row,
test_escape_works_on_export_and_staging_canvases, test_rail_entry_to_
export_after_media_origin_does_not_claim_media) plus 2 tests that fail only
when run in the full combined set and pass in isolation
(test_reader_route_parks_dirty_note_selection_and_preview_without_saving,
test_delete_receipt_is_dismissed_leaving_the_list_for_folder_files) --
pre-existing cross-file test pollution, not a regression from this task.
Discard/task8/60x20-related subset of Tests/UI/test_library_shell.py also
checked: 5 pre-existing baseline reds (verified identical on origin/dev),
25 passed.

Modified: tldw_chatbook/Library/library_notes_lasting_sync_state.py,
tldw_chatbook/UI/Library_Modules/library_notes_controller.py,
tldw_chatbook/UI/Navigation/shortcut_context.py,
tldw_chatbook/UI/Screens/library_screen.py,
tldw_chatbook/Library/library_notes_tree_state.py,
tldw_chatbook/css/components/_agentic_terminal.tcss (source) +
tldw_chatbook/css/screen_agentic_library.tcss (regenerated),
Docs/User_Guide/library/notes.md; test files: Tests/Library/
test_library_notes_lasting_sync_state.py, test_library_notes_tree_state.py,
Tests/UI/test_library_honesty_accessibility.py, test_app_footer_shortcut_
context.py, test_library_notes_reader.py, test_library_notes_w4_data_truth.py,
test_library_notes_wave_editor_keys.py, Tests/Widgets/Library/
test_library_notes_add_from_files_canvas.py.
<!-- SECTION:NOTES:END -->
