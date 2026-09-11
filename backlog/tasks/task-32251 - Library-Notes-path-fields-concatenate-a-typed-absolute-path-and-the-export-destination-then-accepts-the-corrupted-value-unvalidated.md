---
id: TASK-32251
title: >-
  Library Notes path fields concatenate a typed absolute path and the export
  destination then accepts the corrupted value unvalidated
status: Done
assignee:
  - '@claude'
created_date: '2026-09-10 18:05'
updated_date: '2026-09-11 12:00'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - pickers
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cause PROVEN: `enhanced_file_picker.py:2547,2563-2565` `_sync_dir_path_input` writes `str(nav.location)` into `#dir-path-input` on mount and on every navigation, and nothing selects the value on focus. The identical lines are present at `c4a7b1911f` -- pre-existing, newly found; #2542 added the persistent "Folder path" label, which is what made the field visible enough to find.

Reproduced exactly. Clicking the Folder-files picker's field and typing an absolute path leaves `/Users/macbook-dev/private/tmp/claude-501/...` in the field (`R/caps/20`), and the resulting error is rendered **inside the dialog's bottom border** -- `+-- Path not found: ... -+` (`R/caps/21`) -- where nobody looks. On the export destination the same concatenation is accepted silently: the canvas showed `.../exports/Library export 2026-09-10.zip/private/tmp/.../notes-bundle.zip` as the chosen destination with no warning (`R/caps/44`).

Correction to the design assessor's account: running it does **not** create the bogus directory tree. The export fails at write time with a raw Python error surfaced verbatim -- `Error creating chatbook: [Errno 2] No such file or directory: '.../notes-bundle.zip.partial'` -- and nothing lands on disk (`R/caps/45`; `ls "$HOME/Library export 2026-09-10.zip"` -> No such file). So the defect is not data damage: it is that validation is deferred to an OSError repr with an internal noun in it. The vault path had to be typed three times because of this.

Scope against the peer critique, so nobody folds these together: peer task-32229 covers ONLY a focused path Input accepting a *pasted* absolute path and jumping the tree, plus `Ctrl+A` in the file-name box being move-to-start. It does **not** cover pre-fill concatenation, and it does **not** cover the export destination accepting a corrupted value. Those two are this task.

Fix: select-all on focus for any path/name Input that arrives pre-filled -- one change at the picker level, which is also the most on-brand fix available, since this is a terminal app whose path fields behave worse than a shell prompt; move the error to an inline row under the field; validate the destination when it is chosen, not when the zip is opened.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A path Input that arrives pre-filled selects its content on focus, so typing an absolute path replaces the pre-fill instead of concatenating onto it
- [x] #2 An invalid path reports its reason on an inline row under the field, never painted into the dialog border
- [x] #3 The export destination is validated when it is chosen: a destination whose parent is not an existing directory is refused there, with a reason
- [x] #4 An export that still fails at write time reports a user-facing reason rather than an errno repr naming a `.partial` file
- [x] #5 The configured `[notes] sync_directory` is offered as a start location by both pickers where it exists
- [x] #6 Covered by a test for pre-fill select-on-focus and a test for destination validation at choose time
<!-- AC:END -->


## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live: click the pre-filled Folder-path field, type an absolute path, confirm concatenation and the border-painted refusal.
2. Find the ONE seam both picker families share (`FileSystemPickerScreen` / `base_dialog.py`) rather than patching each field.
3. RED test: click-to-focus then type replaces the pre-fill; the refusal renders on a row under the field.
4. Fix at that seam: a shared `PathInput` (task-32229's `FileNameInput` folded into it) and an inline `#picker-error-line`.
5. Export destination: validate at choose time (`describe_unusable_destination`), render the reason on the destination row, and make a residual write failure name the destination rather than a `.partial` file.
6. AC#5: one shared start-directory chain (remembered -> `[notes] sync_directory` -> home) for all four pickers.
7. Live GREEN, guide, stamps.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
Four ACs at one seam plus the export half.

**The field (AC#1).** Textual's `Input` already selects on focus
(`select_on_focus`, default on) -- the reason click-to-focus behaved
differently from Tab-to-focus is that `Input._on_mouse_down` collapses the
selection to the click point, and Textual dispatches `_on_*` to every class
in the MRO with the BASE LAST. So the shared `PathInput` in `base_dialog.py`
arms on the `Focus` that immediately precedes a `MouseDown` (an exact test
for "this click focused me"; `self.has_focus` inside `_on_mouse_down` is
always True and tells you nothing) and defers the select-all past the
dispatch with `call_next`. task-32229's `FileNameInput` became an alias, so
the vendored SelectDirectory/FileOpen/FileSave fields, the Ctrl+L path bar
and the three EnhancedFileDialog fields all get one behaviour.

**The refusal (AC#2).** `_set_error` wrote `Dialog.border_subtitle` --
rendered inside the dialog's bottom border rule. It now owns
`#picker-error-line` under the input bar; two existing tests that pinned the
border behaviour follow it there. `EnhancedFileDialog` keeps its own
`#error-line` override and reaches the base only as a tolerant fallback.

**The destination (AC#3/AC#4).** `describe_unusable_destination` asks the
three questions answerable at choose time, and the reason renders on
`#library-export-destination-line` -- the row under the button pressed --
instead of a toast. A residual `OSError` during the run now names the
destination the user chose, not the `.partial` temp file.

**Start location (AC#5).** `browse_start_directory` is one fallback chain,
remembered -> `[notes] sync_directory` -> home, shared by all four pickers.

Trade-off: the select-all is deferred, not inline, so a focusing click that
immediately DRAGS selects from the start of the field rather than from the
press point. A second click positions the cursor normally (pinned by test).

Files: `Third_Party/textual_fspicker/{base_dialog,select_directory,file_dialog}.py`,
`Widgets/enhanced_file_picker.py`, `Library/{library_browse_location,library_export_state}.py`,
`Widgets/Library/library_export_canvas.py`,
`UI/Library_Modules/{library_export_controller,library_notes_controller}.py`,
`UI/Screens/library_screen.py`, `Widgets/Library/library_file_notes_workspace.py`,
`Chatbooks/chatbook_creator.py`, `Docs/User_Guide/library/{notes,import-and-export}.md`.
<!-- SECTION:NOTES:END -->
