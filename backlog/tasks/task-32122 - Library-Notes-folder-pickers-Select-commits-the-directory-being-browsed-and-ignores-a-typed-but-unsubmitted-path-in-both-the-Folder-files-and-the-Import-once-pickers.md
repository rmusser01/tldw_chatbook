---
id: TASK-32122
title: >-
  Library Notes folder pickers: Select commits the directory being browsed and
  ignores a typed but unsubmitted path, in both the Folder files and the Import
  once pickers
status: Done
assignee: []
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 06:40'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - file-notes
  - import
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PROVEN in code and live: `EnhancedSelectDirectory._select_viewed_directory` returns `_dir_nav().location`; the typed path only takes effect on Enter. The Import once picker is `FileOpen(offer_select_folder=True)`, whose field is labelled 'File name' and is never read by 'Select folder'. Both assessors typed the vault path and pressed Select and got the source repository (Import once) or the home directory (Folder files, which then triggered task-32121). The confirmation shows only a basename ('1 folder selected: notes-review'), so the mistake is invisible, and the picker's own Enter-vs-Select hint never appeared on screen. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pressing Select (or Select folder) after typing an absolute path selects that path, or shows an inline error naming why it did not resolve; it never silently substitutes the browsed directory
- [x] #2 In folder mode the field is labelled 'Folder path', never 'File name'
- [x] #3 The selection confirmation shows the absolute path, not the basename
- [~] #4 Partially satisfied: both pickers share the "else home" start-location fallback (Import once/Keep-synced/Folder files all previously fell back to the process cwd or an ad-hoc home() call; now uniformly `Path.home()` via a shared `resolve_default_location` helper in `base_dialog.py`) and a hint line that is visible in both. NOT implemented: true "last used directory" persistence for the vendored `FileOpen`/`SelectDirectory` pickers -- that mechanism only exists today on the app's own `EnhancedFileDialog` family (`context=` + `filepicker.last_dir_<context>`), which neither Import once, Keep-synced, nor Folder files use; adding it would need either a new `context=` kwarg threaded through call sites this task's brief scoped as copy-only, or a fresh title-keyed persistence scheme not asked for with that specificity. Left for a follow-up if wanted.
- [x] #5 Covered by tests for typed-path-then-Select in both pickers
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. EnhancedSelectDirectory (enhanced_file_picker.py): extract _resolve_typed_directory(value) helper from _on_dir_path_submit; use it in both _on_dir_path_submit and _select_viewed_directory so Select resolves the typed field first; add a persistent 'Folder path' Label.
2. Vendored SelectDirectory (select_directory.py): same fix -- typed #path_input value resolved before dismiss on #select; add 'Folder path' Label; add a hint line via a new _hint_text() hook.
3. Vendored FileOpen/base_dialog.py: fix _select_current_folder/action_select_current_folder to resolve the InputBar's Input value first via a shared _resolve_select_folder_target() helper (falls back to DirectoryNavigation.location when the field is empty); add the same _hint_text() hook to FileSystemPickerScreen.compose() (empty by default, overridden by SelectDirectory/FileOpen(offer_select_folder=True)); default location '.' -> Path.home() in the shared base (fixes 'Import once opens at cwd').
4. file_dialog.py (BaseFileDialog._input_bar): swap the 'File name:' label to 'Folder path:' live when offer_select_folder is set and the typed value resolves to a directory.
5. Confirmation copy: project_library_note_import_snapshot folder-mode selected_names carries the absolute path (not path.name); add Utils.elide_path_middle (basename-preserving) and use it in library_note_import_canvas._bounded_source_name and library_notes_add_from_files_canvas's folder summary.
6. TDD throughout; live-verify on nw-pickers; docs stamps; backlog Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed the shared vendored/enhanced folder pickers, not the call sites, per the constraints doc.

EnhancedSelectDirectory (enhanced_file_picker.py): extracted _resolve_typed_directory(value) from
_on_dir_path_submit; _select_viewed_directory now calls it too, so Select resolves the #dir-path-input
field first and only falls back to the browsed directory when the field is unchanged/empty. Added a
persistent "Folder path:" Label.

Vendored SelectDirectory (select_directory.py, backs Folder files' "Choose folder..."): same
_resolve_typed_directory pattern for #path_input, wired into both Enter-submit and the #select button.
Added a persistent "Folder path:" Label and a _hint_text() override ("Enter Open . Select use this
folder").

Vendored FileOpen(offer_select_folder=True) (backs Import once and Keep-synced): the bug lived in the
shared base_dialog.py -- _select_current_folder/action_select_current_folder always dismissed with
DirectoryNavigation.location, never reading the "File name" field. Added
_resolve_select_folder_target() (reads InputBar's Input, empty -> current dir, else resolve+validate)
and routed both call sites through it. file_dialog.py's BaseFileDialog._input_bar now swaps the label
between "File name:" and "Folder path:" live via an Input.Changed handler, based on whether the typed
text currently resolves to a directory (FileSave never sets offer_select_folder, so it is unaffected).
file_open.py adds the matching _hint_text() override.

Start location (Step 4, partial -- see AC #4 note in the task file): added
base_dialog.resolve_default_location() and used it in FileOpen.__init__ and SelectDirectory.__init__ so
a caller-omitted location=(".") resolves to Path.home() instead of the process cwd -- this is scoped to
those two classes deliberately, NOT the shared FileSystemPickerScreen.__init__: an earlier attempt to
put it there broke ~117 EnhancedFileOpen-family tests that rely on "." meaning "the actual process cwd"
(e.g. monkeypatch.chdir(tmp_path) fixtures) via EnhancedFileDialog's own separate resolve_file_picker_
start mechanism. True per-context "last used directory" persistence for the vendored pickers was NOT
added (no context= concept exists on them, and adding one needs either call-site kwargs the brief
scoped as copy-only, or a new persistence scheme not asked for with that specificity) -- left as a
follow-up.

Confirmation copy (Step 3): project_library_note_import_snapshot's folder-mode branch now keeps
str(path) instead of path.name. Added Utils.elide_path_middle(text, budget) (middle-elide, basename
always intact) and used it in library_note_import_canvas._bounded_source_name (Import once: "1 folder
selected: /.../vault") and library_notes_add_from_files_canvas's folder summary (Keep-synced), which
already carried the full path but with no eliding at all.

REGRESSION CAUGHT AND FIXED: an early version of _resolve_typed_directory always ran the typed value
through .resolve(), including the untouched case where the field just mirrors the browsed directory --
on macOS "/tmp" resolves to "/private/tmp", so a plain no-typing Select silently changed its own
result (caught by Tests/UI/test_library_modal_dismissal.py's SelectDirectory contract test). Fixed by
returning the *unmodified* DirectoryNavigation.location object whenever the field's text equals the
currently-browsed location, resolving only genuinely retyped text.

Verified against dev: swapped all 5 touched files back to HEAD's originals and confirmed test_
workspace_create_modal.py's 13 failures and test_console_modal_dismissal.py's 6 architecture-inventory
failures reproduce identically unmodified -- pre-existing, unrelated to this change. Restored my files
afterward.

Live-verified on nw-pickers (power profile, 235x52): Import once opened at $HOME (not cwd), typed the
vault path (no Enter), the "File name:" label flipped live to "Folder path:", pressed Select folder ->
"1 folder selected: /private/tmp/.../vault" (middle-elided, basename intact). Folder files' "Choose
folder..." opened at $HOME, showed the persistent "Folder path:" label and the "Enter Open . Select use
this folder" hint, typed the vault path, pressed Select -> "Folder Files . Folder: vault / Linked ..
Local folder: vault". Captures saved under notes-crit/wave/pickers/caps/01-06.

Files: tldw_chatbook/Widgets/enhanced_file_picker.py; tldw_chatbook/Third_Party/textual_fspicker/{base_
dialog,file_dialog,file_open,select_directory}.py; tldw_chatbook/Library/library_note_import_state.py;
tldw_chatbook/Widgets/Library/{library_note_import_canvas,library_notes_add_from_files_canvas}.py;
tldw_chatbook/Utils/Utils.py. Tests: new Tests/UI/test_file_open_select_folder.py (8),
Tests/UI/test_select_directory_typed_path.py (6); extended Tests/UI/test_enhanced_select_directory.py
(+3), Tests/Library/test_library_note_import_state.py (+1), Tests/Widgets/Library/test_library_note_
import_canvas.py (+1), Tests/Widgets/Library/test_library_notes_add_from_files_canvas.py (+1). Docs:
notes.md and file-notes.md updated with stamps.
<!-- SECTION:NOTES:END -->
