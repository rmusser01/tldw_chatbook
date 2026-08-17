---
id: TASK-16477
title: Roleplay export-marked-JSON picker joins the enhanced filepicker family
status: Done
assignee:
  - '@robert'
created_date: '2026-08-15 22:26'
updated_date: '2026-08-15 22:45'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Roleplay (personas) screen uses the enhanced file-picker family everywhere except 'Export marked items as JSON', which pushes the bare vendored SelectDirectory dialog: smaller chrome, no breadcrumbs/search/bookmarks/hints, no remembered start directory. Make it consistent with the pickers used across the Roleplay/Console screens.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Export marked rows as JSON opens the enhanced-family directory picker (breadcrumbs, search, bookmarks, hints, 95% modal)
- [x] #2 Directories only are listed (no file entries)
- [x] #3 Select returns the directory currently viewed; Esc/cancel returns None
- [x] #4 Dialog remembers its last directory per context like the other enhanced pickers
- [x] #5 Tests cover mount, dirs-only listing, select-returns-viewed-dir, path-input navigation
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add EnhancedSelectDirectory to Widgets/enhanced_file_picker.py (dirs-only EnhancedFileDialog; select returns viewed dir; synced path input; suppresses file-select handlers)
2. Replace the vendored SelectDirectory push in personas_screen._export_marked_json_worker
3. Add tests (mount, dirs-only, select result, path submit, breadcrumbs on mount, personas worker dialog type)
4. Run picker + personas test suites and lint
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added `EnhancedSelectDirectory` to `tldw_chatbook/Widgets/enhanced_file_picker.py`: an `EnhancedFileDialog` subclass that mirrors the vendored `SelectDirectory` contract (directory-only listing; select returns the directory currently viewed) on the enhanced chrome, so it inherits breadcrumbs, search, bookmarks/recent sidebar, hints bar, the 95%×95% modal sizing, and the per-context remembered start directory (`filepicker.last_dir_{context}` / recents persistence).
  - Directories-only comes from `nav.show_files = False` in `on_mount`, exactly where the vendored dialog sets it.
  - The file-flow select button is suppressed by extending `_SUPPRESSED_BASE_HANDLERS` with `EnhancedFileDialog._on_select_button` (it would query the absent `#filename-input`); a directory-mode `#select` handler dismisses with the viewed location instead.
  - The input bar yields a `#dir-path-input` synced with navigation (and navigable on submit, mirroring the vendored dialog's path input); directory-mode hints say "Enter Open / Select use this folder" rather than the file-mode Enter-confirm.
- Migrated `personas_screen._export_marked_json_worker` from the vendored `SelectDirectory` to `EnhancedSelectDirectory(title="Export N items as JSON", context="character_export_dir")`. Same return type (viewed directory or None), so the write loop is unchanged.
- Tests: new `Tests/UI/test_enhanced_select_directory.py` (9 tests: mount, dirs-only, breadcrumbs-on-open + path-input sync, select-returns-viewed-dir, cancel-returns-None, typed-path navigation, bad-path error, hints text, constructor shape) and `test_bulk_export_marked_pushes_enhanced_directory_picker` in `Tests/UI/test_personas_workbench.py` pinning the dialog type/title/context at the call site.
- Verification: `Tests/UI/test_enhanced_select_directory.py` + the seven existing picker test files (103 passed), full `Tests/UI/test_personas_workbench.py` (319 passed), `Tests/UI/test_non_obscuring_focus_contract.py` (included in the 101-test rerun), `ruff check` clean on all touched files. Rendered the dialog headless under the real app CSS bundle to confirm the enhanced chrome (breadcrumb bar, hints, directory-only list) shows on open.
- Investigation note: the Roleplay screen's other pickers (import card, avatar/expression upload, lore/dictionary import, exports) and the Console's attach/save pickers already share the identical `EnhancedFileDialog` family — the marked-JSON export was the sole bare-vendored dialog in that screen. The Library screen still uses the vendored family throughout; migrating it is out of scope here.
<!-- SECTION:NOTES:END -->

ADR required: no
ADR path: N/A
Reason: UI consistency change within the existing enhanced file-picker family (Widgets/enhanced_file_picker.py); no new boundary, storage, or service contract is introduced.
