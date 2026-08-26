---
id: TASK-14880
title: Clarify File Notes local-folder authority at moderate and compact widths
status: Done
assignee:
  - '@codex'
created_date: '2026-08-10 18:45'
updated_date: '2026-08-10 19:08'
labels:
  - notes
  - library
  - ux
  - accessibility
priority: medium
references:
  - .impeccable/critique/2026-08-10T06-12-44Z__ok-widgets-library-library-file-notes-workspace-py.md
documentation:
  - backlog/decisions/011-chatbook-workbench-ui-system.md
  - backlog/decisions/029-file-notes-disk-authority.md
---

## Description

Make the Library File Notes surface answer where edits happen without clipping its authority explanation or reducing the linked root to middle-elided path telemetry. Preserve exact-path access, disk authority, recovery warnings, retained navigation, and compact-terminal capacity.

## Acceptance Criteria

- [x] The Files/Sync placement sentence is complete at 120x40 and has an intentional treatment of at most two lines at 60x20 instead of silent clipping.
- [x] A linked root presents its state and a friendly local-folder name in the persistent root row; the exact canonical path and any detailed warning remain available through Details and pointer help.
- [x] Empty, checking, offline, warning, Choose folder, and Change flows remain explicit and keyboard reachable without changing disk, replica, Sync, or Session Git behavior.
- [x] Mounted regressions prove complete authority copy, friendly root identity, exact-detail preservation, and stable body geometry at wide, moderate, and compact supported sizes.
- [x] Focused tests, targeted Ruff, Python compilation, CSS/diff integrity, and self-review pass.

## Implementation Plan

ADR required: no
ADR path: N/A; conform to `backlog/decisions/011-chatbook-workbench-ui-system.md` and `backlog/decisions/029-file-notes-disk-authority.md`.
Reason: this is an atomic copy and responsive-presentation refinement within the existing workbench and disk-authority contracts; it changes no storage, synchronization, ownership, or service boundary.

1. Add mounted regressions at 160x45, 120x40, and 60x20 for purpose-copy completeness, bounded wrapping, friendly root identity, exact details, and retained body capacity.
2. Replace the long contrastive sentence with concise Files/Sync authority copy and allow one or two natural rows based on available width.
3. Separate the human root summary from exact root telemetry so the persistent row leads with state and folder name while Details retains the canonical path and warning.
4. Run the complete affected File Notes and CSS contract suites, mutation-check the new geometry/identity guards, and verify no storage or Git path changed.
5. Self-review, record implementation evidence, mark every acceptance criterion complete, and move TASK-14880 to Done.

## Implementation Notes

- Replaced the clipped contrastive purpose sentence with two parallel authority statements: Files edits the selected folder directly; Sync mirrors files into Library. The purpose Static now sizes to one row at 160x45 and 120x40 and wraps completely into exactly two rows at 60x20.
- Split persistent root presentation from exact telemetry. The root row now leads with `Linked`, `Offline`, `Checking`, or `Warning` plus `Local folder: <name>`; the existing Details dialog and pointer help retain the canonical path and full recovery warning.
- Preserved the root chooser, Change and Details actions, empty/offline behavior, responsive navigator/editor body, replica ownership, disk saves, Sync boundary, and Session Git behavior. No storage, service, or generated CSS bundle changed.
- Added seven mounted cases covering complete copy, bounded height, retained body capacity, friendly identity at three supported sizes, warning visibility, tooltip detail, and the keyboard-reachable Details dialog. The pre-implementation run failed all seven against the old clipped/path-heavy presentation.
- Verification: the complete affected battery passed with 157 tests (`test_library_file_notes_workspace.py`, `test_non_obscuring_focus_contract.py`, and `test_css_build_integrity.py`); targeted Ruff, Python compilation, `git diff --check`, and self-review passed. Only pre-existing dependency, SQLite privacy, pytest-cache permission, and pytest-asyncio warnings were reported.
- Qodo follow-up: added Google-style `Args:` documentation to all three newly introduced async tests, satisfying compliance rule 497152 without changing runtime behavior or task scope.
- ADR required: no. The implementation conforms to ADR-011 and ADR-029 without changing disk authority, synchronization, ownership, or long-lived application structure.
