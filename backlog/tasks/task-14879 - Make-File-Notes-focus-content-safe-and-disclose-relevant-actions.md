---
id: TASK-14879
title: Make File Notes focus content-safe and disclose relevant actions
status: Done
assignee:
  - '@codex'
created_date: '2026-08-09 23:20'
updated_date: '2026-08-10 18:27'
labels:
  - notes
  - library
  - ux
  - accessibility
priority: high
references:
  - .impeccable/critique/2026-08-10T06-12-44Z__ok-widgets-library-library-file-notes-workspace-py.md
documentation:
  - backlog/decisions/011-chatbook-workbench-ui-system.md
  - backlog/decisions/029-file-notes-disk-authority.md
---

## Description

Quiet and distill the Library File Notes workbench so keyboard focus never obscures labels or tree entries and the editor exposes only actions relevant to its current state. Preserve every existing local-file operation, disk authority, guarded Session Git behavior, and compact-terminal workflow.

## Acceptance Criteria

- [x] Focused linked-root controls, navigator trees, path/search inputs, editor, and main editor action buttons preserve their labels and perimeter content while retaining a clear keyboard-focus treatment.
- [x] A linked root with no selected document shows New and Refresh; an active file adds its relevant file actions; Save Copy appears only for dirty, conflict, or error recovery; a selected tombstone exposes Restore without unrelated file actions.
- [x] Inapplicable actions are removed from the active choice set, and any visible action that is temporarily unavailable communicates a readable reason without leaving focus on a hidden control.
- [x] Editor action grouping remains legible at 160x45, 120x40, and compact widths; delete confirmation, navigator/editor switching, and Session Git toolbar quieting remain intact.
- [x] Focused mounted regressions, targeted Ruff, compile/diff checks, and production-CSS interaction verification pass with no storage, synchronization, Git-authority, or unrelated UI changes.

## Implementation Plan

ADR required: no
ADR path: N/A; conform to `backlog/decisions/011-chatbook-workbench-ui-system.md` and `backlog/decisions/029-file-notes-disk-authority.md`.
Reason: this is a scoped focus, hierarchy, and action-disclosure refinement within established File Notes ownership and workbench contracts; it introduces no durable architectural boundary or policy.

1. Add failing mounted regressions for content-safe focus and editor-action projection across empty, active, recovery, tombstone, transition, and compact states.
2. Apply a restrained File Notes focus treatment using the existing workbench focus tokens: background, readable foreground, bold underline, and no perimeter outline on one-row controls and trees; retain border-based focus for fields and the editor.
3. Project editor actions from the current document/recovery state, hide empty action rows, and safely redirect focus when a state change removes the focused control.
4. Verify the complete production interaction path at 160x45, 120x40, and compact widths, including delete/restore and Session Git toolbar quieting.
5. Run targeted static checks and self-review, then record evidence and close the task if every criterion passes.

## Implementation Notes

- Replaced the File Notes perimeter focus treatment with page-scoped, content-safe cues: buttons use the shared workbench background/foreground plus bold underline, Trees use the same focused cursor cue with no perimeter outline, and fields keep their stable focus border without the obscuring outline.
- Projected editor actions from retained state: New/Refresh form the baseline, active files reveal Move/Delete/Protect/Reload, Save Copy appears only for recovery states, and tombstones reveal Restore. Structural operations preserve the visible set, disable it with an explicit reason, and restore keyboard focus when an action disappears.
- Kept the retained compose tree and every file operation intact. Added targeted intrinsic relayout for Confirm delete and Protect/Unprotect so labels remain complete without forcing the compact grid when the distilled set already fits.
- Updated mounted regressions for production focus styles, empty/active/recovery/tombstone disclosure, focus repair, busy reasons, 160x45/120x40/64x28 geometry, and existing 40x20 Session Git/editor interactions.
- Qodo follow-up guards busy-start action focus restoration so it runs only when focus was actually lost or stayed on the original action, preserves an intentional focus move while work is in progress, and keeps the optional-MLX test shim behind a contiguous local import. The mounted regression covers focus movement into the editor across a busy transition and fails when unconditional restoration is reintroduced.
- Verification: `Tests/UI/test_library_file_notes_workspace.py` 41 passed; selected compact/Session Git regressions 3 passed; `test_non_obscuring_focus_contract.py` 100 passed; `test_css_build_integrity.py` 9 passed; targeted Ruff, Python compilation, CSS regeneration, and scoped `git diff --check` passed. Pytest reported only pre-existing configuration/deprecation/privacy warnings.
- ADR required: no. The change conforms to ADR-011 and ADR-029 and does not alter storage, synchronization, ownership, or guarded Git behavior.
