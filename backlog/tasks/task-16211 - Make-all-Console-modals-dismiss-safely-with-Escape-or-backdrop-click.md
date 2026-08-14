---
id: TASK-16211
title: Make all Console modals dismiss safely with Escape or backdrop click
status: Done
assignee: []
created_date: '2026-08-14 05:54'
updated_date: '2026-08-14 19:39'
labels:
  - console
  - ui
  - a11y
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make every modal reachable from the Console safely dismissible from the
keyboard or backdrop without bypassing cancel semantics, dirty-state guards,
nested controls, or required setup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every `ModalScreen` reachable from Console can be safely cancelled with `Escape` and with a primary-button click on its backdrop; transient sub-surfaces and unsaved artifacts receive their required guard first.
- [x] #2 Once no transient surface or discard guard claims the gesture, Escape and backdrop dismissal return the exact safe result and invoke the same cancellation callbacks as each modal's visible Cancel or Close control.
- [x] #3 Clicks inside modal content, clicks on descendant overlays such as expanded `Select` options, non-primary clicks, and classifier inputs with unknown provenance do not dismiss the modal.
- [x] #4 Dirty Prompt Workbench state retains its discard guard, nested modals dismiss only the top screen, and the required Console setup overlay remains non-dismissible.
- [x] #5 Existing Composer Menu, RAG Settings, Settings, and Image Viewer click behavior remains functional without duplicate dismissal; Image Viewer retains intentional click-anywhere close.
- [x] #6 Focused tests inventory the actual Console modal launch paths and cover Escape, backdrop, inside/overlay clicks, dirty-state and staged-artifact protection, safe return values, and nested modal behavior.
- [x] #7 Console Settings close requests preserve immediate-reset recovery through the approved Undo and close / Keep reset and close / Return guard; active compaction offers Close anyway / Return without falsely claiming provider cancellation or avoiding billing.
- [x] #8 Cancellation is single-shot and top-screen-only under repeated input or delayed callbacks, and every dismissal or cancelled guard restores its recorded focus destination.
- [x] #9 Inventory coverage follows direct and nested modal launch paths transitively, including shared open/save file pickers, change-review, and cancellation dialogs, while preserving existing non-dismissal click behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a framework-neutral SafeModalDismissMixin with pure backdrop classification, one-shot async cancellation, active-top-screen verification, and focus restoration.
2. Adopt the mixin across ordinary Console-owned modals and reconcile existing MRO click handlers without changing typed success results.
3. Adopt shared and transitively reachable modal components, including both enhanced file dialog variants, boolean confirmations, Video Player, and change-review confirmation.
4. Implement and test the Prompt Workbench, Console Settings, and generated-video capacity guard state machines.
5. Close the runtime/AST transitive inventory, run only related modal tests and targeted static checks, mutation-test safety branches, self-review, and complete Backlog documentation.

ADR required: yes; amend existing ADR
ADR path: backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md
Reason: the shared cross-module cancellation interface and long-lived modal grammar extend ADR-031.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a shared safe-modal dismissal boundary across all 27 Console modal screens and transitively reachable shared dialogs, with typed cancel results, dirty/reset/compaction/staged-video guards, single-shot top-screen dismissal, click-through shielding, and focus restoration. Task 8 added AST plus runtime inventory and fixed-point launch-edge coverage, real non-primary dispatch checks, and narrow Textual 8 MRO typing suppressions with no runtime change. Verification: exact 15-file matrix 614 passed; Ruff check, compileall, and diff check passed; Ruff-format matched the archived 733f03d2d baseline list exactly; MyPy improved from 26 baseline errors in 7 files to 24 errors in 6 files with no new or changed-line diagnostics; four required mutations each went RED then GREEN after restoration. ADR: backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md refinement for task-16211.
<!-- SECTION:NOTES:END -->
