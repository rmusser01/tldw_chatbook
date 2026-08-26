---
id: TASK-16211
title: Make all Console modals dismiss safely with Escape or backdrop click
status: Done
assignee: []
created_date: '2026-08-14 05:54'
updated_date: '2026-08-14 20:46'
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
Implemented a shared safe-modal dismissal boundary across all 28 Console modal screens on the rebased `dev` and transitively reachable shared dialogs, with typed cancel results, dirty/reset/compaction/staged-video guards, single-shot top-screen dismissal, click-through shielding, and focus restoration. The rebase added upstream `AutoSpeakConsentModal` to the explicit contract with exact `False` cancellation for its visible Cancel, Escape, and backdrop paths. Final review added a mount-scoped Prompt Workbench apply latch, responsive worker-based review/editor commits, truthful applying focus/status, top-screen-safe applied results, stale-unmount protection, and once-per-lifecycle MRO coverage. Cumulative review added generation checks immediately after async review validation and before every claimed apply worker can mutate or commit, proven by cancellation-resistant success/failure remount cases. Launch inventory resolves actual constructed runtime modal classes from AST imports, aliases, attributes, and same-module definitions, asserts exact equality at every fixed-point edge, and scans every reachable owner's defining class body in addition to any declared helper/controller sources; injected extra, rowless-owner, and declared-helper-plus-owner-body mutations prove omissions fail. Counts are exactly 28 Console and 37 reachable modal types. Verification before rebase: exact 15-file branch matrix 630 passed. Rebase verification: the full related matrix first exposed only the new upstream modal contract (637 passed, one inventory failure), then the final 16-file matrix passed 684 tests after adoption. Ruff lint, targeted MyPy, compileall, and diff checks passed; the upstream auto-speak module's existing whole-file formatter debt was unchanged outside the new conforming hunks. ADR: backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md refinement for task-16211.
<!-- SECTION:NOTES:END -->
