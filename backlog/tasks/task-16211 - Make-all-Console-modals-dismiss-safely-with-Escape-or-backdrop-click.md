---
id: TASK-16211
title: Make all Console modals dismiss safely with Escape or backdrop click
status: In Progress
assignee: []
created_date: '2026-08-14 05:54'
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
- [ ] #1 Every `ModalScreen` reachable from Console can be safely cancelled with `Escape` and with a primary-button click on its backdrop; transient sub-surfaces and unsaved artifacts receive their required guard first.
- [ ] #2 Once no transient surface or discard guard claims the gesture, Escape and backdrop dismissal return the exact safe result and invoke the same cancellation callbacks as each modal's visible Cancel or Close control.
- [ ] #3 Clicks inside modal content, clicks on descendant overlays such as expanded `Select` options, non-primary clicks, and classifier inputs with unknown provenance do not dismiss the modal.
- [ ] #4 Dirty Prompt Workbench state retains its discard guard, nested modals dismiss only the top screen, and the required Console setup overlay remains non-dismissible.
- [ ] #5 Existing Composer Menu, RAG Settings, Settings, and Image Viewer click behavior remains functional without duplicate dismissal; Image Viewer retains intentional click-anywhere close.
- [ ] #6 Focused tests inventory the actual Console modal launch paths and cover Escape, backdrop, inside/overlay clicks, dirty-state and staged-artifact protection, safe return values, and nested modal behavior.
- [ ] #7 Console Settings close requests preserve immediate-reset recovery through the approved Undo and close / Keep reset and close / Return guard; active compaction offers Close anyway / Return without falsely claiming provider cancellation or avoiding billing.
- [ ] #8 Cancellation is single-shot and top-screen-only under repeated input or delayed callbacks, and every dismissal or cancelled guard restores its recorded focus destination.
- [ ] #9 Inventory coverage follows direct and nested modal launch paths transitively, including shared change-review and cancellation dialogs, while preserving existing non-dismissal click behavior.
<!-- AC:END -->

## ADR Check

ADR required: yes; amend the existing ADR

ADR path: `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`

Reason: the change introduces a reusable cross-module cancellation interface
and long-lived Console interaction grammar. ADR-031 is amended rather than
creating a duplicate decision.

## References

- `Docs/superpowers/specs/2026-08-13-task-16211-console-modal-dismissal-design.md`
- `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`
