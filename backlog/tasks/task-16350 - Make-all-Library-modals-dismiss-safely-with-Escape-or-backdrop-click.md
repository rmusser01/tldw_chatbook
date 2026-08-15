---
id: TASK-16350
title: Make all Library modals dismiss safely with Escape or backdrop click
status: In Progress
assignee: []
created_date: '2026-08-15 04:49'
labels:
  - library
  - ui
  - a11y
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make every modal transitively reachable from Library safely dismissible from the keyboard or backdrop without bypassing transient layers, mutation guards, destructive confirmations, trust boundaries, or typed cancellation semantics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every ModalScreen transitively reachable from Library can be safely cancelled with Escape and with a primary-button click on its backdrop, except an explicitly documented non-dismissible gate.
- [ ] #2 Escape gives a focused descendant overlay first refusal, then peels shared-picker path editing, search, and recent locations in that deterministic order before terminal cancellation; visible Cancel and backdrop remain terminal safe-cancel requests.
- [ ] #3 Terminal Escape, backdrop, and visible Cancel return each modal's exact safe negative result and never confirm, save, delete, install, authorize, trust, or discard by themselves.
- [ ] #4 Clicks inside modal content, descendant overlays, non-primary clicks, and inputs with unknown provenance do not dismiss a modal.
- [ ] #5 Prompt collection create and rename mutations cannot be bypassed by queued Escape, backdrop, or visible Cancel input while a mutation is active; Cancel remains enabled as a guarded request, the first rejected close updates the existing status line once, and later requests do not stack feedback.
- [ ] #6 Cancellation is single-shot, top-screen-only, mount-generation-safe, and restores only an eligible recorded opener or its single eligible stable-ID replacement.
- [ ] #7 Focused tests mount and behavior-test every concrete reachable modal class, exercise real Textual key and overlay dispatch, verify MRO handlers run once, and enforce an explicit bidirectional launch inventory across supported direct, controller-injected, nested-widget, and modal-to-modal owner edges.
- [ ] #8 Shared file picker changes preserve typed results and existing behavior for representative non-Library callers, including EnhancedFileOpen/EnhancedFileSave smart dismissal, handler suppression, persistence, and application import compatibility.
<!-- AC:END -->

## References

- Design: `Docs/superpowers/specs/2026-08-14-task-16350-library-modal-dismissal-design.md`
- ADR: `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`
