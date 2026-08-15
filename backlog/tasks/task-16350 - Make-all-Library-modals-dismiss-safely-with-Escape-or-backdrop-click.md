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
- [ ] #2 Escape peels transient child surfaces such as path editing, search, recent lists, and Select overlays before requesting terminal cancellation; visible Cancel and backdrop remain terminal safe-cancel requests.
- [ ] #3 Terminal Escape, backdrop, and visible Cancel return each modal's exact safe negative result and never confirm, save, delete, install, authorize, trust, or discard by themselves.
- [ ] #4 Clicks inside modal content, descendant overlays, non-primary clicks, and inputs with unknown provenance do not dismiss a modal.
- [ ] #5 Prompt collection create and rename mutations cannot be bypassed by queued Escape, backdrop, or Cancel input while a mutation is active.
- [ ] #6 Cancellation is single-shot, top-screen-only, mount-generation-safe, and restores the recorded opener by stable identity when it still exists.
- [ ] #7 Focused tests mount every concrete reachable modal class, exercise real Textual key and overlay dispatch, verify MRO handlers run once, and enforce an exact fixed-point launch inventory including controller-injected and nested-widget paths.
- [ ] #8 Shared file picker changes retain existing behavior for representative non-Library callers and preserve all typed success results.
<!-- AC:END -->

## References

- Design: `Docs/superpowers/specs/2026-08-14-task-16350-library-modal-dismissal-design.md`
- ADR: `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`
