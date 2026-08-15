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

## Implementation Plan

Detailed executable plan:
`Docs/superpowers/plans/2026-08-14-task-16350-library-modal-dismissal.md`.

1. Extend the shared safe-dismiss boundary to restore only an eligible opener or its single eligible stable-ID replacement.
2. Adopt the shared file-picker base, reconcile Textual full-MRO lifecycle/navigation dispatch, and preserve EnhancedFileOpen/EnhancedFileSave compatibility.
3. Add exact typed safe dismissal to ordinary Library skill, model, Prompt-delete, and Note-folder modals.
4. Add the same contract to File Notes and Git detail/trust/authorization surfaces without changing trust or push policy.
5. Move Prompt collection create/rename/retry callbacks to a screen worker, keep visible Cancel enabled as a guarded request, and reject stale same-instance remount completions.
6. Enforce a narrow bidirectional launch-edge inventory and independently mount every concrete reachable modal for visible Cancel, Escape, and backdrop behavior.
7. Run only the named touched/related test and static matrices, record mutation RED/GREEN evidence, complete review/documentation hygiene, and close the task.

ADR required: yes; amend the existing ADR.

ADR path: `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`.

Reason: Library adoption changes the established cross-module modal cancellation, shared file-picker, and focus-restoration contracts. ADR-031 already owns this interaction grammar and was amended by the approved design work.
