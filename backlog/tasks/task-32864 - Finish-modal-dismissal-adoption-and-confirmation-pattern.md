---
id: TASK-32864
title: Finish modal dismissal adoption and extend the confirmation pattern
status: To Do
assignee: []
created_date: '2026-09-19 08:24'
labels:
  - core-review
  - review-cascade
dependencies: []
parent_task_id: TASK-32850
references:
  - qa/cascade-review-2026-09-19/report.md
  - backlog/decisions/161-component-pattern-library.md
  - backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
130 ModalScreen subclasses exist across 101 files (~44k LOC family; 36 verbatim `_cancel`/`_close` handlers in the census), but the universal declarative base (~8-15k deletable LOC) is EXPLICITLY REJECTED by ADR-161: "Form factor is a CSS-class catalog, not Python builders… two prior builder libraries died of disuse (0 and 2 importers)." This task is the ADR-161-compliant slice only — do not implement a universal modal base:

1. Finish `SafeModalDismissMixin` adoption: 69 of the 130 classes still extend bare `ModalScreen` (ADR-031 mandates the dismissal contract for Console/Library-reachable modals; e.g. `console_save_markdown_modal.py` subclasses the mixin but declares no Escape binding and no `SAFE_MODAL_CONTENT`).
2. Extend the `ConfirmationDialog` declarative pattern (documented pattern class per ADR-161) to the simple prompt/picker/confirm family — starting with the near-verbatim siblings `Widgets/cancel_confirmation_dialog.py` (103 LOC) and `Widgets/delete_confirmation_dialog.py`, which re-roll `ConfirmationDialog`'s shape with different accents.

Out of scope: the ~5 large feature modals (console_settings_modal, console_session_switcher, workspace switcher body, non-dismissible setup gate, vendored fspicker) and any Python builder revival. The modal dismissal inventory tests (`Tests/UI/test_console_modal_dismissal.py`, `test_library_modal_dismissal.py`) are the regression net. ADR required: no — executes ADR-161/031 as written.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The bare `ModalScreen` classes in Console/Library-reachable surfaces adopt `SafeModalDismissMixin` with bindings and `SAFE_MODAL_CONTENT`, or each exception is recorded with its reason
- [ ] #2 The simple confirm siblings collapse into the `ConfirmationDialog` pattern; no new Python builder base is introduced
- [ ] #3 Modal dismissal inventory tests stay green across the adoption
- [ ] #4 The task notes record that the universal-base cascade remains rejected per ADR-161 (pointers to the census evidence, so it is not re-derived)
<!-- AC:END -->
