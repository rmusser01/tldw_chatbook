---
id: TASK-2766
title: 'Decompose _open_console_prompts_modal (397 lines, 17 nested closures)'
status: To Do
assignee: []
created_date: '2026-08-07 06:41'
labels:
  - refactor
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 3 moved this method verbatim into ConsolePromptsController. Review judged its 15-of-18 dependency fan-out inherent rather than a wrong controller boundary, but identified the method itself as a class in disguise: a modal callback-bundle factory whose 17 nested closures each own one dependency. Decomposing it was deliberately kept out of a byte-fidelity wave.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The modal's callbacks are objects or methods with named dependencies rather than closures over a 397-line scope
- [ ] #2 ConsolePromptsController's constructor dependency count falls below 18
- [ ] #3 No behaviour change: the modal-open path's characterisation tests pass unchanged
<!-- AC:END -->
