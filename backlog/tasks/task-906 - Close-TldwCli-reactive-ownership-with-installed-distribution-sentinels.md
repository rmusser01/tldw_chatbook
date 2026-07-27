---
id: TASK-906
title: Close TldwCli reactive ownership with installed-distribution sentinels
status: To Do
assignee: []
created_date: '2026-07-26 23:50'
updated_date: '2026-07-27 00:02'
labels:
  - architecture
  - state
  - packaging
  - verification
dependencies:
  - TASK-647
  - TASK-648
  - TASK-649
  - TASK-650
  - TASK-651
  - TASK-652
  - TASK-904
  - TASK-905
references:
  - backlog/decisions/032-immutable-installed-distribution-assets.md
  - backlog/decisions/033-application-session-state-ownership.md
  - >-
    Docs/superpowers/specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enforce the exact remaining root reactive contract and prove the decomposed production application from a clean installed artifact.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 TldwCli retains exactly current_tab and splash_screen_active from the reviewed 61-descriptor inventory, with no source or dynamic access to any of the 59 removed names.
- [ ] #2 Every relevant registered production route executes without access to a removed root owner.
- [ ] #3 Focused source ownership, static, compile, Ruff, formatting, and diff hygiene checks pass.
- [ ] #4 A wheel and sdist built from the repository install into a clean environment outside the checkout, import only from the installed artifact, and pass resource, product-maturity, and reactive-ownership sentinels.
- [ ] #5 The authorized integrated suite passes without collecting surrogate-application tests, and any excluded legacy collections are explicitly documented.
<!-- AC:END -->
