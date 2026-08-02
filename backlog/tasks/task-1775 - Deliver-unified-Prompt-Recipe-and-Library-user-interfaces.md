---
id: TASK-1775
title: 'Deliver unified Prompt, Recipe, and Library user interfaces'
status: To Do
assignee: []
created_date: '2026-08-01 23:29'
updated_date: '2026-08-02 04:24'
labels: []
dependencies:
  - TASK-1774
references:
  - Docs/superpowers/plans/2026-08-01-console-prompt-improvement-workbench.md
  - Docs/superpowers/specs/2026-08-01-console-prompt-improvement-design.md
  - >-
    backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md
  - TASK-1680
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make saved Prompts and Recipes discoverable and editable in one keyboard-first Console workbench and the existing Library surface. This stage turns the foundation contract into an honest, source-aware interaction without changing live composer or session state as a side effect.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Prompts entry in the Console composer's existing hamburger menu opens one responsive, mode-driven modal that supports Browse, Edit, Improve, and Recipe navigation with focus restoration and dirty-work protections, without adding a top/tab-bar action or another always-visible composer button.
- [ ] #2 Browse and Library label Prompt versus Recipe, paginate empty libraries, use backend search for non-empty queries, and show unavailable, stale, malformed, and foreign artifacts honestly.
- [ ] #3 Shared block editing preserves TextArea cursor, selection, scroll, and undo state; Recipe selection creates an unsaved Prompt working copy and legacy use paths reject Recipes.
<!-- AC:END -->
