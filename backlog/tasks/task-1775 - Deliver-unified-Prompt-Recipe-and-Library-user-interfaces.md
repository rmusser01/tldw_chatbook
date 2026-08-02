---
id: TASK-1775
title: 'Deliver unified Prompt, Recipe, and Library user interfaces'
status: In Progress
assignee: []
created_date: '2026-08-01 23:29'
updated_date: '2026-08-02 04:49'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Task 6: add immutable PromptBlockEditorState operations and a shared stacked System/User PromptBlockEditor that performs incremental control patches and mounted widget moves, with strict RED/GREEN tests for state, validation, application defaults, responsive layout, and TextArea identity/cursor/selection/scroll/undo preservation.
2. Task 7: add the existing composer-menu Prompts action and one responsive mode-driven ConsolePromptsModal Browse/Edit shell with source-aware search, focus restoration, dirty-work guards, and honest unsupported/error states.
3. Task 8: reuse the shared editor in Library > Prompts, add source/capability-aware Prompt and Recipe saves, introduce the built-in Outcome-first Recipe, and guard all legacy prompt execution, picker, apply, usage, and export paths from Recipes.

ADR required: yes
ADR path: backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md
Reason: this stage implements the adopted long-lived unified Console/Library editor and guarded artifact interaction structure.
<!-- SECTION:PLAN:END -->
