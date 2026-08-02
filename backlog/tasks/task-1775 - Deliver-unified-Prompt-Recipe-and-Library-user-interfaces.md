---
id: TASK-1775
title: 'Deliver unified Prompt, Recipe, and Library user interfaces'
status: In Progress
assignee: []
created_date: '2026-08-01 23:29'
updated_date: '2026-08-02 08:20'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 7: Added the stable Prompts entry to the existing Console composer hamburger menu, preserving temporary Save-this-chat precedence and all prior rows/reasons. Added one responsive ConsolePromptsModal Browse/Edit shell with pure source/navigation state, backend list/search/detail/save injection, 200 ms token-gated search, pagination, retry/unavailable/deleted-detail copy, Prompt/Recipe labels, conservative legacy editing, guarded foreign/malformed compatibility views, Recipe-to-unsaved-Prompt copies, capability-gated saves, focus restoration, and dirty-work protection. Reused the Task 6 incremental PromptBlockEditor without composer apply/model side effects. Verification: exact four-file UI suite 235 passed; focused Ruff and format clean; CSS regenerated; diff check clean. Impeccable detector reported one advisory for pre-existing #6f7782 at _agentic_terminal.tcss:3488, outside this task diff. TASK-1775 remains In Progress with ACs unchecked until Task 8 completes Library integration and legacy Recipe guards.

Task 8: Made Prompt and Recipe rows first-class in Library > Prompts with distinct type/source/lane summaries, shared incremental System/User block editing, read-only compiled previews, guarded compatibility conversion, capability/version-aware update, and conflict recovery through Reload or Save as new. Added immutable Outcome-first and Blank Recipe factories plus exact structured save sizing/limit gates. Recipe saves preserve block order, syntax, XML tags, mapping hints, and opt-in starter content without retargeting the active Prompt. Library use opens Recipes as unsaved Prompt copies, while picker, `/prompt`, `/system`, and usage-recording seams reject direct Recipe execution. Markdown export/import now preserves structured Recipe identity and definition; invalid structured exports stop instead of downgrading. Library System Apply is explicitly unavailable until Task 9, with a durable recovery reason, while User Apply remains available. Verification: required Task 8 five-file suite 165 passed; shared editor and PromptScope suites 77 passed; Task 7 modal/menu regression 70 passed; Ruff check/format and diff check clean; one-time Impeccable detector returned no findings; independent review approved with no remaining findings. ADR-040 remains governing. TASK-1775 intentionally remains In Progress with all ACs unchecked pending stage closeout authorization.
<!-- SECTION:NOTES:END -->
