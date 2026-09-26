---
id: TASK-18930
title: >-
  Console prompt stash browser: save, browse, and edit partial drafts against
  the Library prompt collection
status: Done
assignee:
  - '@codex'
created_date: '2026-08-19 09:55'
updated_date: '2026-09-25 20:55'
labels:
  - console
  - prompts
  - library
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Bring the useful outcome of hermes-agent's Ctrl+S prompt stash (2026-08-19 hermes-release review) into Chatbook without claiming that terminal-convention binding. Extend the existing Prompt Workbench with a local Draft Shelf beside Local and Server Prompts: stash the current partial draft without sending; browse with bounded search and truthful states; preview and edit shelf entries; promote a shelf entry into Library Prompts, optionally into a collection; and insert an entry at the current composer caret. The existing structured reusable-prompt flow continues to promote the live draft into a Library Prompt or Recipe. Keyboard access uses the command palette and the sibling-picker focus discipline (filter input keeps focus, synthetic highlight, Esc returns focus to the composer).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A **Save draft to shelf…** composer Menu action and Console command-palette action save the current canonical draft — including full text behind collapsed paste tokens — without sending; the user chooses Save and keep or Save and clear, and persistence failure never clears the composer
- [x] #2 The existing Prompt Workbench presents Draft Shelf beside Local and Server, with bounded paging/search, truthful loading/empty/no-match/error/full states, and stale-worker rejection
- [x] #3 From Prompt Workbench the user can insert a shelf entry at the current composer caret without overwriting existing text, edit a shelf entry in place, promote a shelf entry into a local Library Prompt with an optional existing collection, and hard-delete a shelf entry only after a two-press confirm; promotion never removes the shelf entry, while the existing structured flow continues to promote the live draft into a Library Prompt or Recipe
- [x] #4 Keyboard-only operation works end-to-end: filter input retains focus, arrow/Enter drive a synthetic highlight, Esc dismisses and returns focus to the Console composer — same discipline as the sibling Console pickers
- [x] #5 Draft Shelf storage is local-only, never synced/exported as Prompt records, and capped at 100 entries; the 101st save is blocked until the user explicitly deletes an entry and no automatic eviction occurs
- [x] #6 Tests cover the UI flows, PromptScopeService routing, bounded storage, focus discipline, and paste-token preservation; the user guide documents the workflow
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes.
ADR path: backlog/decisions/184-local-console-prompt-draft-shelf.md.
Reason: this adds durable local working-copy ownership and a local-only PromptScopeService contract that must remain outside Prompt sync/export/server behavior.

1. Add a non-synced LocalPromptDrafts table through the Prompts database v4-to-v5 migration, then expose LocalPromptService CRUD and normalized local-only PromptScopeService operations with optimistic versions and an atomic 100-entry refusal.
2. Add one shared Menu/command-palette Save flow that captures exact composer text and clears only after successful persistence.
3. Add Draft Shelf to the existing Prompt Workbench with focused search, bounded paging, plain-text editing, caret insertion, two-press deletion, and local Prompt/collection promotion.
4. Add targeted storage/service/widget/controller coverage, rebuild governed CSS, document the workflow, and complete live Console UAT.

Detailed plan: Docs/superpowers/plans/2026-09-25-console-prompt-draft-shelf.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added the Prompts database v4-to-v5 `LocalPromptDrafts` migration and local-only `PromptScopeService` CRUD with optimistic versions, literal bounded search, atomic 100-entry refusal, and no sync-log or server path. ADR-184 records the storage and service boundary.
- Added the Console Menu and command-palette save flow with exact canonical composer snapshots, explicit **Save and keep**/**Save and clear** choices, and clear-after-success protection. The Prompt Workbench now includes a Draft Shelf source with truthful states, bounded paging, stale-result rejection, plain-text editing, caret insertion, disarming two-press deletion, and retained-source Library promotion with an optional local collection.
- Addressed independent review findings for delayed source switches, deletion on a trailing page, bounded-search totals, editor Escape hierarchy, intervening delete actions, real collapsed-paste coverage, cached browse rows, and token-only widget styling. The task wording now makes the approved boundary explicit: collection selection belongs to shelf promotion; the pre-existing structured flow continues to promote the live draft.
- Addressed PR review with snapshot-consistent draft reads, shared exact-text Pydantic validation, one UI-neutral shelf-capacity contract, complete bounded collection pagination, blank-promotion refusal, and expanded public Draft Shelf API docstrings. The migration remains an inline Prompts DB step because `DB/migrations/README.md` explicitly assigns Prompts to the module-level migration convention; moving only v4-to-v5 would split that established runner.
- Verified with 35 focused Draft Shelf/runtime-policy tests, 56 Prompts schema/history tests, 8 design-token governance tests, Ruff checks, byte compilation, and `git diff --check`. Live Console UAT covered the Menu and command routes, keep behavior, source browsing, keyboard selection, caret insertion, two-press deletion, and narrow scrolling; promotion is covered by automated UI/service tests. A broader Prompts DB sweep passed 302 tests with the timing-dependent `test_concurrent_updates_to_same_prompt` deselected after confirming it failed 5/5 on the exact base commit; the incident is recorded in `backlog/docs/lessons-testing-evidence.md`. The legacy broad UI fixture remains blocked on the existing `RecoveryRequired("raw_source_selection_changed")` baseline, while its changed Menu ordering is covered by the focused Console path and live UAT.
- After rebasing the PR onto current `dev`, the review regression group passed 7 tests red-to-green, the full focused Draft Shelf/runtime-policy group passed 42 tests, and Prompts migration/history plus design-token governance passed 64 tests. Import/F821 checks and `git diff --check` also passed; a fresh bytecode-only rerun could not allocate its temporary cache because the host volume was full, while the executed test groups imported and exercised the changed modules successfully.
<!-- SECTION:NOTES:END -->
