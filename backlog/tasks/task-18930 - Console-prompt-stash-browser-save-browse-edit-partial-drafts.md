---
id: TASK-18930
title: 'Console prompt stash browser: save, browse, and edit partial drafts against the Library prompt collection'
status: To Do
assignee: []
created_date: '2026-08-19 09:55'
updated_date: '2026-08-19 09:55'
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
The full version of hermes-agent's Ctrl+S prompt stash (2026-08-19 hermes-release review), built on chatbook's existing Library Prompts + collections rather than a parallel store. A composer-adjacent browser modal, patterned on the on-disk Notes workbench UI (File Notes Library workbench, tasks 399.4/399.5): stash the current partial draft without sending; browse the existing prompt collection through the PromptScopeService browse contract shipped by TASK-198 (search, collection filter, bounded paging, truthful empty/error states); preview and edit stashed and existing entries; save the unsaved draft (or an edited stash) into Library Prompts, optionally into a collection; insert any entry into the composer. Keyboard-first with the sibling picker focus discipline (ConsoleSkillPickerModal / ConsoleStylePickerModal pattern: filter input keeps focus, synthetic highlight, Esc dismisses and returns focus to the composer).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A stash affordance (composer Menu action + shortcut) saves the current draft — including the full text behind collapsed paste tokens — into the stash list without sending; the user chooses whether the composer draft stays or clears (no silent data loss either way)
- [ ] #2 The browser modal presents stash entries and Library prompts/collections via the PromptScopeService browse contract with bounded paging, search, and the truthful loading/empty/error states TASK-198 AC5 established (stale-worker rejection included)
- [ ] #3 From the browser the user can: insert a stash or library entry into the composer, edit a stash entry in place, save a stash entry or the live draft into Library Prompts (optionally assigning a collection), and delete stash entries with a two-press confirm
- [ ] #4 Keyboard-only operation works end-to-end: filter input retains focus, arrow/Enter drive a synthetic highlight, Esc dismisses and returns focus to the Console composer — same discipline as the sibling Console pickers
- [ ] #5 Stash storage is local-only, never synced to the server, and bounded (pin the choice: entry cap with oldest-eviction, or explicit-manage-only — eviction must never delete the underlying Library prompts, only stash entries)
- [ ] #6 Tests cover the UI flows, PromptScopeService routing, bounded storage, focus discipline, and paste-token preservation; the user guide documents the workflow
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: UX over the existing Library Prompts service and TASK-198's collections; stash storage is local and bounded, prompts of record stay in the existing store (ADR-057 portable chatbook prompt records governs the records themselves).

1. Local stash store (bounded, JSONL or equivalent — no server sync)
2. Browser modal on the sibling-picker skeleton (filter-focus discipline, debounced rebuild)
3. PromptScopeService integration for browse/save/membership; insert-into-composer action
4. Tests + user guide page section
<!-- SECTION:PLAN:END -->
