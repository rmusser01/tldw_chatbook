---
id: TASK-31248
title: Integrate Meaning into CtrlK and harden character navigation
status: To Do
assignee: []
created_date: '2026-09-04 02:11'
labels:
  - console
  - switcher
  - rag
  - integration
  - ux
dependencies:
  - TASK-31247
references:
  - >-
    Docs/superpowers/specs/2026-09-03-character-conversation-navigation-design.md
  - >-
    Docs/superpowers/plans/2026-09-03-character-conversation-navigation-implementation.md
priority: high
---

## Renumbering provenance

Renumbered from TASK-31240 on 2026-09-04 to keep the eight-task unshipped
sequence contiguous after its first six IDs collided with older worktree tasks.
All dependency and plan references moved with the sequence.

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete the programme by integrating opt-in local Meaning search into Ctrl+K, preserving painted-list stability, and qualifying the full Context-to-switcher-to-Roleplay-to-Console workflow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Character chats exposes focusable Keyword and Meaning strategies, and unavailable Meaning routes to Settings without enabling or downloading anything.
- [ ] #2 Keyword and vector retrieval start together; the 120 ms gate allows either valid leg to paint first without letting zero Keyword suppress healthy Meaning.
- [ ] #3 After any result list paints, late Meaning results never reorder it automatically and require Apply Meaning results while preserving stable selection.
- [ ] #4 Typed leg failures and empty states follow the complete presentation table and a losing leg cannot merge into or replace the winning generation.
- [ ] #5 Context remains Keyword-only and transfers its validated query into Character chats without changing result authority.
- [ ] #6 Cross-surface activation, profile-change, deletion, unavailable-character, repair, draft-veto, and index-lifecycle races preserve the prior trustworthy state.
- [ ] #7 The fixed 10k-conversation benchmark meets Keyword P95 <=300 ms and Meaning P95 <=2 s with the required manifest, model, hardware, and raw evidence.
- [ ] #8 Real-TUI, equal-cell terminal, first-use moderated, accessibility, documentation, and integrated 52x20 evidence satisfy the approved acceptance contract.
<!-- AC:END -->
