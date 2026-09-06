---
id: TASK-31247
title: Add Character chat search controls and Roleplay Meaning
status: To Do
assignee: []
created_date: '2026-09-04 02:11'
labels:
  - settings
  - roleplay
  - rag
  - ux
dependencies:
  - TASK-31246
references:
  - >-
    Docs/superpowers/specs/2026-09-03-character-conversation-navigation-design.md
  - >-
    Docs/superpowers/plans/2026-09-03-character-conversation-navigation-implementation.md
priority: high
---

## Renumbering provenance

Renumbered from TASK-31239 on 2026-09-04 to keep the eight-task unshipped
sequence contiguous after its first six IDs collided with older worktree tasks.
All dependency and plan references moved with the sequence.

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give users explicit control over local character-chat semantic indexing in canonical Settings and ship the first end-to-end Meaning search in Roleplay without conflating staged preferences and immediate maintenance jobs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Settings exposes Character chat search under Settings > RAG with local-only privacy copy and one contextual primary action in a single 52x20 scroll owner.
- [ ] #2 Index existing chats and Keep future chats indexed are separate; model and future-index choices are staged and applied only by Save.
- [ ] #3 Index, rebuild, and delete use saved configuration and are disabled while relevant fields are dirty; Pause, Resume, and Cancel remain available for the running saved-config job.
- [ ] #4 Revert changes only staged fields; Delete cannot run against dirty fields and, on success, disables saved future indexing without deleting conversations.
- [ ] #5 Progress, absent, waiting, ready, paused, cancelled, failed, damaged, storage-full, and model-unavailable states expose truthful status and recovery.
- [ ] #6 Library RAG backfill remains a separate action and category contract.
- [ ] #7 Roleplay exposes focusable Keyword and Meaning strategies only when Meaning is ready and performs direct local semantic retrieval over the same eligibility corpus.
- [ ] #8 Targeted Settings commit-model, lifecycle, compact-layout, privacy, Roleplay search, focus, failure, and real-index tests pass.
<!-- AC:END -->
