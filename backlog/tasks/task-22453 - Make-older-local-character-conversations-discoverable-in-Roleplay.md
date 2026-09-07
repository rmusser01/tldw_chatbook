---
id: TASK-22453
title: Make older local character conversations discoverable in Roleplay
status: Done
assignee:
  - '@codex'
created_date: '2026-08-26 07:29'
labels:
  - roleplay
  - ux
  - conversations
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Roleplay currently shows only the 20 most recent saved conversations for a selected local character. Users with longer-running character histories need to find and open older conversations from the same Roleplay surface instead of detouring through Library.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A user can browse beyond the initial 20 saved conversations for the selected local character without leaving Roleplay.
- [x] #2 Conversation ordering remains stable and no unchanged conversation is skipped or repeated while loading additional results; creations and ordering-key modifications made after browsing begins take effect when the character is reselected, while deletions may be reflected by the next read.
- [x] #3 Every discovered conversation can be previewed and offers the same Resume chat, Send to Console draft, and Open in Library actions.
- [x] #4 Loading, empty, exhausted, and retryable failure states are explicit and keyboard accessible.
<!-- AC:END -->

## Implementation Plan

1. Add deterministic `(last_modified, id)` seek pagination to the existing local character-conversation DB query, preserving legacy offset callers.
2. Add presentation-only conversation tail states and append behavior to the Roleplay inspector.
3. Orchestrate 21-record sentinel reads, retry, deduplication, and stale-result ownership in the conversations controller and screen.
4. Verify database, keyboard, focus, layout, preview-action parity, and isolated live behavior with targeted checks.

ADR required: no
ADR path: N/A
Reason: This is a routine extension of the existing local Roleplay discovery query and inspector list; it does not change storage, ownership, synchronization, security, or service boundaries.

Detailed plan: [2026-08-27-task-22453-older-roleplay-conversations-implementation.md](../../Docs/superpowers/plans/2026-08-27-task-22453-older-roleplay-conversations-implementation.md)

## Implementation Notes

- Added stable local seek pagination ordered by `(last_modified, id)` while preserving legacy offset callers and normalizing mixed SQLite timestamp formats.
- Added a 20-row Roleplay browse controller with a 21st-row sentinel, exact cursor/attempt ownership, deduplication, stale-result suppression, and atomic retry recovery when DB or inspector rendering fails. Raw seek boundaries now advance across duplicate-shadow pages so ordering-key changes cannot hide unchanged older rows; a four-hop per-attempt budget yields back to an actionable Load tail if duplicate-only boundaries keep moving.
- Added append-only inspector rows with keyboard-accessible loading, retry, load-more, empty, and exhausted tails; every appended row retains the existing read-only preview plus Resume chat, Send to Console draft, and Open in Library actions.
- Regenerated the consolidated widget CSS from the inspector source and verified bundle synchronization. Review hardening added typed cursor validation, transaction-scoped reads with deterministic cursor cleanup, structured retry context, complete inspector API docstrings, and one shared page-size constant. The six new controller diagnostics were reviewed before refreshing the production-diagnostic inventory: they expose only local IDs, seek cursors, fixed phase/operation values, and no message content, secrets, paths, or URLs. Targeted verification passed: 16 DB tests, 78 inspector tests, 378 workbench tests (472 total), Ruff across the affected Python/test files, CSS artifact synchronization, and working-tree/branch diff integrity.
- Isolated live TUI acceptance loaded 45 scratch conversations as 20 + 20 + 5, reached **All conversations shown.**, and opened the oldest preview with the full action hierarchy. Real config/data and tracked CSS retained their baseline hashes; the scratch app exited cleanly.

ADR required: no
ADR path: N/A
Reason: This extends the existing local Roleplay query and inspector list without changing storage, ownership, synchronization, security, or service boundaries.
