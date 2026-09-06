---
id: TASK-31707
title: Repair trace settlement boundary regression evidence
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:50'
updated_date: '2026-09-05 19:27'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate oversized settlement and cold recovery failures against current trace contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproduced failure causes are addressed without weakening privacy or recovery guarantees;Affected complete files and scoped static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Checkpoint diagnosis: oversized ASCII response hits the existing one-million-codepoint credential sanitizer before the one-MiB response serializer; use real multibyte text to reach the byte boundary without relaxing either limit, and separately retain fail-closed sanitizer-budget evidence. Cold reserved call uses current SQLite created_at but historical recovery_at, so derive a valid stale timestamp from its actual reservation time while preserving the default grace and exact three-state recovery. No implementation yet. ADR required: no. ADR path: N/A. Reason: planned test-only boundary inputs; runtime privacy and recovery contracts unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Corrected UTF-8 byte-limit fixtures without changing either the one-million-codepoint sanitizer limit or one-MiB response cap; retained the original oversized ASCII input as separate fail-closed sanitizer-budget coverage. Cold restart recovery now derives its stale timestamps from the real SQLite reservation, preserving the default grace period, exact three-state outcomes and idempotence. Fresh resumed baseline reproduced three trace failures; complete corrected trace file:37 passed in11.95s, XML /private/tmp/tldw-review-trace-settlement-fixed.xml. Full-file Ruff lint and changed-region formatting pass; git diff --check passes. Self-review found no runtime or privacy-policy changes. ADR required:no; path:N/A; routine test-only boundary correction.
<!-- SECTION:NOTES:END -->
