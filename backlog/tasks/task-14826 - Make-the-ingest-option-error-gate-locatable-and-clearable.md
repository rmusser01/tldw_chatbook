---
id: TASK-14826
title: >-
  Make the ingest option-error gate locatable and clearable
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-10 21:00'
labels:
  - library
  - ingest
  - ux
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P2 of the 2026-08-10 re-critique. An invalid option value blocks Start with a message the user cannot act on, and the block outlives the selection that caused it.

Observed live: setting `#opt-generic-chunk_size` to `7` produces `Fix the highlighted options to start: Chunk size must be between 100 and 5000.` The message names no group; collapsing the panel highlights nothing (the `-ingest-option-invalid` class is applied to the Input, which is inside the collapsed body); the collapsed title shows `Chunk size: 7` with no error marker. Pressing Clear resets the path and restores the intro but LEAVES the block in place; leaving Ingest and re-entering still blocks. The value is not persisted to config, so the block dies on restart — which also makes it irreproducible in a bug report.

The repo already has the better pattern: Settings' field-level search lands focus ON the offending field.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The gate message names the group and field that is invalid
- [ ] #2 A collapsed panel containing an invalid value is visibly marked as such
- [ ] #3 The gate offers a way to reach the offending field (focus lands on it), rather than only describing it
- [ ] #4 Clearing the staged selection does not leave an unreachable block behind
<!-- AC:END -->
