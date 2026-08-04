---
id: task-2211
title: Staged strip and tray render 'stale' reference status differently
status: To Do
assignee: []
labels:
  - console
  - rag
  - polish
dependencies: []
priority: low
---

## Description

The PR-4 staged-evidence strip maps every non-available EvidenceBundle reference status to "blocked", while the Inspector tray maps "stale" to "running" — two renderers of one bundle coloring the same status differently. The strip's own per-reference branch can reuse normalize_console_source_status (already in console_display_state; the strip's bundleless fallback branch already uses it).

## Acceptance Criteria

- [ ] Strip and tray render the same status vocabulary for the same reference
- [ ] Pinned by a test feeding a 'stale' reference through both renderers
