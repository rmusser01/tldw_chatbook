---
id: TASK-16256
title: Restore Library media render identity
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 15:35'
updated_date: '2026-08-14 15:35'
labels: []
dependencies: []
---

## Description

Preserve the Library media viewer's compose-once contract after viewer construction moved into the active-child builder.

## Acceptance Criteria

- [x] The active-child builder records the exact media-detail object rendered by the viewer.
- [x] A detail arrival already represented on screen does not trigger a second Markdown parse.
- [x] The compose-once media viewer regressions and static checks pass.

## Implementation Plan

ADR required: no
ADR path: N/A
Reason: This restores an existing performance/lifecycle invariant lost during a mechanical extraction.

1. Preserve the two deterministic media-viewer RED regressions.
2. Restore the render-identity assignment at the extracted builder boundary.
3. Run the focused regressions, nearby media-viewer tests, lint, formatter characterization, and diff hygiene.

## Implementation Notes

Restored the exact render-identity assignment at the active-child builder introduced by the entry-worker refactor. This preserves the existing deferred arrival guard and prevents a duplicate parse of large Markdown media. The two deterministic regressions and all 20 media-viewer shell tests pass; Ruff and diff hygiene pass. Both large touched files are formatter-red at HEAD, so no unrelated formatting churn was introduced.
