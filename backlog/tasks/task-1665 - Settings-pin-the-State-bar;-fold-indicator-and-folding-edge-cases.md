---
id: TASK-1665
title: 'Settings: pin the State bar; fold-indicator and folding edge cases'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-31'
labels:
  - settings
  - ux
  - critique-r4
dependencies: []
priority: medium
---

## Description (the why)

Critique round 4 P2 edges on the round-3 batch: RAG showed no State bar at all (each category composed its own inside the scrollable content, so the persistence badge scrolled away mid-task); Overview's inspector cut mid-sentence with no fold indicator (the sync-rows recompose re-mints a hidden hint after the mount-time evaluation); the in-place guidance refresh bypassed _fold_long_tokens ('console.paste_collapse_thresh/old').

## Acceptance Criteria (the what)

- [x] One State banner, pinned between the pane title and the scrollable body, on every category
- [x] The fold indicator is re-evaluated after the sync-rows recompose
- [x] Every in-place guidance refresh path folds dotted keys (console, appearance, storage, rag, provider)

## Implementation Notes

Detail pane restructured to fixed-header-over-scroll-body (#settings-detail-pane-body), mirroring the impact pane; 15 per-category banner yields removed in favor of one composed from active_summary. Test pins repointed from #settings-detail-pane to the -body child.
