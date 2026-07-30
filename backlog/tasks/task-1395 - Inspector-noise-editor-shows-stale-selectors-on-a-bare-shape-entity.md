---
id: TASK-1395
title: Inspector noise editor shows stale selectors on a bare-shape entity
status: To Do
assignee: []
created_date: '2026-07-30 05:20'
labels:
  - watchlists
  - code-health
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`inspector_pane.py:227` (`_ignore_selectors_text`; the fallback read is at `:246`) falls back to a bare `entity["ignore_selectors"]`
key, but the screen's post-save patch (`watchlists_collections_screen.py:3328` region) updates only
the `settings` shape. On an entity carrying both shapes, clearing the field to empty then reopening
would re-display the stale bare-key value.

Unreachable today — `normalize_local_subscription_row` publishes the `settings` shape only — so this
is a latent trap, not a live bug. Found in TASK-1362's Task 6 review, deferred by design. The fix is
to make the reader and the patcher agree on one shape (drop the bare-key fallback, or patch both).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The editor's read path and the post-save patch use the same single shape
- [ ] #2 A test constructs the dual-shape entity and proves a cleared field stays cleared on re-render
<!-- AC:END -->
