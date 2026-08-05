---
id: TASK-1395
title: Inspector noise editor shows stale selectors on a bare-shape entity
status: Done
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
- [x] #1 The editor's read path and the post-save patch use the same single shape
- [x] #2 A test constructs the dual-shape entity and proves a cleared field stays cleared on re-render
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#1: dropped the bare-key fallback from `InspectorPane._ignore_selectors_text` (the reader), so it
reads ONLY the `settings["ignore_selectors"]` shape -- the single shape `normalize_local_subscription_row`
publishes AND the one `_patch_entity_ignore_selectors` (the post-save patch) writes. Reader and patcher
now agree. Verified the bare-key fallback was dead for real entities: every bare `entity["ignore_selectors"]`
reference in tests reads the DB COLUMN (`db.get_subscription(...)["ignore_selectors"]`) or the create-source
API, never a bare-key entity fed to the reader; the one direct reader test (`test_watchlists_inspector.py:1106`)
passes a normalized settings-shape entity.

AC#2: `test_cleared_selectors_stay_cleared_on_a_dual_shape_entity` constructs a dual-shape entity
(settings list + a stale bare key), confirms the settings shape wins when populated, then asserts the
post-clear state (settings entry popped, stale bare key still present) reads back "" -- not the stale
bare value. Mutation-verified: restoring the bare-key fallback reds it.

Files: `tldw_chatbook/UI/Watchlists_Modules/inspector_pane.py`, `Tests/UI/test_watchlists_inspector.py`.
<!-- SECTION:NOTES:END -->
