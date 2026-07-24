---
id: TASK-547
title: 'Console /rewind: clear the stale persisted summary when its boundary is permanently gone'
status: To Do
assignee: []
created_date: '2026-07-24'
labels:
  - console
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`/rewind`'s boundary summary lives in local-only conversation columns (`context_summary` + `summary_boundary_message_id`, PR #844). On resume, `_resolve_context_summary_on_resume` correctly leaves the in-memory state unset when the persisted boundary id no longer maps to a loaded message (fail-open to full history) — but it never clears the DB row, so a stale `(summary, orphaned-boundary)` pair can persist indefinitely (e.g. after the boundary message's branch is hard-deleted, or a foreign client rewrites history). Benign (the payload path validates presence and fails open; the next summarize overwrites it), but the stale row is misleading to anything that reads the column directly and wastes a dangling reference. Fix: when resume detects a dangling boundary, best-effort clear the DB pair via `set_conversation_context_summary(conv_id, None, None)` (same guarded, non-fatal write-through pattern), with a test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Resuming a conversation whose persisted summary boundary maps to no loaded message clears the persisted summary pair (best-effort, non-fatal)
- [ ] #2 A valid boundary continues to restore the in-memory state unchanged
- [ ] #3 Covered by a real-DB resume test
<!-- AC:END -->
