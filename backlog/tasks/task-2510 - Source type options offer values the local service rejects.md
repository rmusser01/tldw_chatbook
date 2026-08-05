---
id: TASK-2510
title: Source type options offer values the local service rejects
status: To Do
assignee: []
created_date: '2026-08-05'
labels:
  - watchlists
  - bug
dependencies: []
priority: medium
---

## Description (the why)

The create-source form's type Select (`_TYPE_OPTIONS`) offers `Feed`,
`Playlist` and `Channel`, none of which `_local_type_for_source_type`
accepts — it raises `ValueError`, so choosing one fails the create with a
generic "Failed to create source." toast that names neither the cause nor the
offending choice. Verified pre-existing on `origin/dev` during UAT batch-3
review.

Two changes in batch 3 (PR #1355) made it **more reachable**, without
introducing it: the type control now carries a visible label (so users find
and use it), and the chosen type persists across a form rebuild instead of
being silently reset to `rss` by the next recompose.

## Acceptance Criteria (the what)

- [ ] The type Select offers only values the active backend can actually
      create, or the unsupported ones are visibly unavailable with a reason.
- [ ] If an unsupported type does reach the service, the failure names the
      type and what to choose instead — never a bare "Failed to create
      source."
- [ ] A test pins the option list against the service's accepted vocabulary,
      so the two cannot drift apart again.
