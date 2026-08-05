---
id: task-2364
title: 'Console message metadata field: engine provenance, interrupted flag, empty-row strand'
status: To Do
assignee: []
created_date: '2026-08-04'
labels: [console, realtime, store]
dependencies: []
priority: medium
---

## Description (the why)

`ConsoleChatMessage` has no metadata field, so the V4 spec's engine provenance
(engine/provider/model) and interruption marking ride a visible " ⏹ interrupted" content
marker plus usage-attach (documented deferral, spec Continuity section). Consequences: the
marker is fed back to the model on reseed only via a strip hack; exports/summaries
string-match UI copy; a legitimately-empty transcript strands an empty user row forever
with nothing recording why. A store-level metadata field closes all three.

## Acceptance Criteria (the what)

- [ ] Messages can carry structured metadata (engine, provider, model, interrupted,
      transcript-status) without content markers.
- [ ] The reseed builder and exports read the field instead of string-matching.
- [ ] The spec's Continuity deferral note is updated to point at the shipped field.
