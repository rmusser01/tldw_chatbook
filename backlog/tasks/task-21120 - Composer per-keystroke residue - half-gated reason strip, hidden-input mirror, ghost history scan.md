---
id: TASK-21120
title: >-
  Composer per-keystroke residue - half-gated reason strip, hidden-input mirror, ghost history scan
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - console
  - composer
priority: low
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21120).

Per printable key in `console_composer_bar.py`: `_sync_send_disabled_reason` calls
`strip.update(Content(reason))` unconditionally (:1588/:1596) - the computed `reason_changed`
gate covers only the ARIA announcement (the audit's known half-gate pattern); the hidden
compatibility `Input.value` is re-set with the full canonical draft (O(draft), firing a second
Changed handler, :1532-1539); ghost text runs a reverse linear scan of prompt history per draft
render AND per 0.5 s blink tick (:4214-4240).

## Acceptance Criteria

- [ ] The reason strip updates only when reason_changed; the hidden-input mirror skips unchanged text; the ghost-text history scan is capped or cached
- [ ] Composer behavior (send gating, ARIA announcements, ghost suggestions) unchanged - existing composer tests green
