---
id: task-2360
title: 'Realtime: reconnect drops mic audio instead of re-buffering'
status: To Do
assignee: []
created_date: '2026-08-04'
labels: [realtime, voice]
dependencies: []
priority: medium
---

## Description (the why)

The mic tap's pre-ready buffer guarantees first words spoken during the CONNECT handshake
are not lost, but `mark_ready()` is one-way: during a mid-loop RECONNECT the tap streams
into a session slot that is momentarily None and frames are dropped. The chip says
"reconnecting…" so it is not invisible, but the entry-time guarantee does not extend across
reconnects (V4 final review M2).

## Acceptance Criteria (the what)

- [ ] Speech during a reconnect window is buffered (bounded) and flushed to the new session
      once ready, mirroring the entry-time guarantee.
- [ ] A failed reconnect discards the buffer with the existing reasoned exit (no stale audio
      sent to a later session).
- [ ] Pinned by a wiring test driving frames during the reconnect window.
