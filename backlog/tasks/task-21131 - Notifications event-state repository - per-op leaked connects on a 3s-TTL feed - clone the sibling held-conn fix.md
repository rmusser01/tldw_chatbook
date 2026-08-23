---
id: TASK-21131
title: >-
  Notifications event-state repository - per-op leaked connects on a 3s-TTL feed - clone the sibling held-conn fix
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - notifications
  - database
priority: low
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21131).

`Notifications/event_state_repository.py:85-106` opens per-op (GC-leaked `with conn:`; no FK on
the file branch), 3+ opens per `build_server_notification_feed` call on the Home screen's 3 s
TTL cache in server mode. The sibling `client_notifications_db.py:69-108` is already the held
thread-local template with a liveness ping - clone it.

## Acceptance Criteria

- [ ] The repository holds a thread-local connection (template shape) with explicit close; FK enabled consistently
- [ ] Server-mode feed behavior unchanged
