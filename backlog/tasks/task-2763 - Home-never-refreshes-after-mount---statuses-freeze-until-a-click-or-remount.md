---
id: TASK-2763
title: 'Home never refreshes after mount — statuses freeze until a click or remount'
status: To Do
assignee: []
created_date: '2026-08-06'
labels: [home, bug, ui]
dependencies: []
---
## Description (the why)

`_sync_home_triage` has exactly three callers: the two on-mount workers
(`UI/Screens/home_screen.py:325, 364`) and the rail-row click handler
(`:719`). The screen has no `set_interval`, no timer, no reactive watcher,
and no event subscription — `@on(Button.Pressed)` is its only handler.

Live-verified (dev @ 84e4b33f0, 2026-08-06): with no clicks, the rail is
byte-identical after 6 s idle — twice the active-work adapter's own 3.0 s
cache TTL. A Library ingest job progressing `queued → parsing → writing →
done` while the user watches Home does not move on screen; the Running
count in the section header is equally frozen. The only refresh paths are
clicking a row or leaving and re-entering the screen (screens are never
reused).

Documented as a Quirk in `Docs/User_Guide/home.md` ("a snapshot with
buttons, not a live dashboard").

## Acceptance Criteria (the what)

- [ ] Home reflects active-work state changes without user interaction
      (poll, subscription, or message-driven re-sync — implementer's choice).
- [ ] The refresh path respects the adapter's cache TTL (no busy-loop).
- [ ] A test drives an ingest-job state change and asserts the rail updates
      without a click.
