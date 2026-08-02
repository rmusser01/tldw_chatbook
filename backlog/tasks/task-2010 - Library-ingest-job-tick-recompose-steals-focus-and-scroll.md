---
id: TASK-2010
title: >-
  Library ingest job-tick recompose steals focus and scroll
status: To Do
assignee: []
created_date: '2026-08-02 21:30'
labels:
  - library
  - ingest
  - ux
  - uat
priority: high
dependencies: []
---

## Description (the why)

Every ingest job transition (`_handle_library_ingest_registry_changed`,
`library_screen.py:5537`) runs a full-screen `refresh(recompose=True)` while
the ingest canvas is selected. A recompose remounts every widget, silently
dropping focus from the Input the user is typing into; later keystrokes hit
the app's global digit bindings and navigate away mid-word (a typed-path
fragment was observed landing in a Console composer). Queue scroll position
also resets on every tick, so watching a folder batch means being yanked to
the top repeatedly. Found in the 2026-08-02 ingest UAT (critique snapshot
`.impeccable/critique/2026-08-02T21-04-04Z__chatbook-widgets-library-library-ingest-canvas-py.md`).

## Acceptance Criteria (the what)

- [ ] Typing into the ingest path (or title/author/keywords) field while a
      queued job transitions keeps focus, text, and cursor position.
- [ ] Queue scroll position survives a job transition.
- [ ] A focused widget that no longer exists after the recompose (e.g. a
      row-action button of a finished job) degrades gracefully: no
      exception, focus falls back to the screen default.
