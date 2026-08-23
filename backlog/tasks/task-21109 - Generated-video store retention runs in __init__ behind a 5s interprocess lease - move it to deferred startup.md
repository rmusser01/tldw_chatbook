---
id: TASK-21109
title: >-
  Generated-video store retention runs in __init__ behind a 5s interprocess lease - move it to deferred startup
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - startup
  - video-generation
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21109).

`_build_generated_video_store()` (app.py:5676 -> 5322-5333) runs `VideoStore.enforce_retention()`
during `TldwCli.__init__`: mkdir, lock-file create, an EXCLUSIVE interprocess portalocker lease
with a 5.0 s poll-timeout (video_store.py:56, 191-233), then a scan+delete pass - all pre-paint.
A held lease from a concurrent instance (a deliberate, supported usage pattern) blocks boot for
up to 5 s.

## Acceptance Criteria

- [ ] App construction only resolves paths; `enforce_retention()` runs from `_schedule_deferred_startup_work` after first paint
- [ ] A probe holding the lease from a second process shows boot is no longer blocked
- [ ] Retention semantics (including the session default) unchanged; existing video-store tests green
