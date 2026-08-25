---
id: TASK-22217
title: >-
  Keep PIL off the warm boot path: lazy import in visual_identity
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - performance
  - startup
  - personas
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22217).

Traced live this review: every boot, `app.py:8784 _init_notes_service` ->
`get_chachanotes_db_lazy()` -> `seed_builtin_content()` (`config.py:7231`) ->
`Character_Chat/visual_identity.py:24` module-level `from PIL import Image, ...`.
`ensure_builtin_samira` preflights and exits early on warm boots — but the PIL import
(~80 modules) is paid before the preflight can run, on the init thread pool, every boot,
in every profile. This undermines TASK-21103/21200 (which keep PIL out of the import
closure) via the construct-time gap the guards cannot see; PIL was confirmed present at
`_ui_ready` on tip.

## Acceptance Criteria

- [ ] A warm boot with seeding already terminal loads no PIL (census `sys.modules` at `_ui_ready` under a default profile)
- [ ] Fresh-profile seeding still works end to end (Samira card + pack created)
- [ ] PIL imports move inside the code paths that actually do image work in `visual_identity.py`
