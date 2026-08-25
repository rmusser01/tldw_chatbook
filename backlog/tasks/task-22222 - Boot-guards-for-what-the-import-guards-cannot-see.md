---
id: TASK-22222
title: >-
  Boot guards for what the import guards cannot see
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - testing
  - startup
  - guard-efficacy
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22222).

This review measured a real user-visible boot regression (+~11% to `_ui_ready`) with every
import guard green, because all of them assert on `import tldw_chatbook.app`:
- no census at `_ui_ready` (the mount leg grew invisibly; PIL present at ready);
- no budget on boot-parsed CSS bytes (770,285 -> 813,605 B since the pin; the TASK-21115
  ratchet design FORCES new widget CSS into the eagerly-parsed bundle with no size budget);
- no census of boot-time worker threads (4 -> 7 unnoticed);
- construct-time runtime imports are invisible (`app.py:7273-7274` re-imports
  `Persona_Visual.*` at construct — harmless today, boundary crossed silently);
- `Tests/App/test_boot_no_feature_db_files.py` is a fixed six-filename list (a seventh
  store, or non-DB side effects like 22216's staging sweep, pass silently);
- no wall-clock or structural TTI regression tripwire at all.
Also: TieAwareStylesheet (`app.py:5893`, new since pin) arms full ~814 KB reparses during
first mount when tie-breakers lower — instrument the count so the cost is visible.

## Acceptance Criteria

- [ ] A `sys.modules` census at `_ui_ready` is pinned (allowlist + budget), so mount-leg growth and construct-time imports land in review
- [ ] Boot-parsed CSS bytes carry a budget with a stated raise procedure
- [ ] Boot worker census pinned (see TASK-22215)
- [ ] The boot-files guard's fixed-list blind spot is documented in the test and extended where cheap
- [ ] TieAwareStylesheet reparse count during a cold boot is measured once and recorded (instrumentation may be temporary); each new guard documents its own blind spots
