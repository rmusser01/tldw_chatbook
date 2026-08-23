---
id: TASK-21123
title: >-
  Relocate the Persona Buddy hook from BaseAppScreen to an app-level overlay owner
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - architecture
  - persona-buddy
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21123).

`UI/Navigation/base_app_screen.py` awaits `reconcile_persona_buddy_view()` at the tail of EVERY
screen recompose (~421 recompose=True sites repo-wide), and every mount AND screen-resume
schedules a reconcile worker - with the widget module imported before the enabled check - even
when the feature is disabled (the default). The disabled-case per-event cost is us-scale
(verified), but the design multiplies lifecycle work across every screen, duplicates five
methods + three fields of state per screen, and spends ~80 lines defending teardown races the
placement itself creates. The app already owns the authoritative entry point (app.py:8610), the
controller is app-owned, and the widget floats via overlay: screen.

## Acceptance Criteria

- [ ] A single app-level overlay owner reacts to screen-change events and controller generation bumps; the per-screen recompose/mount/resume hooks and per-screen buddy state are removed
- [ ] Short-term (or as part of the move): no worker is spawned and no widget module imported when the feature is disabled
- [ ] Buddy behavior when enabled (placement, persistence, unavailable-fence) is unchanged - existing buddy tests green
