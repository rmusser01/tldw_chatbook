---
id: TASK-21123
title: >-
  Relocate the Persona Buddy hook from BaseAppScreen to an app-level overlay owner
status: To Do
assignee: []
created_date: '2026-08-22'
updated_date: '2026-08-23'
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
- [x] Short-term (or as part of the move): no widget module imported when the feature is disabled (shipped separately, see Progress note below); the worker half is still open
- [ ] Buddy behavior when enabled (placement, persistence, unavailable-fence) is unchanged - existing buddy tests green

## Progress note (2026-08-23) - the import half shipped separately

The AC-2 import half of this task shipped on its own, on the wave-7b branch
(`fix/task-21470-wave7b`), because it is a one-line move with no design risk.
`BaseAppScreen.reconcile_persona_buddy_view`'s
`from ...Widgets.Persona_Widgets.persona_buddy_widget import PersonaBuddyWidget` used to be
the FIRST statement of the method, executed before the enabled check. It now sits at the one
branch that constructs the widget. Pinned by
`Tests/Utils/test_optional_import_deferral.py::test_persona_buddy_reconcile_imports_nothing_while_disabled`,
which runs the real coroutine in a fresh interpreter and asserts `persona_buddy_widget`,
`Persona_Buddy.controller`, `Persona_Visual.runtime` and PIL are all absent from `sys.modules`
afterwards, for both the no-controller and the snapshot-says-disabled case. It fails against
the un-moved import.

Measured marginal cost of the import that is now skipped (fresh interpreter, screen module
already imported, then timing the buddy-widget import):

| route | PIL already resident | marginal import |
|---|---|---|
| `home_screen` | no | 27.0 ms, +39 modules (+10 PIL) |
| `settings_screen` | no | 24.6 ms, +31 modules (+10 PIL) |
| `chat_screen` | yes (16) | 14.7 ms, +10 modules |
| `library_screen` | yes (16) | 16.0 ms, +14 modules |

(one-time per process, on the event loop, right after first paint). Note this is ~25 ms, not
the ~1.28 s an earlier task recorded for the cold chain -- that figure did not reproduce on a
warm filesystem here.

### What remains open, and why the relocation is deferred

Everything else: the app-level overlay owner, removing the per-screen recompose/mount/resume
hooks, and removing the per-screen state. An independent review of the relocation found it
would **break the enabled case** as specified: `super().recompose()` removes every child of the
screen, including a mounted Buddy, so an app-level owner that reacts only to screen-CHANGE
events would miss recomposes and the Buddy would silently vanish until the next screen switch.
Any relocation therefore needs a recompose-aware re-mount signal designed in first, and that
design is out of scope for a perf burn-down slice. Do not treat AC-2's import half being ticked
as licence to ship the move without it.

