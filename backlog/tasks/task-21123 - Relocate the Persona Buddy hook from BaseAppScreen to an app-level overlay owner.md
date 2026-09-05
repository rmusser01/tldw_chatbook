---
id: TASK-21123
title: >-
  Relocate the Persona Buddy hook from BaseAppScreen to an app-level overlay owner
status: Done
assignee:
  - '@codex'
created_date: '2026-08-22'
updated_date: '2026-09-04'
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

- [x] A single app-level overlay owner reacts to screen-change events and controller generation bumps; the per-screen recompose/mount/resume hooks and per-screen buddy state are removed
- [x] No widget module imported or Buddy worker started when the feature is disabled (import half shipped separately; worker half completed in this move)
- [x] Buddy behavior when enabled (placement, persistence, unavailable-fence) is unchanged - existing buddy tests green

## Implementation Plan

1. Preserve the accepted pet-only UI and existing modal, navigation, persistence, and unavailable behavior.
2. Move disposable view ownership into one app-owned coordinator. Subscribe to Textual screen changes and a generic screen-rebuilt message so recomposition cannot silently lose the view.
3. Deliver controller generation changes to the app through thread-safe, content-free messages; coalesce reconciliation and skip workers when disabled with no view to remove.
4. Replace per-screen Buddy state and methods with app/owner authority checks; retain the Workbench affordance refresh.
5. Run targeted Buddy lifecycle, widget, controller, Workbench, import-closure, and shutdown checks. No full test sweep.

ADR required: yes (existing decision amendment)
ADR path: backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
Reason: Clarify the app-owned presentation boundary and recompose-aware lifecycle without changing the controller/runtime or user experience.

Design: Docs/superpowers/specs/2026-09-04-task-21123-buddy-overlay-owner.md

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

### Historical deferral (resolved by the 2026-09-04 implementation)

Everything else: the app-level overlay owner, removing the per-screen recompose/mount/resume
hooks, and removing the per-screen state. An independent review of the relocation found it
would **break the enabled case** as specified: `super().recompose()` removes every child of the
screen, including a mounted Buddy, so an app-level owner that reacts only to screen-CHANGE
events would miss recomposes and the Buddy would silently vanish until the next screen switch.
Any relocation therefore needs a recompose-aware re-mount signal designed in first, and that
design is out of scope for a perf burn-down slice. Do not treat AC-2's import half being ticked
as licence to ship the move without it.

## Implementation Notes

Relocated disposable Buddy presentation to one lazy `PersonaBuddyOverlay` owned by
the app. Native screen-change signals, generic post-recompose messages, and
thread-safe content-free controller notifications feed one coalescing worker.
BaseAppScreen no longer owns Buddy fields, methods, or mount/resume workers.
The widget keeps its rendering, controls, timers, and persistence behavior, with
unavailable confirmation now injected from the app-owned boundary.

ADR-074 was amended (no new dependency, schema, or preference format). The disabled
path retains the fresh-process import guard and starts zero Buddy workers; removed
the old worker from the real-app boot census allowlist. Geometry drains before
controller shutdown. Review found and regression-tested two await-boundary races:
shutdown during retirement now prevents late mounts, and canceled retirement cannot
reuse a generation-invalidated view. Production-notification harness wiring and
explicit race gates are documented in `backlog/docs/lessons-testing-evidence.md`.

Verification includes the Buddy domain/widget/lifecycle suites, Workbench Buddy
cases, architecture and fresh-process import guards, real-app boot census, and
terminal-probe tests. The standalone POSIX PTY probe passed all 22 interactions,
including pet-only normal display, alerts, fold, constrained controls, mouse and
keyboard interactions, modal resume, navigation, and persisted geometry restore.
No full repository test sweep was run.

Final combined targeted run: **256 passed, 374 deselected**, 119.35 seconds,
with one inherited RequestsDependencyWarning. Command: shared-venv Python
`-m pytest -q Tests/Persona_Buddy Tests/UI/test_persona_buddy_widget.py Tests/UI/test_persona_buddy_app_mount.py Tests/UI/test_personas_workbench.py Tests/Utils/test_optional_import_deferral.py Tests/Architecture/test_persona_buddy_boundary.py Tests/Packaging/test_persona_buddy_import_closure.py Tests/Live/test_persona_buddy_terminal_probe.py Tests/Performance/test_boot_worker_census.py -k 'buddy or boot_worker' --basetemp=/private/tmp/task21123-verification-final`.
Independent lifecycle review confirmed both regression fixes with 2 passing tests
and reported no remaining blocking findings.

Static checks: modified-code Ruff passes with the 133 pre-existing app.py E402
findings excluded (the same 133 reproduce at HEAD); changed formatting, compilation,
and diff whitespace checks pass. All derived-artifact preflight checks pass.
The diagnostic inventory changed only for removal of the old screen-traversal
fallback's fixed `persona_buddy_geometry_flush_failed` log; reviewed the statement
delta before regenerating the pin, with no new diagnostics or persistent sinks.

Task status and criteria use the documented direct-file workaround for the broken
five-digit Backlog CLI, rather than generating a malformed TASK-TASK- record.
