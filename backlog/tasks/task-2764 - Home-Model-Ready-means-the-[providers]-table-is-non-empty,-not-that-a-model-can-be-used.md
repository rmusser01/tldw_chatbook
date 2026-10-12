---
id: TASK-2764
title: 'Home ''Model: Ready'' means the [providers] table is non-empty, not that a model can be used'
status: Done
assignee: ['@t2-home']
created_date: '2026-08-06'
updated_date: '2026-10-11'
labels: [home, bug, honesty]
dependencies: []
---

## Description (the why)

`model_ready = bool(providers_models)` (`Home/active_work_adapter.py:175`),
where `providers_models` is the `[providers]` config table — which the
shipped default config populates with ~20 providers. So on a fresh install
with zero API keys, Home's header reads `Home | Ready · Local` and Details
reads `Model: Ready` / "Model ready" — while Console's own readiness
(`console_ready`, which does check real provider readiness) can be False.

Consequence: `choose_next_best_action` branch 1 (`Set up Console model` /
"Console needs a working model before live AI tasks.", route `settings`) —
the only Home path into Settings ▸ Providers & Models — can effectively
never fire, and a user believes model setup is complete until a send fails
in Console.

## Acceptance Criteria (the what)

- [x] `model_ready` derives from actual provider readiness (shared with, or
      equivalent to, the `console_ready` computation), not table presence.
- [x] On a key-less default config, Home shows Blocked (and the
      "Set up Console model" suggestion becomes reachable).
- [x] A test pins the key-less-config case.

## Implementation Plan

1. Premise-verify at origin/dev 9f8533686c whether the audit finding still
   holds after dev moved (audit was written at 84e4b33f0, 2026-08-06).
2. If already fixed upstream, verify all three ACs against the landed fix
   and its tests; close as fixed-by-duplicate with evidence.

## Implementation Notes

- VERDICT: fixed upstream by TASK-31805 (Done, 2026-09-06; PR #2461 +
  follow-up; see PR #2461 merge 11b3202b6c) — "Home 'Model: Ready' and Settings Overview
  overstate readiness with no credential present". Closed here as a
  duplicate with re-verification at origin/dev 9f8533686c; no code change
  needed in this task.
- AC1: `HomeScreen._build_dashboard_input`
  (tldw_chatbook/UI/Screens/home_screen.py) overrides the adapter's
  `model_ready` with `_home_console_provider_ready()` — the same send-path
  check Console enforces (`build_console_settings_readiness` →
  `get_provider_readiness`) — computed synchronously from in-memory config
  at first paint, then kept in lockstep with the async content snapshot's
  fresh-config `console_ready`. The adapter-level
  `model_ready=bool(providers_models)` remains only as the base default
  that Home unconditionally overrides before any rendering or
  next-action derivation; the two other `build_dashboard_input` consumers
  (`UI/Workflows_Modules/console_context.py`,
  `UI/Screens/scheduling/schedules_workbench.py`) read only
  `active_work_items` and never `model_ready`.
- AC2: on a key-less config Home's header now reads "Not set up" (the copy
  evolved from "Blocked" via TASK-34100.5's honesty pass — 'a skipped setup
  is not set up, not "Blocked"'), and the "Set up Console model"
  suggestion is reachable (asserted by test).
- AC3: `Tests/UI/test_home_screen.py::
  test_home_model_badge_reports_blocked_without_credential` pins the exact
  audit scenario — non-empty `[providers]` catalog, empty API key →
  "Model: Not set up" + `next_action.action_id == "fix_model_setup"` —
  with the paired
  `test_home_model_badge_reports_ready_with_credential` guarding against
  over-correction. Both re-verified green at 9f8533686c during this
  cluster's run (72 passed in test_home_screen.py).

ADR required: no
ADR path: N/A
Reason: no decision made in this task; upstream TASK-31805 already governed
the readiness-sharing change it landed.
