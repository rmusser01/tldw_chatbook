---
id: TASK-31805
title: >-
  Home 'Model - Ready' and Settings Overview overstate readiness with no
  credential present
status: Done
assignee:
  - '@claude'
created_date: '2026-09-05 19:15'
updated_date: '2026-09-06 15:23'
labels:
  - bug
  - ux
  - settings
  - home
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-09-05 pre-release live UAT sweep (fresh scratch profile, dev tip 8e9d1128d4, real tmux-driven app). On a fresh profile with no API key anywhere, Home reports 'Model: Ready' and Settings Overview shows an OpenAI status implying usability; an actual send then fails with the key-required error. Readiness surfaces should reflect the same resolve_provider_api_key check the send path uses (see the CLAUDE.md configuration notes on readiness/spend agreement).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Home and Settings Overview report not-ready when no valid credential resolves for the selected provider.
- [x] #2 Readiness surfaces and the send path share one credential check.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce: fresh no-key profile -> Home 'Model: Ready' + Overview 'Status: OpenAI/model' while send fails on missing key.\n2. Root cause: Home badge derives from model_ready=bool(providers_models) (adapter); Settings Overview 'Status:' echoes _provider_readiness_label() identity. Neither calls get_provider_readiness.\n3. Fix Home: override model_ready in HomeScreen._build_dashboard_input via build_console_settings_readiness (get_provider_readiness) computed synchronously (allow_fresh_load=False) at first paint, refreshed by console_ready snapshot.\n4. Fix Settings: new _provider_overview_readiness_status() using get_provider_readiness; wire into Overview configuration line.\n5. TDD paired arms (no-credential->not ready; valid key->ready) for both surfaces; live tmux verify.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both readiness surfaces now derive from the SAME send-path check (get_provider_readiness via build_console_settings_readiness), matching what an actual send does.

Root cause (two weak checks):
- Home 'Model: Ready' badge, next-best action, and system summaries all key off HomeDashboardInput.model_ready, which the active-work adapter can only set to bool(providers_models) -- non-empty on any fresh profile whether or not a credential resolves.
- Settings Overview 'Status:' echoed _provider_readiness_label(), which returns provider/model IDENTITY only ('OpenAI / gpt-4o'), never a readiness verdict.

Fix:
- home_screen.py: HomeScreen._build_dashboard_input overrides model_ready from _home_console_provider_ready() (build_console_settings_readiness -> get_provider_readiness). Added allow_fresh_load param so the compose path computes readiness synchronously from in-memory config (honest, flicker-free first paint, no per-compose load_settings disk hit); the async content snapshot keeps the badge in lockstep with the fresh console_ready.
- settings_screen.py: new _provider_overview_readiness_status() wired into the Overview 'configuration' line. Left _provider_readiness_label() (the per-provider inspector identity rows) untouched.

Qodo review of PR #2461 (follow-up commit 78bb63cc99):
- FINDING 2 (real, fixed): _provider_overview_readiness_status() validated only the credential, so a resolvable key + no model still read 'Ready' while the identity showed 'not selected' and the send gateway blocks with 'Select a model before sending.' Now, after the credential check passes, a blank model (normalize_console_model_value(...) is None -- the send path's own predicate) reports 'Not ready: Select a model', in the send path's credential-then-model order. New arm test_settings_overview_status_reports_not_ready_without_model (RED pre-fix: 'OpenAI / not selected; Status: Ready').
- FINDING 1 (setup-flow test crash): FALSE POSITIVE. The 8 failing wizard/navigation tests fail with the IDENTICAL node-id set on clean origin/dev with both changed files reverted to base -- pre-existing and unrelated to this change.

Scope: strictly the no-credential-at-all / no-model overstatement for common keyed providers; the known google env-only gap is not touched.

Tests (paired RED/GREEN arms):
- Tests/UI/test_home_screen.py: blocked-without-credential + ready-with-credential.
- Tests/UI/test_settings_configuration_hub.py: not-ready-without-credential + ready-with-credential + not-ready-without-model.

Live tmux (no-key profile): Home 'Home | Blocked . Local' + 'Set up Console model'; Overview 'Status: Not ready: Missing API key'.

Files: tldw_chatbook/UI/Screens/home_screen.py, tldw_chatbook/UI/Screens/settings_screen.py, Tests/UI/test_home_screen.py, Tests/UI/test_settings_configuration_hub.py.
<!-- SECTION:NOTES:END -->
