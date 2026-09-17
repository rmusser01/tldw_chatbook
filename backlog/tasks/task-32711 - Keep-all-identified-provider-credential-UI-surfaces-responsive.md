---
id: TASK-32711
title: Keep all identified provider credential UI surfaces responsive
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 01:39'
updated_date: '2026-09-17 03:28'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Slow subscription credential reads must not freeze canonical Settings, first-run onboarding or persona handoff readiness.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Canonical Settings remains responsive and displays credential completion or expiry without another user edit
- [x] #2 First-run provider setup stays responsive and can proceed after credentials resolve while preserving correct credential and commit semantics
- [x] #3 Persona handoff readiness remains responsive and automatically refreshes pending completion without stale selection updates
- [x] #4 Targeted mounted regressions cover slow reads completion and failure without contacting real credentials or providers
- [x] #5 Completing subscription setup preserves inactive stored keys and environment bindings unless the user explicitly clears or replaces them
- [x] #6 Home remains responsive during subscription reads and refreshes its model readiness after the worker completes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Reproduce blocking subscription reads on Settings, first-run provider setup and persona handoff using gated fake credential I/O; reuse the existing bounded background readiness snapshot for UI and refresh on completion/expiry; preserve actual request authentication and commit ownership; test stale selection/unmount and pending/error transitions. Final review additionally reproduced deletion of inactive API keys/env bindings during unchanged subscription setup: amend ADR-012 and implement a shared-owner sparse preservation path retaining explicit Clear/replacement and issued-mutation/CAS protections. Run targeted tests/lint and update docs. ADR required: yes (amend existing). ADR path: backlog/decisions/012-provider-credential-settings-boundary.md. Reason: clarify no-change credential semantics at the existing shared writer boundary; no new storage or credentials authority.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Canonical Settings, first-run Provider setup and persona handoff now use the existing bounded background credential snapshot. Mounted status updates cover completion, expiry and TTL refresh while guarding the current provider, draft and owning screen. Settings updates status rows without resetting unsaved inputs; first-run subscription recovery no longer incorrectly requires an API key. Persona polling tracks both rendered header and inspector states, fixing completion before the first poll and completion between the two renders. Actual send authentication remains unchanged and borrowed tokens never enter drafts/config.

Final review reproduced inactive API-key deletion during unchanged subscription setup. The shared provider writer now issues a narrowly validated sparse preservation mutation that leaves api_key, api_key_env_var and credential_source untouched. Explicit Clear/replacement, immutable issuance validation and atomic conflict checks remain authoritative. Authentication mode now participates in the conflict observation. ADR required: yes, amended backlog/decisions/012-provider-credential-settings-boundary.md before implementation; no new store or authority.

Files: UI/Screens/settings_screen.py, UI/Screens/personas_screen.py, UI/Persona_Modules/personas_preview_controller.py, UI/Wizards/FirstRunSetupWizard.py, UI/Wizards/first_run_setup_state.py, Chat/provider_setup_persistence.py; new Tests/UI/test_settings_subscription_readiness.py, Tests/UI/test_personas_subscription_readiness.py, Tests/Wizards/test_first_run_subscription_readiness.py and Tests/Wizards/test_first_run_subscription_preservation.py. Investigation report and plan updated; lessons-console-wiring records the observed polling races.

Validation (targeted runs only; overlapping counts):
- Final combined new UI/preservation tests, all three repaired failure cases and shared subscription controls: 73 passed in 29.83s.
- Provider setup writer, first-run state, subscription readiness and preservation files: 446 passed in 10.96s. Preservation tests include real config save/reload, unchanged fields, explicit edits, invalid preservation, unissued copies and concurrent auth/credential conflicts.
- Existing wizard provider/auth/credential selection: 45 passed, 349 deselected.
- Settings new mounted + provider-draft file: 83 passed; neighboring Overview/save/stale endpoint controls: 10 passed.
- Persona existing action/preview selection: 65 passed; final mounted and readiness/action selection after header-race repair: 16 passed.
- Six new/repaired test files pass Ruff check and format --check. Six production files have zero new Ruff diagnostics versus HEAD and zero findings on changed lines; changed source ranges formatted. Existing whole-file lint/format debt unchanged. Scoped git diff --check passes.
- Credential reads and provider effects are simulated; no real credentials/network, full suite, commits or publication. Existing RequestsDependencyWarning remains informational.

Self-review and independent review completed; TASK-32710 separately closes the three stale fixture/assertion failures.

Qodo PR2703 finding 3: investigate Home synchronous construction and threaded snapshot readiness. Add a mounted slow-credential completion regression, pass background credential policy explicitly at each execution boundary, and verify existing Home state tests. ADR required: no new ADR; restores the existing provider credential/UI boundary under ADR-012.

PR2703 Qodo finding 3 fixed: Home compose explicitly uses nonblocking credential snapshots, while the existing background content-snapshot thread performs resolved readiness and republishes the dashboard. The mounted valid-subscription regression failed with a stuck Model Blocked badge before the fix. Valid/missing/expired states now settle automatically while heartbeat assertions prove UI responsiveness. Home combined run: 8 passed, 64 unrelated cases deselected; five affected existing cases use the private-profile child harness with assertions unchanged. Root combined startup/context/token/Home run: 86 passed. No new Ruff findings; source ranges and new tests formatted. ADR-012 boundary unchanged.
<!-- SECTION:NOTES:END -->
