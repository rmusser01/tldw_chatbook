---
id: TASK-22452
title: Make Anthropic cost-chip tests persist readiness config
status: Done
assignee: []
created_date: '2026-08-26 05:03'
updated_date: '2026-08-26 05:08'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the Anthropic cost-chip UI tests hermetic while exercising the production readiness refresh path. The shared disk-shaped test app reloads its sandboxed TOML, so the fake gateway credentials must be persisted there instead of existing only in a stale in-memory snapshot.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Anthropic cost-chip sends reach the fake gateway and render the expected cost states.
- [x] #2 The test helper writes provider selection and fake credentials only to the per-test sandbox through the production config API.
- [x] #3 Production provider-readiness behavior is unchanged and focused readiness, inspector, and capture regression tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the Anthropic cost-chip failure against the disk-shaped shared test app.
2. Persist the fake Anthropic selection and API key atomically through the production config API while retaining the mounted session selection.
3. Run the cost-chip, provider-readiness, inspector, and exchange-capture regression suites.

ADR required: no
ADR path: N/A
Reason: This is a hermetic test-harness repair that does not alter storage, runtime boundaries, contracts, security policy, or production behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the Anthropic cost-chip test helper to batch-persist its model selection and fake API key through save_settings_to_cli_config before mounting the Console, while keeping the app snapshot aligned for session creation. Production readiness logic was not changed. Removed one pre-existing unused import in the touched test module so focused lint passes. Verification: 124 focused tests passed (cost chip, provider readiness refresh, inspector/exchange capture, cost tracker/status chip); Ruff and git diff --check passed. ADR required: no (test-harness-only repair). No new lessons entry was needed because the task-ID collision pattern is already documented in lessons-backlog-hygiene.md.
<!-- SECTION:NOTES:END -->
