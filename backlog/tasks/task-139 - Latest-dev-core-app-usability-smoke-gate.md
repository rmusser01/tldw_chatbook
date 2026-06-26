---
id: TASK-139
title: Latest-dev core app usability smoke gate
status: Done
assignee:
  - '@codex'
created_date: '2026-06-26 17:16'
updated_date: '2026-06-26 17:36'
labels:
  - qa
  - app-usability
  - console
  - settings
  - library
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a focused latest-dev smoke gate that proves the core app first-use path launches, navigates, and presents recoverable Console setup states without relying on Sync or Persona work. This gives the team a current app-usability signal before starting deferred or owner-blocked backlog items.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Clean first-run app launch reaches Home and the non-Sync/non-Persona core destinations needed for first use without traceback, empty chrome, raw object reprs, or local path leaks.
- [x] #2 Console setup-required state remains recoverable from Home and Console with a visible Settings/provider configuration path.
- [x] #3 Library-to-Console staged-context core loop remains usable with deterministic local fixtures.
- [x] #4 The smoke explicitly avoids taking ownership of Sync and Persona implementation work.
- [x] #5 Focused automated checks and actual QA evidence are recorded; any defects outside this slice become separate backlog tasks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
<!-- SECTION:IMPLEMENTATION_PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a QA/test/evidence gate for existing app launch and recovery behavior. It does not change storage/schema, sync policy, security boundaries, provider/runtime ownership, service contracts, or long-lived UX architecture.

Plan: [Docs/superpowers/plans/2026-06-26-core-app-usability-smoke-gate.md](../../Docs/superpowers/plans/2026-06-26-core-app-usability-smoke-gate.md)

1. Record TASK-88 as not ready because no server-first persisted MCP defaults contract exists in the current client/service/schema surfaces.
2. Add or reuse focused smoke coverage for clean first-run launch, non-Sync/non-Persona destination navigation, Console setup recovery, and Library-to-Console staged context.
3. Run the smoke checks on latest `dev` and fix only defects exposed inside this task scope.
4. Capture actual QA evidence under `Docs/superpowers/qa/core-app-usability-smoke/`.
5. Check off acceptance criteria, add implementation notes, and mark the task Done only after verification passes.
<!-- SECTION:IMPLEMENTATION_PLAN:END -->
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added `Tests/UI/test_latest_dev_core_app_usability_smoke.py` as a latest-dev first-use smoke gate covering Home, Console, Library, Skills, MCP, Settings, Home-to-provider-settings recovery, and Library-to-Console staged context while excluding Sync and Persona ownership. The initial red run exposed two in-scope usability defects: Settings Overview rendered the full local config path, and Home model setup routed to the legacy model route instead of Settings/provider configuration. Settings Overview now renders a non-sensitive config filename/source label, and the Home model setup action routes to Settings with the existing Providers & Models navigation context.

Recorded QA evidence in `Docs/superpowers/qa/core-app-usability-smoke/2026-06-26-latest-dev-core-app-smoke.md`. TASK-88 was left deferred because no confirmed server-first persisted MCP defaults contract exists in the current MCP client/service/schema surfaces.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->
