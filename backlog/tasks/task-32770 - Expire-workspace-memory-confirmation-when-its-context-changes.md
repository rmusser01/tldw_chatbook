---
id: TASK-32770
title: Expire workspace memory confirmation when its context changes
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 05:26'
updated_date: '2026-09-18 05:36'
labels:
  - ui
  - settings
  - design-system
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Workspace memory access must require a current acknowledgement for the workspace and assistant defaults the user is reviewing. Navigating away or changing the saved defaults must not reuse an older read-write acknowledgement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Switching workspace cards or Settings categories and returning requires a fresh first press before read-write can be applied.
- [x] #2 Changing the saved persona, profile, or memory mode after the first press cannot silently apply read-write using the stale acknowledgement.
- [x] #3 Current two-press confirmation and imported-profile first-bind review continue to work; errors and cancellation remain visible and retryable.
- [x] #4 Targeted regressions and native dark/light compact/wide evidence show the current action and outcome with healthy private-profile shutdown.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. ADR path: backlog/decisions/079-workspace-assistant-defaults.md and backlog/decisions/107-portable-tool-use-packs.md (existing), with ADR-150/161 for visible feedback. Reason: repair the existing UI acknowledgement lifetime without changing schema, authority, registry locking or profile review. 1. Reproduce workspace/category roundtrips and saved-default changes through mounted Settings with a real private registry. 2. Scope each pending read-write acknowledgement to its workspace, saved defaults and exact intended defaults; discard it on navigation and validate before applying. Preserve current first-bind token checks and newer staging. 3. Run targeted assistant journeys, registry and governance checks and independent review. 4. Verify dark/light 80x24 and 170x48 in the native private app, inspect captures, account for shutdown, then update the ledger and draft PR. Allocation receipt: fetched origin,316 refs,30 worktrees,maximum32769,no32770 content reference before CLI assigned32770.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Read-write acknowledgement captures the workspace and immutable saved/intended defaults, validates them before applying, and expires on workspace/category/screen navigation. Stale presses show local feedback. Imported-profile cancellation retains staging but requires a fresh memory acknowledgement and new exact-token review; delayed applies still preserve newer staging. Validation: 95 distinct targeted tests (27 assistant, 67 registry/session/governance, one additional cancellation variant), scoped Ruff/format, no new findings against the 116-diagnostic Settings baseline, Backlog/diagnostic guards, and independent review. Four native dark/light compact/wide cells passed with 12 inspected captures. PID 4542 exited 0 normally, with 11 healthy private databases and unchanged default fingerprints. QA: Docs/superpowers/qa/2026-09-17-settings-workspace-confirmation/README.md. Existing ADR-079/107/150/161 apply; no new ADR or schema/locking/authority changes. Native imported-profile review, Change Review and full workspace lifecycle flows remain separate follow-ups.
<!-- SECTION:NOTES:END -->
