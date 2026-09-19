---
id: TASK-32771
title: Keep workspace Change Review readiness and recovery current
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 05:42'
updated_date: '2026-09-18 06:03'
labels:
  - ui
  - settings
  - design-system
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Workspace Change Review must show background preparation outcomes and actionable feedback where users operate its controls. Current Settings can retain preparing after completion, misreport retry failures, and lose visible focus while refreshing the workspace.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Preparing, failed and ready status tracks background completion without discarding unrelated workspace input drafts or moving newer focus.
- [x] #2 Consent conflicts, retry failures and retry outcomes are truthful, plain-text and visible beside the active controls in compact and wide themes.
- [x] #3 Consent remains opt-in and revision-bound; unavailable/global-off states expose no actionable toggle, retry remains bounded, and navigation cannot publish into another workspace.
- [x] #4 Targeted regressions and native private-profile journeys verify real registry and shadow-history behavior, visible focus and clean lifecycle.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. ADR path: backlog/decisions/084-change-review-consent-and-asynchronous-finalization.md (existing), plus ADR-150/161 for presentation. Reason: expose the existing consent/readiness contract faithfully without changing storage, authorization or initialization policy. 1. Preserve three reproduced mounted failures: hidden focus after conflict, retry exception misreported as no failed folders, and preparing remaining after backend readiness completes. 2. Isolate Change Review presentation into a small Settings panel that updates existing rows in place and observes only pending readiness; retain revision-bound toggles and bounded service retry. Keep local plain-text receipts beside controls, preserve unrelated drafts and newer focus, and fence detached/suspended reads. 3. Add dark/light compact/wide keyboard coverage, controlled background failure/retry/completion and navigation checks; run original consent/Settings regressions and governance/static checks with independent review. 4. Verify real private shadow-history initialization and failure recovery in the native app, inspect captures and lifecycle, then update the completion ledger and draft PR. Allocation: origin fetched; 316 refs and 30 worktrees; maximum32770 and no32771 reference before CLI allocation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Change Review now owns a retained Settings panel with local plain-text receipts and asynchronous observation only while preparation is pending. It preserves neighboring inputs and newer focus, fences navigation/suspension and stale read completions, reports retry exceptions truthfully, and captures revision-bound intent on each activation. No backend policy, schema or stylesheet changed; existing ADR-084 and ADR-150/161 apply.

Independent review reproduced and closed a queued Disable-to-Enable race introduced by retaining the button. The seven new UI journeys, 55 related Settings/governance cases and 11 existing consent cases all pass (73 distinct). Existing consent tests now use the established private-profile helper; AST review confirms their bodies/assertions are unchanged. Scoped Ruff/format pass; Settings diagnostic count falls from 116 to 114 with none introduced; backlog/diagnostic guards and diff checks pass.

Native run004 passed dark/light at 80x24 and 170x48 with real registry and shadow Git initialization after controlled failures, preserved rename drafts/focus, one retry, retained history on disable, unchanged fixture files, and clean Ctrl+Q shutdown. All 12 captures were rendered and inspected. Earlier fixture-only failures are retained with their normal-exit receipts. QA: Docs/superpowers/qa/2026-09-17-settings-change-review/README.md. Updated the completion audit and the concrete queued-intent lesson. Full Console agent-turn review/revert, imported-profile native review and workspace lifecycle dialogs remain outside this task; no full suite or provider requests ran.
<!-- SECTION:NOTES:END -->
