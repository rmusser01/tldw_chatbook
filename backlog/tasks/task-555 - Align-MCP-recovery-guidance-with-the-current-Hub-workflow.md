---
id: TASK-555
title: Align MCP recovery guidance with the current Hub workflow
status: Done
assignee: []
created_date: '2026-07-24 21:41'
updated_date: '2026-07-24 21:49'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure the MCP destination's unavailable or unconfigured state gives users a clear, verifiable next recovery action that matches the current MCP Hub information architecture rather than a retired workflow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Users encountering an unavailable or unconfigured MCP registry can identify a concrete next recovery action from the mounted screen
- [x] #2 Recovery guidance and its product-maturity contract use current MCP Hub controls and do not reference retired navigation
- [x] #3 Healthy configured and empty-state MCP behavior remains unchanged
- [x] #4 Focused MCP and product-maturity recovery tests plus static and diff checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the Phase 6 recovery replay failure and compare its retired Inventory assertion with the mounted MCP Hub, redesign spec, and introducing commit history.
2. Update the running-app contract to assert the current Servers-mode recovery controls by widget identity, display state, enabled state, label, and actionable built-in readiness callout instead of searching hidden/static text for the retired Inventory copy.
3. Align the MCP row in the release recovery guide with the current Servers-to-Tools workflow.
4. Run the focused Phase 6 replay, MCP screen/workbench/servers-mode regressions, Ruff/format checks for the changed test, and diff hygiene.
5. Request independent review before documenting and closing the task.

ADR required: no
ADR path: N/A
Reason: This is a stale recovery-test and documentation correction that applies the existing MCP Hub redesign contract; it changes no runtime boundary, application architecture, or long-lived workflow.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Root cause: the Phase 6 replay retained `Next: select Inventory...` after MCP Hub retirement commit 09f5c42b6 removed the legacy Unified MCP panel and its Inventory renderer. The current MCP redesign intentionally starts in Servers mode and uses Add server/import/readiness callouts before the Tools catalog.
- Replaced the stale text search with mounted widget assertions for the exact Add server action and built-in setup callout, including enabled state and nonzero rendered geometry; the replay also rejects the retired sentence explicitly. No production MCP behavior changed.
- Updated the release recovery guide to match the mounted Console status and current MCP Servers add/import/configure → Tools workflow, with a focused table-row contract preventing documentation drift.
- Verification: the Phase 6 recovery file passes 2/2; five neighboring MCP workbench/server interaction tests pass; the original 16-file maturity sweep passes 42/42 in 94.43 seconds. Ruff check, Ruff format verification, and `git diff --check` pass. The only warning is the pre-existing requests dependency-version warning.
- Independent review traced the retired renderer history, checked the redesign spec and production behavior, ran the 7 focused tests, and approved the docs/test-only remediation with no findings.
- ADR required: no; ADR path: N/A; reason: stale recovery-test and documentation correction under the existing MCP Hub redesign contract, with no runtime or architecture change.
<!-- SECTION:NOTES:END -->
