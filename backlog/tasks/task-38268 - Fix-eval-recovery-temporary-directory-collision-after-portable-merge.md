---
id: TASK-38268
title: Fix eval recovery temporary directory collision after portable merge
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-31T23:49:00Z'
updated_date: '2026-11-01T00:24:00Z'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The merged portable bundle introduced an autouse private temporary directory that collides with the existing launch-environment regression on Ubuntu and macOS. Restore meaningful coverage and audit the remaining merged PR CI results.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All three native temporary selectors reach and pass their launch-environment assertions with the autouse private fixture active.
- [ ] #2 The targeted eval-recovery workflow and remaining PR checks are verified or their unresolved results are explicitly recorded.
- [ ] #3 Windows recovery tests and their child processes use owned APPDATA and LOCALAPPDATA sandboxes so ambient installed pet state cannot enter a test backup.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A (existing ADR-126 recovery guards remain unchanged)
Reason: routine test-only namespace collision; no new runtime or trust boundary.
1. Reproduce all three FileExistsError failures from PR #3287 with native private fixture setup.
2. Give the launch-environment test its own temporary directory and retain all selector/secrets assertions.
3. Verify the affected module and exact eval-recovery workflow selection; run scoped lint and review the diff.
4. Audit all remaining PR #3287 check results, record unresolved checks, and publish a scoped follow-up PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Scope update before implementation: clean qualification exposed ambient Windows APPDATA leaking host pet state into the full backup fixture. Add test-only native AppData isolation; production recovery correctly refused wrong relocation and remains unchanged. Existing ADR-126 applies, no new ADR.
<!-- SECTION:NOTES:END -->
