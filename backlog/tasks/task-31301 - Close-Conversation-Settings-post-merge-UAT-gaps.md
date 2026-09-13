---
id: TASK-31301
title: Close Conversation Settings post-merge UAT gaps
status: Done
assignee:
  - '@codex'
created_date: '2026-09-04 18:56'
updated_date: '2026-09-04 19:59'
labels:
  - console
  - ux
  - testing
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the four defects found during post-merge UAT so first-time and experienced users receive consistent setup guidance, deterministic credential return behavior, and trustworthy provider verification without test-network leakage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Conversation Settings uses the full Use for this conversation action label at compact, normal, and wide terminal sizes.
- [x] #2 Missing-credential guidance uses a consistent API key missing for Provider phrase in readiness and setup actions.
- [x] #3 Endpoint-probe tests preserve production lazy imports and intercept the real dependency seam without unintended network access.
- [x] #4 The credential return status-fault flow waits for its semantic completion condition under suite load, without retries, skips, flaky markers, or weakened assertions.
- [x] #5 Focused automated regressions, static checks, localhost endpoint verification, and mounted 80x24, 100x30, and 160x40 UAT pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR paths: backlog/decisions/006-provider-aware-generation-settings.md; backlog/decisions/011-chatbook-workbench-ui-system.md; backlog/decisions/012-provider-credential-settings-boundary.md; backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/097-boot-budget-ratchets.md
Reason: This is copy and test-remediation within existing Conversation Settings, navigation, probe, and lazy-import boundaries.

1. Capture RED evidence for each reported defect, including suite-order reproduction of the return fault.
2. Apply the smallest production and test-seam fixes that satisfy the approved UX copy and isolation contracts.
3. Run focused regressions, static checks, localhost probe validation, and mounted multi-size UAT.
4. Self-review the diff and document verification evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Kept the full primary action label at every supported width and changed compact footer groups to vertical, full-width, auto-height controls so the copy remains readable and pointer/keyboard reachable.
- Standardized missing-credential readiness text as `API key missing for {Provider}` while preserving rejected-credential wording.
- Repaired endpoint tests to patch the source module used by production lazy imports and added an owned-loopback `/v1/models` integration test that verifies returned model IDs.
- Pre-PR review found seven additional stale consumer-alias patches in the provider-draft suite; all now patch the owning endpoint-probe module, and a repository sweep confirms none remain.
- Replaced fixed four-second polling in the status-fault return test with a bounded semantic wait that retains the original exact assertions.
- Stabilized the compact focus-traversal harness by observing the production reveal-settle timer for the first manually focused control, matching every subsequent Tab step without weakening geometry assertions.
- Added a testing-evidence lesson for lazy-import dependency patching. No new ADR was required; the work stays within ADRs 006, 011, 012, 033, and 097.
- Verification after rebasing onto current `origin/dev`: 69 focused workflow tests passed; all 76 provider-draft tests passed; 11 mounted geometry/keyboard tests passed across 80x24, 100x30, and 160x40, followed by three repeated compact-focus passes; the real localhost probe and lazy-import boot closure passed. Compileall, `git diff --check`, stale-seam search, and scoped Ruff checks passed. Independent re-review reported no remaining actionable findings.
<!-- SECTION:NOTES:END -->
