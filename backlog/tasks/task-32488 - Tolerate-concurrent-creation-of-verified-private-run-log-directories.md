---
id: TASK-32488
title: Tolerate concurrent creation of verified private run-log directories
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 06:21'
updated_date: '2026-09-08 06:39'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Concurrent first runs can race to create their shared private log directory, causing one run to silently disable logging despite a valid directory. Accept a safely verified directory created by another caller without weakening the private-path boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Concurrent callers creating the same private directory both succeed and retain owner-only directory permissions.
- [x] #2 A competing file, symlink, or unsafe directory entry is rejected without changing the replacement target.
- [x] #3 Focused private-path and agent run-log tests cover the regression.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/029-local-private-data-boundary.md
Reason: Repair a creation race while preserving the existing descriptor-based privacy boundary.
1. Reproduce concurrent mkdir through synchronized real filesystem callers and assert both usable private results.
2. Reopen a competing entry with the existing no-follow checks; do not classify the losing caller as creator.
3. Verify file/symlink rejection and actual run-log writes under contention.
4. Run focused private-path and run-log tests, lint and formatting; update the review ledger.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Concurrent first-run data-directory creation no longer disables one run log. The losing mkdir caller reopens the existing entry without following links and preserves all owner/type/mode/postcondition checks; only the actual creator is classified as such. Both real writers persist their records under a barrier-controlled race. File, symlink, and unsafe intermediate replacements remain rejected without modifying their targets.

Validation: before the fix, three regression cases failed (two direct creators and the real writer path); afterward 135 targeted private-path/run-log tests passed. Final combined run: 163 targeted private-path, run-log, wake, approval-safety, and retry tests passed. New tests and the edited helper pass Ruff format; new tests pass lint and the helper adds no finding beyond its pre-existing UP035 import issue. Self-reviewed the descriptor path and verified the scoped diff. No full suite or live provider was run.

ADR: existing backlog/decisions/029-local-private-data-boundary.md applies; no new boundary. Files: Utils/private_paths.py, Tests/Utils/test_private_directory_creation_race.py, and the agent-orchestration review ledger. No additional lesson arose beyond the documented race and its reproduction.
<!-- SECTION:NOTES:END -->
