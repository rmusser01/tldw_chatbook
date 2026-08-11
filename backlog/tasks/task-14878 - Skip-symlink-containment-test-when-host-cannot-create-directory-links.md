---
id: TASK-14878
title: Skip symlink containment test when host cannot create directory links
status: Done
assignee:
  - '@codex'
created_date: '2026-08-11 00:40'
updated_date: '2026-08-11 00:42'
labels:
  - tests
  - windows
  - agents
dependencies: []
type: bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the agent file-tool containment regression meaningful on hosts that support directory symlinks without failing the repository suite solely because the current Windows account lacks symlink creation privilege.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The containment test runs and verifies sandbox isolation when the host can create directory symlinks
- [x] #2 The test reports a clear skip when the platform or account cannot create the required symlink
- [x] #3 The change does not weaken production file-tool containment behavior
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Follow the repository's existing symlink-test portability pattern while distinguishing capability failures from unrelated filesystem errors.
2. Request a directory symlink explicitly so Windows creates the intended link type when the privilege is available.
3. Add regression coverage for capability-error skipping and unrelated-error propagation.
4. Run the focused test file, scoped Ruff, and diff checks before updating the task evidence.

ADR required: no
ADR path: N/A
Reason: This is test portability and does not change production behavior, dependencies, or architectural boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Made the agent file-tool symlink containment regression portable without changing production behavior. The fixture now requests a directory symlink, skips only for known platform or privilege capability errors, and re-raises unrelated filesystem failures. Added regression coverage for both error paths. Verification: 22 tests passed and the unsupported-host containment case skipped with explicit privilege copy; scoped Ruff and diff checks passed. ADR required: no; this is test-only portability.
<!-- SECTION:NOTES:END -->
