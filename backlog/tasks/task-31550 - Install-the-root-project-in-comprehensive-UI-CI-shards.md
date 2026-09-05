---
id: TASK-31550
title: Install the root project in comprehensive UI CI shards
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-04 23:59'
updated_date: '2026-09-04 23:59'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore meaningful UI test execution in the comprehensive GitHub Actions workflow. The UI shards currently install dependency files but omit the repository package, so the bundled tldw_profile_core distribution is unavailable and every UI test errors during autouse setup before its test body can run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The UI test job installs the repository package before invoking pytest.
- [ ] #2 A focused workflow contract test fails if the UI job can no longer import bundled project packages.
- [ ] #3 The workflow contract tests and profile-core packaging checks pass.
- [ ] #4 Comprehensive UI CI reaches collected test bodies instead of aborting on a missing tldw_profile_core import.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused CI workflow regression test that proves the UI job installs the root project before pytest, and run it to capture the expected failure.
2. Add the minimal editable root-project installation to the UI job.
3. Run the focused workflow contract, the full CI workflow contract file, and profile-core packaging checks.
4. Re-run comprehensive CI evidence on the corrected workflow when branch publication is available.

ADR required: no
ADR path: N/A
Reason: This is a CI installation correction that preserves existing packaging and runtime boundaries.
<!-- SECTION:PLAN:END -->
