---
id: TASK-16239
title: Make the Git output-cap test stream-agnostic
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 09:59'
updated_date: '2026-08-14 10:00'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the Git subprocess cap evidence portable when platform Git writes an early diagnostic to stderr before stdout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The output-cap test accepts truncation on either bounded pipe.
- [x] #2 The assertion still proves a Git prefix, truncation marker, and bounded combined output.
- [x] #3 The full Git tool module and scoped static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm which Git pipe reaches the cap first on the failing platform.
2. Make the test assert the stream-independent bounded-output contract.
3. Run the focused node, full Git tool module, and scoped static checks.
4. Record verification and close the task.

ADR required: no
ADR path: N/A
Reason: This is a platform-portable test assertion for existing subprocess behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Made the four-byte Git cap assertion stream-agnostic: Apple Git can emit a sandbox/temp-directory warning to stderr before its stdout version line, so either bounded pipe may correctly trigger termination. The test still requires a Git prefix, truncation marker, and combined output under 100 bytes. Full module: 27 passed; focused rerun passed; Ruff check and git diff --check passed. Whole-file Ruff format remains pre-existing baseline-red, so unrelated formatting churn was reverted. ADR required: no.
<!-- SECTION:NOTES:END -->
