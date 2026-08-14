---
id: TASK-16308
title: Restore remote GGUF managed acquisition after local-import guard
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 08:11'
updated_date: '2026-08-14 08:16'
labels:
  - artifacts
  - security
  - tests
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the previously supported pinned Hugging Face GGUF install path without allowing path-private local imports to enter network acquisition.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pinned remote GGUF catalogs with recorded integrity can preflight and install through the managed acquisition flow.
- [x] #2 Path-private local GGUF descriptors remain ineligible for network acquisition even when a source map is supplied.
- [x] #3 Focused Model Artifacts tests and static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce and trace the remote GGUF acquisition failure against ADR-025 and the incumbent local-import guard
2. Add a regression that distinguishes path-private local descriptors from pinned remote descriptors
3. Narrow the acquisition guard to the local path-private shape without weakening source-map validation
4. Run focused and chunk regression gates, mutation-check the guard, and close the task

ADR required: no
ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
Reason: ADR-025 already defines both remote locally-recorded integrity and path-private local import boundaries; this fix reconciles their existing implementation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reconciled the incumbent remote-discovery and local-import boundaries defined by ADR-025. The acquisition guard now rejects exact local-integrity descriptors only when their canonical source URL is empty, preserving the path-private local-import refusal while allowing pinned HTTPS remote GGUF catalogs to use the existing managed preflight/provision flow.

Verification: the original remote integration test was RED before production; the remote plus two local refusal nodes pass 3/3; full acquisition-types and remote-Hugging-Face modules pass 75/75; exact chunk 27 passes 1,708/1,708 with one existing Requests dependency warning. Ruff lint, py_compile, and git diff checks pass. Ruff format remains red on unrelated pre-existing formatting elsewhere in acquisition.py; the edited hunk matches Ruff formatting and no broad formatting churn was introduced.

ADR required: no. ADR-025 already governs both boundaries.
<!-- SECTION:NOTES:END -->
