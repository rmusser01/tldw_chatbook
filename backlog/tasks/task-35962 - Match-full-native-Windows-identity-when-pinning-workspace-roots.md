---
id: TASK-35962
title: Match full native Windows identity when pinning workspace roots
status: Done
assignee:
  - '@codex'
created_date: '2026-10-21 04:36'
updated_date: '2026-10-21 04:56'
labels: []
dependencies: []
documentation:
  - backlog/decisions/101-one-shot-pinned-workspace-tool-execution.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
An unchanged authorized workspace folder must remain readable on Windows when native identity APIs use different device identifier widths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A real unchanged Windows folder pins successfully using the same full device and inode identity captured by Python.
- [x] #2 Identity replacement and reparse roots remain refused; no identity masking or permission widening is introduced.
- [x] #3 Targeted native root-pin and real fs_list contract regressions provide RED/GREEN evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Prerequisite to guidance real-file verification. Follow systematic debugging and TDD: reproduce real unchanged-root refusal, use full native Windows identity matching Python, preserve reparse/access/cleanup checks, verify focused root-pin and fs_list tests, review. ADR required: yes; ADR path: backlog/decisions/101-one-shot-pinned-workspace-tool-execution.md; Reason: identity representation correction preserves the existing authority boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Full native Windows volume/file identity correction in6678eecb0d preserves exact equality, reparse refusal, cwd rechecks and handle cleanup. Native RED/GREEN:15root-pin,5realfs_list,1provenance passed; canonical generated bundle parity/AST checks passed. Independent spec/quality review approved. Mechanical bundle includes documented pre-existing snapshot drift. Unsupported full metadata remains fail-closed; POSIX/remote runtime unqualified. Existing E721 and dependency warnings unchanged. Files: Tools/workspace_root_pin.py, Tests/Tools/test_workspace_root_pin.py, Tools/remote_worker_bundle.py. ADR: backlog/decisions/101-one-shot-pinned-workspace-tool-execution.md.
<!-- SECTION:NOTES:END -->
