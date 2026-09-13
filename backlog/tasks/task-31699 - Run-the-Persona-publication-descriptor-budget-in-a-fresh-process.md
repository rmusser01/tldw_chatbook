---
id: TASK-31699
title: Run the Persona publication descriptor budget in a fresh process
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:41'
updated_date: '2026-09-05 18:52'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Measure the fixed publication descriptor ceiling independently of unrelated open descriptors accumulated by a broad test worker.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The real256asset publication succeeds under unchanged RLIMIT_NOFILE256 in a fresh interpreter.
- [x] #2 Unrelated parent descriptors do not consume the publication measurement or relax its ceiling.
- [x] #3 The complete publication file and deterministic parent-pressure check pass with scoped static checks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the worker failure by opening 300 unrelated parent descriptors; keep the existing real 256-asset publication and RLIMIT_NOFILE 256. 2. Run the exact budget case in a fresh subprocess with close_fds, acquiring its normal database fixture only in the child; preserve identity assertions and propagate child failure. 3. Repeat deterministic parent pressure, run the complete publication file, and scoped static checks. ADR required: no. ADR path: N/A. Reason: test measurement isolation with standard-library subprocess, no runtime or resource-contract change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Moved the exact descriptor measurement to a fresh Python child with close_fds and only its own normal database fixture. The child still creates all 256 assets, enforces RLIMIT_NOFILE=256, publishes through real SQLite/filesystem boundaries, and checks exact active identity. A deterministic probe holding 300 parent descriptors reproduced the original failure and now passes (1 passed/2.41s); the complete publication file passes 53/16.96s. Child nonzero status or missing test execution is an explicit failure; no cap increase or production change. Scoped Ruff/formatter and diff checks pass. ADR not required: measurement isolation only.
<!-- SECTION:NOTES:END -->
