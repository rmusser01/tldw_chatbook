---
id: TASK-21144
title: Local provider probe feedback and auto-detect
status: To Do
assignee: []
created_date: '2026-08-25 06:14'
labels:
  - ux
  - wizard
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT findings P-6, P-7, P-8 (findings.md): Detect and Test give zero feedback when no local server is running (byte-identical frames); the subtitle promises auto-detection that never happens; Detect vs Test is unexplained.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Selecting a local provider auto-probes (debounced, short timeout) with a visible in-progress state
- [ ] #2 Every probe ends in a visible result: found endpoint, or a not-found message naming the address tried
- [ ] #3 Probe buttons are labeled by outcome (find server / test address)
<!-- AC:END -->
