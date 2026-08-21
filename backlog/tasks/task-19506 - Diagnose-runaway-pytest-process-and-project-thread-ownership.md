---
id: TASK-19506
title: Diagnose runaway pytest process and project thread ownership
status: To Do
assignee: []
created_date: '2026-08-21'
labels:
  - testing
  - performance
  - diagnostics
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Determine whether the observed long-running high-CPU pytest process represents a current application or fixture lifecycle defect before changing AgentService, fleet, or global test teardown behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The suspected phase-zero tests run with short per-test bounds, verbose current-test identity, RSS sampling, and classified project-owned thread names and counts
- [ ] #2 Any reproduction is prefix-bisected and captures Python stacks plus producer/task state before termination
- [ ] #3 Suspected fixture teardown tests assert project-owned thread counts return near the measured baseline
- [ ] #4 An unbounded producer-signal wait is changed only when a deterministic test proves the producer can fail or exit without signaling
- [ ] #5 A concrete defect receives an atomic fix task and RED test; otherwise this task closes with captured evidence and no speculative runtime changes
<!-- AC:END -->
