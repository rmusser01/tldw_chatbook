---
id: TASK-16327
title: Add a research report self-eval harness
status: To Do
assignee:
  - '@robert'
created_date: '2026-08-15 05:16'
labels:
  - research
  - evals
dependencies:
  - TASK-16331
  - TASK-16322
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
There is no way to measure whether pipeline changes improve report quality. mole ships self-evaluation with grounding rates; the chatbook already has an Evals module. Add a small eval runner that scores research reports on citation accuracy and grounding using the verification data produced by the citation verification work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An eval runner scores reports on citation accuracy and grounding using verification outcomes from the pipeline,The runner integrates with the existing Evals framework rather than a parallel harness,A baseline metric set is recorded for the current pipeline,Tests cover the scoring logic with synthetic verification payloads
<!-- AC:END -->
