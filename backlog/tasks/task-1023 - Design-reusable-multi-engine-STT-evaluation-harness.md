---
id: TASK-1023
title: Design reusable multi-engine STT evaluation harness
status: To Do
assignee: []
created_date: '2026-07-27 23:38'
updated_date: '2026-07-27 23:38'
labels:
  - stt
  - evaluation
  - architecture
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define how the existing one-shot STT evaluator could later become a canonical Chatbook-owned baseline harness for local and cloud transcription engines without weakening reproducibility or artifact-promotion safety.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A reviewed design separates direct-engine evaluation from production integration smoke tests
- [ ] #2 The design supports local reproducible and remote observational evidence while reserving artifact-promotion gates for local evidence
- [ ] #3 A thin versioned suite-policy boundary preserves strict suite-specific requirements without introducing a policy DSL or plugin framework
- [ ] #4 Corpus identity can be reused across model suites without rebuilding unchanged prepared audio
- [ ] #5 Cloud execution has explicit request budgets, credential redaction, and no promotion eligibility
- [ ] #6 The design records a concrete task breakdown small enough for single-PR implementation tasks
<!-- AC:END -->
