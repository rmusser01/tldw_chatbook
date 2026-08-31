---
id: TASK-24614
title: Research scheduled Agent Lesson improvement runs
status: To Do
assignee: []
created_date: '2026-08-30 01:18'
labels:
  - notes
  - agents
  - research
dependencies:
  - TASK-24309
  - TASK-24613
documentation:
  - >-
    Docs/superpowers/specs/2026-08-29-agent-lessons-notes-organization-sync-design.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Research whether opt-in scheduled improvement runs can safely identify recurring Agent Lesson evidence and prepare reviewable suggestions without creating an autonomous instruction-writing loop.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The research defines candidate sources, scheduling ownership, permission scope, privacy limits, review checkpoints, cancellation, retention, and observability without implementing production behavior
- [ ] #2 The research compares a local foreground workflow, Chatbook-owned scheduling, and the existing server-scheduled-agent boundary without presuming a preferred runtime owner
- [ ] #3 The research specifies how lessons remain untrusted evidence and why no run may automatically edit Notes, repository instructions, managed skills, permissions, or scheduling configuration
- [ ] #4 The research identifies measurable signal-quality and safety evaluations, including false promotion pressure, repeated rejected suggestions, stale evidence, and cross-user or cross-profile leakage
- [ ] #5 Any recommended implementation is split into separately reviewable future tasks and is preceded by a new ADR covering scheduling, execution ownership, data flow, and conflict policy
- [ ] #6 No application code, schema migration, server contract, or automation is created by this task
<!-- AC:END -->
