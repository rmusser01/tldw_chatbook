---
id: TASK-26018
title: 'Compaction: focus-directed summaries'
status: To Do
assignee: []
created_date: '2026-08-31 15:45'
labels:
  - console
  - context
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A compaction summary cannot be steered toward what the user cares about. Verified on origin/dev: a grep for focus across Chat/console_context_compaction.py returns zero, so the summary prompt is fixed and a long debugging session compacts down to whatever the model judged salient. Hermes accepts a topic argument that biases the summary. The change is one string appended to the compaction prompt built for the auxiliary call.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A manual compaction accepts an optional topic that biases what the summary preserves
- [ ] #2 With no topic supplied, the produced summary is byte-identical to today's for the same input
- [ ] #3 The topic is recorded in the compaction provenance so a later reader knows the summary was steered and how
- [ ] #4 The topic is treated as untrusted user text: it is bounded in length and cannot inject instructions that alter the summarizer's role
- [ ] #5 A topic that yields an empty or unusable summary falls back to the unsteered path rather than committing a degraded record
<!-- AC:END -->
