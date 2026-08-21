---
id: TASK-16324
title: Group Console tool activity inside assistant turns
status: In Progress
assignee: []
created_date: '2026-08-21 08:49'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Console transcripts clearly attribute reasoning and tool activity to the assistant response that produced them, reducing ambiguity and transcript clutter.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each user query is followed by one visually coherent Assistant turn container.
- [ ] #2 Tool and reasoning activity is rendered inside its owning Assistant turn.
- [ ] #3 Tool and reasoning details are collapsed by default and can be expanded independently.
- [ ] #4 The final assistant answer remains visible in the same Assistant turn after its activity rows.
- [ ] #5 Existing tool-output expansion and message actions remain usable.
- [ ] #6 Focused transcript tests and live visual verification cover completed, streaming, failed, and resumed turn shapes.
<!-- AC:END -->
