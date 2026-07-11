---
id: TASK-160
title: 'Ingest parallelism: heavy-lane cap for concurrent transcriptions'
status: To Do
assignee: []
created_date: '2026-07-11 22:02'
labels:
  - follow-up
  - ingest
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
F3's parallel parse pool runs all media types through one pool (default min(3, cpu-1)). Two concurrent audio/video transcriptions can be RAM/CPU heavy. Add a dedicated heavy lane capping concurrent transcription parses to 1 while documents fan out wide. Revisit only if RAM pressure shows up in practice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Concurrent transcription parses are capped independently of document parses
- [ ] #2 Config controls the heavy-lane cap
<!-- AC:END -->
