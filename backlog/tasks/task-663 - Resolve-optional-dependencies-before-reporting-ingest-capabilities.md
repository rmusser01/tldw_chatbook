---
id: TASK-663
title: Resolve optional dependencies before reporting ingest capabilities
status: To Do
assignee: []
created_date: '2026-07-26 03:26'
labels:
  - ingest
  - bug
  - p1
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The ingest capability layer reads the optional-dependency registry without ever resolving it, and lazy checking is the default, so every optional feature reports as missing. Users are told to install packages they already have, and every per-type advanced option is permanently disabled, making the PDF engine, OCR, transcription model, language, timestamps, diarization and e-book extraction controls impossible to change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Capability checks reflect what is actually installed rather than an unresolved registry
- [ ] #2 Advanced options for an installed feature are editable
- [ ] #3 Advanced options for a genuinely missing feature remain disabled
- [ ] #4 No tooling warning is shown for a feature that is installed
- [ ] #5 Dependency resolution happens once rather than per field render
<!-- AC:END -->
