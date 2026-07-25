---
id: TASK-530
title: >-
  skill_file runtime tool follow-ups (PR #814 deferred minors)
status: To Do
assignee: []
created_date: '2026-07-24 14:10'
updated_date: '2026-07-24 14:10'
labels:
  - skills
  - agents
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred minors from the reachability final review (PR #814): the 100K read-cap boundary and NUL-bytes-in-valid-UTF8 refusal are exercised only transitively, not unit-pinned; skill_file bindings are per-turn (a $-mention grants access for that reply only - decide whether per-conversation accumulation is wanted, seam is the bridge's per-conversation dicts); the multi-skill 'Bundled files' block lists files without per-row skill attribution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Unit tests pin the exact 100_000-char truncation boundary and the binary/NUL refusal path.
- [ ] #2 A recorded decision (or implementation) on per-conversation binding accumulation.
- [ ] #3 Multi-skill bundle blocks attribute each file row to its skill.
<!-- AC:END -->
