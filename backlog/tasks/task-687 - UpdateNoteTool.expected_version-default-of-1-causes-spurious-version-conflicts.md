---
id: TASK-687
title: UpdateNoteTool.expected_version default-of-1 causes spurious version conflicts
status: To Do
assignee: []
created_date: '2026-07-26 06:06'
labels:
  - tools
  - agents
  - notes
dependencies:
  - TASK-545
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UpdateNoteTool's expected_version parameter (Tools/note_management_tools.py) defaults to 1 when the LLM omits it. An LLM calling update_note on any note that has already been edited more than once (version > 1) hits a spurious optimistic-locking conflict on a perfectly valid call, because the tool schema's default silently disagrees with the note's actual current version. Identified in the design spec for TASK-545 P2 (Docs/superpowers/specs/2026-07-25-port-mutating-tools-design.md, 'Known limitations carried, not fixed'), which explicitly deferred fixing it to keep that phase scoped to porting the tool behind the permission gate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 update_note succeeds on a note whose version has advanced past 1 without the caller having to guess the correct expected_version
- [ ] #2 A test exercises update_note against a note at version > 1 with no expected_version supplied and asserts it does not spuriously conflict
<!-- AC:END -->
