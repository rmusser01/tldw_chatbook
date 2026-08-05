---
id: TASK-402
title: >-
  Fix silent settings-None no-op in set_session_system_prompt and
  set_session_pinned_prefill
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 03:48'
updated_date: '2026-07-21 07:07'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Both store methods silently skip the in-memory settings update when session.settings is None yet still report persisted=True. Currently unreachable via UI handlers (they ensure settings), but the contract is dishonest for direct callers. Fix both twins together.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both methods either seed default settings or report the skipped update honestly
- [x] #2 Unit tests cover the settings-None path for both methods
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. TDD: settings-None tests for both set_session_system_prompt and set_session_pinned_prefill asserting (session, False) honest flag\n2. Implement shared guard in both twins: durable write still runs (persistent truth), flag False + warning log when in-memory update skipped\n3. Docstring the contract; run store suite
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both twins now early-return (session, False) with a warning log when session.settings is None — the durable write is skipped too (no split-brain between live session and saved conversation). Zero production blast radius: the /system caller ensures settings via _ensure_active_console_session_settings and /prefill seeds defaults since PR #729. Tests: 2 new settings-None contract tests; store suite 79 passed.
<!-- SECTION:NOTES:END -->
