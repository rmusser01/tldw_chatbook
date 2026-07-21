---
id: TASK-402
title: >-
  Fix silent settings-None no-op in set_session_system_prompt and
  set_session_pinned_prefill
status: To Do
assignee: []
created_date: '2026-07-21 03:48'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Both store methods silently skip the in-memory settings update when session.settings is None yet still report persisted=True. Currently unreachable via UI handlers (they ensure settings), but the contract is dishonest for direct callers. Fix both twins together.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both methods either seed default settings or report the skipped update honestly,Unit tests cover the settings-None path for both methods
<!-- AC:END -->
