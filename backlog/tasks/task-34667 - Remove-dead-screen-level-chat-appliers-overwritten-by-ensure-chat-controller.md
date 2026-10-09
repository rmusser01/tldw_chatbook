---
id: TASK-34667
title: Remove dead screen-level chat appliers overwritten by ensure_chat_controller
status: To Do
created_date: 2026-10-09 06:27
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from TASK-34435 review: chat_screen.py:14217/14241 defines dictionary/world-info appliers passed into runtime.ensure_chat_controller (chat_screen.py:9793-9794) which unconditionally overwrites both via kwargs.update (console_runtime.py:4286-4292) - dead on every path. Drift precedent: the _library_provider_for_app docstring (console_runtime.py:662-678) records a stale always-overwritten copy that cost 24 Library tools behind one swallowed warning. Delete the dead appliers (or wire them for real) with caller-grep evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Caller-grep evidence pasted,Dead appliers removed or genuinely wired,Console tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See .superpowers/sdd/nonconsole-followups/task-34435-report.md review section
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
