---
id: TASK-34871
title: Triage pre-existing Console suite failures on feat/custom-endpoint-registry
status: To Do
assignee: []
created_date: '2026-09-12 21:39'
labels:
  - console
  - tests
  - regression
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
While verifying task-32483/32484 (2026-09-12), four Console test clusters were found failing on branch feat/custom-endpoint-registry. Every cluster fails IDENTICALLY on the clean tip 7ec9bbda60 with the 32483/32484 changes stashed, so they predate that work — but they erode trust in the suite on this branch, and the largest identity cluster sits next to the task-32481 persona commits (c3c7eee114, 18b88387f2, 7ec9bbda60), which are prime suspects. Clusters (sequential runs, .venv/bin/python -m pytest <file> -q): (1) Tests/Chat/test_console_generation_actions.py — 40 failed, all AttributeError 'ChatScreen' object has no attribute '_retrieval' at chat_screen.py:5571 in console_view_hooks when _bare_generation_screen builds a bare screen; (2) Tests/UI/test_console_native_chat_flow.py — 15 failed in identity/state-restore tests (round_trip_preserves_persona_identity, round_trip_preserves_session_system_prompt, source_aware rejections, legacy identity adaptation, settings_save toast); (3) Tests/UI/test_console_session_settings.py — 2 failed: staged-context rail ordering (children.index(staged_context)==0 got 1) and hung-refresh repair old-screen cleanup; (4) Tests/UI/test_console_launch_wake.py — 1 failed: test_a_launch_built_controller_is_not_sticky_when_console_opens. Root-cause each cluster, fix or split into follow-up tasks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each cluster has its introducing commit identified (bisect or targeted runs), with the task-32481 persona commits explicitly cleared or confirmed for the native_chat_flow identity cluster,Each cluster is either fixed with its file green in a sequential run or has its own linked follow-up task with the failure signature recorded,All four named files pass sequentially (or their follow-up tasks are linked and this task explains the split),No new failures introduced: the four files' pass counts only increase relative to the recorded baseline (40/15/2/1 failures)
<!-- AC:END -->
