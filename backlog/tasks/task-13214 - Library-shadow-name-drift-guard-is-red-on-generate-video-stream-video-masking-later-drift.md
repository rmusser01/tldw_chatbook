---
id: TASK-13214
title: >-
  Library shadow-name drift guard is red on generate-video/stream-video, masking
  later drift
status: To Do
assignee: []
created_date: '2026-08-10 00:29'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/Library/test_library_skills_state.py::test_shadow_name_set_stays_in_sync_with_real_sources fails on origin/dev: ConsoleCommandRegistry names not covered: {'stream-video', 'generate-video'} — both present in console_command_grammar.py but absent from _SHADOWED_BUILTIN_NAMES (library_skills_state.py). Introduced by the video-generation work. Impact beyond the two names: the guard asserts three subsets IN ORDER, so whichever fires first masks every gap underneath it. Demonstrated during supervisor-fleet PR 2a — two newly added runtime tools (wait_agents/check_agents) were missing from the same set, and the RUNTIME_TOOL_NAMES assertion fired first, completely hiding the video gap until the tool names were fixed. While this guard is red, anyone adding a runtime tool or console command gets no drift signal at all, which is precisely the erosion the test's own message warns about ('do not accept this as a baseline failure (task-580)').
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 generate-video and stream-video are covered by _SHADOWED_BUILTIN_NAMES (or deliberately exempted with a documented reason)
- [ ] #2 The guard passes on a clean dev checkout
- [ ] #3 The assertion reports ALL uncovered names across the three sources in one failure rather than short-circuiting on the first subset, so one gap cannot mask another
<!-- AC:END -->
