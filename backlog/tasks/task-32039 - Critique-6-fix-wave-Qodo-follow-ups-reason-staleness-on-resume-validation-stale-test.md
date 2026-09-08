---
id: TASK-32039
title: >-
  Critique #6 fix-wave Qodo follow-ups: reason-staleness on resume, validation,
  stale test
status: To Do
assignee: []
created_date: '2026-09-08 06:18'
labels:
  - library
  - media
  - robustness
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Consolidated follow-ups from Qodo review of the critique #6 fix PRs. Two are correctness bugs surfaced by the merged fixes: (a) the media browse controller's repeated-fault detection (PR #2494/task-31982) compares failures by reason alone and keeps that history across ordinary begin/request_facets calls, so a Library screen-resume auto-refresh or a context change that hits the same normalized reason gets the 'reopen Chatbook' recovery advice on its FIRST failure of the visit; (b) the select-mode bulk Analyze inline reason (PR #2497/task-31981) is cached for the whole select-mode session, so configuring a provider mid-session and returning (screen suspend/reuse preserves the cache) leaves the gate and inline reason stale. The rest are hygiene: bool-validate the new reader_has_item resolver arg (PR #2499); retarget a stale test that expects library CSS in the boot bundle after the screen-split moved it to screen_agentic_library.tcss; and two test-hygiene nits (import grouping, docstring/Args).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The repeated-fault 'reopen Chatbook' recovery step no longer fires on the first failure of a Library resume or a changed page/query/type/nav/facet context (the fault reason is scoped to its context or cleared when the context changes); a genuine consecutive Retry of the same context still escalates
- [ ] #2 The bulk-Analyze inline reason and gate refresh when Library resumes or provider configuration changes, so a provider configured mid-session is reflected without a restart
- [ ] #3 resolve_adaptive_reader_layout raises a clear TypeError for a non-boolean reader_has_item, documented in Raises, with unit coverage
- [ ] #4 Both split-sheet bundle tests (test_generated_stylesheet_includes_library_media_rules AND its twin test_generated_stylesheet_includes_library_shell_rules) assert their library selectors in screen_agentic_library.tcss (the runtime-loaded split sheet), not the boot bundle, and pass
- [ ] #5 The critique-#6 test-hygiene nits are addressed: the new imports form one contiguous local group, and the parameterized painted tests carry a Google-style summary + Args for the size parameter
<!-- AC:END -->
