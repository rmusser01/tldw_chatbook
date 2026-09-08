---
id: TASK-32039
title: >-
  Critique #6 fix-wave Qodo follow-ups: reason-staleness on resume, validation,
  stale test
status: Done
assignee: []
created_date: '2026-09-08 06:18'
updated_date: '2026-09-08 07:05'
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
- [x] #1 The repeated-fault 'reopen Chatbook' recovery step no longer fires on the first failure of a Library resume or a changed page/query/type/nav/facet context (the fault reason is scoped to its context or cleared when the context changes); a genuine consecutive Retry of the same context still escalates
- [x] #2 The bulk-Analyze inline reason and gate refresh when Library resumes or provider configuration changes, so a provider configured mid-session is reflected without a restart
- [x] #3 resolve_adaptive_reader_layout raises a clear TypeError for a non-boolean reader_has_item, documented in Raises, with unit coverage
- [x] #4 Both split-sheet bundle tests (test_generated_stylesheet_includes_library_media_rules AND its twin test_generated_stylesheet_includes_library_shell_rules) assert their library selectors in screen_agentic_library.tcss (the runtime-loaded split sheet), not the boot bundle, and pass
- [x] #5 The critique-#6 test-hygiene nits are addressed: the new imports form one contiguous local group, and the parameterized painted tests carry a Google-style summary + Args for the size parameter
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Five Qodo follow-ups from the critique #6 fix wave. A (correctness): the media browse controller's repeated-fault detection now pairs each reason with a SHA-256 context fingerprint over (query, media_type, sort_by, page) for both the page and facet fences, requires BOTH to match to escalate, clears both on success, and `clear_fault_episode()` runs first in on_screen_resume -- so a resume auto-refresh or a context change no longer gets the 'reopen Chatbook' clause on its first failure, while a genuine same-context Retry still escalates. B (correctness): on_screen_resume drops the bulk-Analyze reason memo so a mid-session provider config is reflected on return; the fix initially cleared a nonexistent screen attribute (dead no-op) and its pin crashed on a removed shim -- corrected in a fix round to `_media_state.analyze_reason_cache` and `_media_state.row_selection.count` (red-first confirmed). C: resolve_adaptive_reader_layout bool-validates reader_has_item (TypeError, Raises: doc, parametrized test). D: both split-sheet bundle tests (media AND the shell twin) retargeted from the boot bundle to screen_agentic_library.tcss where TASK-15450 moved the rules. E: import grouping + docstring Args nits. Files: library_media_browse_controller.py, library_screen.py, adaptive_reader_state.py, Tests/UI/test_library_media_render_fixes.py, Tests/UI/test_library_shell.py, Tests/Library/test_library_adaptive_reader_state.py.
<!-- SECTION:NOTES:END -->
