---
id: TASK-15766
title: Repair a batch of pre-existing red tests found during the latency burn-down
status: To Do
assignee: []
created_date: '2026-08-13 12:31'
labels:
  - test-health
priority: low
---

## Description

Small, unrelated pre-existing test failures surfaced as asides while closing
out the input-latency burn-down (task-15450 - task-15481), each individually
too small to justify its own task but worth tracking as one batch rather than
being silently re-attributed to whatever branch happens to run next to them
(the same rationale task-15512 used for its own cluster). All re-verified
live against dev `6b57458b8` in this worktree on 2026-08-13 except where
noted as a flake.

1. **`Tests/Widgets/test_library_collections_panel.py::
   test_library_collections_panel_empty_state_renders_message_once`** —
   confirmed pre-existing and unrelated by task-15479's notes (reproduces in
   isolation on unmodified files).
2. **`Tests/TTS/test_tts_profile_capabilities.py`** — 3 of 43 tests fail on
   standing pre-existing Protocol-`isinstance` checks, called out in
   task-15479's notes as untouched by that task's change.
3. **`Tests/UI/test_console_control_bar_coalescing.py`** — intermittent;
   task-3070's notes record it failing 2/3 on dev "independently of any
   fleet change" at the commits it measured. Passed 2/2 in this session's
   run — file as a flake, not a deterministic red.
4. **`Tests/UI/test_console_internals_decomposition.py::
   test_console_left_rail_sections_use_available_space`** — confirmed
   red live: `ConsoleWorkspaceContextTray`'s parent is
   `console-rail-section-body-conversations`, expected
   `console-rail-section-body-session`.
5. **`Tests/UI/test_console_citation_sources.py::
   test_zero_only_count_cache_does_not_refresh_unchanged_transcript`** —
   confirmed red live: `AttributeError: 'types.SimpleNamespace' object has no
   attribute 'set_presentation_context'` at `chat_screen.py:15551` — a test
   double drifted behind a production call that now expects the method.
6. **`Tests/UI/test_console_shell_chip_actions.py::
   test_swap_seeds_greeting_only_into_an_empty_chat`** — confirmed red live:
   `TypeError: ConsoleSessionController._swap_console_session_character()
   takes 4 positional arguments but 6 were given`. Flagged as an unrelated,
   pre-existing signature mismatch in task-15476's notes.
7. **`Tests/UI/test_settings_model_catalog_toggles.py::
   test_model_catalog_toggles_initialize_from_saved_config`** — a
   QwenCloud-provider test-data drift, flagged as pre-existing and unrelated
   in task-15470's notes ("caused by an unrelated earlier commit").

## Acceptance Criteria

- [ ] Each of the seven tests above is diagnosed to its causing change (commit
      or drifted contract) and either fixed or the assertion updated to match
      current, intended behavior
- [ ] Item 3 (coalescing flake) is either stabilized (root cause of the
      intermittency identified and fixed) or documented as a known flake with
      its trigger condition, not silently left red
- [ ] All seven pass on dev; no other test in the same files regresses
