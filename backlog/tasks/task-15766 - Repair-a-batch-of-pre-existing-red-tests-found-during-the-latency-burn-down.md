---
id: TASK-15766
title: Repair a batch of pre-existing red tests found during the latency burn-down
status: Done
assignee:
  - '@claude'
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

## Implementation Plan

1. Re-run all seven items at this branch's HEAD (`48ad9e7de`) first — the board
   moves fast; any now-green item is marked resolved-elsewhere with the fixing
   commit (found via `git log` on the test/production file), not re-fixed.
2. For each still-red item: diagnose whether the red is test drift (production
   deliberately changed, pin stale) or a production bug (test pins intended
   behavior); find the introducing commit; fix on the correct side. For
   production-side fixes the existing red test is the born-red evidence; for
   test-side re-pins, justify the new pin against the deliberate-change commit.
3. Item 3 (coalescing flake): rerun several times; root-cause the intermittency
   if reproducible, else document the trigger condition.
4. Any item whose fix would be large/risky (real production bug needing design)
   is documented in the notes and flagged for its own task, not forced.
5. Verify: full runs of every touched test file at the end; ruff check + format
   on touched files only.

## Acceptance Criteria

- [x] Each of the seven tests above is diagnosed to its causing change (commit
      or drifted contract) and either fixed or the assertion updated to match
      current, intended behavior
- [x] Item 3 (coalescing flake) is either stabilized (root cause of the
      intermittency identified and fixed) or documented as a known flake with
      its trigger condition, not silently left red
- [x] All seven pass on dev; no other test in the same files regresses

## Implementation Notes

**Outcome: all six deterministic items were already green at this branch's
HEAD (`48ad9e7de`, a dev merge) — each resolved elsewhere between this task's
2026-08-13 baseline (`6b57458b8`) and now.** Nothing was re-fixed; each item
was re-run in isolation first (all green), then attributed to its fixing
commit by diffing the exact failing assertion/stub. The only change this task
ships is the item-3 known-flake documentation block in the coalescing test's
module docstring.

Per-item attribution (fixing commit verified by diff content, not message):

1. **collections-panel empty-state** — resolved by `288b2a6c6` (2026-08-14,
   "test: align Collections empty-state copy"): rewrote the exact failing
   sentence-count assertion to the current `#library-collections-empty` /
   `state.empty_copy` contract (its own task file rode in the commit).
2. **TTS profile capabilities (3/43)** — resolved by `2daf941a3` ("feat:
   verify OpenAI-compatible voice profiles from samples"): at the baseline,
   `_ProfileRepositoryProtocol` (runtime_checkable, `profile_service.py`)
   already required `create_profile_with_reference`/`get_reference` but the
   test's `_AvailabilityRepository` stub lacked both, failing the structural
   `isinstance`; that commit added exactly those two stub methods. 43/43 now.
3. **control-bar coalescing flake** — not reproducible at HEAD: 18/18
   file-runs green across unloaded (12), 10-process CPU-loaded at 2.5x
   slowdown (3), and default-plugin/random-order (3) conditions, while
   `git log -S` shows the coalescer, the `_console_sync_requested` follow-up
   re-dispatch, and the mount-time scheduling all UNCHANGED since the
   2026-08-10 observation commits (task-3070 notes, 48a54ed9c/762596846).
   Documented as a known flake in the test file's docstring with the
   identified trigger condition: both tests settle on a fixed pause-count
   heuristic, and a still-in-flight `_sync_native_console_chat_ui` yields an
   extra `_sync_console_control_bar` execution via the finally-block
   re-dispatch (breaching test 1's `<= 6` bound) or a trailing execution past
   test 2's `settled` capture (breaking its exactly-one assertion).
4. **left-rail sections tray parent** — resolved by `66c84cd10` (2026-08-14,
   "test: align Console integration contracts"): re-pinned the tray's parent
   to `#console-rail-section-body-conversations`, exactly the live failure
   (the task-15110 rail rework `12166b224` is the deliberate-change side).
5. **citation-sources zero-count cache** — resolved by `486998f35`
   (2026-08-13, "refactor: extract Console image controller"): added
   `set_presentation_context=Mock()` to the drifted test double.
6. **shell-chip swap signature** — resolved by `3bcf12e2b` (2026-08-14,
   "test: reconcile current Console contracts"): the test called the method
   UNBOUND on the class with extra positionals; re-pinned to the bound
   `controller._swap_console_session_character(...)` form.
7. **model-catalog QwenCloud drift** — resolved by `00bffe400` (2026-08-14,
   "test: restore Settings module isolation"): added the missing
   `"QwenCloud": True/False` rows to the saved-config test data.

**Verification** (per-file isolation runs, venv python with PYTHONPATH pinned
to this worktree): the six smaller files fully green — 11+43+2+55+10+8 = 129
passed, 0 failed. `test_console_internals_decomposition.py` alone: 137
passed, 4 failed — all four failures are OUTSIDE this task's seven items,
pre-exist this task's (docstring-only) change, and are flagged for their own
task rather than forced here:

- `test_console_staged_context_tray_stays_quiet_when_populated`,
  `test_console_native_control_bar_and_staged_context_reflect_pending_handoff`,
  `test_console_run_inspector_shows_blocked_provider_and_missing_rag_source`
  — deterministic in isolation, one shared signature: 4s timeout in
  `_wait_for_production_chat_screen` with `active=ChatScreen` (the
  `isinstance`/`region.width > 0` gate at line ~207 never satisfies; the
  helper itself predates the baseline — `75244089d`, 2026-07-28).
- `test_console_composer_shows_cursor_when_focused` — passes in isolation,
  failed only in the full-file run: order/isolation-dependent within the file.

A combined one-process run of all seven files also showed cross-file
contamination failures (the known Tests/UI module-isolation issue that
`00bffe400` addresses); per-file runs are the honest gate and are what the
AC is checked against.

**Files changed**: `Tests/UI/test_console_control_bar_coalescing.py`
(docstring-only known-flake documentation) and this task file. Ruff check +
format clean on the touched file.
