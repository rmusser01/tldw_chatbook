---
id: TASK-34667
title: Remove dead screen-level chat appliers overwritten by ensure_chat_controller
status: Done
created_date: 2026-10-09 06:27
updated_date: 2026-10-10 04:52
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from TASK-34435 review: chat_screen.py:14217/14241 defines dictionary/world-info appliers passed into runtime.ensure_chat_controller (chat_screen.py:9793-9794) which unconditionally overwrites both via kwargs.update (console_runtime.py:4286-4292) - dead on every path. Drift precedent: the _library_provider_for_app docstring (console_runtime.py:662-678) records a stale always-overwritten copy that cost 24 Library tools behind one swallowed warning. Delete the dead appliers (or wire them for real) with caller-grep evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Caller-grep evidence pasted,Dead appliers removed or genuinely wired,Console tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See .superpowers/sdd/nonconsole-followups/task-34435-report.md review section
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Approach: removed the dead screen-level appliers outright rather than wiring
  them, exactly as tasked. `ensure_chat_controller`'s overwrite is the protective
  direction (runtime-owned seams) and was left in place.
- Re-verified at tip (a190654d4c): `ChatScreen._console_chat_dictionary_applier`
  (chat_screen.py ~14227) and `ChatScreen._console_world_info_applier` (~14251)
  were passed at the `runtime.ensure_chat_controller` call site (~9803-9804) and
  unconditionally overwritten by `kwargs.update` (console_runtime.py ~4286-4292).
  Drift was already real: the screen world-info copy called
  `world_info_resolver.apply_world_info_to_message` with a 3-argument signature,
  while the live call site (console_chat_controller.py ~22448-22454) passes a
  4th `frozen_inputs` argument -- the screen copy could not have run at all.
  Caller-grep (production + tests) pasted in
  .superpowers/sdd/task-34667/task-34667-report.md.
- Removed: both applier method definitions, the two pass-through kwargs, and the
  `_CHATDICT_MAX_TOKENS`/`_CHATDICT_STRATEGY` constants (sole consumer was the
  dead dictionary copy).
- Guard added (the trivial in-scope extra): `ensure_chat_controller` now hoists
  its overwrite literal into a `runtime_owned` dict and emits one `logger.debug`
  naming any caller-supplied always-overwritten kwargs before updating. Keys are
  computed from the same dict that overwrites them, so the log can never drift
  from reality.
- Tests: absence pin `test_console_screen_level_applier_copies_stay_removed` in
  Tests/Chat/test_console_prompt_transform_single_collection.py (34433/34437
  style, rationale docstring cites the `_library_provider_for_app` drift
  precedent). Two tests were pinning dead behavior and were updated with
  justification: the module docstrings in the two send-integration files named
  the dead wiring (rewritten to name the live runtime-owned seam), and
  `test_console_world_info_applier_honors_enable_world_info_setting` had bound
  the dead screen method directly -- retargeted to the live
  `_apply_world_info_for_app`, which previously had no unit-level gate coverage.
  Its disabled-config end-to-end sibling's monkeypatch was also retargeted from
  `chat_screen_module.get_cli_setting` (only reachable by the dead copy) to
  `tldw_chatbook.config.get_cli_setting` (the call-time import both live readers
  use).
- Verification: every ensure_chat_controller/applier-touching suite matches its
  pristine-HEAD baseline (re-run in throwaway worktrees at a190654d4c);
  remaining failures are the known sandbox `RecoveryRequired`/environmental set,
  identical before and after. Ruff: no new findings in changed files.
- Modified files: tldw_chatbook/UI/Screens/chat_screen.py,
  tldw_chatbook/Chat/console_runtime.py,
  Tests/UI/test_console_dictionary_send_integration.py,
  Tests/UI/test_console_world_info_send_integration.py,
  Tests/Chat/test_console_prompt_transform_single_collection.py.
- ADR: not required (mechanical dead-code removal preserving the existing
  task-15860 runtime-custody boundary).
- Follow-up observation (out of scope): the same `kwargs.update` also overwrites
  six more kwargs the screen still passes (rag_capture_provider,
  default_session_settings, library_provider_factory, global_user_display_name,
  turn_context_provider, provider_config). Unlike the appliers their referenced
  callables are not dead code (they have other consumers), so they were left for
  a dedicated cleanup task; the new debug log now names them on every
  production ensure call.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the two dead screen-level Console chat appliers (definitions, call-site
pass-through, and their bounds constants) that `ensure_chat_controller`
unconditionally overwrote on every path; added a drift-proof `logger.debug`
guard naming any always-overwritten caller kwarg; retargeted the two tests that
pinned the dead wiring to the live runtime-owned seam; added a 34433/34437-style
absence pin. All targeted suites match their pristine-HEAD baselines.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
