---
id: TASK-34667
title: Remove dead screen-level chat appliers overwritten by ensure_chat_controller
status: Done
assignee:
  - '@codex'
created_date: '2026-10-09 06:27'
updated_date: '2026-10-10 15:18'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from TASK-34435 review: chat_screen.py:14217/14241 defines dictionary/world-info appliers passed into runtime.ensure_chat_controller (chat_screen.py:9793-9794) which unconditionally overwrites both via kwargs.update (console_runtime.py:4286-4292) - dead on every path. Drift precedent: the _library_provider_for_app docstring (console_runtime.py:662-678) records a stale always-overwritten copy that cost 24 Library tools behind one swallowed warning. Delete the dead appliers (or wire them for real) with caller-grep evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Caller-grep evidence pasted,Dead appliers removed or genuinely wired,Console tests pass
- [x] #2 The diagnostic inventory reproduces after review of the new debug statement, and the affected Console send tests reach their assertions under supported profile isolation.
- [x] #3 The override diagnostic names only ignored runtime-owned keys, stays silent without overlap, and preserves both live applier bindings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Remove the two always-overwritten screen appliers and their sole-consumer bounds constants; preserve the runtime-owned appliers and verify their real send paths.

PR #3061 integration follow-up:
1. Rebase on latest dev and review exact range independently.
2. Review the added debug statement for persistent-sink safety and regenerate its diagnostic census.
3. Resolve verified review findings and qualify the affected Console send tests under supported profile isolation.
4. Run targeted tests and lint, verify required CI, then merge and clean up the isolated checkout.
ADR required: no
ADR path: N/A (existing runtime custody and ADR-126 profile-selection rules apply)
Reason: mechanical dead-code removal and test/census repair preserve existing ownership and admission contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Approach: removed the dead screen-level appliers outright rather than wiring
  them, exactly as tasked. `ensure_chat_controller`'s overwrite is the protective
  direction (runtime-owned seams) and was left in place.
- Re-verified at tip (a190654d4c): `ChatScreen._console_chat_dictionary_applier`
  (chat_screen.py ~14227) and `ChatScreen._console_world_info_applier` (~14251)
  were passed at the `runtime.ensure_chat_controller` call site (~9803-9804) and
  unconditionally overwritten by `kwargs.update` (console_runtime.py ~4286-4292).
  Drift was already real: the screen world-info callback accepted three
  arguments, while the live controller passes a fourth `frozen_inputs`
  argument -- the screen callback could not serve that call shape.
  Committed caller evidence: `rg -n "_console_chat_dictionary_applier|_console_world_info_applier|_CHATDICT_MAX_TOKENS|_CHATDICT_STRATEGY" tldw_chatbook Tests` now finds only regression-test assertions and historical test documentation; no production caller or definition remains. `rg -n "ensure_chat_controller\(" tldw_chatbook Tests/Chat Tests/UI/test_console_runtime_ownership.py` locates the live runtime binding and screen caller. The runtime still overwrites the dictionary/world-info kwargs with its app-bound `_apply_*_for_app` functions.
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

PR #3061 review: the new debug statement interpolates only sorted keys drawn from the fixed runtime_owned mapping (no caller values, user content, secrets, paths or URLs). Statement review found one added debug call and no moved/removed calls or sink-topology changes; regenerated the sole console_runtime.py inventory row (40 to 41 calls). Independent review also requested positive/no-overlap guard coverage, committed caller evidence instead of absent scratch reports, and correction of the three-argument SCREEN callback explanation. Mounted send qualification exposed per-test profile selection drift and a stale shared provider double lacking cached_context_window; repaired with the existing private_profile_test helper and pure resolve_context_window.

Final targeted evidence on PR #3061: 10 passed across dictionary/world-info mounted send integration, single collection/frozen edit, removal pin and both diagnostic cases. Removing only the debug branch via a temporary late-collection pytest plugin yields the expected overlap-case assertion failure (1 failed, 1 passed), with no production file edits. Wider runtime construction/ownership run: 103 passed, 81 deselected; three unchanged lifetime tests failed (stream-start timeout, outdated wake-exemption exception expectation, and config profile selection). All three reproduce with the same failure reasons on an exact git archive of dev 0c3ebc6ed76e5753385897c958f80b1c282abaa3; they are unrelated to the deleted appliers/logging change. Full Ruff and formatting pass on the three changed test files; critical Ruff checks pass on both changed production files; git diff --check and diagnostic reproduction pass. Independent review follow-up cleared all four findings. No full suite or live provider traffic was run.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed unreachable screen appliers while preserving runtime-owned frozen transforms; added tested key-only override diagnostics and regenerated the reviewed census. Repaired affected send-test profile/gateway fixtures. All 10 affected behavior checks pass; 103 wider runtime checks pass, with three verified pre-existing dev lifetime failures recorded. Independent review findings are resolved.
<!-- SECTION:FINAL_SUMMARY:END -->
