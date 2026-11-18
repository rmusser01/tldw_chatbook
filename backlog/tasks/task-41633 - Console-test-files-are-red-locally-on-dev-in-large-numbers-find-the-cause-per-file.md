---
id: TASK-41633
title: "Console test files are red locally on dev in large numbers; find the cause per file"
status: In Progress
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- tests
- console
priority: medium
references:
- Tests/UI/test_console_internals_decomposition.py
- Tests/UI/test_console_send_draft_snapshot.py
- Tests/UI/test_console_workbench_contract.py
- Tests/UI/test_console_native_transcript.py
- Tests/Chat/test_change_turn_tracking.py
- Tests/UI/test_console_right_rail.py
- Tests/UI/test_console_rail_reconciliation.py
- Tests/Chat/test_console_async_mcp_snapshot.py
- Tests/MCP/test_console_snapshot_source_contracts.py
dependencies:
- TASK-41308
- TASK-41313
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** The burn-down's local runs (macOS, isolated profile through the scratch wrapper) found whole Console test files red on dev. These files are not on the PR lanes that ran, so nothing flags them. TASK-41313 fixed five such files: they failed behind the `raw_source_selection_changed` config-sandbox trap, and adding the `bootstrap_profile` marker (TASK-38111) cleared them. Some of the files below already carry that marker, so the trap is not the whole story.

**Red counts.** Figures from runs on dev 45b1949ca3 or 5e3c8b2bd4:
- `Tests/UI/test_console_internals_decomposition.py`: 122 (has the marker)
- `Tests/UI/test_console_native_transcript.py`: 110 (has the marker)
- `test_console_session_settings.py`: 97 (TASK-41308's run)
- `Tests/Chat/test_change_turn_tracking.py`: 46 (no marker)
- `Tests/UI/test_console_right_rail.py`: 33
- `Tests/Chat/test_console_async_mcp_snapshot.py`: 32
- `Tests/UI/test_console_rail_reconciliation.py`: 27
- `Tests/UI/test_console_send_draft_snapshot.py`: 20 of 33 (has the marker)
- `Tests/MCP/test_console_snapshot_source_contracts.py`: 16
- `Tests/UI/test_console_workbench_contract.py`: 12
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each listed file has a recorded cause in this task's notes: the marker trap, test drift, or a product defect
- [x] #2 Files red only behind the config-sandbox trap pass with the `bootstrap_profile` marker
- [ ] #3 Every other cause is fixed here or filed as its own task with the failing node ids
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (test-only fixes and cause records; no product change)

1. Re-measure the ten files on dev d0486978dd, one file per run at `-n 4`, and categorize every failure by its first `E` line.
2. Run the files with a throwaway plugin that adds `bootstrap_profile` to every item, to separate the config-sandbox trap from other causes.
3. Add a module `pytestmark` to each file whose reds are the trap, removing the then-redundant per-test markers.
4. For the rest, name the commit that changed the behaviour; fix test drift that is small; collect the remaining node ids as follow-ups for the coordinator to file.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Method.** Each file was run alone at `-n 4` through the scratch-profile wrapper on a detached dev worktree (d0486978dd) and on this branch, at load 60-105 on 18 CPUs. Every failure was grouped by its first `E` line. A throwaway `-p` plugin that adds `bootstrap_profile` to every item showed which reds the config-sandbox trap explained before any marker was added; it was deleted afterwards. Failures that passed when rerun alone are marked as load flakes.

**Finding on "already carries the marker".** Only `test_console_send_draft_snapshot.py`, `test_console_rail_reconciliation.py` and `test_console_workbench_contract.py` had a module `pytestmark`. In the other files the marker was on a few tests: 8 of 142 in internals_decomposition, 1 of 168 in native_transcript, 5 of 452 in session_settings and 1 of 37 in right_rail. Their reds were mostly the trap.

**Per file** (base = dev d0486978dd, head = this branch; failed/passed at `-n 4`):
- `Tests/UI/test_console_right_rail.py`: base 33/4, head 0/37. **Trap**: all 33 were `raw_source_selection_changed`; a module marker replaces the one per-test marker. Behind it, one **drift**: `test_clicking_open_then_collapse_toggles_visibility_and_persists` expected the screen's session controller to own the two project-instruction callbacks, which the Console runtime has owned since b7dd8e53f2 (TASK-34667 overwrites any caller copy); it now checks the runtime.
- `Tests/Chat/test_change_turn_tracking.py`: base 46/58, head 2/102. **Trap** (33 RR; the other 13 base reds pass with the marker too). Of the two left, `test_cancelled_turn_still_schedules_coordinated_finalization` passes alone (load flake), and `test_review_runs_before_baseline_gate_and_tool_dispatch_waits` fails alone too: the baseline gate now finishes before the permission review (`['baseline-finished', 'review-called', 'tool-finished']`). Follow-up.
- `Tests/UI/test_console_rail_reconciliation.py`: base 1/53, head 2/52 under load; both head reds pass alone. **Drift**: `test_chat_screen_inspector_mutation_paths_delegate_one_owner_request[settings]` patched `_build_console_settings_summary_state` on the screen, which 3e618d7119 moved to `screen._context_spend`; it now patches the owner.
- `Tests/UI/test_console_workbench_contract.py`: base 13/62, head 6/70 (one parametrized case added). **Drift**, fixed: the composer menu's Prompts row reads "Prompts · 0 mix-ins…" since 7970fbcada (3 nodes); `_console_empty_recovery_action_copy` stopped parsing the blocker sentence in 939dee8dc2 (#2365), so the four provider-specific cases now expect "Review settings" and an empty-sentence case pins "Choose model" (4 nodes); the status line is a markup string since c5ceca1247, read as rendered text (1 node). Six reds remain; follow-up.
- `Tests/UI/test_console_send_draft_snapshot.py`: base 21/13, unchanged. Not the trap (module marker present). **Product-path change, not fixed**: the probed sends fail at `_admit_console_turn_to_runtime` with `RuntimeError: Console preparation source unavailable; Send again.`, raised by `turn_admission.capture_authority` when a managed `ConsoleRuntime` exists and no precaptured authority is set (added by 11d71e5db5). A probe on `wiring.py`'s `ui_submit` failure branch caught that error in 6 of the 21 tests. The other 15 fail without reaching that branch and were not traced. Follow-up.
- `Tests/MCP/test_console_snapshot_source_contracts.py` (base 16/34) and `Tests/Chat/test_console_async_mcp_snapshot.py` (base 32/32): unchanged. Not the trap. The first imports the second's controls. Causes: private names the product no longer has (`ConsoleChatController._compose_local_provider_async`, never in dev's product code; `MCPToolProvider._freeze_definition_maximum_on_compose`), the same preparation-source refusal, `RecoveryRequired: console_snapshot_owner_changed`, and off-loop assertions. TASK-36161 already tracks the async file (26 reds on 2026-10-22, 32 now); follow-up joins the contracts file to it.
- `Tests/UI/test_console_internals_decomposition.py`: base 122/20, head 12/130 at `-n 4`; serially 7 of those 12 fail, and the five RAG-action tests pass alone. **Trap**: 120 RR. The module marker replaces the 8 per-test markers on the MCP rows. **Drift** fixed behind it:
  - The CSS pin test now reads `features/_console.tcss` and `features/_console_panels.tcss` with design tokens. 2aab79b0fd carved these rules out of `_agentic_terminal.tcss`, and bc1c716e43 (ADR-161) tokenized them.
  - Evidence rows render literally since a2faa20c96 (TASK-32802.4), so the test no longer expects HTML entities.
  - An empty staged tray counts "none" (77b773f7e6).
  - The control bar says "Context rail" (14273c1f17).
  - `_wait_for_production_chat_screen` waits 30 s, not 10 s. At load 70-90 the full app was still starting at 10 s and settled shortly after; 5 of the 7 timeouts then passed.
  Seven reds remain as follow-ups:
  - The tab strip lacks `console-session-tab-strip` and gets a 1-row margin, so the empty panel and header sit one row lower than pinned (3 nodes).
  - `Approvals: 0 pending` where 1 was staged.
  - An empty staged-tray body when sources are staged.
  - The composer cursor-glyph tests read before the focus paint (`shows_cursor`, `hides_cursor`, `cursor_blink`). Each has failed in one run and passed in another, so treat them as timing.
- `Tests/UI/test_console_native_transcript.py`: base 110/58, head 13/155. **Trap**: 69 RR, plus 25 `NoMatches '#console-message-m2'` and the empty-state lookups, which the marker also cleared. The module marker replaces the one per-test marker. Two **drift** fixes behind it:
  - `test_console_transcript_empty_state_is_centered_in_stylesheets` read `_agentic_terminal.tcss`, whose Console rules 2aab79b0fd moved to `features/_console.tcss`.
  - `test_console_transcript_click_action_row_background_preserves_selection` passed a `SimpleNamespace` click, but the transcript input fence (95ca4bf47b) reads the event's `_sender` and `time`, so it now passes a real `textual.events.Click`.
  `test_activity_header_click_and_o_share_expansion_state` failed once under load and passed alone. The other 12 reds are follow-ups.
- `Tests/UI/test_console_session_settings.py`: base 98/354, head 19 failed + 8 errors / 433 passed. **Trap** (87 RR); module marker added, the 5 per-test markers removed. What remains:
  - `test_console_readiness_unblocks_after_provider_save_without_restart` still hits `raw_source_selection_changed`. It redirects `TLDW_CONFIG_PATH` and force-reloads config itself, the TASK-38111 wizard shape that the marker cannot fix.
  - Eight nodes now error at teardown with `Native app workers remain uncertain; config retirement refused` (`Tests/conftest.py`). Two of them also show `thread constructor failed`, which the roleplay-writer test injects. All eight were red on base behind the trap. Under the shared bootstrap profile, the guard refuses to hand config back while the test app's native workers have not retired.
  - About 15 other assertion failures. All were red on base, behind the trap or as base's own non-trap reds.
  - One new red: `test_console_discovery_identity_is_canonical_and_each_request_is_monotonic` passes on base and passes alone with the marker. It fails at `-n 4` and in a serial batch after the other reds, so it depends on test order under the shared profile.
  - Serially, 4 of the 27 passed: `test_cancelled_unmounted_drain_finishes_latest_plan`, `test_console_workspace_conversation_row_switches_native_tab`, `test_mounted_console_cancel_latest_waiter_keeps_durable_c` and `test_roleplay_repair_marker_retries_partial_then_consumes`.
  All of these are follow-ups.

**AC #3.** Every cause this task could fix in the tests is fixed. The coordinator files the rest as follow-ups from the stream report, with the node ids named above. This stream may not create task ids. AC #3 stays open until they are filed.

**Lanes.** No lane file was edited. The new module markers make three census node ids in `test_console_session_settings.py` `bootstrap_profile`. They still pass `scripts/check_ui_pr_gate_census.py`, but TASK-32873's rule for such nodes is for the coordinator to apply. None of the ten files is newly green as a whole except `test_console_right_rail.py` (37 passed at `-n 4` in 188 s at load 86).

Files: `Tests/UI/test_console_internals_decomposition.py`, `Tests/UI/test_console_native_transcript.py`, `Tests/UI/test_console_session_settings.py`, `Tests/UI/test_console_right_rail.py`, `Tests/UI/test_console_rail_reconciliation.py`, `Tests/UI/test_console_workbench_contract.py`, `Tests/UI/test_console_scope_row.py` (marker; TASK-41632's file, 40 trap reds, 1 left), `Tests/Chat/test_change_turn_tracking.py`, `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
