---
id: TASK-38977
title: "Console prompts-modal and response-rule run-copy tests pin superseded copy and a stale gateway call, so they are red on dev"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-06 01:42
updated_date: 2026-11-07 22:31
labels:
- follow-up
- tests
- console
- prompts
- response-rules
priority: low
references:
- Tests/UI/test_console_prompts_modal.py
- Tests/UI/test_console_response_rules.py
- tldw_chatbook/Widgets/Console/console_prompts_modal.py
- tldw_chatbook/Prompt_Management/prompt_improvement_service.py
- tldw_chatbook/UI/Console_Modules/agent.py
- tldw_chatbook/UI/Screens/chat_screen.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05.

**What fails.** Five nodes are red 3 of 3 alone at load 12-14. The prompts file, run whole at `-n 4`, gives 4 failed and 121 passed in 20.9 s. No branch due to merge before this one changes these tests or the code they pin. None of the five runs on a PR lane, which is how they drifted.

**Copy drift (two nodes in `Tests/UI/test_console_prompts_modal.py`).**
- `test_no_matches_retry_and_source_unavailable_are_explicit` (`:639`) fails at `:666`: `assert 'source is unavailable' in 'Local is unavailable — its items cannot be shown. Retry or switch source.'`.
- `test_source_switch_clears_foreign_rows_before_unavailable_result` (`:1555`) fails at `:1600`; it expects "Server Prompt source is unavailable".
- The pins date from 8abb970364 (2026-08-02). 378413f0f4 (2026-09-25) changed the copy, at `tldw_chatbook/Widgets/Console/console_prompts_modal.py:1172`.

**Stale gateway stub (two nodes in the same file).**
- `test_real_recipe_fill_with_additional_context_reaches_block_review` (`:3351`) and `test_mapped_context_blocks_recipe_save_until_deleted` (`:3380`) fail at `:446` on `assert gateway.requests`.
- The stub `_OneShotAuxiliaryGateway.complete_auxiliary(self, request)` (`:375`) takes no `route` keyword. Since c78f641adf (2026-08-30), the service calls `complete_auxiliary(request, route=None)` (`prompt_improvement_service.py:452`).
- The resulting TypeError is caught at `:463`, so the outcome reads as a generic `provider_error`.
- TASK-38756 fixed the same drift in the Prompt_Management gateway pin, but not in this UI stub.

**Run-copy drift (one node).**
- `Tests/UI/test_console_response_rules.py::test_active_run_copy_takes_priority_over_an_earlier_rule_verdict` fails at `:297`: `assert 'Waiting for reply…' == 'Streaming response.'`.
- 077e5bfa09 (2026-10-31) moved streaming copy to `chat_screen.py:17512`, which calls `Agent.console_run_feedback_copy` (`agent.py:1167`). Before the first chunk arrives, that returns "Waiting for reply…" (`agent.py:676`).
- The product change was intended, so the pin is stale.

TASK-38536, TASK-38747 and TASK-38703 recorded some of these as pre-existing. None of them owns them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The five nodes pass 3 of 3 runs alone, and both files pass 3 of 3 whole-file runs at -n 4 on dev (red before: 5 of 5, and 4 failed with 121 passed for the prompts file)
- [x] #2 Each updated pin still fails under a temporary edit that breaks what it pins (an unavailable result without source-unavailable copy; a recipe fill that never reaches the gateway; active-run copy losing priority to an earlier rule verdict), with the red runs recorded
- [x] #3 If complete_auxiliary's call shape changes again, the recipe tests fail naming the call mismatch rather than reading it as a generic provider_error
- [x] #4 The five nodes join the UI census with their measured serial times, and their shard stays at least 5 minutes under its cap
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-run the five nodes on base 8b762b52fa to confirm the reds (done: 5 failed, 144 passed for both files at -n 4).
2. Update the two copy pins to the shipped unavailable copy (378413f0f4).
3. Make `_OneShotAuxiliaryGateway.complete_auxiliary` take any call shape and bind it against the real `ConsoleProviderGateway.complete_auxiliary` signature; record a mismatch and have the helper assert no mismatch (naming it) before `assert gateway.requests`.
4. Pin the active-run copy to the pre-first-chunk run copy ("Waiting for reply…", 077e5bfa09) instead of `run_state.visible_copy`.
5. Negative controls by temporary source edits: unavailable branch without the unavailable copy; the service's call shape changed (and a recipe fill that never reaches the gateway); the active-run branch returning the rule verdict.
6. Append the five nodes at the UI census tail with measured serial times; raise MINIMUM_FILES; check their shard's measured time stays at least 5 min under the 20-min cap.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All five reds were stale pins, not product defects. Base 8b762b52fa, 2026-11-07.

**What changed (test-only).**
- Copy: the two unavailable-source pins now read the shipped wording, "Local is unavailable — its items cannot be shown" and "Server is unavailable — its items cannot be shown". 378413f0f4 (draft shelf, 2026-09-25) replaced "<Source> Prompt source is unavailable — its Library cannot be shown".
- Gateway: `_OneShotAuxiliaryGateway.complete_auxiliary(*args, **kwargs)` binds each call against `inspect.signature(ConsoleProviderGateway.complete_auxiliary)` and records any mismatch. `_real_additional_context_recipe_outcome` asserts there is none, naming it, before `assert gateway.requests` (AC#3). The old stub took `(request)` only, and the service has called `complete_auxiliary(request, route=None)` since c78f641adf.
- Run copy: `test_active_run_copy_takes_priority_over_an_earlier_rule_verdict` now pins `"Waiting for reply…"`, the pre-first-chunk run copy that 077e5bfa09 moved to `Agent.console_run_feedback_copy`. The provider is held before any chunk.
- Flake found while laning: the two recipe tests pressed buttons after one `pilot.pause()`. They failed 1-4 of 160 runs at load about 40 (NoMatches on a button the remount had not mounted yet, a stale Save menu, and once the product's `_mount_recipe_editor` raising NoMatches because the Fill press reached a not-yet-mounted button). They now wait on each press's own end condition: the next button mounted, the Fill worker finished (`_improvement_worker.is_finished` with no activation pending), the block delete reaching the modal, and the save notice. After that: 160 of 160 at load 41-44.

**Red/green evidence.**
- Base, both files at `-n 4`: the five nodes red, 5 failed / 144 passed at load 35-40.
- After, the five nodes alone at `-n 4`: 3 of 3 (load 38-44). The two copy nodes and the run-copy node: 90 of 90 repeats (load 40-44).
- Negative controls (AC#2), all temporary edits, run at load 33-39:
  - The unavailable branch made to use the "failed to load" copy: both copy nodes red.
  - `complete_auxiliary` replaced by a raise, a fill that never reaches the gateway: both recipe nodes red at `assert gateway.requests`.
  - The service calling `complete_auxiliary(request, None)`: both recipe nodes red with "the service's complete_auxiliary call does not match the real gateway: … too many positional arguments" (AC#3).
  - The rule verdict returned ahead of the active run's copy: the run-copy node red.

**Whole files (AC#1, second half): not met at this host's load.** Six whole-file runs at `-n 4` and load 42-98 each failed only on tests outside these five, all in `test_console_response_rules.py`:
- `test_fixture_failure_retires_held_transport[primary|helper]`;
- `test_console_syncs_never_wait_for_a_config_rebuild[full-ui-held|full-ui]`;
- `test_console_mount_refreshes_never_wait_for_a_config_rebuild[roleplay|mount]`.

They failed 1-6 per run, plus `test_compact_recipe_working_copy_scrolls_to_the_block_editor` and `test_leaving_saved_recipe_chooser_clears_selection_intent[escape]` once each. Base at load 52-75 failed the same four response-rules tests (9 failed / 140 passed: those four plus the five reds). These are load-sensitive tests already on dev, several of them in the UI census. The five nodes passed in every run. AC#1 stays unticked until both files run 3 of 3 on a quieter host.

**Lane (AC#4).** The five nodes are appended at the UI census tail, one entry per test (0.6-9.1 s each serially at load about 27; 13-17 s for all five). They land on shards 18-22 at the current census length. The slowest of the last 12 PR runs on those shards was 14.15, 14.4, 13.72, 13.95 and 13.25 min (collected 2026-11-07 06:05Z), so each stays more than 5 min under the 20-min cap.

**Files.** `Tests/UI/test_console_prompts_modal.py`, `Tests/UI/test_console_response_rules.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
<!-- SECTION:NOTES:END -->
