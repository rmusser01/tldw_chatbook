---
id: TASK-38977
title: "Console prompts-modal and response-rule run-copy tests pin superseded copy and a stale gateway call and are red on dev"
status: To Do
created_date: 2026-11-06 01:42
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
- [ ] #2 Each updated pin still fails under a temporary edit that breaks what it pins (an unavailable result without source-unavailable copy; a recipe fill that never reaches the gateway; active-run copy losing priority to an earlier rule verdict), with the red runs recorded
- [ ] #3 If complete_auxiliary's call shape changes again, the recipe tests fail naming the call mismatch rather than reading it as a generic provider_error
- [ ] #4 The five nodes join the UI census with their measured serial times, and their shard stays at least 5 minutes under its cap
<!-- AC:END -->
