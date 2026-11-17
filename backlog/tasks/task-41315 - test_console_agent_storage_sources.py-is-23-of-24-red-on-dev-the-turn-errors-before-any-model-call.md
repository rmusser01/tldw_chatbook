---
id: TASK-41315
title: "test_console_agent_storage_sources.py is 23 of 24 red on dev: the turn errors before any model call"
status: In Progress
created_date: 2026-11-16 19:55
labels:
- follow-up
- tests
- console
- mcp
priority: medium
references:
- Tests/Chat/test_console_agent_storage_sources.py
- Tests/Chat/test_console_skill_substitution.py
- Tests/console_provider_doubles.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the 2026-11-13..16 burn-down, which saw this file at 23 failing. Re-run on dev 5e3c8b2bd4 (2026-11-16, load 33-45, scratch-profile wrapper, `-n 4`): 23 failed, 1 passed. Line numbers are that tree's.

**Why.** This file proves that a Console agent turn's original storage and MCP sources (execution log, run-log writer, workspace bindings, permission store) reach the child model unchanged. A changed source must be refused without effects. With 23 of 24 nodes red, none of that is checked, and the file runs on no PR lane (it is not in `scripts/ui_pr_gate_census.txt` or a workflow target). The file already has its `bootstrap_profile` marker, so this is not the config-sandbox trap.

**Failures.**
- `test_original_console_mcp_context_reaches_child_model`, 22 params (`none`, `unrelated`, `workspace*`, `writer*`, `root`, `log_dir`, `database`, `mcp_deny`, `late_permission`, …): the gateway records too few payloads, e.g. `assert 0 == 3` at `:405` for `[none-1]`. Other params fail the `:398` count check. For `[none-1]`, the log shows the primary run's first step is an error (`agent run step kind=error`), then "Console submit requires attention | accepted=True status=failed". No model call is made.
- `test_original_mcp_capture_retains_structured_names_with_definition_hashes`: `KeyError: 'local:fixture::fixture'` at `:468`. The fixture tool is missing from the captured `mcp_definition_maximum`.
- Lead, not confirmed: the file's gateway subclasses `Tests/Chat/test_console_skill_substitution.py`'s `_RecordingGateway`. Its resolution (`Tests/console_provider_doubles.provider_resolution`) has no `continuation_protocol`. TASK-40715's notes say child runs in these hosts need one. TASK-41028's ordinary-tree file uses the same double.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The task notes name the cause of the primary run's first-step error and of the missing fixture definition, from the tests' own tracebacks
- [x] #2 Every node passes on dev at `-n 4`, or is retargeted or retired with the reason recorded; any product defect found is fixed with a test that is red first
- [ ] #3 The file runs on a PR lane, or the task notes record why it should not
<!-- AC:END -->

## Implementation Plan

ADR required: no. Test-double drift plus one comparison bug in existing agent-source admission; no new policy or interface.

1. Reproduce `[none-1]` and read the primary run's error step from AgentRuns (not the log line).
2. Fix the shared provider double (`Tests/console_provider_doubles.provider_resolution`) once, if a child run needs fields the real resolution has.
3. If the primary error is a product defect, keep `[none-1]` as the red test, fix at the shared comparison, add a focused unit test, mutation-check both.
4. Classify every remaining node (drift, product, retire) from its own failure text; record the cause of the missing `local:fixture::fixture` definition.
5. Covering files on base and head; preflight; Tests/Architecture.

## Implementation Notes

Outcome: all 22 nodes pass at `-n 4`. AC #3 (PR lane) is left for the lane PR.

**Cause of the primary run's first-step error (AC #1).** The run's `error` step, read from AgentRuns, is `unexpected provider error (Command event activation source changed.)`. It is raised by `_CommandEvent.refresh_permissions` at turn start. The MCP context event captured its sources inside the submit's `_AgentSources.activate()` on the loop (profile observation first, then the bridge's). The turn's worker chain starts with `worker_guard(bridge)` (bridge order). Same set of sources, different order, and they were compared as tuples. Fix: `Agents/activation.sources_changed()` compares them as a set, so any added, removed or moved source still refuses. `_CommandEvent.check`/`refresh_permissions` use it. Red first: `[none-1]`, plus the unit test `test_reordered_admission_is_not_a_source_change`; both fail again with the ordered comparison.

**Cause of the missing `local:fixture::fixture` definition (AC #1).** The fixture's MCP catalog uses `manifest_provider=dict`, which is not the stock manifest. Under a managed Console runtime, `capture_turn_configuration_snapshot` therefore defers (`mcp_capture_pending=True`, empty maximum; 11d71e5db5). Retargeted, per the 2026-11-17 decision, to `test_original_mcp_capture_defers_a_non_stock_catalog`. It asserts what the deferred capture guarantees: pending, no definition hash and no structured name pinned, not complete. The send treats a pending capture as opaque.

**Changed source at turn finish (product defect, fixed).** Unmasked by the first fix. The changed source is refused, and the run row is terminal (`error`); my earlier note that it "stays running" came from reads taken mid-persist. The defect was elsewhere. `OrdinaryTurn._finish_primary_if_returned` entered `source.activate()`, which raised before `host_returned`/`finish_root`. The `RuntimeError` escaped `submit_draft` (skipping `_unregister_submit_task`). The turn stayed sealed with a live primary that was never host-returned, so `unproven_only()` was False and disposal waited forever. Fixed at the shared seam: the new `OrdinaryTurn.return_host(owner, release)` settles a refused owner unproven, which means its own durability read is refused too. The claim and scratch stay held (TASK-40716) and disposal leaves the turn. The primary finish and the child `_finish_execution` both use it; the child re-raises so its worker still records unproven cleanup. Red first: `test_changed_source_at_finish_leaves_the_primary_returned_unproven` and `test_changed_source_at_child_return_leaves_the_child_unproven`; both fail when the helper re-raises without settling. End to end, `root`, `log_dir`, `writer_source`, `bound_directory`, `database` and `late_permission` now pass with no teardown hang.

**Retired (owner decision 2026-11-17, same as TASK-41314):** `writer`, `writer_segment_path`, `writer_write_bytes`, `writer_manifest`, `writer_close`. They replaced writer methods and expected a refusal based on implementation identity, which ADR-356 decision 4 dropped (06f8b0bf94, TASK-39147).

**`workspace*` (6 nodes):** these were cancelled before their first step. Cherry-picked TASK-41312's harness fix 3ff4b82d8d (`_RecordingGateway` gains `execution_key`; that branch's TASK-41312 task-file hunk was left out). Now 6/6 pass. The double's explicit `continuation_protocol`/`api_mode` match the shared `provider_resolution` defaults.

**Left unchanged (follow-up):** the same ordered source comparison at `Agents/legacy_hook_ownership.py:205`, `Chat/console_plugin_hook_preparation.py:177`, `Chat/console_reference_preparation.py:312`, and `Chat/console_workflow_runs.py:241, 390, 943, 2262`.

**AC #3.** Not ticked, because this task does not edit workflows. `Tests/Chat/test_console_agent_storage_sources.py` should join `pr-fast-lane-admission` ("Run admission-sensitive suites"). It finishes in about 2 minutes at `-n 4` with no hang.

Files: `tldw_chatbook/Agents/activation.py`, `tldw_chatbook/Agents/hooks_v2/native_commands.py`, `tldw_chatbook/Agents/ordinary_tree.py`, `tldw_chatbook/Agents/agent_native_execution.py`, `Tests/Chat/test_console_agent_storage_sources.py`, `Tests/Chat/test_console_skill_substitution.py` (cherry-pick), `Tests/Chat/test_console_ordinary_tree_widening.py` (cherry-pick).
