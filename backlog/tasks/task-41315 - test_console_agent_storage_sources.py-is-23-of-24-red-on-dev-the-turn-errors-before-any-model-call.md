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
- [ ] #2 Every node passes on dev at `-n 4`, or is retargeted or retired with the reason recorded; any product defect found is fixed with a test that is red first
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

Outcome: stopped for decisions. One product defect is fixed (red first). The remaining 18 red nodes are classified below from their own failure text. None of them is the `continuation_protocol` lead, which was a third cause hidden behind the first.

**Cause of the primary run's first-step error (AC #1).** The run's `error` step, read from AgentRuns, is `unexpected provider error (Command event activation source changed.)`. It is raised by `_CommandEvent.refresh_permissions` at turn start (`_phase_turn_start` calls `admit_input`, which calls `context.refresh_sources`). The MCP context event captured `_guard.captured_sources(None)` inside the submit's `_AgentSources.activate()` on the loop. That tuple starts with the fresh profile observation (config, the `db.agent_runs` selector, workspaces, sandbox), then the bridge's sources. The turn's worker chain starts with `worker_guard(bridge)`'s observation (bridge order), and `activate()` nests inside it. The two hold the same set of `(owner, path)` sources in a different order, and they were compared as tuples. This is a product defect: any Console agent turn with a returned-context MCP hook was refused before its first model call.

**Fix.** `Agents/activation.sources_changed(captured, service=None)` compares the admitted sources as a set; `_CommandEvent.check`/`refresh_permissions` use it. Adding, removing or moving a source still refuses. Red first: `[none-1]` failed with this error on the base tree and again after the shared-double fix, and it passes with the fix. New unit test: `test_reordered_admission_is_not_a_source_change`. The test's own ordered assertion (`model_sources == [...] * 3`) now compares sets.

Not changed: the same ordered comparison in `legacy_hook_ownership.py:205`, `console_plugin_hook_preparation.py:177`, `console_reference_preparation.py:312` and four sites in `console_workflow_runs.py`. No failing test shows them.

**Cause of the missing `local:fixture::fixture` definition.** `capture_turn_configuration_snapshot` returns `mcp_capture_pending=True` with an empty maximum. The fixture's MCP service uses `manifest_provider=dict`, which is not the stock manifest, so `_mcp_metadata_is_closed` is False. The rig also has a managed Console runtime (`_has_managed_console_runtime`), so the synchronous capture defers (11d71e5db5). `_capture_mcp_definition_selection(app, _admitted=True)` does return the definition. Whether to retarget this node to the deferred/async capture or give the fixture a stock manifest is a decision.

**Remaining nodes after the fix** (load 64-117):
- Pass (6 + unit test): `none-1`, `none-2`, `logging_disabled`, `admission_refused`, `mcp_deny`, `unrelated-2`.
- `workspace`, `workspace_distinct_registry`, `workspace_disable`, `workspace_switch`, `workspace_retarget`, `workspace_custom_reader` (6): the primary run is `cancelled` before any step and no project notice is shown. This is TASK-41312 (project-instruction dispatch cancels before its confirm callback).
- `writer`, `writer_segment_path`, `writer_write_bytes`, `writer_manifest`, `writer_close` (5): the replaced writer method runs. This was confirmed alone for `writer_manifest`, which wrote "changed writer invoked" into the user file, and for `writer`, where the child ran to `done`. The other three failed in the `-n 4` run and were not probed alone. 06f8b0bf94 (TASK-39147, ADR-356: "callback/code identity is not an authority check") removed `_check_writer`'s method-identity check. Candidates to retire, which needs a decision.
- `root`, `log_dir`, `writer_source`, `bound_directory`, `database`, `late_permission` (6): the changed source is refused ("original agent log writer changed" / "...log binding changed" / "...storage source changed"). But the `RuntimeError` escapes `submit_draft` to the caller, the primary row stays `running`, and the fixture teardown then hangs (custody never retires; pytest-timeout did not fire). Product, needs a decision.
- `test_original_mcp_capture_retains_structured_names_with_definition_hashes`: see above.

Before the fix, all of these were masked: every turn was refused at its first step, so no mutation ran.

**AC #3.** Not ticked. The file should not join a lane until the hangs above are resolved: a hung teardown would hold a lane worker. After that it belongs with `pr-fast-lane-admission`.

Files: `tldw_chatbook/Agents/activation.py`, `tldw_chatbook/Agents/hooks_v2/native_commands.py`, `Tests/Chat/test_console_agent_storage_sources.py` (shared double: see TASK-41316).
