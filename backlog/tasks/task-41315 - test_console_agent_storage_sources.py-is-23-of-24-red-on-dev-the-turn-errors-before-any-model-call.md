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
- [ ] #1 The task notes name the cause of the primary run's first-step error and of the missing fixture definition, from the tests' own tracebacks
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
