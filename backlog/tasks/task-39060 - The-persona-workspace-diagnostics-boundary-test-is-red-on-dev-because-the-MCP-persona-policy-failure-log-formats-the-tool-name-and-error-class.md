---
id: TASK-39060
title: "The persona workspace diagnostics boundary test is red on dev because the MCP persona-policy failure log formats the tool name and error class"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 12:31
updated_date: 2026-11-06 23:20
labels:
- follow-up
- logging
- tests
- mcp
priority: medium
references:
- Tests/test_persistent_diagnostic_boundary.py
- tldw_chatbook/Agents/mcp_tool_provider.py
- tldw_chatbook/Logging_Config.py
- .github/workflows/derived-artifacts.yml
dependencies:
- TASK-38989
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f and on e8eeae7db9 (`fu-base-r10`), 2026-11-06. Line numbers are dev's (the same at e8eeae7db9).

**The red.**
- `Tests/test_persistent_diagnostic_boundary.py:95` `test_persona_workspace_diagnostics_do_not_interpolate_private_values` requires a constant, single-argument message for each prefix in `_CONSTANT_DIAGNOSTIC_PREFIXES` (`:20-44`), including "MCPToolProvider: persona_policy_provider failed".
- `tldw_chatbook/Agents/mcp_tool_provider.py:1635-1639` logs:

  ```
  logger.warning("MCPToolProvider: persona_policy_provider failed for {}; error_type={}", tool.name, type(exc).__name__)
  ```
- Run alone through run.sh, it is red at both commits: `AssertionError: tldw_chatbook/Agents/mcp_tool_provider.py: 'MCPToolProvider: persona_policy_provider failed' must not format runtime values`, `assert 3 == 1` (1 failed in 0.34 s).

**History.**
- The pin came from 4ee4c5af5a (PR #2232, merged to dev 2026-08-30 10:23 PDT) and the three-argument call from b6653e3806 (PR #2231, merged 15 minutes later). Neither PR saw the other, so the history suggests the test has been red since #2231 merged. That commit was not run.

**Who already knows.**
- TASK-31977 (In Progress) records this failure as a baseline limitation and does not own it.
- Architecture PR #3436 (head b7c8e809b3, TASK-38989) lanes only this file's new forwarder node (`test_forwarder_renders_only_a_plain_bound_exception_type`), not the whole file, because of this red.
- No PR lane runs the file on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `test_persona_workspace_diagnostics_do_not_interpolate_private_values` passes; it is shown red first (`assert 3 == 1` on the persona-policy failure log)
- [x] #2 With a `persona_policy_provider` that raises, the line the production forwarder emits still names the error class and holds no value the boundary test forbids; a test through the real forwarder shows this
- [x] #3 `Tests/test_persistent_diagnostic_boundary.py` runs whole on a PR lane (never the fast PR contract), replacing the single-node entry if TASK-38989 added one, with three timings of that lane's command recorded
<!-- AC:END -->

Possible approach (not required): after TASK-38989 the forwarder renders a bound `exception_type` (a plain class name only), so a constant message plus `logger.bind(exception_type=...)`, the shape the Persona Buddy logs use, would keep the class visible.

## Implementation Plan

1. Confirm the red at base: `test_persona_workspace_diagnostics_do_not_interpolate_private_values` fails with `assert 3 == 1` on the persona-policy failure log.
2. Add a test that runs a raising `persona_policy_provider` through `MCPToolProvider._persona_floor` and the real forwarder (`_forward_loguru_to_standard`), and asserts the forwarded line. Show it red while the old three-argument call is in place.
3. Change the log to a constant message, with the class bound as `exception_type` (the Persona Buddy shape), so the forwarder renders `<message>: <ErrorClass>`.
4. Update the pins this touches: the `exception_type` bind-site census (`Tests/Persona_Buddy/test_controller_failure_logging.py`), the reviewed metadata fields in `Tests/Architecture/test_persistent_diagnostic_inventory.py`, and the production diagnostic inventory (read the drifted rows before regenerating).
5. Lane `Tests/test_persistent_diagnostic_boundary.py` whole on the Architecture guards job in place of its one-node entry, then time the job command three times.

## Implementation Notes

**Status: Done.**

Owner decision (2026-11-06): "Bind class, drop count. All 10 sites log a constant message with logger.bind(exception_type=…) — the forwarder already renders it, so no detail is lost for the class name. Drop rule_count and its test assertion. The boundary stays strict: no runtime values in the message."

**The red, in full.** `test_persona_workspace_diagnostics_do_not_interpolate_private_values` stops at its first offender, which hid most of the drift. Walking every prefix in `_CONSTANT_DIAGNOSTIC_PREFIXES` at base 2684e5453f found 10 offenders, not 1:
- MCP `mcp_tool_provider.py:1635` (the filed `assert 3 == 1`);
- `Agents/persona_policy.py:62` and `:76`;
- `Character_Chat/local_character_persona_service.py:89`;
- `UI/Screens/personas_screen.py:16002`;
- `Workspaces/agent_provisioning.py:95` and `:177`;
- `Workspaces/registry_service.py:897`, `:910` and `:3976`.

The class and count arguments came from b6653e3806 and 0b2f6209de; 4ee4c5af5a's constant-message rule landed the same day, 2026-08-30.

**Changes.**
- All 10 sites now log a constant message and bind the class: `logger.bind(exception_type=type(<x>).__name__).<level>("<message>")`. The forwarder renders that as `<message>: <Class>`.
  - `persona_policy.py:62` binds the dropped entry's type under `exception_type`, the one key the forwarder renders.
  - The personas screen keeps the constant `mode=local` in its message and no longer logs the rule count.
- **Deliberate test change (owner decision):** `Tests/UI/test_persona_policy_rules_editor.py::test_policy_rule_save_failure_logs_metadata_only` no longer asserts `rule_count=1`. It asserts the bound `exception_type == "RuntimeError"` instead. With a temporary unbound call that assertion failed (`assert False`); with the bound call it passes.
- New test `Tests/test_persistent_diagnostic_boundary.py::test_persona_policy_failure_forwards_only_its_error_class` (AC#2). It drives a raising `persona_policy_provider` through `_persona_floor` and the real `_forward_loguru_to_standard`, and asserts the line `MCPToolProvider: persona_policy_provider failed: RuntimeError`. It also checks that neither the tool-name sentinel nor the error-text sentinel survives `RedactingFileFormatter`. It was red before the change: the line held the tool name.
- Pins updated:
  - In `Tests/Persona_Buddy/test_controller_failure_logging.py`, 8 reviewed sites were added to `EXCEPTION_TYPE_BIND_SITES`, with matching `EXCEPTION_TYPE_TOKEN_ROWS` tripwire rows. The stale "7-site census" wording now reads "bind-site census".
  - The reviewed fields in `Tests/Architecture/test_persistent_diagnostic_inventory.py` no longer list MCP's `tool.name` or personas_screen's `len(rules)`.
  - `Docs/security/production-diagnostic-inventory.json` was regenerated after reading the drift. `--statements` showed only the rewritten calls, and each file's diagnostic count is unchanged.

**Evidence.**
- AC#1: red at base with `assert 3 == 1`. After the MCP fix alone it failed at `persona_policy.py` with `assert 2 == 1`. Now `Tests/test_persistent_diagnostic_boundary.py` passes 6 of 6, and an all-prefix walk finds no offender.
- AC#3: the whole file is laned on the Architecture guards job, in place of the two forwarder-node entries. The step and `ARCHITECTURE_GUARD_TARGETS` are both updated.
- The covering-file runs on head and base, and the job run, are under "Final runs" below.

**Final runs** (code at 7d4f2b3f7a, through run.sh, 2026-11-06, after a reboot). Other agents kept the load between 15 and 200, so several heavy nodes hit their timeouts.
- Persona and log covering files (13 files, `-n 4`):
  - Head: 438 passed, 9 failed, 1 skipped, at load 64-201.
  - Base 2684e5453f: 421 passed, 7 failed, 1 skipped, at load 64-142.
  - None of head's 9 failures comes from this change:
    - 4 `test_persona_policy_rules_editor.py` nodes fail locally with RecoveryRequired (TASK-38111); base has the same 4. Under a temporary `bootstrap_profile` mark the file passes 17 and fails 1, `test_switcher_suffix_available_default_appends_label`, which is red at base with the same assertion.
    - 5 `test_persistent_diagnostic_inventory.py` nodes timed out at the 300 s default: 4 `task_15103` ledger nodes (base timed out 2 of them) and `test_production_diagnostic_inventory_and_sink_topology_are_unchanged`.
  - With `--timeout=1500`, the ledger and inventory nodes all pass on head (49 passed, load 65-99). The production-inventory node alone took a 284.9 s call.
  - Base's extra red is this task's boundary test.
- Architecture guards job command (`jobstep.sh`, 35 targets): 386 passed, 1 failed, load 19-138.
  - The failure is `test_profile_owned_path_inventory.py::test_compatibility_and_runtime_policy_constants_have_no_new_runtime_owners`, the 180 s timeout.
  - That node passes alone on head with a longer timeout (332.9 s call at load 65-106).
  - At base it also hits 180 s, at load about 79 (177.4 s call).
  - The node has been on this job since TASK-38989 and passed in all three earlier runs at load 24-57.
- bootbudget: 32 passed.
- `Tests/CI` at `-n 4`: 1,215 passed, 1 skipped.
