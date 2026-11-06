---
id: TASK-39060
title: "The persona workspace diagnostics boundary test is red on dev because the MCP persona-policy failure log formats the tool name and error class"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-06 12:31
updated_date: 2026-11-06 17:59
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
- [ ] #1 `test_persona_workspace_diagnostics_do_not_interpolate_private_values` passes; it is shown red first (`assert 3 == 1` on the persona-policy failure log)
- [x] #2 With a `persona_policy_provider` that raises, the line the production forwarder emits still names the error class and holds no value the boundary test forbids; a test through the real forwarder shows this
- [ ] #3 `Tests/test_persistent_diagnostic_boundary.py` runs whole on a PR lane (never the fast PR contract), replacing the single-node entry if TASK-38989 added one, with three timings of that lane's command recorded
<!-- AC:END -->

Possible approach (not required): after TASK-38989 the forwarder renders a bound `exception_type` (a plain class name only), so a constant message plus `logger.bind(exception_type=...)`, the shape the Persona Buddy logs use, would keep the class visible.

## Implementation Plan

1. Confirm the red at base: `test_persona_workspace_diagnostics_do_not_interpolate_private_values` fails with `assert 3 == 1` on the persona-policy failure log.
2. Add a test that runs a raising `persona_policy_provider` through `MCPToolProvider._persona_floor` and the real forwarder (`_forward_loguru_to_standard`), and asserts the forwarded line. Show it red while the old three-argument call is in place.
3. Change the log to a constant message, with the class bound as `exception_type` (the Persona Buddy shape), so the forwarder renders `<message>: <ErrorClass>`.
4. Update the pins this touches: the `exception_type` bind-site census (`Tests/Persona_Buddy/test_controller_failure_logging.py`), the reviewed metadata fields in `Tests/Architecture/test_persistent_diagnostic_inventory.py`, and the production diagnostic inventory (read the drifted rows before regenerating).
5. Lane `Tests/test_persistent_diagnostic_boundary.py` whole on the Architecture guards job in place of its one-node entry, then time the job command three times.

## Implementation Notes

**Status: In Progress (NEEDS_CONTEXT). AC#2 is met; AC#1 and AC#3 are blocked on an owner question. The test hid nine more offenders behind the one filed.**

**Done (AC#2).**
- `MCPToolProvider._persona_floor` (`Agents/mcp_tool_provider.py`) used to format `tool.name` and the error class. It now logs a constant message and binds the class: `logger.bind(exception_type=type(exc).__name__).warning("MCPToolProvider: persona_policy_provider failed")`. The forwarder renders that as `MCPToolProvider: persona_policy_provider failed: RuntimeError`.
- New test `Tests/test_persistent_diagnostic_boundary.py::test_persona_policy_failure_forwards_only_its_error_class`. It drives a raising provider through `_persona_floor` and the real `_forward_loguru_to_standard`. It asserts the exact forwarded line, that `RuntimeError` survives `RedactingFileFormatter`, and that neither the tool-name sentinel nor the error-text sentinel does.
  - Red before the change: the forwarded line was `...failed for tool-PRIVATE-PROMPT-SENTINEL-...; error_type=RuntimeError`.
  - Green after.
- Pins that follow the change:
  - the `exception_type` bind-site census and the package tripwire in `Tests/Persona_Buddy/test_controller_failure_logging.py` each gained one reviewed row. The stale "7-site census" wording now reads "bind-site census".
  - The reviewed fields in `Tests/Architecture/test_persistent_diagnostic_inventory.py` changed from `("tool.name", "type(exc).__name__")` to `("type(exc).__name__",)`.
  - `Docs/security/production-diagnostic-inventory.json` was regenerated after reading the drift. One row changed: `mcp_tool_provider.py 10/eef0ce33… -> 10/7d804c0e…`, the reviewed call.
- The new node runs on the Architecture guards job beside the existing forwarder node. The workflow step and `ARCHITECTURE_GUARD_TARGETS` were both updated.
- Covering runs:
  - the boundary file, the bind census, the diagnostic inventory and `Tests/Agents/test_mcp_provider_profile.py`: 101 passed, 1 skipped, 1 failed (the boundary test, below);
  - `Tests/CI`: 1,215 passed, 1 skipped.

**Blocked (AC#1, AC#3).** With the MCP line fixed, `test_persona_workspace_diagnostics_do_not_interpolate_private_values` fails at its next file: `persona_policy.py: 'Dropping non-mapping persona policy rule' must not format runtime values`, `assert 2 == 1`. The test stops at the first offender. A walk over all of `_CONSTANT_DIAGNOSTIC_PREFIXES` found 10 offenders at base 2684e5453f, not 1. The nine besides MCP are:
- `Agents/persona_policy.py:62` (`entry_type`) and `:76` (`error_type`);
- `Character_Chat/local_character_persona_service.py:89` (`error_type`);
- `UI/Screens/personas_screen.py:16002` (`mode=local; rule_count={}; error_type={}`);
- `Workspaces/agent_provisioning.py:95` and `:177`;
- `Workspaces/registry_service.py:897`, `:910` and `:3976`.

Every one of them formats `type(<x>).__name__`, and personas_screen also formats `len(rules)`.

Two reviewed policies conflict here:
- b6653e3806 ("reconcile inventory and enforce privacy") and 0b2f6209de ("retain safe persona save context") added those arguments on 2026-08-30. They pinned them as reviewed metadata in `REVIEWED_METADATA_ONLY_DIAGNOSTICS`, and `Tests/UI/test_persona_policy_rules_editor.py::test_policy_rule_save_failure_logs_metadata_only` asserts `rule_count=1` and `mode=local` in the rendered message.
- 4ee4c5af5a, merged the same day, requires a constant, single-argument message for each of these prefixes.

The owner needs to choose one of:
- (a) bind the class at all nine sites, as here. personas_screen then loses `rule_count`, and that editor test's `rule_count=1` assertion has to go.
- (b) let the boundary test accept reviewed class-name and count arguments.

I did neither. (a) removes an assertion from another test, and (b) loosens the boundary test, so either one needs the owner's ruling. Until then the whole file cannot run on a lane, so AC#3 stays open. Only its two forwarder nodes are laned.

Files: `tldw_chatbook/Agents/mcp_tool_provider.py`, `Tests/test_persistent_diagnostic_boundary.py`, `Tests/Persona_Buddy/test_controller_failure_logging.py`, `Tests/Architecture/test_persistent_diagnostic_inventory.py`, `Docs/security/production-diagnostic-inventory.json`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.

**Architecture guards job timings (shared by TASK-39054, TASK-39055, TASK-39060 and TASK-39061).**
- Before adding anything: CI ran the step in 402 s (job 7 m 38 s) on #3436's run 42218508176, 2026-11-06.
- The step's command, run with `bash $FU/jobstep.sh fu-r11-arch "Run Architecture guard contracts"` (serial, `--timeout=180 --tb=short`). Each run of the final 36 targets gave 383 passed:

| Command | Targets | Nodes | pytest time | Load |
|---|---|---|---|---|
| Base 2684e5453f | 29 | 160 | 153.9 s | 7-11 |
| Base 2684e5453f | 29 | 160 | 457.0 s | 27-45 |
| Base 2684e5453f | 29 | 160 | 283.9 s | 32-38 |
| This branch, run 1 | 36 | 383 | 563.4 s | 25-48 |
| This branch, run 2 | 36 | 383 | 451.2 s | 24-48 |
| This branch, run 3 | 36 | 383 | 440.6 s | 24-57 |
| This branch, right after the 283.9 s base run | 36 | 383 | 413.8 s | 32-48 |

- Six agents shared the machine, and the load swung from 24 to 57, so the totals mostly measure the load.
- One more run, with `--durations=0` (862.4 s at load 50-85), attributes setup, call and teardown time per file. The added files cost 146.9 s of its 858.3 s:
  - `test_no_undefined_module_globals.py`: 72.5 s for 3 nodes, with no periodic-GC teardown;
  - `test_log_sanitizer.py`: 40.6 s for 168 nodes, of which 31.8 s is the conftest's every-25th-test `gc.collect()`;
  - the four ChatScreen files: 33.5 s for 51 nodes, 8.9 s of it GC;
  - the two boundary nodes: 0.3 s.
- An extra job run overlapped my temporary source reverts for a boot-census bisect (12 failed). I discarded it.
