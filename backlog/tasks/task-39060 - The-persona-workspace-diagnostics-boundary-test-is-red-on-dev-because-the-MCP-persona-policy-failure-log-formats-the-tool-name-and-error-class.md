---
id: TASK-39060
title: "The persona workspace diagnostics boundary test is red on dev because the MCP persona-policy failure log formats the tool name and error class"
status: To Do
created_date: 2026-11-06 12:31
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
- [ ] #2 With a `persona_policy_provider` that raises, the line the production forwarder emits still names the error class and holds no value the boundary test forbids; a test through the real forwarder shows this
- [ ] #3 `Tests/test_persistent_diagnostic_boundary.py` runs whole on a PR lane (never the fast PR contract), replacing the single-node entry if TASK-38989 added one, with three timings of that lane's command recorded
<!-- AC:END -->

Possible approach (not required): after TASK-38989 the forwarder renders a bound `exception_type` (a plain class name only), so a constant message plus `logger.bind(exception_type=...)`, the shape the Persona Buddy logs use, would keep the class visible.
