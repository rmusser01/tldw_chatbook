---
id: TASK-547
title: Fix tools config unreachable via get_cli_setting
status: Done
assignee: []
created_date: '2026-07-24 12:00'
labels: [tools, config, bug]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`get_tool_executor()` reads `tools_config = get_cli_setting("tools", {})` — the `{}` is treated as the DEFAULT-value slot (following get_cli_setting's non-string-2nd-arg convention), so it returns `{}` regardless of the actual `[tools]` TOML section. Net effect: all `[tools]` flags (`read_file_enabled`, `write_file_enabled`, `create_note_enabled`, `update_note_enabled`, `cache_enabled`, etc.) are unreachable and always False in real usage — the entire tools config section is dead and cannot be enabled by users.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] The dead `get_cli_setting("tools", {})` call site is gone — `get_tool_executor()` and `Tools/tool_executor.py` (System A) were deleted outright in TASK-545 P3, rather than the call being repaired in place, since System A had no live caller left to repair for
- [x] Enabling a `[tools]` flag in `config.toml` actually enables that tool on the live path: `BuiltinToolProvider` (System B, the path the agent runtime actually calls) reads its `[tools]` gate keys (`write_file_enabled`/`create_note_enabled`/`update_note_enabled`/etc.) correctly, and the Settings screen's save path was repaired to write them correctly
- [x] Unit test confirms that setting a `[tools]` enable flag (e.g. `write_file_enabled = true`) makes `BuiltinToolProvider` register/enable that tool
- [x] Reviewed other `get_cli_setting` calls for the same default-value-slot misuse: found four further live instances, filed as TASK-699, TASK-700, TASK-701 (three subsystems), plus the earlier TASK-658; the bug class itself is tracked by TASK-703
<!-- AC:END -->

## Implementation Notes

The original AC assumed `get_tool_executor()`'s broken `get_cli_setting("tools", {})` call would be repaired in place. That premise no longer held once TASK-545 P3 decided System A's fate: `Tools/tool_executor.py` and `get_tool_executor()` were deleted entirely (they had no live caller — the Console never reached System A, see TASK-545's description) rather than fixed, so there was no longer a broken call site to repair.

The underlying intent — "enabling a `[tools]` flag in config actually enables that tool" — is satisfied on the one remaining (live) path: `Agents/tool_catalog.py`'s `BuiltinToolProvider`, which the agent runtime (System B) actually calls, reads its own `[tools]` gate keys correctly (this predates P3; P2 used the same pattern to register `write_file`/`create_note`/`update_note` behind default-off gate keys). The write side of the loop — the Settings screen actually persisting those flags — was repaired as part of P3's work retargeting Settings at the live system.

ACs reworded above to match what shipped rather than the original (System-A-shaped) wording, per TASK-545's guidance not to leave stale ACs describing deleted code.
