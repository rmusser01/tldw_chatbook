---
id: TASK-547
title: Fix tools config unreachable via get_cli_setting
status: Done
assignee: []
created_date: '2026-07-24 12:00'
updated_date: '2026-07-26 20:44'
labels:
  - tools
  - config
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`get_tool_executor()` reads `tools_config = get_cli_setting("tools", {})` — the `{}` is treated as the DEFAULT-value slot (following get_cli_setting's non-string-2nd-arg convention), so it returns `{}` regardless of the actual `[tools]` TOML section. Net effect: all `[tools]` flags (`read_file_enabled`, `write_file_enabled`, `create_note_enabled`, `update_note_enabled`, `cache_enabled`, etc.) are unreachable and always False in real usage — the entire tools config section is dead and cannot be enabled by users.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 get_cli_setting call is fixed to read the [tools] section correctly (e.g. get_cli_setting("tools") without default, or using the section-dict accessor)
- [x] #2 Enabling a [tools] flag in config.toml actually enables that tool in get_tool_executor()
- [x] #3 Unit test confirms that setting read_file_enabled = true in [tools] section enables read_file
- [x] #4 Review other get_cli_setting calls for similar default-value slot misuse
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed 2026-07-26: the defect was confined to `get_tool_executor()`'s two-argument `get_cli_setting("tools", {})` call. `get_cli_setting`'s non-string-second-arg convention treats a non-str second positional as the DEFAULT slot, not a key, so `{}` was silently read back as the default on every call regardless of the real `[tools]` TOML section — the entire section was unreachable through that one call site.

The `[tools]` section itself was never dead: the three-argument form (`get_cli_setting(section, key, default)`) reads it correctly, and two other call sites already used that form successfully — TASK-584's `read_file_enabled`/`list_directory_enabled` gates in `Agents/tool_catalog.py`, and `Tools/file_operation_tools.py`'s `_resolve_sandbox_config()` (`get_cli_setting("tools", "file_sandbox_root", default_root)`). `get_tool_executor()` was the only caller using the broken two-argument shape. `refactor(tools): remove the ToolExecutor, code audit tool, and settings switches` deleted that function along with the rest of `ToolExecutor` (no remaining callers post task-577), which removed the only broken caller — there is no in-place fix left to make.

The live keys have since migrated to `[agent_tools] enabled_packs` (`Agents/builtin_pack_config.py`), read via the correct three-argument form. The old per-tool `[tools] read_file_enabled`/`list_directory_enabled` flags keep working as a deprecated fallback so an existing user's config isn't silently switched off. Per the project owner's decision (recorded in `builtin_pack_config.py`), that fallback grants *exactly* the tool each old flag names — `read_file_enabled` grants only `read_file`, not the whole `files` pack — so migrating to pack-level config did not widen any existing user's grant. Covered by `Tests/Agents/test_builtin_pack_config.py` (`test_legacy_read_file_flag_restricts_to_read_file_only` and siblings).

AC #4 (review other `get_cli_setting` calls for the same default-slot misuse): grepped the codebase for the `get_cli_setting("<section>", {})` shape. Two look-alikes checked out fine — `Skills_Interop/local_skills_service.py` already documents the trap in a comment and deliberately uses the three-argument form; `Local_Ingestion/transcription_service.py:2777`'s `get_cli_setting("transcription.remote_whisper", {})` is safe because the dotted section name routes through `get_cli_setting`'s dotted-format branch, which reassigns the second positional to `default` correctly. One live instance of the actual bug remains: `Local_Ingestion/local_file_ingestion.py:1148`'s `quick_ingest()` (`get_cli_setting("database", {})`, no dot, so `{}` is silently returned as-is) — already tracked upstream as TASK-760 on origin/dev (this worktree branch predates that fix), so no new task was filed here.
<!-- SECTION:NOTES:END -->
