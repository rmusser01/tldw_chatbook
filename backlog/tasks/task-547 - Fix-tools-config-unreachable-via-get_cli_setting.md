---
id: TASK-547
title: Fix tools config unreachable via get_cli_setting
status: To Do
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
- [ ] get_cli_setting call is fixed to read the [tools] section correctly (e.g. get_cli_setting("tools") without default, or using the section-dict accessor)
- [ ] Enabling a [tools] flag in config.toml actually enables that tool in get_tool_executor()
- [ ] Unit test confirms that setting read_file_enabled = true in [tools] section enables read_file
- [ ] Review other get_cli_setting calls for similar default-value slot misuse
<!-- AC:END -->
