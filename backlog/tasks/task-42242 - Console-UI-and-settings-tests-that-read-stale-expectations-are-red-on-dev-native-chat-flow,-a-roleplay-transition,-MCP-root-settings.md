---
id: TASK-42242
title: "Console UI and settings tests that read stale expectations are red on dev: native chat flow, a roleplay transition, MCP root settings"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- tests
- console
- ui
priority: medium
references:
- Tests/UI/test_console_native_chat_flow.py
- Tests/UI/test_mcp_root_settings.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the providers2 stream (round 19), which reported about 47 red nodes in `test_native_chat_flow*`, one roleplay transition test and two `mcp_root_settings` tests, all stale on dev. Checked at e5626ea043 through the scratch wrapper at `-n 4`:
- `Tests/UI/test_console_native_chat_flow.py`: 18 nodes fail at line 222, the `assert config_module.save_setting_to_cli_config("console", "agent_runtime", False)` in `_build_console_send_test_app`; one fails at line 5276, `test_console_native_blocked_send_preserves_composer_text_and_shows_recovery`, which asserts `composer.draft_text() == ""` while the composer still holds `blocked draft`. The other nodes of that run fail with `RecoveryRequired: raw_source_selection_changed`, the known local trap, which is not a cause to fix here.
- `Tests/UI/test_mcp_root_settings.py::test_local_config_shutdown_drains_and_fences_both_config_owners[workspace_root]` fails; the second node the stream reported was not identified.
- The roleplay transition test was not located by name; the stream did not name it and no file at the base matches `native_chat_flow` or a roleplay transition name beyond `Tests/UI/test_library_style_transitions.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of each red node is named and fixed for every red node in `test_console_native_chat_flow.py`, the roleplay transition test (identified by name in the task's notes) and the two `test_mcp_root_settings.py` nodes
- [ ] #2 A node whose expectation was stale is updated to the current behaviour; a node that exposes a product defect has its own task filed
- [ ] #3 Each file named here passes through the scratch wrapper at `-n 4` apart from nodes that fail only with the local `RecoveryRequired` trap, which are listed
<!-- AC:END -->
