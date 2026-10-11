---
id: TASK-15260
title: >-
  Console local tools: turn-context snapshot defaults False while the master
  switch defaults True
status: Done
assignee: []
created_date: '2026-08-11 14:40'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while repairing the dev test baseline (the MCP permission tests). PR #1474 flipped local tools on by default via LOCAL_TOOLS_DEFAULT_ENABLED = True (Agents/builtin_tool_gate.py:554), and _compose_local_provider (Chat/console_chat_controller.py:3840-3850) resolves the flag as turn_context.tool_configuration.get('local_tools_enabled', LOCAL_TOOLS_DEFAULT_ENABLED) — i.e. the turn context WINS when present. But the turn-context snapshot itself is built with the opposite default: console_chat_controller.py:6508-6510 does coerce_bool_setting(get_cli_setting('console','local_tools_enabled', False), False). Consequence for a user upgrading with an existing config.toml that lacks the key: the snapshot writes False, _compose_local_provider consumes it and disables local tools for every run, while the MCP hub's master switch reads LOCAL_TOOLS_DEFAULT_ENABLED and displays ON. Switch says enabled, tools are absent, no error. New installs are unaffected because the shipped config now writes the key explicitly, which is likely why this has gone unnoticed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both resolution sites agree on one default constant; a config.toml missing the key produces the same answer at the snapshot, the provider, and the hub switch
- [x] #2 A regression test covers the upgrade shape: config with no local_tools_enabled key, assert the run actually gets local tools and the hub switch agrees
<!-- AC:END -->

## Closeout audit 2026-10-11

Board-prune audit (branch chore/board-prune-c, base a3c47c6c72): completed-but-unmarked. Evidence gathered 2026-10-11 against origin/dev:

AC dispositions:
- AC#1: met - All resolution sites now agree on LOCAL_TOOLS_DEFAULT_ENABLED: Chat/console_chat_controller.py:16031/16035/16038/16245/16249/16380/16384 use the one constant (premise quoted coerce_bool_setting(..., False) no longer exists).
- AC#2: met - Tests/Agents/test_builtin_tool_gate.py:536-551 asserts 'missing key -> ON by default' for the upgrade shape.
- git grep LOCAL_TOOLS_DEFAULT_ENABLED origin/dev -- tldw_chatbook/Chat/console_chat_controller.py shows every snapshot/provider read using the shared constant (Agents/builtin_tool_gate.py:800).
- Regression evidence: test_builtin_tool_gate.py:536 'missing key -> available by default', :546/:551 'missing key -> ON by default'.
