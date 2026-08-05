---
id: TASK-331
title: Tool-executor security hardening
status: Done
assignee: []
created_date: '2026-07-20 18:45'
updated_date: '2026-07-24 12:00'
labels: [security, tools]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Low-severity hardening in the shared tool executor and file tools, grouped as one pass. Bundled per finding; can be split.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `file_operation_tools` passes a real sandbox root to `validate_path` instead of the literal strings `"file"`/`"directory"` (`file_operation_tools.py:65,176,338`); it currently confines ops to `<cwd>/file` — fails closed but is unintended and fragile
- [x] #2 The tool-result cache no longer uses `pickle.load` (`tool_executor.py:204`); it is replaced with JSON or another safe serializer
- [ ] #3 fs-mutating built-in tools (`read_file`/`write_file`/`list_directory`) require a confirmation/governance gate before auto-executing on model `tool_calls`, consistent with the MCP Allow/Ask/Off model
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#1 and AC#2 are complete: the sandbox root is now read from `[tools] file_sandbox_root` configuration (defaulting to `get_user_data_dir()/tool_sandbox`) and passed to `validate_path` in `file_operation_tools.py`, confining all file operations to a real, configurable directory; the tool-result cache in `tool_executor.py` was converted from pickle to JSON serialization, fixing an `ImportError` crash when the cache was enabled. AC#3 (confirmation/governance gate for fs-mutating tools) is deferred as a dedicated follow-up task-545 because it requires cross-system integration: wiring `ToolExecutor` (call site: `Event_Handlers`, main-loop) and/or the agent-runtime `BuiltinToolProvider` (call site: worker-thread) into the existing `MCP/permission_store.py` model (`resolve_effective_state`, `EffectiveToolState`), adding a risk-tag field to the `Tool` ABC, tagging mutating tools, and reusing `Widgets/Chat_Widgets/chat_approval_card.py` for ask confirmations — work well beyond the scope of this hardening bundle.
<!-- SECTION:NOTES:END -->
