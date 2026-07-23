---
id: TASK-331
title: Tool-executor security hardening
status: To Do
assignee: []
created_date: '2026-07-20 18:45'
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
- [ ] #1 `file_operation_tools` passes a real sandbox root to `validate_path` instead of the literal strings `"file"`/`"directory"` (`file_operation_tools.py:65,176,338`); it currently confines ops to `<cwd>/file` — fails closed but is unintended and fragile
- [ ] #2 The tool-result cache no longer uses `pickle.load` (`tool_executor.py:204`); it is replaced with JSON or another safe serializer
- [ ] #3 fs-mutating built-in tools (`read_file`/`write_file`/`list_directory`) require a confirmation/governance gate before auto-executing on model `tool_calls`, consistent with the MCP Allow/Ask/Off model
<!-- AC:END -->
