---
id: TASK-331
title: Harden built-in file-tool sandbox and confirmation governance
status: To Do
assignee: []
created_date: '2026-07-20 18:45'
updated_date: '2026-07-23 14:21'
labels:
  - security
  - tools
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Correct the built-in file tools sandbox boundary and require explicit governance before model-initiated filesystem mutation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `file_operation_tools` passes a real sandbox root to `validate_path` instead of the literal strings `"file"`/`"directory"` (`file_operation_tools.py:65,176,338`); it currently confines ops to `<cwd>/file` — fails closed but is unintended and fragile
- [ ] #2 Built-in filesystem tools (`read_file`/`write_file`/`list_directory`) require a confirmation/governance gate before auto-executing on model `tool_calls`, consistent with the MCP Allow/Ask/Off model
- [ ] #3 Behavioral tests cover allowed, denied, confirmation-required, and sandbox-escape attempts.
<!-- AC:END -->
