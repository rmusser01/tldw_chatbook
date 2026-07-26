---
id: TASK-702
title: Rehome file-operation auditing off the deleted Settings side effect
status: To Do
assignee: []
created_date: '2026-07-26 08:00'
labels:
  - tools
  - agents
  - tech-debt
dependencies:
  - TASK-545
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`install_claude_code_hooks()` (`Tools/file_operation_hooks.py`) monkeypatched `WriteFileTool.execute` to audit file writes. Its only caller was System A's registration path, so the patch only ever took effect if a user happened to open the Settings screen — while the live agent runtime (`BuiltinToolProvider`, System B) executes the very same `WriteFileTool` class through a different path that never called `install_claude_code_hooks()`. TASK-545 P3 deleted System A's registration entirely, so `Tools/file_operation_hooks.py` is now unreferenced by any install path in the codebase.

If auditing agent file writes is still wanted, its correct home is the seam every built-in tool call already passes through — `BuiltinToolProvider.invoke` (the same gate/provider seam TASK-545 P1/P2 wired permissions through) — not a side effect of instantiating a UI screen. Note that `install_claude_code_hooks()`'s `INTEGRATION_INSTRUCTIONS` docstring still references `_global_executor`, a System-A-era symbol that no longer exists anywhere in the codebase after P3.

Also stale in the same file: `INTEGRATION_INSTRUCTIONS` (around line 346) still instructs the reader to "register the audit tool in tool_executor.py by adding `_global_executor.register_tool(CodeAuditTool())`" -- a global and a file section that no longer exist after P3. Fix or delete it as part of whatever this task decides.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] A decision is made and recorded: either (a) file-operation auditing is reimplemented at the `BuiltinToolProvider.invoke` seam so it covers every real write regardless of UI state, or (b) `Tools/file_operation_hooks.py` is removed as dead code
- [ ] If kept and reimplemented, the stale `_global_executor` reference in `INTEGRATION_INSTRUCTIONS` is corrected to describe the actual current wiring
- [ ] If removed, a grep confirms no remaining references to `install_claude_code_hooks` or `Tools/file_operation_hooks.py` anywhere in the codebase
<!-- AC:END -->
