---
id: TASK-743
title: Resolve file-operation auditing subsystem
status: To Do
assignee: []
created_date: '2026-07-26 08:00'
updated_date: '2026-08-12 21:18'
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
The file-operation audit subsystem has no live owner. `CodeAuditTool`, `FileAuditSystem`, `Tools/file_operation_hooks.py`, its demo, feature tests, and live documentation still describe a System-A registration side effect that TASK-545 removed. `code_audit` never became a current provider capability, and `install_claude_code_hooks()` is no longer installed.

This task owns one complete keep/redesign/delete decision and its implementation for the whole subsystem. A retained or redesigned feature must cover every live Console file-mutation seam: built-in `write_file` and local `fs_write`, `fs_edit`, and `fs_patch`. A deletion must remove the implementation, hook, demo, feature-specific tests and documentation, and stale references together. Merely adding a `BuiltinToolProvider` write hook would leave the local file-mutation paths unaudited and is not a complete outcome.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One keep/redesign/delete decision covers `CodeAuditTool`, `FileAuditSystem`, `Tools/file_operation_hooks.py`, the demo, feature tests, live documentation, and stale integration references; the chosen outcome is implemented in this task's single PR
- [ ] #2 Before retaining or redesigning the feature, the task's Implementation Plan records the ADR check and links the applicable existing or new ADR; a deletion records why no feature ADR is required
- [ ] #3 If kept or redesigned, successful Console file mutations through built-in `write_file` and local `fs_write`, `fs_edit`, and `fs_patch` all reach the same audit owner, with focused tests proving all four seams and proving refused or out-of-workspace operations cannot reach auditing or mutation
- [ ] #4 If kept or redesigned, the audit-state owner, workspace/session scope, creation/reset/teardown lifecycle, finite bound, and deterministic eviction behavior are explicit; tests prove audit content cannot cross workspace or Console-session boundaries and no unscoped global buffer is used
- [ ] #5 If kept or redesigned, the operator can select the audit provider and model without code changes, and unavailable or invalid selection produces a deterministic, user-visible outcome
- [ ] #6 If kept or redesigned, diagnostics are payload-free and tests prove file contents, patches, audit prompts, model responses, and credentials are absent from logs and error details
- [ ] #7 If kept or redesigned, auditing neither bypasses nor weakens the existing permission gate or workspace confinement, and audit prompts/content are disclosed only to the selected audit provider
- [ ] #8 If deleted, the audit implementation, installation hook, demo, feature-specific tests and live documentation are removed, and a repository scan finds no stale claims or references to the deleted subsystem
- [ ] #9 The surviving live documentation describes the chosen behavior; retained/redesigned audit tests pass, or deletion scans and the surrounding built-in/local tool suites pass
<!-- AC:END -->
