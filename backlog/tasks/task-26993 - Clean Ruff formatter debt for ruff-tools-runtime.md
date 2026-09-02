---
id: TASK-26993
title: Clean Ruff formatter debt for ruff-tools-runtime
status: To Do
assignee: []
created_date: '2026-08-31 18:31'
updated_date: '2026-08-31 18:31'
labels:
  - maintenance
  - formatting
  - quality
dependencies:
  - TASK-26000
references:
  - Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md
  - Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json
priority: medium
---

<!-- TASK-26000-BATCH: ruff-tools-runtime -->
<!-- TASK-26000-PATHS-SHA256: f23e5415b0e659af3d7949462ab2b9ad79b780a5f56fe397acfe7d22dc9e7535 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-tools-runtime` Ruff formatter batch at the owner boundary recorded as: Local, Git, web, workspace-dispatch, and virtual CLI tools with direct tests.. The focused test surface recorded by TASK-26000 is `["Tests/Tools"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Tools/test_code_audit_repoint.py",
  "Tests/Tools/test_document_expansion_tool.py",
  "Tests/Tools/test_file_tool_sandbox.py",
  "Tests/Tools/test_file_tools_workspace_roots.py",
  "Tests/Tools/test_git_tool_impls.py",
  "Tests/Tools/test_git_tool_sensitive_paths.py",
  "Tests/Tools/test_glob_grep_files.py",
  "Tests/Tools/test_local_tool_impls.py",
  "Tests/Tools/test_local_tool_impls_properties.py",
  "Tests/Tools/test_local_tool_sensitive_paths.py",
  "Tests/Tools/test_patch_tool_impls.py",
  "Tests/Tools/test_virtual_cli_impls.py",
  "Tests/Tools/test_watchlists_command_service.py",
  "Tests/Tools/test_watchlists_tool_service.py",
  "Tests/Tools/test_web_crawl.py",
  "Tests/Tools/test_web_deep_search.py",
  "Tests/Tools/test_web_search_tool.py",
  "Tests/Tools/test_web_tool_impls.py",
  "Tests/Tools/test_workspace_root_pin.py",
  "Tests/Tools/test_workspace_tool_executor.py",
  "Tests/Tools/test_workspace_tool_protocol.py",
  "Tests/Tools/test_write_file_diff_capture.py",
  "tldw_chatbook/Tools/_grep_worker.py",
  "tldw_chatbook/Tools/document_expansion_tool.py",
  "tldw_chatbook/Tools/git_tool_impls.py",
  "tldw_chatbook/Tools/local_tool_impls.py",
  "tldw_chatbook/Tools/note_management_tools.py",
  "tldw_chatbook/Tools/patch_tool_impls.py",
  "tldw_chatbook/Tools/virtual_cli_impls.py",
  "tldw_chatbook/Tools/watchlists_command_service.py",
  "tldw_chatbook/Tools/watchlists_tool_service.py",
  "tldw_chatbook/Tools/web_search_tool.py",
  "tldw_chatbook/Tools/web_tool_impls.py",
  "tldw_chatbook/Tools/workspace_root_pin.py",
  "tldw_chatbook/Tools/workspace_tool_dispatch.py",
  "tldw_chatbook/Tools/workspace_tool_executor.py",
  "tldw_chatbook/Tools/workspace_tool_protocol.py",
  "tldw_chatbook/Tools/workspace_tool_worker.py"
]
```

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [ ] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [ ] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [ ] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [ ] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [ ] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [ ] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [ ] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->
