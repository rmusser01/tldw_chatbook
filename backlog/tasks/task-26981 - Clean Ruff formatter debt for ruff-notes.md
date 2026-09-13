---
id: TASK-26981
title: Clean Ruff formatter debt for ruff-notes
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

<!-- TASK-26000-BATCH: ruff-notes -->
<!-- TASK-26000-PATHS-SHA256: 51d8e0189d61b89127b84990aafbfc323ba2ec1c95309b44006b1d7976b4f6b9 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-notes` Ruff formatter batch at the owner boundary recorded as: Notes persistence/sync services and direct Notes tests.. The focused test surface recorded by TASK-26000 is `["Tests/Notes"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Notes/git_process_tree_test_helper.py",
  "Tests/Notes/test_agent_lesson_mutation_authority.py",
  "Tests/Notes/test_agent_lesson_secret_validation.py",
  "Tests/Notes/test_agent_lessons.py",
  "Tests/Notes/test_agent_lessons_seed.py",
  "Tests/Notes/test_file_notes_git_commit_integration.py",
  "Tests/Notes/test_file_notes_git_integration.py",
  "Tests/Notes/test_file_notes_git_push.py",
  "Tests/Notes/test_file_notes_git_push_integration.py",
  "Tests/Notes/test_file_notes_git_push_service.py",
  "Tests/Notes/test_file_notes_git_push_transport.py",
  "Tests/Notes/test_file_notes_git_service.py",
  "Tests/Notes/test_file_notes_replica.py",
  "Tests/Notes/test_file_notes_service.py",
  "Tests/Notes/test_file_notes_session_owner.py",
  "Tests/Notes/test_git_process_containment.py",
  "Tests/Notes/test_note_folder_repository.py",
  "Tests/Notes/test_note_import_executor.py",
  "Tests/Notes/test_note_import_receipts.py",
  "Tests/Notes/test_note_organization_transaction.py",
  "Tests/Notes/test_notes_library_unit.py",
  "Tests/Notes/test_notes_scope_service_folders.py",
  "Tests/Notes/test_notes_sync_executor.py",
  "Tests/Notes/test_notes_sync_observation_reuse.py",
  "Tests/Notes/test_notes_sync_version_states.py",
  "Tests/Notes/test_notes_sync_watcher.py",
  "Tests/Notes/test_notes_sync_worker_coroutine.py",
  "Tests/Notes/test_server_notes_workspace_service.py",
  "tldw_chatbook/Notes/Notes_Library.py",
  "tldw_chatbook/Notes/agent_lessons.py",
  "tldw_chatbook/Notes/file_notes_git_network.py",
  "tldw_chatbook/Notes/file_notes_git_push.py",
  "tldw_chatbook/Notes/file_notes_git_service.py",
  "tldw_chatbook/Notes/file_notes_service.py",
  "tldw_chatbook/Notes/file_notes_session_owner.py",
  "tldw_chatbook/Notes/git_process_containment.py",
  "tldw_chatbook/Notes/note_folder_repository.py",
  "tldw_chatbook/Notes/notes_device_state_store.py",
  "tldw_chatbook/Notes/notes_organization_repository.py",
  "tldw_chatbook/Notes/notes_scope_service.py",
  "tldw_chatbook/Notes/notes_sync_authority.py",
  "tldw_chatbook/Notes/notes_sync_executor.py",
  "tldw_chatbook/Notes/notes_sync_runtime.py",
  "tldw_chatbook/Notes/notes_sync_watcher.py",
  "tldw_chatbook/Notes/server_notes_workspace_service.py"
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
