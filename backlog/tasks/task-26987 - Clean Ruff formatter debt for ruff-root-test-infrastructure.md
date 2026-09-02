---
id: TASK-26987
title: Clean Ruff formatter debt for ruff-root-test-infrastructure
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

<!-- TASK-26000-BATCH: ruff-root-test-infrastructure -->
<!-- TASK-26000-PATHS-SHA256: fd7b774e7d2ef6cf2d616462d2e5eaecb60ab9736d06737711554cfe9fbc5158 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-root-test-infrastructure` Ruff formatter batch at the owner boundary recorded as: Root pytest guards, fixtures, and cross-suite test infrastructure.. The focused test surface recorded by TASK-26000 is `["Tests"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/conftest.py",
  "Tests/junit_outcome_diff.py",
  "Tests/network_guard.py",
  "Tests/test_call_from_thread_guard.py",
  "Tests/test_config_app_config_encryption.py",
  "Tests/test_config_chunking_defaults.py",
  "Tests/test_config_delete_settings.py",
  "Tests/test_config_encryption_effective_path.py",
  "Tests/test_config_private_bootstrap.py",
  "Tests/test_config_read_fastpath_task21124.py",
  "Tests/test_config_runtime_snapshot.py",
  "Tests/test_config_save_settings_semantics.py",
  "Tests/test_database_path_privacy.py",
  "Tests/test_hypothesis_profile.py",
  "Tests/test_keyring_isolation.py",
  "Tests/test_logging_private_files.py",
  "Tests/test_logs_buffer_single_record_per_emission.py",
  "Tests/test_network_guard.py",
  "Tests/test_persistent_diagnostic_sentinel_matrix.py",
  "Tests/test_persistent_log_is_not_empty.py",
  "Tests/test_probe_import_provenance.py",
  "Tests/test_smoke.py",
  "Tests/test_tiktoken_vendored_cache.py"
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
