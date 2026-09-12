---
id: TASK-27010
title: Clean Ruff formatter debt for ruff-utils-config
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

<!-- TASK-26000-BATCH: ruff-utils-config -->
<!-- TASK-26000-PATHS-SHA256: 6d05db3da2d66bd7888500c30bb68c4bfe99c2c2479604464da33038f880d52f -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-utils-config` Ruff formatter batch at the owner boundary recorded as: Shared utilities and direct Utils/config tests.. The focused test surface recorded by TASK-26000 is `["Tests/Utils"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Utils/test_atomic_file_ops.py",
  "Tests/Utils/test_config_encryption.py",
  "Tests/Utils/test_config_nested_settings.py",
  "Tests/Utils/test_db_status_manager.py",
  "Tests/Utils/test_download_caps_wiring.py",
  "Tests/Utils/test_egress.py",
  "Tests/Utils/test_egress_adoption_census.py",
  "Tests/Utils/test_fd_protection.py",
  "Tests/Utils/test_fts5_quoting_adoption_census.py",
  "Tests/Utils/test_git_url_validation.py",
  "Tests/Utils/test_github_api_client.py",
  "Tests/Utils/test_image_protocol_warmup.py",
  "Tests/Utils/test_instance_lock.py",
  "Tests/Utils/test_log_sanitizer.py",
  "Tests/Utils/test_mosaic_render.py",
  "Tests/Utils/test_optional_import_deferral.py",
  "Tests/Utils/test_private_paths.py",
  "Tests/Utils/test_sensitive_config_keys.py",
  "Tests/Utils/test_sensitive_paths.py",
  "Tests/Utils/test_startup_polish_regressions.py",
  "Tests/Utils/test_text_wrap_index.py",
  "Tests/Utils/test_tls_trust.py",
  "tldw_chatbook/Utils/fd_protection.py",
  "tldw_chatbook/Utils/fts5_match_forms.py",
  "tldw_chatbook/Utils/github_api_client.py",
  "tldw_chatbook/Utils/instance_lock.py",
  "tldw_chatbook/Utils/log_sanitizer.py",
  "tldw_chatbook/Utils/path_validation.py",
  "tldw_chatbook/Utils/private_paths.py",
  "tldw_chatbook/Utils/sensitive_config_keys.py",
  "tldw_chatbook/Utils/sensitive_paths.py",
  "tldw_chatbook/Utils/terminal_utils.py",
  "tldw_chatbook/Utils/text_wrap_index.py",
  "tldw_chatbook/Utils/tls_trust.py"
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
