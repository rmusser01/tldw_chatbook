---
id: TASK-26989
title: Clean Ruff formatter debt for ruff-skills-runtime
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

<!-- TASK-26000-BATCH: ruff-skills-runtime -->
<!-- TASK-26000-PATHS-SHA256: 039c0b882508fa3ae476d03f75cc50a8235835064066f025f7ae7881ad118089 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-skills-runtime` Ruff formatter batch at the owner boundary recorded as: Skill discovery, trust, import, package, and script execution with direct tests.. The focused test surface recorded by TASK-26000 is `["Tests/Skills"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Skills/test_e2e_run_skill_script.py",
  "Tests/Skills/test_import_skill_directory.py",
  "Tests/Skills/test_local_skills_bundle_io.py",
  "Tests/Skills/test_project_skills_discovery.py",
  "Tests/Skills/test_project_skills_import_modal.py",
  "Tests/Skills/test_project_skills_startup_gate.py",
  "Tests/Skills/test_read_skill_file.py",
  "Tests/Skills/test_skill_fingerprint_executable.py",
  "Tests/Skills/test_skill_package_inspection.py",
  "Tests/Skills/test_skill_remote_fetch.py",
  "Tests/Skills/test_skill_script_grants.py",
  "Tests/Skills/test_skill_script_runner.py",
  "Tests/Skills/test_skill_script_service.py",
  "Tests/Skills/test_skill_trust_scanner_recursive.py",
  "Tests/Skills/test_skill_trust_store.py",
  "Tests/Skills/test_skill_trust_store_reset.py",
  "Tests/Skills/test_skill_trust_store_scoping.py",
  "Tests/Skills/test_skills_import.py",
  "Tests/Skills/test_skills_library_flow.py",
  "Tests/Skills/test_trust_tolerates_unsupported.py",
  "Tests/Skills/test_verify_content_binary.py",
  "Tests/Skills/test_web_research_skill.py",
  "Tests/Skills/test_zip_import_bundle.py",
  "tldw_chatbook/Skills_Interop/project_skills_discovery.py",
  "tldw_chatbook/Skills_Interop/skill_package_inspection.py",
  "tldw_chatbook/Skills_Interop/skill_remote_fetch.py",
  "tldw_chatbook/Skills_Interop/skill_trust_scanner.py"
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
