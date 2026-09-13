---
id: TASK-27000
title: Clean Ruff formatter debt for ruff-ui-personas
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

<!-- TASK-26000-BATCH: ruff-ui-personas -->
<!-- TASK-26000-PATHS-SHA256: 46d36f76100c63f6ae8f76474794245c08c3a08fc174f406ae4641e07562c22b -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-ui-personas` Ruff formatter batch at the owner boundary recorded as: Persona and character UI surfaces with direct UI/Character tests.. The focused test surface recorded by TASK-26000 is `["Tests/Character_Chat", "Tests/UI"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/UI/test_actor_pack_staging_sweep_seam.py",
  "Tests/UI/test_character_display_text.py",
  "Tests/UI/test_persona_policy_rules_editor.py",
  "Tests/UI/test_persona_profile_widgets.py",
  "Tests/UI/test_personas_center_canvas_layout.py",
  "Tests/UI/test_personas_character_editor_avatar.py",
  "Tests/UI/test_personas_character_widgets.py",
  "Tests/UI/test_personas_character_world_books.py",
  "Tests/UI/test_personas_character_world_books_screen.py",
  "Tests/UI/test_personas_deferred_center_views.py",
  "Tests/UI/test_personas_editor_save_in_place.py",
  "Tests/UI/test_personas_expression_generate.py",
  "Tests/UI/test_personas_inspector_pane.py",
  "Tests/UI/test_personas_lore.py",
  "Tests/UI/test_personas_preview.py",
  "Tests/UI/test_personas_preview_restore.py",
  "Tests/UI/test_personas_workbench.py",
  "Tests/UI/test_personas_workbench_foundation.py",
  "Tests/UI/test_personas_workbench_state.py",
  "Tests/UI/test_uat_first_time_character_chat.py",
  "tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py",
  "tldw_chatbook/UI/Persona_Modules/personas_conversations_controller.py"
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
