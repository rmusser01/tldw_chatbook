---
id: TASK-26942
title: Clean Ruff formatter debt for ruff-character-persona
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

<!-- TASK-26000-BATCH: ruff-character-persona -->
<!-- TASK-26000-PATHS-SHA256: b538e3a29758a03fa04e3e2c46fc71161cf40919bc8357c00d785f7170e1617e -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-character-persona` Ruff formatter batch at the owner boundary recorded as: Character, persona, and actor-pack ownership with direct tests.. The focused test surface recorded by TASK-26000 is `["Tests/Actor_Packs", "Tests/Character_Chat"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Actor_Packs/test_actor_pack_import.py",
  "Tests/Actor_Packs/test_persona_actor_pack_coordinator.py",
  "Tests/Character_Chat/test_apply_world_info_to_message.py",
  "Tests/Character_Chat/test_character_card_lenient_import.py",
  "Tests/Character_Chat/test_character_generation_controller.py",
  "Tests/Character_Chat/test_character_tts_portability.py",
  "Tests/Character_Chat/test_character_world_book_send_path.py",
  "Tests/Character_Chat/test_compose_character_card_text.py",
  "Tests/Character_Chat/test_emote_directives.py",
  "Tests/Character_Chat/test_expression_set_io.py",
  "Tests/Character_Chat/test_persona_list_paging.py",
  "Tests/Character_Chat/test_persona_policy_rules.py",
  "Tests/Character_Chat/test_placeholder_aliases.py",
  "Tests/Character_Chat/test_resolve_character_world_books.py",
  "Tests/Character_Chat/test_resolve_world_info_injection.py",
  "Tests/Character_Chat/test_summarize_active_world_books.py",
  "Tests/Character_Chat/test_world_book_import.py",
  "Tests/Character_Chat/test_world_book_manager.py",
  "Tests/Character_Chat/test_world_info_diagnostics.py",
  "Tests/Character_Chat/test_world_info_regex.py",
  "tldw_chatbook/Character_Chat/Character_Chat_Lib.py",
  "tldw_chatbook/Character_Chat/character_card_formats.py",
  "tldw_chatbook/Character_Chat/character_generation.py",
  "tldw_chatbook/Character_Chat/character_persona_scope_service.py",
  "tldw_chatbook/Character_Chat/expression_set_io.py",
  "tldw_chatbook/Character_Chat/persona_list_paging.py",
  "tldw_chatbook/Character_Chat/server_character_persona_service.py",
  "tldw_chatbook/Character_Chat/world_book_manager.py",
  "tldw_chatbook/Character_Chat/world_info_processor.py",
  "tldw_chatbook/Character_Chat/world_info_regex.py",
  "tldw_chatbook/Persona_Visual/importer.py"
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
