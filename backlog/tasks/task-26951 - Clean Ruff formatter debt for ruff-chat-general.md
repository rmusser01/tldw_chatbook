---
id: TASK-26951
title: Clean Ruff formatter debt for ruff-chat-general
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

<!-- TASK-26000-BATCH: ruff-chat-general -->
<!-- TASK-26000-PATHS-SHA256: 845fb91c16932e965061e116d600f846d1b6f4cbc51468a6a859fe9509dfc3c3 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-chat-general` Ruff formatter batch at the owner boundary recorded as: Remaining cohesive Chat orchestration helpers and direct Chat tests.. The focused test surface recorded by TASK-26000 is `["Tests/Chat"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Chat/test__zz_probe2.py",
  "Tests/Chat/test_anthropic_caching_degrade.py",
  "Tests/Chat/test_change_notes_db.py",
  "Tests/Chat/test_change_turn_tracking.py",
  "Tests/Chat/test_chat_api_call_endpoint_redaction.py",
  "Tests/Chat/test_extract_response_content.py",
  "Tests/Chat/test_fleet_attention.py",
  "Tests/Chat/test_fleet_autowake.py",
  "Tests/Chat/test_fleet_settle_fanout.py",
  "Tests/Chat/test_local_server_discovery.py",
  "Tests/Chat/test_local_thinking_wire_formats.py",
  "Tests/Chat/test_openai_reasoning_sampling_params.py",
  "Tests/Chat/test_reply_sentence_sequencer.py",
  "Tests/Chat/test_turn_context_posture.py",
  "Tests/Chat/test_visual_renderer_decoupling.py",
  "tldw_chatbook/Chat/assistant_generation_state.py",
  "tldw_chatbook/Chat/document_generator.py",
  "tldw_chatbook/Chat/reply_sentence_sequencer.py",
  "tldw_chatbook/Chat/thinking_blocks.py"
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
