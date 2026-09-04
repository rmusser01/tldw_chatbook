---
id: TASK-26947
title: Clean Ruff formatter debt for ruff-chat-console-foundation
status: In Progress
assignee:
  - '@codex'
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
  - Docs/superpowers/plans/2026-09-04-task-26947-ruff-chat-console-foundation.md
priority: medium
---

<!-- TASK-26000-BATCH: ruff-chat-console-foundation -->
<!-- TASK-26000-PATHS-SHA256: c4150a472d5ef3d79bcc9e6795e0db669d8a27268b8cef71d5d9a71e5d86bf5a -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-chat-console-foundation` Ruff formatter batch at the owner boundary recorded as: Console service foundations outside narrower context, fleet, library, observability, and interaction owners.. The focused test surface recorded by TASK-26000 is `["Tests/Chat"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Chat/test_console_agent_bridge_cancel_all.py",
  "Tests/Chat/test_console_agent_bridge_local.py",
  "Tests/Chat/test_console_agent_bridge_steering.py",
  "Tests/Chat/test_console_agent_lesson_approval.py",
  "Tests/Chat/test_console_agent_lesson_promotion_approval.py",
  "Tests/Chat/test_console_agent_swap.py",
  "Tests/Chat/test_console_agent_tool_result_cap.py",
  "Tests/Chat/test_console_auto_speak.py",
  "Tests/Chat/test_console_capture_policy_repository.py",
  "Tests/Chat/test_console_capture_purge.py",
  "Tests/Chat/test_console_chat_controller_exchanges.py",
  "Tests/Chat/test_console_chat_fork.py",
  "Tests/Chat/test_console_chat_fork_persistence.py",
  "Tests/Chat/test_console_chat_store_before_first.py",
  "Tests/Chat/test_console_chat_store_exchanges.py",
  "Tests/Chat/test_console_chat_store_parent_persist.py",
  "Tests/Chat/test_console_chat_store_summary.py",
  "Tests/Chat/test_console_chat_store_tree.py",
  "Tests/Chat/test_console_close_during_durable_postcommit.py",
  "Tests/Chat/test_console_conversation_actions.py",
  "Tests/Chat/test_console_conversation_hydration.py",
  "Tests/Chat/test_console_durable_commit_offload.py",
  "Tests/Chat/test_console_exchange_capture.py",
  "Tests/Chat/test_console_exchange_export.py",
  "Tests/Chat/test_console_first_chat_handoff.py",
  "Tests/Chat/test_console_gateway_tls_trust.py",
  "Tests/Chat/test_console_generate_video.py",
  "Tests/Chat/test_console_generation_store.py",
  "Tests/Chat/test_console_hands_free.py",
  "Tests/Chat/test_console_history_budget.py",
  "Tests/Chat/test_console_image_view.py",
  "Tests/Chat/test_console_prompt_queue.py",
  "Tests/Chat/test_console_prompt_queue_coordinator.py",
  "Tests/Chat/test_console_provider_continuation.py",
  "Tests/Chat/test_console_provider_failure_copy.py",
  "Tests/Chat/test_console_provider_gateway.py",
  "Tests/Chat/test_console_rail_priority_no_eviction.py",
  "Tests/Chat/test_console_rail_state.py",
  "Tests/Chat/test_console_raw_cli_persistence.py",
  "Tests/Chat/test_console_raw_shell_progress.py",
  "Tests/Chat/test_console_raw_shell_revocation.py",
  "Tests/Chat/test_console_realtime_loop.py",
  "Tests/Chat/test_console_settings_apply_store.py",
  "Tests/Chat/test_console_side_chat_service.py",
  "Tests/Chat/test_console_skill_script_confirm.py",
  "Tests/Chat/test_console_skill_substitution.py",
  "Tests/Chat/test_console_speech_preferences.py",
  "Tests/Chat/test_console_style_picker.py",
  "Tests/Chat/test_console_turn_file_entries.py",
  "Tests/Chat/test_console_turn_preparation.py",
  "Tests/Chat/test_console_user_sibling_nav.py",
  "Tests/Chat/test_console_video_capacity.py",
  "Tests/Chat/test_console_viewless_hooks.py",
  "Tests/Chat/test_console_voice_dictation_model.py",
  "Tests/Chat/test_console_voice_input.py",
  "tldw_chatbook/Chat/console_agent_bridge.py",
  "tldw_chatbook/Chat/console_capture_policy_repository.py",
  "tldw_chatbook/Chat/console_conversation_actions.py",
  "tldw_chatbook/Chat/console_conversation_hydration.py",
  "tldw_chatbook/Chat/console_conversation_markdown.py",
  "tldw_chatbook/Chat/console_ephemeral.py",
  "tldw_chatbook/Chat/console_exchange_export.py",
  "tldw_chatbook/Chat/console_generate_image.py",
  "tldw_chatbook/Chat/console_generate_video.py",
  "tldw_chatbook/Chat/console_hands_free.py",
  "tldw_chatbook/Chat/console_prompt_queue_coordinator.py",
  "tldw_chatbook/Chat/console_provider_gateway.py",
  "tldw_chatbook/Chat/console_provider_support.py",
  "tldw_chatbook/Chat/console_raw_cli.py",
  "tldw_chatbook/Chat/console_realtime_loop.py",
  "tldw_chatbook/Chat/console_runtime.py",
  "tldw_chatbook/Chat/console_scratch_space.py",
  "tldw_chatbook/Chat/console_turn_preparation.py"
]
```

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [ ] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [ ] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [ ] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and normalized significant-token position within the nearest logical owner (the same-line `except` clause for an `ExceptHandler` header, otherwise the nearest containing AST statement), excluding only parenthesis pairs independently proven AST-neutral by shadow parse/dump equality; preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [ ] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [ ] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [ ] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [ ] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reconcile all 73 TASK-26000 assigned paths against current origin/dev, record upstream lineage and the untouched focused-test baseline, and capture Python 3.12.11 AST/comment/directive evidence.
2. Run Ruff 0.15.22 format with all 73 paths supplied explicitly, reject any unassigned Python diff, and require the structural comparison to match.
3. Run Ruff lint/format checks, the 55 assigned Console foundation test modules, backlog task-ID uniqueness, and git diff --check; require the post-format focused result to introduce no failure beyond the untouched origin/dev baseline.
4. Commit only formatter-owned Python changes, request independent review, then record exact evidence and close TASK-26947 in a task-only commit.

ADR required: no
ADR path: N/A
Reason: Mechanical formatter cleanup under TASK-26000 introduces no architectural, persistence, security, dependency, or long-lived UX decision.

Detailed plan: Docs/superpowers/plans/2026-09-04-task-26947-ruff-chat-console-foundation.md
<!-- SECTION:PLAN:END -->
