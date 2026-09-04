---
id: TASK-26947
title: Clean Ruff formatter debt for ruff-chat-console-foundation
status: Done
assignee:
  - '@codex'
created_date: '2026-08-31 18:31'
updated_date: '2026-09-04 18:03'
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

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-chat-console-foundation` Ruff formatter batch at the owner boundary recorded as: Console service foundations outside narrower context, fleet, library, observability, and interaction owners.. The focused test surface recorded by TASK-26000 is `["Tests/Chat"]`.
<!-- SECTION:DESCRIPTION:END -->

<!-- TASK-26000-BATCH: ruff-chat-console-foundation -->
<!-- TASK-26000-PATHS-SHA256: c4150a472d5ef3d79bcc9e6795e0db669d8a27268b8cef71d5d9a71e5d86bf5a -->
<!-- TASK-26000-FINAL: false -->

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
- [x] #1 After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [x] #2 Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [x] #3 Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [x] #4 Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and normalized significant-token position within the nearest logical owner (the same-line `except` clause for an `ExceptHandler` header, otherwise the nearest containing AST statement), excluding only parenthesis pairs independently proven AST-neutral by shadow parse/dump equality; preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [x] #5 Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [x] #6 Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [x] #7 `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [x] #8 The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reconcile all 73 TASK-26000 assigned paths against current origin/dev, record upstream lineage and the untouched focused-test baseline, and capture Python 3.12.11 AST/comment/directive evidence.
2. Run Ruff 0.15.22 format with all 73 paths supplied explicitly, reject any unassigned Python diff, and require the structural comparison to match.
3. Run Ruff lint/format checks, mechanically remove any immutable-base unused imports found only in assigned test paths, and run the 55 assigned Console foundation test modules, backlog task-ID uniqueness, and git diff --check; require the final focused result to introduce no failure beyond the untouched origin/dev baseline.
4. Commit only formatter-owned Python changes plus any mechanically lint-fixed assigned test imports, request independent review, then record exact evidence and close TASK-26947 in a task-only commit.

ADR required: no
ADR path: N/A
Reason: Mechanical formatter cleanup under TASK-26000 introduces no architectural, persistence, security, dependency, or long-lived UX decision.

Detailed plan: Docs/superpowers/plans/2026-09-04-task-26947-ruff-chat-console-foundation.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Authority and lineage: TASK-26000 authority cut `e555df102c950c29beed5e7119f433d35eee1f3c`; initial task base `282c733d6120c81d5823262d7bb4f3b61952e896`; strict-latest rebase base `origin/dev` `46b1aed544e1921a5d5ad49958f8ce16d02958ed`. Since `65e11df720`, exactly two assigned paths drifted upstream: `tldw_chatbook/Chat/console_agent_bridge.py` and `tldw_chatbook/Chat/console_runtime.py`; no assigned path was dropped or substituted.
- Strict-latest implementation: formatter commit `d92d10859a`; assigned-test lint commit `68d5bd875a`; one-file post-rebase formatter follow-up `d4acf0243a`. Ruff 0.15.22 formatted 71 of 73 assigned paths (2 were unchanged) and changed no unassigned Python path. A missed blank-line formatter change in `console_agent_bridge.py` was caught after rebase and fixed by the one-file follow-up. The v3 guard preserved ASTs, ordered comments, directive/range metadata, and strict-latest replay passed 73/73: 70 format-only files plus three format-then-safe-F401-fix test files.
- The original physical-NEWLINE guard was corrected twice: nearest logical-owner and AST-neutral-parenthesis handling, then fail-closed same-line `ExceptHandler` header handling. These plan deviations were necessary to avoid rejecting semicolon splitting and formatter grouping parentheses while retaining attachment checks. Ruff then exposed three immutable-base F401 imports in assigned tests; safe removal was limited to those demonstrably unused, side-effect-free bindings, with formatter replay kept separate.
- Ruff `check` and `format --check` passed on all 73 paths, and the 70-path v3 compare (including `console_agent_bridge.py`) passed; base-to-final v3 evidence has only the three expected import AST deltas and no comment/directive/fmt deltas. The exact focused command form was Python 3.12.11 `-m pytest --tb=line --disable-warnings --junitxml=<artifact>` with the 55 literal `Tests/Chat` paths in this task's Assigned Paths order, run once from detached strict-latest current-dev and once at final HEAD. Each JUnit report contains 2,255 cases: 2,201 passed, 52 known failures, 0 errors, and 2 skips; normalized failure/error keys have no additions or removals. Base hash: `94c78a3dc7945933f63f39d40d0eb5b7d6bfb6c27db72e13d7828e7e8b496e3d`; final hash: `862bb0a4dbf03e1c102374aa54f342f1d237bbc95e3bb509d87923d569fee4ab`. This gate is **not green**: it is exact baseline parity. Protected pytest temporary-directory cleanup messages are environment cleanup warnings, not JUnit failures or errors.
- Governance: Backlog uniqueness passed 3/3; `git diff --check` passed; the persistent diagnostic inventory was unchanged and passed with 572 owners / 1337 TASK-492 calls / 7599 TASK-494 calls / 10 sink files, so no diagnostic-inventory refresh was made. Strict-latest evidence report SHA-256: `9620f68c628413261ef8f73709645aaabcf6b624bd07e29466598a985de0b86d`; temporary detached worktrees were removed. Independent formatter, lint-cleanup, and post-rebase evidence reviews found no Critical, Important, or Minor findings. No full suite was run. ADR required: no; no new ADR was needed because this is mechanical formatter cleanup under TASK-26000.
<!-- SECTION:NOTES:END -->
