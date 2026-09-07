---
id: TASK-2819
title: 'Local agent tools phase 1: plumbing + fs_list pilot'
status: Done
assignee: []
created_date: '2026-08-05 00:45'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec: Docs/superpowers/specs/2026-08-04-local-agent-tools-design.md. Plan: Docs/superpowers/plans/2026-08-04-local-agent-tools-phase1.md. ADR: backlog/decisions/032. Build LocalToolProvider + approval-hook generalization + workspace-root config, proven end-to-end with fs_list.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 LocalToolProvider lists/schemas/invokes fs_list through the agent runtime loop
- [x] #2 Approval card gates fs_list with allow/session/always/deny wired to the permission store under local:__local__
- [x] #3 Kill switch and fail-closed no-callback paths return the pinned refusal strings
- [x] #4 workspace_root and local_tools_enabled config keys coerce and default correctly
- [x] #5 All new tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-08-04-local-agent-tools-phase1.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

Implemented the phase-1 plumbing on branch `feat/local-agent-tools-p1` via subagent-driven development with per-task spec + quality review.

- `Utils/path_validation.py`: `validate_path` gained `allow_hidden` (default False, byte-identical for existing callers).
- `Tools/local_tool_impls.py` (new): sync cores — `resolve_workspace_path` (confines to workspace root, hidden components allowed under root) and `list_directory` (fs_list).
- `Agents/local_tool_provider.py` (new): `LocalToolProvider` implementing the `ToolProvider` protocol with MCP-parity approval discipline: clear-first per-turn stamps, fail-closed `invoke()` with pinned refusal strings (`LOCAL_DENY_REFUSAL`/`LOCAL_TIMEOUT_REFUSAL`/`LOCAL_KILL_SWITCH_REFUSAL`), `stamp_scope()` sub-agent isolation, 32 KiB byte-fitting, injected `resolve_state`/`kill_switch`/`approval_callback`/`is_session_approved`/`persist_approval` seams.
- `config.py`: `[console] local_tools_enabled` (default False) + `workspace_root` (default "" = cwd), coerced + templated.
- `Chat/console_chat_controller.py`: `build_local_review_hook` + `build_combined_review_hook` (clear-first across all providers even when an earlier hook raises) + `_compose_local_provider` (permission store under `local:__local__` via `service.gate_tool_test`; `always_allow` persisted with `definition_hash`; `approve_session` via `approve_for_session`).
- `Chat/console_agent_bridge.py`: local provider registered per-run (order Builtin → Local → Skill → MCP), `_combined_review_state_scope` composes both providers' stamp scopes, skill/MCP collision filters include local names so `fs_*` can never be shadowed.
- Tests: provider (29), hooks (15), bridge (5), impls (4), config (2), end-to-end fence-protocol integration (2, approve + deny paths through `AgentService.run_turn`).

Review-driven hardening beyond the initial plan: fail-closed verdict fallthrough (garbage decisions refuse), all injected callables guarded against raising across the worker-thread boundary, real args threaded to the approval card, combined-hook clear-first on exception, skill-shadowing collision fix.

Trade-offs / deviations: `fs_list` landed in phase 1 as the pilot (spec had it in phase 2 — phase 2 task should drop it); deny/timeout audit recording (`record_tool_decision`) deliberately not wired for local tools this phase (documented in `_compose_local_provider`, track for phase 2); AGENTS.md Tool-Calling rewrite deferred to phase 2/3 per plan.

Final whole-implementation review: Ready to merge; all 5 ACs verified. Branch base..HEAD test runs: 546 passed + 1078 passed (two pre-existing failures on the base: anthropic native tools, github api client).
