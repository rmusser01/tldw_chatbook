---
id: TASK-16789
title: Console approval prompts wait indefinitely by default
status: Done
assignee:
  - '@Robert'
created_date: '2026-08-15 21:46'
updated_date: '2026-08-16 02:51'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Blocking human prompts (tool approval card, skill install confirm, skill script confirm) currently auto-deny after 120s. A user who steps away loses the run. Prompts should stay armed until the user answers or the run is stopped, without reopening the abandoned-thread double-execution hazard that the 120s ceiling exists to prevent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tool approval card does not auto-deny when `[mcp] approval_timeout_seconds` is unset (new default 0 = no deadline); it resolves only on user decision or run cancellation
- [x] #2 Approval wait no longer consumes the per-call tool timeout: `_call_with_timeout` pauses its deadline while a human decision is pending for the run; a genuinely hung tool still times out and cancellation still works
- [x] #3 Skill install confirm and skill script confirm prompts also wait indefinitely by default (defaults flip 120 -> 0)
- [x] #4 A configured positive timeout still auto-denies (existing seam values and existing timeout tests keep working)
- [x] #5 Approval card shows no countdown copy when no deadline is armed
- [x] #6 Tests cover wrapper deadline pause, no-deadline rounds for all three prompts, and default resolution
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: a human prompt now waits indefinitely by default, made safe by
pausing — not removing — the per-call tool ceiling. `Agents/human_input_wait.py`
is a new stdlib-only, lock-guarded, refcounted registry keyed by run id
(deliberately NOT a ContextVar: the mark is set on the round's worker thread
and polled on the wrapper's thread). `AgentService._call_with_timeout` gained
an optional `pauses_deadline` predicate; while it polls true the deadline
re-arms each 0.5s slice, so the ceiling counts tool execution time, not human
deliberation — a hung tool still dies on schedule, Stop still cancels within
one slice, and the wait cannot lose the abandoned-thread double-execution race
the old `approval_timeout < max_tool_call_seconds` invariant existed to bound.
`_make_invoke_tool` wires `pauses_deadline=human_input_wait_active(run_id)`.

Controller: `_DEFAULT_{MCP_APPROVAL,SKILL_INSTALL_CONFIRM,SKILL_SCRIPT_
CONFIRM}_TIMEOUT_SECONDS` flip 120 -> 0; each of the three wait loops treats a
resolved timeout <= 0 as "no deadline armed" (nullable `deadline`), wraps its
wait in `use_human_input_wait(owning_run_id)`, and carries the 0 through to
the card payload (the card already renders no countdown for 0). New no-app
guard in `request_mcp_approvals`: an app-less controller can never surface or
resolve a round, so it fails every call closed to "deny" immediately instead
of arming an unresolvable round (mirrors the skill confirms' existing guards;
a wired app with a missing card seam is intentionally NOT guarded — the round
stays resolvable and `_marshal_pending_approval` no-ops).

Decisions: `[mcp] approval_timeout_seconds` keeps its key; positive values
keep the old auto-deny semantics, 0 (new default) waits indefinitely —
confirmed with the user as the desired default, along with covering both
skill confirm prompts, not just the approval card. ADR-067
(`backlog/decisions/067-indefinite-human-approval-waits.md`) records the
superseded invariant; the `max_tool_call_seconds` docstring and the
controller constants were rewritten to cite it. The unconsumed
`UnifiedMCPControlPlaneService.approval_timeout_seconds` (no product callers)
keeps its own 120.0 default — noted in the ADR for any future consumer.

Tests (TDD — each watched fail first): registry semantics incl. cross-thread
visibility and same-run refcounting (`Tests/Agents/test_human_input_wait.py`);
wrapper pause/resume + `_make_invoke_tool` wiring
(`Tests/Agents/test_agent_service.py`); zero-timeout rounds surviving past the
old first-poll bail for all three prompts, run-keyed human-wait marking,
no-app fail-closed, and the 0.0 default resolution + constants pin
(`test_console_mcp_approval.py`, `test_console_skill_install_confirm.py`,
`test_console_skill_script_confirm.py`); countdown-copy-hidden pin
(`test_chat_approval_card.py`).

Verification: Tests/Agents (1428 passed), test_console_mcp_approval (74),
skill install/script confirm suites (17/10/29/5), agent_swap + fleet_wake +
diff_channel + probe --run-slow (60+4), parallel_runs + agent_bridge +
skill_remote_fetch (270), chat_approval_card — all green; mypy adds no new
errors (30 pre-existing in agent_service/console_chat_controller, identical
on HEAD).

Incidental test fixes required by the flip: the script-confirm payload test
now round-trips a positive seam value (45.0) instead of pinning `> 0`; the
cross-session-cancel approval test wires the fake app (an app-less controller
now fails closed before arming, which is that path's own new test);
`make_controller` gained a teardown that fails armed rounds closed so a
failing test can no longer hang interpreter shutdown (lesson recorded in
`backlog/docs/lessons-testing-evidence.md`).

Files: tldw_chatbook/Agents/human_input_wait.py (new),
Agents/agent_service.py, Agents/agent_models.py (comment only),
Chat/console_chat_controller.py, config.py (template comment);
Docs/User_Guide/console/agent-runs-and-tools.md;
Tests/Agents/test_human_input_wait.py (new), Tests/Agents/test_agent_service.py,
Tests/UI/test_console_mcp_approval.py, Tests/UI/test_chat_approval_card.py,
Tests/UI/test_console_skill_install_confirm.py,
Tests/Chat/test_console_skill_script_confirm.py;
backlog/decisions/067-indefinite-human-approval-waits.md.
<!-- SECTION:NOTES:END -->

## Implementation Plan (the how)

1. ADR-067: record the new invariant (human-decision waits pause the per-call clock; timeout <= 0 disables auto-deny)
2. TDD: failing tests for `_call_with_timeout` deadline pause, `human_input_wait` registry, no-deadline approval/confirm rounds, default resolution
3. Implement `Agents/human_input_wait.py` (thread-safe run-id-keyed registry)
4. Add `pauses_deadline` to `_call_with_timeout`; wire from `_make_invoke_tool` via `human_input_wait_active(run_id)`
5. Controller: treat resolved timeout <= 0 as no deadline in `request_mcp_approvals` + both skill confirms; flip the three defaults to 0; wrap waits in `use_human_input_wait(owning_run_id)`
6. Update docs/config example; run suites; task hygiene
