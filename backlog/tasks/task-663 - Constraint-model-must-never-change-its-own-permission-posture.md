---
id: TASK-663
title: 'Constraint: the model must never be able to change its own permission posture'
status: To Do
assignee: []
created_date: '2026-07-25'
labels: [tools, security, agents, constraint]
dependencies: [TASK-545, TASK-627]
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Standing constraint**, filed from the comparative spike behind TASK-627 (`Docs/superpowers/specs/2026-07-25-builtin-tool-permissions-ui-design.md`), to be checked against any future permission-mode work (TASK-662) or any other feature that gives the agent runtime a tool capable of touching its own permission state — not a defect against code that exists today.

CheetahClaws (`SafeRL-Lab/cheetahclaws`) ships `EnterPlanMode`/`ExitPlanMode` as built-in tools that are **always auto-approved** (they are special-cased ahead of the normal permission check) and whose handlers **mutate `permission_mode` directly** — the same field every other tool call's permission decision is computed from. The model can therefore call `ExitPlanMode` to widen its own authority mid-run, and that specific call is exempt from the approval gate that governs everything else it does. The mechanism that is supposed to constrain the agent is, for this one pair of tools, controlled by the agent.

This repo has no equivalent today: nothing in `Agents/`, `Chat/console_agent_bridge.py`, or the built-in tool catalog lets a tool call change `BuiltinToolGate`'s resolved state, the permission-store payload, a kill-switch value, or (if TASK-662 is ever built) a permission mode. This task exists to keep it that way as new capability is added, and to give reviewers a named thing to check a design against rather than re-discovering the CheetahClaws failure mode from scratch each time.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] No built-in tool, MCP tool, or agent-runtime primitive can write to the permission-store payload, change a resolved `EffectiveToolState`, toggle the kill switch, or (if it exists) change a permission mode, as a *side effect of being invoked by the model* — any such change must originate from a human-driven UI action (the Permissions-mode matrix, kill-switch toggle, or equivalent), not a tool call's own handler
- [ ] If a "plan mode" or equivalent construct is ever introduced (TASK-662), entering or exiting it is not a tool the model can call with standing auto-approval; any mode transition available to the model goes through the same review/approval path as every other consequential action, not a special-cased bypass
- [ ] This constraint is referenced explicitly in the design/spec of any future task that adds a permission-mode axis or any tool capable of touching agent-runtime configuration, so it is checked at design time rather than caught in review
- [ ] A regression test (or a design-review checklist item, if no such tool exists yet to test against) exists that would fail if a future built-in tool's handler is wired to mutate permission state directly
<!-- AC:END -->
