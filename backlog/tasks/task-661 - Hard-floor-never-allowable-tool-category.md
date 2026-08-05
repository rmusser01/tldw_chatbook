---
id: TASK-661
title: Hard-floor / never-allowable tool category
status: To Do
assignee: []
created_date: '2026-07-25'
labels: [tools, security, agents]
dependencies: [TASK-545, TASK-627]
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the comparative spike behind TASK-627 (`Docs/superpowers/specs/2026-07-25-builtin-tool-permissions-ui-design.md`). CheetahClaws (`SafeRL-Lab/cheetahclaws`) puts its destructive-command denylist **inside the execution primitive itself**, below the permission-mode branch — so no permission mode, and no per-tool override, can bypass it. The bash tool refuses a small hardcoded set of commands (e.g. `rm -rf /`) unconditionally, before any mode/allow-list decision is even consulted.

This repo has no equivalent. Today's ceiling is per-tool: `resolve_builtin_state`/`resolve_effective_state` can resolve a tool to `deny`, and TASK-627 made that persistent and reversible for `agent:builtin` — but "Off" is a *decision*, not a *floor*. Nothing stops a stored `allow` (or a future permission-mode axis, if one is ever built) from making a tool the design considers should never be fully unattended-executable available anyway. There is currently no tool in this codebase's catalog that needs this — this task is about having the *mechanism* ready before one is added (e.g. an eventual shell/exec built-in), not about a live gap in `calculator`/`datetime` or the current fs/note tools TASK-545/P2 will port.

**This repo already has the shape to copy.** `Skills_Interop/local_skills_service.py` (task-582, PR #893) defines `MAX_SCRIPT_WALL_CLOCK_SECONDS = 600.0` and applies it as `overrides["wall_clock_seconds"] = min(wall, MAX_SCRIPT_WALL_CLOCK_SECONDS)` — a ceiling a *configuration value* cannot exceed regardless of what the user or a profile requests, justified in-comment because "an unbounded override would strand the turn". A hard-floor tool category needs the identical shape one level up the decision stack: a value a *permission decision* cannot override, checked in the tool's own execution path (or the gate itself, ahead of the resolved state), not expressed as just another entry in the store.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] A mechanism exists for marking a built-in tool (or a specific invocation shape, e.g. a command pattern) as never-allowable, checked ahead of / independent from `resolve_builtin_state`'s resolved decision
- [ ] No stored permission-store state (`allow`, a persistent override, a server default) can cause a hard-floored tool/pattern to execute — a test constructs a store payload that sets `allow` for such a tool and asserts execution is still refused
- [ ] The hard floor is enforced in the tool's own execution path (defense-in-depth, mirroring `BuiltinToolProvider.invoke`'s existing enforcement layer), not only at the UI/permission-store layer
- [ ] The floor value/category is a named constant with an in-comment rationale, following `MAX_SCRIPT_WALL_CLOCK_SECONDS`'s precedent, not a magic literal buried in a conditional
- [ ] The mechanism is generic to "a decision layer above the resolved permission state," not wired specifically to today's per-tool toggles, so a later permission-mode axis (if ever built) can enforce against it without re-deriving a separate floor
<!-- AC:END -->
