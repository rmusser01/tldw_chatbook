---
id: TASK-16866
title: >-
  Headless skill confirms fail closed instantly and silently, diverging from
  tool approvals
status: Done
assignee: []
created_date: '2026-08-17 00:00'
labels:
  - console
  - agents
  - safety
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #1752 (task-15860 Task 5) made a headless MCP tool approval WAIT and announce itself app-wide: a wake turn firing with Console closed now raises a toast on whatever screen the user is on, and the card is mounted the moment Console opens. Skill install/script confirms were deliberately left untouched by that landing and still take the opposite path: with no view attached their pending-slot is None, which their read site treats as fail-closed-at-once, so the confirm denies immediately and NOTHING surfaces to the user.

That divergence is now a user-visible inconsistency in the same feature: one class of gated action asks the user wherever they are, the other refuses silently. It needs an owner ruling rather than a default, because the two options differ in posture: make skill confirms wait-and-announce like tool approvals (consistent, but skill scripts execute immediately after confirm with no cancel checkpoint - see the tool-approval security stream), or keep them fail-closed and instead SURFACE the refusal so the user learns the skill did not run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The behaviour of a headless skill install confirm and a headless skill script confirm is decided deliberately and recorded, not left as an artefact of a None slot
- [x] #2 Whichever posture is chosen, the user learns the outcome: either a card they can answer from any screen, or a visible notice that the action was refused
- [x] #3 The chosen behaviour is pinned by a test driven through the production path with Console genuinely unmounted
- [x] #4 Docs/User_Guide/console/agent-runs-and-tools.md states the skill-confirm behaviour alongside the tool-approval behaviour
<!-- AC:END -->

## Closeout audit 2026-10-11

Board-prune audit (branch chore/board-prune-c, base a3c47c6c72): completed-but-unmarked. Evidence gathered 2026-10-11 against origin/dev:

AC dispositions:
- AC#1: met - Posture is deliberate and symmetric with tool approvals: request_skill_install_confirm now mirrors request_mcp_approvals' park/mount/retain contract (console_interrupt_rounds.py:4159+, TASK-910) and waits under ADR-067 human_input_wait with no deadline by default; only a genuinely unwired UI fails closed.
- AC#2: met - User always learns the outcome: card mounts immediately for the active session or parks with a sanitized app-wide notice for background sessions (remounted on switch).
- AC#3: met - Tests/Chat/test_console_viewless_hooks.py drives request_skill_install_confirm/request_skill_script_confirm on a detached (viewless) controller.
- AC#4: met - Docs/User_Guide/console/agent-runs-and-tools.md:169 documents approvals queued behind a skill confirmation.
- Implementation: Chat/console_interrupt_rounds.py:4159 (install) and :4268 (script) round registry, park/remount, per-round cancel; Agents/human_input_wait.py marks both prompt classes.
