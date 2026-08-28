---
id: TASK-22514
title: Console turns survive screen navigation
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-28 00:35'
updated_date: '2026-08-28 01:32'
labels:
  - console
  - agents
  - architecture
  - navigation
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-27-console-turns-survive-navigation-design.md
  - backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md
  - Docs/superpowers/plans/2026-08-27-console-turns-survive-navigation.md
documentation:
  - Docs/superpowers/plans/2026-08-27-console-turns-survive-navigation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make every accepted Console turn continue when the user switches screens. Navigation detaches the Console view without interrupting provider streaming, agent work, tools, or pending human decisions. Stop, session close, and confirmed app exit remain cancellation boundaries. Approved design: Docs/superpowers/specs/2026-08-27-console-turns-survive-navigation-design.md. Architecture decision: backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every accepted normal chat and agent turn continues across real screen navigation without a screen-owned task or dead-view callback.
- [ ] #2 Tool approval, skill-install confirmation, and skill-script confirmation notify once and wait safely while Console is detached.
- [ ] #3 Stop, session close, and revision-pinned app quit cancel only their intended scope while navigation never signals turn cancellation.
- [ ] #4 Returning to Console reconciles continuing and completed work from the live app-owned store without duplicate or missing transcript content.
- [ ] #5 Targeted deterministic tests and an isolated live-provider journey verify navigation races, approval timing, multi-session isolation, bounded shutdown, unread restart behavior, privacy, and narrow-terminal navigation rendering.
- [ ] #6 The User Guide removes obsolete screen-scoped cancellation warnings and documents background continuation, approvals, attention markers, and shutdown behavior.
- [ ] #7 Completion and terminal failure while hidden produce one app-wide notice and a durable local-only Console attention marker that clears only after the owning session renders the matching terminal result.
- [ ] #8 Send-time turn context and accepted attachments transfer to app-owned runtime custody without broadening existing provider requests or leaking raw content into attention state, logs, exports, sync, or unrelated remote APIs.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add app-owned turn custody and transfer prompt ownership before composer clear.
2. Make navigation a generation-fenced pure view detach and freeze all send-time inputs.
3. Retain MCP, skill-install, and skill-script decisions with answerable-time clocks.
4. Atomically persist terminal receipt marks and project hidden outcomes into shell attention.
5. Fence Stop, session close, and app quit to exact scopes with bounded cleanup.
6. Verify real navigation, races, restart durability, privacy, narrow layouts, and an isolated live-provider journey.
7. Update the User Guide and complete targeted review/verification.
<!-- SECTION:PLAN:END -->
