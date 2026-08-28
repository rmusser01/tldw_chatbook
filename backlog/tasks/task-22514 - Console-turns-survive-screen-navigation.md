---
id: TASK-22514
title: Console turns survive screen navigation
status: To Do
assignee: []
created_date: '2026-08-28 00:35'
updated_date: '2026-08-28 00:38'
labels:
  - console
  - agents
  - architecture
  - navigation
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-27-console-turns-survive-navigation-design.md
  - backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md
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
- [ ] #3 Completion and terminal failure while hidden produce one app-wide notice and a durable local-only Console attention marker that clears only after the owning session renders the terminal result.
- [ ] #4 Stop, session close, and revision-pinned app quit cancel only their intended scope while navigation never signals turn cancellation.
- [ ] #5 Send-time turn context and accepted attachments transfer to app-owned runtime custody without leaking raw content into attention state, logs, exports, sync, or remote payloads.
- [ ] #6 Returning to Console reconciles continuing and completed work from the live app-owned store without duplicate or missing transcript content.
- [ ] #7 Targeted deterministic tests and an isolated live-provider journey verify navigation races, approval timing, multi-session isolation, bounded shutdown, unread restart behavior, privacy, and narrow-terminal navigation rendering.
- [ ] #8 The User Guide removes obsolete screen-scoped cancellation warnings and documents background continuation, approvals, attention markers, and shutdown behavior.
<!-- AC:END -->
