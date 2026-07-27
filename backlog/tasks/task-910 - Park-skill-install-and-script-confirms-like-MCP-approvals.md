---
id: TASK-910
title: 'Park skill-install/skill-script confirms like MCP approvals'
status: To Do
assignee: []
created_date: '2026-07-27 03:55'
labels: [console, agents, approvals]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The parallel-agents train (per-session runs) made background runs able to raise skill-install and skill-script confirm cards. Both bridges remain single-slot: a background run's confirm card mounts OVER the currently viewed tab, and switching sessions denies the pending confirm (deny-on-any-switch). MCP approvals got full park/badge/toast/round-identity treatment; the two skill-confirm bridges (request_skill_install_confirm, request_skill_script_confirm in console_chat_controller.py) did not. Fail-closed today, but the interruption and spurious denies degrade the multi-agent UX.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A background session's skill-install/skill-script confirm does not mount over the viewed tab; it parks with the needs-approval marker and a single toast, mounting on visit.
- [ ] #2 Switching sessions no longer denies another session's pending confirm; only that session's own stop/shutdown does.
- [ ] #3 Confirm decisions carry round identity so a decision cannot resolve a different session's confirm.
- [ ] #4 Never auto-approve; timeout behavior unchanged.
<!-- AC:END -->
