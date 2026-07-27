---
id: TASK-1130
title: 'Restored pending_skill_install card is dead-but-clickable'
status: To Do
assignee: []
created_date: '2026-07-27 15:20'
labels: [console, approvals, resume-state]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TaskResumeState.from_dict restores pending_skill_install across screen navigation, but skill rounds live in the controller's request_id-keyed registries and every navigation builds a fresh ChatScreen/ConsoleChatController — so the restored card can mount with no live round behind it; clicking it strict-match no-ops (fail-closed, never auto-approves). TASK-1051 established this chain and deliberately documented the script-side asymmetry (pending_skill_script is dropped); the install side keeps the hazard for round-trip data-fidelity reasons pinned by TASK-910 tests. Either stop restoring pending_skill_install too (mirroring the script decision, updating the fidelity tests) or build a real reconnection path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A restored ChatScreen never shows a skill-install confirm card whose decision cannot reach a live round.
- [ ] #2 Never-auto-approve and round-identity invariants unchanged.
- [ ] #3 TASK-910's round-trip fidelity tests updated coherently with the chosen branch.
<!-- AC:END -->
