---
id: TASK-33091
title: Register pending decision kinds as data in the round registry
status: To Do
assignee: []
created_date: '2026-09-27 19:46'
labels: [refactor, console]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Six pending-decision kinds (approval, skill_install, skill_script, chat_create, user_questions, worktree_merge) each own a five-function family of structurally identical flow — request, resolve, ids, marshal, remount — plus per-kind revoke families and kind-string dispatch ladders, roughly 1.5-2k LOC in console_chat_controller.py. The round registry they feed is already generic; the kinds just are not registered into it. Today a new confirm surface costs five new functions. A kind descriptor (payload schema, marshaler, resolver, timeout policy, revocation) registered into the registry makes new confirm surfaces data; genuinely kind-specific payload enrichment stays as descriptor hooks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A kind descriptor covering payload schema, marshaler, resolver, timeout policy, and revocation registers into the generic round registry.
- [ ] #2 All six existing kinds migrate with behavior parity, including per-kind timeouts and payload enrichment.
- [ ] #3 A test demonstrates that adding a new kind requires only a descriptor.
- [ ] #4 Approval flow and pending-round tests pass.
<!-- AC:END -->
