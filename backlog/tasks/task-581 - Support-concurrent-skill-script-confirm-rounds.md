---
id: TASK-581
title: Support concurrent skill-script confirm rounds
status: To Do
assignee: []
created_date: '2026-07-25 14:35'
labels:
  - skills
  - chat
  - reliability
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The skill-script confirm bridge in `console_chat_controller.py` holds a single pending slot (`_pending_skill_script_event` / `_pending_skill_script_decision` / `_pending_skill_script_request_id`). If two confirm rounds are ever armed at once, the second overwrites the first's state and both worker threads then block until their 120s timeout expires.

This currently fails closed and is bounded, so it is not a security defect — and it is not reachable today, because the agent loop dispatches tool calls one at a time on a single worker thread. It becomes reachable the moment anything runs tool calls concurrently, or a second agent run overlaps the first.

The same single-slot shape exists in the sibling MCP approval and skill-install confirm flows, so a fix is worth designing once and applying consistently rather than patching one seam.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Two confirm rounds armed concurrently each resolve to their own decision, with no cross-talk
- [ ] #2 Neither round can be resolved by the other's decision (the per-round request_id remains authoritative)
- [ ] #3 Teardown of one round does not clear another round's pending state
- [ ] #4 A test pins the concurrent case, failing against the current single-slot implementation
- [ ] #5 A decision is recorded on whether the sibling MCP-approval and skill-install flows adopt the same fix
<!-- AC:END -->
