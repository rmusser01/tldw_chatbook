---
id: TASK-581
title: Support concurrent skill-script confirm rounds
status: Done
assignee:
  - '@claude'
created_date: '2026-07-25 14:35'
updated_date: '2026-07-25 20:41'
labels:
  - skills
  - chat
  - reliability
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The skill-script confirm bridge in `tldw_chatbook/Chat/console_chat_controller.py` holds a single pending slot (`_pending_skill_script_event` / `_pending_skill_script_decision` / `_pending_skill_script_request_id`). If two confirm rounds are ever armed at once, the second overwrites the first's state and both worker threads then block until their 120s timeout expires.

This currently fails closed and is bounded, so it is not a security defect — and it is not reachable today, because the agent loop dispatches tool calls one at a time on a single worker thread. It becomes reachable the moment anything runs tool calls concurrently, or a second agent run overlaps the first.

The same single-slot shape exists in the sibling MCP approval and skill-install confirm flows, so a fix is worth designing once and applying consistently rather than patching one seam.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Two confirm rounds armed concurrently each resolve to their own decision, with no cross-talk
- [x] #2 Neither round can be resolved by the other's decision (the per-round request_id remains authoritative)
- [x] #3 Teardown of one round does not clear another round's pending state
- [x] #4 A test pins the concurrent case, failing against the current single-slot implementation
- [x] #5 A decision is recorded on whether the sibling MCP-approval and skill-install flows adopt the same fix
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the single pending slot with a request_id-keyed registry guarded by a lock.

Before: _pending_skill_script_event / _decision / _request_id were one slot, so a second armed round overwrote the first and BOTH worker threads then blocked to their 120s deadline. Fails closed, and not reachable today because the agent loop dispatches tool calls one at a time on one worker thread — but reachable the moment anything runs them concurrently.

After: self._pending_skill_script_rounds maps request_id -> {event, decision}. Arming inserts; resolving looks up by id and sets only that round's event; teardown pops only its own round. The card surface is cleared only when NO round remains armed — clearing unconditionally would have hidden a sibling round's card, which is the same clobbering bug in a different coat.

_deny_pending_skill_script_on_context_change now denies EVERY armed round rather than just the newest: a conversation switch invalidates the context all of them were raised in.

Added a public pending_skill_script_ids() accessor. This also let me migrate the sibling test module off four private attributes it had been poking (_pending_skill_script_event/_request_id) onto a supported contract — those tests now assert behaviour rather than internals.

AC#5 DECISION: the sibling MCP-approval and skill-install flows keep their single-slot design for now. They share the shape but are equally unreachable today, and changing three HITL surfaces in one pass would widen the blast radius well past what this task justifies. The keyed pattern now exists here as the reference if either becomes concurrent — recorded rather than silently skipped.

Tests: Tests/Chat/test_skill_script_concurrent_confirms.py (4) written RED-first, all four failing against the single-slot implementation. Tests/Chat + the card module: 2208 passed / 69 skipped. ruff clean.

Files: tldw_chatbook/Chat/console_chat_controller.py, Tests/Chat/test_skill_script_concurrent_confirms.py, Tests/Chat/test_console_skill_script_confirm.py
<!-- SECTION:NOTES:END -->
