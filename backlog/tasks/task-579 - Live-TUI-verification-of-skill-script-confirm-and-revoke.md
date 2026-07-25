---
id: TASK-579
title: Live TUI verification of skill-script confirm and revoke
status: To Do
assignee: []
created_date: '2026-07-25 14:35'
labels:
  - skills
  - qa
  - ui
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The trust-gated skill script execution feature (PR #871) is covered by unit, widget and end-to-end tests, but no part of its UI has ever been driven in a running application. Two surfaces carry real risk if their wiring is subtly wrong, because both are security controls the user relies on:

- the in-chat confirm card (Allow once / Always allow this skill / Deny), including the per-round `request_id` handshake — if the id is not echoed correctly the resolve is silently dropped by design, the card appears to do nothing, and the agent's worker thread blocks until its 120s timeout;
- the Library ▸ Skills "Revoke script access" button, which is the user's only way to withdraw a skill's standing permission.

A silently broken wire-up on either would leave every automated test passing while the control does nothing. This task is the live smoke pass that closes that gap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 In a running app, an agent-initiated script run displays the confirm card with the correct skill name, script path, invocation mechanism and arguments
- [ ] #2 "Allow once" runs the script and returns its output to the conversation; "Deny" refuses and the script does not execute
- [ ] #3 "Always allow this skill" suppresses the prompt on a subsequent run of the same skill
- [ ] #4 Editing the skill's content afterwards causes the next run to prompt again
- [ ] #5 The Library ▸ Skills trust panel shows the grant state, and "Revoke script access" clears it so the next run prompts again
- [ ] #6 Switching conversations while a confirm card is pending does not leave the run blocked
- [ ] #7 Findings are captured with screenshots or a terminal capture under Docs/superpowers/qa/
<!-- AC:END -->
