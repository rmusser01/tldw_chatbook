---
id: TASK-579
title: Live TUI verification of skill-script confirm and revoke
status: Done
assignee: []
created_date: '2026-07-25 14:35'
updated_date: '2026-07-25 18:33'
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
- [x] #1 In a running app, an agent-initiated script run displays the confirm card with the correct skill name, script path, invocation mechanism and arguments
- [x] #2 "Allow once" runs the script and returns its output to the conversation; "Deny" refuses and the script does not execute
- [x] #3 "Always allow this skill" suppresses the prompt on a subsequent run of the same skill
- [x] #4 Editing the skill's content afterwards causes the next run to prompt again
- [x] #5 The Library ▸ Skills trust panel shows the grant state, and "Revoke script access" clears it so the next run prompts again
- [x] #6 Switching conversations while a confirm card is pending does not leave the run blocked
- [x] #7 Findings are captured with screenshots or a terminal capture under Docs/superpowers/qa/
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ALL SEVEN ACs verified live against the running app with real subprocesses. Only the model is simulated; every gate, card and process is genuine.

AC#1 card names skill/script/resolved interpreter. AC#2 Allow -> exit_code 0 + the script's genuine stdout; Deny -> script never executed (transcript kept exactly ONE output). AC#3 Always-allow persists a digest-pinned grant and the next run shows ZERO cards. AC#4 (strongest) editing the script -> quarantined_modified + grant False, and RE-APPROVING TRUST DOES NOT RESTORE IT; the next run re-prompts and executes the NEW content. AC#5 the trust-panel grant line renders, and Revoke script access flips the copy AND clears the on-disk grant to {} — the button->handler->panel-refresh path, previously the single most valuable untested path. AC#6 with a card pending, switching session cleared it and released the run within ~45s (vs the 120s timeout). AC#7 evidence per criterion in Docs/superpowers/qa/skills-script-execution-2026-07-25/.

TWO DEFECTS FOUND, both filed and one already fixed: task-624 (keyring convenience never auto-unlocked — a unit-tested method with ZERO callers; fixed + merged in PR #883) and task-625 (the local-llm provider reads a top-level settings key instead of api_settings.local-llm, making it unusable from its documented config; blocked this UAT until switched to llama_cpp).

OBSERVATIONS not filed: the card clear on session switch is not instantaneous (still rendered at 24s, gone by 45s); opening a NEW TAB with a card pending does NOT clear it (the deny hook is wired to switch_session) — defensible and fail-closed, but a card that follows you into another conversation is worth a UX look.

DRIVING LESSONS for future live QA: clicking a card steals composer focus; this Gemma build needs max_tokens>=3000 or it burns its budget on reasoning and returns nothing; a vague skill body makes it delegate via spawn_subagent which then gets stuck — name the tool and its exact args; the skill row IS a Button (.library-skill-row) so click it directly, and the trust panel needs scrolling to reach.
<!-- SECTION:NOTES:END -->
