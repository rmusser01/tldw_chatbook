---
id: TASK-428
title: Roleplay handoffs stage into a fresh conversation instead of the active tab
status: Done
assignee: []
created_date: '2026-07-21 09:38'
updated_date: '2026-07-22 13:48'
labels:
  - roleplay
  - console
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). Observed live: with an existing Console conversation open, Start Chat staged the character handoff into that same (unrelated, already polluted) tab, and the conversation ends up named after the prefilled meta-instruction ("Continue this con..."). Handoffs from the Roleplay workbench should always land in a fresh conversation/tab so prior context does not bleed into the character chat and vice versa.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Start Chat / Open in Console with another conversation active creates and focuses a new conversation rather than reusing the active tab
- [x] #2 The new conversation is not named after prefilled instruction text
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Builds on TASK-427 (PR #754, stacked): 427 already routed the character "Start Chat" handoff to a fresh, activated session titled "Chat with {name}". This closes the remaining half — the "Open in Console" preview-transcript handoff.

ROOT CAUSE: the Personas preview "Open in Console" handoff (source=personas, item_type=preview-conversation) carries no start_chat intent, so it misses 427's character path and falls into the shared _stage_handoff_as_console_live_work, whose store.ensure_session(...) REUSED whatever Console conversation was active — polluting an unrelated tab and (on send) risking a 'Continue this con...' auto-title.

FIX (Personas-only, chat_screen.py): new module predicate _is_personas_preview_handoff(payload); in the staging method's suggested_prompt block, personas-preview handoffs go through store.create_session(...) (fresh + pre-activated) titled from the sanitized handoff title, seeding the new session's draft. AC#1 (create+focus new conv, not reuse): create_session sets active_session_id and the native sync pass focuses it. AC#2 (not named after prefill): the title is a real, non-default title so the send-time _maybe_auto_title_session leaves it. Every other 'Use in Console' source keeps ensure_session (scope choice: Personas-only).

CORRECTNESS TRAP AVOIDED (design-review finding): the fresh branch does NOT run the inline composer.load_draft — the composer still points at the previously-active session, and _sync_console_session_draft (TASK-339 draft-swap) would otherwise save the prefill back into THAT session. Regression-tested (test_open_in_console_does_not_pollute_prior_session_draft).

SCOPE/HONESTY (follow-up, not 428): the fresh 'Open in Console' session uses default settings — no character system_prompt/character_id — so it ISOLATES context (the AC) but does not itself make the continuation 'in character'; a send retains the P0-2 agent-loop limitation. The Start-Chat card-fetch FAILURE fallback deliberately still reuses (427's error path).

TESTS (Tests/UI/test_chat_first_handoffs.py): 3 new (fresh-session AC#1/AC#2, no-cross-session-draft-pollution, non-personas-reuse scope guard), all RED->GREEN. Full handoff+console suites 231 passed; personas preview 21 passed. Regression: test_console_live_work_handoffs.py 13 failures PROVEN pre-existing (base combined 13 fail/227 pass == with-change 13 fail/227 pass; all schedules/subscriptions, unrelated).

Files: tldw_chatbook/UI/Screens/chat_screen.py, Tests/UI/test_chat_first_handoffs.py.
<!-- SECTION:NOTES:END -->
