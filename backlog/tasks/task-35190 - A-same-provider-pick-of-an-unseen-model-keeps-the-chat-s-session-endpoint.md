---
id: TASK-35190
title: A same-provider pick of an unseen model keeps the chat's session endpoint
status: To Do
created_date: 2026-10-17 11:17
dependencies:
- TASK-33006.19
labels:
- model-config-redesign
- follow-up
priority: medium
references:
- tldw_chatbook/Chat/console_chat_controller.py
- tldw_chatbook/Chat/console_session_settings.py
- tldw_chatbook/Widgets/Console/console_model_popover.py
- tldw_chatbook/Widgets/Console/console_settings_field_row.py
- Tests/UI/test_console_endpoint_discovery.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the TASK-33006.19 review, round 2 (2026-10-17). A chat can run on an Endpoint other than the configured URL, for example after a user applies a second local server to one chat (session 127.0.0.1:9200, config :9099). In Chat settings, picking a model on that same provider that the draft has not seen moves the chat to the configured URL: the controller's settings rebaser (`rebase_console_settings_draft`) starts an unseen target from its default chain, and with a clean endpoint draft it takes that chain's base URL, not the chat's session URL. For llama.cpp that base URL is None, and None here means the configured URL, not 'no endpoint': `default_console_session_settings` leaves it unset on purpose, and the configured endpoint is resolved later (the modal shows it through `_initial_base_url_for_provider`). A fix that reads None as 'no endpoint' is wrong. So even a saved-model pick on the same server lands on :9099. Since TASK-33006.19, pick mode offers a model that Test connection listed only while a pick of it stays on the endpoint that listed it, which is correct, but it means a listing at :9200 says '3 models listed' and then Change offers none of them except the current model. The only way to pick them today is to retype the Endpoint. TASK-33006.19's test_a_listing_is_offered_only_while_its_pick_keeps_the_listed_endpoint pins today's reset, so this task changes that test's expectation. The rebaser is not Chat settings' own. Switch model (Alt+M) picks through it too (`ConsoleModelPopover._rebase_to`, on the quick surface, `QUICK_MODEL_DEFAULT_FIELDS`), and a probe at the TASK-33006.19 head (2026-10-17) showed the same reset there: a llama.cpp chat on :9200 with a clean endpoint draft, picking an unseen same-provider model, got base_url None and no endpoint draft on both the full and the quick surface. So the fix belongs in the rebaser, and AC #4 says what Alt+M then does. The rebaser's other two callers do not take this path: the live commit (`ChatScreen._commit_console_settings_submission_live`) rebases a draft onto its own pair, and session adoption (`session.py`) replaces the rebased base_url with the configured one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 In Chat settings, picking a saved model on the chat's own provider keeps the chat's session Endpoint (:9200) instead of resetting it to the configured URL (:9099), pinned by a test
- [ ] #2 A model that Test connection listed at the chat's session Endpoint (:9200) is offered in pick mode without retyping the Endpoint, and picking it keeps the chat on :9200, pinned by a test
- [ ] #3 A pick onto a different provider still takes that provider's own endpoint, so no session URL is carried across providers
- [ ] #4 Switch model (Alt+M) picking an unseen model on the chat's own provider also keeps the session Endpoint (:9200), so it and Chat settings land the same pick on the same Endpoint, and the quick surface still saves no Endpoint as a default, pinned by a test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
