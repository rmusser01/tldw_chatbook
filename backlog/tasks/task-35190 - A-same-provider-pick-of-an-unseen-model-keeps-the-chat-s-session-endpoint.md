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
- tldw_chatbook/Widgets/Console/console_settings_field_row.py
- Tests/UI/test_console_endpoint_discovery.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the TASK-33006.19 review, round 2 (2026-10-17). A chat can run on an Endpoint other than the configured URL, for example after a user applies a second local server to one chat (session 127.0.0.1:9200, config :9099). In Chat settings, picking a model on that same provider that the draft has not seen moves the chat to the configured URL: the controller's settings rebaser starts an unseen target from its default chain, and with a clean endpoint draft that chain's base URL is the configured one, not the chat's session URL. So even a saved-model pick on the same server lands on :9099. Since TASK-33006.19, pick mode offers a model that Test connection listed only while a pick of it stays on the endpoint that listed it, which is correct, but it means a listing at :9200 says '3 models listed' and then Change offers none of them except the current model. The only way to pick them today is to retype the Endpoint. TASK-33006.19's test_a_listing_is_offered_only_while_its_pick_keeps_the_listed_endpoint pins today's reset, so this task changes that test's expectation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 In Chat settings, picking a saved model on the chat's own provider keeps the chat's session Endpoint (:9200) instead of resetting it to the configured URL (:9099), pinned by a test
- [ ] #2 A model that Test connection listed at the chat's session Endpoint (:9200) is offered in pick mode without retyping the Endpoint, and picking it keeps the chat on :9200, pinned by a test
- [ ] #3 A pick onto a different provider still takes that provider's own endpoint, so no session URL is carried across providers
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
