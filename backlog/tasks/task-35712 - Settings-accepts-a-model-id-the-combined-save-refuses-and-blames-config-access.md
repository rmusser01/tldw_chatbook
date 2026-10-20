---
id: TASK-35712
title: Settings accepts a model id the combined save refuses and blames config access
status: To Do
created_date: 2026-10-20 09:30
labels:
- follow-up
- settings
- providers
- model-config
priority: low
references:
- tldw_chatbook/Chat/provider_setup_persistence.py
- tldw_chatbook/UI/Settings_Modules/providers_models_card.py
- tldw_chatbook/UI/Screens/settings_screen.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of TASK-35011 (2026-10-20). Settings ▸ Providers & Models takes a Default model id of any length: the `settings-model-value` Input sets no `max_length`, and a Custom ID is staged as typed. The combined provider save refuses any model id over 120 characters (`_MAX_MODEL_CHARS` in provider_setup_persistence.py; `persist_provider_settings_atomic` raises `ValueError("Provider settings model is invalid.")` before it validates anything else). Settings catches that ValueError as a failed write, so the user sees "Provider settings not fully applied: the file was not written. Draft retained; check config access and retry". Retrying fails the same way, and the message sends the user looking at file permissions instead of at the id. Discovered ids are already capped at 120 by the discovery cache, so this is reached through a typed Custom ID (or a hand-edited draft), not through Discover models.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A model id the save cannot write is refused where it is entered, with a message that names the limit, before s is pressed
- [ ] #2 If such an id still reaches the save, the message names the model id as the cause, not config access
- [ ] #3 Settings and the combined save agree on one limit, defined in one place
<!-- AC:END -->
