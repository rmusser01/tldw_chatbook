---
id: TASK-41144
title: "Detaching a custom endpoint whose family is unknown moves its conversations to a provider named after the family string"
status: To Do
created_date: 2026-11-15 06:30
labels:
- follow-up
- settings
- providers
- custom-endpoints
priority: low
references:
- tldw_chatbook/UI/Screens/settings_provider_view_model.py
- tldw_chatbook/Chat/custom_endpoint_registry.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top d4e8d51ca8 (endpoint stream b3c75c8234, on dev ecad404bd4); line numbers are that tree's.

**Why.** A refused entry with a family Chatbook does not know (for example one hand-written in `config.toml`) can still be deleted with its conversations detached. Those conversations then point at a provider no adapter serves, so every later send from them fails. Found by review r14ep-rev4 (declined there as pre-existing, #4).

**Evidence.** `detach_and_delete_entry` (`settings_provider_view_model.py:543-`) loads the entry with `include_refused=True` (`:576-579`) and moves each referencing conversation to `provider=family_execution_key(entry.family)` (`:581-591`). `family_execution_key` (`custom_endpoint_registry.py:592-603`) returns `"custom"` only for `openai_compatible` and passes any other string through; `ENDPOINT_FAMILIES` is `llama_cpp`, `openai_compatible` and `ollama` (`:41`), and `validate_entry` refuses any other family (`:575-577`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Detaching an entry whose family is not in ENDPOINT_FAMILIES either refuses with a named reason or moves its conversations to a provider an adapter serves; the notes record which
- [ ] #2 A test detaches such an entry and checks where its conversations end up, red before the change
<!-- AC:END -->
