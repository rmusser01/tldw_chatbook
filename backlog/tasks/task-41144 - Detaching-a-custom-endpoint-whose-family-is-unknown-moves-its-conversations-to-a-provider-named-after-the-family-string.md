---
id: TASK-41144
title: "Detaching a custom endpoint whose family is unknown moves its conversations to a provider named after the family string"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-15 19:11'
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
- [x] #1 Detaching an entry whose family is not in ENDPOINT_FAMILIES either refuses with a named reason or moves its conversations to a provider an adapter serves; the notes record which
- [x] #2 A test detaches such an entry and checks where its conversations end up, red before the change
<!-- AC:END -->

## Implementation Plan

1. `_detach_refusal` refuses, with a named reason, to move conversations of an entry whose family is not in ENDPOINT_FAMILIES (no adapter serves the family string); an entry with no conversations still deletes.
2. Test: detach such an entry with a referencing conversation; nothing moves and the copy names the entry. Red first.

## Implementation Notes

**Chosen: refuse with a named reason** (AC#1 allows either). Moving the conversations to a guessed provider (for example `custom`) would send their content over a protocol the entry never named, so Detach refuses instead. `_detach_refusal` returns, before any URL check, when the entry's family is not in `ENDPOINT_FAMILIES` and at least one conversation would move: "Detach blocked: '<name>' has an endpoint family Chatbook does not know, so its conversations have no provider to move to. Switch those conversations to another provider, then delete it." Nothing moves and nothing is deleted. The copy names the entry and never the family string or the URL (the family is untrusted config text). An unknown-family entry with no conversations still deletes, so a bad hand-written entry can always be removed.

- Red/green: `test_detach_never_moves_conversations_to_an_unknown_family` (with and without a conversation-own URL) was red at base (`DID NOT RAISE DetachRefused`; the conversation moved to provider `vllm_native`), green after. `test_an_unknown_family_entry_with_no_conversations_still_deletes` is a control, green at base and after.
- User Guide: `Docs/User_Guide/settings.md` (Custom endpoints ▸ Delete row) states the new refusal.
- Files: `tldw_chatbook/UI/Screens/settings_provider_view_model.py`, `Tests/Chat/test_local_server_discovery.py`, `Docs/User_Guide/settings.md`.
