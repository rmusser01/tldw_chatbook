---
id: TASK-41704
title: "Saving a native cloud provider's own base URL in Settings appends /v1 and breaks its requests"
status: To Do
assignee: []
created_date: 2026-11-18 05:33
updated_date: 2026-11-18 06:23
labels:
- follow-up
- providers
- settings
priority: high
references:
- tldw_chatbook/Chat/provider_setup_persistence.py
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
dependencies:
- TASK-41700
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by review r16cu-rev2 of TASK-41700. `_persisted_endpoint_for_owner` reads a cloud provider's path that has no `v1` segment through the endpoint contract's generic reading, which appends `/v1`. Saving Google's own `https://generativelanguage.googleapis.com/v1beta` writes `.../v1beta/v1`, and the adapter then posts to `.../v1beta/v1/models/m:generateContent` (404). Cohere's `https://api.cohere.com` is saved as `.../v1`, and the adapter posts to `/v1/v2/chat` (404).

The same happens to a gateway base without a `v1` (`https://litellm.corp/gemini/v1beta`), and to DeepSeek's origin, which DeepSeek tolerates. It triggers on an explicit Settings entry, on "Save endpoint & use model", and on an existing configured value.

Confirmed to predate TASK-41522 and TASK-41700 (review r16cu-rev2, identical at 45b1949ca3). Fixing it changes the canonical identity of values accepted today, so project-instruction keys must be handled.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Saving each native cloud provider's own documented base, or a gateway base without `v1`, writes it unchanged, and the adapter posts to the right URL
- [ ] #2 Project-instruction keys for values saved before the fix still resolve, or are migrated, with a test
- [ ] #3 Tests pin Google, Cohere and one gateway base, red before the fix
- [ ] #4 The User Guide's known-gap sentence about the `/v1` append is removed
<!-- AC:END -->
