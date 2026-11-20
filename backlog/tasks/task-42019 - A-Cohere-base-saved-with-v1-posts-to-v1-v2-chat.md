---
id: TASK-42019
title: "A Cohere base saved with /v1 posts to /v1/v2/chat"
status: Done
assignee: ['@claude']
created_date: 2026-11-19 19:39
updated_date: 2026-11-20 15:58
labels:
- follow-up
- providers
priority: low
references:
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
- tldw_chatbook/LLM_Calls/LLM_API_Calls.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by review r17cu-rev1 (round 17). Typing `https://api.cohere.com/v1` still saves `/v1` (the rule for a path whose `v1` is last), and the Cohere adapter appends `/v2/chat`, so it posts `/v1/v2/chat`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Cohere base ending in `/v1` on Cohere's own host saves as the bare host, the same rewrite TASK-42002 applies to values already saved
- [x] #2 A test covers it, red before the fix
<!-- AC:END -->

## Implementation Plan

1. Shares TASK-42002's rule: the contract reads Cohere's own-host `/v1` (and the other listed shapes) as the documented base, so every save path that resolves through the contract writes the bare host.
2. Test in `Tests/Chat/test_local_server_discovery.py`: typing `https://api.cohere.com/v1` saves `https://api.cohere.com` and the adapter posts `/v2/chat` under it; red before.

## Implementation Notes

Fixed by TASK-42002's rule in the endpoint contract (`_V1_APPENDED_PATHS`, `_read_v1_appended_path`). Every save that resolves through the contract (Settings' **s**, Console's Save endpoint & use model, first-run setup, Make default) now writes Cohere's own-host `https://api.cohere.com/v1` as the bare host. The adapter then posts `https://api.cohere.com/v2/chat`, not `/v1/v2/chat`. It is one rule, so a save also writes the documented base for TASK-42002's other own-host shapes (Google `.../v1beta/v1`; Groq, OpenRouter and QwenCloud `<host>/v1`), exactly the values the load migration would rewrite. A gateway or foreign host's `/v1` saves as typed.

**Test:** `Tests/Chat/test_local_server_discovery.py::test_a_base_saved_with_v1_appended_saves_its_documented_base[cohere]` (and one node per other shape). The save writes the documented base, and the adapter, given that value, posts its own route under it. Red with d85e90bc9e's contract: 5 of 5 (`'https://api.cohere.com/v1' == 'https://api.cohere.com'`, and so on); green after.

Files: see TASK-42002 (same commit).
