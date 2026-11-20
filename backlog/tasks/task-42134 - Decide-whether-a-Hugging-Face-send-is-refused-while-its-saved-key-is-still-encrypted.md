---
id: TASK-42134
title: "Decide whether a Hugging Face send is refused while its saved key is still encrypted"
status: To Do
assignee: []
created_date: 2026-11-20 20:06
updated_date: 2026-11-20 20:06
labels:
- follow-up
- providers
- decision
priority: medium
references:
- tldw_chatbook/LLM_Calls/LLM_API_Calls.py
- tldw_chatbook/Chat/provider_readiness.py
- tldw_chatbook/config.py
- Docs/User_Guide/settings.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the providers stream (TASK-42007, round 18). While the saved `api_settings.huggingface.api_key` is still `enc:` ciphertext (a locked session), readiness reports "Saved API key is still encrypted" (`tldw_chatbook/Chat/provider_readiness.py:921-935`). `chat_with_huggingface` (`LLM_Calls/LLM_API_Calls.py:4397`) does not refuse: `resolve_provider_api_key` returns `None` for ciphertext (`config.py:1377`), and the handler logs "API key is missing. Public Inference API or unsecured TGI assumed." and sends with no `Authorization` header (`LLM_API_Calls.py:4435-4449`).

TASK-42007's test (round-18 providers branch) pins this: with a ciphertext slot key, eight bridged providers send nothing through `chat_api_call` and Hugging Face sends without a key. A caller that does not gate on readiness therefore sends a request without the key the user saved, and the user sees the provider's auth error instead of a prompt to unlock the key. A keyless send is valid for a public or self-hosted TGI endpoint, so this needs an owner decision.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner's decision is recorded: whether a Hugging Face send is refused while its saved key is still ciphertext
- [ ] #2 If refused: a send through `chat_api_call` with a still-encrypted Hugging Face key and no pinned key raises before any request is made, and a test shows it, red before the change
- [ ] #3 Either way, a Hugging Face send with no saved key at all keeps its keyless behaviour, and a test pins it
- [ ] #4 `Docs/User_Guide/settings.md` says what a Hugging Face send does while its key is locked
<!-- AC:END -->
