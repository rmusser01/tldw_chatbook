---
id: TASK-42134
title: "Decide whether a Hugging Face send is refused while its saved key is still encrypted"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 20:06
updated_date: 2026-11-21 18:22
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
- [x] #1 The owner's decision is recorded: whether a Hugging Face send is refused while its saved key is still ciphertext
- [x] #2 If refused: a send through `chat_api_call` with a still-encrypted Hugging Face key and no pinned key raises before any request is made, and a test shows it, red before the change (N/A — the decision is not to refuse; the keyless send stays)
- [x] #3 Either way, a Hugging Face send with no saved key at all keeps its keyless behaviour, and a test pins it (N/A per the coordinator's brief — the decision changes no code, so the no-key path is untouched; see notes)
- [x] #4 `Docs/User_Guide/settings.md` says what a Hugging Face send does while its key is locked (N/A per the coordinator's brief — no behaviour changes; see notes)
<!-- AC:END -->

## Implementation Plan

1. Record the owner's decision (2026-11-21) in this file.
2. The decision is "no", so nothing changes in code: tick AC#1 and mark the conditional criteria not applicable with the reason.

## Implementation Notes

**Owner decision, 2026-11-21 (verbatim): "134 - no".** A Hugging Face send is not refused while its saved `api_settings.huggingface.api_key` is still `enc:` ciphertext. `chat_with_huggingface` keeps reading ciphertext as no key and sends without an `Authorization` header (public Inference API or an unsecured TGI endpoint), and readiness keeps reporting "Saved API key is still encrypted".

- No code, test or doc change (coordinator's brief: "No code change").
- AC#2 does not apply: it was conditional on refusing.
- AC#3 and AC#4 are worded "either way" and unconditionally. Per the coordinator's brief they are marked not applicable, because the decision keeps today's behaviour and changes no code. For the record: no existing test pins a Hugging Face send with no saved key at all (the nearest pins are `Tests/Chat/test_encrypted_api_key_never_sent.py::test_huggingface_sends_no_ciphertext`, ciphertext reads as no key and the request goes out without it, and TASK-42007's `test_a_still_encrypted_slot_key_keeps_the_legacy_key_out[huggingface]`), and `Docs/User_Guide/settings.md` does not yet say that a Hugging Face send goes out keyless while its key is locked. Both were reported to the coordinator as a possible follow-up.
