---
id: TASK-42134
title: "Decide whether a Hugging Face send is refused while its saved key is still encrypted"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 20:06
updated_date: 2026-11-21 21:05
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
- [x] #3 Either way, a Hugging Face send with no saved key at all keeps its keyless behaviour, and a test pins it
- [x] #4 `Docs/User_Guide/settings.md` says what a Hugging Face send does while its key is locked
<!-- AC:END -->

## Implementation Plan

1. Record the owner's decision (2026-11-21) in this file.
2. The decision is "no", so the send path does not change; AC#2 does not apply.
3. AC#3 and AC#4 apply either way: pin the keyless send with a test in a lane-run file, and say in `Docs/User_Guide/settings.md` what a send does while the key is still encrypted.

## Implementation Notes

**Owner decision, 2026-11-21 (verbatim): "134 - no".** A Hugging Face send is not refused while its saved `api_settings.huggingface.api_key` is still `enc:` ciphertext. `chat_with_huggingface` keeps reading ciphertext as no key and sends without an `Authorization` header (public Inference API or an unsecured TGI endpoint). Readiness keeps reporting "Saved API key is still encrypted".

- AC#2 does not apply: it was conditional on refusing. The send path is unchanged.
- AC#3: `Tests/Chat/test_local_server_discovery.py::test_a_hugging_face_send_without_a_usable_saved_key_goes_out_keyless[absent|still-encrypted]` (lane-run). It uses a real scratch config with `HUGGINGFACE_API_KEY` unset and the `_cloud_world` stub transport, so nothing leaves the process. With no saved key, and with a still-encrypted one, `chat_api_call("huggingface", ...)` sends a request that has no `Authorization` header and no ciphertext. The test pins an invariant dev already keeps, so red was shown under temporary edits to `chat_with_huggingface`, each reverted. Refusing when the key is missing failed both nodes ("the send was refused"); sending a placeholder `Authorization` failed both on the header assertion. Green after: 2 passed.
- AC#4: `Docs/User_Guide/settings.md` (the paragraph on a provider's own key that is still encrypted) now says that a Hugging Face send is not refused: while its key is still encrypted the provider reads as not ready, but a send made anyway goes out with no key, as with no key saved.
- Correction: the first version of these notes marked AC#3 and AC#4 not applicable on the coordinator's first brief ("No code change"). The coordinator corrected that, because both criteria apply either way.

Files: `Tests/Chat/test_local_server_discovery.py`, `Docs/User_Guide/settings.md`.
