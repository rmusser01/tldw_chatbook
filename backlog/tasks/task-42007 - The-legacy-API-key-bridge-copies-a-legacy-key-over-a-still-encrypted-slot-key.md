---
id: TASK-42007
title: "The legacy API key bridge copies a legacy key over a still-encrypted slot key"
status: Done
assignee: ['@claude']
created_date: 2026-11-19 19:39
updated_date: 2026-11-20 16:21
labels:
- follow-up
- providers
- security
priority: medium
references:
- tldw_chatbook/config.py
- tldw_chatbook/Chat/provider_readiness.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41709 (round 17). For the nine bridged providers, `config._normalize_legacy_provider_api_key` treats an `enc:` value in `api_settings.<provider>.api_key` as no key and copies the legacy `[API]` key over it, so readiness reports Ready with the older key instead of "still encrypted". TASK-41710's Custom 2 bridge already treats ciphertext as a key of its own.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A still-encrypted slot key is never overwritten by the legacy key, and readiness says the key is still encrypted
- [x] #2 For an encrypted slot, the bridge returns no legacy key to the legacy `<provider>_api` dict either, so the provider's own send path and readiness agree (as TASK-41710's Custom 2 rule does)
- [x] #3 Tests cover each bridged provider's encrypted-slot case for both readiness and the send path, red before the fix
<!-- AC:END -->

## Implementation Plan

1. In `config._normalize_legacy_provider_api_key`, a slot `api_key` that is still `enc:` ciphertext is a key of its own (TASK-41710's Custom 2 rule): the legacy `[API]` key is neither copied into the slot nor returned for the legacy dict.
2. Tests in `Tests/Chat/test_local_server_discovery.py` (beside the Custom 2 tests): for each of the nine bridged providers, a real scratch config with an encrypted slot key and a legacy key: readiness says still encrypted, and `chat_api_call` without a pinned key sends nothing; red before.

## Implementation Notes

- `config._normalize_legacy_provider_api_key`: after the env branch, a slot `api_key` that is still `enc:` ciphertext returns `None`, so the legacy `[API]` key is neither copied into the slot nor returned for the legacy `<provider>_api` dict (TASK-41710's Custom 2 rule). The environment variable still outranks it, as readiness's resolver does; a recorded `credential_source` keeps deciding first (`"none"`/`"environment"` return earlier, `"stored"` skips the variable and now stops here).
- Test: `Tests/Chat/test_local_server_discovery.py::test_a_still_encrypted_slot_key_keeps_the_legacy_key_out`, one node per bridged provider (parametrized over `_LEGACY_PROVIDER_API_KEY_BRIDGE`): a real scratch config (`select_config_source`, real loader) with a ciphertext slot key and a legacy key; readiness is `STILL_ENCRYPTED_REASON`, the slot keeps the ciphertext, the legacy dict holds no key, `chat_api_call` without a pinned key raises and no request carries the legacy key or the ciphertext. Eight providers send nothing; Hugging Face sends keyless by design ("Public Inference API or unsecured TGI assumed"), as Custom 2 does with no key.
- Red/green: 9 failed against the previous `config.py` (`assert (True, 'Ready') == (False, 'Saved API key is still encrypted')`); with the readiness assertion removed as a negative control, the send-path half was also red for all 9 (8 on the legacy dict holding `legacy-key-0000`, Google on the key reaching the wire); 9 passed after.
- `config.py` grew 4 lines; `Tests/Architecture/test_module_size_ratchet.py` 8871 -> 8875 with a dated comment.
- Docs: `Docs/User_Guide/settings.md` (the nine legacy keys paragraph).
- Files: `tldw_chatbook/config.py`, `Tests/Chat/test_local_server_discovery.py`, `Tests/Architecture/test_module_size_ratchet.py`, `Docs/User_Guide/settings.md`.
