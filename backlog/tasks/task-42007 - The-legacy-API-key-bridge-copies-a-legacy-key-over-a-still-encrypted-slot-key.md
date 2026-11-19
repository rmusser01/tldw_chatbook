---
id: TASK-42007
title: "The legacy API key bridge copies a legacy key over a still-encrypted slot key"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:39
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
- [ ] #1 A still-encrypted slot key is never overwritten by the legacy key, and readiness says the key is still encrypted
- [ ] #2 Tests cover each bridged provider's encrypted-slot case, red before the fix
<!-- AC:END -->
