---
id: TASK-41709
title: "Dreams sends without checking provider readiness, so keyless and still-encrypted keys reach the handler"
status: To Do
assignee: []
created_date: 2026-11-18 06:57
updated_date: 2026-11-18 06:57
labels:
- follow-up
- providers
priority: medium
references:
- tldw_chatbook/Dreams/story_service.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in review of TASK-41523 (round 16). `story_service.resolve_dreams_chat` takes the readiness result's `api_key` without checking `ready` (~story_service.py:257). For a slot whose stored key is still `enc:` ciphertext, readiness blocks ("still encrypted") while Dreams sends keyless; with an `enc:` slot key plus a plaintext legacy `[API] custom_openai2_api_key` and no `credential_source`, Dreams sends the legacy key while readiness says the key is still encrypted. Ingest checks readiness; Dreams does not. It affects every keyless local provider, not only Custom.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Dreams send whose provider readiness is not ready is refused with readiness's reason and sends nothing
- [ ] #2 Tests cover a still-encrypted slot key and the enc-plus-legacy case, red before the fix
<!-- AC:END -->
