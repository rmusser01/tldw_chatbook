---
id: TASK-32236
title: >-
  Library Search/RAG: the blocked RAG Answer panel repeats one sentence four
  times and names an env var instead of Settings
status: To Do
assignee: []
created_date: '2026-09-10 14:52'
labels:
  - library
  - search-rag
  - copy
  - critique-9
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With no provider the panel prints `Blocked.` / `Unavailable: …` / `Why: The configured provider has no usable API key. Set OPENAI_API_KEY or add api_key under [api_settings.openai].` / `Next: …` / `Recovery: <same sentence>` / `Owner: LLM provider credential.` The Media reader's identical condition says 'Set one in Settings ▸ Providers & Models.' One missing key, two remedies, one of them TOML; 'Owner' is an internal concept. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 5.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The blocked panel is one line in the Media grammar (`No analysis provider is configured · Set one in Settings ▸ Providers & Models.`) plus an 'Open Settings ▸ Providers' action
- [ ] #2 Why/Recovery/Owner are no longer painted; the structured record stays in the log
- [ ] #3 Every provider gate in Library resolves reason and remedy from one place
<!-- AC:END -->
