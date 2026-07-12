---
id: TASK-176
title: >-
  Local-provider first send fails: raise Console gateway 30s timeout and default
  streaming on for local providers
status: To Do
assignee: []
created_date: '2026-07-12 02:47'
labels:
  - ux
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Core-loop UAT 2026-07-11 (Docs/superpowers/qa/core-loop-uat-2026-07): with llama.cpp configured per the setup card, the very first send fails after 30s with 'Provider stream failed: [failed]'. The Console provider gateway hardcodes httpx.AsyncClient(timeout=30.0) and llama_cpp defaults to streaming off, so any large local model exceeds the ceiling on a non-streamed completion. The out-of-box path a local user is funneled into cannot succeed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 First send succeeds out-of-box against a local llama.cpp server whose full generation takes >30s,Non-streaming local completions honor the provider api_timeout (or a >=120s default) instead of a hard 30s client ceiling,Streaming defaults to on for local providers (or the default path otherwise avoids the timeout)
<!-- AC:END -->
