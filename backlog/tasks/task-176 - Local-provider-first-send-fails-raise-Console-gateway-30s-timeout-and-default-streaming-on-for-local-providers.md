---
id: TASK-176
title: >-
  Local-provider first send fails: raise Console gateway 30s timeout and default
  streaming on for local providers
status: Done
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
- [x] #1 First send succeeds out-of-box against a local llama.cpp server whose full generation takes >30s
- [x] #2 Non-streaming local completions honor the provider api_timeout (or a >=120s default) instead of a hard 30s client ceiling
- [x] #3 Streaming defaults to on for local providers (or the default path otherwise avoids the timeout)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed on branch claude/uat-core-loop-2026-07 (PR #606, commits 6fd4a60f..88c0475b) with focused tests; re-verified live against llama.cpp on a fresh profile (remediation captures in Docs/superpowers/qa/core-loop-uat-2026-07).
<!-- SECTION:NOTES:END -->
