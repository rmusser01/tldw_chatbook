---
id: TASK-267
title: Native tool-calls for Cohere
status: To Do
assignee: []
created_date: '2026-07-17 02:55'
labels:
  - agents
  - providers
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Sibling of the Google conversion task: chat_with_cohere normalizes responses and drops tool-call data, keeping Cohere fence-only after task-263. Convert inside the handler (OpenAI tools/history <-> Cohere tool shapes, both response paths), flip gated on a real API round-trip.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 chat_with_cohere accepts OpenAI-format tools and converts request/response/streaming shapes end-to-end,cohere joins NATIVE_TOOLS_PROVIDERS only after a live round-trip against the real API,Fence fallback intact until the flip (no partial states)
<!-- AC:END -->
