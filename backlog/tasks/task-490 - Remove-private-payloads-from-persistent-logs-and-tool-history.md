---
id: TASK-490
title: Remove private payloads from persistent logs and tool history
status: To Do
assignee: []
created_date: '2026-07-23 13:55'
labels:
  - security
  - privacy
  - logging
  - tools
dependencies:
  - TASK-488
priority: high
---

## Description

Prevent prompts, provider payloads, response bodies, API-key fragments, tool
arguments, and tool results from being retained in normal logs or unbounded
execution history while preserving useful operational diagnostics.

## Acceptance Criteria

- [ ] Normal and debug persistent logs contain metadata rather than prompt payload or response content.
- [ ] API keys and partial API-key fragments are never logged.
- [ ] Tool execution logs contain tool identity, status, timing, and argument names without argument values.
- [ ] Tool execution history is bounded and stores payload-free metadata while tool call return values remain unchanged.
- [ ] Provider and summarization error logging excludes raw response bodies.
- [ ] Sentinel-based tests prove representative chat, provider, summarization, and tool payloads are absent from captured logs and history.

## Architecture

- [ADR-022: Local Private Data Boundary](../decisions/022-local-private-data-boundary.md)
- [Local Privacy Containment Design](../../Docs/superpowers/specs/2026-07-23-local-privacy-containment-design.md)
