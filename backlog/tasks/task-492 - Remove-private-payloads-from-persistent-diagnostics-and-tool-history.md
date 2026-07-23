---
id: TASK-492
title: Remove private payloads from persistent diagnostics and tool history
status: To Do
assignee: []
created_date: '2026-07-23 14:23'
updated_date: '2026-07-23 14:23'
labels:
  - security
  - privacy
  - logging
  - tools
dependencies:
  - TASK-490
references:
  - backlog/decisions/022-local-private-data-boundary.md
documentation:
  - Docs/superpowers/specs/2026-07-23-local-privacy-containment-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent Chat, provider, summarization, ToolExecutor, and MCP diagnostic paths from retaining user, model, credential, argument, or result values while preserving bounded operational metadata.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Normal and debug persistent diagnostics owned by this task contain metadata rather than prompts, messages, request/response bodies, tool argument values, or tool result values.
- [ ] #2 API keys and partial API-key fragments are never written to application or MCP logs.
- [ ] #3 A checked repository-wide inventory classifies every Chatbook-owned production diagnostic that reaches a persistent application or MCP sink, assigning remediation to TASK-492 or TASK-494 with a reason for any exclusion.
- [ ] #4 Tool diagnostics retain tool identity, status, timing, and only registered schema argument names; unknown argument keys are counted rather than persisted.
- [ ] #5 ToolExecutor history is bounded to 100 payload-free metadata records while immediate tool-call return values and bounded in-memory result caching remain unchanged.
- [ ] #6 MCP execution records are metadata-only, and error records use sanitized categories/status/exception types without raw exception or response text.
- [ ] #7 Parameterized sentinel tests cover success, HTTP/parsing failure, timeout, streaming, and cache paths through standard logging, loguru, MCP JSONL, and the real rotating file sink.
<!-- AC:END -->
