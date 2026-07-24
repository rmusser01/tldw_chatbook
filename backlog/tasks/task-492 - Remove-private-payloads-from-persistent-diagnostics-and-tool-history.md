---
id: TASK-492
title: Remove private payloads from persistent diagnostics and tool history
status: Done
assignee:
  - '@codex'
created_date: '2026-07-23 14:23'
updated_date: '2026-07-24 15:31'
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
- [x] #1 Normal and debug persistent diagnostics owned by this task contain metadata rather than prompts, messages, request/response bodies, tool argument values, or tool result values.
- [x] #2 API keys and partial API-key fragments are never written to application or MCP logs.
- [x] #3 A checked repository-wide inventory classifies every Chatbook-owned production diagnostic that reaches a persistent application or MCP sink, assigning remediation to TASK-492 or TASK-494 with a reason for any exclusion.
- [x] #4 Tool diagnostics retain tool identity, status, timing, and only registered schema argument names; unknown argument keys are counted rather than persisted.
- [x] #5 ToolExecutor history is bounded to 100 payload-free metadata records while immediate tool-call return values and bounded in-memory result caching remain unchanged.
- [x] #6 MCP execution records are metadata-only, and error records use sanitized categories/status/exception types without raw exception or response text.
- [x] #7 Parameterized sentinel tests cover success, HTTP/parsing failure, timeout, streaming, and cache paths through standard logging, loguru, MCP JSONL, and the real rotating file sink.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/022-local-private-data-boundary.md
Reason: TASK-492 implements the accepted metadata-only diagnostics and tool-history boundary.

1. Add a file-sink-only metadata admission boundary with standard logging and Loguru sentinel tests.
2. Make ToolExecutor history bounded and payload-free while preserving return/cache behavior.
3. Replace MCP argument/result/error persistence with a metadata-only record schema and migrate legacy generations.
4. Check in and guard the repository-wide production diagnostic inventory and sink topology.
5. Run the focused and parameterized sentinel matrix, reconcile acceptance criteria, and document evidence.

Detailed plan: Docs/superpowers/plans/2026-07-24-metadata-only-high-risk-diagnostics.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-022's high-risk diagnostic boundary. The private rotating application sink now admits only schema-validated metadata records while UI/terminal diagnostics remain available; the Loguru bridge preserves source ownership. ToolExecutor history is a defensive, payload-free 100-record deque with registered argument names, unknown-key counts, timing, cache state, and result metadata while immediate results/cache behavior remain unchanged. MCP JSONL uses a metadata-only public schema and atomically scrubs legacy active/rotated payload rows and torn lines on read or append. Added the checked owner/topology inventory and guard.

Verification: 52 focused sentinel/inventory tests passed under /private/tmp/tldw-task492-494-sentinel-F0irEw/pytest; recursive scan found zero sentinel/key fragments and generated regular logs were 0600. MCP/UI integration: 293 passed. Broad Chat/MCP/Tools/agent run: 1463 passed, 69 skipped; two loopback tests passed with sandbox permission. One Anthropic native-tool test fails identically at baseline commit 0197ec790 and is unrelated. Ruff, compileall, inventory check, and git diff --check passed.

Plan: Docs/superpowers/plans/2026-07-24-metadata-only-high-risk-diagnostics.md
ADR: backlog/decisions/022-local-private-data-boundary.md
<!-- SECTION:NOTES:END -->
