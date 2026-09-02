---
id: TASK-25900
title: 'MCP client: Streamable HTTP and SSE transports'
status: To Do
assignee: []
created_date: '2026-08-31 15:07'
updated_date: '2026-09-02 06:50'
labels:
  - mcp
  - interop
  - parity
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Chatbook's MCP client can only reach servers it spawns as a local process, so the entire hosted-MCP ecosystem is unreachable. Verified on origin/dev: the only transport values in the package are "stdio" (MCP/server.py:1010,1016) and "in_process" (MCP/local_runtime_delegate.py:310,379); MCP/client.py:815,851 connects solely via asyncio.create_subprocess_exec, and MCP/local_store.py records command/args/env with no URL field. Ranked #1 in the 2026-08-31 hermes parity report because MCP is chatbook's declared extension mechanism and remote endpoints (Linear, Sentry, Notion, Stripe, and hermes's 50+ vendor catalog) are all off-limits. Carries a dependency decision that needs recording: adopt the official mcp Python SDK's streamablehttp_client/sse_client versus extending the hand-rolled JSON-RPC client.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A remote MCP server reachable over Streamable HTTP can be added, connected to, and its tools invoked from the Console
- [ ] #2 A remote MCP server reachable over SSE can be added, connected to, and its tools invoked from the Console
- [ ] #3 The server record persists a URL-based target distinctly from a command-based target, and round-trips through the store
- [ ] #4 Remote servers pass through the same permission gate, definition-hash rug-pull guard, and execution log as stdio servers - verified by tests
- [ ] #5 Existing stdio servers continue to work byte-identically; no regression in the stdio path
- [ ] #6 Connection failure, TLS failure, and auth-required responses each surface a distinct, honest readiness state rather than a generic error
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
RESHAPED by ADR-111 Option C (2026-09-02, supersedes the hand-roll plan):
1. [tldw_server repo] Implement Streamable HTTP + SSE ExternalFederationTransport impls in mcp_unified/federation (httpx + websockets deps already present); release; bump the mcp-unified pin here.
2. [chatbook] Wire remote server records through mcp_unified.federation's manager in the hub path: same permission gate, rug-pull hash, execution log as stdio (AC#4); URL-based server records in local_store round-trip distinctly (AC#3); distinct readiness for connect/TLS/auth failures (AC#6).
3. stdio path untouched and extra-free (AC#5); remote servers require the mcp extra.
Cross-repo dependency: step 1 first.
<!-- SECTION:PLAN:END -->
