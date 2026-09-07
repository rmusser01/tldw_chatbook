# ADR-053: MCP-Unified Standalone Runtime Boundary

Status: Accepted
Date: 2026-08-09
Amended: 2026-08-10 (post-rebase inventory reconciliation with TASK-4000)
Related Tasks: [TASK-2512 - Migrate MCP server from FastMCP to tldw_server's mcp-unified package](../tasks/task-2512%20-%20Migrate-MCP-server-from-FastMCP-to-tldw_servers-mcp-unified-package.md); [TASK-1354 - Complete web_search and web_fetch Console and MCP exposure](../tasks/task-1354%20-%20Complete-web_search-and-web_fetch-Console-and-MCP-exposure.md)
Supersedes: N/A

## Decision

Replace the standalone Chatbook MCP server's FastMCP dependency and transport
with the public `mcp-unified==0.2.1` package. Adapt Chatbook's existing tool,
resource, and prompt handlers to the package's narrow `GatewayCoreRuntime`
contract and serve stdio through `mcp_unified.gateway.serve_stdio`.

Keep the adapter Chatbook-owned and deliberately small. It provides only the
decorator-compatible registration surface used by `MCP/server.py`, canonical
result projection, URI-template routing, bounded resource continuation, and
prompt argument validation. It does not import tldw_server's internal
`BaseModule`, `ModuleRegistry`, policy, or application service graph.

The standalone stdio catalog contains the nine implemented legacy built-in
tools plus the explicitly enabled phase-4 filesystem, git, and web tools. The
retired `ingest_media` placeholder remains absent: it fabricated a queued
result without submitting ingestion work, so external clients must use
Chatbook's Library Import flow for persistent URL/file ingestion. The server
does not consume the combined in-app capability manifest and does not expose the eighteen
`library_*` tools. Those Library tools remain owned by ADR-030's in-process
descriptor-backed runtime, including its gated execution path and raw
`tools/call` refusal.

Map existing Chatbook return shapes at the adapter boundary:

- A tool's JSON value is returned raw to `mcp-unified`, which authoritatively
  projects protocol-specific `content` and `structuredContent`.
- An explicit local-tool permission or execution refusal becomes a bounded
  `GatewayToolExecutionError`; ordinary application dictionaries containing
  an `error` field remain application data.
- A resource dictionary becomes a canonical `contents` result. Long text is
  returned in bounded UTF-8 chunks with an opaque continuation URI, exact
  position/total metadata, and content-change detection.
- A prompt message list becomes canonical MCP `messages`. A contiguous
  leading system-instruction block is folded into the first user message;
  system messages elsewhere fail closed because MCP prompt results permit
  only `user` and `assistant` roles.

Use the existing AST-derived built-in descriptor source and the local
provider's real JSON schemas rather than regenerating a second contract.
Prompt descriptors gain their real required/optional arguments. All
standalone tool input schemas reject undeclared properties.

Retain the upstream gateway's bounded validation, rate, concurrency, result,
and shutdown defaults. Synchronous local-agent calls run off the event loop.
Cancellation suppresses late protocol output but does not claim to undo a
filesystem, process, or network side effect that already began in a worker
thread.

The adapter classifies local-tool failures only by exact equality with the
provider's existing stable refusal constants; every other failed
`ToolResult` is a generic local execution failure. Each reason code selects a
fixed public message, and raw provider errors are neither emitted nor logged
by the adapter. Optional local tools are staged and published all-or-none;
composition or staging failure retains the complete built-in catalog and
emits one fixed payload-free diagnostic without an exception or traceback.

Handler metadata is namespaced under
`_meta["tldw.chatbook/resource"]`. Resource continuation cursors carry bounded
offset and SHA-256 state, but are explicitly not an authentication boundary;
authorization remains the resource read itself. Prompt results must be
non-empty, and a leading system block is folded only when immediately
followed by the first user message.

The optional dependency is pinned exactly to `mcp-unified==0.2.1`. Chatbook
keeps its existing `python -m tldw_chatbook.MCP` client configuration and
hand-written MCP client, including legacy `2025-03-26` negotiation. The client
will follow bounded catalog cursors and fail rather than return a partial
catalog when a cursor or page/item bound is invalid. Integration coverage
also pins `2025-11-25` object-only structured results and rejects batching for
every profile except `2025-03-26`.

## Context

The standalone server imports `mcp.server.fastmcp`, while the declared
`mcp[cli]>=1.0.0` extra now resolves to official SDK `mcp==2.0.0`, which no
longer provides that module. A fresh optional-extra install therefore leaves
the documented standalone server unusable. The existing development
environment does not contain the old dependency, so FastMCP binding tests are
skipped rather than proving the server works.

The public `mcp-unified` 0.2.1 release was built and published from
tldw_server after its multi-revision strict-stdio release gates passed. Its
public API supports programmatic stdio serving, tools, resources, resource
templates, prompts, `2025-03-26`, current `2026-07-28`, bounded schema/result
projection, and typed application errors. These resolve the unknowns recorded
when TASK-2512 was filed.

Chatbook has two different local MCP surfaces whose distinction is
security-significant:

1. `TldwMCPServer` is a subprocess-launched external stdio server. It exposes
   the nine implemented legacy built-ins and, when configured,
   permission-gated local-agent tools.
2. `LocalMCPRuntimeDelegate` is an in-process control-plane runtime. Its
   combined manifest also contains eighteen private Library tools, while raw
   protocol `tools/call` is refused and execution proceeds only through the
   gated and logged local-tool action.

Using the combined manifest as the standalone gateway's catalog would attach
private Library descriptors without their in-process dispatch/policy seam and
could accidentally publish them to external clients. The adapter therefore
registers standalone handlers explicitly and treats the combined manifest as
out of scope.

FastMCP previously inferred schemas and prompt arguments from Python
signatures. A transport-only replacement that records generic dictionaries
would lose parameter contracts, surface malformed prompt descriptors, and
defer unknown arguments until a Python `TypeError`. The migration must retain
one schema source and validate before dispatch.

The strict gateway also enforces a result ceiling that the legacy server did
not. Conversations and media resources can exceed that ceiling. Raising the
gateway limits would weaken a reviewed upstream boundary, while silently
truncating would lose data. Continuation-aware resource projection preserves
access without unbounded JSON-RPC lines.

This is a runtime, dependency, wire-contract, cancellation, permission, and
local-data exposure decision, so it requires a canonical ADR under the
repository's architecture rules.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Keep FastMCP as a fallback | The currently declared extra no longer supplies it, two server stacks would double compatibility/security testing, and the owner selected a full migration. |
| Rewrite `server.py` as explicit `GatewayCoreRuntime` maps | It can work but would replace every existing nested handler and AST capability seam, creating unnecessary behavioral churn. |
| Import tldw_server's internal module registry and policy graph | Those are application internals, not the released package contract, and would couple Chatbook to server configuration, auth, and service ownership it does not share. |
| Feed `describe_local_mcp_capabilities()` directly to the standalone adapter | The combined manifest includes eighteen in-app-only Library tools and would violate their distinct execution and privacy boundary. |
| Infer every schema dynamically with Pydantic | It would create a second descriptor authority and can emit dialect-sensitive references; Chatbook already has AST descriptors for built-ins and exact provider schemas for local tools. |
| Treat every mapping with an `error` key as a tool error | Several existing handlers intentionally return structured data containing errors; shape guessing would misclassify valid application results. |
| Relabel prompt `system` messages as `assistant` | That changes instruction authorship and semantics. Folding the leading instruction into the user request is protocol-valid and semantically closer. |
| Raise gateway output limits for large resources | Large single-line responses increase memory and denial-of-service risk and discard the upstream bounded-default guarantee. |
| Claim cancellation stops worker-thread side effects | Python cannot force-kill an arbitrary running thread safely. The honest contract is cancellation of the await/output path, with existing permission and timeout controls around side effects. |
| Restore the old `ingest_media` entry to keep a ten-tool count | Its success response was fabricated and performed no ingestion. Preserving TASK-4000's absence/refusal contract is safer than advertising work that never happens; real ingestion remains a separate feature. |

## Consequences

### Benefits

- Fresh `[mcp]` installs provide the documented standalone server again.
- One reviewed multi-revision stdio implementation replaces obsolete FastMCP
  transport and projection behavior.
- Existing handler code, AST inventory, client command, local-tool permission
  store, and in-app Library runtime remain stable.
- Resource and prompt return shapes become explicit, bounded, and testable.
- Local permission refusals reach MCP clients as authoritative tool errors.
- Current and legacy protocol behavior can be tested against one runtime.
- The standalone server cannot accidentally bypass the in-app Library
  execution policy by reusing its combined manifest.

### Accepted trade-offs

- The adapter is a small Chatbook-owned compatibility layer that must be kept
  aligned with the public `GatewayCoreRuntime` contract.
- Exact pinning requires an intentional dependency update for future
  `mcp-unified` releases.
- Existing non-Library tool outputs that exceed the gateway's safe result
  ceiling fail closed; this migration does not redesign the nine implemented
  legacy tool result contracts.
- Retired `ingest_media` remains unavailable over standalone and in-process
  direct dispatch; persistent ingestion continues through Library Import.
- Resource continuation adds an opaque query token to follow-up reads.
- That token detects stale/mismatched state but is not authenticated; it does
  not add a separate authorization boundary beyond the resource read.
- A leading prompt system block becomes visible inside the first user message
  because MCP has no system role in prompt results.
- Sync local-agent work may finish after its awaiting request is cancelled;
  protocol output remains suppressed, but completed side effects are not
  rolled back.
- HTTP serving remains unsupported. This decision covers stdio only.

## Links

- [TASK-2512](../tasks/task-2512%20-%20Migrate-MCP-server-from-FastMCP-to-tldw_servers-mcp-unified-package.md)
- [TASK-1354](../tasks/task-1354%20-%20Complete-web_search-and-web_fetch-Console-and-MCP-exposure.md)
- [Migration specification](../../Docs/superpowers/specs/2026-08-09-mcp-unified-standalone-server-migration-design.md)
- [ADR-030: Direct Local Library Tool Boundary](030-local-library-agent-tool-boundary.md)
- [ADR-032: Local Agent Tool Permission Boundary](032-local-agent-tool-permission-boundary.md)
- [ADR-033: Local Agent Process Execution Boundary](033-local-agent-process-execution-boundary.md)
