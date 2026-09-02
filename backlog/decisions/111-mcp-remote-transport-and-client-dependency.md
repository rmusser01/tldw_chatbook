# ADR-111: MCP remote transport and client dependency

Status: **Accepted**
Date: 2026-09-01
Accepted: 2026-09-02 — owner chose **Option B** (hand-roll Streamable HTTP + SSE
on the already-present httpx; no new core dependency, single asyncio
concurrency model, transports reuse the existing permission gate, rug-pull
hash, and execution log). TASK-25900 is unblocked; TASK-26032 builds OAuth on
this transport.
Related Task: [TASK-25900](../tasks/task-25900%20-%20MCP-client-Streamable-HTTP-and-SSE-transports.md)
Number swept against local decisions and origin/dev 2026-09-01; 107–109 and 111 were free (max in use locally: 110; on origin/dev: 106). Re-verify at merge time — ADR numbers collide constantly in this repo.

> Accepted 2026-09-02 (Option B). Transport implementation may proceed under
> TASK-25900.

## Context

Chatbook's MCP client can only reach servers it spawns as a local subprocess.
Verified on origin/dev:

- The only transports in the package are `"stdio"` (`MCP/server.py:1010,1016`)
  and `"in_process"` (`MCP/local_runtime_delegate.py:310,379`).
- `MCP/client.py` connects **solely** via `asyncio.create_subprocess_exec`
  (`client.py:851`); the message channel is `_StdioJSONRPCConnection`
  (`client.py:322`), a hand-rolled JSON-RPC-over-stdio connection with a
  stdout `_read_loop`, stdin `_send_message`, pending-request correlation by
  id, per-call timeouts, and transport-failure cleanup.
- `MCP/local_store.py`'s server record persists `command`/`args`/`env` with
  **no URL field**.
- The declared dependency `mcp-unified==0.2.1` is an **optional extra**
  (`pyproject.toml:145`), not the official `mcp` SDK, and is not importable in
  the working venv. The client does **not** use any MCP SDK — it hand-rolls
  JSON-RPC (`"jsonrpc": "2.0"` framing at `client.py:508,536,747,756`).

The 2026-08-31 hermes parity report ranked this the **#1 gap**: MCP is
chatbook's declared extension mechanism, and every remote endpoint (Linear,
Sentry, Notion, Stripe, hermes's 50+ vendor catalog) is off-limits because the
client cannot speak Streamable HTTP or SSE.

TASK-25900 requires adding both transports (AC#1/#2), persisting a URL target
distinctly from a command target (AC#3), routing remote servers through the
**same** permission gate, definition-hash rug-pull guard, and execution log as
stdio (AC#4), keeping the stdio path byte-identical (AC#5), and giving
connection / TLS / auth-required failures **distinct honest readiness states**
(AC#6).

## The decision to make

How do we obtain Streamable HTTP + SSE transport?

### Option A — Adopt the official `mcp` Python SDK

Use the SDK's `streamablehttp_client` / `sse_client` and (likely) its
`ClientSession` for remote servers.

- **Pros:** reference implementation maintained by the spec authors; handles
  protocol-version negotiation, Streamable HTTP's dual request/stream mode,
  SSE reconnection/resumption, and (relevant to the dependent TASK-26032)
  OAuth 2.1 discovery for free; tracks spec churn for us.
- **Cons:** a **new core dependency** for a local-first TUI, pulling in
  `anyio`, `httpx-sse`, and the SDK's own surface. The SDK is built on
  **`anyio` task groups**, not raw `asyncio`; this client is `asyncio`-native
  (raw tasks, `asyncio.wait_for`, `create_subprocess_exec`). Bridging anyio
  scopes into the existing `MCPClient`/`_StdioJSONRPCConnection` lifecycle is a
  real concurrency-model seam, and AC#4/#5 demand the remote path reuse the
  existing gate/hash/log wiring rather than the SDK's own session semantics —
  so we adapt the SDK to our control plane, not the reverse.

### Option B — Hand-roll HTTP/SSE onto the existing client (recommended)

Add `_HttpJSONRPCConnection` (Streamable HTTP) and `_SseJSONRPCConnection`
siblings to `_StdioJSONRPCConnection`, using **`httpx`** (already a core
dependency) for POST + streaming and `httpx`'s response streaming (or a tiny
SSE line parser) for server→client events. Reuse the existing pending-request
correlation, timeout, transport-failure-cleanup, and `MCPClient`-layer
gate/hash/log machinery unchanged.

- **Pros:** **no new core dependency** (httpx is already present); **one
  concurrency model** (asyncio) across all transports; the hard parts
  (id-correlation, timeouts, failure cleanup, permission gate, rug-pull hash,
  execution log) already exist and are reused verbatim, which is exactly what
  AC#4/#5 ask for; smallest blast radius on the stdio path.
- **Cons:** we implement and maintain the transport-level protocol details
  ourselves (session-id header handling, SSE reconnection, Streamable HTTP's
  POST-that-may-return-a-stream), and we track MCP transport spec changes by
  hand. OAuth (TASK-26032) is then also ours to build on top.

## Recommendation

**Option B.** For a local-first TUI whose client already hand-rolls JSON-RPC
by deliberate choice, reusing the existing asyncio connection machinery over
the already-present `httpx` is the smaller, more coherent change and lands
squarely on AC#4/#5 (same gate/hash/log, untouched stdio). The transport
protocol surface we take on is bounded and well-specified; the anyio/asyncio
mismatch and dependency weight of Option A are the larger long-term costs.

Choose **Option A instead** if the owner weighs spec-fidelity and
zero-maintenance protocol tracking (including OAuth discovery for TASK-26032)
above dependency minimalism and a single concurrency model.

## Consequences (if Option B is Accepted)

- A transport-neutral connection interface is extracted from
  `_StdioJSONRPCConnection`; stdio becomes one implementation of it.
- `local_store.py`'s server record gains a `url` + `transport` shape persisted
  distinctly from `command`/`args` (AC#3), defaulting to stdio for backward
  compatibility (AC#5).
- Readiness (`MCP/readiness.py`) gains distinct states for connect-refused,
  TLS-failure, and auth-required (AC#6).
- TASK-26032 (OAuth 2.1) builds the authorization-code + token-refresh flow on
  this hand-rolled HTTP transport.

## Consequences (if Option A is Accepted)

- `mcp` becomes a core (or a clearly-required-for-remote) dependency; the extra
  vs. core placement is decided here.
- An anyio↔asyncio bridge is introduced and its lifecycle/cancellation
  semantics are documented; remote sessions are adapted to the existing
  permission gate, rug-pull hash, and execution log (AC#4).
- TASK-26032 uses the SDK's OAuth support.
