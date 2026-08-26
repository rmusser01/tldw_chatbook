# MCP-Unified Standalone Server Migration Design

**Date:** 2026-08-09

**Status:** Approved for implementation; inventory amended 2026-08-10 after TASK-4000 integration

**Task:** TASK-2512

**ADR required:** yes

**ADR path:** `backlog/decisions/053-mcp-unified-standalone-runtime-boundary.md`

**Reason:** This changes the standalone MCP dependency, runtime and stdio
transport, cross-module adapter contract, cancellation behavior, permission
error projection, and external local-data exposure boundary.

## 1. Goal

Restore and modernize Chatbook's documented standalone MCP server by replacing
the removed FastMCP 1.x surface with the released `mcp-unified==0.2.1`
programmatic stdio gateway.

Success means:

- `python -m tldw_chatbook.MCP` launches a strict standalone stdio server from
  a fresh `tldw_chatbook[mcp]` installation.
- The nine implemented legacy built-in tools, five resource templates,
  dynamic resource catalog, five prompts, and optionally exposed phase-4 local-agent tools are
  present and callable through `mcp-unified`.
- The server negotiates Chatbook's existing `2025-03-26` client and the
  package's current `2026-07-28` protocol behavior.
- Existing Chatbook JSON return shapes are projected canonically and large
  resource content is continuation-safe.
- Local filesystem, git, and web tools keep the current permission store,
  workspace confinement, kill switch, and external-caller fail-closed policy.
- The eighteen in-process `library_*` tools and their raw `tools/call` refusal
  remain unchanged and are not published by the standalone stdio server.
- No production import or declared optional dependency on FastMCP or the
  official `mcp` SDK remains.

Post-rebase correction (approved 2026-08-10): TASK-4000 retired
`ingest_media` because it returned a fabricated queued result without
submitting work. The migration preserves that newer product contract:
`ingest_media` is absent from standalone discovery and refused by direct
in-process dispatch, while persistent ingestion remains available through
Library Import. Restoring the placeholder or implementing real ingestion is
outside TASK-2512.

## 2. Verified Upstream Contract

The migration targets the public PyPI package `mcp-unified==0.2.1`, published
from tldw_server after the following upstream gates completed:

- multi-revision protocol validation and projection;
- programmatic strict binary stdio serving;
- Linux Python 3.10, 3.11, 3.12, and 3.13 coverage;
- Windows Python 3.11 coverage;
- wheel and sdist isolated installation;
- official Python SDK `mcp==2.0.0` stdio interoperability;
- fresh PyPI wheel and sdist hash/install verification.

The public package provides:

- `mcp_unified.gateway.GatewayCoreRuntime`;
- `mcp_unified.gateway.GatewayRequestContext`;
- `mcp_unified.gateway.GatewayApplicationError` and
  `GatewayToolExecutionError`;
- `mcp_unified.gateway.GatewayLimits`;
- `mcp_unified.gateway.serve_stdio(runtime, ...)`;
- tool, resource, resource-template, and prompt catalogs;
- `tools/call`, `resources/read`, and `prompts/get` dispatch;
- protocol revisions `2024-11-05`, `2025-03-26`, `2025-06-18`,
  `2025-11-25`, and current `2026-07-28`;
- batching only where the negotiated profile permits it;
- bounded JSON/schema/result validation and bounded shutdown.

These facts resolve TASK-2512's original resource/prompt, revision, and
programmatic-serving unknowns. Chatbook must use the public gateway API rather
than tldw_server's private `BaseModule`, `ModuleRegistry`, policy, or service
modules.

## 3. Scope

### 3.1 Migrated standalone surface

The standalone adapter registers exactly these existing handlers:

**Built-in tools (9)**

1. `chat_with_llm`
2. `chat_with_character`
3. `search_rag`
4. `search_conversations`
5. `create_note`
6. `search_notes`
7. `list_characters`
8. `get_conversation_history`
9. `export_conversation`
**Resource templates (5)**

1. `conversation://{conversation_id}`
2. `note://{note_id}`
3. `character://{character_id}`
4. `media://{media_id}`
5. `rag-chunk://{chunk_uuid}`

The existing dynamic resource catalog continues to list the five most recent
conversations and five most recent notes.

**Prompts (5)**

1. `summarize_conversation`
2. `generate_document`
3. `analyze_media`
4. `search_and_synthesize`
5. `character_writing`

**Optional local-agent tools**

The current `fs_*`, `fs_patch`, `git_*`, `web_fetch`, `web_search`,
`web_crawl`, and conditionally registered `web_deep_search` catalog remains
behind `[mcp].expose_local_tools`. The standalone composition supplies no
Console `SessionTodoStore`, so `todo_create`, `todo_update`, `todo_get`, and
`todo_list` are unregistered externally. `todo_write` is retired by
TASK-13216 and is also absent.

### 3.2 Explicitly unchanged in-process Library surface

`describe_local_mcp_capabilities()` currently combines the nine implemented legacy
built-ins with eighteen descriptor-backed `library_*` tools. The latter are
owned by ADR-030 and executed by `LocalMCPRuntimeDelegate` through
`LocalLibraryToolService`. Raw in-app protocol `tools/call` remains refused;
the Console executes only through its gated and logged local-tool action.

The standalone adapter must never use that combined manifest as its dispatch
catalog. It consumes `_describe_local_tools()` for the nine standalone
built-ins and explicit registrations for phase-4 local tools. Tests must prove
that:

- all eighteen Library descriptors remain in the in-process manifest;
- no `library_*` name appears in standalone `tools/list`;
- raw in-app `tools/call` remains refused;
- normal gated direct Library execution remains green.

### 3.3 Non-goals

- Replacing or redesigning the in-process local MCP control plane.
- Exposing the eighteen `library_*` tools to external stdio clients.
- Changing the Console `[console].direct_library_tools` retrieval-mode toggle.
- Replacing the hand-written external MCP client with the official SDK.
- Adding HTTP, WebSocket, or network-listening transport.
- Porting tldw_server application modules or its policy database.
- Redesigning the nine implemented legacy tool result contracts.
- Restoring the retired `ingest_media` placeholder or implementing a real
  ingestion submission path; use Library Import for persistent ingestion.
- Adding semantic or embedding search to direct Library tools.
- Claiming worker-thread cancellation reverses completed side effects.

## 4. Architecture

Add one focused module, `tldw_chatbook/MCP/gateway_runtime.py`, containing a
Chatbook-owned runtime adapter. It implements the public
`GatewayCoreRuntime` methods and the optional resource-template method:

- `list_tools(context)`
- `call_tool(name, arguments, context)`
- `list_resources(context)`
- `list_resource_templates(context)`
- `read_resource(uri, context)`
- `list_prompts(context)`
- `get_prompt(name, arguments, context)`

The adapter exposes the small decorator-compatible registration surface that
`MCP/server.py` already uses:

- `tool(...)`
- `resource(uri_template)`
- `list_resources()`
- `prompt(...)`

This retains the existing nested handlers and AST inventory while replacing
only registration, dispatch, canonical mapping, and transport. It is not a
general plugin framework and exposes no unused extension system.

`TldwMCPServer` retains its public identity and `self.mcp` compatibility
attribute, but that attribute holds the new runtime. Its stdio path calls
`await serve_stdio(self.mcp)` and returns the integer status. HTTP continues
to raise `NotImplementedError`.

The package entry point propagates the returned status as the process exit
code. It writes human diagnostics only to stderr; protocol stdout remains
newline-delimited JSON exclusively.

## 5. Registration and Descriptor Authority

### 5.1 Built-in tools

The existing AST walker remains the built-in descriptor authority. Its
`_signature_to_input_schema` result gains `additionalProperties: false`,
matching the actual Python signatures and rejecting undeclared arguments
before dispatch.

The runtime receives the `_describe_local_tools()` descriptors keyed by name.
Decorating a built-in binds its callable to that descriptor. Missing,
duplicate, or descriptor-without-handler identities fail during server
construction rather than producing a partial catalog.

The combined `describe_local_mcp_capabilities()` result is never passed into
the standalone runtime.

### 5.2 Local-agent tools

Each `LocalToolRegistration` already carries the provider's canonical JSON
schema. The new binding passes that schema directly to the adapter. The
FastMCP-only generic `arguments: dict` schema and appended parameter-summary
copy are removed.

The runtime invokes local handlers through `asyncio.to_thread`, keeping sync
filesystem/git/web work off the protocol event loop. Local registrations are
all-or-none: build the complete registration list first, validate every name,
description, object-root input schema, callable handler, duplicate, and
built-in collision against temporary maps, then publish the staged maps with
one update. A composition or staging failure discards the entire optional
local set and leaves the built-in catalog unchanged.

That recoverable failure emits exactly `Local MCP tools unavailable;
continuing with built-in tools.` to stderr, with no exception interpolation
or traceback. A path, secret, raw argument, or provider exception must never
reach stdout or stderr through this diagnostic. The real gateway profile
tests compile every provider schema; an unexpected schema rejection after
publication fails closed instead of serving a partial catalog.

### 5.3 Resources

Each resource decorator registers:

- its literal RFC 6570 template;
- function name and first doc line;
- a compiled, anchored matcher for the template's single identifier;
- its async handler.

Only the five known one-variable custom-scheme templates are supported. The
base matcher rejects path/fragment ambiguity, duplicate templates, malformed
percent encoding, and unknown schemes before invoking a handler. Query
parsing happens first under the exact continuation grammar in section 7; the
matcher itself receives only the normalized query-free base URI.

The dynamic resource list handler returns existing canonical resource
descriptors (`uri`, `name`, optional `description`, `mimeType`).

### 5.4 Prompts

The AST manifest gains prompt `arguments` derived from the same function
signature:

- every public parameter becomes one MCP prompt argument;
- parameters without defaults are `required: true`;
- optional/defaulted parameters are `required: false`;
- argument names are unique and bounded.

The runtime also retains the primitive expected type for dispatch validation.
The five current prompts use only `str`, `int`, and `Optional[str]`, so a small
stdlib coercer is sufficient. It accepts already-correct JSON values and the
string form sent by common MCP clients, rejects booleans as integers, rejects
unknown/missing arguments, and leaves Python defaults to the handler.

## 6. Canonical Tool Mapping

The runtime returns a tool handler's JSON value directly. `mcp-unified` owns
revision-specific projection:

- dictionaries and lists become deterministic JSON text content;
- supported profiles also receive `structuredContent`;
- profiles without structured-content support receive content only;
- invalid, too-deep, non-finite, or oversized results fail closed through the
  package's bounded application errors.

The adapter does not infer an error from a JSON shape. A built-in return such
as `{"error": "..."}` remains structured application data for compatibility.

The local-agent wrapper has explicit knowledge of `ToolResult.ok`. On a
refusal or execution failure it raises/returns `GatewayToolExecutionError`
with a safe message capped by the upstream 512-character limit and one stable
reason code:

| Condition | Reason code | Exact public message |
| --- | --- | --- |
| External call requires operator approval | `operator_approval_required` | `Operator approval is required for this local tool.` |
| Tool permission is Off | `tool_permission_denied` | `This local tool is disabled by operator policy.` |
| Local-tool kill switch is engaged | `local_tools_disabled` | `Local tools are disabled.` |
| Permission state could not be resolved | `permission_state_unavailable` | `Local tool permission state is unavailable.` |
| Provider invocation failed | `local_tool_failed` | `Local tool execution failed.` |

`ToolResult` does not retain the provider's internal verdict, so the adapter
uses exact equality against the provider's existing stable refusal constants:

- `EXTERNAL_NO_CALLBACK_REFUSAL` and `LOCAL_TIMEOUT_REFUSAL` map to
  `operator_approval_required`;
- `LOCAL_DENY_REFUSAL` maps to `tool_permission_denied`;
- `LOCAL_KILL_SWITCH_REFUSAL` maps to `local_tools_disabled`;
- `LOCAL_GATE_ERROR_REFUSAL` maps to `permission_state_unavailable`;
- every other `ok=False` value maps to `local_tool_failed`.

This is deliberately a narrow adapter mapping. It does not widen the shared
`ToolResult` contract or classify by substrings. The adapter never forwards
or logs `ToolResult.error`; it selects only the fixed public message above.

This produces `isError: true`, bounded text content, and gateway-owned error
metadata. It must not include stack traces, secrets, raw paths, SQL, or an
unbounded exception representation.

## 7. Canonical Resource Mapping and Continuation

Existing resource handlers return:

```json
{
  "uri": "conversation://id",
  "name": "Title",
  "mimeType": "text/markdown",
  "content": "...",
  "metadata": {}
}
```

The adapter returns:

```json
{
  "contents": [
    {
      "uri": "conversation://id",
      "mimeType": "text/markdown",
      "text": "..."
    }
  ],
  "_meta": {
    "tldw.chatbook/continuation": {
      "startChar": 0,
      "endChar": 100,
      "totalChars": 100,
      "totalBytes": 100,
      "returnedBytes": 100,
      "hasMore": false,
      "nextUri": null
    },
    "tldw.chatbook/resource": {"message_count": 2}
  }
}
```

Rules:

- One text block is returned per read.
- Text is fitted to at most 256 KiB of UTF-8 before result-envelope overhead,
  remaining comfortably below the gateway's default result ceiling.
- Chunk boundaries never split a UTF-8 sequence.
- `startChar`, `endChar`, `totalChars`, `totalBytes`, and `returnedBytes` are
  exact for the materialized resource text.
- A base read accepts no query parameters and no fragment.
- If more text remains, `nextUri` is the same base resource URI with exactly
  one bounded `tldw_continue` query parameter containing the opaque token.
- A continuation read rejects fragments, duplicate `tldw_continue`
  parameters, empty tokens, and every unknown query parameter. The adapter
  removes and validates that parameter first, then matches the normalized
  query-free base URI to the registered template and handler identifier.
- The token contains a format version, the next character offset, a SHA-256
  digest of the normalized base URI, and a SHA-256 content revision. It
  exposes no content, path, database identifier beyond the already-public
  resource URI, or secret.
- A token is accepted only for the same normalized base URI. Offsets must be
  within the current content bounds.
- If the resource text changes between chunks, the continuation fails with a
  bounded `resource_changed` application error and instructs the caller to
  restart from the base URI. It never silently skips or repeats changed text.
- Malformed, base-mismatched, or out-of-bounds tokens fail as invalid resource
  arguments. The token is an opaque continuation cursor, not an authentication
  or authorization boundary: its unkeyed digests detect stale/mismatched
  state but do not claim to make a client-provided offset unforgeable.
- A non-empty handler `metadata` mapping is recursively JSON-normalized and
  stored only under `_meta["tldw.chatbook/resource"]`. Absent or empty handler
  metadata omits that key. Invalid metadata fails closed; it is never flattened
  or silently dropped, and gateway-reserved keys cannot be overwritten.
- `Not Found` and existing textual resource-error responses remain compatible
  application content; the migration does not redesign backend lookup
  semantics.

The result-level `_meta` mapping is available in every supported profile. The
wire-level connection exposes it as `getattr(result, "_meta")`, and
`MCPClient.read_resource` returns it under the exact public `"_meta"` key so
callers can follow continuation without parsing text. Missing wire metadata
becomes an empty mapping; it is never renamed or merged into resource
content.

## 8. Canonical Prompt Mapping

Existing prompt handlers return a list of `{role, content}` dictionaries.
The adapter returns canonical MCP content blocks:

```json
{
  "messages": [
    {
      "role": "user",
      "content": {"type": "text", "text": "..."}
    }
  ]
}
```

Mapping rules:

- `user` and `assistant` messages pass through as text blocks.
- One or more contiguous leading `system` messages must be immediately
  followed by a `user` message. Their content is joined with two newlines and
  replaces that first user's text with exactly
  `System instructions:\n{joined}\n\nUser request:\n{original}`.
- A system message after any user/assistant message, a trailing system block
  not immediately followed by a user message, an unknown role, a non-string
  content value, or an invalid overall shape fails closed as an invalid
  application result.
- Empty message lists fail closed. All five Chatbook prompt handlers are
  required to return at least one message even though the upstream projector
  accepts an empty list.

This folding is performed only at the external MCP adapter. The underlying
`MCPPrompts.character_writing_prompt` continues to return its internal
system-plus-user form to in-process consumers.

The existing `search_and_synthesize_prompt` currently calls async
`keyword_search` without awaiting it. Migration coverage must first reproduce
the coroutine/iteration failure, then add the missing `await`; a prompt cannot
be declared migrated while one of the five real handlers always returns its
error fallback.

## 9. Protocol, Client, and Lifecycle Compatibility

### 9.1 Negotiation

The server leaves protocol negotiation to `mcp-unified`. Required integration
coverage includes:

- legacy initialize at `2025-03-26` using Chatbook's own hand-written client;
- initialize/catalog/call/read/get at `2025-11-25`, including its object-only
  `structuredContent` behavior;
- current `2026-07-28` request metadata behavior;
- a deterministic unsupported-version error;
- batch acceptance only for `2025-03-26`, with explicit rejection at
  `2025-11-25` and `2026-07-28`;
- notification and cancellation behavior;
- bounded EOF, output failure, and shutdown behavior inherited from the
  package.

### 9.2 Catalog pagination

`mcp-unified` paginates catalogs and returns `nextCursor`. Chatbook's client
currently reads only the first page. Add one private bounded aggregation
helper shared by `list_tools`, `list_resources`, and `list_prompts`:

- request the first page without a cursor;
- only an absent or JSON `null` `nextCursor` ends aggregation;
- follow every valid non-empty string `nextCursor`;
- reject empty or non-string cursors;
- reject repeated cursors;
- accept at most 100 pages and 10,000 items;
- if another page would be required after page 100, or another item would
  exceed 10,000, raise a bounded client error rather than returning a partial
  catalog;
- reject malformed page item arrays;
- preserve original item order.

Current standalone catalogs fit on one page, but following the protocol cursor
prevents future tools or third-party servers from being silently truncated.

### 9.3 Entry point

- `TldwMCPServer.run("stdio")` returns the integer from `serve_stdio`.
- `main()` returns that integer.
- `MCP/__main__.py` exits with it.
- Ctrl-C and fatal diagnostics use stderr only.
- The redundant runtime `sys.path` mutation is removed; installed and source
  module execution must resolve normally.
- `python -m tldw_chatbook.MCP` remains the documented client command. No MCP
  client configuration migration is required.

## 10. Dependency and Availability Migration

Replace both optional dependency declarations:

```toml
mcp = ["mcp-unified==0.2.1"]
```

and replace the `mcp[cli]` entry inside `all-tools` with the same exact pin.

Update every live dependency/availability surface:

- `tldw_chatbook/MCP/__init__.py::is_mcp_available` checks `mcp_unified`;
- `MCP/server.py` conditionally imports the new adapter/gateway;
- `Utils/optional_deps.py` lists `mcp-unified`;
- package-to-import test mappings use `mcp-unified -> mcp_unified`;
- recovery copy and `Docs/Design/MCP.md` install instructions use
  `tldw_chatbook[mcp]` or the new distribution name;
- FastMCP-specific tests and comments become runtime-adapter tests;
- historical specs/tasks remain historical and are not rewritten merely to
  erase the word FastMCP.

The pin is exact because `mcp-unified` is public-alpha and this work is built
against a verified concrete API. A later upgrade is an explicit dependency
review, not an unbounded resolver event.

## 11. Security and Privacy

- Stdio remains subprocess-local and does not open a network listener.
- Any external MCP client that launches Chatbook inherits the user's OS-level
  access to the configured local databases. Documentation must say plainly
  that tools, resources, and prompts can disclose private local Library data
  and that data sent onward to a cloud model leaves the device.
- `[mcp].expose_local_tools` remains default-off.
- Ask-state external local tools fail closed because no Console approval card
  exists in the subprocess.
- Tool permission state is loaded fresh for every call; operator grant/revoke
  and the kill switch take effect without restarting the already-running
  server where the existing provider supports that behavior.
- Workspace confinement and the `local:__local__` permission identity remain
  unchanged.
- The adapter never logs result payloads, prompt text, resource content,
  arguments, secrets, paths, or raw exceptions.
- Resource continuation metadata contains lengths, offsets, a digest, and a
  continuation URI only.
- The eighteen direct Library tools stay outside external stdio exposure.

## 12. Error and Cancellation Semantics

- Unknown tool/prompt names and malformed request arguments are handled by
  the strict gateway's fixed protocol errors.
- Invalid application catalogs/results fail closed without leaking internal
  values.
- Local permission/execution refusals use typed tool errors as defined above.
- Resource continuation errors use bounded application errors with resource
  kind and stable reason codes.
- A runtime task cancellation propagates into async built-in handlers.
- A sync local-agent invocation is awaited through `asyncio.to_thread`.
  Cancelling that await prevents a late response but cannot terminate the
  already-running Python thread or undo a completed file/process/network side
  effect. Existing permission, confinement, tool-specific timeouts, and result
  bounds remain the governing controls.
- Server shutdown stays within `mcp-unified`'s configured deadline and does
  not claim that arbitrary non-cooperative application work was force-killed.

## 13. Testing and Verification

Implementation follows strict red-green-refactor. Required coverage includes:

### 13.1 Pure adapter tests

- duplicate/missing registrations fail construction;
- nine built-in descriptors map bijectively to nine handlers;
- real local-provider schemas appear unchanged in `tools/list`;
- optional local registrations publish all-or-none, and a duplicate,
  built-in collision, invalid descriptor shape, or non-callable handler leaves
  the built-in catalog unchanged;
- no `library_*` tool appears in standalone catalog;
- tool dict/list/string projection through the real gateway;
- typed local allow/ask/deny/kill-switch/gate-error/execution-error behavior;
- exact stable-refusal-constant to reason-code mapping, with every other
  failure classified as `local_tool_failed`;
- every typed local failure uses its exact fixed public message and a sentinel
  `ToolResult.error` containing a path/secret reaches neither the wire nor
  adapter diagnostics;
- sync local handlers run off the event loop;
- local-tool registration failure emits one fixed payload-free diagnostic and
  does not echo a sentinel path, secret, exception, or traceback;
- all five URI templates route exact valid identifiers and reject malformed,
  ambiguous, unknown, or mismatched URIs;
- small and multi-chunk resource reads, exact namespaced metadata, UTF-8
  boundaries, malformed/base-mismatched/out-of-bounds cursor rejection, and
  content-change behavior without treating the cursor as authentication;
- all five prompt descriptors include correct arguments;
- prompt primitive coercion, missing/unknown/invalid arguments;
- user/assistant pass-through and exact leading-system folding;
- empty-list and invalid prompt role/order/content failures.

### 13.2 Real handler tests

- all nine implemented built-in handlers remain registered and
  `ingest_media` remains absent/refused;
- representative read and write tools execute against temporary real SQLite
  databases and never the user's configured database;
- all five resource handlers map through the adapter;
- all five prompt handlers map through the adapter;
- `search_and_synthesize` awaits real async search behavior;
- large conversation/media content returns continuation rather than exceeding
  the gateway result limit.

### 13.3 Protocol and process tests

- in-memory strict stdio initialization and core method flow for
  `2025-03-26`, `2025-11-25`, and `2026-07-28`;
- object-only `structuredContent` at `2025-11-25` and batch rejection there,
  with batch acceptance only at `2025-03-26`;
- Chatbook client subprocess launch against `python -m tldw_chatbook.MCP`;
- catalog cursor aggregation, malformed/repeated-cursor rejection, exact
  100-page/10,000-item boundaries, and fail-closed over-bound behavior;
- tool call, resource read/continuation, and prompt get;
- exact `"_meta"` preservation through both client layers;
- clean EOF and nonzero fatal return-code propagation;
- protocol stdout contains JSON only;
- cancellation emits no late duplicate output.

### 13.4 Boundary and packaging tests

- no production `mcp.server.fastmcp`, `from mcp`, or `import mcp` remains;
- both optional extras pin `mcp-unified==0.2.1`;
- availability and optional-feature gates resolve `mcp_unified`;
- build wheel and sdist;
- install each artifact's `[mcp]` extra independently in a clean environment;
- assert `mcp_unified` and `tldw_chatbook` import from that environment's
  site-packages, not the checkout;
- run each artifact smoke from a temporary working directory with a fresh
  temporary `HOME`, no checkout `PYTHONPATH`, and no inherited Chatbook
  config/data/path override or provider-credential environment variables;
- before launch, assert the resolved Chatbook databases, permission store,
  configuration, and default workspace all remain under that temporary root,
  and place the artifact virtual environment and `TMPDIR` under the same root;
- run the standalone MCP protocol smoke against both isolated artifacts;
- dependency/license metadata and documentation checks remain green.

### 13.5 Regression scope

Run, at minimum:

- the complete `Tests/MCP` suite;
- direct Library tool contract/security/service tests;
- optional-dependency and installed-distribution tests;
- documentation/architecture contract tests that inventory MCP behavior;
- Ruff format/check, type checking for changed modules, Bandit for changed
  production code, Python 3.11 syntax compilation, and `git diff --check`.

Compare any broader-suite failures with the identical command on a clean
`origin/dev` worktree and report failure sets, not counts.

## 14. Documentation and Task Hygiene

Update:

- `Docs/Design/MCP.md` for `mcp-unified`, current supported revisions,
  standalone catalog boundaries, continuation behavior, and privacy warning;
- any user-facing install/recovery copy that still names `mcp[cli]`;
- TASK-2512 with split measurable acceptance criteria, ADR/spec/plan links,
  implementation notes, exact test evidence, and final status;
- TASK-2511 to record that the obsolete FastMCP smoke was superseded by the
  completed migration and artifact-level `mcp-unified` smoke.

No new lesson is required unless implementation uncovers a reusable trap not
already captured by the repository's testing, live-verification, or backlog
hygiene lessons.

## 15. Acceptance Summary

The migration is complete only when:

1. The exact released public package is the only standalone MCP server
   dependency.
2. Every intended standalone tool/resource/prompt is discoverable and usable
   through the strict gateway.
3. No in-app-only Library tool is accidentally exposed through stdio.
4. Canonical mappings, prompt roles/arguments, typed local errors, long
   resources, and catalog pagination are bounded and tested.
5. The existing client command and `2025-03-26` behavior remain compatible.
6. Fresh wheel and sdist `[mcp]` installations pass the standalone smoke.
7. Documentation states the local-data privacy boundary honestly.
8. Tests, static analysis, security checks, documentation, ADR hygiene, and
   backlog closeout satisfy the repository Definition of Done.
