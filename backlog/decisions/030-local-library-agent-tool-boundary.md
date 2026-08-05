# ADR-030: Direct Local Library Tool Boundary for Console and MCP

Status: Accepted
Date: 2026-08-02
Related Task: [TASK-1337 - Add direct local Library tools for Console agents and MCP](../tasks/task-1337%20-%20Add-direct-local-Library-tools-for-Console-agents-and-MCP.md)
Supersedes: N/A

## Decision

Expose local Library reads through one descriptor-backed, synchronous service
contract shared by Console adapters and local MCP adapters. Provide separate
list, get, and lexical-search tools for Media, Notes, Prompts, Skills,
Conversations, and Collections. Direct operations use only local SQLite FTS,
parameterized literal substring matching, and bounded managed-Skill scans;
they never invoke RAG, embeddings, vector indexes, or semantic similarity.

List and search return an exact distinct total with a deterministic bounded
page and an opaque type-prefixed stable ID. Get requires that returned ID and
uses revision-aware continuation for large text. All serialized results are
byte-fitted below 32 KiB, and storage projections exclude binary columns and
unbounded related content before materialization.

Console owns one global `[console].direct_library_tools` preference, enabled by
default. When enabled, Console agents may automatically call the 18 direct
read tools. When disabled, direct list/count/view/search tools are absent and a
bounded Library RAG tool is the default retrieval capability. The UI must state
that Library data returned to a cloud model leaves the device and that the
setting does not disable MCP access.

Console-specific MCP composition suppresses the built-in server's 18 new
Library tool names and overlapping legacy Library read/RAG tools so MCP cannot
duplicate or bypass the selected Console mode. This suppression does not alter
the MCP server's public contracts or external MCP profiles. Local MCP clients
receive the 18 tools under MCP's existing transport and read-policy boundary.

Blocked local Skills remain discoverable through safe metadata but never
return body or supporting-file content. Media vector embeddings, Conversation
image BLOBs, binary payloads, and local filesystem paths are never selected or
returned by these tools.

## Context

Users need factual Library answers such as item counts, exact-name checks, and
keyword matches. The current Library RAG seam is designed for retrieval and
synthesis, not exact inventory totals, stable paging, or complete coverage of
Prompts, Skills, and Collections. Coupling direct factual reads to RAG also
makes answers depend on index readiness and can introduce semantic matches
where the caller requested literal local data.

Console, the built-in MCP server, and the direct local MCP delegate currently
have separate tool-registration paths. Defining the 18 contracts independently
in each path would create schema and behavior drift. At the same time, routing
Console through MCP alone would apply MCP naming and approval behavior to a
native local capability and would let the built-in MCP catalog bypass the
Console retrieval-mode setting.

Tool results enter model history. Large Media, Prompt, Note, Skill, or
Conversation content therefore needs a service-owned byte limit and explicit
continuation rather than relying on provider truncation. Existing detail paths
may also materialize vector or image BLOBs and all Skill supporting files, so
response-only trimming is not a sufficient privacy or performance boundary.

This decision extends the global-settings ownership described by ADR-003,
follows ADR-009 for blocked Skill trust behavior, follows ADR-013 for raw text
versus FTS expressions, and preserves the versioned Prompt artifact semantics
in ADR-029.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Use the existing Library RAG service for every agent query | It cannot guarantee exact inventory totals or stable type-complete paging and violates the lexical-only requirement for direct tools. |
| Expose one polymorphic list/search/get tool for every type | Large union schemas are harder for models and MCP clients to use correctly and weaken wrong-type stable-ID validation. |
| Implement 18 independent Console tools and 18 independent MCP tools | Duplicated schemas, validation, normalization, and limits would drift across runtimes. |
| Expose only the built-in MCP versions to Console | MCP names and permissions are a different boundary, and the catalog could bypass the Console direct-versus-RAG setting. |
| Return complete records from list and search | This would defeat pagination, inflate model context, and expose more local data than the user asked to retrieve. |
| Add new FTS tables for all six types | Existing FTS tables, link relations, and bounded literal scans satisfy this scope without a schema migration. |
| Bound only the serialized response | Existing detail paths can still load large supporting files and BLOBs before trimming, causing avoidable memory and privacy risk. |

## Consequences

### Benefits

- Console and MCP share one testable result and validation contract.
- Exact counts, stable IDs, keyword matching, and deterministic paging do not
  depend on an embedding index.
- Large content remains useful through continuation without overflowing model
  history.
- The Console retrieval mode cannot be silently bypassed by its own built-in
  MCP inventory.
- Skill trust and binary-data boundaries remain explicit and testable.

### Accepted trade-offs

- Existing local services need additive read/query methods for exact totals,
  literal keyword matching, byte-safe projections, and child pagination.
- Console exposes 18 discoverable tools, so agents normally use the existing
  find/load catalog flow before calling one.
- Direct Library reads are automatic when the global Console preference is
  enabled; there is no additional per-call approval prompt.
- Disabling direct tools reduces Console retrieval to the RAG types currently
  supported: Notes, Media, and Conversations. Missing dependencies or indexes
  fail visibly and do not re-enable direct tools.
- Exact totals describe each response transaction, not a frozen snapshot
  across later pages while concurrent edits occur.
- The standalone MCP bootstrap must construct all six current local services
  from canonical configured paths and retain legacy tool compatibility.

## Links

- [TASK-1337](../tasks/task-1337%20-%20Add-direct-local-Library-tools-for-Console-agents-and-MCP.md)
- [Design specification](../../Docs/superpowers/specs/2026-08-02-local-library-agent-tools-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-08-02-local-library-agent-tools.md)
- [ADR-003: Settings Library/RAG Defaults Boundary](003-settings-library-rag-defaults.md)
- [ADR-009: Local Skill Trust Boundary](009-local-skill-trust-boundary.md)
- [ADR-013: Separate media plain-text search from FTS MATCH expressions](013-media-search-plain-text-fts-boundary.md)
- [ADR-029: Versioned Prompt Artifacts and Safe Improvement Transactions](029-versioned-prompt-artifacts-and-safe-improvement-transactions.md)
