# Local Library Agent and MCP Tools Design

**Date:** 2026-08-02

**Status:** Approved design

**Scope:** Read-only Console-agent and local MCP access to Media, Notes,
Prompts, Skills, Conversations, and Collections stored in the local Library

## Goal

Let Console agents and local MCP clients answer factual questions about the
user's Library without routing direct list, view, count, or lexical-search
requests through the RAG pipeline.

Examples include:

- "How many notes do I have?"
- "Do I have a note named Y?"
- "Do I have any media relating to X?"
- "Find prompts tagged with brainstorming."
- "Open the conversation returned by that search."

Success means:

- Each Library item type has a dedicated list, get, and lexical-search tool.
- List and search responses contain a bounded page and an exact total count.
- Every result contains an opaque stable item ID, and get tools require that
  ID rather than accepting a title or name.
- Search matches names or titles, content or descriptions, and supported tags
  or keywords without embeddings or semantic similarity.
- Large content is returned in bounded, revision-aware chunks with explicit
  continuation metadata.
- Console and MCP expose the same validated operation contracts and normalized
  result shapes while retaining their existing registration and policy seams.
- Users are visibly warned that Library data returned to a cloud model leaves
  the device.

## Non-goals

- Replacing or redesigning the existing RAG pipeline.
- Semantic, embedding, vector, or similarity search inside the 18 direct
  Library tools.
- A single polymorphic tool spanning all item types.
- Separate count-only tools; the exact count is part of list and search.
- Creating, updating, deleting, importing, exporting, or executing Library
  items.
- Searching inside binary media, returning binary payloads, or exposing local
  filesystem paths.
- Recursively searching the full content of Collection members.
- Adding new FTS tables or a schema migration solely for this feature.
- Changing the existing legacy MCP tool names or behavior.
- Expanding RAG fallback coverage beyond its current Notes, Media, and
  Conversations scope as part of this work.

## Existing Boundaries and Constraints

The implementation follows these observed repository seams:

- `Agents/tool_catalog.py` owns Console tool discovery and activation. A
  provider may contribute more tools than the active-tool limit because agents
  can discover and load them through the existing catalog tools.
- `Chat/console_agent_bridge.py` composes the per-run tool registry. The direct
  Library provider belongs after built-ins and before Skills and MCP providers.
- The Console MCP bridge renames built-in MCP tools to model-facing names such
  as `mcp__tldw_chatbook__<tool>`. Name-collision filtering alone therefore
  cannot prevent the new Library MCP tools from reappearing in Console when
  the direct provider is absent. Console composition needs a source-aware
  exclusion for the built-in server's raw Library tool names, including
  overlapping legacy read/RAG tools.
- `UI/Screens/chat_screen.py` is the actual Console bridge construction and
  settings-injection seam. The app instance already owns the local services
  needed to construct the Library service.
- Ordinary Console subagents inherit the parent's allowed tools, excluding the
  existing special cases. Skill-triggered child agents retain their current
  deliberately narrowed tool behavior.
- Tool results enter model history without a second general-purpose truncation
  layer. The Library service must therefore enforce output bounds itself.
- `MCP/server.py` declares FastMCP tools and derives the local capability
  manifest. `MCP/local_runtime_delegate.py` separately dispatches in-process
  tool calls, while `MCP/local_control_service.py` maps tools to policy action
  IDs. All three seams must stay in sync.
- Existing MCP tools such as `search_notes` and `search_conversations` return
  older result shapes. They remain unchanged; the new tools use a `library_`
  namespace.
- The standalone MCP bootstrap currently has stale service construction and
  media-path assumptions. Supporting the new tools requires correcting those
  scoped bootstrap defects and using the configured local database-path
  helpers.
- The existing `LibraryLocalRagSearchService` remains the RAG boundary. It is
  not reused as the implementation of direct lexical tools because it does not
  provide the required type coverage, exact totals, stable pagination, or
  non-RAG contract.

This design follows the plain-text/FTS boundary in
`backlog/decisions/013-media-search-plain-text-fts-boundary.md` and the local
skill trust boundary in
`backlog/decisions/009-local-skill-trust-boundary.md`. A new ADR is required
for the cross-module service contract, Console/MCP runtime boundary, and
privacy-sensitive mode setting.

## 1. Tool Surface

Each type receives exactly three namespaced tools:

| Type | List | Get | Search |
| --- | --- | --- | --- |
| Media | `library_list_media` | `library_get_media` | `library_search_media` |
| Notes | `library_list_notes` | `library_get_note` | `library_search_notes` |
| Prompts | `library_list_prompts` | `library_get_prompt` | `library_search_prompts` |
| Skills | `library_list_skills` | `library_get_skill` | `library_search_skills` |
| Conversations | `library_list_conversations` | `library_get_conversation` | `library_search_conversations` |
| Collections | `library_list_collections` | `library_get_collection` | `library_search_collections` |

The singular/plural naming in this table is the canonical public API. Existing
unnamespaced MCP tools are not aliases and are not removed.

List tools accept `limit` and `offset`. Search tools accept a required `query`
plus `limit` and `offset`. Get tools accept only the expected stable `id` for
identity and optional bounded-content or child-pagination parameters. A get
tool never accepts a title, display name, or raw database row number in place
of the returned opaque ID.

Console registers the tools with a `LibraryToolProvider`. Because 18 tools
exceed the active-tool cap, their names, descriptions, and schemas remain
discoverable through the existing `find_tools`/`load_tools` flow rather than
all being forced into every model request. MCP clients see all 18 normal MCP
tools.

The direct provider is registered after built-ins and before Skills and MCP.
Its names participate in Skill and MCP collision filtering, while the
`builtin_names` set used to constrain skill-triggered child agents remains the
original safe built-in set. This keeps the existing narrowed child-agent trust
boundary intact.

## 2. Shared Service and Descriptor Source

Add one synchronous `LocalLibraryToolService` core that owns the public
operation contract and delegates storage work to the existing local services:

- `LocalMediaReadingService`
- `NotesInteropService` / the corrected local Notes scope seam
- `LocalPromptService` / local Prompt scope seam
- `LocalSkillsService`
- `ChatConversationService`
- `LocalLibraryCollectionsService`

The service depends only on local backends. It never selects a server scope,
calls an embedding model, or invokes the RAG service for any of its 18 direct
operations.

The synchronous core matches the Console `ToolProvider.invoke` contract and
runs inside the Console agent's existing worker thread. Small adapters may use
`asyncio.run` to call local service methods that are declared async but perform
local work. FastMCP and direct MCP runtime handlers run the synchronous core
with `asyncio.to_thread`, so SQLite and filesystem reads never block the MCP or
Textual event loop. The core is not called synchronously from an event-loop
thread.

One descriptor table is the source of truth for tool names, descriptions,
input JSON schemas, and operation routing. The Console provider, FastMCP
registration, capability description, and local runtime delegation derive
from those descriptors. This prevents the Console and MCP contracts from
drifting while allowing each runtime to retain its own registration adapter.

`MCP/local_runtime_delegate.py` receives a generic, allow-listed
`library_*` dispatch path backed by the descriptor table, rather than 18
unrelated hand-written dispatch methods. Only names present in the table may
dispatch. `MCP/local_control_service.py` maps every new tool to a read-only
Library policy action, including a distinct Collections read action rather
than reusing a semantically unrelated reading-list action.

The local capability manifest retains AST-derived legacy entries and appends
descriptor-derived Library entries with their input schemas. Hub inventory
normalization preserves those schemas. Runtime diagnostics treat an
allow-listed descriptor-dispatched Library tool as implemented even though it
does not have a dedicated `_tool_<name>` method.

The standalone MCP constructor uses the configured path helpers, including
`get_media_db_path`, and constructs the current local service classes with
their real signatures. These corrections are limited to what is required to
make direct in-process and standalone stdio execution behave consistently.

## 3. Stable IDs

All brief and search results include an opaque, type-prefixed `id`. The
prefix lets a get tool reject an ID for the wrong Library type before querying
storage.

Preferred backing identities are:

- Media, Notes, Prompts, and Conversations: their existing UUIDs.
- Collections: the existing stable `collection_id`.
- Skills: the existing local skill record identity. A folder/name rename is a
  new skill identity under the current storage model.

The public ID must not encode page position, search rank, mutable title text,
or an unrestricted filesystem path. Raw backing IDs stay behind the codec.
Illustrative forms are `media:<opaque>`, `note:<opaque>`, and
`collection:<opaque>`; callers must treat the entire value as opaque.
Encoded public IDs use an ASCII-safe alphabet and are bounded to 128 bytes.

Every get operation validates and decodes the expected prefix. Missing items
return `not_found`; wrong-type or malformed IDs return `invalid_argument`.

## 4. Pagination, Exact Totals, and Ordering

List and search return this common envelope:

```json
{
  "items": [],
  "total": 0,
  "limit": 20,
  "offset": 0,
  "has_more": false,
  "next_offset": null,
  "response_truncated": false,
  "omitted_fields": []
}
```

Rules:

- Default `limit` is 20; maximum `limit` is 50.
- `offset` is zero-based and non-negative.
- `total` is the exact number of distinct matching active items at the time of
  that response, before pagination.
- SQL-backed stores calculate the distinct count and page within the same
  SQLite read transaction.
- Skills are enumerated once into a consistent in-memory result set; its length
  is the exact total for that response before slicing.
- Exactness applies to each response, not to a frozen snapshot across multiple
  calls. Concurrent Library edits can legitimately change later totals.
- Joins against tags, keywords, or Collection members deduplicate by stable
  item identity before counting or paging.
- Ordering is deterministic and always ends with the stable ID as a tie-break.
  Lists use the type's existing user-facing order, normally most recently
  updated first. Search prefers exact title/name matches, then backend lexical
  rank where available, then recency and stable ID.
- SQL-backed offset pagination uses `LIMIT`/`OFFSET` on the final deduplicated
  query. It must not fetch and materialize all rows before the requested
  offset, even where an existing adapter currently emulates offsets that way.
- `has_more` and `next_offset` are derived from `offset`, returned item count,
  and `total`; an empty terminal page returns `next_offset: null`.

Inactive, soft-deleted, and trashed items are excluded unless a type's current
Library definition of active records explicitly differs. The direct tools do
not introduce an include-deleted switch.

## 5. Lexical and Keyword Search

Search is literal, case-insensitive lexical search. A result matches when the
query is found in any supported field:

- Title or name.
- Primary content or description.
- Tags or keywords.

The query is not an FTS expression supplied by the caller. The implementation
builds safe FTS token queries internally, parameterizes SQL values, and escapes
`LIKE` wildcard characters for substring branches. It may combine:

- Exact case-insensitive title/name matches.
- Case-insensitive substrings.
- Safely generated FTS token matches.
- Keyword/tag link-table or normalized metadata matches.

These branches are ORed, deduplicated, and counted exactly. No direct tool
consults embeddings, vector indexes, similarity scores, or semantic reranking.
An empty or whitespace-only query returns `invalid_argument`; callers use the
corresponding list tool to retrieve all items.

Search briefs include bounded evidence:

- `matched_fields`, using stable values such as `title`, `content`,
  `description`, `keywords`, or `member_title`.
- `matched_keywords`, bounded by the keyword response limits.
- A maximum 240-character textual preview where the trust policy permits it.

Search implementation by type:

- **Media:** extend the current FTS title/content path with keyword-name
  matching and `COUNT(DISTINCT ...)`; return the Media UUID in briefs.
- **Notes:** fix local offset forwarding, add an exact-count search path, and
  combine title/content FTS or substrings with keyword matching.
- **Prompts:** preserve the underlying database's exact totals and include the
  keywords field in lexical search instead of discarding those capabilities in
  the adapter. Searchable text is name, details, compiled system prompt,
  compiled user prompt, structured prompt definition, and keywords.
- **Skills:** perform a bounded case-insensitive scan over managed local skill
  name, description, `SKILL.md` body, and metadata keywords. No SQLite index is
  added. Trust restrictions still govern returned evidence and content.
- **Conversations:** combine title matching, message FTS, and conversation
  keyword matching, deduplicated at conversation identity. RAG-context
  messages are never included.
- **Collections:** search Collection name and description plus direct member
  titles. It does not recursively search member content and does not invent a
  tag system where none exists.

## 6. Brief Results and Keyword Bounds

List and search items are summaries, not complete records. Every brief contains
at least:

- Stable `id` and `type`.
- Title or name.
- A bounded summary or preview where available and permitted.
- Relevant timestamps or status metadata already available from the store.
- Supported keywords/tags, bounded to 20 values.
- `keyword_total` and `keywords_truncated` when more keywords exist.

Brief titles/names are display-normalized by replacing control characters with
spaces and are bounded at a UTF-8 boundary to 160 bytes, including an ellipsis
when truncated. They include `title_truncated` or `name_truncated` when the
display value is shortened. Previews are bounded to 240 displayed characters,
and individual returned keyword values are bounded to 120 displayed
characters. The underlying exact match and count use the complete stored
values; these are response bounds only.

The exact fields may differ by Library type, but Console and MCP receive the
same normalized shape for a given operation. Missing source metadata is
represented explicitly or omitted according to the descriptor schema; it is
not fabricated.

Blocked Skills remain discoverable by name and safe metadata as required by
the skill trust model. Their body, supporting-file excerpts, and content-based
preview are withheld. A match may report that restricted content matched, but
must not reproduce that content.

## 7. Bounded Reads, Get, and Continuation Contract

Output bounds apply at the storage projection as well as at serialization.
The implementation must not load a BLOB or an unbounded related-content set
and merely discard it afterward:

- Media queries explicitly exclude `vector_embedding` and return only the
  textual fields needed by the contract.
- Conversation message queries explicitly exclude `image_data`; they may
  return bounded MIME/attachment-presence metadata but never image bytes.
- Skill detail gains bounded, trust-aware read methods that reuse the existing
  validated-path boundary. The current eager detail method, which reads every
  supporting file, is not used by these tools.
- Prompt reads project only the selected textual fields or section instead of
  materializing unrelated version-history data.

Every operation has a 32 KiB UTF-8 serialized-result hard ceiling. Page
serialization reserves at most 24 KiB for the envelope and mandatory item
fields: ASCII stable ID, fixed type, byte-bounded display title/name,
title/name truncation flag, exact keyword count, and keyword-truncation flag.
The serializer measures the actual encoded JSON bytes. If mandatory fields
would exceed their reserved budget, it deterministically shortens display
titles/names further, never below a 32-byte display floor, and sets the
corresponding truncation flag.

The ASCII ID and fixed-field bounds guarantee that a 50-item mandatory page
fits below the reserve. If optional data would cross the full ceiling, the
serializer trims in this fixed order: additional keyword values, previews,
then optional metadata.

It preserves every item in the requested page and always preserves each item's
`id`, `type`, bounded title/name, and keyword counts. The envelope then returns
`response_truncated: true` plus `omitted_fields`; otherwise it returns
`response_truncated: false`. It never relies on the model runtime to truncate
an oversized result.

If response-level trimming removes keyword values, the affected item's
`keywords_truncated` remains true and `keyword_total` remains the exact stored
count. `omitted_fields` reports stable field paths rather than including any
omitted values.

Get responses return metadata plus one bounded content segment or child page.
The default text budget is 8,000 characters and the maximum caller-selectable
budget is 16,000 characters. These are requested character ceilings, not a
promise that every response contains that many characters. After bounded
metadata is assembled, the serializer measures the actual UTF-8 JSON bytes and
shortens returned text at a Unicode character boundary until the complete
response fits below 32 KiB. It uses the largest prefix that fits. The returned
`end`, `has_more`, and continuation cursor are computed from the actual
character prefix, never the requested ceiling.

Content metadata includes `requested_max_chars` and `returned_chars`. `start`,
`end`, and `total_chars` are canonical text character offsets; they are not
byte offsets. JSON escaping and multibyte characters therefore cannot make a
cursor skip or repeat text.

Text continuation metadata includes:

```json
{
  "content": {
    "text": "...",
    "start": 0,
    "end": 8000,
    "total_chars": 24000,
    "requested_max_chars": 8000,
    "returned_chars": 8000,
    "revision": "opaque-revision",
    "has_more": true,
    "next_cursor": "opaque-cursor"
  }
}
```

Cursors are opaque, bound to the stable item ID, content section, offset, and
revision. Continuation validates that the current item revision still matches.
If content changed, the tool returns `content_changed` and a fresh-start hint
instead of splicing different revisions together.

The revision is the stored item/message version when that version changes with
the returned content. Where a store cannot guarantee that property, it is a
hash of the exact canonical text section being paged. Conversation
within-message cursors bind to the message version or message-content hash;
Skill cursors bind to the selected file-content hash.

Type-specific get behavior:

- **Media:** returns textual metadata/content only. It never returns a binary
  payload or local path.
- **Notes:** returns its content in chunks using the common cursor contract.
- **Prompts:** returns metadata plus a bounded section manifest for `details`,
  `system_prompt`, `user_prompt`, and `prompt_definition`. The optional
  `section` argument selects one manifest section for chunked reading. When it
  is omitted, the response includes a bounded overview of details and compiled
  System/User text plus the manifest; canonical structured definitions remain
  directly retrievable through `section="prompt_definition"`.
- **Skills:** returns safe metadata and the main `SKILL.md` content when the
  skill is trusted. Supporting files are first returned as a bounded manifest.
  A later call selects a manifest file through its opaque file token and reads
  that file in chunks; callers cannot submit arbitrary paths. Blocked skills
  never return body or supporting-file content.
- **Conversations:** returns conversation metadata and a text-only paginated
  message page with an exact `message_total`. The structured continuation
  state identifies message offset, stable message ID, and within-message
  character offset so a single long message can be continued without losing
  message boundaries. `message_limit` is a maximum: a response may stop before
  that many messages, or shorten the final included message, to honor the byte
  ceiling. It reports `returned_message_count`, the next message offset, and
  any within-message cursor from the actual returned content.
  `include_rag_context` is always false.
- **Collections:** returns Collection metadata and a paginated direct-membership
  page with an exact `member_total`. Each member includes its stable
  `membership_id`, stored source type, title, and a type-prefixed `item_id`
  accepted by the corresponding get tool when the source type is supported.
  Unsupported source types return `item_id: null` and an opaque source
  reference. Collection get does not inline member content.

## 8. Console Mode Setting and RAG Fallback

Console adds one global application preference under
`[console].direct_library_tools`, enabled by default. It is presented in
Settings > Library/RAG under a Console agent retrieval subsection, not stored
per conversation:

> **Use direct Library tools**
>
> On: Console agents may automatically list, count, read, and lexically search
> your local Library.
>
> Off: Direct list, count, view, and lexical search tools are unavailable.
> Console agents use Library RAG as the default retrieval method. RAG currently
> covers Notes, Media, and Conversations and requires an available, populated
> index.
>
> **Privacy:** Retrieved titles, metadata, content, and RAG excerpts are
> included in model requests. If you use a cloud model, this Library data
> leaves your device and is handled by that provider. Use a local model if the
> data must remain on-device.
>
> **Scope:** This setting affects Console agents only. MCP Library access is
> controlled separately.

The privacy warning is visible below the toggle, not hidden in a tooltip.
Changing the setting applies to the next Console agent run.

When the setting is on, the Console registry includes the 18 direct Library
tools. When it is off, those tools are omitted and a bounded
`search_library_rag` agent tool is available as the default Library retrieval
method. The agent invokes RAG only when retrieval is relevant; the app does not
run RAG for every message. If dependencies or an index are unavailable, the
RAG tool returns a clear `index_unavailable` or setup response and never
silently re-enables direct access.

This setting controls Console agent exposure only. The 18 local MCP tools
remain available under MCP's own enablement, transport, and read-policy
controls. The toggle is not described as a privacy kill switch: both direct
tool results and RAG excerpts may be transmitted to the selected model.

Enabling the setting is the Console consent boundary for automatic read-only
Library tool execution; this design does not add a second per-call approval
prompt. To prevent duplication or bypass in either setting state, the
Console-specific `MCPToolProvider` composition always excludes tools whose
source identity is `builtin:tldw_chatbook` and whose raw name is in a
Console-only exclusion set. That set is the 18 descriptor-defined Library
names plus the overlapping legacy tools `search_rag`, `search_notes`,
`search_conversations`, `get_conversation_history`, and
`export_conversation`. Suppression affects Console exposure only; these legacy
tools retain their existing MCP contracts for MCP clients. The filter does not
exclude similarly named tools from external MCP profiles, which remain
governed by existing MCP permissions.

## 9. Errors, Validation, and Security

Expected failures use a normalized structured error with a stable code,
human-readable message, retryability, and bounded details. No response includes
a Python stack trace, SQL text, secret, unrestricted path, or raw exception
representation.

Required codes are:

- `invalid_argument`
- `not_found`
- `content_changed`
- `index_unavailable`
- `feature_unavailable`
- `storage_error`

The common error payload is:

```json
{
  "error": {
    "code": "not_found",
    "message": "The requested Library item was not found.",
    "retryable": false,
    "details": {}
  }
}
```

The service produces one JSON-safe error object. MCP returns that object
directly. Console preserves the existing `ToolResult` model: successful
payloads are JSON-serialized into `content`, while failures JSON-serialize the
same error object into the string `error` field with `ToolResult.ok = false`.
After JSON decoding, Console and MCP expose the same Library payload/error
shape; the shared agent runtime does not need a new error type.

JSON schemas enforce type and numeric bounds, and runtime validation repeats
security-critical checks because not every caller is schema-conforming. SQL
values are parameterized. Opaque ID and continuation codecs fail closed.

All 18 tools are read-only. Console calls obey the global setting, catalog
allow-list, and agent-run tool controls described above; there is no per-call
Library approval prompt. MCP calls obey their existing read policies. Returned
Library text is untrusted data, not instructions. Tool descriptions and
model-facing wrappers must make that boundary explicit. The service returns
structured fields rather than concatenating Library text into instruction-like
prose.

## 10. Data Flow

Direct Console execution is:

1. Console constructs its per-run bridge with the persisted direct-tools
   setting.
2. Console composes MCP with the built-in Library names source-filtered out.
3. `LibraryToolProvider` contributes descriptor-backed tools when enabled, or
   the bounded RAG provider contributes `search_library_rag` when disabled.
4. The model discovers and loads the required tool.
5. The provider validates arguments and calls `LocalLibraryToolService` on the
   existing agent worker thread.
6. The service queries the existing local backend, normalizes stable IDs and
   fields, enforces bounds, and returns structured data.
7. The existing Console tool-result path records the bounded result for the
   model and UI.

Local MCP execution is:

1. FastMCP exposes descriptor-backed `library_*` schemas.
2. The control service authorizes the corresponding read action.
3. Direct or standalone delegation resolves the allow-listed operation and
   offloads the synchronous service call from the event loop.
4. The same `LocalLibraryToolService` returns the same normalized payload used
   by Console.

The RAG-off mode is a separate flow. It calls the existing Library RAG service
through a bounded agent adapter and never passes through the direct lexical
service.

## 11. Testing and Verification

Tests use temporary or in-memory local SQLite databases and managed temporary
skill directories. They must cover:

- Descriptor completeness and uniqueness for all 18 canonical tool names.
- Console catalog discovery/loading and per-run provider composition.
- Source-aware suppression of the 18 built-in Library MCP duplicates and the
  five overlapping legacy built-in read/RAG tools in both Console toggle
  states, without suppressing external MCP profiles or changing MCP-client
  contracts.
- MCP registration, capability/schema exposure, generic delegation, and
  read-policy mapping.
- Standalone MCP bootstrap using configured current database paths.
- Exact list and search totals before pagination for every item type.
- Deterministic ordering and correct `has_more`/`next_offset` behavior,
  including empty and out-of-range pages.
- Stable ID presence, round-trip get, malformed ID rejection, wrong-type ID
  rejection, and not-found behavior.
- Case-insensitive title/name, content/description, keyword/tag, and Collection
  member-title matches.
- Literal handling of FTS operators, quotes, percent, and underscore; callers
  cannot inject FTS syntax or broaden `LIKE` matches.
- Deduplication when one item matches multiple fields, keywords, messages, or
  members.
- Keyword value/count bounds and search evidence bounds.
- Maximum-page serialization with multibyte Unicode and JSON-escaped input,
  proving byte-bounded mandatory fields fit their reserve, required item IDs
  are preserved, display titles shorten deterministically when necessary, and
  `response_truncated` identifies omitted optional fields below the hard
  ceiling.
- Text chunk sizes, serialized hard ceiling, cursor continuation, terminal
  chunks, tampered cursors, and revision-change rejection. Multibyte and
  JSON-escaped content tests prove the final serialized get response stays
  below 32 KiB, returned character offsets reflect byte-driven shortening, and
  continuation neither skips nor repeats text.
- Conversation message totals, message pagination, long-message continuation,
  exclusion of RAG-context messages, and proof that message image BLOBs are not
  selected or returned.
- Collection membership totals/pagination without member-content expansion.
- Trusted-skill content access, blocked-skill discoverability, and blocked
  content/supporting-file withholding.
- Media content responses containing no binary data or local filesystem path.
- Media detail queries that do not select vector-embedding BLOBs, Skill reads
  that do not eagerly load supporting-file content, and large-offset SQL pages
  that do not materialize all preceding rows.
- Prompt section manifests, bounded overview/default behavior, direct section
  selection, and structured-definition continuation.
- Collection member `item_id` round trips for supported Library source types.
- Console/MCP input-schema and normalized-result parity.
- Toggle default, persistence, next-run application, direct-tool omission, RAG
  replacement, unavailable-index behavior, global-not-session scope, automatic
  read consent, and exact visible privacy/scope copy.
- MCP runtime diagnostics recognizing descriptor-dispatched tools as
  implemented and built-in inventory preserving descriptor input schemas.
- Compatibility tests proving existing unnamespaced MCP tools retain their
  current public behavior.

Verification should run focused service, Console-agent, MCP, settings, and DB
tests first, followed by the broadest suite supported by the local optional
dependency environment. Any pre-existing import abort or unavailable optional
dependency must be reported separately rather than represented as a feature
failure.

## 12. Rollout and Compatibility

No Library data migration is planned. Existing UUIDs, primary identities, FTS
tables, keyword relations, message indexes, and Collection membership rows are
used through additive service queries and adapters.

The feature is additive:

- Existing MCP names and result contracts stay intact.
- Existing visible manual Library RAG behavior stays intact.
- The Console setting adds agent retrieval mode selection without changing the
  selected model or provider.
- If one optional backend capability is unavailable, only the affected tool
  returns `feature_unavailable`; other Library types remain usable.

Implementation should land behind the setting and contract tests, with
documentation listing the 18 tools, their exact-count semantics, the lexical
search boundary, content limits, and cloud-model privacy consequence.

### Implementation deviations (recorded 2026-08-07)

- **FastMCP-free local MCP surface.** The design assumed the descriptor-backed
  tools would also register with FastMCP (Existing Boundaries, §2, §10). During
  implementation the owner directed that FastMCP is deprecated in this
  repository and must not be used: the `mcp` dependency is absent from the
  development environment, and `mcp[cli]>=1.0.0` resolves to a release line
  that removes `mcp.server.fastmcp`. The shipped surface is therefore entirely
  FastMCP-free: `MCP/server.py` appends the 18 descriptor-derived entries to
  the local capability manifest (`_describe_local_library_tools`),
  `MCP/local_runtime_delegate.py` dispatches them to the shared synchronous
  service via `asyncio.to_thread`, `MCP/local_control_service.py` maps them to
  policy action IDs, and `runtime_policy/registry.py` carries the
  `library_collections` capability. The legacy `TldwMCPServer`/`__main__.py`
  module is untouched; retiring it is separate scope. Every payload and schema
  parity guarantee is unchanged — both runtimes still derive from the single
  descriptor table, and the cross-runtime parity tests in
  `Tests/Library/test_cross_runtime_parity.py` pin that.

## 13. Alternatives Considered

### Reuse the existing RAG search service for all queries

Rejected because it couples count/list/view questions to indexing state,
cannot provide the required exact totals and stable paging, lacks full type
coverage, and violates the explicit no-semantic-search contract for direct
tools.

### Expose one polymorphic `search_library` and one `get_library_item`

Rejected because type-specific schemas are clearer to models and MCP clients,
make stable-ID validation simpler, avoid a large union result, and let each
backend express the metadata and child pagination it actually supports.

### Create new FTS indexes for every type

Rejected for this scope. Existing FTS/search seams plus parameterized
substring and relation-table matching can satisfy the functional contract
without schema churn. Skills remain a bounded filesystem-managed scan.

### Return complete records from list and search

Rejected because tool results enter model history and large Media,
Conversation, Prompt, Note, or Skill content would make pagination ineffective
and create unnecessary privacy and context-window exposure.

## 14. Architecture Decision Record

ADR required: **yes**

ADR path: `backlog/decisions/030-local-library-agent-tool-boundary.md`

Reason: the feature introduces a durable cross-module read contract shared by
Console and MCP, a runtime selection boundary between direct lexical tools and
RAG, stable-ID/continuation semantics, and explicit privacy behavior.

The ADR must be created and linked from the Backlog task and implementation
plan before implementation begins. It should reference ADR-009 and ADR-013
rather than duplicating their skill-trust and Media FTS decisions.
