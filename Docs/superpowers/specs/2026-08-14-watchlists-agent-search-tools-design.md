# Watchlists Agent Search Tools Design

Date: 2026-08-14
Status: User-approved and independently reviewed; ready for implementation planning
Task: [TASK-16222](../../../backlog/tasks/task-16222%20-%20Expose-local-Watchlists-search-tools-to-Console-and-MCP.md)
Related decisions:

- [ADR-030 — Local Library Agent Tool Boundary](../../../backlog/decisions/030-local-library-agent-tool-boundary.md)
- [ADR-032 — Local Agent Tool Permission Boundary](../../../backlog/decisions/032-local-agent-tool-permission-boundary.md)

ADR required: yes

ADR path: `backlog/decisions/032-local-agent-tool-permission-boundary.md`
(addendum)

Reason: ADR-030 already establishes local domain-tool naming and bounded-result
precedents, while ADR-032 owns Console registration, the synthetic local MCP
permission boundary, external MCP exposure, and approval semantics. However,
ADR-032 currently describes workspace, web, Git, and session-task tools under a
“Local workspace + web tools” principal. Giving that principal access to
private feed/article data is a material privacy and provider-boundary
expansion. An ADR-032 addendum must record the accepted shared-principal
trade-off, Watchlists naming/data exposure, external MCP gating, and required
permission/help-copy correction before implementation.

## Summary

Expose local Watchlists evidence to the Console agent and approved external MCP
clients through two read-only tools:

- `watchlists_search_items` searches the existing local Watchlists corpus and
  returns newest-first, source-linked, collection-aware evidence with stable
  continuation.
- `watchlists_get_item` retrieves bounded detail for one canonical local item
  returned by search.

The tools reuse the existing title/content/author FTS5 index, literal-term LIKE
fallback, local Watchlists services, runtime-source state, local-tool provider,
permission store, kill switch, approval flow, definition hashes, and MCP
exposure gate. The agent interprets the returned evidence. The tools do not
perform semantic similarity, LLM summarization, threat scoring, or server-side
Watchlists search.

## Problem

Watchlists already collects and searches monitored feed items, but the primary
agent surface cannot query that evidence. A user must leave Console, navigate to
Watchlists, manually scope and inspect results, then restate the relevant facts
to the agent. External MCP clients have the same gap.

That prevents natural questions such as:

- What are the newest Watchlists items involving a topic?
- When did a named source last check successfully, and what did it publish most
  recently?
- Find all recent feed items relating to a topic.
- Within a named Watchlist collection, what evidence should inform a
  threat-hunting assessment?

The data and most query predicates already exist. The missing capability is a
small, permission-gated retrieval contract that gives the model bounded,
explicitly untrusted evidence.

## Goals

- Search local Watchlists items by literal full-text terms over title, content,
  and author.
- Support newest-first browsing without requiring a query.
- Scope by human-readable or numeric source and Watchlist collection identity.
- Filter by item statuses and a date floor.
- Return source links, collection memberships, match-centered excerpts, and
  date fields with their actual meanings.
- Continue through bounded pages without offset drift when newer items arrive.
- Retrieve bounded full detail for one selected item.
- Share one implementation across Console and external MCP registration.
- Preserve every existing local-tool permission and exposure control.
- Fail honestly in server Watchlists mode.

## Non-goals

- Semantic or embedding similarity, hybrid retrieval, synonym expansion, or a
  new vector index.
- LLM calls, summaries, threat-severity scores, or fixed threat-hunting rules
  inside a tool.
- Server Watchlists item search or changes to the server API.
- Mutating items, statuses, flags, sources, collections, or runtime mode.
- A new settings screen, permission store, tool provider, or MCP transport.
- Transactional snapshot isolation across a multi-page search.
- Claiming that marking content untrusted guarantees a model will ignore prompt
  injection. The tool can label and delimit evidence; it cannot control model
  behavior.

## User-facing semantics

“Similar to X” means literal full-text topic matching in this version. Input is
split into whitespace-delimited terms. Every term is quoted as an FTS5 literal
and ANDed, using the same semantics as the current Watchlists Read search.
Operator-looking input such as `OR`, `NEAR`, column filters, quotes, `%`, and
`_` cannot change the query structure. If FTS5 is unavailable, the existing
AND-of-terms, OR-across-title/content/author LIKE fallback applies.

Search results are ordered by `effective_date DESC, item_id ASC`, where
`effective_date` is Watchlists' existing normalized publication date falling
back to item creation time. Search is recency-first, not BM25-ranked. This
directly answers “most recent involving X”; it does not claim semantic nearest
neighbors.

All item statuses are included by default. A user who wants unread, reviewed,
ingested, ignored, or failed items can request one or more of those statuses.

## Tool contracts

### `watchlists_search_items`

Read-only search and browse tool.

Parameters:

| Name | Type | Required | Contract |
|---|---|---:|---|
| `query` | string | no | Blank or absent means newest items. Maximum 512 characters and 32 whitespace-delimited terms. |
| `collection` | string or integer | no | Watchlist collection/folder name (maximum 256 characters), canonical `local:watchlist:<id>`, or local row ID. |
| `source` | string or integer | no | Source name/configured URL (maximum 2,048 characters), canonical `local:subscription:<id>`, or local row ID. |
| `statuses` | array of strings | no | Any of `new`, `reviewed`, `ingested`, `ignored`, `error`; absent means all. |
| `since` | string | no | Inclusive effective-date floor as `YYYY-MM-DD` or RFC 3339; normalized to UTC. |
| `limit` | integer | no | Requested results, default 10, minimum 1, maximum 50. |
| `cursor` | string | no | Continuation returned by a prior call using the same normalized filters. |

`collection` and `source` compose as an intersection. A source that does not
belong to the selected collection produces an empty result, not a widened
search.

The response is valid JSON and has this logical shape:

```json
{
  "status": "ok",
  "query_mode": "literal_full_text",
  "ordering": "effective_date_desc_item_id_asc",
  "as_of": "2026-08-14T21:30:00Z",
  "returned_count": 2,
  "has_more": true,
  "next_cursor": "...",
  "scope": {
    "collection": {"id": "local:watchlist:7", "name": "Threat Intel"},
    "source": null
  },
  "items": [
    {
      "id": "local:watchlist_item:41",
      "title": "Example advisory",
      "url": "https://example.invalid/advisory",
      "url_redacted": false,
      "author": "Example CERT",
      "status": "new",
      "effective_date": "2026-08-14T20:00:00Z",
      "published_date": "2026-08-14T20:00:00Z",
      "created_at": "2026-08-14T20:05:00Z",
      "updated_at": "2026-08-14T20:05:00Z",
      "source": {
        "id": "local:subscription:3",
        "name": "Example CERT feed",
        "type": "rss",
        "url": "https://example.invalid/feed.xml",
        "url_redacted": false,
        "is_active": true,
        "is_paused": false
      },
      "collections": [{"id": "local:watchlist:7", "name": "Threat Intel"}],
      "evidence": {
        "content_is_untrusted": true,
        "snippet": "...matched topic in context...",
        "snippet_truncated": true
      }
    }
  ]
}
```

`query_mode` is `literal_full_text` when `query` is non-blank and `browse`
when the tool is only applying scope/date/status filters.

When one source is selected, `scope.source` also reports that source's distinct
`created_at`, `updated_at`, `last_checked`, and `last_successful_check` values.
Source objects also report `is_active` and `is_paused`, so an agent does not
misread an intentionally dormant source's stale dates as evidence of silence.
The response never invents a single “last updated” field. Item
`effective_date`, `published_date`, `created_at`, and `updated_at` remain
separate because they answer different questions.

The tool description tells callers that a request for “all” matching items
requires following `next_cursor` until `has_more` is false, subject to the
agent run's normal tool-call budget. One call never silently raises the page
ceiling or returns an unbounded corpus.

The snippet is centered on a matched title, author, or body passage when a
query is present. A blank-query browse returns a bounded leading-body preview.
HTML/control text is rendered as JSON data, not executable terminal control
sequences. The excerpt may normalize markup and whitespace; it is evidence,
not a byte-for-byte archival body.

### `watchlists_get_item`

Read-only detail tool for one search result.

Parameters:

| Name | Type | Required | Contract |
|---|---|---:|---|
| `item_id` | string | yes | Canonical `local:watchlist_item:<positive integer>` ID returned by search. |

Bare integers, foreign-backend IDs, source IDs, and malformed composite IDs
are rejected rather than guessed. The response includes the same source,
collection, URL, author, status, and explicit date metadata as search, plus a
bounded `evidence.content` field. It includes
`evidence.content_is_untrusted: true` and `evidence.content_truncated`.
Article bodies reuse the existing
`Subscriptions.html_text.readable_body_text()` path, with explicit
`content_normalized: true`, so HTML/control markup does not consume the model's
evidence budget. The normalized text is readable evidence, not an archival
byte copy; `content_format` and `content_kind` remain separate metadata.

If the item has no article body but is a change item, its existing renderable
change evidence (such as `diff_summary`) is returned under an explicitly named
field rather than relabeled as article content.

## Scope resolution

Bare JSON integer IDs and canonical local IDs resolve directly and must exist.
Booleans are rejected even though Python treats them as integers. Numeric
strings are treated as names, not silently coerced into row IDs. Text
resolution is deterministic:

1. Strip surrounding whitespace.
2. Prefer a case-insensitive exact name match.
3. For sources, also allow an exact configured-URL match.
4. Otherwise accept exactly one case-insensitive partial-name match.
5. If several matches remain, return `status: "needs_disambiguation"` with a
   byte-bounded list of candidate canonical IDs and names/URLs. Sources use
   `local:subscription:<id>` and collections use `local:watchlist:<id>`; each
   candidate ID is valid input to the same parameter that produced it.
6. If none remain, return a concise not-found error.

The tool never silently chooses the first ambiguous row. A
`needs_disambiguation` response is a successful tool result rather than an
execution failure, so the agent can immediately retry with a candidate ID.

Collection memberships for all returned sources are loaded in one bounded
query. The result path must not issue one membership query per item or source.

## Continuation

Public continuation is cursor-based, not offset-based. The first call records
the current maximum `subscription_items.id` as `snapshot_max_item_id` and
fetches one lookahead row. Every page admits only rows whose ID is at or below
that boundary. This excludes later inserts without excluding a pre-existing
item whose feed supplied a future publication date. A wall-clock `as_of` is
also returned for user context, but it is not used as an effective-date filter.
The next position uses the query's actual keyset ordering:

```text
(effective_date DESC, item_id ASC)
```

The keyset handles tied effective dates and the existing null-date sink
explicitly. A newly inserted row has an ID above `snapshot_max_item_id`, so it
cannot shift or duplicate later pages regardless of its supplied publication
date.

The versioned cursor contains only:

- cursor format version;
- original traversal `as_of` (context only, not a filter);
- `snapshot_max_item_id`;
- last effective-date/null-state and item ID;
- SHA-256 fingerprint of the normalized query, resolved scope IDs, statuses,
  date floor, and ordering contract.

Normalization collapses query whitespace without changing literal term text,
sorts the unique status set, converts resolved scopes to numeric row IDs, and
normalizes the date floor to UTC. Equivalent status ordering therefore does
not invalidate a cursor, while any filter that can change results does.

It contains no query text, names, URLs, article text, credentials, or database
path. `as_of` is repeated in later response envelopes so every page identifies
the same traversal start, but it is not an admission predicate. The caller
repeats the search filters with the cursor. A fingerprint
mismatch, malformed encoding, unknown version, or invalid key returns a clear
validation error before executing the item query. The fingerprint detects
accidental context reuse; it is not presented as an authentication mechanism.

This is stable keyset continuation, not snapshot isolation. Updates to an
admitted item's effective date, status, or searchable content,
collection-membership changes, and deletions may change later pages. Inserts
made after the first page cannot enter the traversal because their IDs exceed
`snapshot_max_item_id`. Persisting a full result-ID snapshot solely for tool
paging would add state, cleanup, and ownership not justified by this feature.
The response exposes `as_of` and `snapshot_max_item_id`, and documentation
states this limitation.

## Result-size discipline

`LocalToolProvider` has a 32 KiB result ceiling whose generic fallback slices
text bytes. Slicing a JSON string could produce invalid JSON. These tools
therefore pack their own responses to a 30 KiB internal ceiling before
returning, leaving headroom below the provider boundary.

Search serialization adds items one at a time. If the next complete item would
exceed the tool's lower internal budget, it stops, sets `has_more: true`, and
returns a cursor at the last included row. At least one normal metadata-only
item must fit; an individually oversized title, URL, collection name, or
snippet is Unicode-safely truncated with an explicit field-level indicator.

Detail serialization reserves space for the envelope and metadata, then
Unicode-safely truncates body content to the remaining byte budget and sets
`content_truncated`. The serialized final string is re-measured before it
crosses the provider boundary. Serialization uses `allow_nan=False`.
Non-finite stored numeric evidence, such as a malformed change percentage, is
normalized to JSON `null` with a neighboring `<field>_invalid: true` marker
rather than emitting non-standard `NaN` or `Infinity`. Every successful result
remains standards-compliant, parseable JSON.

## Untrusted evidence boundary

Feed titles, authors, URLs, snippets, bodies, diffs, and source names are
external data. Tool descriptions tell the agent to treat `evidence` and other
feed-supplied fields as untrusted facts, never as instructions. Responses keep
article text inside a clearly labeled `evidence` object with
`content_is_untrusted: true`.

JSON encoding escapes control characters, including terminal escape bytes.
The tool does not remove or rewrite substantive hostile-looking text because
doing so would corrupt evidence. Tests can prove delimiting, labeling,
escaping, and byte bounds; they cannot prove that every model will follow the
warning.

Response shaping uses an explicit field allowlist; it never serializes
`auth_config`, `custom_headers`, `rate_limit_config`, `extracted_data`, raw
processing errors, source `last_error`, database paths, or any other column not
named by this contract. Every emitted source or item URL strips URL userinfo
and removes the entire query and fragment. This deliberately does not try to
maintain an exhaustive secret-key vocabulary: signatures, SAS values, OAuth
codes, valueless keys, and future query credential shapes are all removed.
`url_redacted` is true whenever userinfo, a query, or a fragment existed or the
displayed value otherwise differs from storage, so the agent does not mistake a
safety-redacted URL for an archival byte copy. The rule applies equally to
item/source objects, scope metadata, and disambiguation candidates. Resolution
may compare against the raw configured URL internally, but raw userinfo,
queries, and fragments are never returned or logged. The remaining HTTP(S)
path is intentionally preserved for source linkage and is covered by the
operator's Watchlists-tool permission; the contract does not claim arbitrary
stored paths are credential-free. If a stored URL cannot be parsed safely, the
response uses `url: null` and `url_redacted: true` rather than failing the whole
page or echoing the malformed value. Only absolute HTTP(S) URLs with a host are
emitted; hostless or other schemes such as `file:` and `javascript:` also
become null/redacted evidence.

## Architecture

### Shared read-only service

A small synchronous `WatchlistsToolService` in `tldw_chatbook/Tools/` owns:

- argument validation;
- runtime-source check;
- source and collection resolution;
- search/detail orchestration;
- cursor encoding/validation;
- membership enrichment;
- response shaping and byte-safe JSON.

It receives the existing synchronous `SubscriptionsDB` read owner and a
runtime-source loader. Tests inject an in-memory database and fake source
loader. Production does not read UI widget state and does not create an event
loop inside a synchronous tool handler. A narrow handler adapter catches
unexpected storage/implementation exceptions, logs only a bounded exception
category without payloads or paths, and raises one fixed public failure string;
this is required because `LocalToolProvider.invoke()` otherwise exposes the
first 300 characters of an exception message to Console callers.

Console injects the app's existing long-lived `app.subscriptions_db` when it
composes the per-run `LocalToolProvider`. This avoids both an `asyncio.run()`
bridge to `LocalWatchlistsService`'s async facade and reconstruction of
`SubscriptionsDB`: its constructor performs schema initialization and migration
probes, while the app deliberately holds one thread-safe instance with
thread-local SQLite connections.

External MCP supplies a Watchlists-only lazy database resolver to its
persistent local provider. The resolver constructs and caches one
`SubscriptionsDB` read-only view only on the first **local-mode** Watchlists
invocation, after the per-call runtime-source check, not during whole-provider
composition and not once per tool call. The read-only view skips
`BaseDB`/`SubscriptionsDB` schema initialization and opens the existing file
through `connect_private_sqlite(..., read_only=True, must_exist=True)` under a
dedicated registered SQLite owner. It cannot create, migrate, or write the
database file, schema, or rows; SQLite itself rejects writes. Resolving the
configured path may still ensure the profile's private parent directory, an
existing config behavior not represented as a Watchlists database mutation. A
missing or pre-migration database therefore returns `feature_unavailable` and
tells the operator to open the normal application once, rather than silently
changing the database from a read-tagged MCP call. A server-mode call never
resolves or constructs the local database. Construction or schema-probe
failure closes the uncached candidate immediately and becomes a
structured `feature_unavailable` Watchlists outcome and cannot prevent the
filesystem, Git, web, or task tools from registering.

The minimal storage change is a keyword-only read-only construction path on
`SubscriptionsDB`, backed by a keyword-only “skip schema initialization” seam
on `BaseDB` whose default remains initialization-on. Normal application and
test callers are unchanged. The read-only connection branch does not issue
write-oriented PRAGMAs such as `journal_mode=WAL`; it sets only safe
connection-local read behavior and row factories. A small readiness probe
checks the exact core tables and columns needed by these tools before the
successful view is cached. FTS coverage is a separate search-time state: it is
complete only when an item-ID anti-join finds no `subscription_items.id`
missing from `subscription_items_fts_docsize`. Count equality is insufficient
because equal-cardinality sets can contain different IDs. Missing or
incompletely backfilled FTS does not fail the readiness probe: the search path
uses the existing literal LIKE fallback and repeats the anti-join on later
searches. It caches only the monotonic complete state, after which existing
insert/update/delete triggers preserve coverage. This matters for standalone
MCP, whose read-only connection cannot run the app's asynchronous FTS backfill;
the mere presence of an FTS table must not cause permanent false negatives.

Because external MCP dispatches local handlers concurrently through worker
threads, lazy initialization is protected by a `threading.Lock` with a
double-checked cached value. Exactly one successful `SubscriptionsDB` instance
is retained. A failed construction is not cached: that call receives a bounded
`feature_unavailable` outcome, while a later call may retry under the same
lock. Every failed candidate is closed before the lock is released; concurrent
callers never race object construction or readiness probes or accumulate
uncached SQLite handles.

Catalog-only/default `LocalToolProvider` composition may have no Watchlists
database dependency. It still registers the two schemas, but its handlers
return structured `feature_unavailable` results. The dependency is explicit
and optional so existing catalog projection and direct-provider tests do not
touch the user's subscriptions database merely by listing tools.

The handler's dependency order is fixed: validate arguments, load the current
runtime source, return `unsupported` for server mode, and only then resolve the
local database. This keeps the pinned server response authoritative even when
the optional local dependency is absent or broken.

### Existing query path

The implementation reuses `SubscriptionsDB.get_new_items` for list predicates
and adds the following explicit, additive synchronous read seams:

- extended item-search projection/predicates for `effective_date`,
  `snapshot_max_item_id`, keyset continuation, lookahead, and match-centered
  excerpts;
- one authoritative item-detail read joined to its source, which distinguishes
  a missing row from a present row with null content;
- bounded exact/partial source and collection resolvers that do not inherit
  `list_sources(q=...)`'s 1,000-row scan ceiling;
- one batched source-to-collection membership lookup;
- one-row lookahead.

Existing source, collection, status, `since`, literal FTS, LIKE fallback,
projection, and normalization behavior stays single-sourced. The tool must not
reimplement the Watchlists corpus query in a parallel SQL module. These read
seams may live on `SubscriptionsDB` or the existing synchronous
`WatchlistBundleService` as appropriate; no second storage owner is introduced.
The shared search seam chooses LIKE when FTS is absent or the item-ID anti-join
shows incomplete docsize coverage, and returns to FTS when a later check on the
same long-lived owner proves coverage complete.

### Provider registration

Two `LocalToolSpec` entries are added to the default local catalog with names
`watchlists_search_items` and `watchlists_get_item`, source `local`, catalog IDs
`local:<name>`, and empty mutation tags. This follows ADR-030's local domain
prefix precedent and ADR-032's permission boundary.

Both JSON schemas set `additionalProperties: false`. Core validation also
rejects unknown keys, bounds source/collection strings, requires a non-empty
unique `statuses` array when supplied, and rejects booleans anywhere an integer
ID or limit is expected. Schema validation improves clients; core validation
remains authoritative because not every provider path enforces JSON Schema.

Because Console and external MCP already compose `LocalToolProvider`, register
its catalog, and expose approved local tools through `MCP/local_server_tools.py`,
the feature extends those composition seams rather than creating a new
provider. Existing controls continue to apply:

- `[console] local_tools_enabled` for Console;
- the MCP permission store under `local:__local__`;
- per-tool/default permission precedence and session approvals;
- kill switch;
- definition-hash guard;
- `[mcp] expose_local_tools` for external MCP;
- headless external MCP cannot satisfy an `ask` prompt and requires an
  operator-granted allow.

ADR-032's addendum and user-facing help rename the group from “Local workspace
+ web tools” to wording that explicitly includes Watchlists data. The master
switch controls catalog availability, while each Watchlists call still has its
own permission state and default-ask behavior.

## Runtime-source behavior

Each call reads the current profile-scoped runtime source through an injected
loader backed by the existing `runtime_policy.json` contract. It does not
capture the source at provider construction, because the user can switch
between local and server while the process remains open.

If the source is `server`, the tool returns a successful, structured expected
outcome rather than `ToolResult.error`:

```json
{
  "status": "unsupported",
  "retryable": false,
  "message": "server Watchlists search is not supported; switch Watchlists to Local before retrying"
}
```

It performs no local database search in that path. An absent or malformed
runtime-policy file follows the existing runtime-policy loader's local default.

## Error behavior

- No matches: successful `status: "ok"` response with `items: []`.
- Ambiguous scope: successful `status: "needs_disambiguation"` response.
- Missing scope or item: successful structured `status: "not_found"`,
  `retryable: false` outcome.
- Invalid query, unknown key, status, date, limit, ID, or cursor: successful
  structured `status: "invalid_argument"`, `retryable: false` outcome before
  the item query.
- Server mode: successful structured `status: "unsupported"`,
  `retryable: false` outcome with the pinned message.
- Missing/failed Watchlists dependency: successful structured
  `status: "feature_unavailable"` outcome. It is retryable only when the
  failure is plausibly transient.
- FTS5 unavailable or query-time FTS operational failure: existing literal
  LIKE fallback.
- Other database failure: concise tool error without SQL, paths, content, or
  credentials. The Watchlists handler adapter supplies a fixed public message;
  raw exception text never reaches `LocalToolProvider`'s generic exception
  formatter.
- Permission deny/timeout/kill-switch/gate error: unchanged ADR-032 provider
  result, before the tool core runs.

Expected domain outcomes deliberately cross `LocalToolProvider` as
`ToolResult(ok=True, content=<valid JSON>)`. This is load-bearing for external
MCP: `ChatbookGatewayRuntime.call_tool()` preserves successful content but
maps any unrecognized `ToolResult.error` to the generic “Local tool execution
failed.” Permission refusals remain `ToolResult.error` and use the gateway's
existing security-reviewed allowlist. Unexpected storage/implementation
failures also remain generic errors. The shared service returns structured
expected outcomes; `LocalToolProvider.invoke` remains the never-raise public
boundary.

## Testing

### Database and service tests

- Blank query returns newest effective items across every status.
- Full-text terms match title, author, and a deep-body occurrence.
- FTS operators are literal; FTS5 failure takes the LIKE fallback with `%`,
  `_`, and backslash remaining literal.
- A present but partially backfilled FTS table also takes the LIKE fallback and
  returns a deep-body match missing from FTS. Equal counts with wrong member IDs
  remain incomplete, and a partial-to-complete transition on the same
  long-lived owner switches a later search back to FTS.
- Search excerpts center on the matched field and remain bounded.
- Collection, source, status, `since`, `snapshot_max_item_id`, and their
  intersections compose.
- Keyset continuation covers multiple pages, equal dates, null-date sink,
  deletion, and insertion of newer or future-dated rows. The first call's
  `snapshot_max_item_id` admits every row that already existed, including a
  future-dated item, while excluding every later insert.
- One lookahead row drives `has_more` and is not returned.
- Membership enrichment is bounded and batched, with a query-count assertion
  guarding against N+1 behavior.

### Tool-core tests

- Defaults and every argument boundary: 512 characters, 32 terms, status
  allowlist, date formats, limit 1/10/50, canonical IDs.
- `additionalProperties: false`, unknown-key rejection, bounded scope strings,
  non-empty unique statuses, and boolean-as-integer rejection.
- Exact case-insensitive, exact URL, unique partial, ambiguous, missing, and
  collection/source-intersection resolution.
- Bare integer and canonical source/collection IDs round-trip from every
  disambiguation candidate; numeric strings remain names.
- Cursor round trip, version rejection, malformed encoding, filter fingerprint
  mismatch, and absence of raw filters/content from decoded cursor payload.
- Distinct source and item timestamp fields remain distinct.
- Source active/paused state accompanies freshness dates.
- Stored raw URL userinfo, fragments, and queries never appear in any
  source/item/scope/disambiguation output, and `url_redacted` reports that
  transformation.
- Non-HTTP(S), hostless, and malformed stored URLs become null/redacted rather
  than executable or local-path-shaped evidence.
- Only contract-allowlisted fields serialize; auth/header/raw-payload/error
  canaries do not appear anywhere in output or logs.
- Every search/detail response parses as JSON and remains below the provider
  ceiling, including Unicode and individually oversized fields.
- A stored non-finite change percentage is normalized to `null` with its
  invalid marker; strict serialization with `allow_nan=False` never emits
  `NaN` or `Infinity`.
- Detail returns bounded article or change evidence with accurate truncation.
- Detail reuses `readable_body_text` and labels normalization while preserving
  separate content-format/kind metadata.
- Prompt-injection-shaped and terminal-control-shaped feed text remains
  delimited, labeled untrusted, and JSON-escaped.
- Server mode returns the pinned message and a spy proves the database was not
  queried.
- Structured `invalid_argument`, `not_found`, `unsupported`, and
  `feature_unavailable` outcomes cross the provider as successful valid JSON;
  unexpected storage failures remain scrubbed tool errors.

### Provider and integration tests

- Both schemas appear in the default local catalog with `local:` IDs and no
  mutation tags.
- Console composition injects `app.subscriptions_db` and exposes both tools
  only when local tools are enabled.
- Catalog-only composition performs no database construction and returns
  `feature_unavailable` only if a Watchlists handler is actually invoked.
- External MCP's first local-mode call opens an existing database through the
  registered read-only SQLite seam; tests prove a missing database file is not
  created, schema SQL cannot run, a write statement fails, and an old schema
  is not migrated.
- A failed external lazy-construction/readiness candidate is closed before a
  later retry; a concurrency probe proves one successful cached owner and no
  leaked failed-candidate handles.
- One representative allow/ask/deny path proves these registrations use the
  existing provider boundary; the generic permission matrix remains owned by
  existing provider tests rather than being duplicated per tool.
- External MCP registration exposes both schemas only behind
  `expose_local_tools`; headless ask fails closed and operator allow executes.
- A guarded external Watchlists dependency failure leaves every unrelated
  local tool registered and callable.
- The agent runtime can discover, load, and invoke both tools through its
  normal progressive-disclosure path.
- At least one max-size response is parsed after
  `LocalToolProvider.invoke(...).content`, proving `_fit_result` did not slice
  it, and expected-error JSON is parsed through
  `ChatbookGatewayRuntime.call_tool()`, proving external MCP does not replace
  it with the generic local failure.

### Isolated live verification

Use a scratch profile that sets `TLDW_CONFIG_PATH`, `XDG_CONFIG_HOME`, and
`XDG_DATA_HOME` before importing the app. Write an explicit scratch data
directory and subscriptions database path into that profile's TOML; do not
repurpose `HOME` or rely on an omitted path falling back to the real profile.
Assert every resolved runtime-policy, config, and database path is under the
scratch root before seeding synthetic sources, collections, and items,
including hostile text and a deep-body match. Verify:

1. Console discovers and invokes search, continuation, and detail.
2. An allowed external MCP client invokes the same tools and sees equivalent
   JSON.
3. Scope disambiguation and separate timestamps are visible.
4. Switching the scratch runtime policy to server produces the structured
   non-retryable unsupported result with no local read.
5. The real runtime-policy and subscriptions files are unchanged afterward.

No live test uses the user's real feeds or relies on a bare interpreter without
the repository's test isolation.

## Documentation

- Add tool names, parameters, literal-full-text semantics, permissions, and
  local-only limitation to Console/local-tool and MCP exposure documentation.
- State that external content is untrusted evidence and that continuation is
  stable keyset ordering, not snapshot isolation.
- Amend ADR-032 for Watchlists/private-domain reads under the synthetic local
  principal and update “Local workspace + web tools” permission/help copy to
  name Watchlists evidence.
- Link TASK-16222, ADR-030, and amended ADR-032 from the implementation plan and
  final task notes.

## Acceptance mapping

| TASK-16222 criterion | Design section |
|---|---|
| Search local items with filters and continuation | Tool contracts; Scope resolution; Continuation |
| Newest-first, linked, dated, bounded evidence | User-facing semantics; Result-size discipline; Untrusted evidence boundary |
| Retrieve one item | `watchlists_get_item` |
| Resolve human scopes | Scope resolution |
| Stable keyset and cursor validation | Continuation |
| Honest server-mode behavior | Runtime-source behavior |
| Preserve permission/exposure boundary | Provider registration |
| Automated and isolated live proof | Testing |
