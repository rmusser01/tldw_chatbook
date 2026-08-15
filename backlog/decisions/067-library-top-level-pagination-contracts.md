# ADR-067: Library Top-Level Pagination Contracts

Status: Accepted
Date: 2026-08-14
Related Tasks: TASK-16481, TASK-16482, TASK-16483, TASK-16484, TASK-16485,
TASK-16486, TASK-16487, TASK-16488
Supersedes: N/A

## Decision

Top-level flat Library browse sources use source-owned, exact-total, bounded
page contracts rather than a generic Library data controller. Conversations,
Prompts, Media items, Skills, and Collections expose at most 20 summary records
per applied page. Each source retains its existing filter, sort, selection,
mutation, detail, worker, and authority owner.

Every service page response includes bounded summary items, source-native page
coordinates, and an exact total read coherently with the items. Limit/offset
ordinary browse responses echo the exact request. Conversation/Collection
stable-ID locator responses instead return a resolved page-aligned offset/page;
they also return the target's zero-based rank and must contain the target at
`rank - offset`. Validation requires
`offset = (rank // page_size) * page_size`, an agreeing resolved page, and the
same source order, in addition to total/cardinality/identity validation. Prompt requests use `page`/`page_size`, while
their existing normalized response uses resolved `current_page`, compatibility
alias `page`, and `per_page`: `per_page` matches the request, `current_page` is
the deterministic clamp implied by the requested page and exact total, and
`page` must equal `current_page`. The Prompt request fingerprint binds the
original requested scope. Ordering uses the source's established sort plus a
stable final identity key. Pages are live reads: totals describe one response
transaction and do not freeze a multi-page browse session.

Responses must contain the exact number of rows implied by their coherent
total and returned coordinates: `min(page_size, remaining_total)`. Undersized
non-final pages and oversized pages fail closed so no range becomes
unreachable or falsely represented. Source-specific summary validation also
requires each item to carry a valid stable identity that is unique within the
page; malformed items fail closed rather than being silently discarded.

The Library screen owns requested and applied scope, last-good page records,
request generation, and `uninitialized`/`fresh`/`stale` presentation state.
Exact total/range metadata exists only while fresh; stale presentation carries
a source-owned recovery reason without fabricating ranges.
Only a current generation may apply. One pure shared display function may
derive range, total-page, disabled-control, and recovery copy; existing source
canvases render it with their own IDs. There is no generic pager widget or
composer, and the function owns no source data, workers, widgets, or messages.
The broad Library snapshot remains valid for rail, landing, and RAG consumers
and cannot overwrite a source's dedicated page state.

Conversation and Media selection remains current-page scoped and clears when
their applied scope changes. Prompt selection preserves its existing immutable
cross-page, version-captured basket and explicit total/on-page summary; paging
or Prompt scope changes neither clear nor add entries and do not flatten it
into the current-page model.

Cross-visit screen snapshots persist only the last successfully applied scope,
never records, loading/error state, failed requests, or unsubmitted drafts. A
fresh Library screen re-fetches that scope. Explicit navigation context keeps
precedence under ADR-033. Unmount invalidates source request generations; an
already-running local thread read may finish but cannot apply to the old or new
screen.

Conversation deep links and Collection mutation placement use source-specific,
bounded page-containing-ID reads. Each such read returns the owning page and
exact total from one coherent snapshot under the same deterministic ordering;
its zero-based target rank, resolved page/offset, page-local target position,
and target ID must agree. It does not walk pages, materialize an unbounded list, or
inject an extra row into page 1. Malformed locator envelopes fail closed.

Any fresh exact limit/offset response that proves its requested page is out of
range causes one generation-guarded reload of the last valid page before
application. This applies to local mutations and external/concurrent source
shrinkage. A validated Prompt response already contains the coherently clamped
page and applies directly without a second read. Only one automatic
limit/offset clamp is attempted. If its second coherent response is also out
of range because the source changed again, the last good records enter the
existing stale presentation: exact total/range copy and row/bulk/pager actions
are suppressed, while Retry and scope recovery remain available. Invalid page
metadata is never applied, and only an authoritative success clears stale.

The existing Conversation page read is brought under the same rule: its count
and rows execute in one read transaction rather than two independent DB calls.

Media paging uses true storage-level limit/offset execution and a complete
distinct-type facet read. The type chooser is a bounded scrollable list rather
than one mounted Button per type. Skills filter name and description only,
before sorting and slicing. Their page envelope also carries the source-wide
blocked total and first blocked Skill name so trust recovery never depends on
which page is visible.

Source mutations invalidate older reads before committing. A successful
mutation is not reclassified as failed when its follow-up page read fails. The
screen reconciles the known affected record locally, marks retained results
stale, suppresses exact-total copy, disables stale row/bulk actions, and offers
an authoritative Retry. Pager controls remain disabled without an exact
boundary, while scope controls may recover through a fresh page-1 request.

Page diagnostics are metadata-only. The modified Media read path must not log
queries, titles, record bodies, paths, credentials, or stable private IDs.

## Context

The Library currently mixes capped broad snapshots, full in-memory lists, a
50-row Prompt controller, and a 20-row Conversation controller. Users can be
unable to reach records beyond a canvas-sized cap, while large sources can
mount many Textual widgets. Applying a second UI slice to a partial snapshot
would make the interface look paged without making the full source reachable.

The in-scope sources already have distinct service, mutation, trust, and detail
boundaries. Replacing them with one polymorphic paging controller would move
domain state away from the owners established by ADR-003 and expand the root
application state contrary to ADR-033. At the same time, service additions for
true Media offset paging, stable-ID owning-page reads, complete facets, and
Skills trust aggregates are cross-module contracts that need one durable
decision.

Exact totals and stable bounded pages extend ADR-030. They do not promise a
frozen session: concurrent writes may move later boundaries. Generation guards
protect presentation correctness when cancelled Textual workers leave an
already-running local thread read to finish.

Notes folders, Media Trash, and Collection members are hierarchical or nested
surfaces with separate ownership and are intentionally handled by atomic
follow-up tasks.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Add one generic Library page controller | It would centralize unrelated source scopes, mutations, trust, and recovery state and conflict with existing ownership ADRs. |
| Page only the rows already present in the broad snapshot | Records outside that partial snapshot would remain unreachable and totals/facets would be false. |
| Fetch every record and slice in the screen | It bounds mounted rows but not storage work or memory and makes deep pages increasingly expensive. |
| Use infinite scrolling | It complicates deterministic focus, recovery, exact ranges, and mounted-widget bounds without being required for 20-row reachability. |
| Freeze a read snapshot across the whole browse session | It would hold storage snapshots or copy full result sets, complicate mutations, and exceed the established live-read contract. |
| Persist page records across navigation | Records become stale and duplicate service ownership; ADR-033 already provides lightweight memory-only scope snapshots. |
| Keep current Conversation deep-link row injection | It makes page 1 contain a record outside its deterministic ordering and invalidates its range/total semantics. |
| Derive Media types or Skill trust totals from the current page | Facets and trust recovery would vary falsely by page and could hide reachable data or actions. |

## Consequences

### Benefits

- Every in-scope top-level record is reachable in a consistent 20-row UI.
- Exact totals, filters, facets, trust actions, and deep links describe the
  complete source rather than the current page.
- Mounted list widgets and Media type chooser widgets remain bounded.
- Source-specific mutation, detail, policy, and runtime ownership is preserved.
- Stale requests and committed-but-unrefreshed mutations remain truthful and
  recoverable.
- Navigation restores lightweight scope without persisting private records.

### Accepted trade-offs

- Source services expose parallel page envelopes instead of one polymorphic
  interface.
- Live pages may shift under concurrent writes.
- Local Skills still scan and classify their filesystem index before returning
  a bounded page.
- Page-containing-ID queries may rank the active source to return one bounded
  page.
- Failed/unsubmitted filter drafts are not restored across a fresh destination
  screen.
- Notes, Media Trash, and Collection-member paging remain separate follow-up
  work.

## Links

- [Design specification](../../Docs/superpowers/specs/2026-08-14-library-top-level-source-pagination-design.md)
- [ADR-003: Settings Library/RAG Defaults Boundary](003-settings-library-rag-defaults.md)
- [ADR-030: Direct Local Library Tool Boundary for Console and MCP](030-local-library-agent-tool-boundary.md)
- [ADR-033: Application Session State Ownership](033-application-session-state-ownership.md)
