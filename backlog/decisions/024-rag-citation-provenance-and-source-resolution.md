# ADR-024: Adopt canonical RAG citation provenance and governed source resolution

Status: Accepted
Date: 2026-07-23
Related Task: N/A — create or select a Backlog epic and atomic child tasks
before implementation planning
Supersedes: N/A

## Decision

Chatbook will represent answer-level RAG provenance as one immutable, versioned
`CitationTrace` sealed at the terminal generation boundary. The trace records
retrieval runs, the exact evidence submitted to each provider request, bounded
answer and repair attempts, structural citation mappings, semantic trust
results when available, and policy state.

The sealed trace contains immutable identities, hashes, validation results, and
opaque payload references. Exact submitted text and retained non-final attempt
bodies live in separately governed payload records so revocation or secure
purge can remove text without rewriting historical trace metadata.

Submitted and cited text will use authority-scoped `EvidenceSnapshot` records.
Current source access will use versioned `SourceLocatorEnvelope` values resolved
through a static allowlisted registry. Snapshot storage, source identity,
current resolution, native open, external open, comparison, and export are
separate policy capabilities.

Local traces will use database-backed, transactional persistence in the message
ownership boundary. Existing evidence bundles, citation validation metadata,
and sidecar records remain compatibility inputs and synthesize partial
`legacy_inferred` traces. New writes use only the canonical trace.

Citation provenance will be a message-owned adjunct for Sync v2 and will
synchronize only when the server advertises a compatible trace schema and
snapshot mode. The optional RAG API `grounding_trace` field may carry the
canonical trace while existing server document and citation arrays remain
supported.

## Context

The existing citation implementation has useful but fragmented contracts:
`EvidenceReference`, `EvidenceBundle`, `CitationRef`, answer citation parsing,
Console staging state, widget validation metadata, and a JSON sidecar. These
objects do not establish one authoritative record of:

- which retrieval execution produced evidence
- which transformed text was actually submitted to the provider
- which answer attempt or repair cited it
- whether citation syntax, semantic support, and current-source state differ
- how provenance survives reload, artifacts, export, import, and sync

The sidecar rewrites a whole JSON document and is unsuitable for durable,
deduplicated snapshots or atomic message-plus-provenance persistence.

The reviewed `tldw_server` pipeline already distinguishes retrieved evidence,
derived evidence, chunk lineage, citation structure, claim verification, and
trust. It may rerun retrieval or repair an answer before the final response.
Therefore a client trace assembled before the terminal boundary can describe
evidence that did not produce the final answer.

Source navigation also crosses security boundaries. Historical traces can
contain local paths, URLs, server identities, or workspace-governed text.
Treating those values as executable metadata would enable path traversal, SSRF,
unsafe external opens, tenant leaks, and revoked-content disclosure.

A canonical ADR is required because the solution changes storage and migration,
data ownership, Sync v2 behavior, client/server service contracts, cross-module
interfaces, source navigation, and privacy and authorization policy.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Extend `EvidenceBundle`, validation dictionaries, UI state, sidecars, and exports independently | Preserves several competing sources of truth and cannot reliably model retries, prompt transformations, cache reuse, migration, or revocation. |
| Store only final citation markers and source IDs | Cannot prove which text was submitted, distinguish cited from merely supplied context, or preserve a historical answer after source changes. |
| Reconstruct complete provenance from existing server `documents` and citation arrays | Those arrays may omit prompt transformations, retries, and finalization history. Reconstructed traces must remain explicitly partial. |
| Use a full append-only provenance event ledger | Adds ordering, replay, compaction, retention, and sync machinery beyond what a single-user TUI needs. |
| Resolve arbitrary classes, paths, or URLs directly from trace metadata | Makes imported and historical data executable and bypasses authority, path-containment, URL, and SSRF policy. |
| Store all snapshots in the existing JSON sidecar | Whole-file rewrites, poor transactionality, unbounded growth, and weak garbage collection make it unsuitable. |
| Deduplicate snapshots globally | Identical text across tenants or governance domains could bypass revocation and confidentiality boundaries. |
| Automatically refresh sources during inspection or export | Makes historical viewing non-deterministic, adds latency, and can trigger unwanted network or source access. |
| Treat structurally valid markers as proof of support | Conflicts with the server trust contract and would mislabel related but non-supporting evidence as verified. |

## Consequences

- `CitationTrace` becomes the canonical answer-level provenance contract.
- Mutable trace construction remains request-scoped and seals exactly once.
- Retrieval runs, prompt evidence sets, and answer attempts remain distinct.
- The exact submitted snapshot is durable when policy allows; otherwise a
  server reference, ephemeral, or redacted record is used.
- Ephemeral and seal-time-redacted evidence cannot produce a persisted fully
  grounded trust state. Later revocation preserves the seal-time record but
  changes current access and produces an explicit warning.
- Content-addressed deduplication is limited to a compatible authority,
  confidentiality, tenant, and source-governance scope.
- Structural validity, claim support, and current-source observations remain
  independent.
- Current-source observations use separate bounded mutable storage and never
  rewrite the sealed historical trace.
- Active message ownership is bound to the selected answer body hash. Editing,
  importing, replacing, or conflict-resolving different text invalidates the
  grounded association without deleting the historical trace.
- Inline `[S#]` markers and a compact Sources footer expose provenance without
  rewriting the answer.
- The shared inspector uses Console's right rail, existing Library detail
  regions, or a narrow full-screen fallback rather than default modal flow.
- Current resolution is lazy and asynchronous; historical snapshots render
  first.
- Resolver capabilities are allowlisted and policy-derived. Imported locators
  are inert until explicitly validated and rebound.
- Personal deletion normally retains historical snapshots. Governed revocation
  and secure purge can redact or remove them.
- Snapshot revocation does not silently rewrite independently owned assistant
  messages; policies requiring derived-answer removal use an explicit wider
  secure purge or quarantine action.
- Local provenance moves from sidecar-only storage to versioned SQLite tables
  with atomic message ownership, bounded JSON, snapshot references, retention,
  and migration journaling.
- Citation snapshots are excluded from FTS and RAG indexing.
- Legacy records remain readable as partial traces and are not silently
  upgraded to complete provenance.
- Legacy free-form paths, URLs, and content references remain inert. Only a
  fresh allowlisted resolver lookup under current authority can produce a
  native locator.
- Sync provenance is disabled until a compatible server capability is
  advertised. The active Sync v2 server contract remains authoritative.
- Server RAG responses may add `grounding_trace` without removing existing
  response fields.
- A valid supported server trace takes precedence over legacy arrays. Unsupported
  or malformed traces may fall back only to validated partial legacy
  provenance; tenant or authority mismatches are rejected.
- Export is deterministic and policy-filtered. Source refresh is always an
  explicit separate action.
- Academic and bibliographic citations remain separate typed records.
- Implementation requires an epic and atomic Backlog tasks split by contract,
  persistence, migration, pipeline stage, UI surface, resolver family,
  artifact/export/import/sync surface, and qualification gate; this is not a
  single-PR change.

## Rollback plan

- Disable new trace writes through a recovery switch while leaving supported
  stored traces readable.
- Do not down-migrate or drop provenance tables.
- Do not delete legacy sidecars automatically.
- If optional server `grounding_trace` handling is disabled, continue consuming
  existing document and citation arrays as partial provenance.
- If a resolver is disabled, retain snapshots and show current-source
  resolution as unavailable rather than falling back to unsafe native or URL
  behavior.
- If sync compatibility is withdrawn, keep local traces authoritative and sync
  messages without claiming remote provenance completeness.

## Links

- [Design spec](../../Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md)
- [Prior citation carry-through design](../../Docs/superpowers/specs/2026-05-23-citation-snippet-carry-through-epic-design.md)
- [ADR-003: Settings Library/RAG defaults](003-settings-library-rag-defaults.md)
- [ADR-005: Invest in local RAG](005-invest-in-local-rag-mirroring-tldw-server.md)
- [ADR-008: Sync v2 M1 contract](008-sync-v2-client-m1-contract-alignment.md)
- [`tldw_server` reviewed revision](https://github.com/rmusser01/tldw_server/commit/d9c245ac14c40df855d1ab6cd19b3c137b16b47b)
