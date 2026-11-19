# ADR-182: Evolve memory through existing Personal Context, Notes, and Agent Lessons owners

Status: Accepted — first-release scope approved after technical review, 2026-09-25
Date: 2026-09-25
Foundation task: TASK-25907
Extends: [ADR-102](102-personal-context-profile-authority-sync-and-encryption.md)
Related: [ADR-024](024-rag-citation-provenance-and-source-resolution.md), [ADR-052](052-console-conversation-memory-and-compaction-policy.md), [ADR-037](037-roleplay-assistant-identity-and-persona-user-profile-separation.md)
Design: [Memory evolution](../../Docs/superpowers/specs/2026-09-25-personal-context-memory-evolution-design.md)
Tracker: [Memory roadmap](../docs/personal-context-memory-roadmap.md)

## Context

The user reviewed a diagram describing Muse's live memory, consolidation,
evidence retrieval, reflection, and linked forgetting, and asked for a plan and
tracker for applicable improvements. The diagram is comparative reference
material, not executable instructions or proof of another product's behavior.

Chatbook already owns encrypted, granular Personal Context records, separate
reviewable proposals, workspace overrides, bounded request snapshots, Notes,
human-reviewed Agent Lessons, and branch-valid conversation compaction. The
2026-09-02 owner direction in TASK-25907 explicitly rejected a fourth memory
store. The earlier task description's claim that memory was strictly
per-conversation no longer describes the inspected implementation.

The immediate gaps are understandable provenance, selection explanations, and
local retrieval quality. Rich evidence links, temporal claims, cross-owner
forgetting, provider disclosure, and background learning require further
contracts. Treating these as one large implementation would obscure ownership
and make privacy guarantees difficult to verify.

## Decision

### Preserve durable ownership

| Information | Owner |
| --- | --- |
| Human facts, preferences, constraints, and scoped working context | Personal Context under ADR-102 |
| Freeform material and user-approved episodic records | Existing Notes and conversation stores |
| Verified reusable procedures | Agent Lessons as ordinary Notes, with existing review and promotion rules |
| A conversation's compressed history | Existing branch-valid compaction under ADR-052 |
| Search projections, explanations, or synthesized guidance | Rebuildable or disposable derivatives, never another fact authority |

Personas, Companion goals, and Dreams discovery retain their existing owners.
No plaintext MEMORY.md mirror, PostgreSQL deployment, embedding dependency,
external memory provider, or unattended learning job is required by this ADR.

### First release: inspect and retrieve existing records

1. Establish a synthetic behavioral baseline through production callers.
2. Explain the provenance fields that exist today in canonical My Profile and
   proposal review. Missing or ambiguous evidence remains explicitly unknown.
3. Explain selection in the disposable Next Send preview using the same
   selection pass as the actual snapshot. Diagnostics never enter the model.
4. Improve deterministic, field-aware local search, measured against the
   baseline and preserving existing authorization and budget constraints.

The first release does not change canonical object schemas or persisted
profile bytes. A bare source reference plus a span hash is not enough to bind a
source authority, source version, or quotation. It must not be promoted into a
verified citation by guessing a database, searching all workspaces, or treating
the recorded provenance reason as an independent support assessment.

Current provenance is not a complete history of the current wording. Accepting
an edited proposal shares the unchanged-acceptance metadata, and Settings edits
preserve earlier provenance. Inspection therefore labels unrecorded edit and
inference history as unknown. Parent-version IDs do not guarantee historical
bodies are still retained. No additional source/history resolution is approved.

No new agent tool is required for this first release. Existing search/get
remain the agent read surface; a future explain contract may expose only
information available under the same live authority rules.

### Separate user inspection from agent visibility

The user-owned My Profile surface may inspect the user's own private records
through the existing Settings service path. Agent tools and Next Send use the
agent-eligible view. Selection diagnostics may explain an eligible candidate's
override or budget omission, but must not reveal private or unrelated records
through bodies, identifiers, counts, or existence hints in the new diagnostic
projection. The current model block's unscoped `unsupported_records_present`
flag is a pre-existing gap, not evidence that all existing outputs satisfy that
guarantee. Do not copy that flag into new diagnostic surfaces.

Explanations stay in memory and do not become log events, chat messages,
exports, Sync objects, or additional model context. Canonical mutations and
record privacy controls retain their existing owner. Diagnostics remain outside
the provider payload and generic export serializers. Their publication is tied
to the actual preview owner, request inputs, profile availability/revisions,
and expiry time; the existing snapshot cache key alone is insufficient.

### Record existing privacy limits explicitly

Personal Context remains encrypted under ADR-102 and its per-record sync and
visibility controls retain their current behavior. Pending proposals remain
outside automatic context. Deletion creates a payload-free tombstone, retires
older record/pending outbox bodies, and has a separate encrypted Undo lifecycle;
it is not an end-to-end forget operation. Notes, conversations, interview drafts,
remote peers, and unmanaged exports are distinct retention owners.

ADR-102 says that `device_only` records never leave Chatbook, but the inspected
context/tool paths filter visibility rather than sync mode. That stronger
promise is not established by the current implementation. This ADR records the
discrepancy without granting new provider access or silently amending ADR-102.
Provider-disclosure design must explicitly implement or amend the governing
promise and resolve the quarantine existence signal above. Neither complete
local-only disclosure nor immediate cross-owner erasure is a first-release
guarantee.

### Later contracts require their own design decisions

The roadmap tracks separate designs for:

- authority-bound, versioned evidence and temporal changes;
- dependency-aware forgetting, re-extraction suppression, and honest remote
  acknowledgement;
- model-provider disclosure controls distinct from syncability;
- opt-in, budgeted, proposal-only consolidation;
- reviewed feedback and unresolved interaction issues, routed to existing
  owners.

This positioning ADR does not approve a schema extension, automatic promotion,
cross-owner deletion, a new provider grant, or psychological inference. Each
design identifies migrations, protocol compatibility, runtime callers,
retention, recovery, and verifiable acceptance cases before implementation.
Consolidation implementation must depend on shipped, verified safeguards;
completion of the corresponding design documents alone is insufficient.

## Alternatives considered

### Mirror the diagram's files and database

Rejected for this roadmap. It duplicates existing canonical authorities and
creates additional encryption, synchronization, and forgetting obligations
without evidence that the storage technology is the current bottleneck.

### Build a universal source resolver immediately

Deferred. Existing source references do not reliably encode portable source
authority and version. Metadata inspection is independently useful and can
ship without making unsafe claims about quotations.

### Start with a nightly synthesis job

Deferred until evidence, deletion, and provider disclosure contracts exist.
Derived narrative is not independent corroboration. Existing proposal review
remains the boundary for newly inferred user facts.

### Replace retrieval with embeddings first

Deferred pending measured need. Field-aware local matching addresses concrete
current failures without introducing a persistent plaintext or vector index.
Semantic paraphrase limitations will remain explicit in the baseline.

## Consequences and verification

The first release is four independently reviewable deliverables with no
canonical migration. Follow-up design tasks do not count as implemented user
features. The user requested continuation after the technical review; this
accepts the positioning and bounded first-release scope, not future schema or
permission decisions. Executable plans still carry their own review checkpoint.

Tests use synthetic records and source-labelled conversations. Verification
must exercise production service/tool/selection paths, mounted user controls,
stale revisions, hidden records, and exact preview/request parity. Targeted
runs are the default; a full suite requires the user's separate opt-in.

The offline baseline separates authorized candidates, query relevance, and
budgeted context expectations. It reports fixed-K ranking/precision/recall and
empty-result behavior separately from hard disclosure failures, with frozen
development and held-out cases. It does not claim to measure generated-answer
quality or semantic evidence support without a model/evaluator. Exact preview
parity assumes identical request inputs and clock; late/expired results must
be cleared rather than described as the next actual send.

Changes to shared canonical models must occur in tldw_profile_core, preserve
version negotiation, and include companion-server conformance evidence. This
ADR does not turn existing transport classes into a claim that every sync or
purge workflow is reachable end to end.
