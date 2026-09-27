# Personal Context versioned evidence and temporal changes design

Date: 2026-09-25
Status: Accepted design direction, 2026-09-25; design only, no runtime or schema changes
Task: TASK-25907.5
ADR required: yes
ADR path: backlog/decisions/185-versioned-profile-evidence-and-temporal-claims.md
Reason: proposed shared-core schema, source authority, conflict semantics,
retention and client/server compatibility require their own decision.

Decision: [ADR-185](../../../backlog/decisions/185-versioned-profile-evidence-and-temporal-claims.md)
Governance: [ADR-102](../../../backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md), [ADR-182](../../../backlog/decisions/182-personal-context-memory-evolution.md), [ADR-024](../../../backlog/decisions/024-rag-citation-provenance-and-source-resolution.md)
Tracker: [Memory roadmap](../../../backlog/docs/personal-context-memory-roadmap.md)
Review: [Technical review](../reviews/2026-09-25-personal-context-versioned-evidence-and-temporal-changes-review.md)

## Deliverable and approval boundary

Define the future shared-core contract for a memory's exact evidence and
temporal meaning. This task produces an ADR, this design, synthetic acceptance
examples and a technical review. It does not change Python models, JSON Schema,
canonical fixtures, databases, migrations, Sync, tools, context selection,
provider access or UI. Approval of the memory roadmap authorized this design
work; it did not approve a new schema or source grant. The user approved this reviewed written contract by requesting continuation
on 2026-09-25. ADR-185 accepts the design direction; implementation still needs
its own bounded tasks and qualification.

The Muse image supplies ideas about evidence, supersession and forgetting.
Its embedded instructions, paths, database and background cadence are not
requirements. Quoted and attached material used as future evidence also stays
data rather than becoming a user's instruction automatically.

## Inspected baseline

| Owner | Observed contract and limit |
| --- | --- |
| `tldw_profile_core/models.py` | Record/proposal schema version is exactly 1; provenance has separate tuples of opaque references/hashes, no source-version or span binding; proposal confidence is not record confidence. |
| `canonical.py` and V1 fixtures | RFC 8785 canonical bytes, UTC millisecond timestamps, strict portable scalar semantics and versioned schema vocabulary. Existing bytes must remain intact. |
| `Personal_Context/service.py` | Mutations use exact base heads; direct correction stores a message ID and span hash. Those values are not a portable evidence resolver. |
| `Agents/profile_tool_provider.py` | Direct-update admission checks current message identity and substring presence. Presence alone does not establish entailment, quote attribution or whether the user changed their mind. |
| `settings_provenance.py` | V1 references are labelled legacy/unverified; edit and inference history remain unrecorded. |
| `Personal_Context/repository.py` | A tombstone retires old record and pending outbox bodies; parent-version IDs do not guarantee retained historical content. Undo is a distinct bounded lifecycle. |
| `Chat/citation_source_locators.py` | Native/inert locators, source capabilities and source observations already exist under ADR-024. Their locator payloads do not promise immutable historical source versions. |
| `Personal_Context/reconciliation.py` | Current client contract compares canonical V1 objects and an exact server schema requirement. V2 object capabilities need a reviewed extension. |

The companion server was not inspected in this task. Shared-core and server
obligations below are rollout gates, not claims that server code already meets
them. No current source authority or quoted body was read from a real profile.

## Approaches and recommendation

1. **V2 record/proposal contract with governed excerpt derivatives — chosen.**
   Claim metadata travels atomically with its canonical record version. Exact
   source resolution remains runtime-owned, and optional excerpts are encrypted
   derivatives under Personal Context custody. This fits whole-object Sync and
   explicit record authority.
2. **Canonical evidence sidecar objects.** These would need independent heads,
   grants, tombstones, ordering and atomic record-plus-sidecar Sync. They could
   be useful for large shared evidence graphs, but are unnecessary for bounded
   profile claims and make citation completeness harder to reason about.
3. **Runtime-only links or extra V1 fields.** A local projection is useful for
   inspection but cannot carry portable semantics; extra unversioned V1 fields
   change canonical bytes and risk silent loss at older peers. Neither is the
   durable contract.

The design does not introduce an append-only truth ledger, a universal resolver,
global snapshot deduplication, automatic support verification or newest-wins
learning.

## Shared-core V2 data contract

Publish distinct `ProfileRecordV2` and `ProfileProposalV2` models and schemas;
do not widen the existing V1 classes in place. A distinct V2 manifest carries
profile-wide required context semantics and an evidence-retirement epoch;
scope models need no new schema unless their own meaning changes. Negotiated
capabilities name each supported object schema and semantic vocabulary rather
than infer support from package version or a single transport integer. Pending
proposals carry a V2 proposed record and bind their base version as strictly as
V1. The profile-wide activation fence is required even if some records remain
V1: an opaque V2 relation can change whether a related V1 claim is usable.

V2 retains record identity, controls, immutable version/parent links and typed
payloads. It adds the following bounded concepts, all data-only:

| Concept | Required meaning |
| --- | --- |
| `claim_basis` | `direct_user_assertion`, `inference`, `imported_assertion` or `legacy_unknown`; a trusted capture path records origin, not a model's guess about the speaker. |
| `evidence_bindings` | At most eight identified bindings to exact source representations; no executable path, URL, class name or resolver function. |
| `support_assessments` | At most one current assessment per binding ID, binding digest and claim digest: `not_assessed`, `supports`, `contradicts` or `insufficient`, attributed to its method/actor and assessment time. |
| `approval_receipt` | Optional user action (`authored`, `accepted`, `accepted_edited`) bound to the exact canonical claim digest, record version and action time. It is separate from support assessment. |
| `confidence_estimate` | Optional finite 0–1 estimate with actor, method, assessment time and claim digest. It is an inference estimate; no calibration is claimed without separate calibration evidence. |
| `salience_hint` | Optional explicit priority hint with actor/time. Durable high salience requires user choice or accepted proposal; it never changes truth or permissions. |
| `temporal_validity` | Tagged `unknown`, explicitly chosen `standing`, or a half-open UTC-millisecond interval with its basis. Missing dates do not silently mean standing. |
| `relations` | At most four exact record/version edges: `correction_of`, `change_from`, `supersedes` or `workspace_exception_to`, with the reviewed operation that created them. |

The payload keeps its existing 16 KiB canonical ceiling. Proposed V2 claim
metadata has a 16 KiB canonical ceiling; the whole canonical record is bounded
to 64 KiB. IDs use bounded, nonblank, control-free strings of at most 128
characters and 512 UTF-8 bytes. Lists reject duplicate binding/edge identities.
Unknown enum values, malformed dates, nonfinite numbers, wrong scalar types and
cross-field inconsistencies are rejected by both models and semantic schema
validation. These proposed bounds are frozen before future implementation and
covered by overflow fixtures; they do not modify V1's accepted values.

Support, approval and confidence bind a versioned `claim_digest`: SHA-256 of
the canonical projection containing payload, claim basis, temporal validity and
relation edges. It excludes assessments, approval receipts, current source
observations and the digest itself, so it is not self-referential. Changing
wording, effective dates or a correction/exception relation invalidates old
attribution. The canonical fixture fixes the projection's keys, defaults and
ordering; salience and privacy actions have their own version-bound review and
never inherit a content approval as permission.

Evidence has a separate `binding_digest`: SHA-256 of the canonical complete
binding, including authority, object, representation, version/digests, span,
source role, capture time and optional excerpt identity. It excludes its own
digest and disposable access observations. Every support assessment names the
binding ID, binding digest and claim digest. Changing any bound evidence value
requires a fresh assessment even when the claim digest is unchanged; no prior
assessment can attach to a substituted source. A Settings edit records the new
authoring action rather than inheriting old source support. Imported approval,
source-role and assessment labels describe peer assertions; they cannot prove
local user intent or grant current local authority.

### Exact source binding

Each binding has a unique binding ID and contains:

- source owner/kind and authority kind (`local_profile` or authenticated tenant),
  opaque authority and governance-scope IDs, and source object ID;
- a source representation identifier, such as exact user-message text or note
  body, with no normalization or joined-field transformation;
- an immutable owner version token plus digest of that representation, or an
  explicit captured-representation token when the owner has no history;
- zero-based Unicode codepoint span offsets `[start, end)` into that exact
  decoded representation and SHA-256 of the exact UTF-8 span;
- source role: direct user message, quoted material, attachment, tool result
  or imported material, and trusted capture time;
- an optional governed excerpt handle, never inline excerpt text in the
  canonical claim metadata.

A captured-representation token proves what was captured, not that the source
owner retains that historic version. Its exact excerpt may be inspectable only
while a governed snapshot exists. A bare hash cannot reconstruct a quote.
Source adapters must declare how they obtain exact immutable text and validate
offsets/digests. Initially admit only conversation-message and note-revision
adapters that can meet that contract; unsupported kinds stay inert. Attachments
and tool results cannot be relabelled direct user assertions.

Core validates shape and bounds. The runtime's source owner validates actual
authority, source identity, version and span. Canonical binding data is never
itself an access token or a native binding. A peer/import cannot set `native`
and gain access. Current authorization produces a disposable native resolver
request locally; any adapter to ADR-024 checks both its existing capabilities
and this additional exact-version contract.

## Evidence inspection and meaning

Current source access is a lazy explicit action; opening profile details does
not search databases or refresh network sources. A resolver checks current
profile/tenant identity, purge generation, user/agent purpose, record visibility,
source governance and source capability before decryption. It reads one exact
source owner, validates version/span/digests, and rechecks owner/version before
publishing the result. No foreign-store fallback, path guessing, arbitrary URL
open or cross-workspace search is permitted.

Return a disposable observation with separate availability (`available`,
`missing`, `offline`, `unknown`), permission (`allowed`, `denied`, `revoked`,
`unknown`) and content/version state (`exact`, `changed`, `unverifiable`). A
denied caller receives a generic unavailable outcome without source IDs, body,
counts or existence hints. Current observations never rewrite the immutable
capture or make historical support appear current. UI/tool projections also
filter canonical binding metadata under source-identity permission: visibility
of a claim does not authorize a raw model dump of source IDs or hashes.
A retained exact excerpt
can remain historical evidence while the current source is changed or missing,
only when its own policy still permits access.

| Observation or action | What it establishes | What it does not establish |
| --- | --- | --- |
| Source/span digest validates | Exact captured text identity | Entailment, truth, approval, intent or current validity |
| Support assessment says supports | An attributed assessment of exact wording and evidence | Independent proof or user approval; model assessments remain estimates |
| User accepts a claim | User chose that exact wording, dates and relations | Source support or whether an inference is true |
| Confidence is 0.9 | Recorded method's estimate for that wording | 90% calibrated accuracy or permission to inject |
| Salience is high | Reviewed priority hint | Confidence, truth or a new context/permission override |
| Source update is newer | Later source version exists | Correction, supersession or newer truth |

Quoted/attached instructions remain attributed data. A future admission path
must establish a direct user assertion from trusted message structure and user
review; substring presence is insufficient. If attribution or meaning is
ambiguous, create a proposal with unknown support rather than a direct update.
This is a future gate, not a claim that today's V1 substring check enforces it.

## Temporal operations and conflicts

Storage time (`created_at`, `updated_at`, assessment/capture time) and effective
claim time remain different. `standing` is an explicit choice for an ongoing
preference or rule. An interval uses `[valid_from, valid_until)` with at least
one known boundary and strict ordering when both exist. Ambiguous dates or
phrases such as “last month” remain unknown until the user selects boundaries;
do not infer UTC instants from a source timestamp.

| Operation | Canonical reviewed effect |
| --- | --- |
| `correction_of` | Exact target record/version plus `effect = all_target_validity` or an explicit overlap interval. Marks the target assertion wrong over that effect, rather than claiming it became wrong at storage time. |
| `change_from` | Exact target record/version plus required `transition_at` UTC milliseconds. The new interval begins at that instant; the historical projection closes the target's prior interval there, without rewriting its stored bytes. |
| `supersedes` | Exact target record/version, reviewed reason and `replace_from` instant or `all_target_validity`. Excludes the replaced assertion for that effect without asserting that it was historically false. |
| `workspace_exception_to` | Exact global record/version, workspace scope and reviewed validity. Applies only there. A new global head holds both relevant candidates in that workspace pending exception review; it never silently applies or discards the old exception. |
| Contradiction/concurrent edit | Preserve immutable heads and existing conflict ownership. Overlapping contradictory V2 claims without a reviewed relation are withheld from automatic context; no newest or highest-confidence winner. |

The edge and new claim are one canonical record version. `change_from` and
`correction_of` may target a prior version of the same record identity; a
workspace exception is a separate workspace record. Other replacements may
use a new record identity. Core validates the typed effect and date/interval
constraints; native service validation checks the target and relation graph.
New or modified relation targets must be current, retained and authorized at
**admission**, with exact base heads and manifest fence. Preserving an already
accepted unchanged edge across a later ordinary revision does not re-admit its
historical target or reinterpret its effect. Reject cross-profile targets, invalid scope
direction, cycles, missing review and stale targets. A previously accepted
historical edge does not become invalid just because its target later gains a
successor. Missing/deleted target content stays unavailable, never reconstructed
from the edge. Only a current accepted successor chain affects current selection;
competing heads hold the affected candidates until conflict resolution.

Repository admission commits the new record/edge and manifest transition
atomically, comparing all involved target heads in the same transaction. A
server must apply the same checks and cannot expose a new head without its
relation effect. Old canonical bytes are unchanged. Historical selection
computes the reviewed effect from retained authorized versions; it does not
create an unapproved earlier start date. Unknown transition time creates a
proposal, never a fabricated disjoint interval. Source changes only signal
review and cannot perform temporal mutations. Edges confer no permission to
read targets; deletion still retires content-bearing evidence and relation data.

### Concrete change and correction

At review time, record `preference-A`, version `A1`, says “Use detailed replies”
with interval `[2026-08-01T00:00:00.000Z, null)`. On September 15 the user explicitly
says “From today, use concise replies,” then confirms the UTC boundary below
in review. The runtime does not derive midnight from “today.” Accepted version
`A2`, parent `A1`, has
interval `[2026-09-15T00:00:00.000Z, null)` and this reviewed edge:

```json
{"kind":"change_from","target_record_id":"preference-A","target_version_id":"A1","transition_at":"2026-09-15T00:00:00.000Z"}
```

`A1`'s bytes still have their original interval. The accepted edge makes its
historical projection `[2026-08-01T00:00:00.000Z, 2026-09-15T00:00:00.000Z)`;
`A2` is selected from September 15. The review binds the exact target, boundary
and new payload. A different head arriving before admission rejects the operation;
a later `A3` does not invalidate this already admitted historical edge.

Alternatively, “That was wrong; I have preferred concise replies since August 1”
produces `A2-correction` over `[2026-08-01T00:00:00.000Z, null)` with
`correction_of(A1, effect=all_target_validity)`. It does not preserve a period in
which `A1` was correct. These are alternative histories, not simultaneous heads.
A user-approved supersession with the September 15 replacement boundary would
stop using `A1` then without making either historical truth assertion.

### Eligibility of a V2 claim

Selection requires current record visibility/purpose authorization, resolved
heads/relations, a version-bound user approval and explicit standing or a valid
current interval. `unknown` validity remains withheld until reviewed. Salience
cannot displace hard constraints, authority or budget rules.

A direct user-authored claim may remain usable if its source later becomes
missing: absence alone is not falsity. Inference/imported claims additionally
require at least one exact bound `supports` assessment and a currently permitted,
exact source observation or governed snapshot; contradictory bound assessments
hold the claim for review. Those are the required support conditions, not an
implicit confidence threshold. A contradiction hold uses an explicit structured
relation, a bound contradictory assessment or an existing conflict group; the
contract does not promise detection of every semantic contradiction in free
text. Core shape validation cannot judge entailment. Automatic selection does
not open a new source:
without a valid observation whose authority/version can still be checked, it
withholds the claim and requests explicit refresh. Source-policy revocation or
forgetting takes precedence, including the retirement hold below. A user may
explicitly author a new direct assertion; the system cannot silently convert an
inference to one. V1 eligibility stays unchanged within a compatible active
profile, subject to the profile-wide V2 compatibility fence.

## Retention, disclosure and derivatives

Optional excerpt payloads stay in Personal Context's encrypted repository as
separately governed derivatives, with a random revocation identity and exact
claim-version/binding ownership. Each is at most 1,000 codepoints and 4 KiB UTF-8;
no full conversation or attachment is copied to obtain a short citation.
Snapshots are not FTS/RAG content, logs, crash metadata or global deduplication
keys. Managed exports require a separate source-and-record export capability.

The least permissive source and claim policy governs the excerpt and binding
metadata. A local-only binding requires the canonical claim to stay device-only
unless the user explicitly creates a new reviewed claim without that binding;
whole-object Sync never silently strips evidence. A syncable portable binding
may reach a compatible peer without its optional excerpt only if the canonical
reference truthfully says the excerpt is not part of that object's transport.
The peer cannot invent a source body or native authority from the handle.

New evidence is Settings-inspectable only by default. Tools, automatic model
context, child runs, interviews, exports, embeddings and maintenance cannot
receive it just because the claim is agent-visible or syncable. Their future
disclosure gate must independently authorize the evidence source and purpose.
The current ADR-102 device-only discrepancy and unscoped quarantine signal are
not repaired by this design task and are not new permissions.

Record tombstones retire excerpt content, old record/evidence versions, pending
outbox bodies and local derived caches according to one transactional cleanup
inventory. Source deletion alone neither proves a claim false nor authorizes
silently deleting an independently approved claim. Source-policy revocation, however,
must also retire restricted **inline metadata**, not just excerpt bodies.

### Privacy retirement without deleting an approved assertion

1. The native source-policy owner signals the encrypted dependency inventory.
   Fence linked inspection, automatic selection, export, Sync and late resolver
   publication immediately under an exact policy/manifest generation check.
   The inventory covers bindings, excerpts, assessments and other metadata
   derived from them, plus canonical history, outboxes, recovery/Undo copies and
   managed caches. Do not log source IDs, hashes or body-bearing receipts.
2. Create a service-authored privacy-only successor under exact heads. Preserve
   the approved assertion's payload, controls and temporal meaning, remove the
   restricted binding and derivative metadata, and record a content-free
   retirement receipt with a random identity. Retire entire older immutable
   envelopes containing restricted metadata; never rewrite their bytes in
   place. Preserve only permitted content-free lineage. Do not mint or rebind
   a human approval receipt: any retained historical approval still names its
   original version. Hold automatic use until the user reviews the sanitized
   successor; inference support is not silently upgraded.
3. Advance the V2 manifest `evidence_retirement_epoch` and distribute a bounded,
   encrypted control receipt naming retired envelope/derivative random IDs,
   without source locators, source digests or quotations. Each receipt has at
   most 32 retired random IDs and a 16 KiB canonical ceiling; a larger inventory
   is partitioned into a bounded-page retirement operation. The manifest stays
   fenced until every page in that operation is durably applied; partial cleanup
   cannot acknowledge the epoch. Atomically retire local
   histories, pending outbox bodies, managed recovery/Undo copies, snapshots
   and caches. Workers and restore/import paths compare the epoch before
   publishing; older bodies cannot be reapplied or re-exported.
4. The home server fences old writes and releases only the sanitized successor
   plus retirement receipt to compatible peers. Peers retire all managed
   matching copies and acknowledge the epoch before context use and Sync resume.
   An offline/unacknowledged peer cannot resume through the server or replay an
   old envelope. Local decryption/disclosure on an offline device cannot be
   promised stopped before it receives the retirement signal; UI must report
   outstanding acknowledgements rather than claim completed remote erasure.

A source policy that forbids retaining source identity must be enforceable at
admission and later revocation; otherwise that source cannot be admitted as
portable evidence. Remote evidence admission requires this lifecycle on every
active peer. The epoch is an evidence-retirement barrier, not a replacement for
profile purge generation or a secret transport grant. Explicit forgetting,
suppression of re-extraction and concurrent-job cancellation are completed and
verified in the separate forgetting design before rollout. Unmanaged exports
and backups cannot be promised erased. Managed restore must apply current
retirement receipts before making recovered content usable.

## V1 migration and compatibility

1. Keep all V1 schemas, custom vocabulary and canonical-byte fixtures unchanged.
   Read V1 references, support, approval history, confidence, salience and temporal
   meaning as legacy/unknown; preserve V1's current eligibility behavior.
2. Upgrade only through the canonical service with exact base-version and
   manifest fences. A user-reviewed V2 rewrite gets a new version. Do not pair
   independent V1 reference/hash tuples by position, find matching local IDs,
   infer support from reason codes or infer dates from update times.
3. Publish V2 JSON Schema plus required semantic vocabulary, deterministic
   serialization, models and fixtures in a new pinned shared-core release.
   Retain RFC 8785 and portable UTC-millisecond rules. Package both V1 and V2
   fixture sets in the source and distribution paths.
4. Negotiate V2 manifest/record/proposal/evidence semantics with the home server
   before V2 writes or sync on a linked profile. Activation requires every
   registered context consumer to acknowledge support or be explicitly disabled
   with its old profile grant retired. The server refuses cutover while any
   active V1-only consumer remains; it does not deliver V2 semantics to such a
   consumer and hope opaque retention will make context safe. A dormant offline
   copy cannot be guaranteed stopped remotely: it must upgrade or be explicitly
   removed from active use before cutover, and cannot reconnect on its old grant.
   Unlinked profiles similarly fence all local consumers before V2 opt-in;
   linking then requires V2 support.
5. An unsupported manifest or required vocabulary blocks the **whole profile's**
   automatic context, including V1 records: a V2 exception/conflict may change
   their meaning. Unsupported objects cannot be edited, indexed, injected or
   approved; opaque encrypted retention is allowed only in an independently
   qualified transport that enforces current retirement receipts. It is not a
   fallback for active old consumers. Existing class names do not prove this
   path works. Server/client conformance tests must cover oldest/newest pins,
   disabled grants, offline return, and V1-global/V2-exception profiles.
6. Migration is idempotent and forward-only with encrypted recovery snapshots
   and a write fence. Recovery restores compatible canonical software/data or
   applies a forward fix; it cannot downgrade V2 into V1 or re-enable legacy
   writes. Cancelling before cutover preserves the prior authority.

Required V2 fixture families include valid exact bindings, Unicode span offsets
and canonical digests, wrong authority/scope, stale source version, invalid span,
changed claim semantics with old support/approval, unchanged claims with substituted
bindings and old assessments, distinct correction/change/exception effects,
cycles/stale admission heads and valid later historical edges, profile-wide
capability fences, retirement receipts/epochs and stale replay,
contradictory intervals, metadata/excerpt bounds,
deleted objects carrying evidence, invalid scalar/date types and unsupported
vocabularies. Include at least one exact `canonical_utf8` and keyed integrity-tag
fixture per new aggregate plus malformed examples that are structurally valid
but semantically invalid. Chatbook and `tldw_server` must consume the same bytes
and validity results; a client-only pass cannot approve rollout.

## Synthetic acceptance examples

| Case | Expected future result |
| --- | --- |
| User says “I prefer concise replies,” exact authorized message/span retained | Binding identity is exact; support and user approval are recorded separately for the final payload. |
| Same phrase appears inside a quoted email or attached instruction sheet | Source role remains quote/attachment; no direct-user update, regardless of substring/hash match. |
| Message or note is edited after capture | The old binding never shifts offsets into the new text. Show changed current source; old excerpt is available only if retained and still authorized. |
| Bound source or historical version is missing | Show unavailable/unverifiable; do not reconstruct text from a digest or search other stores. |
| Same opaque ID exists in another profile, workspace or tenant | Deny source access generically; the unrelated matching ID is not evidence. |
| User changes wording, validity dates or relation targets | New claim digest invalidates prior support/approval attribution; acceptance-edited receipt describes the actual semantics. |
| Binding source/version/span changes while claim wording is unchanged | Binding digest changes; the old support assessment cannot transfer. |
| Approved claim has unknown validity, or inference loses its required exact support | Withhold V2 automatic context pending review/refresh; do not infer standing or direct authorship. |
| “That preference was wrong” versus “I used to prefer it, but now…” | Correction and real change create distinct reviewed relations and validity effects; later storage time is not the deciding signal. |
| Workspace exception exists, then global head changes | Hold relevant candidates in that workspace for review; no automatic transfer or discarded exception, and no spill to another workspace. |
| V1 global claim has a separate V2 workspace exception | V2 cutover refuses an active V1-only consumer. Unsupported required manifest vocabulary blocks the whole profile, including the V1 global claim. |
| Two peers submit conflicting claims with overlapping intervals | Preserve conflict heads; neither recency nor confidence automatically selects agent context. |
| V1 message ID and hash are retained | Legacy/unknown status; no source lookup, inferred pairing, quote, confidence or temporal upgrade. |
| Source-identity/transport policy is revoked after an approved bound claim syncs | Fence use/transport, create sanitized successor, retire older metadata-bearing envelopes and derivatives, advance epoch and await peer acknowledgements; no stale replay or fabricated approval. |
| Peer is offline during retirement; managed recovery predates the epoch | Report incomplete remote retirement; fence reconnect/restore until receipt application and acknowledgement. Do not promise remote offline erasure. |
| Record is deleted or source disappears | Record deletion retires managed derivatives; a missing source alone does not declare an independently approved claim false. |
| V1-only server or unknown V2 vocabulary | Refuse V2 cutover/active use, apply profile-wide fence and qualify any opaque retention separately; never downcast to an applied V1 record. |

### Exact Unicode span vector

The message representation is `Hi 👋 — café`. Its 11 codepoints include one
astral emoji and precomposed `é`; no Unicode normalization is allowed. For
`[7, 11)`, the exact span is `café`, UTF-8 hex `636166c3a9`, with SHA-256
`850f7dc43910ff890f8879c0ed26fe697c93a067ad93a7d50f466a7028a9bf4e`.
The full representation's SHA-256 is
`bdd355f021e0095bbd9ce9da729cfd0b9556b01bc01c5b59461ebc4e12e55bef`. A binding must contain both representation and span digests,
an exact owner version/capture token and authorized scope. Decomposed `e` plus
combining acute, UTF-16 offsets, or offsets into an edited message must fail
this vector. These are concrete input/digest examples, not V2 canonical-byte
fixtures or a runtime source-verification claim.

These are declared future acceptance requirements, not passing runtime tests.
Each must become a production-entry regression with successful authorized
controls before the corresponding implementation is considered shipped.

## Future implementation boundaries

The bounded implementation sequence is shared-core contract/conformance,
encrypted evidence and migration, exact source capture/resolution, reviewed
temporal operations and V2 selection, then client/server qualification. Forgetting
and provider disclosure must be shipped prerequisites for enabling governed
excerpts and agent evidence use. These are areas for later atomic tasks, not
placeholder task IDs or implementation authorization. No background extraction,
automatic approval or consolidation is added by accepting this design.

## Technical review checklist

- [x] No source ID, imported locator or hash is treated as authority or support.
- [x] Direct assertion, quote, approval, confidence, salience and validity remain distinct.
- [x] Correction, real change, exception and concurrent conflict have separate behavior.
- [x] V1 truthfulness, immutable bytes, negotiation, recovery and server gates are explicit.
- [x] Evidence retention/disclosure and revocation cover derivatives without promising unmanaged erasure.
- [x] All six task criteria have concrete design sections and synthetic examples.
- [x] Only documentation/tracker files changed; production code and fixtures remain untouched.
