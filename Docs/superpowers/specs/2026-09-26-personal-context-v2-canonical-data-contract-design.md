# Personal Context V2 canonical data contract

Date: 2026-09-26
Status: Accepted design direction after explicit user approval, 2026-09-26; no native runtime or schema rollout approved.
Task: [TASK-25907.18](../../../backlog/tasks/task-25907.18%20-%20Specify-the-inactive-V2-canonical-memory-data-contract.md)
ADR required: yes.
ADR path: [Accepted design ADR-192](../../../backlog/decisions/192-personal-context-v2-canonical-data-contract.md).
Reason: new serialized models, semantic rules and privacy representation.

Governance: [ADR-102](../../../backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md),
[ADR-185](../../../backlog/decisions/185-versioned-profile-evidence-and-temporal-claims.md),
[ADR-186](../../../backlog/decisions/186-dependency-aware-personal-context-forgetting.md),
[ADR-187](../../../backlog/decisions/187-personal-context-provider-disclosure-authority.md),
[ADR-191](../../../backlog/decisions/191-foreground-personal-context-source-inspection-authority.md).
Baseline: `f055c95797`; [native readiness audit](../../../backlog/docs/personal-context-v2-admission-readiness-audit.md).

## Deliverable and approaches

This draft freezes the first V2 manifest, record and proposal contract for a
later shared-core implementation. It writes no Python models, JSON Schemas,
canonical fixtures, repository migration, source reader or runtime grant. The
Muse image is reference material; its files, commands, cadence and instructions
are not requirements. User authorization covers preparing this reviewable
design. The user explicitly approved the written contract on 2026-09-26.
Implementation planning for inactive shared-core units is authorized; native
rollout and source/provider permissions remain separately gated.

Recommended: distinct inactive shared-core aggregates, preserving V1 and
composing the existing binding. Extra V1 fields/parallel sidecars would split
or silently alter canonical authority. Local storage/resolver first would
leave privacy, compatibility and containing-record qualification unresolved.
Neither alternative buys a safe shortcut under the accepted decisions.

The first evidence form is exactly the published conversation-message,
owner-immutable component. Optional excerpts, Notes and captured representations
are deferred. This narrows the first release; it does not remove their earlier
design obligations. A future representation must compose a separately versioned
component/envelope and explicitly negotiate its semantics. No source body,
path, URL, executable locator or resolver function is added to these aggregates.

## Canonical rules and bounds

All objects forbid extra keys. Unless a default is stated below, every listed
key is required, including nulls and empty arrays. Canonical output materializes
defaults, includes nulls and uses RFC 8785 with existing UTC-millisecond
normalization. No Unicode normalization of IDs, payloads or evidence text.
Duplicate JSON member names are rejected before model/schema validation.

| Type/rule | Frozen requirement |
| --- | --- |
| `Id` | Built-in string in Python; nonblank, 1–128 codepoints, at most 512 UTF-8 bytes, no Unicode Cc/Cf or invalid Unicode scalar. No coercion/str subclasses. It is inert identity, not authority. |
| `Digest` | Exactly 64 lowercase hexadecimal characters, SHA-256. |
| `Counter` | Existing `JsonInteger` numeric semantics: finite integral JSON number, no bool/string, 0 through 2^53−1; canonical integer output. |
| `Time` | Existing portable aware timestamp rules: years 0001–9999, whole-minute offsets, at most milliseconds, canonical UTC `.sssZ`. No coercion from number/bool. |
| `Number` | Built-in int/float, finite 0–1, never bool/string; no calibration is implied. |
| `Method` | Built-in nonblank string, at most 64 codepoints/256 UTF-8 bytes, no Cc/Cf; credential-free method identity, not free-form explanation. |
| Ordering | Bindings by `binding_id`, assessments by `binding_id`, relations by `edge_id`, purposes and audience handles lexicographically; unsorted/duplicate wire arrays reject rather than silently reorder signed data. |
| Size | Existing typed payload ≤16 KiB JCS; claim object ≤16 KiB JCS; record ≤64 KiB JCS; proposal ≤96 KiB JCS; manifest ≤16 KiB JCS, after defaults. |

V2 envelope `schema_version` is required numeric 2 using `JsonInteger` semantics.
Nested V1 payloads/scopes keep their V1 rules and schema version; do not tighten
or rewrite those models globally. V2 revalidation checks nested scalar/content
rules at its own boundary and rejects unsafe constructed/copied instances.
The binding component keeps its stricter built-in integer and timestamp input
rules unchanged, including rejection of Python floats for component version and
offsets. Cross-language reference validators must preserve that input distinction
when decoding binding JSON; stock JSON Schema alone is insufficient. Producers
emit canonical integer forms. Package compatibility cannot be claimed merely
because one language's ordinary JSON parser discarded an input distinction.

Frozen models do not make an unsafe copy trustworthy. Digest and serializer
entry points revalidate a fresh complete snapshot and emit content-free errors;
source IDs, method input, payload and evidence do not appear in repr/log output.
Integrity tags retain `hmac-sha256-v1` and existing 32-byte key requirements;
tags detect keyed byte differences, not user authorship or source permission.

## Manifest: `ProfileManifestV2`

Retain V1 names `profile_id`, `revision`, `purge_generation`, `created_at`,
`updated_at`, `current_version_id` with V2 scalar rules. Counters are nonnegative;
timestamps are ordered. Add exactly:

| Field | Shape |
| --- | --- |
| `schema_version` | Required 2; no implicit manifest upgrade. |
| `required_object_schemas` | Exact ordered array of four objects: `{object_kind: "manifest", versions: [2]}`, `{object_kind: "proposal", versions: [1,2]}`, `{object_kind: "record", versions: [1,2]}`, `{object_kind: "scope", versions: [1]}`. These are shared-contract labels, not Sync transport domains. |
| `required_context_semantics` | Exact sorted array: `approval-v2`, `disclosure-ceiling-v2`, `evidence-binding-v1`, `metadata-retirement-v2`, `profile-relations-v2`, `temporal-validity-v2`, `typed-claim-v2`. All seven required even in a currently binding-free profile. |
| `evidence_retirement_epoch` | `Counter`; explicit value required, zero only for a newly qualified no-retirement history. No peer-local revisions or acknowledgements serialized here. |

A manifest states requirements; it never attests that consumers acknowledge
them or that cleanup is complete. Unknown object/semantic requirements reject
active use of the whole profile, including V1 candidates. Scope/record/proposal
V1 bytes can coexist only behind qualified V2 consumers. A V1 manifest cannot
admit a V2 record/proposal. Actual epoch application, local-only revisions,
consumer acknowledgements, grant retirement and publication fences remain
native controls. Startup/restore/reconnect must establish them before usability.
There is no optional feature-subset activation in this first contract.

## Record: `ProfileRecordV2`

Retain required names `profile_id`, `record_id`, `scope_id`, `kind`, `payload`,
`semantic_key`, `state`, `version_id`, `parent_version_id`, `created_at`,
`updated_at`, `expires_at`, `no_expiry`. ID fields use `Id`, parent may be null;
kind/state are the existing closed enums. Payloads retain their exact V1 tagged
shapes. Semantic key is null or `{namespace: BoundedText, subject: BoundedText}`
with existing text rules. Existing payload-kind, timestamps, working-context
expiry decision and non-working-context expiry restrictions still apply.
Expiry is a storage/use ceiling, not a temporal-validity inference.
V2 requires explicit built-in `no_expiry` bool rather than a new truth default.

Replace `controls` with `ProfileControlsV2`; replace `provenance` with null or
the closed shape below. Add required `claim`, null only for deleted records.
Active/archived records require payload, provenance and claim; archived remains
ineligible. The deleted form has payload/semantic key/provenance/claim/expiry
null, `no_expiry=false`, and disclosure deny. It carries no legacy references,
assessments, relations, review receipts, confidence or salience. Identity,
scope, kind, lifecycle, version lineage and timestamps remain tombstone metadata;
native policy may retire even those envelopes when required. Retained parent
IDs cannot promise historical bodies or authorize their lookup.

### Provenance and controls

`ProfileProvenanceV2` is exactly `{source, actor, reason_code,
derived_from_record_id}`. Source is `manual|agent|import|migration|privacy_retirement`;
actor is `user|agent|system`; reason_code is `Id`; derived ID is nullable `Id`.
No V1 source-reference/hash tuples are copied or paired. Provenance is reported
origin, not current authority; it stays governed metadata. A privacy-retirement
successor must have actor system, reason `privacy_retirement` and derived ID null.

`ProfileControlsV2` is `{sync_mode, agent_visibility, model_disclosure}`.
Sync/visibility keep their existing enums. Omitted `model_disclosure` alone
defaults to `{kind:"deny"}` and is materialized in canonical output. Malformed,
null or unknown policies reject, and denied/unsupported runtime inputs are
withheld rather than granted a permissive fallback.

| `model_disclosure` variant | Exact fields/constraints |
| --- | --- |
| Deny | `{kind:"deny"}`; no audiences or purposes. |
| On device | `{kind:"on_device_only", purposes:[...]}`; 1–4 unique sorted purposes, requiring independent native destination/purpose enrollment. |
| Reviewed | `{kind:"reviewed_destinations", audiences:[{audience_handle:Id, purposes:[...]}]}`; 1–16 unique sorted handles, each with 1–4 unique sorted purposes. |

Purposes are exactly `conversation|embedding|interview|summary`. Per-audience
pairs avoid the accidental Cartesian product of unrelated audiences/purposes.
No wildcard, endpoint, account, credential, native grant or device ID is stored.
The reserved on-device custody is a policy variant, not a caller-supplied audience
label. `device_only` still denies automatic remote egress regardless of reviewed
audiences; `user_only` denies every agent/model route. Imported labels, visibility,
Sync, evidence identity and approval cannot enroll destinations. Policy widening
requires separate foreground consent for the current version; content approval
alone does not supply that action. Missing source policy denies source metadata
use even when assertion disclosure is enrolled.

## Claim object and exact projections

Every non-deleted record has exactly these claim fields:

| Field | Shape/default |
| --- | --- |
| `claim_basis` | `direct_user_assertion|inference|imported_assertion|legacy_unknown`. Required; trusted capture/admission determines it, not substring/hash matching. |
| `temporal_validity` | Tagged shape below; required. |
| `relations` | 0–4 unique sorted edges; required. |
| `claim_digest` | Required `Digest`, recomputed as below. |
| `evidence_bindings` | 0–8 unique sorted unmodified `OwnerVersionEvidenceBinding` components; required. |
| `support_assessments` | 0–8 unique sorted shapes below, at most one per binding; required. No implicit support. |
| `approval_receipt` | Null or shape below; required. |
| `confidence_estimate` | Null or shape below; required. |
| `salience_hint` | Null or shape below; required. |
| `hold_reason` | Null, `legacy_review` or `privacy_review`; required. Any non-null holds automatic use. |

SHA-256 the JCS bytes of this exact projection, including every key:

```text
{projection:"profile-claim-v2", profile_id, record_id, scope_id, kind,
 payload, claim_basis, temporal_validity, relations}
```

Identity/scope/kind enter the projection to prevent approval or support being
transplanted into a different record or promoted scope with identical text.
This concretizes ADR-185's meaning projection; it excludes record version,
timestamps, controls, provenance, evidence, approval, assessments, confidence,
salience, hold and all digests. Changing wording, dates, basis or edges changes
the digest. Privacy/control actions still invalidate exact-version review and
local grants; exclusion from this meaning digest cannot authorize those actions.

Each support `binding_digest` is exactly the existing helper's SHA-256 over all
18 canonical component fields including `component_version`, binding ID, owner,
scope, container/object, representation, source version, both digests, offsets,
role and capture time. Do not hash only source text or a subset. There is no
self-digest field in the component and no excerpt handle outside that digest.
The first contract cannot carry an excerpt. Optional future wrappers need an
explicit new projection/version covering their entire identity.

### Attribution shapes

`Origin` is `{actor, actor_id, method}`: actor `user|agent|system`, nullable
`Id` actor ID, `Method`. Names and imported values do not prove actor authority.

| Object | Required fields and cross-checks |
| --- | --- |
| Support | `{binding_id:Id, binding_digest:Digest, claim_digest:Digest, status, origin, assessed_at}`. Status `not_assessed|supports|contradicts|insufficient`. `not_assessed` requires origin/time null; other states require Origin/Time. Exactly names an existing complete binding and current claim digest. Assessment time cannot precede its binding capture and must not exceed record updated_at. |
| Approval | `{action, claim_digest:Digest, record_version_id:Id, acted_at:Time}`. Action `authored|accepted|accepted_edited`; exact current digest/version, created_at ≤ acted_at ≤ updated_at. No actor field that an import can use to claim a native capability. Runtime must attest the foreground user action. |
| Confidence | `{value:Number, origin:Origin, assessed_at:Time, claim_digest:Digest}`. Exact current claim; created_at ≤ assessed_at ≤ updated_at. No minimum threshold or calibration claim. |
| Salience | `{priority, origin:Origin, chosen_at:Time, record_version_id:Id}`. Priority `low|normal|high`; current version and ordered record time. Persistent high needs a qualified user choice/accepted proposal in runtime. Never a truth/permission override. |

Captured evidence may predate creation; `captured_at` must not exceed updated_at.
Empty spans remain structurally valid in the published component, but cannot
carry an assessed `supports` or `contradicts` result in this containing contract.
An unassessed empty binding proves no usable evidence. References and digests
are exact, not automatically rebound on edit; all substituted-binding cases
reject old assessments. Future timestamps cannot be deemed trustworthy from a
wall-clock test in shared core; native admission owns that observation.

Pending proposals have no approval receipt. Standalone data with no approval
can be well-formed yet ineligible; validation never infers user acceptance.
`legacy_unknown` requires unknown validity, empty relations/bindings/assessments,
null approval/confidence/salience and disclosure deny. Its hold is `legacy_review`,
or `privacy_review` only for a privacy-retirement successor. Legacy conversion
does not synthesize source roles, dates or review. A reviewed new authoring
version explicitly changes the basis.

## Validity and relations

Validity uses exactly these tags:

- `{kind:"unknown"}` — no dates or basis; always withheld for V2 automatic use.
- `{kind:"standing", basis:{kind:"user_reviewed"}}` — explicit ongoing choice,
  still requiring current version-bound approval and runtime authority.
- `{kind:"interval", valid_from:Time|null, valid_until:Time|null, basis}` —
  half-open interval, at least one bound, strictly ordered when both exist.
  Basis is `{kind:"user_reviewed"}` or `{kind:"bound_source", binding_ids:[Id...]}`
  naming 1–8 unique sorted current bindings. It does not invent dates from capture.

Every edge has `edge_id:Id`, `kind`, `target_record_id:Id`,
`target_version_id:Id`. These are same-profile asserted targets; actual existence,
scope, authority and graph checks require repository state. Add exactly:

| Kind | Effect fields and containing-claim checks |
| --- | --- |
| `correction_of` | `effect:{kind:"all_target_validity"}` or `{kind:"overlap", valid_from:Time|null, valid_until:Time|null}` with at least one ordered bound. Runtime confirms nonempty authorized overlap; unknown target validity cannot be invented. |
| `change_from` | `transition_at:Time`; containing validity must be interval with valid_from equal to this value. At most one such edge per successor. Native projection closes the reviewed target there without mutating old bytes. |
| `supersedes` | `effect:{kind:"all_target_validity"}` or `{kind:"replace_from", replace_from:Time}`, plus `reason_code:Id`; no assertion of historical falsity. |
| `workspace_exception_to` | `workspace_scope_id:Id` equal to containing scope; runtime requires workspace→global exact target and validates reviewed validity. |

Reject duplicate edge IDs, multiple edges to the same exact target, and targets
equal to the containing current record/version. Same-record prior-version
correction/change remains valid. Shared core checks shapes and local invariants;
current authorized heads, stale targets, cross-profile identity, actual scopes,
cycles, incompatible effects and conflicts require native atomic admission.
Already admitted unchanged historical edges are preserved without re-admitting
an obsolete target head; new/modified edges require exact current targets.
A new global head holds the relevant workspace candidates pending review.
Neither recency, confidence nor source disappearance resolves contradiction.

## Proposal: `ProfileProposalV2`

Required fields: `schema_version:2`, `proposal_id:Id`, `profile_id:Id`,
`scope_id:Id`, `operation`, `target_record_id:Id|null`, `base_version_id:Id|null`,
`proposed_record:ProfileRecordV2|null`, `provenance:ProfileProvenanceV2|null`,
`state`, `created_at:Time`, `expires_at:Time`. State/operation retain V1 enums.
There is no additional top-level confidence field; an estimate for create/update
belongs to the nested claim and binds its exact digest. V1 proposals remain V1.

Pending requires provenance and expiry exactly created_at + 90 days. Create
requires target/base null and an active proposed record with null parent;
update requires target/base and an active proposed record with matching record
ID and parent. Nested profile/scope equal the proposal. Proposed record's
created_at/updated_at cannot exceed proposal created_at and its approval is
null. Archive/promote requires target/base, no proposed record; exact native
review/admission owns the resulting version, scope and action. A candidate
cannot carry a reviewed policy widening as pre-authorized consent.

Resolved states require proposed_record and provenance null. Create receipts
keep target/base null; update/archive/promote keep exact target/base. They retain
no claim, evidence or estimate. Expiry is ordered but not retroactively changed
to resolution time. Content-free means no content-bearing fields beyond those
explicit receipt identities/lifecycle values; it is not physical erasure.

## Retirement, disclosure and automatic-use boundary

A privacy-only successor preserves permitted assertion payload and temporal
meaning under the native policy owner, removes prohibited binding/assessment
metadata, sets approval/confidence/salience null and hold `privacy_review`, and
uses privacy-retirement provenance. It must not preserve forbidden relations or
source-basis IDs: native review either supplies a permitted validity representation
and recomputes the meaning digest, or withholds content under unknown validity.
This does not silently assert new dates. Other retained bindings remain only
when their complete metadata is independently permitted; no inherited support
is kept in this first sanitized form. A fresh user-reviewed successor is required
to clear the hold. No copied human receipt can name the new record version.

The manifest epoch alone does not retire history/outboxes/Undo/recovery/caches.
Shared retirement receipt/page schemas, native dependency journal, peer-local
revision, source marks and profile purge remain separately scoped under ADR-186.
This first contract publishes no receipt and cannot qualify an epoch transition.
Portable evidence waits for qualified receipt semantics and all managed owners;
local device-only metadata still needs applicable retirement safeguards.

For eventual automatic use, require current authority, supported profile-wide
semantics, active resolved heads/relations, no hold, known current validity,
exact qualified approval and restrictive destination/purpose admission. Inference
and imported claims additionally require current permitted exact evidence and
at least one bound supports assessment. Any current bound contradicts assessment
holds the claim for review, including a direct assertion. A missing
source does not prove an independently approved direct assertion false, but
policy retirement still takes precedence. Selection never opens a source to
create that observation. Hidden/denied data cannot affect ranking, model-visible
counts or omission signals. V1 eligibility remains V1 only inside a compatible
profile, with separately applicable policy migration and cutover rules.

## Dispatch, schemas and qualification

Proposed identifiers: `urn:tldw:profile-core:schema:personal-context:2`,
`urn:tldw:profile-core:json-schema:dialect:2`, and
`urn:tldw:profile-core:json-schema:vocabulary:semantic:2`.
Keyword stays `x-tldw-profile-semantics`, but the V2 vocabulary/dialect is required.
Publish separate `personal-context-v2.json` and `personal-context-v2-meta.json`
source/distribution copies in a later implementation. Leave V1 files unchanged.

The V2 semantic rule map freezes: `canonicalization=rfc8785-v1`,
`canonicalDateTime=utc-milliseconds-v1`, `iJsonMaxSafeInteger=9007199254740991`,
`canonicalPayloadMaxUtf8Bytes=16384`, `canonicalClaimMaxUtf8Bytes=16384`,
`canonicalRecordMaxUtf8Bytes=65536`, `canonicalProposalMaxUtf8Bytes=98304`,
`canonicalManifestMaxUtf8Bytes=16384`, `pendingProposalExpiryDays=90`,
`aggregateRules=profile-aggregates-v2`, `claimProjection=profile-claim-v2`,
`bindingProjection=owner-version-evidence-binding-v1`,
`attributionRules=claim-attribution-v2`, `temporalRelations=profile-relations-v2`,
`disclosureRules=profile-disclosure-ceiling-v2`,
`manifestRequirements=profile-context-requirements-v2`.
All entries required exact values; tags name the closed rules specified here,
not capabilities. Semantic validation includes byte ceilings, normalized dates,
digest equality, nested identity equality, sorted uniqueness, component strict
input rules, temporal/attribution constraints and content-free shapes.
Stock Draft 2020-12 structural validation does not establish those guarantees.

Keep public V1 models, `SERIALIZED_SCHEMA_VERSION=1`, exporter and existing
`CanonicalObject` union unchanged in the first data-only release. Expose an
explicit V2-only validator/exporter and types separately; invoking them is not
native acceptance. No repository, recovery, Sync/bootstrap, tools or context
imports the new types until its qualified native cutover unit. Package versions,
SQLite schema 8 and Sync V2 transport do not imply canonical V2 support.
No lossy V2→V1 serialization or fallback on invalid V2 input.

Migration preserves unmigrated V1 bytes and marks any explicit new V2 rewrite
with unknown legacy authority/support/approval/validity pending review. Old
active consumers block cutover or are disabled with grants retired. Offline
copies are outstanding, not erased. Native atomic heads/manifest admission,
retirement/disclosure enforcement and companion-server pins/conformance are
separate required gates before usability. No server code was inspected here.

## Conformance matrix: future checks, not passing runtime evidence

| Family | Positive control | Required negative control |
| --- | --- | --- |
| Wire/version | Distinct V2 aggregates; V1 scopes/payload defaults | Extra/missing keys, bool/string version, unknown tags, duplicate JSON keys; V1 bytes unchanged |
| Scalars/size | Boundary IDs, integers, ordered portable dates and exact size | Cc/Cf/surrogates, subclasses, NaN/Inf, overflow and UTF-8 multibyte byte ceilings |
| Canonical package | Fixed full JCS bytes + keyed integrity tag for manifest, active/archived/deleted record and pending/resolved proposal; source/installed copies identical | Model/structural/semantic disagreement, changed defaults, missing wheel fixtures, opaque downcast |
| Claim projection | Fixed identity/payload/validity/relation projection bytes and hash | Same text moved to another scope/record; changed dates/basis/edge with old support/approval |
| Binding composition | Published 18-field fixture/digest unchanged | Any single field substitution, subset digest, stale assessment, unsupported excerpt/Notes/snapshot |
| Attribution | Exact claim+binding supports; independent current-version approval | Hash→support upgrade, imported receipt→authority, future/inconsistent attribution time, empty-span support |
| Validity/relations | Standing/interval and distinct correction/change/supersession/exception shapes | Unknown validity auto-use, unordered bounds, duplicate edges, impossible transition and same-current-version target |
| Lifecycle/privacy | Empty tombstone/resolved proposal; unapproved sanitized successor | Evidence/provenance in tombstone, copied approval, retained denied basis IDs, privacy hold silently cleared |
| Proposal/disclosure | 90-day matching active candidate; deny default; explicit audience-purpose pairs | Wrong nested IDs/parent, embedded approval, receipt content, wildcard/cartesian audience widening |
| Native admission later | Authorized current target and manifest transaction, preserved historical edge | Stale head, foreign scope, cyclic/conflicting effects, partial commit; data-only pass cannot qualify it |
| Activation/retirement later | Every consumer acknowledged/retired; current receipt pages applied | Unknown semantics block entire profile incl. V1, offline old grant, restore/replay resurrection, unqualified owners/server |
| Source/disclosure later | Independently authorized current native span and qualified destination | Source locator as grant, reader-only publication lock, queued retry, child route, metadata egress or unqualified cache |

The first implementation must create fixed valid and structurally-valid-but-
semantically-invalid aggregate fixtures, with independent expected JCS UTF-8,
SHA-256 and `hmac-sha256-v1` results using a public synthetic 32-byte fixture
key. A generator cannot serve as its own oracle. Source/package validators and
installed-wheel fixture paths must agree. Native and server suites later consume
the same fixed bytes and outcomes. Today only documentation and the illustrative
projection below can be checked; no V2 aggregate or runtime conformance exists.

## Illustrative meaning digest

This is a projection example, not a V2 aggregate fixture or an approved claim.
IDs and payload are synthetic. JCS input/output (ASCII, one line):

```json
{"claim_basis":"direct_user_assertion","kind":"preference","payload":{"kind":"preference","polarity":"like","schema_version":1,"subject":"replies","value":"concise"},"profile_id":"p1","projection":"profile-claim-v2","record_id":"r1","relations":[],"scope_id":"s1","temporal_validity":{"basis":{"kind":"user_reviewed"},"kind":"standing"}}
```

This projection is exactly **337 UTF-8 bytes**. SHA-256:
`4d91768974a491f84e8ef67d9b8577975e8615c47065cd97a16260a7f4c28593`.
Changing only scope to `s2` yields
`de7ef8e9bffe5f5854390682357e8f8cbc08c4706dfa3e2c288910a08f7b1382`.
The byte string above has no newline in the hash input.
Replacing `scope_id` alone changes the digest; adding approval or confidence
outside this projection cannot change it. No source text or model is needed.

## Technical self-review and evidence

Self-review resolved scope-transplanted approval by including identity/scope in
the meaning projection; preserved the published binding instead of extending
its digest implicitly; avoided audience/purpose cross-products; removed mutable
source observations and local grants from canonical fields; made tombstones and
resolved proposals drop provenance; and prevented sanitized privacy successors
from reusing approval or leaking source-basis IDs. Core rules and native admission
rules are separated, including the remaining strict component parser requirement.

Native Python 3.12 verified the exact ASCII projection using independently
sorted compact stdlib JSON and RFC 8785 serialization, then SHA-256 and the
changed-scope negative control. The temporary documentation guard checks all
local links, task/ADR identities, fixed vectors, prior tracked/task/core/fixture
bytes and independent follow-up hashes. No V2 tests are claimed and no prior
36-case V1 receipt is relabelled V2 evidence.
The documentation guard passed 88 resolving local links, 19 unique memory-family
IDs, both fixed projection digests and the three identity-substitution controls;
prior runtime/schema/fixture/task bytes and independent follow-up hashes stayed
unchanged. Receipt: `/private/tmp/memory-v2-contract-draft-20260926.json`.
No full suite, real profile/keyring/provider/source/server or network was used.
The user approved the written contract on 2026-09-26. Implementation planning
may now proceed for inactive shared-core units; later native gates remain.
Acceptance does not qualify V2 aggregate conformance, migration or activation.
