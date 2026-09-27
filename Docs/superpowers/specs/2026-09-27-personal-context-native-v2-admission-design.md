# Native V2 profile compatibility and atomic admission design

Date: 2026-09-27
Task: [TASK-25907.21](../../../backlog/tasks/task-25907.21%20-%20Specify-native-V2-profile-compatibility-and-atomic-admission.md)
Status: Accepted design direction after explicit user approval, 2026-09-27. Design only; native V2 remains inactive.
Baseline: `c5ea63a6ae`; native Python 3.12.11. Python >=3.12 is the floor.
ADR required: yes.
ADR path: backlog/decisions/193-native-v2-profile-compatibility-and-admission.md.
Reason: define native service, encrypted control-state and compatibility boundaries under existing authority/privacy decisions.

## Purpose and limits

The inactive shared profile library can validate V2 bytes. The application still
uses V1 throughout its repository, service, recovery and Sync paths. This design
specifies the native barrier and transaction contract needed before V2 data can
become usable. It does not qualify sources, model destinations, forgetting owners
or the companion server. Structural validity, source truth, user approval, agent
read permission and model consent remain separate decisions.

Build on the existing one profile repository and one authorized service. Local
qualification/admission controls belong to that encrypted repository; they are
privacy/control records, never another store of user facts. Existing binding and
meaning components, V1 bytes, scopes and payloads remain unchanged. Do not widen
the published V2 wire contract or import native controls into shared core.

Governing decisions: [ADR-102](../../../backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md),
[ADR-185](../../../backlog/decisions/185-versioned-profile-evidence-and-temporal-claims.md),
[ADR-186](../../../backlog/decisions/186-dependency-aware-personal-context-forgetting.md),
[ADR-187](../../../backlog/decisions/187-personal-context-provider-disclosure-authority.md),
[ADR-191](../../../backlog/decisions/191-foreground-personal-context-source-inspection-authority.md)
and [ADR-192](../../../backlog/decisions/192-personal-context-v2-canonical-data-contract.md).
The [readiness audit](../../../backlog/docs/personal-context-v2-admission-readiness-audit.md)
predates the [completed aggregate implementation](../../../backlog/docs/personal-context-v2-aggregates.md):
its shared-data gaps are now closed, its native gaps remain.

## Alternatives and recommendation

1. **One profile-wide barrier plus exact native admission, staged behind closed
   release gates (recommended).** All ingress/read/write paths consult one
   current compatibility view; repository commits enforce its version fences.
   This is more work than dispatch alone, but prevents a V1 consumer from missing
   a V2 exception, privacy successor or restriction.
2. **Decode V2 and omit unknown rows.** Rejected: a permitted V1 global fact may
   have a V2 workspace exception. Quarantining the exception does not make the
   old global fact safe to use. Getters and Settings also bypass strict snapshots.
3. **Local storage/source inspection first.** Deferred: device-only is not
   metadata-retirement coverage, disclosure consent, or server conformance.
   No local-only exception to ADR-185/192 is introduced.

## Inspected seams and required enforcement

These are current source anchors, not claims that a future gate already exists.
The proposed consumer IDs are native registration names, not a new shared vocabulary.
Each family expands into explicit routes; registering one family must not hide
an unregistered route inside it.

| Proposed consumer family | Current seam | Required future boundary |
| --- | --- | --- |
| `profile-bootstrap` | [bootstrap](../../../tldw_chatbook/Personal_Context/bootstrap.py#L25), [schema inspection](../../../tldw_chatbook/Personal_Context/repository.py#L478) | Check storage support before loading keys/content, then authenticated manifest requirements before any ordinary record access. Unsupported binaries fail closed. |
| `profile-owner-read` | [manifest](../../../tldw_chatbook/Personal_Context/repository.py#L1416), [record](../../../tldw_chatbook/Personal_Context/repository.py#L2880), [Settings read](../../../tldw_chatbook/Personal_Context/service.py#L1499) | Guard individual getters/lists as well as snapshots; no V1 fallback on a blocked V2 profile. Owner-only status is separate from content inspection. |
| `profile-mutation` | [record/manifest commit](../../../tldw_chatbook/Personal_Context/repository.py#L2415), [proposal acceptance](../../../tldw_chatbook/Personal_Context/repository.py#L3351), [interview commit](../../../tldw_chatbook/Personal_Context/service.py#L989) | Enforce exact native admission and current compatibility in the owner transaction, including direct-write, Undo, promotion and interview batches. |
| `profile-context-tools` | [authorized view](../../../tldw_chatbook/Personal_Context/service.py#L1647), [strict snapshot](../../../tldw_chatbook/Personal_Context/repository.py#L3201), [tool recheck](../../../tldw_chatbook/Agents/profile_tool_provider.py#L149), [cache key](../../../tldw_chatbook/Personal_Context/context_service.py#L72) | Profile-wide guard before scope filtering, ranking, serialization and tool advertising; current stamps invalidate snapshots and prepared results. |
| `profile-export-recovery` | [export snapshot](../../../tldw_chatbook/Personal_Context/export_service.py#L113), [recovery decoder](../../../tldw_chatbook/Personal_Context/export_service.py#L274), [recovery loader](../../../tldw_chatbook/Personal_Context/export_service.py#L352) | Explicit mixed-version format and strict native import admission; grants and local qualification receipts are never exported/imported as authority. |
| `profile-sync-link` | [link planning](../../../tldw_chatbook/Personal_Context/link_service.py#L151), [reconciliation schema check](../../../tldw_chatbook/Personal_Context/reconciliation.py#L463), [inbound adapter](../../../tldw_chatbook/Sync_Interop/personal_context_adapter.py#L205), [dispatchable outbox](../../../tldw_chatbook/Personal_Context/repository.py#L4153) | Authenticated cohort/server negotiation, staging and replay checks before canonical apply or content push. Sync V2 transport is not canonical Profile V2 support. |
| `profile-derived-routes` | ADR-186/187 inventory: history, root/child/fallback rounds, tool arguments, interviews, summaries, capture/trace, run logs, embeddings, managed prompt caches, consolidation and repair | Each actual owner/route must qualify lineage, retirement and final publication or be disabled for governed data with authority retired. This family is an inventory obligation, not a claim of coverage. |
| `profile-source-inspection` | ADR-191 foreground Settings/app/Console owner contract | Qualified containing-record admission first; exact source owner/version/current permissions and final display fence remain separate prerequisites. No generic source resolver. |

No gate is added to deprecated Settings surfaces. Any later UI uses the canonical
F9 Settings route and the design-token rules. No UI change is part of this design.

## 1. Exact compatibility requirements

The V2 manifest requires these exact sorted object schema entries:

| Object | Required versions |
| --- | --- |
| manifest | 2 |
| proposal | 1, 2 |
| record | 1, 2 |
| scope | 1 |

It also requires these seven exact semantics: `approval-v2`,
`disclosure-ceiling-v2`, `evidence-binding-v1`, `metadata-retirement-v2`,
`profile-relations-v2`, `temporal-validity-v2`, `typed-claim-v2`.
The V2 dialect additionally requires its complete 16-entry semantic map.
The manifest tuple, dialect requirements and pinned byte/semantic fixture bundle
are distinct checks. A package version, installed decoder or self-reported
capability list is insufficient. Unknown/missing/reordered/duplicate requirements
fail closed; there is no best-effort subset or lossy V2-to-V1 conversion.

Native registration is compiled at the application composition boundary. A route
has a stable consumer ID, exact build/contract/fixture identity, owner ID and
supported operation set. Only a trusted native owner can attest a qualification
result. Agents, canonical data, plugins, configuration and imported receipts
cannot register a consumer or declare it qualified. New/unregistered profile
routes deny access. A route that only rejects unsupported operations must show
that rejection at all its entry points; decoding is not semantic interpretation.

Maintain an encrypted peer-local qualification record containing: profile ID,
manifest version, purge generation, shared evidence-retirement epoch, local
retirement revision, registry revision and digest, native build/contract/fixture
identity, local grant-policy revision, peer-cohort revision and owner receipts.
Each owner receipt binds consumer ID, operation set, qualification revision and
its current enabled/disabled state. Receipts contain no claim text, quotes,
source locators, paths or labels and are not proof of physical deletion.
Their authentication prevents tampering; it does not turn an assertion into
qualification. Runtime evidence must support each enabled owner.

A consumer is eligible only with a current matching receipt. A disabled consumer
is removed from the active set only after its native owner confirms jobs drained,
prepared payloads invalidated, grants retired and replay/reconnect fenced for the
same revisions. A toggle, crashed process, timeout or offline peer is not a
retirement receipt. If the application cannot enumerate a route or establish
its closure, the profile remains blocked. Local checks cannot retire a server
or an offline client's authority.

For linked profiles, the home peer must supply an authenticated current cohort
and equivalent server-side grant fences before cutover. The wire/cohort protocol
is a separate server-qualified unit; absence blocks cutover. For unlinked
profiles, the absence of a linked cohort must be verified from native state;
server byte/semantic conformance required by ADR-185/192 remains a release gate.
No canned server receipt or empty list stands in for that evidence.

## 2. Profile states and native interfaces

Proposed native modules are `Personal_Context/native_compatibility.py` and
`Personal_Context/native_admission.py`, with owner integration in existing
repository/service/bootstrap. No modules are created by this task.

`ProfileCompatibilityView` is an immutable process-local result from an
owner-authenticated repository snapshot and the compiled registry. It binds all
qualification fields above and yields one of these states:

| State | Permitted behavior |
| --- | --- |
| `legacy_v1` | Existing V1 service behavior under current authority; no invented V2 review/support/disclosure. Installing V2 helpers changes nothing. |
| `v2_blocked` | No ordinary profile content reads/writes, context, tools, replay or model use, including V1 records inside the profile. Only separately authorized native maintenance/control operations can run. |
| `v2_storage_qualified` | All compatibility/storage/privacy owners qualified; authorized native canonical transactions may run. Each record and operation still needs admission and current authority. No model or source access implied. |

This is not a replacement for LOCKED/DISABLED/REMOVED or Sync attention. Service
status projects a generic unavailable/attention reason; current owner maintenance
may inspect a content-free reason such as capability missing or retirement
pending. No global unsupported-record flag, hidden ID, denied-count or evidence
metadata enters model output or ordinary Next Send. A blocked profile must not
look ABSENT and trigger automatic Start Fresh. Existing content is preserved;
blocking is not deletion or cleanup acknowledgement.

Repository guards apply before decrypting record/proposal bodies and inside
write transactions, independent of caller discipline. The authenticated manifest
and encrypted qualification controls can be read through a narrowly owned
maintenance path to compute the barrier. That path cannot read arbitrary record
content or be exposed as an agent/export fallback. Unsupported manifest semantics
permit generic status/opaque retention only, never content inspection. Any future
opaque staging owner itself needs bounded encryption/retention/retirement coverage.

An upgraded store needs a new native storage version that unsupported old binaries
reject before normal access. The current native storage version is 8 and current
AAD marker is 1; neither is canonical Profile V2. The later storage unit must
allocate its migration version and explicitly version any changed AAD/envelope
format. Never rewrite or reinterpret existing authenticated rows in place. No
physical migration/DDL/envelope version is implemented or approved here.

`ProfileAdmissionStamp` is an immutable, non-exportable native assertion created
by `PersonalContextService` after review/authorization. It binds operation ID,
actor/root/run scope, exact incoming version and canonical digest, expected
manifest/record heads, related target heads/scopes, compatibility and authority
revisions, purge generation, shared epoch and local retirement revision.
Foreground approval additionally binds the exact reviewed claim/record version,
validity and relations. Source-dependent decisions bind owner-issued current
source/version/permission stamps. An agent cannot supply or reconstruct a stamp;
canonical attribution/approval fields alone never mint it.

A stamp is not a durable authorization token. Repository commit requires the
same live native authority and publication coordinator, rechecking every bound
revision in the final guarded transaction. Restarts, retries, Sync and recovery
must obtain fresh native admission. Durable receipts record completed outcomes
without granting future access. Native controls are encrypted and excluded from
Sync/recovery/ordinary logging; content-free shared retirement controls remain
owned by the separately qualified ADR-186 protocol.

## 3. Atomic admission and relation resolution

Validate incoming raw bytes with duplicate-aware, bounded explicit V1 or V2
validation, then compare authenticated envelope identity to the decoded object.
Use the V2 helper for V2 canonical bytes/digests; never route V2 through root V1
dispatch or accept Pydantic override/coercion settings from an ingress caller.
Unknown canonical objects remain unavailable under owner-controlled opaque
retention; they are never partially decoded into usable V1 records.

Admission has three independent outcomes: reject an invalid/unauthorized
transaction; retain an authorized but held/conflicted object unavailable for
automatic use; or commit an authorized qualified object. Pending proposals are
always unavailable to automatic context. Imported/authenticated claims do not
inherit native approval or source access. Direct user assertions may remain
assertions without documentary evidence; missing support is not proof they are
false. Inference/import automatic-use eligibility requires qualified exact support,
known reviewed validity and resolved relations as required by ADR-185. Confidence
and salience never repair missing authority, review or evidence.

The existing `BEGIN IMMEDIATE` transaction remains the profile atomic boundary.
The service must first prepare permitted owner information outside SQLite, then
enter the qualified ADR-186 publication coordinator, then the repository write
transaction. Do not resolve sources, await a provider/network, run a model or
invoke arbitrary callbacks inside that transaction. Cross-owner permissions and
retirement changes must participate in the same coordinator/fence protocol;
an optimistic read before commit alone is insufficient. An owner without that
protocol cannot qualify its source-dependent operation.

Inside the guarded transaction, read the current manifest, qualification controls,
local policies/mappings, all expected record/proposal heads and relation targets.
For a new edge the named target version must be the exact current authorized
head, in the same profile and permitted canonical scope; its lifecycle and
validity must permit the effect. Corrections, changes and supersession are
same-scope; workspace exceptions originate in the named mapped workspace and
target the authorized global claim. Do not search across sources/scopes to make
an invalid edge resolve. Validate the reachable exact relation graph for cycles
using nodes identified
by `(record_id, version_id)`, not a graph that substitutes current heads for
historical targets. A reviewed correction/change may target the current prior
version of its own record; that is not a cycle back to the new version. If
required ancestry is unavailable/unsupported, reject new admission generically.

A successfully admitted historical edge stays bound to its original target
version after later revisions; do not reinterpret it as targeting today's head.
Immutable native admission receipts preserve the fact of prior admission. Content
retirement removes prohibited material while retaining only permitted content-free
lifecycle controls; receipt history cannot resurrect a retired target or confer
fresh authority. Receipts themselves participate in retirement: retaining history
must not retain withdrawn claim/binding digests or source metadata. An imported
old edge has no local historical receipt and needs qualified imported lineage
handling, otherwise it stays held. Copying an unchanged admitted edge across an ordinary
revision can preserve its historical admission, while the new claim/record
version still needs its own required foreground approval and current authority;
old approval cannot authorize changed wording/validity/relations.

Commit the immutable candidate, head CAS, manifest successor, admission outcome,
proposal resolution and any required encrypted outbox/Undo atomically in the
same profile transaction. When requirements, registry, cohort and owner
qualification revisions are unchanged, that transaction also rebinds the local
qualification record to the successor manifest; it does not require every owner
to rerun fixture qualification for each content edit. Requirement/authority/epoch
changes cannot use this rebind and require the applicable fresh owner checks.
Prepared stamps never auto-refresh across a manifest change. For Sync no outbound
echo is fabricated. A batch's participating objects either all commit or none do. Outbox/Undo eligibility and
metadata coverage are prerequisites; a stamp is not permission to retain an
unsafe before-image. A crash before SQLite commit leaves no admitted head; a
committed operation ID yields its exact native receipt on recovery rather than
creating a second candidate. Recover ambiguous cross-owner publishing tickets
through their native owner before retry; lease expiry is not evidence of rollback.

If a global target changes while a workspace-exception candidate waits, its review
is stale: require fresh review of the new head. Do not relink automatically. If
contradictory candidates race, exact CAS and existing durable Sync-conflict review
prevent newest-wins. An unresolved affected claim is omitted or uses only a still
permitted mutually acknowledged version under ADR-102; neither candidate gets a
fresh automatic-use receipt by timestamp. No parallel conflict/fact authority.

Failures expose generic unavailable/conflict/changed results to agents. Owner
review may show only currently permitted specifics. No target existence oracle.

## 4. Startup, cutover, recovery and replay

The eventual foreground cutover prepares one exact V1 snapshot, consumer/grant
inventory and review. Before fencing, cancellation can discard the plan. Fencing
stops admission and drains registered publishers. Persist durable content-free
intent before changes spanning native owners. If an owner retirement is ambiguous,
remain blocked/needs attention; do not roll grants back into service.

Only after current owner acknowledgements, privacy controls, server conformance
and other release gates qualify may one transaction publish the new native
storage marker, V2 manifest and matching encrypted local qualification state.
Existing immutable V1 bodies/IDs remain unchanged. This is forward-only; an old
binary is not a rollback strategy. A compatible controlled recovery or forward
fix is required. No cross-database atomicity promise is made.

Within a V2 profile, V1 facts lack V2 approval/support/temporal/disclosure: deny
model disclosure and automatic semantic use pending explicit native review.
An optional V2 legacy successor uses `legacy_unknown`, unknown validity and a
legacy review hold with no bindings/support/approval/confidence/salience or
relations. Storage timestamps and old provenance are not evidence of truth dates
or user approval. Fresh foreground authorship creates a new exact reviewed
version with independently qualified inputs; migration never forges a receipt.

Startup loads current destruction/purge/retirement and compatibility controls
before admitting restored/outbox/staged data or advertising tools. Restore
validates a separately versioned mixed-envelope format, preserves canonical bytes
and histories only where their owner policy allows, and applies current fences
before any imported content becomes usable. Local grants, consumer receipts and
source capabilities cannot be restored as authority. A new destination still
requires fresh native enrollment. If current retirement knowledge cannot be
established, recovery stays blocked; an old backup cannot lower epochs/revisions.

The shared evidence-retirement epoch, peer-local retirement revision and whole
profile purge generation are separate monotonic controls. A local-only retirement
need not change a synchronized epoch, but still invalidates local admission and
prepared outputs. Any old-generation/retired replay is rejected by the relevant
native owner; equal counters do not establish permission or completion.

An offline peer remains outstanding until it acknowledges the exact required
capabilities/retirement controls, or the home owner verifies its active grant is
retired and reconnect denied. Reconnection and every content push/pull require
fresh cohort/contract checks. Retiring a grant does not claim stopped offline
decryption or physical erasure. Existing `server_trusted_v1` custody is unchanged.

## 5. Bounded qualification sequence and release gates

These are ordered deliverables, not references to uncreated task IDs. Each later
unit gets its own acceptance criteria, applicable ADR review and native plan.

| Unit | Independently reviewable outcome | Activation boundary |
| --- | --- | --- |
| A. Native barrier and explicit codec | New source-backed consumer registry, mixed-version validation and repository/service read/write guards; real temporary encrypted SQLite verifies V1 success and every V2 path denies without current qualification. App bootstrap/status and tools consume the guard, so it is not an unconsumed utility. | Production V2 creation/migration/import/Sync/source/model routes stay closed. Synthetic fixture admission cannot qualify a production owner. |
| B. Retirement/publication owners | Qualified ADR-186 control journal, suppression, owner receipts, managed lineage coverage and restart/final-publication races for the actual enabled scope. | Unsupported logs/capture/caches or unknown coverage prevent that scope from being offered. No universal deletion claim. |
| C. Deny disclosure and clean local enrollment | ADR-187 filtering, derivative restrictions, destination/purpose grants and final adapter gates cover all enabled routes and clean owned process/slot state. | Remote/replay/interview/summary/tool-egress routes remain disabled until each qualifies. No URL-label enrollment. |
| D. Atomic native admission and foreground review | Real owner transactions bind exact heads, support/source authority, actor intent, scopes/relations and all current revisions; app/service callers use them. | V2 data still cannot be activated until all release prerequisites, including server conformance, are satisfied. |
| E. Companion contract/cohort qualification | Real client/server pinned fixture conformance, current cohort/grant retirement, bootstrap/Sync/recovery controls and honest offline acknowledgement. | No mocked server result qualifies rollout; no server work is done in this checkout. |
| F. Reviewed cutover and source inspection | Eligible profile cutover plus ADR-191 exact foreground containing-record/source/display admission; no automatic source opening. | Only the fully qualified declared scope becomes usable; future consolidation/repair remain separate. |

A and D share the same barrier/admission contract. A may land first with every
V2 permit path closed; D must not bypass B/C/E by implementing a convenient
local-only accept path. A new release decision is needed to change these gates.
No periodic worker, new dependency, external provider or fourth memory store is
needed for the barrier itself. Physical schemas and cross-owner protocols are
not guessed in this document; their specified units must qualify them before
release. This document is not an executable implementation plan.

## 6. Future qualification matrix

All cases below are **unrun V2 native acceptance cases**, not passed runtime
evidence. Positive controls are mandatory so deny-all behavior cannot qualify
use. Use native Python >=3.12, real temporary SQLite and synthetic credentials;
no live profile, keyring, source, provider or server in the first native unit.

| Area | Required cases |
| --- | --- |
| Compatibility | V1 profile still works after library installation; fully qualified synthetic V2 control; each missing semantic/consumer blocks the whole profile including a V1 global claim; forged/imported/old-build receipts deny; unsupported storage binary refuses. |
| Coverage | Snapshot, individual get/list, Settings, direct-write, proposal/interview commit, Undo, export/recovery, Sync bootstrap/apply/dispatch and tool advertising all use the guard; new unregistered route denies. |
| Revisions | Changed registry/build, local policy/mapping/grant, cohort, manifest, purge generation, shared epoch and local retirement revision each invalidate prepared state; loss of owner acknowledgement blocks immediately. |
| Transactions | Successful exact record+manifest+proposal/outbox outcome; failure after each mutation rolls all participating profile rows back; actor string/HMAC/approval JSON alone cannot admit; duplicate operation recovers the exact outcome without a new head. |
| Relations | Current-head positive controls for each of four edge kinds; stale target, missing target, wrong profile/scope, unmapped workspace, cycle and unknown required ancestry reject; later admitted history keeps the original target without granting new use. |
| Conflicts | Two contradictory concurrent candidates cannot get newest-wins eligibility; target change during workspace review requires fresh review; last-ack fallback only when still permitted by current controls. |
| Legacy | Mixed V1/V2 wire validation preserves V1 bytes; migrated legacy is held/unknown/deny; dates, attribution, grants and review are never invented; fresh independently reviewed authorship can qualify when prerequisites do. |
| Recovery/replay | Fence before startup/restore/outbox/reconnect; old backup cannot lower controls; imported local receipts/grants inert; offline peer stays outstanding; verified grant retirement denies reconnect without claiming erasure. |
| Publication/privacy | Revocation before guarded commit, before final display and before adapter entry cancels/fences; crashed publishing ticket is recovered, never assumed cancelled; opaque staging/before-images/outboxes need registered retirement; source/model restrictions survive derivatives. |
| Non-disclosure | Blocked/hidden/wrong-scope/unsupported records yield generic outcomes; no denied counts, global quarantine flag, raw metadata or owner receipts reach model/ordinary preview/logs; positive currently permitted owner status control. |

## Self-review and evidence boundary

The source review corrected four plausible shortcuts: a snapshot-only gate misses
individual getter and mutation paths; a compiled support declaration is not a
current qualification receipt; and a pre-transaction source-permission read is
not a race-safe cross-owner authorization fence. Self-review also caught the
need to rebind the local qualification record in the manifest transaction, so
normal content edits do not accidentally invalidate the whole profile or trigger
blanket owner requalification. The design now requires each at its owning
boundary. It also separates profile compatibility from per-record
admission and storage qualification from model/source permission.

Document validation checks resolving links/line anchors, exact shared requirements,
unique task/ADR allocation, prior runtime/core/task bytes and the independent
TASK-25907.10 task/roadmap suffix. Those checks establish document consistency and
preservation only. Previous 728 shared-library cases establish inactive data
validation; they do not qualify any case in this native matrix. No V2 native
runtime tests or real server conformance are claimed.

Native document guard passed on Python 3.12.11: 117 resolving local links,
19 verified source line anchors, 22 unique task-family IDs, all seven manifest
semantics, the four exact object-schema entries and the 16 dialect rules.
Application/shared-library/tests and prior task bytes are unchanged relative to
`c5ea63a6ae`; the foreign task and roadmap suffix retain their exact recorded
SHA-256 values. Receipt: `/private/tmp/native-v2-admission-design-20260927.json`.
The native command was `.venv/bin/python -I /private/tmp/check_native_v2_admission_design.py`.
No full suite, profile/keyring/provider/source/server or network access.

The user explicitly approved the written design on 2026-09-27. The next deliverable
is a native implementation plan for Unit A with all production V2 permit paths closed.
