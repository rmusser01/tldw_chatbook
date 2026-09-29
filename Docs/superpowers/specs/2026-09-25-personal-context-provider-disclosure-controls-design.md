# Personal Context provider disclosure controls design

Date: 2026-09-25
Status: Accepted design direction, 2026-09-25; design only, no runtime rollout
Task: TASK-25907.7
ADR required: yes
ADR path: backlog/decisions/203-personal-context-provider-disclosure-authority.md
Reason: model egress, destination-bound runtime grants, canonical compatibility
and the governing device-only promise change privacy/provider contracts.

Decision: [Accepted design ADR-203](../../../backlog/decisions/203-personal-context-provider-disclosure-authority.md)
Governance: [ADR-102](../../../backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md), [ADR-182](../../../backlog/decisions/182-personal-context-memory-evolution.md), [ADR-201](../../../backlog/decisions/201-versioned-profile-evidence-and-temporal-claims.md), [ADR-202](../../../backlog/decisions/202-dependency-aware-personal-context-forgetting.md), [ADR-147](../../../backlog/decisions/147-agent-provider-routing.md), [ADR-080](../../../backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md), [ADR-119](../../../backlog/decisions/119-llamacpp-prompt-cache-snapshot-ownership.md)
Tracker: [Memory roadmap](../../../backlog/docs/personal-context-memory-roadmap.md)
Review: [Technical review](../reviews/2026-09-25-personal-context-provider-disclosure-controls-review.md)

## Purpose and approval boundary

Define independent synchronization, agent visibility and model-disclosure
controls. A record being readable by an agent or syncable is not consent to
send it to a model endpoint. Authorize native destinations/purposes explicitly,
carry restrictions into registered derivatives, and recheck before adapter entry.
This task writes contracts and future synthetic acceptance cases only. It makes
no provider calls, grants, migrations, model/schema edits, UI or runtime changes.
The Muse diagram supplies comparative ideas, not disclosure instructions.
The user explicitly approved the reviewed written contract. ADR-203 accepts
design direction only; no model grant, policy migration or runtime rollout is
approved by closing this design task.

## Inspected behavior and egress owners

| Owner/seam | Current code evidence | Future contract |
| --- | --- | --- |
| Shared core `ProfileControls` | `packages/tldw_profile_core/src/tldw_profile_core/models.py` has `sync_mode` and `agent_visibility`; V1 schema/version is fixed | Add a separately versioned restrictive disclosure policy; no permissive interpretation of absent fields |
| Canonical context view | `Personal_Context/service.py::authorized_context_view` checks profile/scope runtime authority and active workspace mapping; no resolved model audience input | Native audience projection after current authority, before selection/serialization |
| Ordinary Console preparation | `context_service.py` selects records using active/visible/expiry/conflict rules; request provider/model are used for token estimation. Controller appends the serialized block into a system row | Destination/purpose admission before profile block construction and final gateway entry; stale payloads cannot merely change labels |
| Profile tools | `Agents/profile_tool_provider.py` get/search/update return canonical record JSON under run authority, without a destination-bound disclosure grant | Return only records and metadata permitted for the requesting actor's current model route/purpose; result derivatives keep policy |
| Children, steering, reports and provider routing | Agent service/bridge owns root/child snapshots, tool results, reports and ADR-147 route choices | Every child and subsequent round needs its own resolved-destination projection within the root grant ceiling; parent permission does not transfer |
| General agent tool invocation and managed publication | `Agents/agent_service.py::invoke_tool` passes model arguments to `ToolCatalogRegistry.invoke_by_name`; web/MCP/skills/shell/file tools have independent execution/publication owners | Model-generated arguments retain governed input restrictions. Ordinary tool permission does not authorize fact disclosure. External/unqualified invocation and publication stay disabled in governed runs until separately qualified. |
| Interviews | Coordinator pins provider/model and currently filters existing records to visible syncable records. Configured adapter serializes records and prior turns | Separate interview-purpose consent for records and draft answers; syncable is not provider consent; recheck each question/resume |
| Summaries and compaction | `Chat/console_context_compaction.py` uses gateway auxiliary/native calls and input lineage, including prior summaries | Apply policy to full prepared input, admit the summary route separately and retain input restrictions in the output |
| Embeddings/RAG | Existing embedding wrapper supports configurable backend selection; the completed Personal Context matcher is local lexical search | No new profile embedding/index job is implied. Any future model embedding request requires an embedding-purpose audience grant and derived-index ownership |
| Captures, audit/logs and caches | ADR-202 inventories Safe/Full first-system copies, semantic artifacts, AgentRunsDB, run logs and managed cache binaries | Preserving a local copy does not authorize later model replay/search/export; policy and lineage must accompany registered managed copies |
| Export/recovery/Sync | Profile export has separate explicit plaintext/recovery publication; Console exchange export has its own Trace profiles. Sync is the ADR-102 trusted-server boundary | No model grant from export/Sync/import. Local export approval is its own exact action; model uploads require current audience consent |
| Quarantine status | Service derives `unsupported_records_present` from unscoped record quarantine; context serializer may emit it even with no selected record | Remove that global existence signal from model block and normal Next Send diagnostics; owner-only maintenance remains separate |

The inventory is code inspection, not proof of shipped destination enforcement.
No real profile or server was inspected. Unknown/unregistered derivatives remain
an explicit migration/qualification limit.

## Alternatives and chosen direction

1. **Restrictive canonical policy plus destination-bound native grants — chosen.**
   Portable records carry a ceiling, while each runtime authorizes concrete
   custody and purpose. Existing profile/source permissions remain prerequisites.
2. **Allow by provider display name or URL label.** Rejected: aliases, changed
   endpoints/accounts, LAN relays, fallback routes and child overrides can change
   actual custody without changing that name.
3. **Treat agent visibility or syncability as model consent.** Rejected: these
   answer different questions. Blanket consent would perpetuate the current
   device-only mismatch and create new disclosure on migration.

## Three controls and the ADR-102 amendment

| Control | Question answered | Does not grant |
| --- | --- | --- |
| Synchronization | May this canonical record cross the profile Sync boundary? | Model execution, export/upload or child permissions |
| Agent visibility/authority | May an authorized agent read/propose/update within this scope? | A model audience or source access outside current floors |
| Model disclosure | Which admitted model custody/purpose may receive this version and its governed derivatives? | Sync, agent read access, broader sources or export publication |

This proposal **explicitly amends**, rather than silently reinterprets, ADR-102's
literal “device_only records never leave Chatbook” wording. The future rule is:
`device_only` prevents automatic synchronization and any automatic disclosure
outside qualified Chatbook-owned on-device execution. A local model process
qualifies only under native owned execution and no external forwarding. A
loopback URL, private network address or local-looking provider name is not
proof. LAN servers, SSH tunnels, remote relays and unknown processes do not
qualify as on-device custody.

Explicit owner-controlled plaintext/recovery export remains a distinct existing
operation with exact retention/recipient review; it does not grant model upload.
The amendment clarifies that exception and local execution custody. It grants
nothing today: ADR-102 and current permissions remain until a separately
qualified rollout. The current implementation does not yet provide the future
model-egress guarantee, and UI/documentation must say so at rollout.

`device_only` is a hard automatic egress ceiling: even a remote audience grant
cannot authorize it. To allow remote model use, the user must separately review
a record-control change as well as the concrete model grant; changing Sync mode
alone still grants no model permission. `user_only` remains absent from all
agent/model paths, including tools and derivatives. A fixed local questionnaire
with no model call needs no model grant but retains normal draft/source privacy.

## Proposed canonical policy and local grants

Extend the not-yet-shipped ADR-201 V2 design with a strict `model_disclosure`
policy: `deny`, `on_device_only`, or `reviewed_destinations`. These are proposed
versioned vocabulary values, not fields added to V1. `reviewed_destinations`
names at most 16 bounded opaque audience handles and explicit purposes from
`conversation`, `interview`, `summary` and `embedding`. An audience handle is a
restrictive user-selected category requiring native enrollment; it contains no
credentials, endpoint paths or runtime permission. No wildcard/all-providers
policy is offered. `on_device_only` uses a reserved local audience namespace
with the same explicit destination/purpose enrollment; it does not grant every
local-looking model. An empty, unknown or unsupported policy means deny.
If ADR-201 ships first, this addition needs a subsequent version; never mutate
previously shipped canonical bytes/schema semantics in place.

Each peer keeps separate encrypted, user-reviewed native grant bindings for
at most 16 destinations per selected audience. A grant binds profile/scope,
canonical policy revision, purpose, qualified destination identity, actor/run
ceiling, consent revision and optional expiry. Default is no binding. Grants
never synchronize, enter profile exports/model context or authorize a source.
Peer enrollment is independent; importing a record cannot enroll an audience.
Any expiry, revocation, record policy change, scope/actor narrowing, purge or
retirement invalidates the old grant at the next admission boundary.
Only foreground user review can widen a canonical disclosure ceiling or enroll
a native audience. Model proposals/direct-update tools cannot turn read/write
authority into consent. New unreviewed records default to deny; generated
records/derivatives inherit the input intersection until a distinct owned
review. Scope promotion preserves restrictions and obtains consent for the new
scope; a global binding is not automatically applied to a promoted private copy.

Destination identity binds adapter/provider identity, resolved model, effective
endpoint origin and required route/base-path identity, credential-free account
or tenant binding when available, endpoint provenance, custody classification
and native configuration revision. The UI may use a safe label; enforcement
uses the qualified identity, not the label. Secrets and signed URL query values
are never identifiers or diagnostic fields. Credential replacement that changes
account/principal or leaves its identity uncertain requires renewed enrollment;
no auto-rebind to a similarly named provider. Frozen credential-free snapshots
must retain only permitted endpoint metadata under existing provenance rules.

For on-device qualification, native execution must identify the owned model
process/socket and establish the adapter's no-forwarding contract. Existing
`ConsoleEgressClass.ON_DEVICE` is a classification hint, not a new proof. Unknown
ownership/capability is denied. For network enrollment, transport must pin its
reviewed effective destination and account/route semantics, prevent unreviewed
cross-origin redirects/fallback/proxy changes, and expose uncertain custody.
A direct provider service's subsequent retention is outside local enforcement;
consent names that service custody, not a guarantee of its internal behavior.
An aggregator/relay with unresolved selectable downstream destinations is not
eligible for the first future release. No model/prompt can enroll or widen grants.

A disclosure policy is a ceiling, not a portable capability or data truth. The
shared core validates shape/semantics; native runtimes resolve destinations,
consent, source authority, transport and OS/process ownership. Never put API
keys, authentication, network clients or grant consumption in the shared core.

## Admission, mixed sources and cached state

Evaluate the complete governed input under current profile/source authority,
active scope, record visibility, lifecycle, model policy, purpose, native grant
and purge/retirement controls. Filter by audience **before** ranking, workspace
exceptions, token budgeting, explanatory candidates and serialization. A hidden
or audience-denied record cannot suppress an otherwise admitted workspace/global
candidate, consume budget, or alter a model-visible omission reason.

Every generated registered artifact inherits the intersection of all input
model restrictions and the existing stricter source-owner controls. Never union
permissions or widen because an LLM paraphrased a fact. A mixed artifact denied
at a destination is omitted as a whole; no substring redaction or generative
sanitization during dispatch. User-authored originals remain independent owners;
a selected governed derivative does not confer consent to read its originals.
Admitted record text and evidence/provenance fields each remain subject to
ADR-201 current source display/transport authority. A record-level model grant
is no bypass for restricted quotations/locators. Required inline metadata that
cannot be disclosed requires the reviewed sanitized successor contract; do not
fabricate an evidence-free canonical object during provider serialization.
An explicitly reviewed replacement requires new inputs/lineage/approval under
ADR-201/202, not a transfer of an old audience receipt.

Stored transcript/tool rows, child messages, summaries, captures/log search
results and prepared snapshots retain governed dependency/policy identity.
Legacy copies with unknown lineage cannot be treated as unrestricted at a new
route. On route migration, omit whole managed units or stop preparation if
required units cannot be safely separated. No claim of retroactively identifying
unregistered copied facts or censoring independently authored source transcripts
is made.
ADR-119 model-state binaries and live/working slots are governed inputs too.
A new conversation or empty visible history does not establish a clean model
state after Save/Restore or slot reuse. The native launch/cache owner must bind
complete input lineage and inherited policy to snapshots and slots, recheck
current grants/retirement controls before Save/Restore and every model reuse,
and prevent cross-audience/actor reuse. Unknown preloaded state cannot qualify
by searching binary bytes or trusting a model response. Until that owner
qualifies, managed Save/Restore/reuse for governed input is disabled; the first
release needs an explicitly reviewed isolated clean owned process/slot or must
refuse. Do not implicitly reset/delete an independently owned cache or process. A conversation containing unmanaged historical profile-derived text
cannot qualify the complete-history first-release guarantee; expose a generic
coverage failure only under the current conversation owner's authority.

An immutable request-owned admission token binds the qualified destination and
purpose, actor/root grant ceiling, exact selected artifact versions, current
policy/grant revisions and purge/retirement controls. Native admission and final
adapter entry share the qualified ADR-202 publication gate; no unlocked final
check/callback or model decision substitutes. Tokens are opaque, content-free,
short-lived, single-attempt and cannot be transferred to another child/model.
Queued requests are still managed; final entry consumes current authority and
records a content-free outcome before provider execution without holding a
network await under the gate. Each app-owned retry, fallback, routed round and
resume needs fresh evaluation, even after an earlier attempt entered an adapter.

If destination or authority changes after review, invalidate the prepared block,
tool projection and Next Send explanation. Rebuild under the resolved route and
ask for a new grant only if the owner chooses to disclose there. Ordinary chat
may continue with no Personal Context when denied material is optional; an
explicit profile-dependent operation fails with a generic unavailable result.
Interviews, summaries or required mixed history stop unless an exact valid
filtered-input plan exists. Unknown destinations never receive governed content.
Already entered network work is potentially begun/uncertain disclosure, not
confirmed delivery or recallable solely by local revocation. Request cancellation
where supported; fence future callbacks/saves and retain local cleanup ownership.

### Model-generated tool arguments and publication

A qualified local model can still place governed facts in a web URL, MCP call,
child task, skill/shell command or file destined for independent publication.
Every generated call conservatively inherits all governed model-input
restrictions unless complete native dependency lineage proves a narrower set;
no text classifier or the model itself can certify that arguments are harmless.
The existing tool catalog, permission store, hook/approval and root authority
remain necessary floors, but permission to invoke a tool is not disclosure consent.

The first governed release disables external or unqualified web/MCP/skill/shell
invocations and independent file/export/sync publication. Only qualified native
local tools whose execution, storage and output preserve those restrictions may
run. Root/child requests and reports still need their exact model admission;
spawning a tool/child is not an escape from the root's policy ceiling. A later
tool-egress/publication release requires a distinct reviewed purpose/custody
contract and owner authorization; a conversation-model grant never authorizes
an external tool audience. No new tool consent schema or tool grant is implied
by this design's model-purpose vocabulary. Unknown tool custody fails closed.

## Noninterference and user-facing explanation

Remove `unsupported_records_present` from the provider/model serialization and
normal Next Send projection, including the empty-record case. Keep quarantine
maintenance under an explicit user-owned Settings action; there is no generic
“hidden records exist” flag/count in model text, tool failures, retries or child
reports. A deny result is the same shape for absent, hidden, wrong-scope,
unsupported or audience-denied records. Unknown canonical policy is not read
for a nicer explanation. Integrity controls may internally invalidate a token
without revealing which private item changed. Such an internal revision triggers
a fresh permitted projection; do not serialize the global authority hash, global
quarantine size or a hidden-item change reason. The content/diagnostic contract
does not claim constant-time execution or hide all filesystem/network patterns.

Next Send shows the effective provider/model and safe destination/custody label,
selected allowed records, purpose and grants actually applicable to that exact
prepared request. Any omission explanation is derived only from candidates the
user and requesting actor are authorized to inspect for that surface. Hidden
records never appear as names/counts/reasons. A pre-request consent preview is an
explicit user-only surface with separate privacy-control authority; it may show
an owned record only when current record/source permissions allow. It must not
be mirrored in provider payloads, capture metadata, logs, telemetry or exports.
No grant is created by opening or dismissing the preview.

Explain Sync, agent visibility and model audience separately in canonical F9
Settings. Legacy `device_only` is described as current Sync behavior until the
qualified egress release lands; never imply the new guarantee is already shipped.
No UI implementation is part of this task. Later UI work must read ADR-150 and
compose the existing design tokens; no parallel legacy Settings surface.

## Migration and compatibility gates

- Keep V1 bytes, fixtures, schemas and behavior unchanged in this design task.
  A future rollout gives all legacy/unreviewed records a deny model policy;
  existing visibility, Syncability, saved credentials or previous sends create
  no grant. User review may enable qualified local use or exact remote audiences.
- Use ADR-201's profile-wide version/capability activation, not silent V1
  downcasting. Every active context/tool/Sync/restore consumer must understand
  restrictive policy before activation. Unsupported consumers block cutover or
  have their profile grants explicitly retired. Retain deny for unknown policy.
- Canonical policy can travel only through existing authorized profile Sync;
  native model grants never travel. `server_trusted_v1` is unchanged: Sync is a
  separate trusted-server disclosure, not E2EE or a model execution consent.
  The server may store policy but must not use it as a grant for its own LLM jobs.
- Policy edits and grant revocations require versioned native revisions and
  stale-request invalidation. Restrictive concurrent conflicts/unknown states
  deny pending review, rather than choosing the most permissive policy.
- Recovery/import restores restrictive policy and current retirement controls,
  but never restores grant authority from an old backup. Enrollment must be
  re-established locally; current authoritative controls precede content reuse.
- Shared-core schema/vocabulary/canonical fixtures, runtime destination/purpose
  adapters, native grant store and server compatibility are future rollout
  requirements. No server conformance is inferred from a client-only test.

## Independently testable first future release

Begin with local native deny enforcement and explicit enrollment of a qualified
Chatbook-owned on-device model for **new, registered-lineage V2 inputs in new
conversations**. The package includes ordinary profile context, profile tools,
root/child rounds, prepared state and stored tool/results needed to continue
those conversations. Model process/slot state must also qualify as clean
or fully policy-bound; new conversation identity alone is insufficient. Required
owners must preserve restrictions and enforce final entry. No partial feature may advertise a global model-disclosure guarantee
while a tool, child, replay or history path can bypass it. External/unqualified
model-generated tool arguments and independent filesystem/publication paths
remain disabled for governed runs as specified above. Unqualified model
routes may run ordinary chats only with no governed input and verified separable
history; otherwise refuse.

Adaptive model interviews, summaries/compaction, model embeddings, background
consolidation, managed-log/model replay, prompt-cache Save/Restore/reuse and
remote audiences remain disabled for
these governed inputs until their own adapters qualify the same contract. Fixed
questionnaires remain possible without model egress. A separately qualified
remote-audience release can enroll exact direct provider destinations. General
exports retain their explicit owner action; importing/uploading them grants no
model permission. Known legacy conversations remain outside the first release
until qualified classification/retirement; they must not be silently declared safe.

No new external database, semantic classifier, provider probe, scheduled job or
paid call is required by this design. Qualification uses synthetic records,
real native preparation/gateway seams and deterministic observing adapters.
No full test sweep or provider execution is authorized here.

## Synthetic future acceptance cases

| Case | Required result |
| --- | --- |
| Syncable/visible legacy record with no reviewed model policy | Deny governed model use; no silent grant from migration/old credentials |
| Device-only record and explicit cloud grant | Hard ceiling denies; local qualified execution needs its own current grant |
| Loopback endpoint is an SSH tunnel or unowned relay | No on-device qualification; no governed content sent |
| Same provider label points to new origin/model/account/route | Invalidate snapshots/grants; no label-based fallback |
| Qualified local grant receives a fallback to remote/unknown destination | Re-evaluate/deny before adapter entry; no automatic prompt forwarding |
| Hidden/wrong-workspace/unsupported record changes while permitted records stay fixed | Model/tool/Next Send permitted content and omission shape remain unchanged; no global quarantine existence flag |
| Denied workspace exception and an admitted global record | Denied item does not affect eligible ordering/overrides/budget |
| Profile get/search/update result on a child route with no grant | Generic unavailable projection; no canonical body/provenance leakage |
| Child report or mixed summary includes local-only and remote-allowed inputs | Intersection stays local-only; remote route omits whole governed artifact or stops |
| Prepared payload waits in a queue while a grant is revoked | Final entry denied; queued payload retired; no TTL/check-to-entry bypass |
| Retry, summary route or resumed interview changes audience/purpose | Fresh admission required; conversation-purpose receipt cannot authorize another purpose |
| Adaptive interview includes visible syncable records and earlier answers | Review interview input audience separately; omit/stop unpermitted input, not send due to Syncability |
| Future embedding backend switches to a network model | No embedding-purpose grant means no profile text/vector request |
| Qualified local model emits a web URL, MCP argument, shell command or independent file containing governed input | Preserve conservative dependency restrictions and refuse external/unqualified invocation/publication before its native boundary; ordinary tool permission is insufficient. No exfiltration merely because the model itself is local. |
| Local model has a restored cache binary or reused slot with unknown earlier input | No qualified send from a new conversation merely because its history is empty; use an explicitly reviewed clean owned process/slot or refuse. Unknown cache Save/Restore/reuse stays disabled, with no implicit reset. |
| Log/capture/history reconstructs a profile tool result at a new destination | Restrictions survive reconstruction; unknown legacy lineage blocks complete guarantee |
| Import/Sync/old backup includes audience handles or old grant metadata | Restrictive ceiling survives; native authority is not imported and current controls apply first |
| User opens Next Send or rejects the consent preview | No new grant; no selected private content in automatic diagnostics/logs |
| Old peer/server lacks policy capability or offers native model execution | Block policy activation/consumer use; delivery/storage is not execution consent |
| Explicit plaintext export then a request to upload to another model | Export is a separate action; upload still needs current model audience/source authority |

These are future cases, not passing runtime/provider evidence.

## Technical review checklist

- [x] All seven task criteria mapped to concrete contracts and future cases.
- [x] Current code facts are separated from unimplemented policy/grants.
- [x] Device-only amendment, local qualification and export exception are explicit.
- [x] Every governed egress path is enforced or disabled in the bounded release.
- [x] Migration, unknown audiences/lineage and restricted diagnostics deny safely.
- [x] Scoped documents/tracker links, criteria, provisional ID and whitespace verified.
