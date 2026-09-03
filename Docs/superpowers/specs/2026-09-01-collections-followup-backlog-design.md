# Collections Follow-up Backlog Design

**Date:** 2026-09-01

**Status:** Approved for backlog creation after independent and practical review

**Repositories:** `tldw_chatbook`, `tldw_server`

**Foundation:** TASK-18919 and ADR-107

## Goal

Turn the four known post-reader opportunities into an ordered, atomic backlog without reopening
TASK-18919 or implying that unsupported Server behavior is safe. Cross-repository capabilities are
implemented and attested by `tldw_server` before Chatbook enables them. Review of the shipped APIs
found that the broader management workflows also need bounded capability prerequisites: output
templates are shared and lack reference-safe deletion, digest schedule listing lacks an exact page
envelope, and export is page-capped rather than a complete restart-safe job. Broader Collections
management is therefore split by workflow and paired with the smallest truthful Server contract so
each implementation task can ship in one PR. The legacy recovery lifecycle begins with a decision
task because migration and retirement have materially different data-loss, rollback, and
compatibility consequences.

## Approaches considered

### Selected: atomic cross-repository tasks

Create six Server capability tasks, five corresponding Chatbook workflow integrations, one
Chatbook tracking parent, and one Chatbook ADR decision task. This produces thirteen records whose
ownership and completion evidence are unambiguous. Four Server tasks harden only the concrete
gaps found in otherwise-shipped template, digest, and import/export contracts; they do not redesign
those products.

### Rejected: six coarse tasks

One task per requested line plus two Server prerequisites would make templates, digests, and
import/export one multi-PR implementation unit. It would also leave the Chatbook children with no
truthful capability evidence and mix the legacy lifecycle decision with an implementation whose
desired behavior is not yet known.

### Rejected: duplicate all management work in both repositories

The Server already exposes output-template, Reading digest, and Reading import/export contracts.
Replacing those surfaces would duplicate existing product work. The selected Server prerequisites
retain those routes and address only observed blockers: shared-template ownership/deletion,
digest-page attestation, and complete native export/re-import lifecycle.

## Task structure

### `tldw_server` capability prerequisites

#### S1. Add atomic revision-guarded hard delete for Reading items

Add a positive, monotonically increasing integer `revision` to persisted Reading items and every
Reading summary/detail response. Every item-owned mutation that can change the user's deletion
decision increments that revision, including metadata/status/tag changes and capture-owned content
changes. The permanent-delete request accepts `expected_revision` and enforces `WHERE id = ? AND
user_id = ? AND revision = ?` or its backend-equivalent in the same transaction as child cleanup
and deletion. A stale precondition returns a documented 409/412 conflict without deleting the item;
missing items remain distinct. The operation removes capture-owned children and artifacts but does
not delete linked external Media or Notes. Docs-info advertises exact
`hasReadingOptimisticDeletesV1=true` only when the response token and atomic mutation are active.
SQLite/PostgreSQL migration, concurrent mutation/delete, authorization, conflict, cascade, and
diagnostic-privacy behavior are covered.

#### S2. Add coherent scoped tag and domain aggregates for Reading items

Expose bounded, deterministic, fully pageable tag and domain aggregate results for an explicitly
documented Reading scope. It uses distinct `capture_q` and `facet_q` parameters so facet-value
search never means filtering only already-loaded pages and cannot be confused with capture search.
Aggregate rows and exact aggregate totals are evaluated in one snapshot with the accepted capture
search/status/favorite/date/tag/domain filters. The contract
uses self-excluding semantics: the requested facet ignores that facet's active filter while retaining
all other scope filters, allowing the user to change it without losing alternatives. Normalized
value plus stable tie-break ordering makes every matching value reachable. Results are user-scoped
and exclude capture content and sensitive URL components. Docs-info advertises exact
`hasReadingAggregateFacetsV1=true` only when the endpoint and its SQLite/PostgreSQL snapshot
guarantees are active.

#### S3. Establish safe ownership and capability evidence for Collections output templates

Keep the existing `/outputs/templates` API but define which template types are governed by the
Collections umbrella and expose the current live reference owner: Reading digest schedules.
Historical rendered outputs do not block deletion because they are self-contained. Update and
delete remain user-scoped; deletion refuses a template referenced by a Reading digest schedule
unless that reference is first removed or reassigned. The task records a source-level reference
inventory at its branch base; finding another live owner requires updating the task acceptance
criteria before implementation rather than silently expanding scope. The API returns bounded
conflict reasons without leaking another user's objects. Docs-info advertises exact
`hasCollectionsOutputTemplateManagementV1=true` only when bounded paging, CRUD, reference-safe
deletion, and the documented ownership set are active. Existing template rendering remains intact.

#### S4. Attest bounded Reading digest schedule and output management

Keep all existing digest routes and response shapes unchanged. Add an exact bounded schedule-page
route beside the current bare-list route; the existing output-page route retains its envelope. Both
use deterministic ordering and user scoping. Schedule creation accepts a caller-generated,
user-scoped `client_request_id`: repeating the same key and normalized payload returns the original
schedule, while reusing it with a different payload fails with a bounded conflict. Exact lookup by
that key lets a client reconcile a lost create response. Preserve schedule `last_status`,
`last_run_at`, `next_run_at`, and output history as the only run evidence; do not invent a distinct
run-history API. Update/delete responses remain non-optimistic and are documented for refresh-based
reconciliation after transport uncertainty. Docs-info advertises exact
`hasReadingDigestManagementV1=true` only when these additive guarantees are active and configured
scheduler/worker availability is reported separately.

#### S5a. Add complete restart-safe Reading export jobs

Keep the current page-scoped streaming `/reading/export` route and response unchanged for existing
clients. Add asynchronous, user-scoped export-job routes that evaluate one explicit filter scope
coherently, write every matching item exactly once to a private managed artifact, expose bounded job
history/detail, and support authorized download and confirmed cleanup. Interruption is restart-safe
and never publishes a partial artifact as complete. A caller-generated export request key prevents
duplicate artifacts after a lost create response; reusing a key with a different normalized scope or
content payload returns a bounded conflict. Each artifact carries the versioned Server-native
portable-schema identifier and manifest consumed by S5b. Docs-info advertises exact
`hasReadingExportJobsV1=true` only when complete-scope export, job lifecycle, artifact retention,
and cleanup guarantees are active.

#### S5b. Add Server-native Reading export re-import

Retain Pocket JSON and Instapaper CSV import, then add Server-native JSONL/ZIP admission for
artifacts produced by S5a. The portable schema includes submitted URL, title, summary, freeform
note, status, favorite, tags, published/read timestamps, optional sanitized text/clean HTML, and
capture-owned highlights. It excludes database/user IDs, authoritative created/updated timestamps,
Media and linked-Note identities, generated audio, offline/archive files, and internal metadata.
Import allocates new identities, recomputes canonical URLs, and records source timestamps only as
bounded import provenance. On a canonical-URL collision it preserves existing scalar/state/content
fields, unions tags, and adds only nonduplicate capture-owned highlights using a documented stable
fingerprint. Importing into an empty authority reproduces every portable field; repeated import is
idempotent and never creates another capture or highlight. Docs-info advertises exact
`hasReadingNativeImportV1=true` only when this versioned field and collision contract is active.

### `tldw_chatbook` capability integrations

#### C1. Enable capability-gated Server capture hard delete

Reference S1 as `tldw_server:TASK-N` rather than using a same-repository Backlog dependency or a
pre-merge `blob/dev` URL. Chatbook keeps Server hard delete visibly unavailable until exact
`hasReadingOptimisticDeletesV1=true` is positively established and refuses a response with a
missing/invalid revision instead of using its current fallback value. It sends the loaded revision
with the destructive request, preserves the item on conflict or unknown outcome, refreshes
authoritative state, and removes the row only after confirmed deletion. Confirmation and cleanup
semantics continue to follow ADR-055 and ADR-107, and linked Media or Notes are never presented as
deletion targets.

#### C2. Present complete capability-gated Server tag and domain facets

Reference S2 as `tldw_server:TASK-N`. Until exact
`hasReadingAggregateFacetsV1=true` exists, retain typed filters and never label suggestions from
returned rows as complete facets. When supported, every facet browse and value search calls the
Server endpoint with bounded paging; Chatbook never filters a loaded prefix to claim a complete
result. Expose exact counts, deterministic paging, complete-scope filtering, generation fencing,
and explicit loading/empty/error/retry states. Source changes discard prior-authority facet state,
and narrow layouts preserve the adaptive reader.

### `tldw_chatbook` Collections-management program

#### C3. Tracking parent: complete Server Collections management workflows

Track three independent children and close only when all three are Done. The parent has a
non-implementing coordination/closeout plan: maintain child links and status, verify their combined
navigation and capability honesty, record integrated evidence, complete its acceptance criteria,
and close after all children. It does not own production code.

#### C3a. Manage Server Collections output templates

Reference S3 as `tldw_server:TASK-N` and require exact
`hasCollectionsOutputTemplateManagementV1=true`. Add bounded browse, detail, create, edit, and
reference-safe delete only for the Server-owned Collections template types. The UI explains that a
template may serve multiple Collections workflows and surfaces Server in-use conflicts without
offering a destructive bypass. Create/update/delete transport uncertainty never auto-retries;
Refresh reconciles authoritative state. Delete requires title-specific permanent confirmation.
Local mode does not claim template parity. Validation, paging, authorization, unknown outcomes,
and destructive confirmation are covered by service and mounted tests.

#### C3b. Manage Server Reading digest schedules and outputs

Reference S4 as `tldw_server:TASK-N` and require exact
`hasReadingDigestManagementV1=true`. Add exact bounded schedule browse/detail/create/edit,
enable-disable/delete, and output history using existing Reading digest contracts. Present only
`last_status`, `last_run_at`, `next_run_at`, and output history; do not label these as a complete run
ledger. Every create submission carries one stable request key; after an unknown response, Check
Status performs exact lookup and an explicit retry reuses that key, so it cannot duplicate the
schedule. Timezone and recurrence values are presented in human terms. Delete is permanent and
confirmed. Ambiguous update/delete outcomes never auto-retry and instead retain input plus offer
Refresh/reconciliation; only explicit Server conflicts are called conflicts. Scheduler/worker
unavailability remains distinct from CRUD capability. Local mode remains unsupported unless a
separate Local design is approved.

#### C3c. Manage full Server Reading import and export workflows

Reference S5a and S5b as `tldw_server:TASK-N` identities and require both exact
`hasReadingExportJobsV1=true` and `hasReadingNativeImportV1=true`. Add bounded
Pocket/Instapaper/Server-native import admission, import/export job history and detail, explicit
export scope/format/content controls, authorized artifact download, and retention/cleanup
presentation. Artifact cleanup is permanent and requires export-title/date confirmation; an unknown
cleanup response preserves the row as stale until Refresh proves whether the artifact remains. The
UI distinguishes accepted, running, partially successful, completed, failed, cancelled,
interrupted, and unknown outcomes; it does not load unbounded files, retain private paths, or present
a partial artifact as complete. Round-trip means the exact S5b portable field set through a
Server-native export and import; Pocket/Instapaper are import-only compatibility formats.
Production-shaped tests use more than one API page and verify collision idempotency. Local capture
import/export is not inferred from the Server contract.

### `tldw_chatbook` legacy lifecycle decision

#### C4. Decide migration or retirement of legacy generic Collections recovery

Inventory the v1 schema, code reachability, compatibility promises, and downgrade paths using only
repository evidence and synthetic fixtures. No real-user database, telemetry, row count, path,
identifier, name, item text, URL, or membership is inspected or collected. Compare at least:
retained export-only recovery, explicit user-approved migration into captures, and retirement after
a defined release/notice window. The task produces a new accepted ADR defining authority mapping,
canonical-URL collisions, membership handling, consent, backup/export requirements, rollback,
retention, privacy boundaries, and removal gates. No legacy data is mutated or deleted in this
decision task. Atomic implementation tasks are created only after the ADR selects an outcome.

## Filing metadata

All records are created **To Do** and unassigned. Implementation plans are intentionally absent
until a task is moved to In Progress.

| Key | Repository | Title | Priority | Labels |
| --- | --- | --- | --- | --- |
| S1 | `tldw_server` | Add atomic revision-guarded hard delete for Reading items | high | collections, reading-list, api, concurrency, deletion |
| S2 | `tldw_server` | Add coherent scoped tag and domain aggregates for Reading items | medium | collections, reading-list, api, facets, pagination |
| S3 | `tldw_server` | Establish safe Collections output-template ownership and deletion | medium | collections, output-templates, api, deletion |
| S4 | `tldw_server` | Attest bounded Reading digest schedule and output management | medium | collections, reading-list, digests, pagination |
| S5a | `tldw_server` | Add complete restart-safe Reading export jobs | medium | collections, reading-list, export, jobs |
| S5b | `tldw_server` | Add Server-native Reading export re-import | medium | collections, reading-list, import-export, portability |
| C1 | `tldw_chatbook` | Enable atomic Server capture hard delete | medium | library, collections, reading-list, deletion, server-parity |
| C2 | `tldw_chatbook` | Present complete Server capture tag and domain facets | medium | library, collections, reading-list, facets, server-parity |
| C3 | `tldw_chatbook` | Complete Server Collections management workflows | medium | library, collections, server-parity, tracking |
| C3a | `tldw_chatbook` | Manage Server Collections output templates | medium | library, collections, output-templates, server-parity |
| C3b | `tldw_chatbook` | Manage Server Reading digest schedules and outputs | medium | library, collections, reading-list, digests, server-parity |
| C3c | `tldw_chatbook` | Manage Server Reading import and export workflows | medium | library, collections, reading-list, import-export, server-parity |
| C4 | `tldw_chatbook` | Decide legacy Collections migration or retirement | low | library, collections, legacy, adr, decision |

Server records reference `tldw_chatbook:TASK-18919` and
`tldw_chatbook:Docs/superpowers/specs/2026-09-01-collections-followup-backlog-design.md`.
Chatbook records reference TASK-18919, ADR-107, and this design. C1, C2, C3a, C3b, and C3c also
reference their exact `tldw_server:TASK-N` prerequisites after the Server records allocate IDs.

## Draft acceptance criteria

### S1 — atomic Reading hard delete

- [ ] Every Reading item summary/detail exposes a positive monotonic `revision`, and every
  item-owned change relevant to deletion advances it in SQLite and PostgreSQL.
- [ ] Hard delete atomically requires the authenticated user's exact `expected_revision`; stale
  requests return a documented conflict and remove nothing.
- [ ] Confirmed hard delete removes the Reading item and capture-owned children/artifacts without
  deleting linked external Media or Notes.
- [ ] Schema migration, concurrent mutation/delete, wrong-user, missing-item, rollback, and cascade
  behavior have focused SQLite/PostgreSQL regression coverage.
- [ ] Docs-info advertises `hasReadingOptimisticDeletesV1=true` only when the complete contract is
  active, and public API documentation describes the precondition and responses.
- [ ] A new or applicable Server ADR records the schema, destructive precondition, and ownership
  decision before implementation begins.

### S2 — coherent Reading aggregate facets

- [ ] Authenticated callers can page every tag or domain aggregate through deterministic bounded
  results with exact aggregate totals and server-side `facet_q` search.
- [ ] Aggregate count and rows use one SQLite/PostgreSQL snapshot and apply `capture_q` plus all
  non-self facet filters before aggregation.
- [ ] Self-excluding tag/domain semantics are documented and verified for combined filters,
  concurrent writers, empty scopes, deep pages, and normalization collisions.
- [ ] Responses reveal no capture body, note, highlight, sensitive URL component, or other user's
  values.
- [ ] Docs-info advertises `hasReadingAggregateFacetsV1=true` only with the complete contract, and
  the ADR check plus focused API/database/security tests are recorded.

### S3 — safe Collections template ownership

- [ ] The API and documentation identify the exact output-template types governed by Collections
  and record a branch-base inventory of live template reference owners.
- [ ] Template listing remains bounded and user-scoped, while create/get/update/delete preserve the
  existing rendering contract and expose bounded errors.
- [ ] Deleting a template referenced by a Reading digest schedule is refused atomically without
  exposing another user's objects; historical self-contained outputs do not block deletion.
- [ ] Docs-info advertises `hasCollectionsOutputTemplateManagementV1=true` only when ownership,
  CRUD, paging, and reference-safe deletion guarantees are active.
- [ ] Focused database/API/security tests and the required Server ADR check pass.

### S4 — bounded Reading digest management

- [ ] Existing digest routes and response shapes remain compatible, and an additive schedule-page
  route returns exact totals with deterministic bounded rows.
- [ ] Schedule creation uses a user-scoped `client_request_id`; identical replays return the same
  schedule, payload mismatch conflicts, and exact lookup reconciles a lost response.
- [ ] Schedule and output reads never cross users and expose only `last_status`, `last_run_at`,
  `next_run_at`, and bounded output history rather than claiming a complete run ledger.
- [ ] Update/delete outcomes and scheduler/worker availability are documented separately so clients
  can refresh after uncertainty without inventing optimistic conflicts.
- [ ] Docs-info advertises `hasReadingDigestManagementV1=true` only when the additive paging and
  idempotent-create contract is active.
- [ ] Focused SQLite/PostgreSQL, API, concurrency, compatibility, and security tests pass; the ADR
  check is recorded.

### S5a — complete Reading export jobs

- [ ] Existing page-scoped streaming export remains compatible, while additive authenticated job
  routes expose bounded create/history/detail/download/cleanup behavior.
- [ ] Each export job reads one explicit filter scope coherently and writes every matching capture
  exactly once to a private managed artifact, including scopes larger than one API page.
- [ ] A user-scoped request key makes identical creates idempotent and rejects a different normalized
  payload with a bounded conflict.
- [ ] Interrupted or failed jobs resume or terminate truthfully and never publish a partial artifact
  as complete; retention and confirmed cleanup are restart-safe.
- [ ] Every completed artifact includes the versioned Server-native schema identifier and manifest
  required by S5b, and unauthorized users cannot discover or download it.
- [ ] Docs-info advertises `hasReadingExportJobsV1=true` only with the full contract; a Server ADR
  plus focused job/database/API/security tests are complete.

### S5b — Server-native Reading re-import

- [ ] Import accepts the versioned Server-native JSONL/ZIP artifacts produced by S5a while retaining
  Pocket JSON and Instapaper CSV compatibility.
- [ ] Empty-authority import reproduces every declared portable field and allocates fresh local
  identities without importing user/database IDs, external links, private paths, or internal
  metadata.
- [ ] Canonical-URL collisions preserve existing scalar/state/content fields, union tags, and add
  only nonduplicate capture-owned highlights using the documented stable fingerprint.
- [ ] Re-importing the same artifact is idempotent and produces no duplicate capture or highlight;
  malformed, future-version, oversized, and partially corrupt input fails safely.
- [ ] Docs-info advertises `hasReadingNativeImportV1=true` only with the field/collision contract and
  public portability documentation.
- [ ] S5a is a same-repository dependency, and focused parser/job/database/API/security and
  multi-page round-trip tests pass with the applicable ADR recorded or amended.

### C1 — Chatbook Server hard delete

- [ ] Server hard delete remains visibly disabled with a reason unless
  `hasReadingOptimisticDeletesV1=true` is positively established for the active authority.
- [ ] Server captures with a missing/invalid revision fail closed; no fallback revision is used for
  destructive actions.
- [ ] Confirmed deletion sends the loaded revision, requires title-specific permanent confirmation,
  and removes the row only after authoritative success.
- [ ] Conflict or unknown outcome preserves the capture, marks state stale, and offers authoritative
  Refresh without automatic retry; external Media and Notes are never deletion targets.
- [ ] Local behavior remains unchanged and service/controller/mounted/live regressions cover source
  switches, late responses, narrow layouts, and ADR-055/ADR-107 semantics.

### C2 — Chatbook complete Server facets

- [ ] Without `hasReadingAggregateFacetsV1=true`, typed filters remain available and current-page
  suggestions are never labelled complete facets.
- [ ] With the capability, tag/domain browse and `facet_q` search use bounded Server pages with exact
  totals; the client never filters a loaded prefix to claim completeness.
- [ ] Counts and values reflect the active capture scope and documented self-excluding semantics,
  with deterministic navigation across deep pages.
- [ ] Authority/scope changes and unmounts fence late results; loading, empty, stale, failure, and
  Retry states remain explicit.
- [ ] Service/state/mounted and 160×50/120×35/100×30/80×24 regressions preserve the adaptive reader
  and Local typed-filter behavior.

### C3 — Collections management tracking parent

- [ ] C3a, C3b, and C3c are linked as children with current status and external Server prerequisites.
- [ ] The parent carries a non-implementing coordination/closeout plan after entering In Progress and
  owns no production code.
- [ ] Integrated navigation and fail-closed capability behavior across all three workflows have one
  recorded production-shaped verification pass.
- [ ] The parent closes only after every child is Done and its combined evidence and residual risks
  are documented.

### C3a — Chatbook Server template management

- [ ] The surface remains disabled unless `hasCollectionsOutputTemplateManagementV1=true` is
  positively established and exposes only the Server-declared Collections template types.
- [ ] Users can browse exact bounded pages, inspect, create, edit, and preview templates without
  claiming Local parity.
- [ ] In-use deletion conflicts identify the blocking workflow without offering a destructive bypass;
  eligible deletion requires title-specific permanent confirmation.
- [ ] Create/update/delete uncertainty never auto-retries, retains user input where applicable, and
  reconciles through Refresh.
- [ ] Service/schema/controller/mounted/live tests cover validation, paging, authorization,
  capability changes, conflicts, uncertainty, focus, and narrow layouts.

### C3b — Chatbook Server digest management

- [ ] The surface remains disabled unless `hasReadingDigestManagementV1=true` is positively
  established, with scheduler/worker availability shown separately.
- [ ] Users can browse exact bounded schedule pages, inspect/create/edit/enable-disable/delete
  schedules, and browse bounded output history.
- [ ] Recurrence and timezone are presented in human terms; the UI exposes only Server-proven
  last/next/status/output evidence and never claims a complete run ledger.
- [ ] Create uses one stable request key with Check Status and explicit idempotent retry after an
  unknown response; it cannot duplicate a schedule.
- [ ] Delete is permanently confirmed, while uncertain update/delete outcomes retain context and
  reconcile through Refresh without automatic retry.
- [ ] Service/schema/controller/mounted/live tests cover paging, validation, authority changes,
  capability/worker states, uncertainty, focus, and narrow layouts.

### C3c — Chatbook Server import/export management

- [ ] The workflow remains disabled unless both `hasReadingExportJobsV1=true` and
  `hasReadingNativeImportV1=true` are positively established.
- [ ] Users can admit bounded Pocket/Instapaper/Server-native imports and inspect exact bounded
  import/export job history and detail.
- [ ] Export controls define scope, format, included portable content, retention, authorized download,
  and permanently confirmed artifact cleanup without revealing private paths.
- [ ] Accepted/running/partial/completed/failed/cancelled/interrupted/unknown states are distinct;
  partial artifacts are never presented as complete and uncertain cleanup remains stale until Refresh.
- [ ] Server-native round-trip and canonical-collision behavior match S5b; Pocket/Instapaper remain
  import-only and Local import/export parity is not claimed.
- [ ] Service/schema/controller/mounted/live tests use more than one API page and cover idempotency,
  authority changes, recovery, focus, and all standard terminal geometries.

### C4 — legacy Collections lifecycle ADR

- [ ] Repository code/schema/recovery reachability, compatibility promises, and downgrade behavior
  are inventoried with synthetic fixtures only; no real-user data or telemetry is inspected.
- [ ] Retained export-only recovery, explicit user-approved capture migration, and time-bounded
  retirement are compared with evidence and stated trade-offs.
- [ ] A new accepted ADR selects the lifecycle and defines authority mapping, canonical collisions,
  memberships, consent, backup/export, rollback, retention, privacy, and removal gates.
- [ ] The decision task mutates or deletes no legacy data and changes no production lifecycle.
- [ ] Any approved implementation is decomposed into atomic Backlog tasks only after the ADR selects
  an outcome; those tasks do not reference uncreated future IDs.
- [ ] Decision evidence, ADR linkage, documentation checks, and task closeout satisfy repository DoD.

## Ordering and references

- S1 through S4, S5a, and S5b are Server tasks; S5b depends on S5a's versioned portable export
  schema, while the others are independent.
- C1 references S1 and remains blocked on its attested capability.
- C2 references S2 and remains blocked on its attested capability.
- C3a, C3b, and C3c are independent children of C3, depend on completed TASK-18919 behavior, and
  reference S3, S4, and both S5 tasks respectively.
- C4 depends on TASK-18919 and references ADR-107; it requires a new ADR but no code change.
- Chatbook tasks use stable `tldw_server:TASK-N` references rather than declaring a same-repository
  dependency on a potentially colliding numeric task ID. Canonical `blob/dev` URLs may be added
  after the Server task files merge, when those links can resolve.

## Backlog quality rules

- Each implementation record describes user-visible or externally verifiable outcomes, not a file
  edit sequence.
- Every task is sized for one PR; the tracking parent is explicitly non-implementing.
- Capability tasks fail closed until positive versioned evidence exists.
- No task claims Local/Server parity where one authority lacks the feature.
- No task references an uncreated future task ID.
- Implementation plans and final ADR decisions are added only when their tasks enter progress.

## ADR assessment

- S1: a Server ADR is required because it adds a schema revision, destructive precondition, and
  public capability contract.
- S2: a Server ADR check is required during planning because it adds a public aggregate contract;
  reuse an applicable existing ADR or create one then.
- S3: a Server ADR check is required because it formalizes shared template ownership and durable
  reference-safe deletion.
- S4: no new Server ADR is expected because it hardens the existing digest contract's paging and
  attestation; planning must record the check.
- S5a: a Server ADR is required because it adds managed export-job artifacts, retention, restart,
  and cleanup ownership.
- S5b: the S5a ADR must govern the portable artifact; planning must amend it or create a linked ADR
  if native re-import adds a materially different data-ownership or collision decision.
- C1 and C2: no new Chatbook ADR is expected because they implement ADR-107's existing fail-closed
  capability boundary; planning must still record the check.
- C3a, C3b, and C3c: planning must assess whether existing Server ownership contracts fully govern
  the long-lived UI workflows; no storage decision is made by these backlog records.
- C4: a new ADR is mandatory and is the task's primary deliverable.
