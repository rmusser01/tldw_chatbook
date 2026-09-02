# Collections Follow-up Backlog Design

**Date:** 2026-09-01

**Status:** In review

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

Create five Server capability tasks, five corresponding Chatbook workflow integrations, one
Chatbook tracking parent, and one Chatbook ADR decision task. This produces twelve records whose
ownership and completion evidence are unambiguous. Three Server tasks harden only the concrete
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
documented Reading scope. The endpoint accepts a facet-value `q` search so searching never means
filtering only already-loaded pages. Aggregate rows and exact aggregate totals are evaluated in one
snapshot with the accepted capture search/status/favorite/date/tag/domain filters. The contract
uses self-excluding semantics: the requested facet ignores that facet's active filter while retaining
all other scope filters, allowing the user to change it without losing alternatives. Normalized
value plus stable tie-break ordering makes every matching value reachable. Results are user-scoped
and exclude capture content and sensitive URL components. Docs-info advertises exact
`hasReadingAggregateFacetsV1=true` only when the endpoint and its SQLite/PostgreSQL snapshot
guarantees are active.

#### S3. Establish safe ownership and capability evidence for Collections output templates

Keep the existing `/outputs/templates` API but define which template types are governed by the
Collections umbrella and expose their cross-workflow references. Update and delete remain
user-scoped; deletion refuses a template referenced by any digest schedule or other durable Server
workflow unless that reference is first removed or reassigned. The API returns bounded conflict
reasons without leaking another user's objects. Docs-info advertises exact
`hasCollectionsOutputTemplateManagementV1=true` only when bounded paging, CRUD, reference-safe
deletion, and the documented ownership set are active. Existing template rendering remains intact.

#### S4. Attest bounded Reading digest schedule and output management

Retain the existing digest routes while returning exact bounded page envelopes for schedules and
outputs, with deterministic ordering and user scoping. Preserve schedule `last_status`,
`last_run_at`, `next_run_at`, and output history as the only run evidence; do not invent a distinct
run-history API. Create/update/delete responses and errors are documented sufficiently for a client
to reconcile after transport uncertainty, but no speculative optimistic revision contract is added.
Docs-info advertises exact `hasReadingDigestManagementV1=true` only when those guarantees are
active and the configured scheduler/worker availability is reported separately.

#### S5. Add complete restart-safe Reading export and native re-import

Retain Pocket JSON and Instapaper CSV admission, then add Server-native JSONL/ZIP re-import for the
Server's own export schema. Replace page-capped pseudo-export with an asynchronous, user-scoped
export job that evaluates one explicit filter scope coherently, writes every matching item exactly
once to a private managed artifact, exposes bounded job status, and supports authorized download
and cleanup. Interruption is restart-safe and never publishes a partial artifact as complete.
Docs-info advertises exact `hasReadingImportExportManagementV1=true` only when native round-trip,
complete-scope export, job lifecycle, and artifact retention guarantees are active.

### `tldw_chatbook` capability integrations

#### C1. Enable capability-gated Server capture hard delete

Reference S1 by its full repository-qualified URL rather than using a cross-repository Backlog
dependency identifier. Chatbook keeps Server hard delete visibly unavailable until exact
`hasReadingOptimisticDeletesV1=true` is positively established and refuses a response with a
missing/invalid revision instead of using its current fallback value. It sends the loaded revision
with the destructive request, preserves the item on conflict or unknown outcome, refreshes
authoritative state, and removes the row only after confirmed deletion. Confirmation and cleanup
semantics continue to follow ADR-055 and ADR-107, and linked Media or Notes are never presented as
deletion targets.

#### C2. Present complete capability-gated Server tag and domain facets

Reference S2 by its full repository-qualified URL. Until exact
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

Reference S3 by full repository-qualified URL and require exact
`hasCollectionsOutputTemplateManagementV1=true`. Add bounded browse, detail, create, edit, and
reference-safe delete only for the Server-owned Collections template types. The UI explains that a
template may serve multiple Collections workflows and surfaces Server in-use conflicts without
offering a destructive bypass. Create/update/delete transport uncertainty never auto-retries;
Refresh reconciles authoritative state. Delete requires title-specific permanent confirmation.
Local mode does not claim template parity. Validation, paging, authorization, unknown outcomes,
and destructive confirmation are covered by service and mounted tests.

#### C3b. Manage Server Reading digest schedules and outputs

Reference S4 by full repository-qualified URL and require exact
`hasReadingDigestManagementV1=true`. Add exact bounded schedule browse/detail/create/edit,
enable-disable/delete, and output history using existing Reading digest contracts. Present only
`last_status`, `last_run_at`, `next_run_at`, and output history; do not label these as a complete run
ledger. Timezone and recurrence values are presented in human terms. Delete is permanent and
confirmed. Ambiguous create/update/delete outcomes never auto-retry and instead retain input plus
offer Refresh/reconciliation; only explicit Server conflicts are called conflicts. Scheduler/worker
unavailability remains distinct from CRUD capability. Local mode remains unsupported unless a
separate Local design is approved.

#### C3c. Manage full Server Reading import and export workflows

Reference S5 by full repository-qualified URL and require exact
`hasReadingImportExportManagementV1=true`. Add bounded Pocket/Instapaper and Server-native import
admission, import/export job history and detail, explicit export scope/format/content controls,
authorized artifact download, and retention/cleanup presentation. The UI distinguishes accepted,
running, partially successful, completed, failed, cancelled, interrupted, and unknown outcomes; it
does not load unbounded files, retain private paths, or present a partial artifact as complete.
Round-trip means Server-native export followed by Server-native import; Pocket/Instapaper are
import-only compatibility formats. Production-shaped tests use more than one API page. Local
capture import/export is not inferred from the Server contract.

### `tldw_chatbook` legacy lifecycle decision

#### C4. Decide migration or retirement of legacy generic Collections recovery

Inventory the v1 schema, code reachability, compatibility promises, and downgrade paths using
repository evidence and synthetic fixtures. An optional local probe may report only aggregate row
counts and whether recovery has been invoked for the currently selected database; it never records
names, item text, URLs, memberships, stable identifiers, raw paths, or cross-user telemetry. The ADR
must remain decidable without any real-user probe. Compare at least: retained export-only recovery,
explicit user-approved migration into captures, and retirement after a defined release/notice
window. The task produces a new accepted ADR defining authority mapping, canonical-URL collisions,
membership handling, consent, backup/export requirements, rollback, retention, privacy boundaries,
and removal gates. No legacy data is mutated or deleted in this decision task. Atomic
implementation tasks are created only after the ADR selects an outcome.

## Ordering and references

- S1 through S5 are independent Server tasks.
- C1 references S1 and remains blocked on its attested capability.
- C2 references S2 and remains blocked on its attested capability.
- C3a, C3b, and C3c are independent children of C3, depend on completed TASK-18919 behavior, and
  reference S3, S4, and S5 respectively.
- C4 depends on TASK-18919 and references ADR-107; it requires a new ADR but no code change.
- Chatbook tasks reference the eventual Server task file or repository URL rather than declaring a
  same-repository dependency on a potentially colliding numeric task ID.

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
- S5: a Server ADR is required because it adds managed export-job artifacts, retention, restart,
  and native round-trip ownership.
- C1 and C2: no new Chatbook ADR is expected because they implement ADR-107's existing fail-closed
  capability boundary; planning must still record the check.
- C3a, C3b, and C3c: planning must assess whether existing Server ownership contracts fully govern
  the long-lived UI workflows; no storage decision is made by these backlog records.
- C4: a new ADR is mandatory and is the task's primary deliverable.
