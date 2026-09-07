# Task 4 report — Research Sources workbench

## Status

Complete after review fix round 6. Research Workspace now has a compose-once, authority-explicit Sources
workbench with durable intake receipts, paged sources, exact desired selection,
readiness/status inspection, device-only folders/annotations, and honest owner
capability gates. Quick Notes bodies and Task 5 ownership were not added.

ADR required: no new ADR

ADR path: `backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md`

Reason: this task implements ADR-078's accepted authority, canonical catalog,
association-only removal, durable qualified intake, and device-overlay rules.

## Implementation

- Replaced the Sources placeholder with 25 stable source slots, 20 stable
  receipt slots, page/search/filter/sort/selection controls, explicit folder
  actions, row actions, and recovery states. Async owner updates patch existing
  widgets so focus and identity survive refresh.
- Added the five-path intake modal with exact Local/Server vocabulary:
  `Import Files`/`Upload`, `Local Library`/`My Media`, URL, Paste, and
  `Search Local`/`Search Server`. URL batch intake creates one durable operation
  per item; directory expansion with one operation is rejected by the app seam.
- Captured qualified workspace refs are persisted before ingest submission.
  `required_origin` makes app ingest fail before queue mutation when the active
  backend differs. Existing-catalog attachment routes through the explicit
  captured ref even after visible navigation and repaints only a still-current
  workspace.
- Added bounded qualified recent-operation lookup and receipts with independent
  Library/Media, workspace-association, and index/readiness stages. Retry emits
  the exact failed stage; readiness uses Refresh/Recheck and never claims a
  server indexing retry.
- Migrated the private overlay from schema v1 to v2 for bounded device-only
  source folders and annotations. V1 migration invents no folders; records are
  isolated across data source, server profile, principal, and workspace axes.
- Source selected intent and readiness are rendered separately. Select all
  reads the exact paged owner, remove unlinks only the workspace association,
  and row/batch controls fail closed from typed capabilities. Move/Copy remains
  visible and disabled because neither current owner exposes that canonical
  action.
- Added a status/preview/annotation inspector with lifecycle, reason, source of
  truth, progress disclosure, retry eligibility, stale state, readiness,
  identifiers, next action, honest missing-preview copy, and Device-only label.

## Parity action matrix

| Surface | Implemented owner action | Honest gate |
| --- | --- | --- |
| Quick URL, Upload/Import, URL batch, Paste | durable operation before app ingest; exact required origin | disabled/recovery when no workspace; submit failure remains a receipt |
| Local Library/My Media and Search Local/Server | bounded explicit-authority catalog search and receipt-first attach | unavailable callback reports the selected authority |
| Attached search, filters, date, sort, pagination | local stable view over a bounded owner page | unavailable owner fields are named and never guessed |
| Select all/visible/clear and row Select | exact owner desired IDs through typed selection capability | unknown/unavailable selection disables controls with reason |
| Preview/details/annotation | owner preview plus normalized status; annotation stays device-only | missing preview and progress are explicit |
| Remove | workspace association only | canonical Library/Media item is never deleted |
| Reorder | exact owner mutation when capability and owner bound allow | Local unsupported, temporary sort disabled, Server preflight remains typed |
| Move/Copy | none | visible disabled control and canonical-owner reason |
| Retry | exact failed catalog/association stage; readiness recheck | no fake server indexing retry |

## Verification evidence

- Fresh full focused Research Workspace/UI/app-origin gate: `359 passed,
  1 warning in 56.15s`.
- Explicit inverse matrix: `30 passed, 1 warning in 25.63s`, covering no Local
  fallback, operation-before-submit, captured-ref attach, late fencing,
  association-only removal, exact Server reorder preflight, stage retry, v1
  migration/qualified isolation, owner clearing, enabled event behavior, ASCII
  handles, and all production geometries.
- Production geometry: all 15 cases pass, including active Sources at 160x40,
  120x30, 100x30, 84x24, 80x24, and 60x20.
- Scoped Ruff lint passes. Eighteen Task 4-owned/heavily changed files are Ruff
  format-clean; the large legacy app/tiny seam files were kept free of unrelated
  whole-file formatting churn.
- Changed production `compileall`, `git diff --check`, and explicit Unicode
  arrow scan pass. Requested collapse handles remain exact ASCII `<---`/`--->`.
- Two consecutive timestamp-normalized CSS bundle hashes match
  (`145da747…`); the bundle-sync checker passes all five outputs.
- Impeccable detector result: `[]`.
- The only warning is the accepted environment `RequestsDependencyWarning`.
  Full pytest was not run, per repository policy.

## Inverse mutation evidence

- Removing the app `required_origin` precondition admitted a Local queue write
  for captured Server intake and made the no-mutation trace red.
- Submitting before operation creation inverted the URL trace and made the
  per-item durable ordering test red.
- Deriving attach from the currently visible workspace dropped an old modal's
  captured intent; the explicit-ref controller test was red until the owner ref
  was routed directly and validated on every identity axis.
- Restoring current-page-only Select all lost owner row 101. Restoring stale
  controller acceptance allowed late Local results to repaint another context.
- Enabling preview/remove/reorder without typed capabilities and retaining old
  folders during an authority switch made the mounted fail-closed guards red.
- Replacing unlink with canonical deletion, advertising indexing retry, or
  moving Server reorder before its exact-owner preflight makes the retained
  Task 2/3 inverse guards red.
- Decoding v1 as v2 without deterministic empty source fields and collapsing
  qualified keys made the overlay migration/isolation guards red.
- Replacing 25 stable source slots with 100 materially increased the mounted
  tree; the final bounded page uses 25 slots. The mounted guard measures 577
  descendants, requires fewer than 650 and mount under 1.0 second, and
  preserves focus without normalizing a slower wait.

All inverse mutations were restored before the final gates.

## Files and boundaries

Changes are limited to Research source controller/overlay/receipt seams, the
Research Workspace Sources UI and CSS, exact app ingest origin precondition,
and focused tests. No canonical content database, server folder/annotation API,
Quick Notes body, cross-authority transfer, source deletion, or keybinding was
introduced.

## Review fix round 1

The review's authority, pagination, worker, overlay, accessibility, and paste
lifecycle findings were verified against the actual app/runner and mounted
Textual seams before repair. No new ADR was required: the fixes close gaps in
ADR-078's existing authority, owner, conflict, and private-overlay contracts.

| Review family | Closure and executable owner |
| --- | --- |
| Qualified Server intake | Admission and delayed dispatch recover the durable operation and validate `data_source`, `server_profile_id`, and `principal_id`; identity changes produce no catalog/job/server mutation. App admission and delayed-dispatch race tests exercise the real helper seams. |
| Search/selection honesty | Server web search remains visible but disabled with provider/setup recovery and never calls My Media catalog search. Upload and Local Library/My Media explicitly say one file/item per action. |
| Worker safety | One screen wrapper runs source actions with nonfatal worker policy, contains expected capability/API/network/conflict/validation failures, preserves drafts/receipts, and logs unexpected defects. A mounted failing-port matrix exercises intake, selection, reorder, preview, remove, folder, and retry. |
| Multi-page semantics | Labels say `Filter current page`, `Preview visible selected`, and `Remove visible selected`. Reorder fetches exact owner order, works for 26, refuses 101, and never partially mutates. Off-page folder selection is disabled with exact recovery. |
| Add capability | Add/Quick URL and modal submission require typed `attach_existing` before operation, catalog, or ingest writes; a static denial writes nothing while a post-create race remains a durable failed receipt. |
| Receipt independence | Qualified recent receipts load first and survive owner capability/source failures, restart, and late completion; only the source projection is cleared. |
| Overlay CAS | Conflicts retain the local draft and mount explicit Reload, metadata-only Export, and in-memory Fork/copy actions. There is no silent or force overwrite. The two-store mounted test exercises each recovery action and verifies exported JSON omits annotation text, paths, and URLs. |
| Annotation CRUD | The inspector lists qualified annotations and creates, edits, deletes, reopens, and reloads stable IDs from the real overlay store without crossing source refs. |
| Nested folders | The bounded device-only hierarchy honors `parent_folder_id`, validates missing parents/cycles/depth, renders ancestry with ASCII expand/collapse controls, and requires explicit `Select folder sources`. |
| Safe dismissal/removal | Add, Inspector, removal confirmation, and overlay-conflict modal use Escape-safe dismissal. Add restores opener focus. Row/batch unlink mutation starts only after copy stating that the workspace association is removed while Library/Media is retained. |
| Disabled accessibility | Research disabled controls keep full opacity/text opacity, add `[Unavailable]`/reason copy, and preserve visible focus distinction under the consolidated theme. |
| Paste staging lifecycle | An app-owned 0700 directory and 0600 operation-derived files are indexed without bodies or paths. Operation creation precedes staging; success/cancel deletes, retryable failure retains, pre-job submission failure deletes, and a bounded startup sweep removes terminal/orphan artifacts without touching user upload paths. |
| Owner-independent detail and recovery | Details/folders remain explicitly device-only and qualified. Status exposes a real Refresh/Recheck action plus exact stage retry rather than a fake server indexing endpoint. |

### Fix-round RED and regression evidence

- Review-owned inverse matrix: **27 passed** in 10.36s after the fixes. It
  drives the production app/controller/store handlers and mounted screens for
  every row above (including three terminal staging states).
- Whole Research Workspace package: **256 passed** in 1.85s.
- App submission seam: first fresh run found a direct-helper compatibility
  regression (**3 failed, 130 passed**); making the optional Research operation
  link optional for ordinary Server helper calls produced **133 passed** in
  1.87s.
- Whole Library runner file: **145 passed, 1 Windows-only skipped** in 14.11s.
- Mounted Task 4 UI: **39 passed** in 15.56s; screen/app-wiring neighbors:
  **22 passed** in 17.61s.
- Six-size geometry initially exposed six stale assertions for the renamed
  page-local filter (**6 failed, 9 passed**); exact copy was corrected and the
  gate finished **15 passed** in 22.63s.
- Honest Upload/My Media single-selection copy was added test-first
  (**2 failed** before production, **2 passed** after).
- CSS build completed consecutively with identical timestamp-normalized hash
  `52463ed25c2a099680e864a8b9795d70f8ad6ce9f61caa51392b2b88837ac0c2`;
  integrity/bundle/staleness checks are **28 passed**.
- Impeccable detector `[]`; scoped Ruff, changed-production `compileall`,
  `git diff --check`, ASCII/keybinding, staging privacy/path, and bounded-tree
  scans pass. The only changed-suite warning is the accepted environment
  `RequestsDependencyWarning`. The full App neighbor additionally emits
  pre-existing third-party SWIG deprecations in an untouched PDF test.

Full pytest was not run, per repository policy. TASK-21508 remains
controller-owned and excluded from the fix commit.

## Review fix round 2

The three latest findings were reproduced before production edits and closed at
the existing owner boundaries; no new ADR or storage owner was introduced.

- Batch preview/remove IDs now come only from mounted, displayed stable row
  slots. Search, type, date/status, and device-folder filtering exclude hidden
  selected associations while the separate owner desired count remains exact.
- Quick URL, modal single/batch URL, and direct screen intake share one bounded
  `http`/`https` validator. It rejects credentials, relative/file URLs,
  control/format characters, malformed or oversized input before operation,
  staging, registry, or Server mutation, while preserving supported Unicode and
  percent-encoded URLs.
- Research intake now prepares and durably queues through the existing Library
  registry without dispatch, persists the operation's catalog/in-progress job
  link, and only then dispatches through the existing runner. Link failure
  durably cancels the undispatched job and deletes managed paste staging;
  dispatch failure durably fails the linked job and retains retryable staging.
  Immediate Local and Server terminal listeners observe the link exactly once.
- Ordinary non-Research Local, Server, and web-clip submissions keep their
  wrapper behavior. Strict persistence is opt-in for Research prepare/settle,
  and restart tests read the prepared or terminal state from the real SQLite
  job store.
- The Task 4-owned overlay-conflict test now waits for the observable remote
  state and remounted pane instead of transition timing; it passed five
  consecutive isolated runs.

### Fix-round-2 evidence

- Initial RED: 15 expected failures across visible selection, URL safety/no
  writes, and prepare-link-dispatch ordering (plus one valid-URL control pass).
  Follow-up strict durability/settlement and Local/Server prepare guards were
  also observed red before their production seams.
- Restored-code split gates: Research Workspace plus Task 4 UI **305 passed**;
  app ingest seam **140 passed**; Library runner **145 passed, 1 Windows-only
  skipped**; inspector/receipt/app-wiring/screen/all geometry **44 passed**;
  CSS integrity/bundle/staleness **28 passed**.
- Inverse mutations were independently red and restored: legacy page-level
  selection (**1 failed**), shared-validator bypass (**7 failed**), dispatch
  before link (**1 failed**), and omitted prepared-job cancellation on link
  failure (**1 failed**).
- Scoped Ruff lint and format, changed-production `compileall`,
  `git diff --check`, production privacy/Unicode-arrow scans, CSS gates, and
  Impeccable detector (`[]`) pass. Warnings are the accepted environment
  Requests dependency warning and pre-existing third-party SWIG deprecations;
  the runner skip is its Windows-only spawn/resource-tracker boundary.

Full pytest was not run, per repository policy. TASK-21508 remains
controller-owned and excluded from this fix commit.

## Review fix round 3

The latest crash-window, startup-recovery, filtered-preview, and privacy
findings were reproduced before production edits. ADR-078 was amended rather
than creating a duplicate ADR: schema v7's dispatch hold is the durable
eligibility mechanism for ADR-078's already accepted qualified two-owner ingest
transaction.

- Library ingest-job schema v7 adds a constrained `dispatch_held` column with
  atomic genuine-v6 migration and fresh-schema parity. Research preparation and
  catalog retry persist held queue rows; ordinary intake remains unheld. Every
  queue selector excludes held jobs, release is persistence-first and one-way,
  and restore preserves and protects held rows from the history cap.
- Startup scans one SQL-filtered bounded page of held queued Research jobs.
  Matching pending receipts are CAS-linked to the exact job before release;
  already-linked receipts release directly. Transient reads/writes isolate the
  row and retain staging. Missing, terminal, origin-incompatible, or differently
  linked receipts require durable job settlement before staging cleanup.
- Runtime link exceptions reread the durable receipt. An ambiguous committed
  link dispatches; an exact still-pending receipt remains visible for startup
  recovery with its job held and paste retained. Incompatible receipts clean
  only after confirmed terminal cancellation, and cancellation-store failure
  retains the managed artifact.
- The startup paste sweep now recognizes a missing operation with a durable held
  job, closing the concurrent sweep/reconcile deletion window. Held Research
  retries also require a persistence store before mutating registry history.
- Batch Preview is enabled by exactly one displayed selected association,
  independent of the global desired-selection count. Mounted search, type,
  readiness status, date, and folder filters target the exact displayed
  association.
- Unexpected Local classification logs contain only operation/job context and
  origin; managed paste paths and `source_path` representations are absent.

### Fix-round-3 RED evidence

- Initial focused review matrix: **17 expected failures** — DB/migration/restore
  4, registry eligibility 1, app authority/startup/privacy 7, runtime link 3,
  and filtered Preview 2.
- Follow-up inverse guards independently reproduced the concurrent staging
  sweep deletion and held-history pruning hazards before their production
  repairs.

### Fix-round-3 GREEN and closeout evidence

- Final review/inverse matrix after all edits: **23 passed** with the accepted
  environment Requests warning. It includes genuine v6 migration/history,
  rollback, held selector/release/retry/pruning, Local+Server preparation,
  restart and listener dedupe, transient/incompatible/cancellation cleanup,
  concurrent staging sweep, mounted filtered Preview targeting, and exception-
  text path privacy.
- Complete ingest DB plus registry neighbors: **128 passed**. App submission
  seam: **146 passed**. Complete Library runner neighbor: **145 passed, 1
  Windows-only skipped**. Complete Research Workspace package: **257 passed**.
- Task UI modal/inspector/receipt/Sources/app-wiring/screen/geometry split:
  **97 passed**, including all 15 production geometry cases. CSS integrity,
  bundle, class coverage, and staleness: **32 passed**.
- Scoped Ruff lint passes. Eleven owned/heavily changed files are Ruff
  format-clean; the large legacy app and runner retain their pre-existing
  whole-file format/lint baseline outside the edited lines. Changed production
  `py_compile`, `git diff --check`, privacy/ASCII scans, and the one-shot
  Impeccable detector (`[]`) pass.
- Warnings are the accepted environment `RequestsDependencyWarning` and the
  App neighbor's pre-existing third-party SWIG deprecations. The full suite was
  not run, per repository policy. TASK-21508 remains controller-owned and is
  excluded from the fix commit.

## Review fix round 4

The Library/Home/provider retry review finding was reproduced at the shared app
seam. A failed Research-owned ingest had been sent through the ordinary
`LibraryIngestJobRegistry.requeue` path, creating an unheld replacement,
topping up the Local parse pool, and leaving the durable source operation linked
to the failed job. No new ADR is required: this fix enforces ADR-078 through the
catalog retry owner and the dispatch-hold contract added in fix round 3.

- The shared retry seam now detects `research_source_operation_id` and routes
  only Research-owned catalog failures through
  `ResearchSourceAssociationScheduler.retry`. The scheduler's existing fence,
  held requeue, operation relink, and release/dispatch flow remain the sole owner.
- Exact operation/job lineage and Local/Server authority are checked before the
  scheduler can mutate. Missing owners, mismatches, and scheduler failures fail
  closed with fixed path-free Research Workspace recovery; no generic requeue,
  provider override, dispatch, or parse-pool top-up occurs.
- Concurrent retry requests converge on the one durable replacement selected by
  the scheduler fence. The async app worker returns only the released replacement
  named by the exact receipt, while the public UI-thread seam remains nonblocking.
- Ordinary Library jobs preserve the legacy synchronous replacement return and
  top-up behavior. Home reports the async Research request honestly, and Library
  rows suppress provider-specific recovery buttons that would bypass the saved
  operation options.

### Fix-round-4 RED and inverse evidence

- Initial production probe/new app matrix: **6 expected failures, 1 ordinary
  control pass**. It captured an unheld replacement, one generic top-up, stale
  operation linkage, generic Home copy, mutation with no scheduler, lineage
  mismatch mutation, and scheduler bypass under concurrent clicks.
- Disabling the final Research branch made the Local direct contract red with an
  immediate unheld replacement. Restoring provider actions for a Research row
  made the mounted honest-recovery contract red. Both mutations were restored
  and the exact controls returned **3 passed**.

### Fix-round-4 GREEN and closeout evidence

- Direct Local+Server, provider, Home, missing-owner, scheduler-error privacy,
  lineage, authority, concurrency, restart, and ordinary contracts plus the
  Library canvas: **146 passed, 1 baseline case deselected**.
- App/ingest DB/registry/Research association neighbors: **309 passed**, including
  the final **11 passed** Task 4 app contract. Home/Library screen/adapter
  neighbors: **140 passed, 1 baseline case
  deselected**. Library runner: **145 passed, 1 Windows-only skipped**. Focused
  Research source/workspace/UI neighbors: **194 passed**.
- Two broader UI failures were reproduced unchanged in a detached worktree at
  base `5cc0292eefe251db7083e36c49bde7df14cf5555`: the retained server-only
  prompt fixture still reports `server`, and the legacy Library fixture cannot
  find `#library-ingest-top-button`. They are excluded as proven baselines, not
  hidden regressions.
- Scoped production/new-test Ruff lint, new-test Ruff format, changed-production
  `compileall`, `git diff --check`, and the one-shot Impeccable detector (`[]`)
  pass. Warnings are the accepted Requests dependency warning and pre-existing
  third-party SWIG deprecations in an untouched PDF neighbor. Full pytest was
  not run, per repository policy.

TASK-21508 remains controller-owned and is excluded from the fix commit.

## Review fix round 5

The release-then-raise reviewer finding was reproduced against the real retry
scheduler, operation store, and ingest-job database for both Local and Server.
The catalog coordinator terminalized the operation after its dispatcher raised,
but did not settle the replacement that the dispatcher had already durably
released. The app then accepted that failed receipt as a successful worker
result. No new ADR is required: this closes another failure window in ADR-078's
existing dispatch-hold transaction.

- On dispatcher exception, the coordinator now durably marks the exact
  replacement failed before persisting the catalog-stage failure. A crash after
  replacement settlement but before receipt settlement leaves an in-progress
  operation pointing to a durable failed job, which startup resume can safely
  terminalize; a terminal failed operation never points at a queueable row.
- The app replacement validator accepts only exact receipts whose catalog stage
  is in progress or succeeded. A terminal failed receipt returns no replacement
  and emits the existing fixed, path-free Research Workspace recovery.
- Ordinary Library retries, successful Research dispatch, concurrent scheduler
  fencing, and explicit Local/Server authority checks remain unchanged.

### Fix-round-5 RED and inverse evidence

- Initial Local+Server RED: **2 expected failures**. Each worker returned the
  released replacement while its receipt was failed, the replacement remained
  `QUEUED` with `dispatch_held=False`, `next_queued()` selected it, and no
  recovery warning was emitted.
- Removing durable replacement settlement made the restored Local inverse fail
  with a queued replacement. Removing the catalog-status gate made the worker
  return the terminal failed replacement. Both mutations were restored; the
  final exact Local+Server, concurrency, and ordinary controls are green.

### Fix-round-5 GREEN and closeout evidence

- Final direct retry contract: **13 passed**, including Local+Server durable
  replacement failure, zero later queue selection, sanitized recovery, and
  ingest/operation SQLite reopen. App/DB/registry/association split: **311
  passed**.
- Complete Library runner: **145 passed, 1 Windows-only skipped**. Source
  operation/readiness/controller neighbors: **101 passed**.
- One parallel first pass transiently observed the retained concurrency test
  return the durable winner to one rather than both waiters (**1 failed, 310
  passed**). The exact test then passed six consecutive isolated runs, the
  release/concurrency sequence passed eight consecutive runs, and the fresh
  complete split finished **311 passed** without a production change.
- Scoped Ruff lint/format, changed-production `compileall`, and
  `git diff --check` pass. Warnings are the accepted Requests dependency warning
  and pre-existing third-party SWIG deprecations in an untouched PDF neighbor.
  No UI changed in this round, so the prior final-candidate Impeccable `[]`
  remains applicable. Full pytest was not run, per repository policy.

TASK-21508 remains controller-owned and is excluded from the fix commit.

## Review fix round 6

The failed-settlement-write reviewer finding was reproduced against the real
retry scheduler, operation store, registry, and ingest-job SQLite database for
both Local and Server. After a dispatcher durably released a replacement and
raised, an injected failure of the required replacement `FAILED` upsert left
the receipt in progress and the replacement `QUEUED`/unheld. The app's fallback
reread then returned that row as a successful worker result, and the same-process
generic selector could claim it. No new ADR is required: this closes a crash
window in ADR-078's existing durable dispatch-owner boundary.

- Releasing a held Research retry now atomically persists its dispatch claim as
  `PARSING` together with `dispatch_held=False`. A later dispatcher or terminal
  write failure can therefore leave an interrupted active row, but can never
  expose generic `QUEUED` work. Initial Research intake and ordinary Library
  release behavior remain unchanged.
- Local atomically claimed retries enter a small owner-pending set. The existing
  parse-pool top-up subtracts those not-yet-submitted claims from its in-flight
  count, respects total and heavy-lane capacity, submits each claim once, and
  removes it on submission or terminal dispatch failure. Server dispatch retains
  its explicit Server owner.
- Only the scheduler's expected CAS conflict uses the durable-winner reread.
  Other scheduler/coordinator failures return no replacement and emit the fixed,
  path-free Research recovery notice. Exact operation lineage and authority
  validation remain unchanged.
- A raw SQLite reopen retains the interrupted row as nonqueueable `PARSING`.
  Production `plan_restore` then normalizes that interrupted row to `FAILED`,
  after which a bounded coordinator resume terminalizes the still-in-progress
  receipt. This is containment and later recovery, not a claim that raw restore
  itself settles the operation.

### Fix-round-6 RED and inverse evidence

- Initial Local+Server integration RED: **2 expected failures**. Both workers
  returned a `QUEUED`/unheld replacement after the injected failed-state upsert;
  the receipt remained in progress, `next_queued()` selected the replacement,
  and no sanitized recovery was emitted.
- The atomic-release unit guard was independently red because release persisted
  `QUEUED` rather than `PARSING`.
- Three restored inverse mutations failed at the intended invariant: removing
  the atomic claim produced **3 failures** across the unit and Local+Server
  containment cases; restoring generic exception fallback returned a false
  replacement (**1 failure**); omitting the Local pending owner prevented later
  capacity dispatch (**1 failure**).

### Fix-round-6 GREEN and closeout evidence

- Exact Local+Server failed-upsert, atomic-release, and capacity controls:
  **4 passed**. The complete direct app retry file passed **15 tests** on five
  consecutive runs after the SQLite preflight was kept on the event-loop turn.
- App retry/submission split: **161 passed**; ingest registry/DB: **129 passed**;
  complete Library runner: **146 passed, 1 Windows-only skipped**; Research
  association/store/readiness/controller: **125 passed**. Across those targeted
  gates, **561 tests passed** with one platform skip.
- Scoped Ruff lint passes for changed production and tests. Ruff format passes
  the owned formatted files; the legacy runner retains two unrelated lint
  findings and unrelated whole-file format drift, while the new block is clean.
  Changed-file `compileall` and `git diff --check` pass.
- Warnings are the accepted environment `RequestsDependencyWarning` and
  pre-existing third-party SWIG deprecations in an untouched PDF neighbor. No UI
  changed in this round, so the previous final-candidate Impeccable result
  remains applicable. Full pytest was not run, per repository policy.

TASK-21508 remains controller-owned and is excluded from the fix commit.
