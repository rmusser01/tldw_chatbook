# Task 4 review package — Research Sources workbench

Fix-round-6 review base:

`8b5defce02bca90fafd3690bbab8be0aa8c258e4`

The controller-owned dirty TASK-21508 file remains excluded from the
implementation and commit.

## Fix-round-6 review order

1. `library_ingest_jobs.py`: Research retry hold release atomically persists the
   explicit owner's active claim, so no fallible later settlement can reopen
   generic queue eligibility.
2. `app.py`: Local claimed retries wait in the existing bounded parse-pool owner
   path; only expected scheduler CAS conflicts use durable-winner reread, while
   other failures return no replacement and notify with sanitized recovery.
3. Real-store Local+Server regression: injected replacement-failure upsert,
   same-process selector exclusion, raw SQLite reopen, production restore
   containment, truthful worker recovery, and bounded later operation resume.
4. Ordinary queue and retry parity, concurrent fence stability, capacity exact-
   once dispatch, split gates, static checks, and inverse evidence in the Task 4
   report.

### Fix-round-6 high-risk invariants

- A Research replacement is never simultaneously unheld, `QUEUED`, and
  operation-owned after explicit dispatch admission, even if every later
  failure-settlement write raises.
- Same-process generic `next_queued()` cannot claim an interrupted Research
  retry. A raw reopen remains nonqueueable, and production restore settles the
  interrupted active row before normal top-up.
- Scheduler/coordinator failures never become false successful worker results;
  exception text, secrets, and source paths remain absent from recovery copy.
- Local atomically claimed retries still respect total/heavy parse capacity and
  dispatch once when capacity becomes available. Server retries never cross to
  Local, and ordinary Library queue/retry behavior is unchanged.

### Fix-round-6 verification snapshot

- Direct exact controls **4 passed**; App retry/submission **161 passed**;
  registry/DB **129 passed**; Runner **146 passed, 1 Windows-only skipped**;
  Research association/store/readiness/controller **125 passed**.
- All three inverse mutations failed at their intended guards and were restored.
  Scoped Ruff/format, compilation, and whitespace gates pass. The legacy runner's
  two unrelated lint findings and whole-file format baseline were not rewritten.
  Full pytest was not run by repository policy.

Fix-round-5 review base:

`58dc62050fe6db2ad8c79624eff18d0c6d938f15`

The controller-owned dirty TASK-21508 file remains excluded from the
implementation and commit.

## Fix-round-5 review order

1. `source_association.py`: dispatcher exceptions durably fail the exact retry
   replacement before the catalog receipt can become terminal.
2. `app.py`: worker replacement validation rejects terminal failed catalog
   receipts while preserving successful and concurrently reread replacements.
3. Real-store Local+Server regression: release, raise with sensitive text,
   durable job/receipt settlement, no next queued work, sanitized recovery, and
   both SQLite owners reopened.
4. Split app/DB/registry/association, runner, source-operation/readiness,
   inverse, and static evidence in the Task 4 report.

### Fix-round-5 high-risk invariants

- A terminal failed Research catalog retry never leaves its replacement in an
  unheld queueable state, in memory or after restart.
- Replacement settlement is durable before operation failure settlement; a
  crash between them remains resumable and cannot create terminal orphan work.
- Failed receipts never produce a successful worker result or expose dispatcher
  exception text, secrets, or source paths.
- Local and Server replacements preserve their captured authority; ordinary
  Library retry return/top-up behavior and successful scheduler fencing remain
  unchanged.

### Fix-round-5 verification snapshot

- Direct contract **13 passed**; App/DB/registry/association **311 passed**;
  Runner **145 passed, 1 Windows-only skipped**; source-operation/readiness/
  controller **101 passed**.
- Both inverse mutations failed at their intended invariant and were restored.
  Scoped Ruff/format, compilation, and whitespace gates pass. Full pytest was
  not run by repository policy.

Fix-round-4 review base:

`5cc0292eefe251db7083e36c49bde7df14cf5555`

The controller-owned dirty TASK-21508 file remains excluded from the
implementation and commit.

## Fix-round-4 review order

1. `app.py`: the public Library/Home/provider retry seam detects Research
   operation ownership, validates exact lineage and authority, and delegates to
   the existing catalog scheduler instead of generic requeue/top-up.
2. `library_ingest_state.py` and `library_ingest_canvas.py`: Research ownership
   reaches the row projection and replaces provider-specific recovery actions
   with one honest Research retry action.
3. Direct real-SQLite/store/scheduler tests: Local+Server held replacement,
   durable relink-before-release, restart, missing owner, scheduler exception
   privacy, forged lineage, cross-authority denial, concurrent fence, Home,
   provider delegation, and ordinary legacy compatibility.
4. Split Library/Home/runner/Research/static/inverse/detector evidence and the
   two exact base-proven unrelated UI failures in the Task 4 report.

### Fix-round-4 high-risk invariants

- A Research-owned failure never enters the ordinary unheld requeue/top-up path,
  even when retry starts from Library, Home, or a provider recovery callback.
- The durable operation still names the clicked failed job and the same explicit
  Local/Server authority before the scheduler can replace it.
- Held replacement persistence, operation relink, and release/dispatch remain
  scheduler-owned and fenced; concurrent clicks create one lineage and dispatch.
- Missing scheduler/store, mismatched lineage/authority, and worker exceptions
  mutate nothing and expose no exception text, secret, or local path.
- Ordinary Library retries retain their exact synchronous return and top-up
  contract; Research rows do not advertise unsupported provider overrides.

### Fix-round-4 verification snapshot

- New direct app contract: **11 passed**; combined Library canvas gate: **146
  passed, 1 base-proven case deselected**.
- App/DB/association neighbors: **309 passed**; Home/Library/adapter neighbors:
  **140 passed, 1 base-proven case deselected**; Runner **145 passed, 1
  Windows-only skipped**; Research source/workspace/UI neighbors **194 passed**.
- Scoped Ruff/format, compilation, whitespace, restored inverse controls, and
  one-shot Impeccable detector (`[]`) pass. Full pytest was not run by policy.

Fix-round-3 review base:

`119c55b624f8918840fe9587ae52adfbbdede4a2`

The controller-owned dirty TASK-21508 file remains excluded from the
implementation and commit.

## Fix-round-3 review order

1. `Library_Ingest_Jobs_DB.py` and `library_ingest_jobs.py`: genuine v6→v7
   migration, explicit held-column round trip, queue exclusion, one-way durable
   release, held retry durability, and restart-cap protection.
2. `app.py` and `paste_staging.py`: bounded startup held reconciliation,
   per-row transient isolation, exact operation/job link recovery, durable
   settlement-before-cleanup, and the concurrent startup-sweep hold guard.
3. Research screen intake: ambiguous link-write reread, pending recovery without
   `finally` cleanup, incompatible cancellation failure/success, and dispatch
   only after durable release.
4. Sources region and mounted screen tests: exactly one displayed selected
   association owns Preview across search/type/status/date/folder filters and
   emits the exact stable association ID.
5. Migration/history/rollback, Local+Server, runner, UI, CSS/static, privacy,
   inverse, and detector gates; ADR-078 and Task 4 report evidence.

### Fix-round-3 high-risk invariants

- A crash after job preparation but before operation linking leaves a durable
  queue row that no queue selector or startup runner can dispatch.
- The hold is released on disk only after the exact operation is durably linked
  to that job; an ambiguous write answer is resolved by rereading the receipt.
- A transient or failed cancellation never deletes managed paste staging.
  Startup sweep cannot classify a missing operation as orphaned while a held
  queued job still owns it.
- Local and Server authority remain explicit. An origin mismatch settles the
  held job and never dispatches through the other owner.
- Global desired-selection count neither disables nor retargets a Preview of
  the one displayed selected association.
- Classification diagnostics cannot reveal managed staging paths or source
  representations.

### Fix-round-3 verification snapshot

- Final focused review/inverse matrix: **23 passed**.
- DB/registry **128 passed**; App **146 passed**; Runner **145 passed, 1
  Windows-only skipped**; Research Workspace package **257 passed**.
- Task UI and 15 geometry cases **97 passed**; CSS gates **32 passed**.
- Scoped Ruff lint/format, changed-production compilation, whitespace,
  privacy/ASCII scans, and one-shot Impeccable detector (`[]`) pass.
- Only the accepted Requests dependency warning and pre-existing third-party
  SWIG deprecations remain. Full pytest was not run by repository policy.

Fix-round-2 review base:

`2bcf9c46c70bbd137543bd8f29ec06e8cb4313df`

Expected commit subject:

`fix: close Research source intake ownership gaps`

The controller-owned dirty TASK-21508 file is excluded from this implementation
and commit.

## Fix-round-2 review order

1. `source_urls.py`, add-source modal, and screen boundary: one exact URL gate
   before any Local/Server write, including direct non-modal calls.
2. `library_ingest_jobs.py` and `app.py`: strict durable Research prepare and
   settlement, existing runner dispatch, ordinary caller compatibility, and
   terminal listener idempotence.
3. Screen intake and tests: prepare, persist catalog/job link, dispatch; cancel
   and clean on link failure; fail and retain retry staging on dispatch failure.
4. Sources region and mounted test: displayed stable slots own batch IDs while
   owner desired count stays separate across search/type/status/date/folder
   views.
5. Split regression, inverse, geometry, CSS, privacy/static, and detector
   evidence in the Task 4 report.

### New high-risk invariants

- A hidden selected association cannot enter the exact preview/remove list or
  mutation merely because the owner still desires it.
- URL admission cannot be bypassed through Quick URL, modal batch intake, or a
  direct screen call, and credential-bearing/local paths never reach an
  operation, staging file, registry row, or Server request.
- No Research job dispatch can occur before its qualified operation link is
  durable. A link-write failure leaves a durable cancelled job, no dispatch,
  cleaned managed paste staging, and no canonical catalog mutation.
- Immediate terminal completion observes the already-linked operation once;
  dispatch exceptions cannot overwrite a terminal result or schedule a second
  listener transition.

## Review order

1. `app.py`, screen intake, and app tests: complete durable authority validation
   at admission and delayed dispatch, plus terminal paste cleanup.
2. `paste_staging.py`, `private_paths.py`, and tests: app-owned private files,
   operation binding, terminal cleanup, retry retention, and bounded orphan
   sweep.
3. `controller.py`, `sources_region.py`, and tests: exact-owner reorder bounds,
   explicit page-local semantics, typed Add gate, nested folders, and mounted
   worker containment.
4. Inspector/overlay conflict/modal tests: annotation CRUD/restart, CAS draft
   retention and explicit recovery, Escape/focus restore, and unlink
   confirmation.
5. CSS, all six geometries, and detector output: full-opacity disabled state,
   non-color recovery, stable tree/perf guard, and exact ASCII handles.

## High-risk invariants to challenge

- A captured Server intake cannot route to Local, another profile/principal, or
  the newly visible workspace at either admission or delayed/restarted dispatch.
- Every ingest operation exists durably before the app queue is mutated; one
  operation cannot ambiguously own multiple expanded files.
- Local desired IDs remain catalog Media IDs while Server desired IDs remain
  workspace-source association IDs; selected intent never impersonates
  readiness.
- Remove never deletes canonical Library/Media. Readiness retry only refreshes
  or rechecks; Server reorder never mutates before exact-owner preflight.
- Authority/workspace switches clear old source/folder projections but never
  erase qualified recent receipts. Late source/capability/attach results cannot
  repaint current state.
- Overlay v1 migration invents no organization, stores no paths/URLs/secrets or
  source/Quick-Note bodies, remains bounded, and isolates every qualified key
  axis.
- Every visible enabled control has a mounted event/action test. Unsupported
  controls remain visible, disabled, and carry a reason.
- Source/receipt widgets are composed once, bounded at 25/20, and remain
  reachable at 60x20 without covering the footer or pane mode strip.

## Verification snapshot

- Split focused gates: `638 passed, 1 Windows-only skipped` before the final
  single-selection copy rerun; final affected modal tests add `2 passed`.
- Review-family inverse matrix: `27 passed`.
- Geometry: `15 passed`, including all six requested active Sources sizes.
- Ruff lint, compileall, diff whitespace, ASCII/keybinding/privacy scans,
  deterministic CSS parity/checker, and Impeccable detector: pass.
- One accepted Requests dependency warning in changed gates; full pytest
  intentionally not run.
