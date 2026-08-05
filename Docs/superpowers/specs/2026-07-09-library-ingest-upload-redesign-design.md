# Library Import Media Redesign

**Date:** 2026-07-09

**Status:** Approved design (user, 2026-07-09); implementation planning has not started.

**Parent direction:** `Docs/superpowers/specs/2026-07-04-home-library-redesign-design.md`

**Refines and supersedes:** the Phase L3b interaction, job-registry, runtime, security, and administration details in `Docs/superpowers/specs/2026-07-07-library-l2b-l3-design.md`

**Architecture decision:** `backlog/decisions/014-library-ingest-service-authority-and-recovery.md`

**Design anchor:** the shipped Library rail/canvas workbench and ADR-011.

## 1. Purpose

Importing is primarily the act of adding material to the Library. It is not a
separate top-level destination and it is not a server-administration console.
This design replaces the legacy four-tab ingestion screen with a
Library-native `Ingest > Import media` canvas that accepts local files and
URLs, stages them for review, and keeps each source visible through validation,
submission, processing, success, or failure.

The redesign preserves the capable parts of the existing ingestion surface
without preserving its navigation and form structure. The result follows the
same interaction grammar as Console and Library: a persistent rail, a focused
canvas, selectable rows, a selected-item inspector, visible actions, honest
authority labels, and recoverable states.

## 2. Goals

- Make files and URLs equally prominent in one intake flow.
- Let users verify a mixed batch before committing it.
- Apply shared defaults safely while allowing per-source overrides.
- Keep one source row visible through its entire lifecycle.
- Make Local versus Server support explicit before submission.
- Keep work observable after the user leaves the Import media canvas.
- Preserve current multi-file and multi-URL capability.
- Use the existing `media_reading_scope_service` ingest-job seam as the
  lifecycle authority instead of introducing a second job store.
- Move source configuration and enrichment administration out of the import
  task flow.
- Retire the legacy Ingest destination only after capability, responsiveness,
  and route-migration gates pass.

## 3. Non-goals

- A second UI-owned persistence system for job history. Existing Local service
  records and Server job state remain authoritative.
- Parallel local ingestion in v1.
- Recursive folder ingestion. Users may select multiple files; directory sync
  belongs to server-source administration.
- A raw API console for server batches, Web Clipper payloads, or enrichment
  JSON.
- Rebuilding Import / Export. That remains a separate Library rail row and is
  governed by its own capability inventory.
- Changing the underlying media database schema.

## 4. Resolved product decisions

1. Library owns import. The canonical route is
   `Library > Ingest > Import media`.
2. Global commands and legacy links may deep-link to that canvas, but they do
   not create another implementation or destination.
3. One unified source field accepts a local path or URL and reports the
   detected source kind explicitly.
4. Sources are staged first. The user commits ready sources with
   `Start N ready`.
5. Batch defaults apply to staged sources, with optional per-item overrides in
   a selected-source inspector.
6. The staged list becomes the live job list. Rows transition in place instead
   of moving into a separate results log.
7. Wide canvases show list and inspector side by side. Compact canvases show
   one at a time.
8. Server-source configuration and Web Clipper administration do not remain in
   Import media.
9. A bad source does not block ready sources. It remains staged with a clear
   recovery action while the action reports exactly how many items will start.
10. The app owns staged drafts and a read-only display projection. The existing
    media-reading scope service owns submitted job lifecycle state.
11. Staged drafts bind to their destination authority when added. Switching
    Library authority does not silently retarget them.
12. One user confirmation may fan out into several backend batches grouped by
    authority, effective media type, and compatible processing options.
13. Background preflight must complete before a source becomes ready.

## 5. Information architecture and ownership

```text
Library
  Browse
    Media
    Conversations
    Notes
    Collections
    Search / RAG
  Create
    ...
  Ingest
    Import media       canonical user task surface
    Import / Export    separately inventoried Library capability
  Details
    ingestion readiness and links to Settings

Settings
  Ingestion sources   source definitions, policies, sync, clipper defaults
```

Import media owns submissions plus current and recent visible job states.
Settings owns durable server-source configuration, sync policies, archive
source configuration, and Web Clipper defaults. Library Details presents
read-only readiness and routes to the exact Settings controls. Raw capture and
enrichment JSON controls are not part of the primary product UI.

The app-level Import media state owns draft identity, selection, effective
defaults, and a display projection. Submitted lifecycle state comes from
`media_reading_scope_service`, which already exposes Local and Server submit,
detail, list, observe, and cancel operations.

The existing `ingest` route remains a compatibility alias during migration. It
deep-links to the Library canvas. It becomes unreachable only after the parity
and responsiveness gates in section 16 pass.

## 6. Core layout

```text
+ Library rail -------------+ + Import media ------------------------------------------+
| [ Search Library...     ] | | Add local files or URLs to your Library.                |
|                           | |                                                          |
| Browse                  - | | Source                                                   |
|   Media (17)              | | [ Path or URL...                 ] [Browse] [Add]         |
|   Conversations (128)     | | Detected: local file                    [Add multiple]  |
|   Notes (42)              | |                                                          |
|   Collections (5)         | | Batch defaults                                           |
|                           | | Author: none | Keywords: none | Analysis: off | [Edit] |
| Create                  - | +-------------------------------+--------------------------+
|   ...                     | | Sources (3)                   | Source inspector         |
|                           | |                               |                          |
| Ingest                  - | | > [ ] attention.pdf          | attention.pdf            |
| > Import media            | |       file | ready            | Type: document           |
|   Import / Export         | |   [?] example.com/article    | Title [Attention paper]  |
|                           | |       checking               | Author [              ]  |
| Details                 + | |   [!] lecture.mp4            | Keywords [research, ai]  |
|                           | |       needs input             |                          |
|                           | |                               | Inherits batch defaults  |
|                           | |                               | [Reset overrides]        |
|                           | +-------------------------------+--------------------------+
|                           | | 2 ready | 1 needs input [Clear staged] [Start 2 ready] |
+---------------------------+ +----------------------------------------------------------+
```

The primary action remains in a stable action row. When other work is already
active, its label changes to `Queue N ready`. It never claims it will submit a
source that remains in `needs input`.

### Multiple-source entry

The one-line field is the fast path. `Add multiple` expands an inline text area
that accepts one path or URL per line. `Browse` supports multi-file selection.
Both paths feed the same detector and staging model.

```text
Add multiple
+----------------------------------------------------------+
| C:\research\paper.pdf                                  |
| https://example.com/article                              |
| https://example.com/lecture.mp4                          |
+----------------------------------------------------------+
[Add 3 sources]
```

The initial batch cap is 100 staged sources. Larger directory-scale work uses
configured ingestion sources rather than an unbounded canvas list.

## 7. Staging and metadata semantics

Adding a source performs only non-blocking string work on the UI path:

- trim and classify the value as a path or an `http`/`https` URL;
- normalize it for exact within-draft duplicate detection;
- bind the draft to the active Library authority;
- create a stable draft row in `checking` state.

Background preflight performs all file-system, network-policy, service,
capability, and database work:

- verify a local path exists, resolves safely, is a readable file, and has a
  supported type;
- resolve effective media type and processing capability;
- validate URL network policy without fetching content;
- check existing Library identity and duplicates;
- set `ready`, `needs input`, or `needs review` with readable reason and
  recovery actions.

Network-drive access, DNS, content hashing, service calls, and database checks
run in workers. `Start N ready` excludes `checking`, `needs input`, and
`needs review`, closing the confirmation race.

Batch defaults contain only fields that sensibly apply across mixed source
types:

- author;
- keywords;
- analysis enabled or disabled;
- chunking enabled or disabled plus common chunking defaults.

Title is always derived per source and can be overridden only in the inspector.
Media-specific settings such as PDF parser, transcription language,
timestamps, diarization, crawl mode, or media type appear only for a selected
compatible source.

Defaults remain live for staged fields that have not been overridden. An
override stores only the changed field. Queuing snapshots the effective
settings and makes them read-only. `Retry` reuses the frozen snapshot;
`Edit and retry` creates a new staged copy and leaves the failed record as
history until cleared.

## 8. Source and runtime capability matrix

The active Library authority determines the destination. The user does not
accidentally create a Local record while looking at Server Library without a
visible explanation.

| Destination | Local file | URL | Required behavior |
| --- | --- | --- | --- |
| Local Library | Supported | Supported | Use the local media reading/ingest service. |
| Server Library | Capability-gated | Supported | Use server upload only when an upload adapter is available; submit URLs as server jobs. |

A source unsupported for the current destination remains staged as
`needs input`. Recovery names the boundary, for example:

```text
Local file upload is unavailable for Server Library.
[Switch Library to Local] [Remove]
```

Local and Server keep separate draft sets. Switching Library authority reveals
that authority's drafts and does not mutate or re-preflight the other set.
`Move to Local` or `Move to Server` is an explicit per-source or selected-source
action that changes authority and then restarts preflight. Active work always
retains its submission authority. When work is active under the other
authority, the canvas shows a quiet count and a `View` action rather than
hiding it.

An adapter can refine a source from `auto` to a media type during background
preflight; the row and inspector show the result.

## 9. Lifecycle and selected-source inspector

```text
[ ] staged       attention.pdf         document | ready
[?] checking     example.com/article   running preflight
[!] needs input  missing-notes.txt      local path not found
[!] needs review duplicate.pdf          possible Library match
[..] queued      example.com/article   waiting
[>>] processing  lecture.mp4            transcribing
[ok] added       research.epub          added to Local Library
[x] failed       blocked.example        HTTP 403
```

Text labels are authoritative; glyphs and color only aid scanning.

The lifecycle model supports:

`checking`, `staged`, `needs_input`, `needs_review`, `queued`, `submitting`,
`processing`, `succeeded`, `failed`, `interrupted`, `cancel_requested`, and
`cancelled`.

Adapters may omit states they cannot report. Stage labels such as extracting,
transcribing, chunking, indexing, or waiting on server are shown when real.
Percentages appear only when backed by real progress.

Inspector actions depend on state and adapter capability:

- staged: edit settings, reset overrides, remove;
- queued: view frozen settings, remove only if the coordinator has not claimed
  the job;
- processing: show authority, stage, and real progress; show Stop only when the
  adapter supports cancellation;
- succeeded: `Open in Library`, then `Use in Console` where the existing
  Library handoff supports the returned record;
- failed: `Retry`, `Edit and retry`, and `Remove`.

`Open in Library` is enabled only after the returned authority and record ID
resolve to a readable Library item.

## 10. Job architecture

```text
Unified intake
      |
      v
Authority-bound draft store
      |
      v
Background preflight
      |
      v
Submission planner
partition by authority + media type + processing profile
      |
      v
media_reading_scope_service
      |
      +-- Local service: persistent job rows + synchronous execution
      |
      +-- Server service: remote batch/job IDs + continued execution
      |
      v
App-owned read-only job projection
      |
      +-- Library rows and inspector
      +-- Home Running / Needs Attention
```

The app-owned draft store contains stable draft identity, bound authority,
defaults and overrides, preflight state, selection, and mappings to returned
backend batch/job IDs. Its job projection is a display cache, never the
lifecycle authority. Submitted state reconciles from
`media_reading_scope_service`. All draft and projection mutations are
serialized onto the UI thread. Widgets receive explicit snapshots and emit
typed Textual messages; they do not query databases, call services, or mutate
the store directly.

One user confirmation is planned into submission groups keyed by authority,
effective media type, and compatible frozen processing options. A heterogeneous
eight-source confirmation can therefore create several backend batches while
remaining one visible user action. Each source row records its returned batch
and job identity independently.

Local submission currently executes synchronously while persisting
`local_ingestion_jobs`. It runs in a worker, with one active Local submission
group in v1. Draft projections show later groups as queued until claimed. After
backend identity exists, Local service records are authoritative. On startup or
Local Library entry, stale Local `queued` or `running` records from an
interrupted process are reconciled to `interrupted` with explicit Retry; they
must never remain falsely running.

Server submission records stable server job and batch IDs, then releases the
submission path. A deduplicated observer follows authoritative server events or
polling. Retry frequency and backoff are bounded; observation duration follows
the server job lifecycle. Only one observer owns a given server job or batch.
Route changes and canvas recomposition do not create duplicate streams,
pollers, or timers.

Automatic Server recovery uses a bounded receipt ledger under
`library.ingest.server_receipts`. One receipt represents a submitted backend
batch and carries its batch/job IDs, authority, safe display label, and
submission timestamp. All nonterminal receipts are retained; terminal history
is capped at the 20 most recent receipts. The ledger stores no source
credentials, raw URL query values, or job payloads. On Server Library entry,
receipts are reconciled through job detail/list APIs. An unscoped recent-event
stream may supplement recovery but is not treated as durable history.

Home consumes read-only job projections on its normal refresh cadence:

- queued or running work appears under `Running`;
- failures appear under `Needs Attention` and deep-link to Import media with
  the failed row selected;
- Home never cancels, retries, or mutates an ingest job.

## 11. Duplicate policy

Exact normalized duplicates inside the same authority's draft set are rejected
immediately and focus the existing row. Existing Library duplicates require
background preflight:

- URLs compare normalized/canonical URL identity where available;
- files use existing stored identity first and content hashing only in a
  worker when necessary;
- a likely existing item becomes a warning, not a silent skip.

An existing-item warning remains `needs review` and is excluded from Start
counts until the user chooses a disposition. The inspector offers
`Open existing` and `Ingest anyway` when the backend can support both safely.
If the backend enforces uniqueness, the recovery copy says so and only
`Open existing` is shown.

## 12. Security and privacy boundaries

The existing basic URL syntax validator is not a fetch authorization check.
Every adapter must enforce its own boundary before network or file access.

Local paths:

- expand and resolve paths through project path-validation helpers;
- require an existing, readable, non-directory file;
- re-check supported type and size before execution;
- handle symlinks deliberately and never infer a directory crawl;
- redact sensitive path content from telemetry where it is not necessary.

URLs:

- accept only `http` and `https`;
- reject embedded credentials;
- normalize host and port before dispatch;
- revalidate every redirect and resolved address;
- default to public-network destinations;
- allow loopback/private-network fetches only through an explicit Local
  advanced override with warning;
- never offer that client override for Server jobs unless the server reports a
  corresponding policy capability;
- enforce request timeout, redirect limit, and response-size limit;
- redact credentials, tokens, and sensitive query values from logs and errors.

The server remains authoritative for its own SSRF and fetch policy. A client
preflight cannot weaken server rejection.

## 13. Responsive, keyboard, and accessibility behavior

Compact behavior is based on available canvas width after the Library rail,
not on total terminal width.

```text
Wide canvas
+ Sources ----------------------+ Inspector -------------------------+
| rows                           | selected-source details             |
+--------------------------------+-------------------------------------+

Compact canvas
+ Sources -----------------------------------------------------------+
| rows                                                              |
+-------------------------------------------------------------------+

Enter on a row
+ Source inspector -------------------------------------------------+
| [Back to sources]                                                  |
| selected-source details                                            |
+-------------------------------------------------------------------+
```

- `F6` follows the shared pane cycle: intake, source list, inspector, action
  row.
- `Enter` adds from intake or opens the selected row from the list.
- `Escape` returns from the compact inspector or collapses expanded defaults
  and multi-source entry.
- Contextual actions appear in the existing footer-hint system and command
  palette, but the primary task remains visibly completable.
- Focus uses the established accent outline without changing dimensions.
- Important states always include text; color is never the only signal.
- Progress ticks target the affected row and inspector. Full canvas recompose
  is reserved for structural transitions such as adding or removing rows.

No new tab strip, nested card grid, decorative border, permanent log panel, or
fabricated progress animation is introduced.

## 14. Component and service boundaries

| Unit | Responsibility |
| --- | --- |
| `Library/library_ingest_state.py` | Pure source, defaults, override, capability, lifecycle, and display-state models. |
| App-owned draft store | Authority-bound staged sources, defaults, overrides, preflight, selection, and backend-ID mappings. |
| App-owned job projection | Read-only reconciliation of authoritative service job records for Library and Home. |
| Submission planner | Partition ready drafts by authority, media type, and processing profile. |
| `media_reading_scope_service` | Canonical Local/Server submit, detail, list, observe, and cancel seam. |
| Local execution coordinator | One worker-owned Local submission group at a time; stale-job reconciliation. |
| Server observer | Receipt-backed reconciliation, cancellation capability, and deduplicated observation. |
| `Widgets/Library/library_ingest_canvas.py` | Canvas composition and responsive state only. |
| Focused child widgets | Intake, defaults, source list, inspector, and action row. |
| `UI/Screens/library_screen.py` | Rail selection, canvas mounting, message handling, and Library routing only. |
| Home adapter | Maps read-only job snapshots into Running and Needs Attention rows. |

Exact filenames may be refined during planning, but these responsibility
boundaries are binding. Domain widgets cannot reach into parent screens or app
services directly.

## 15. Error handling and recovery

- Validation errors remain attached to staged rows.
- Preflight failure becomes `needs input` or `needs review`; it never leaves a
  row indefinitely in `checking`.
- Worker and adapter failures become safe failure summaries in the inspector;
  raw tracebacks and secrets do not enter user-facing copy.
- A failed source does not erase successful results from the same batch.
- Server observation loss becomes `status unavailable | reconnecting`, not a
  false job failure. Bounded retry with backoff precedes a manual Refresh
  action.
- Cancellation uses `cancel requested` until the adapter confirms cancellation.
- If cancellation is unsupported, the action is absent and the inspector says
  the job will continue.
- Switching Library authority never silently moves, resubmits, or cancels
  active work.
- Stale Local jobs become `interrupted`, not failed or still running.
- Missing or expired Server receipts produce honest recovery copy and a manual
  batch-ID recovery path; the UI does not invent a job state.

## 16. Testing and migration gates

### Pure-state tests

- source classification and normalization;
- within-authority draft duplicate handling;
- 100-source cap;
- checking, needs-input, needs-review, and interrupted transitions;
- batch-default inheritance and override reset;
- settings freeze at queue time;
- authority-bound draft separation and explicit moves;
- heterogeneous submission partitioning;
- capability-matrix readiness and recovery copy;
- every lifecycle transition, including invalid transitions;
- ready and needs-input action counts;
- compact versus wide display-state decisions.

### Adapter and integration tests

- a real in-memory MediaDatabase and a small text fixture through the Local
  adapter;
- local URL extraction through an injected deterministic scraper;
- server URL submission, returned IDs, event reconciliation, observer restart,
  cancellation, and service-unavailable behavior;
- Local stale-job interruption recovery;
- bounded Server receipt persistence, redaction, and rehydration;
- unsupported Server local-file upload recovery;
- URL policy tests for schemes, credentials, loopback/private addresses,
  redirects, size, timeout, and redaction;
- existing Library duplicate detection and direct open-by-ID routing.

### Widget and screen tests

- one-line and multi-line source entry;
- multi-file browse selection;
- selection and inspector editing;
- `Start N ready` and `Queue N ready` semantics;
- queued-setting immutability;
- retry, edit-and-retry, remove, cancel, Open in Library, and Use in Console;
- focus order and footer hints;
- compact, default, and large geometry without horizontal overflow;
- targeted progress updates without full-canvas mount churn.

### Parity and responsiveness gates

Before the legacy screen becomes unreachable, inventory and disposition every
capability currently under Local Files, Server Sources, Server Jobs, Web
Clipper, and Results. Each capability must be present in Import media, moved to
its approved owner, or explicitly retired by this design.

Capture event-loop heartbeat, worker backlog, timer count, mount/remove churn,
route-switch soak, and server-observer reconnect evidence under ADR-011. The
replacement cannot retire the legacy path if it increases stalls, leaks
workers/timers, duplicates server observers, or loses active-job visibility.

Screenshot QA requires populated empty, checking, staged, needs-input,
needs-review, processing, interrupted, succeeded, and failed states at compact,
default, and large terminal sizes. User screenshot approval is required before
legacy route retirement.

## 17. Delivery phases

The design is intentionally delivered as four mergeable phases. No phase adds
ingest runtime logic directly to the already-large `library_screen.py`.

### I1: Ingest contracts and safety foundation

- pure draft, preflight, lifecycle, partitioning, and display-projection state;
- authority-bound draft sets;
- `media_reading_scope_service` adapter contract and real Local/Server seam
  tests;
- URL/path policy and safe error/redaction contract;
- stale Local reconciliation and bounded Server receipt model;
- no route or legacy-screen change.

### I2: Local Import media canvas

- `ingest-import-media` becomes a Library canvas for Local scope;
- unified one-line and multi-line intake plus multi-file browse;
- batch defaults, selected-source inspector, responsive behavior, and action
  counts;
- Local file/URL submission, lifecycle projection, Retry, Open in Library, and
  compact/default/large screenshot approval;
- legacy Ingest route remains reachable for unported Server/admin capability.

### I3: Server jobs and Home visibility

- Server file/URL capability preflight and partitioned submission;
- receipt-backed recovery, deduplicated observation, cancellation, and manual
  batch-ID recovery;
- active-other-authority visibility;
- read-only Home Running and Needs Attention projection;
- Server-state screenshot and route-switch/observer soak approval.

### I4: Administration migration and legacy retirement

- durable ingestion-source and Web Clipper-default administration in Settings;
- read-only Library Details readiness and exact Settings routing;
- disposition every remaining Local Files, Server Sources, Server Jobs, Web
  Clipper, and Results capability;
- compatibility alias audit, responsiveness gate, final screenshot approval,
  and only then legacy Ingest deregistration.

## 18. Parallel RAG/Search PR coordination

A parallel RAG/Search screen PR is expected to land before ingest
implementation planning. It is a sequencing dependency because it may change
Library canvas routing, `library_screen.py`, focus/footer behavior, shared
widgets, and agentic-terminal TCSS.

Implementation planning must begin from the merged post-PR branch, not from the
2026-07-09 pre-merge tree inspected during this design. Before writing the
file-level plan:

1. re-read the merged RAG/Search spec, ADRs, and implementation notes;
2. inventory its changes to Library canvas composition, shell state, focus
   order, worker ownership, command/footer hints, tests, and TCSS;
3. adopt any newer decomposition or shared primitives as the source of truth;
4. keep ingest domain state and services independent from RAG/Search domain
   state;
5. re-run focused Library and RAG/Search tests before establishing the ingest
   baseline;
6. amend this design if the merged PR changes Library ownership or route
   contracts rather than papering over the conflict in implementation.

The ingest phases must not modify RAG/Search internals unless a shared contract
requires a separately reviewed compatibility change.

## 19. ADR check

```text
ADR required: yes
ADR path: backlog/decisions/014-library-ingest-service-authority-and-recovery.md
Reason: The design fixes long-lived destination ownership, authoritative
service and projection boundaries, draft and receipt persistence, local/server
job recovery, cross-screen status flow, security policy, administrative
ownership, and legacy-route retirement.
```

## 20. Relationship to the earlier L3b design

This document preserves the earlier parity bar, real-database smoke test,
stage-label rule, Home integration, UI-thread registry mutation rule, and
legacy retirement gate. It supersedes these earlier details:

- the single stacked form becomes unified intake plus a staged list and
  selected-source inspector;
- the existing single queue runner becomes heterogeneous submission planning
  with separate server observation;
- the existing media-reading scope service remains authoritative while the app
  owns drafts and a read-only projection;
- URL and file support follow an explicit Local/Server capability matrix;
- multi-source entry and security policy become binding requirements;
- Local stale-job recovery and receipt-backed Server rehydration are explicit;
- delivery is split into four mergeable phases and planning is sequenced after
  the parallel RAG/Search PR;
- server-source and enrichment administration move to exact owners outside
  Import media.
