# Library Import Media Redesign

**Date:** 2026-07-09

**Status:** Approved design (user, 2026-07-09); implementation planning has not started.

**Parent direction:** `Docs/superpowers/specs/2026-07-04-home-library-redesign-design.md`

**Refines and supersedes:** the Phase L3b interaction, job-registry, runtime, security, and administration details in `Docs/superpowers/specs/2026-07-07-library-l2b-l3-design.md`

**Architecture decision:** `backlog/decisions/013-library-ingest-ownership-and-job-lifecycle.md`

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
- Move source configuration and enrichment administration out of the import
  task flow.
- Retire the legacy Ingest destination only after capability, responsiveness,
  and route-migration gates pass.

## 3. Non-goals

- Persistent local job history across application restarts.
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
|   Import / Export         | |   [ ] example.com/article    | Title [Attention paper]  |
|                           | |       URL | ready             | Author [              ]  |
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

Adding a source performs cheap, non-fetching checks on the UI path:

- trim and classify the value as a path or an `http`/`https` URL;
- normalize it for exact within-batch duplicate detection;
- verify a local path exists, is a readable file, and has a supported type;
- identify an available adapter for the active Library authority;
- set `ready` or `needs input` with readable reason and recovery action.

Network access, content hashing, and database duplicate checks run in workers,
not in the input handler.

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

Switching authority re-runs capability preflight for every staged source but
does not silently submit anything. An adapter can refine a source from `auto`
to a media type during background preflight; the row and inspector show the
result.

## 9. Lifecycle and selected-source inspector

```text
[ ] staged       attention.pdf         document | ready
[!] needs input  missing-notes.txt      local path not found
[..] queued      example.com/article   waiting
[>>] processing  lecture.mp4            transcribing
[ok] added       research.epub          added to Local Library
[x] failed       blocked.example        HTTP 403
```

Text labels are authoritative; glyphs and color only aid scanning.

The lifecycle model supports:

`staged`, `needs_input`, `queued`, `submitting`, `processing`, `succeeded`,
`failed`, `cancel_requested`, and `cancelled`.

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
Detector + cheap validation
      |
      v
App-owned ingest registry
      |
      v
FIFO submission coordinator
      |
      +-- Local adapter: one active local execution in v1
      |
      +-- Server adapter: submit, store server ID, release coordinator
                              |
                              v
                    deduplicated server observer
                    stream or poll with bounded backoff
```

The app-owned registry stores source identity, authority, frozen request
settings, lifecycle state, adapter capability, progress, result reference, and
safe user-facing failure details. All registry mutations are serialized onto
the UI thread. Widgets receive explicit snapshots and emit typed Textual
messages; they do not query databases, call services, or mutate the registry
directly.

The submission coordinator is FIFO, but it does not wait for remote processing
to finish. A server submission records its stable server job or batch ID and
then moves to a deduplicated observer. Retry frequency and backoff are bounded;
observation duration follows the authoritative job lifecycle. Only one observer
owns a given server job or batch. Route changes and canvas recomposition do not
create duplicate streams, pollers, or timers.

Local jobs remain serial in v1. They end if the application exits. Server jobs
continue at the server and are rehydrated from authoritative recent/active job
queries when Server Library is entered again. Reconciliation prefers server
state while preserving the local submission label and safe failure summary.

Home consumes read-only registry snapshots on its normal refresh cadence:

- queued or running work appears under `Running`;
- failures appear under `Needs Attention` and deep-link to Import media with
  the failed row selected;
- Home never cancels, retries, or mutates an ingest job.

## 11. Duplicate policy

Exact normalized duplicates inside the staged batch are rejected immediately
and focus the existing row. Existing Library duplicates require background
preflight:

- URLs compare normalized/canonical URL identity where available;
- files use existing stored identity first and content hashing only in a
  worker when necessary;
- a likely existing item becomes a warning, not a silent skip.

The inspector offers `Open existing` and `Ingest anyway` when the backend can
support both safely. If the backend enforces uniqueness, the recovery copy says
so and only `Open existing` is shown.

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
| App-owned ingest registry | Serialized source/job records and read-only snapshots. |
| Submission coordinator | FIFO claim and dispatch without waiting for server completion. |
| Local ingest adapter | Local file/URL preflight and execution through existing media services. |
| Server ingest adapter | Server submission, cancellation capability, reconciliation, and bounded observation. |
| `Widgets/Library/library_ingest_canvas.py` | Canvas composition and responsive state only. |
| Focused child widgets | Intake, defaults, source list, inspector, and action row. |
| `UI/Screens/library_screen.py` | Rail selection, canvas mounting, message handling, and Library routing only. |
| Home adapter | Maps read-only job snapshots into Running and Needs Attention rows. |

Exact filenames may be refined during planning, but these responsibility
boundaries are binding. Domain widgets cannot reach into parent screens or app
services directly.

## 15. Error handling and recovery

- Validation errors remain attached to staged rows.
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

## 16. Testing and migration gates

### Pure-state tests

- source classification and normalization;
- within-batch duplicate handling;
- 100-source cap;
- batch-default inheritance and override reset;
- settings freeze at queue time;
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

Screenshot QA requires populated empty, staged, needs-input, processing,
succeeded, and failed states at compact, default, and large terminal sizes.
User screenshot approval is required before legacy route retirement.

## 17. ADR check

```text
ADR required: yes
ADR path: backlog/decisions/013-library-ingest-ownership-and-job-lifecycle.md
Reason: The design fixes long-lived destination ownership, local/server
adapter boundaries, job lifetime, cross-screen status flow, security policy,
administrative ownership, and legacy-route retirement.
```

## 18. Relationship to the earlier L3b design

This document preserves the earlier parity bar, real-database smoke test,
stage-label rule, Home integration, UI-thread registry mutation rule, and
legacy retirement gate. It supersedes these earlier details:

- the single stacked form becomes unified intake plus a staged list and
  selected-source inspector;
- the existing single queue runner becomes a FIFO submission coordinator with
  separate bounded server observation;
- URL and file support follow an explicit Local/Server capability matrix;
- multi-source entry and security policy become binding requirements;
- local session lifetime and server rehydration are distinguished;
- server-source and enrichment administration move to exact owners outside
  Import media.
