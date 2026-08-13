# TASK-15513: Ingest Option Local Parity Design

## Purpose

The Library ingest canvas exposes one workflow for Local and Server imports. Its
high-value controls must therefore describe real behavior for the selected
backend. A successful-looking import may not silently discard a visible option.

This design adds four shared controls and one Server-only control:

- Overwrite existing: Local and Server.
- Custom prompt: Local and Server, available only when Analyze after import is on.
- System prompt: Local and Server, available only when Analyze after import is on.
- Generate embeddings: Local and Server.
- Keep original file: Server only. It is absent in Local mode because local
  ingestion already leaves the user's source file untouched; this task does not
  introduce a managed-copy or archive feature.

## Product and interaction design

The canvas remains a dense, keyboard-first workbench. The controls live in the
existing always-reachable generic options panel, renamed from "Plain text & HTML"
to "Import behavior" so its title matches its cross-file scope. Encoding remains
in that panel but is labeled as applying only to plain text and HTML.

Overwrite existing and Generate embeddings are checkboxes. Custom prompt and
System prompt are multiline fields because analysis instructions routinely
exceed one terminal line. Both prompt labels carry the disabled reason when
Analyze after import is off, and disabled controls do not gate Start.

Keep original file is filtered from the capability schema at render time unless
the active ingest backend is Server. It is not rendered disabled in Local mode:
there is no local action the user can take to make that server storage policy
meaningful.

All controls preserve the current panel expansion, reset-to-default, retry, and
config-persistence behavior. Backend changes may structurally recompose the
canvas because the Server-only control genuinely enters or leaves the form; tests
must re-query widgets after that transition.

## Defaults

- Overwrite existing: off, preserving duplicate-safe imports.
- Custom prompt and System prompt: blank.
- Generate embeddings: on. Local semantic indexing is already enabled by default
  under ADR-005; keeping this default avoids silently regressing newly imported
  content out of semantic search. Server imports receive the same explicit value.
- Keep original file: off.

## Data flow

The capability schema remains the source of defaults and field metadata. New
values are stored under `ingest_options["generic"]`, so form persistence, retry
snapshots, and per-job receipts use the existing mechanism.

For Local jobs:

1. The screen snapshots the generic options into `LibraryIngestJob.ingest_options`.
2. `_ingest_job_options` copies prompt values only when analysis is enabled.
3. The writer passes Overwrite existing and Generate embeddings to
   `persist_parsed_media`.
4. Overwrite is forwarded to `MediaDatabase.add_media_with_keywords`.
5. Generate embeddings off enters a context-local suppression scope consumed only
   by the RAG post-ingest hook. The source row still commits normally. The scope is
   reset in `finally`, cannot leak to later jobs, and does not change global RAG
   configuration or concurrent threads.

For Server jobs, `build_server_ingest_kwargs` explicitly projects the five
relevant generic values to the declared ingest-jobs fields. Prompt values are
omitted when analysis is off. Keep original file is only expected from a Server
canvas, but the request builder remains defensive and accepts a persisted Server
snapshot.

## Error handling and honesty

Prompt values are ignored only while their visible controls are disabled because
analysis is off; the labels state that dependency. Values remain in form state so
temporarily switching analysis off and back on does not erase user work.

Local indexing suppression changes only the derived semantic projection. Per
ADR-030, the SQLite media row remains authoritative and its successful commit is
not coupled to vector indexing. Turning Generate embeddings on retains the
existing best-effort indexing failure behavior and guidance.

The request contract guard continues to assert that every Server field sent is
declared by the captured live-server fixture.

## Verification

Tests must prove:

- Local and Server canvases render the four shared controls.
- Only Server renders Keep original file.
- Prompt fields are multiline, disabled with readable reasons when analysis is
  off, and enabled when it is on.
- The form snapshot persists values and retry preserves them.
- Server request kwargs and the final request model contain the exact declared
  fields.
- Local overwrite changes the observable duplicate outcome.
- Local Generate embeddings on invokes the real post-ingest hook seam, while off
  suppresses it for only that write and a following write indexes normally.
- Rendered-frame and compact-viewport checks confirm labels and neighboring
  controls remain visible.

## ADR check

ADR required: no new ADR.

ADR paths:

- `backlog/decisions/005-invest-in-local-rag-mirroring-tldw-server.md`
- `backlog/decisions/030-derived-index-lifecycle-and-atomic-media-migrations.md`

Reason: ADR-005 already requires local indexing at ingestion time and ADR-030
already defines semantic indexes as best-effort derived projections after the
authoritative media commit. This task adds an explicit per-import opt-out and UI
routing without changing storage ownership, schema, conflict policy, or the
source-first lifecycle boundary.
