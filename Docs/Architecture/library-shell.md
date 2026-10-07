# The Library shell

This document describes the Library destination: the rail/canvas navigation model, the adaptive reader shell, media browsing and read-it-later, the prompts and skills surfaces, collections, and the durable ingest job queue with its parse/write stages.

## Authoritative files

| Concern | File | Key symbols |
| --- | --- | --- |
| Screen | `UI/Screens/library_screen.py` | `LibraryScreen(BaseAppScreen)`; `apply_navigation_context`, `_build_library_entry_active_child` (composes exactly one canvas child) |
| Rail state | `Library/library_shell_state.py` | pure (Textual-free) `build_library_shell_state`; the `LIBRARY_ROW_*` row ids (browse-conversations/-media/-notes/-prompts/-skills/-search/-collections, create-*, ingest-*) |
| Nav controller | `UI/Library_Modules/library_navigation_controller.py`, `screen_constants.py` | route admission, pending character-repair lifecycle, `LIBRARY_NAV_MODE_TO_ROW_ID` (the 60-line comment above it is the authoritative live-vs-forward-compat mode list) |
| Canvases | `Widgets/Library/*` | notes/skills/prompts canvases, `LibraryMediaViewer`, `LibraryBrowseReaderShell`, search/RAG panel, ingest canvas, collections capture/reader |
| Canvas state | `Library/library_*_state.py` | pure display contracts per canvas (media list/viewer/reader/ingest/prompts/skills/conversations/notes) |
| Media seam | `Media/local_media_reading_service.py`, `Media/media_reading_scope_service.py` | reading progress; read-it-later lifecycle (ADR-014 scope authority) |
| Collections | `DB/Library_Collections_DB.py` (schema v4), `Library/library_collections_service.py` | collections + capture tables (items/tags/highlights/saved searches/offline files), review sets; legacy generic Collections tables are recovery-only |
| Ingest jobs | `DB/Library_Ingest_Jobs_DB.py` (schema v7), `Library/library_ingest_jobs.py`, `app.py` `LibraryIngestQueueMixin` | durable job rows, in-memory registry, `plan_restore` |

## Navigation model (dataflow)

1. Another destination posts `NavigateToScreen("library", context)` with keys from `Constants.py`: `mode`, `conversation_id`, `note_id`, `notes_create`, `open_source_type`/`open_source_id`, `ingest_media`, character repair/inspection/browse contexts.
2. The controller rejects context application while a prompts mutation is in flight, bumps a generation counter (stale async applies drop), and gives character contexts dedicated admission branches.
3. Row resolution starts from the mode→row map, then applies priority overrides (conversation_id → conversations; `notes_create` → create-note; ingest → ingest row; any note_id → browse-notes; a validated open-source pair → its browse row). Unknown modes resolve to no-op — quiet degrade by contract.
4. Leaving side-effects: media browse invalidates unless the target is media; collections capture unmounts unless the target is collections. When mounted, context application runs as a worker that first flushes pending editor saves — a save-guard refusal aborts the switch and keeps the current route.
5. Row selection → `build_library_shell_state` → one canvas child (conversations / media / notes / prompts / skills / export / search / ingest / landing). **Collections is the exception** — it changes whole-route topology (rail scopes + Items + Work panes) and goes through the central recompose path instead of the one-child seam.

## Adaptive reader

`LibraryBrowseReaderShell` is **one resident shell** shared by the Media and Notes routes (route-neutral id, marker classes, single writer `apply_route`) — the decomposition spec records that per-route shells previously caused 177 mounts per media switch, 55 of them pure rebuild waste. Reader modes: read / analysis / highlights / info. Pane visibility and widths persist to `[library.reader]` and per-destination `[library.<dest>_reader]` sections.

## Media, read-it-later, progress

Media canvas views: list / viewer / trash through one shared builder. Read-it-later is a `MediaReadItLaterState` row in the media DB (Home deep-links via the browse-subview context); reading progress is upserted per item and surfaced through the reading service. Saving a single reading item deliberately restores a trashed duplicate (an explicit user decision); the **bulk** import path stays non-restoring.

## Prompts and skills surfaces

- Prompts canvas consumes `PromptsDatabase` row shapes with typed conflicts (`ExpectedVersionConflictError`, `PromptNameConflictError`) classified at save; create vs browse is an editor-view sentinel, not a separate canvas. Prompt insertion into the Console is covered in [chat-pipeline.md](./chat-pipeline.md).
- Skills canvas consumes `LocalSkillsService` envelopes (available/blocked + trust fields); import staging lives on the screen. Skills was retired as a destination into Library (ADR-015; route alias `skills` → `library`).

## Ingest job queue (dataflow)

1. **Submit** (UI thread only): directory sources expand to one job per file; a missing media DB fails immediately with exact copy; an origin mismatch raises (fail-closed Research authority). Job enters `QUEUED`.
2. **Parse stage** (UI thread coordination): a lazily created spawn-context multiprocessing pool; ebook batches get one-worker generations. Parse failures classify in-worker and go straight to FAILED; successes stash payloads.
3. **Write stage**: exactly one background writer thread claims the oldest payload-ready job via a **single synchronous UI-thread call** (`call_from_thread`) that atomically transitions PARSING → WRITING — one WRITING job ever, honoring SQLite's single-writer nature. The writer persists through `persist_parsed_media` (the pipeline documented in [media-ingestion.md](./media-ingestion.md)).
4. **Persistence**: the registry writes through to `Library_IngestJobsDB` (upsert); `research_source_operation_id` is immutable once persisted; a durable `dispatch_held` release cannot be re-held.
5. **Restore on boot**: a worker (never blocking boot) loads persisted jobs (capped at 500), plans the restore — in-flight jobs become retryable FAILED "Interrupted by app restart"; held Research rows are exempt from both interruption and pruning — applies it, then attaches the store **before** merging so jobs submitted in the mount window persist too. Research-held jobs reconcile afterwards.

Quit semantics (accepted v1 limits): a WRITING job finishes its DB write; PARSING/QUEUED jobs are lost and nothing resumes them next launch.

## Config keys

`[library]`: `ingest_directory_scan_limit` (1000), `ingest_url_preflight_probe` (**false** by default — a pasted link never auto-probes a host), `ingest_options`, `ingest_parse_workers` (`min(3, cpu-1)`), `ingest_heavy_lane_max_workers` (1). `[library.reader]` (+ legacy `[library.media_reader]` fallback) and per-destination `[library.<dest>_reader]` with env overrides. DB path overrides: `library_collections_db_path`, `library_ingest_jobs_db_path`.

## Boundaries

- Library owns user-facing import UI + canvases; **job lifecycle authority is the media reading scope service** (ADR-014); Settings owns durable ingestion-source administration (ADR-013/014).
- The job registry is UI-thread-only with no internal locking; background threads marshal through `call_from_thread`; returned jobs are copies.
- Parse workers never touch the media DB; the writer never touches parse workers.
- The RAG search surface is documented in [rag.md](./rag.md) — this shell only hosts the canvas.

## Failure behaviors

| Case | Behavior |
| --- | --- |
| Corrupt ingest store | Registry starts empty and store-less with a warning; boot continues |
| Restore-window race | Store attaches before merge so window-submitted jobs displace stale rows |
| Collections count read failure | Renders "(—)" with a deadline sentence, never a fake 0 |
| Save guard refuses on nav switch | Switch aborts; current route kept |
| Same-key Escape bindings | Resolved in declaration order with disjoint `check_action` gates — position is the contract (10+ chained bindings) |

## Governing decisions and docs

ADR-013 (`013-library-ingest-ownership-and-job-lifecycle.md`, superseded), ADR-014 (`014-library-ingest-service-authority-and-recovery.md`), ADR-015 (destination IA; Skills folded into Library). Key specs: `2026-08-13-library-compose-once-design.md`, `2026-08-24-library-destinations-adaptive-reader-design.md`, `2026-09-01-library-screen-decomposition-design.md`, `2026-07-10-library-f3-parallel-parse-design.md`, `2026-07-12-library-persistent-job-history-design.md` (all under `Docs/superpowers/specs/`). User guide: `Docs/User_Guide/library.md` + `Docs/User_Guide/library/`.

## Verified gotchas

1. **SKIPPED jobs cannot persist**: the store's CHECK constraint omits `skipped`, so skip outcomes are in-memory only and vanish on restart (the best-effort persist swallows the rejection).
2. The writer-claim must stay one synchronous UI-thread call — a two-call exit path previously stranded jobs behind a stale runner flag.
3. v2→v3 of the jobs store is a table **rebuild** (SQLite can't alter a CHECK) with an explicit column list.
4. The ingest store runs with `isolation_level = None`; `transaction()`'s explicit BEGIN is the only transaction owner.
