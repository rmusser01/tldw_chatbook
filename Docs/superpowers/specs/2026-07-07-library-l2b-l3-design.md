# Library L2b + L3 Design — Notes absorption; Search, Ingest, Create

**Date:** 2026-07-07
**Status:** Approved direction (user, 2026-07-07); each phase still requires its own screenshot approval gate before merge.
**Parent spec:** `Docs/superpowers/specs/2026-07-04-home-library-redesign-design.md` (§2, §5, §6). This document refines its L2/L3 phases against what actually shipped through L2a (PR #585) and what already exists on `dev`.
**Design anchor:** the shipped Library shell + media viewer (PRs #582, #585) — their canvas grammar, service-offload discipline, and QA gates are the binding house pattern.

## Context: what already exists (verified 2026-07-07)

- **Notes** is a Console-styled three-pane workbench (`UI/Screens/notes_screen.py`, ~3,300 lines: navigator / editor / inspector + Notes/Sync/Templates mode strip) with pure scope models and a `notes_scope_service` seam. It is its own screen (TAB_NOTES); the Library rail routes to it.
- **Search/RAG** already executes retrieval in-Library (`Widgets/Library/library_search_rag_panel.py`, `Library/library_rag_state.py`, `library_rag_service` protocol, exclusive-worker execution) but is parked as a legacy "mode" canvas, and the rail-top search box only filters conversations.
- **Ingest:** TAB_INGEST renders `MediaIngestWindowRebuilt` (working). `Widgets/NewIngest/` (SmartFileDropZone → UnifiedProcessor → ProcessingDashboard) exists but is **routed nowhere and therefore unproven**. Library's "Import / Export" row renders placeholder copy.
- **Create:** study/flashcards/quizzes rows are handoff buttons to the Study screen; real DBs and `get_due_flashcards` exist; no counts on the rail; no Home mirror.
- **Home:** generic active-work section exists; no ingest feed, no due-count mirror.

## Resolved decisions (user, 2026-07-07)

1. **Notes: rebuild in Library grammar** (list rows → in-canvas editor), not pane reuse, not partial absorb. The standalone Notes tab deprecates at parity.
2. **Parity line is split:** L2b.1 ships the core canvas; L2b.2 ships Sync + Templates management; the tab deprecates after L2b.2.
3. **Ingest: rebuild in Library grammar**; the standalone Ingest screen deprecates at parity with what TAB_INGEST ships today.
4. **Search scope: promote + wire + history.** First-class canvas, rail-top search feeds it, lightweight query history. Saved searches deferred.
5. **Create/Home scope: counts + hooks, keep handoffs.** Live counts on rail rows; due>0 mirrors to Home Needs Attention; ingest jobs feed Home Running (L3b). The Study screen is NOT absorbed.
6. **Packaging: four phases** — L2b.1 → L2b.2 → L3a → L3b — each independently mergeable with its own plan, worktree, and screenshot approval gate.
7. **Collections item-browsing** (list/add/remove members inside a collection) is a tracked follow-up, not part of any of these phases.

## Global constraints (binding, all phases)

- Canvas grammar: stacked full-width render-verified widgets (Static / Input / TextArea / Markdown / Collapsible / Button / VerticalScroll). No `Select`; cycling buttons instead. Never a `Horizontal` mixing a `1fr` sibling with fixed-width children. **Render smoke-check in the served TUI before building on any unproven widget** (Checkbox, file-picker modal).
- Width rule (L2a lesson): a canvas child that fills the canvas uses `1fr`, never its own `Nfr`; long text bodies get `width: 100%` and the container `overflow-x: hidden`.
- All service calls through `_run_library_service_call(..., isolate_in_worker=True)`; services accessed via `getattr(app, "<service>", None)` with quiet degrade.
- Mutation→refetch→recompose — **amended for editor surfaces:** autosave persists silently and never recomposes (targeted `update()` for status; version bumped in memory from the service response); refetch+recompose only on transitions (Back, row switch, rail re-entry).
- State reset discipline on every canvas entry/exit path (L2a `_select_library_rail_row` lesson).
- Real-backend regression tests for every mutation; test fakes must mirror the real service's allowlists/method seams (L2a false-green lesson).
- Input sanitization on every persisted field (`input_validation` helpers); local paths through `path_validation`; URLs restricted to http/https.
- CSS in `css/components/_agentic_terminal.tcss` → `./build_css.sh` → commit both.
- Fetch caps: L2b.1 replaces the shared `LIBRARY_SOURCE_PAGE_SIZE = 5` with per-source page sizes; rail badges always show true DB counts; capped counts use the `(N+)` grammar.
- Live textual-serve QA at 2050×1240 with populated seeded states; explicit user screenshot approval per phase before merge.
- Deprecations: the phase removes routing/registration (tab unreachable); screen-file deletion folds into the tracked Library dead-code sweep.

---

## Phase L2b.1 — Notes core canvas

**Rail:** `browse-notes` flips `screen`→`canvas` (target `notes`); count becomes a real known count. `create-note` flips to canvas: creates a blank note via the service (persists immediately — today's behavior; accepted trade-off: an untouched blank leaves an "Untitled" note, Delete is one action away) and lands in the editor.

**List mode** (mirrors the media canvas): header `Notes (N)`; one full-width filter input that **queries the notes search seam on Enter** (never client-side filtering of a capped snapshot); `sort: Newest ▸` cycling button; rows = title + muted relative age. One scoped list from `notes_scope_service.list_notes` for the active runtime scope (no Local/Server/Workspaces triple-grouping in-canvas — the `Library | Local` header governs scope).

**Editor mode** (selecting a row): `‹ Back to list` → title `Input` → body `TextArea` (bounded: min ~12 / max ~20 rows, internal scroll — never `1fr` inside the scrolling canvas) → keywords `Input` → muted meta line (`Created · Modified · vN · sync state · word count · autosave status`) → actions toolbar `Save · Preview · Use in Console · Export .md · Export .txt · Copy` with `Delete` separated at the far end behind an inline confirm.

- **Autosave always on** (the toggle is dropped — deliberate simplification); debounced worker; **never recomposes**; status via targeted `update()`.
- **Flush-on-exit:** every path out of editor mode (Back, row switch, rail switch) flushes pending changes first.
- **Conflict policy:** version conflict during autosave → pause autosave, keep editor content untouched, quiet line `changed elsewhere · [Overwrite] [Reload]`. Never silently reload an editor.
- **Preview** toggles the TextArea to a `Markdown` widget.
- **Create from Template:** in-canvas template *rows* (list grammar, not a cycling button) reading the existing template service read-only.
- **Dropped:** the emoji-picker button (palette survives). **Moved to L2b.2:** Import Note.

**Units:** `Library/library_notes_state.py` (pure), `Widgets/Library/library_notes_canvas.py`, screen orchestration only. Optimistic-version handling mirrors the notes service contract.

**Tests:** pure-state units (list/filter/sort/editor/meta-line builders); real-ChaChaNotes-DB tests for create/update/delete/keywords (+ conflict path); pilots for every action and every exit-flush path; hardened fake mirroring `notes_scope_service`.

## Phase L2b.2 — Sync + Templates + Notes deprecation

- **Templates management:** `Note templates (n)` row joins the Create section → canvas with template list → view/edit/create/delete (list→detail grammar).
- **Sync:** an action inside the Notes canvas (list-header area) → in-canvas sync view with Back. Capability parity with today's Sync pane (profiles, run, status), rebuilt stacked. **Import Note** lands here.
- **Deprecation:** remove TAB_NOTES registration + nav entry; re-point every internal TAB_NOTES route (grep-audited: Home recents, palette, any deep links) to the Library notes canvas. `notes_screen.py` + panes + their tests are deleted by the dead-code sweep, not this PR. Gate: user screenshot approval of full parity before the deregistration commit.
- **Follow-up logged:** background sync failures should later mirror to Home Needs Attention via the L3a hook pattern.

## Phase L3a — Search promotion + Create counts + Home due-mirror

**Search:**
- `browse-search` flips `mode`→`canvas`; `browse-collections` gets the same vestigial-flag flip.
- **Smoke first:** capture the *entire current panel* in the served TUI and inventory what renders (it predates all L2a rendering discoveries). Checkbox scopes fall back to `✓/○` toggle buttons if needed; any Select-like control becomes a cycling button.
- **Single query truth:** one `library.search.query` state field; the rail-top input and the canvas input both seed from it and write it on submit. After a rail submit, focus returns to the rail input.
- **Rail-top rewire:** rail submit selects the Search canvas, prefills, and executes immediately in fast `search` mode. Placeholder becomes `Search Library…` unconditionally. A visible status line (`searching · notes, media…`) is required — execution takes seconds; the exclusive worker already handles re-submits.
- **Conversations keep filtering:** the conversations canvas gains its own in-canvas filter input (Enter-to-apply) over the raised snapshot — same semantics as today, honestly documented; service-backed FTS filtering is a tracked follow-up. Pilots re-anchored.
- **History:** last 10 submitted queries — exact-match dedupe, entries truncated to 200 chars — persisted under `library.search.history` via `save_setting_to_cli_config`. Rendered in an always-available `Collapsible("Recent searches")` (expanded when idle, collapsed when results show); click re-runs. Pure list-management function + units.
- **Result → Open:** ships per source type only where the result row carries a resolvable *parent* record id (verify during planning — media results may carry chunk ids). Open routes by id **straight to the detail surface** (media viewer fetch, notes editor fetch, conversations deep-link) — never via list selection, so caps are irrelevant. Types without a resolvable id: no Open action, quiet.
- **RAG-mode degrade:** planning verifies behavior with embeddings/providers absent; requirement: `search` mode always works, `rag` mode unavailable → quiet line + setup routing (Console convention).

**Shared unit (built here, reused by L3b):** an internal `open library item by (source_type, record_id)` route.

**Create counts:** rail rows gain counts loaded in the same rail-counts worker (each via its scope service, quiet degrade): `Flashcards due: N` (bright >0, dimmed at 0; capped fetch renders `N+`), `Study decks (N)`, `Quizzes (N)`. Rows remain handoffs to the Study screen.

**Home due-mirror:** `build_home_dashboard_state` input gains `flashcards_due_count`; >0 renders a Needs Attention row `Flashcards due: 12 · Library` whose `detail_route` goes **directly to the Study screen's flashcards surface** (one hop — Library is not an artificial middle stop). Staleness contract: refreshed on Home's normal entry/refresh cadence. Service absent → no row.

## Phase L3b — Ingest canvas + Running feed + final cleanup

**Parity bar:** capability parity with **whatever TAB_INGEST ships today**. Planning opens with an inventory of `MediaIngestWindowRebuilt` — including which backend seams it actually calls and whether it carries server/API ingest. The Ingest screen deprecates only when the inventory is covered.

**Backend seam (decided by inventory, not assumption):** default to the seams the working ingest screen uses today. `Widgets/NewIngest/`'s `ProcessingJob`/`ProcessingState` are reused as *model shapes only*; its `UnifiedProcessor`/`BackendIntegration` are adopted only if the inventory shows the working screen already uses them. **Task 0:** headless smoke — ingest a small text file through the chosen seam to a real MediaDatabase row before any UI is built.

**Rail:** `ingest-import-media` flips `screen`→`canvas`. `ingest-import-export`: planning inventories real export/import seams (chatbook, document export); the canvas lists exactly the real actions — **if none exist, the row is removed** (a missing row beats an apologizing canvas) and its build is a tracked follow-up. This row is not part of the deprecation bar.

**Form** (stacked): `Import media` header → source input (path or URL) + `[Browse…]` (existing file-picker; if the modal fails the served-TUI smoke, the path input alone is the interface) → `type: auto ▸` cycling button → optional title/author/keywords → advanced options in a `Collapsible` (toggle buttons) → `[Start ingest]`. A terminal has no drag-drop; the path input + picker is the drop zone.

**Job queue** (below the form): one row per job — `● running · intro.mp4 · transcribing` / `✓ done · 2m` / `✗ failed · reason` — with `[Open in Library]` on success (via the shared open-by-id route, straight into the media viewer) and `[Retry]` on failure. **Stage labels, never fabricated percentages** — real progress only if the seam emits it.

**Job registry (architecture):** app-level `library_ingest_jobs` — pure job model + one long-lived **queue-runner worker** (exclusive in its own group) pulling jobs FIFO; submissions append to the registry and never spawn their own runner (a per-job exclusive worker would cancel running transcriptions). Registry mutations happen **only on the UI thread via `call_from_thread`**; progress ticks are targeted row `update()`s; recompose only on job add/remove/state-change. Library canvas and Home both read it: running → Home `Running` (`HomeActiveWorkItem`), failures → Home `Needs Attention` with `detail_route` to the ingest canvas; Home reads on its normal refresh cadence. **Accepted v1 limits (stated):** registry is in-memory — queue history dies with the app; a running job dies on quit (same as today); serial queue, parallelism is a follow-up.

**Server scope:** ingest writes to the local MediaDatabase. Under `Library | Server`, default behavior is local-only with a quiet `ingest runs on Local` line; the inventory may upgrade this decision if TAB_INGEST demonstrably carries server ingest today.

**Final cleanup (isolated last commit-series; pre-authorized to split into a trivial L3c if L3b runs hot):** retire `LIBRARY_MODES`, mode-chip remnants, `_active_mode` plumbing; study/flashcards/quizzes become plain handoff-canvas kinds; deleted screens go to the dead-code sweep.

## Architecture summary

| Unit | Phase | Role |
|---|---|---|
| `Library/library_notes_state.py` | L2b.1 (+templates state L2b.2) | notes list/editor/templates pure state |
| `Library/library_shell_state.py` | per phase | per-source page sizes (L2b.1); row-table flips |
| `Library/library_rag_state.py` (extend) | L3a | single query truth, history, status line |
| shared open-by-id route | L3a | `(source_type, record_id)` → detail surface; reused by L3b |
| `Library/library_ingest_state.py` + `library_ingest_jobs` registry | L3b | form/queue pure state; app-level job model + queue-runner |
| `Widgets/Library/library_notes_canvas.py`, `library_ingest_canvas.py` (+ search panel polish in place) | per phase | one posting-style widget file per canvas |
| `Home/dashboard_state.py` (extend) | L3a / L3b | `flashcards_due_count`; ingest job feed |

## Testing & gates (every phase)

Pure-state units first; pilot tests including re-anchors for every behavior change; real-backend mutation tests (real ChaChaNotes DB for notes/templates; real MediaDatabase for the e2e ingest of a small text file — no transcriber dependency; audio/video via faked processors behind optional-dep markers); hardened fakes mirroring service seams; render smoke-checks gating unproven widgets; generated-CSS discipline; live 2050×1240 populated QA; per-phase user screenshot approval before merge; fresh worktree off `origin/dev` per phase with its own plan and ledger.

## Tracked follow-ups (out of scope, logged)

Collections item-browsing; saved searches; service-backed conversations FTS filter; sync-failure → Home mirror; ingest parallelism + persistent job history; Import/Export build-out (if row removed); server-scope ingest (if not covered by inventory); dead-code sweep grows `notes_screen.py`, MediaIngest screen, legacy mode machinery.
