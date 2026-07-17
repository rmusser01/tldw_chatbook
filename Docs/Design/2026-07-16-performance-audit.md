# Performance & Snappiness Audit — 2026-07-16

Five parallel read-only audits (render/refresh paths, timers/workers, DB access
patterns, startup/imports, non-Console screens) against dev @ `006345bf`.
Every finding below was verified against the actual code (file:line quoted) and
the load-bearing numbers were **measured** with probe scripts against real
classes / synthetic data, not estimated. Goal per the product ask: make the UX
as snappy as possible **without affecting functionality**.

Fix tracking: backlog tasks **246–262** (each finding names its task).

Path convention: file references are relative to the `tldw_chatbook/` package
root (e.g. `DB/ChaChaNotes_DB.py` = `tldw_chatbook/DB/ChaChaNotes_DB.py`).

## Headline diagnosis

Three structural patterns account for most of the perceived sluggishness:

1. **The Console 0.2 s streaming tick does unconditional work** — including
   real sqlite queries and full widget recomposes — on the event loop, five
   times a second, for the entire duration of every streaming response.
   Measured: **11.2 ms median per tick** (3 k conversations / 8 workspaces,
   empty search), **~70 ms per tick with an active browser search string** —
   i.e. up to a third of wall-clock time during a stream spent off-render.
2. **Whole-screen `refresh(recompose=True)` instead of targeted updates** —
   Library has **124** whole-screen recompose sites (including per-checkbox
   handlers) while its purpose-built `sync_state()` methods have **zero
   callers**; Home, MCP, and two Console widgets have smaller instances of the
   same pattern.
3. **~1.3 s of the ~1.8–2.3 s cold start is import-time work for features the
   session never uses** — 469 ms building 1,313 Pydantic models for the
   remote-server API, ~550 ms of optional-extra machinery (chromadb/OTel/gRPC,
   playwright/trafilatura/dateparser, pymupdf/onnxruntime) imported eagerly
   despite existing lazy patterns in the same files/packages.

A fourth, self-inflicted one: a **commented-out log guard** in the DB layer
stringifies every query's params unconditionally — up to **14 ms per
image-message insert** to build a debug string that is never emitted.

---

## P0 — Quick wins (trivial diffs, immediate effect)

### A1. DB layer stringifies every query's params for a disabled debug log — task-246
`DB/ChaChaNotes_DB.py:2635-2636`: the `isEnabledFor(logging.DEBUG)` guard is
**commented out** (its comment still says "Avoid formatting query/params if
not debugging") so the f-string — including `str(params)` over raw
`image_data` BLOBs — runs on *every* query. Measured: 0.87 ms @200 KB,
4.96 ms @1 MB, **14.3 ms @3 MB**, paid on the send-completion persist path of
every image message. Systemic: same pattern at `DB/Prompts_DB.py:433`,
`DB/Client_Media_DB_v2.py:626` (full ingested document text), and
`DB/Sync_Client.py:667,674`. Fix: loguru has no isEnabledFor — use logger.opt(lazy=True) with callables (or a loguru min-level check);
truncate params *before* stringifying. Risk: none (logging-only).

### A2. Console `save_state()` runs twice per tab-switch-away — task-247
`app.py:4611-4621` calls `save_state()` explicitly and stores the result;
Textual's `ScreenSuspend` then fires `ChatScreen.on_screen_suspend`
(`chat_screen.py:10651-10655`) which calls `save_state()` **again and
discards the result**. `save_state` walks O(sessions × messages) serializing
every native Console message. Fix: delete the suspend-hook call. Risk: none
(return value provably unused).

### A3. 100 ms guaranteed event-loop freeze every 2 s on the Embeddings screen — task-248
`Widgets/performance_metrics.py:196` calls
`psutil.Process.cpu_percent(interval=0.1)` inside a sync `set_interval(2.0)`
callback — `interval=0.1` **sleeps 100 ms on the event loop** every tick while
Embeddings Management is open (`Embeddings_Management_Window.py:288`). The
correct non-blocking form (`interval=None`) is already used at
`Metrics/metrics.py:200`. Latent copies (currently no UI callers):
`metrics_logger.py:153`, `RAG_Search/simplified/health_check.py:277`.

### A4. Conversation search uses a correlated `LIKE` scan; the FTS index already exists — task-249
`DB/ChaChaNotes_DB.py:4924-4936` (`search_conversations_page`) matches
message content with `EXISTS(SELECT 1 … m.content LIKE '%q%')` — per-candidate
correlated scan, index-hostile leading wildcard — while the schema defines and
triggers `messages_fts` (line 326-356) and `search_messages_by_content`
(6336-6360) already uses it correctly. Measured: 1.4 ms → **7.8 ms per scope**
with a search string; multiplied by the P1-B1 tick this hits ~70 ms/tick.
Fix: join `messages_fts MATCH`. Risk: low (in-schema template exists).

### A5. Chatbook import commits ~1,500 transactions for a 50×30 import — task-250
`Chatbooks/chatbook_importer.py:314-388` never opens an outer transaction;
`add_conversation`/`add_message` each commit individually.
`TransactionContextManager` is depth-tracked/reentrant
(`ChaChaNotes_DB.py:9703-9722`) so wrapping the loop is a pure win.

---

## P1 — Interaction hot paths

### B1. The Console 0.2 s sync tick — task-251 (umbrella)
`chat_screen.py:7419-7431` arms `set_interval(0.2, _poll_transcript)` for the
duration of any active run; the tick runs `_sync_native_console_chat_ui`
(7356-7381) = **10 sub-syncs, most unconditional**:

- **Worst**: `_sync_console_workspace_context` → `_sync_persisted_console_browser_rows`
  (3386-3520) re-queries the DB — `search_conversations_page` +
  `count_messages_for_conversations` + keywords — **per scope (global + every
  workspace), every tick, on the event loop** (the sync worker is a coroutine,
  not `thread=True`), then `ConsoleWorkspaceContextTray.sync_state`
  (console_workspace_context.py:203-221) does `refresh(recompose=True)`
  **unconditionally**. Measured 11.2 ms/tick median; ~70 ms/tick with an
  active search. The template fix exists 700 lines away: the sub-agent badge
  count was previously ~75 queries/tick and got batched + TTL-cached
  ("Finding A", chat_screen.py:4037-4068). Apply the same gate/TTL — or drop
  this sub-sync from the tick entirely (8 explicit invalidation sites exist).
- `ConsoleSettingsSummary.sync_state` (console_settings_summary.py:33-54) has
  **no equality guard** (sibling `ConsoleRunInspector` has one) → ~13 forced
  `Static.update()`/`refresh(layout=True)` per tick with agent-section and
  system-line helpers (chat_screen.py:1897-1909, 1778-1786).
- `ConsoleRunInspector` recomposes wholesale on any change; selecting the
  streaming message makes its excerpt change every tick → full panel teardown
  5×/s; it is also synced while `display:none` (6163-6164).
- Store-side per-tick tax: `messages_for_session` dataclass-copies **every
  message** per tick (console_chat_store.py:529-534, 1187-1188);
  `_materialize_stream_buffer` re-joins the entire chunk list per tick
  (1175-1184) — measured negligible at chat sizes, structural fix cheap.
- Guidance copy-blocks `.update()` unconditionally (5398-5417); rail-state
  tuple computed twice per tick (7372-74 + 9477).

Good news preserved: the transcript itself is properly fingerprint-gated and
its row reconciler is genuinely incremental (`_reconcile_rows`,
console_transcript.py:695-738); streaming does **not** hit sqlite per chunk
(persist = one INSERT at first content + one UPDATE at finalize) — keep both.

### B2. Library: 124 whole-screen recomposes; targeted `sync_state()` has zero callers — task-252
`library_screen.py` calls `self.refresh(recompose=True)` (BaseAppScreen: full
remove-and-remount of nav bar, footer, ~20-row rail, and 50–100-row canvas)
from per-row checkbox/select handlers (5926-5956, 6020-6045, 10268-10312) and
~110 more sites. `LibraryRail.sync_state()`, `LibraryMediaCanvas.sync_state()`,
`LibraryConversationsCanvas.sync_state()` exist for exactly this and are never
invoked. This pattern already caused the app-wide mouse-capture bug that
`base_app_screen.py:52-84` defensively works around. Fix: route state changes
through the existing `sync_state` methods; patch single-row toggles directly.
Risk: moderate (rail counts must stay in sync) — stage by interaction class.

### B3. Home dashboard: 3 sync DB queries on the UI thread per visit/click — task-253
`home_screen.py:417-444` inline-executes watchlist snapshot, notification
queue (limit 100), and server-event feed queries synchronously at every
compose, triage sync, and rail click — while the sibling
`_home_content_seam_call` (350-383) already does `asyncio.to_thread`
correctly. Also `HomeRail`/`HomeCanvas.sync_state()` always recompose
(home_rail.py:64, home_canvas.py:47). Fix: thread + cache the seam calls;
targeted rail/canvas patches.

### B4. Debounced searches run sync sqlite/FTS on the event loop — task-254
`run_worker(coroutine)` is **not** a thread. Console browser search
(chat_screen.py:661-671 → chat_conversation_scope_service.py:144-147 →
`chat_conversation_service.list_conversations`, a plain `def`) blocks the loop
×(1 + workspaces) per debounce fire (~3 ms p50 @1.5 k conversations, grows
with data). Same shape: CCP search (conv_char_events.py:615/1248), media
search (media_events.py:363/518). The `_maybe_await` plumbing looks async but
the local mode never yields. Fix: `asyncio.to_thread` at the service leaf.
Risk: medium — thread-local connections are fine; note `exclusive=True`
cancellation cannot interrupt an in-flight thread call.

### B5. Library RAG panel rebuilds ~100+ widgets per keystroke — task-255
`library_screen.py:11821-11827`: `Input.Changed` on the RAG query box calls
`_refresh_search_rag_panel_state_widgets` (12375-12425), which tears down and
remounts the whole Evidence results list and Recent-searches history
(individually awaited `remove()`/`mount()`) even though neither depends on
unsubmitted text. Fix: per-keystroke path updates only the run-button/status
line (already a separable function).

---

## P2 — Startup (~1.8–2.3 s to first paint; import time dominates, `TldwCli()` warm is only ~31 ms)

### C1. `tldw_api` package: 1,313 eager Pydantic models, ~469 ms (31 % of import) — task-256
`tldw_api/__init__.py` (1,681 lines) eagerly imports 54 schema files
(16,376 lines). Forced by `app.py:353` and **69 other files**. Fix: PEP 562
lazy `__getattr__` re-exports — the pattern already exists, documented, in
`Local_Ingestion/__init__.py` — plus `Server*Service` modules importing their
own schema submodule directly. Longer-term: don't construct ~30
`Server*Service` objects in `TldwCli.__init__` in local-only mode.

### C2. Optional-feature imports paid eagerly, ~550 ms — task-257
- `Embeddings/Chroma_Lib.py:39-40`: module-scope `get_safe_import('chromadb')`
  does a **real import** (chromadb → OTel → gRPC → protobuf, ~154 ms via
  RAG_Admin which app.py imports unconditionally). Move into
  `ChromaDBManager.__init__`.
- `Tools/__init__.py:17` eagerly imports `WebSearchTool` →
  `Article_Extractor_Lib.py:69-73` module-scope playwright + trafilatura
  (dateparser 75 ms, htmldate 78 ms) — **~197 ms** for a feature disabled by
  default (`web_search_enabled=False`; the executor gate at
  tool_executor.py:643 is already correct). The same file already fixed
  pandas with a `find_spec` probe — apply identically.
- `app.py:124` direct-submodule import of `local_file_ingestion` bypasses
  Local_Ingestion's own lazy `__init__` → pymupdf/onnxruntime (~170 ms) +
  Document lib (~59 ms) for optional `pdf`/`ebook` extras. Fix:
  import-per-format at dispatch (classify/dispatch split already exists).

### C3. `config.py` import hygiene — task-258
The 1,285-line embedded default TOML is parsed **twice** (2753 and 3840 —
the second just to read `providers`); `load_cli_config_and_ensure_existence()`
and `load_settings()` each independently re-open, re-parse, and re-merge the
same user config at import (3865-3866); `Utils/Utils.py:48` imports `chardet`
(~21 ms, 40 submodules) for two rarely-called functions. Fix: reuse the parse,
consolidate the double load (watch `_CONFIG_CACHE`/`_SETTINGS_CACHE`
semantics), lazy-import chardet. Related, lower value for the app itself:
`tldw_chatbook/__init__.py:24`'s compatibility shim imports `textual.widgets`
(~53 ms) for every non-UI consumer (tests, tooling) — guard it.

### C4. Monolithic CSS parse ~90–130 ms before first paint — task-262 (investigation)
16,064 lines / 2,278 rules parsed unconditionally. Textual supports
per-`Screen` CSS; splitting needs a cross-screen selector audit and
per-screen visual QA — staged work, not a quick win. (Related but larger:
`chat_screen.py`'s 11 k lines cost ~161 ms of pure module-exec as the default
tab — a file split is noted for future consideration, no task filed.)

---

## P3 — Targeted polish

### D1. Console transcript/stream polish — task-259
Per-message row-signature cache (`_transcript_rows` re-derives every row per
changed tick: 0.86 ms @200 msgs → **5.15 ms @1,000 msgs**, measured);
stream-buffer collapse-after-join; `_stage_console_library_rag_launch`
(chat_screen.py:5729-5731) currently recomposes the **entire ChatScreen** for
one pending card → targeted widget; `ConsoleRunInspector` per-row updates +
skip-when-hidden.

### D2. RAG conversation search: N+1 + BLOB overfetch — task-260
`RAG_Search/pipeline_functions_simple.py:96-115`: one
`get_messages_for_conversation` per matched conversation (≤20 queries/search),
each SELECTing `image_data` BLOBs to build text snippets.
`get_messages_for_conversations_batch` (ChaChaNotes_DB.py:5870-5929) exists,
unused. Add `include_image_data=False` variant for text-only callers (also:
mindmap_integration.py:88).

### D3. Sundries — task-261
Background-effect `render_line` recomputes the full W×H grid **per line** =
O(W·H²)/repaint (console_background_effect.py:92-116; cache per frame tick);
10 s footer token-count re-tokenizes the whole visible history without a
dirty check (app.py:5950-5954); `SELECT 1` liveness ping per `get_connection`
per query (~2× raw call count, ChaChaNotes_DB.py:2431-2434);
`BookmarksManager.__init__` sync `Path.exists()`×5 (cloud dirs can stall) +
first-run TOML write on every picker construction
(enhanced_file_picker.py:85-123); MCP full rebuilds ×2 per lifecycle action
incl. hidden Tools canvas (mcp_rail.py:198, mcp_servers_mode.py:459-500) —
N+2 store loads already tracked as task-236; missing indexes on
`conversations.deleted`/`last_modified` (browser ORDER BY) for large tables.

---

## Verified-fine (do not "fix"; preserve as templates)

- Transcript row reconciler is genuinely incremental; streaming persists
  exactly twice per turn (no per-chunk sqlite).
- No search-as-you-type DB hits anywhere in Library (all `Input.Submitted`);
  console browser search is properly debounced with a cancellation token.
- Library pagination is real, gathered, capped, `include_keywords=False`;
  mount is instant-then-reconcile with a 5 s TTL cache.
- `list_conversations` batches counts+keywords (no N+1);
  `set_message_attachments` is DELETE+executemany in one transaction.
- Subscriptions scheduler is the correct `thread=True` periodic-sync-DB
  template; rail-preference prune is one-shot + threaded.
- Screen registry lazy-imports correctly; no torch/transformers on the start
  path; config reads are cache-backed; picker directory scans are threaded.
- Composer cursor-blink and setup-modal snow timers are correctly paused.
- All non-Console screens' `save_state`/`restore_state` are O(1)-cheap.
- File-picker search box is **unwired** (silent no-op) — a correctness
  follow-up, not a perf issue.

## Measurement provenance

Probe scripts (scratch, not committed): `probe_search_cost.py`,
`probe_sqlite_write.py`, `probe_console_rail.py`, `bench_transcript*.py`,
plus `python -X importtime` traces under isolated `HOME`. Synthetic data
sizes stated inline with each number.
