# Holistic performance review — dev `35d4bf3a1` (2026-08-22)

**Symptom:** users report the app has recently slowed down — general sluggishness, input/click
lag, and slow startup/screen switches; historically 3–5× worse on constrained hardware, more
when fsync-bound.
**Method:** seven parallel read-only review lanes (regression re-check of every 2026-08-11
audit fix · startup & first interaction · input paths · cross-cutting hooks & timers · DB & I/O
· new-subsystem cost-when-idle · rendering & CSS) against a pinned worktree of dev `35d4bf3a1`,
plus live headless-Pilot probes run in an isolated second worktree (scratch
HOME/XDG/`TLDW_CONFIG_PATH`, fresh profile, no live-config writes). Top claims re-verified
first-hand; lane contradictions resolved by measurement (two below). Baseline for
"regression?" is the 2026-08-11 input-latency audit pin `82b595049`
(`Docs/Design/2026-08-11-input-latency-audit.md`); dev absorbed ~3,800 commits / +231k package
LOC (+24%) in those 11 days.

**Verdict in one paragraph.** None of the 13 audited 2026-08-11 fix families is broken —
regression re-check found them all structurally intact. The recent slowdown is instead
(1) a **first-boot-after-upgrade wall**: schema v34→v46 replays 12 migrations inside
`TldwCli.__init__`, and the brand-new v46 migration rewrites the entire `messages_fts` index
in one transaction; (2) **boot-cost accretion** from new subsystems — ~15k LOC of Chunking
imported to read one string constant, Persona_Buddy dragging 93% of Persona_Visual (and PIL)
eagerly, 155 pydantic models through two lazy-facade leaks, seven feature DBs schema'd in
`__init__` for features the user may never touch; (3) **new per-input-event work** in the
rewritten Console surfaces (drag-selection re-wraps the message body per mouse-move; the
right-rail forces a layout refresh per scroll frame; every keystroke does a synchronous
Workspace-DB read — measured live, ~1.25 reads/key); and (4) **fsync-shaped stores that dodged
the audit's WAL+NORMAL sweep** because they live outside `DB/`, above all the notes
device-state store (DELETE+FULL, fresh connection AND a ~60-statement schema census per
transaction). The CSS parse-cache cliff is NOT re-crossed on the plain tab tour (47 sources
measured vs the 64 cliff), but 34 new `DEFAULT_CSS` classes since the consolidation put a
feature-rich session (~10 distinct modal opens) over it — and the guard test that would say so
is red for an unrelated reason (below) while CI hasn't completed since June.

The filed backlog for this review is **task-21100 – task-21134**. This document is the durable
evidence record those tasks cite (the backlog CLI drops custom sections, so measurements live
here). The burn-down's own follow-ups, filed at close-out on 2026-08-23, are **task-21230 –
task-21249**; what shipped, and the two findings the implementations proved wrong, are recorded
in "What the burn-down changed" and "Corrections found during implementation" below.

---

## Live measurements (fast M-series, headless Pilot, isolated profile)

### Boot and import

| probe | result |
|---|---|
| `import tldw_chatbook.app`, cold (first import: .pyc compile + first dylib loads — what a user pays on first launch after an upgrade) | **3.10 s** |
| same, warm | **0.825 s** |
| largest cold chain | `Chat.console_runtime` → `Persona_Buddy.console_adapter` → `Persona_Buddy.controller` → `Persona_Visual.assets` → `PIL.Image` → **`PIL._imaging` 1.276 s (41% of cold import)** |
| largest warm chains | `Library.ingest_capabilities` 83 ms · `Chat.chat_conversation_scope_service` 65 ms (pulls Library → Sync_Interop) · `TTS` 58 ms · scheduler handlers → Subscriptions → bs4 ~80 ms · `Notes.file_notes_git_service` 39.5 ms self |
| `TldwCli()` construction (UI thread, pre-paint) | **0.528 s** |
| `run_test()` boot-to-ready | **2.212 s** |
| total to interactive | ≈ **3.5 s** fast HW → 10–17 s constrained (matches known ~10–12 s tmux cold starts) |

### 13-destination tab tour

Stylesheet sources: 27 at boot → **47 after the full tour** (cliff = 64, guard soft limit 56;
Aug-11 pre-fix: 93, post-fix: 49). Warm `stylesheet.parse()` after the tour: **0.8 / 0.2 /
0.1 ms** — the parse cache is healthy on the plain tour. Per-switch wall (first visit):
Chat **1.398 s**, Personas **1.381 s**, LLM 0.842 s, Library 0.576 s, Settings 0.577 s,
Schedules 0.544 s, the rest 0.21–0.45 s. Warm re-visits (screens are never cached): 0.33–0.62 s
⇒ construct+compose+teardown floor of ~0.3–0.6 s fast HW → 1–3 s constrained on every switch.

### CSS guard tests on the pin

`Tests/UI/test_widget_css_consolidation.py::test_full_destination_tour_stays_under_the_parse_cache_cliff`
**FAILS — but not on the count**: `_build_test_app()` crashes inside `TldwCli.__init__` →
`_wire_character_persona_services` (app.py:6010→6612) → Actor_Packs
`persona_coordinator.recover()` → `ActorPackRepository.list_persona_intents()` → DB query with
the ChaChaNotes DB unassigned in the harness. **The cliff guard is disarmed on dev**, invisibly
(CI has not completed since 2026-06-26). The Performance-suite twin
(`test_destination_tour_css_sources_stay_below_parse_cache`) passes via its own app factory.
Net: 1 failed, 32 passed.

### Pre-importer A/B (suspect eliminated)

20 keystrokes immediately after boot, then idle to 30 s — `TLDW_SCREEN_PREIMPORT` default vs
`=0`: keystroke median 82.1 vs 99.0 ms, p90 120.8 vs 160.7 ms, 30 s CPU 2.51 vs 2.75 s —
**no measurable pre-importer penalty on multi-core hardware** (differences within noise).
Residual concern is 1–2-core boxes only (GIL competition is structural: the import loop has no
sleep/yield between the ~21 modules / ~117k lines it executes); filed as design-hardening, not
as a measured harm.

### Idle burn

| state | 30 s / 15 s CPU | attribution (timer census) |
|---|---|---|
| Console, **setup state** (no provider configured) | **9.9% of a core** | `ConsoleSetupBackdrop._tick` at **5 Hz** — the setup modal's decorative snow field; every tick refreshes the full-screen widget → `Screen._on_timer_update` → layout+render (75 Layout messages/15 s) |
| Console, **configured state** | **1.8% of a core** | nav overflow 2 Hz (no-op-gated) + composer caret blink ~1.9 Hz (28 Static refreshes/15 s) + 1 Hz heartbeat |

The configured-state idle floor is acceptable (×3–5 ⇒ ~5–9% constrained). The setup state —
every new user's first experience, on whatever hardware they have — burns 5.5× more for a
decoration. A `reduced_motion` flag exists but is not the default.

### Per-keystroke Workspace-DB reads (lane contradiction, resolved by measurement)

Lane L1 (static) cleared the keystroke builders as DB-free; lane L3 (static) traced a
synchronous SQLite read one level deeper. Live counter: 20 printable keys in the configured
composer → **25 × `ensure_default_workspace` + 25 × `get_active_workspace`** (~1.25 each per
key), 3.10 ms cumulative for all 50 calls (≈62 µs/call, warm page cache, fast SSD). L3 is
right; the cost profile is benign on fast hardware and disk-sensitive (cold cache, slow disk,
and the repair branch's DELETE write are the risk cases).

### Measurement-artifact disclosure

An intermediate census attributed ~940 posted `Callback` messages per keypress; attribution by
qualname showed 18,640/18,648 were `Pilot._wait_for_screen.<locals>.decrement_counter` —
**Textual Pilot's own `pause()` machinery (one callback per widget per pause), not app
behavior**. All absolute keystroke latencies measured through `pilot.press()+pause()` in this
review are inflated by that overhead and are quoted only as A/B comparisons, never as app
latency. App-attributable per-key work in the configured Console measured ~7 widget refreshes.

### Pragma read-back (new stores)

`NotesDeviceStateStore`: **journal_mode=delete, synchronous=2 (FULL)** — live confirmation.
(TTS profile repository read back statically by the DB lane: WAL+NORMAL, dedicated executor —
conforms; its v3→v4 migration memory spike is a separate finding.)

---

## Findings → tasks

Severity is stated for constrained hardware (×3–5). "REGRESSION" = absent at `82b595049`.

### P0 — the upgrade wall and the store that dodged the sweep

- **21100** · First boot after upgrade replays v34→v46 (12 migrations) inside
  `TldwCli.__init__`, and `chachanotes_v45_to_v46_sync_log_retention.sql` (merged 2026-08-22,
  PR #1974) unconditionally rebuilds the whole `messages_fts` index (`delete-all` + reinsert of
  every non-deleted message) plus nine full-`sync_log` purge scans, all in ONE transaction —
  tens of seconds to minutes of silent pre-paint hang on real profiles on slow disks, WAL
  ballooning to index size. REGRESSION. Fix: chunked resumable FTS backfill outside the
  version-bump transaction (in-repo exemplar: `Subscriptions/fts_backfill.py`) + an "upgrading
  database…" splash state. Runner: `ChaChaNotes_DB.py:5936-6021`; boot placement `app.py:5808`
  → `_init_notes_service`. (ChaChaNotes runs WAL+NORMAL — the wall is I/O volume, not fsync.)
- **21101** · `Notes/notes_device_state_store.py:443-472` — the app's only remaining
  DELETE+FULL store (live-confirmed above), fresh connection per op, and
  `initialize_notes_device_schema` re-runs a full `sqlite_schema` census + re-executes 16
  `CREATE INDEX IF NOT EXISTS` (~60 statements) inside **every** transaction, including pure
  reads. Behind the notes-sync runtime (boots unconditionally) and the notes import executor
  (2–4 receipt transactions per imported note ⇒ a 500-note import ≈ 1,000+ open/census/fsync
  cycles). REGRESSION. Fix: the sanctioned template (held thread-local conn, WAL+NORMAL,
  `isolation_level=None`) + schema init once at `initialize()`.

### P1 — startup cost accretion

- **21102** · ~15k LOC of `Chunking/` (incl. 28/38 vendored engine modules, a real
  `import langdetect`, an nltk `find_spec` path scan, and the Internal_Prompts package) is
  imported eagerly through SIX entry points, the first being
  `Local_Ingestion/local_file_ingestion.py:172` importing `ENGINE_VERSION` — a string literal
  (`Chunk_Lib.py:150`). All six must break (ingest_preflight.py:26, web_clip_request.py:27,
  RAG_Search/__init__.py:21, RAG_Admin/local_rag_admin_service.py:17-18, app.py:1997-2007);
  fixing app.py alone buys zero. Durable guard: a test asserting `"Chunking" not in
  sys.modules` after `import tldw_chatbook.app`. REGRESSION.
- **21103** · `Persona_Buddy` (eager at app.py:393) drags **93% of Persona_Visual (6,633 LOC)**
  via `controller.py:18,23` + `rendering.py:11-13` (the latter imports the tree for one `int`
  constant) — this is the chain that puts `PIL._imaging` (1.28 s cold) on the boot path. Both
  consumers already tolerate absence (`app.py:8582`, `console_runtime.py:468,519`). Fix: lazy
  controller property (house pattern: `_build_rag_admin_services`, app.py:6054-6124), split the
  stdlib-only `console_adapter` out of the package-init surface, constant moved. REGRESSION.
- **21104** · `Subscriptions/monitoring_engine.py:37` imports `bs4` unguarded (eager via
  scheduler handlers ← app.py:429) while `beautifulsoup4` exists only in
  `[project.optional-dependencies]` (four extras; verified none in core) — a base
  `pip install .` cannot `import tldw_chatbook.app` at all, and everyone else pays the import.
  PRE-EXISTING. Fix: promote to core deps or guard-and-degrade; add an import-closure test for
  extras-gated packages.
- **21105** · Seven feature databases are created + schema'd synchronously inside
  `TldwCli.__init__` for features a never-user doesn't touch: research (5 tables+migrations),
  notifications, event_state (10 DDL) + sync_state (16 DDL) server-parity stores, writing (16
  DDL), kanban (24 DDL — zero UI consumers found at all), notes_sync_state. The lazy seam
  already exists (`BaseDB.__init__(initialize_schema=False)`, base_db.py:43). Fix: open-on-
  first-use uniformly + per-store test asserting no DB file exists after boot without feature
  use. Mostly PRE-EXISTING; notes-sync leg REGRESSION.
- **21106** · Actor_Packs: `recover()` runs synchronous SQLite inside `__init__`
  (app.py:6612) — this is ALSO what crashes `_build_test_app()` and disarms the CSS cliff
  guard; and `creation.py:17` imports `tldw_api.character_persona_schemas` (79 pydantic
  models, ~34 ms) eagerly. Fix: move recovery to Personas-surface mount (its own docstring
  only requires "before affected surfaces mount"); TYPE_CHECKING the schema import; re-arm and
  re-run the cliff guard. REGRESSION.
- **21107** · `Kanban_Interop/server_kanban_service.py:10` module-level `from ..tldw_api
  import` (31 names) forces `kanban_schemas` (76 pydantic models, ~44 ms), defeating tldw_api's
  PEP-562 lazy facade — one of exactly two leaks (the other fixed by 21106). Fix +
  `sys.modules` guard test for `tldw_chatbook.tldw_api.kanban_schemas` after app import.
- **21108** · app.py import diet (top-level imports 194→220): `speech_tts_settings_panel`
  (5,618-line widget module imported for one payload dataclass — move
  `SpeechTTSPanelDraftSnapshot` to a types module), `TTS/voice_bundle_service` (1,857),
  `Notes/notes_sync_runtime` chain, Notifications package init. None needed before first
  paint. Also tighten `Tests/Performance/test_app_import_weight.py` (currently 8.0 s / 4,000
  modules — far above drift). REGRESSION.
- **21109** · `_build_generated_video_store()` runs `VideoStore.enforce_retention()` in
  `__init__` behind an EXCLUSIVE interprocess portalocker lease with a **5.0 s poll-timeout**
  (video_store.py:56,191-233) — a held lease from a concurrent instance blocks boot up to 5 s;
  scan+delete runs pre-paint. Fix: construction stays path-only; retention moves to
  `_schedule_deferred_startup_work`. REGRESSION.
- **21110** · Splash serializes with, instead of hiding, the initial chat_screen import: splash
  runs 1.5 s doing nothing, THEN ~20k lines import+compose on the loop
  (app.py:8343-8386 → 12611-12642 → 11166-11222); the pre-importer only starts post-mount.
  Fix: kick the resolved initial route's module import on a thread at splash-mount.
  PRE-EXISTING.
- **21111** · Startup-init hygiene bundle: (a) the `__init__` parallel-task timing log measures
  durations AFTER `future.result()` returns (app.py:5823-5825) so every task logs ~0 s — the
  existing STARTUP TIMING SUMMARY cannot attribute the parallel phase (fix first, it makes all
  other startup work measurable); (b) 2–3 `keyring.get_keyring()` backend discoveries during
  `__init__` (server credentials ~13 ms + Security.framework ctypes; skills trust ×2) → lazy
  properties; (c) `_restore_ingest_jobs` does open+read+reconcile-writes on the UI thread in
  `on_mount` (app.py:2211-2239) → `to_thread`; (d) `ensure_builtin_samira` full-scans
  `character_cards` parsing every `extensions` JSON per boot
  (`Character_Chat/visual_identity.py:3044-3060`) → targeted `json_extract` + LIMIT 1 or cached
  id. Mixed provenance.
- **21112** · Notes-sync runtime hardening: starts unconditionally at `app.py:10297-10303`
  (`cutover_admitted=True` hardcoded) — zero-profile boots still create the state DB, run ≥3
  censused transactions, and first boot runs two unbounded SELECTs over chachanotes.db
  (notes_sync_legacy.py:603-628); with ≥1 active root the watcher does a **full recursive
  stat walk of every root every 1 s forever** (notes_sync_watcher.py; bounds 10k entries —
  over-bounds roots pay the full scan every tick before bailing). Fix: gate `start()` on
  non-empty `list_root_summaries()` (+ legacy config key for migration; Library already falls
  back to `InertLastingSyncRuntime`), back the watcher off on unchanged signatures (1 s →
  5–15 s) or use FS events, bound the legacy scan. REGRESSION.
- **21113** · Screen pre-importer niceness (design hardening; A/B above showed no fast-HW
  harm): `_preimport_screens` (app.py:11762-11886) is a tight no-yield loop over ~117k lines
  on a GIL-holding daemon thread starting 0.2 s after first paint. Fix: sleep between routes,
  order by configured default_tab first, skip while `_screen_navigation_lock` is held,
  consider disabling below 4 cores. Keep `TLDW_SCREEN_PREIMPORT` overrides. REGRESSION
  (mechanism itself shipped by task-15472's arc).

### P1 — Console input paths and rendering

- **21114** · Transcript drag-selection does, per MouseMove at 50–100 Hz:
  uncached `get_display_text()` rebuild of the full message body (plain rows), a full
  `Content(text).wrap(width)` over the body, an ungated `set_selection_range` →
  `body.update()` full re-render even when the range didn't move, and a sweep over every
  mounted row calling `clear_selection()` (console_transcript.py:5036-5064, 4896-4928,
  2278-2311, 1740-1787). Tens of ms per event on multi-KB rows ⇒ seconds of cumulative lag per
  drag on slow HW. REGRESSION (whole feature post-baseline). Fix: cache display text
  (invalidate in `sync_message`), memoize the wrap table per (text,width) per drag, early-out
  on unchanged range, remember the single selected row.
- **21115** · CSS parse-cache headroom nearly consumed: **34 new `DEFAULT_CSS` declarations /
  29 files since the consolidation** (Console modals/inspector rail/turn-file card, Library
  dialogs, trajectory, speech). Arithmetic on the pin: plain tour ≈47 measured (empty
  transcript, no modals); + conversation-row classes + ~10 distinct modal opens **crosses the
  64 cliff today**; accretion rate ~+8 classes/3 days with the tour guard red (21106) and CI
  not running. Every one is a plain string block that can ride the sanctioned
  `BUNDLED_CSS`/`BUNDLED_SCREEN_CSS` + `build_css.py` mechanism; also convert the last
  class-level `CSS` (`UI/SiteConfigSettings.py:41`). Durable guard: a STATIC allowlist ratchet
  (the AST walk already exists in test_widget_css_consolidation.py) failing on any new
  `DEFAULT_CSS`/`CSS` outside the list — so the invariant stops depending on the slow
  integration tour. REGRESSION by accretion.
- **21116** · library_screen.py still performs whole-screen `refresh(recompose=True)` on
  per-click paths — ~105 statement-level sites (99 `self.`) on a screen that grew 26k→34.8k
  lines; confirmed hot: `_open_library_item_by_id` (rail row / RAG result / media open),
  ~~`_apply_library_row_toggle`~~ (**FALSE POSITIVE — corrected during implementation**;
  instrumented at the base commit it performs **0 recomposes and 0 widget constructions** per
  click, because the cited recompose statement sits inside an exception-fallback arm that a
  normal toggle never reaches), media-viewer back, skills/prompts import open/cancel, export
  open. Continue the 15457 canvas-scoped conversion (`library_canvas_sync` seam, 82 call
  sites) + a ratchet test on the count. PRE-EXISTING pattern, cost-per-event regressed with
  screen growth (9 sites added post-fix are low-frequency admin flows).
- **21117** · Right-rail (Inspector) scroll pipeline: `watch_scroll_y` → geometry reconcile
  runs `self.refresh(layout=True)` on the whole rail per scroll frame plus two DOM queries and
  a second `call_after_refresh` hop, even when nothing changed (right_rail.py:116-145,
  298-375; file grew 249→1,092). Fix: split the pure-scroll path (hint update only) from the
  layout+refold reconcile (resize/section-demand only). REGRESSION.
- **21118** · Per-keystroke synchronous Workspace-DB reads (measured: ~1.25 ×
  `ensure_default_workspace` + `get_active_workspace` per key): memoize the workspace context
  on the screen, invalidate on workspace-change events, make the keystroke path read-only
  (repair side-effects move to session-start/workspace-switch); also cache staged-launch
  `EvidenceBundle.from_payload` (re-parsed ≥2× per keystroke during staged launches).
  PRE-EXISTING, partially mitigated by 15452/15465.
- **21119** · Chat-screen click dismissal walks the whole screen DOM **three times per handler
  invocation** — `query(ConsoleTranscript)`, `query(ConsoleSelectionMenu)`, and a third inside
  `_remove_selection_menu`, which runs its own query (chat_screen.py:18939-18990). The handler
  runs **once or twice per physical press** depending on who swallows the Click: a composer
  press is 1 invocation = **3 walks**; a rail press is 2 invocations = **6 walks**. Fix:
  mounted-menu flag / cached transcript ref, early-return when nothing is mounted. REGRESSION.
  **CORRECTED during implementation** — the original text said "twice … on BOTH MouseDown and
  Click … (~4 traversals per click)". See "Corrections found during implementation" below.
- **21120** · Composer per-key residue: `_sync_send_disabled_reason` does an unconditional
  `strip.update(Content(...))` per key (the `reason_changed` gate covers only the ARIA
  announcement — the audit's half-gate pattern, console_composer_bar.py:1582-1607); hidden
  compatibility `Input.value` mirror re-set with the full draft per key (fires a second
  Changed handler); ghost-text reverse scan of prompt history per render AND per 0.5 s blink
  tick. Mixed provenance.
- **21121** · The 0.2 s run tick gained `_console_changed_files_scope()` — a full
  shallow-snapshot pass over every session message per tick when no change-review marker
  exists (the common case), making ≥2 full passes per tick with the cost path
  (chat_screen.py:11467-11497 → console_chat_store.py:2858-2865). Fix: memoize newest run-id
  on the store, bump on marker append (pattern: the estimate cache). REGRESSION.

### P1/P2 — cross-cutting hooks, timers, DB & I/O

- **21122** · Persona Buddy runtime costs (feature-enabled case): the widget repaints
  unconditionally at 10 Hz (`set_interval(0.10, refresh_from_controller)` with no change gate
  — verified; three `Static.update` per tick while a separate frame timer already animates);
  the resolution loop can retry a failed visual at 10 Hz with four `to_thread` hops + DB reads
  per iteration; geometry keys spawn one non-exclusive config-write worker per keypress.
  Fixes: gate the poll on (snapshot generation, preferences generation, visual identity) or
  replace with controller-posted messages; capped backoff on unconfirmed-unavailable;
  coalesce geometry persists. REGRESSION (feature is new).
- **21123** · Persona Buddy hook placement: `BaseAppScreen` awaits
  `reconcile_persona_buddy_view()` at the end of EVERY screen recompose, and every
  mount/resume schedules a reconcile worker — even with the feature disabled (default), and
  the widget module imports before the enabled check (base_app_screen.py:357-524). Disabled-
  case per-event cost is µs-scale (verified) — the payload is multiplied lifecycle work,
  per-screen duplicated state, and teardown-race defense. Fix: relocate to a single app-level
  overlay owner reacting to screen-change events + controller generation bumps; short-term,
  early-return before `run_worker`/import when disabled. REGRESSION.
- **21124** · `get_cli_setting` (398 call sites, many on the event loop) takes the global
  config file lock BEFORE the cache check (config.py:5107), so any concurrent config write —
  which holds the lock through 2 fsyncs + 3 full TOML parses + a settings rebuild — stalls
  every loop-side read. Amplified by per-click writers (Logs filter chip = 2 rewrites/4 fsyncs
  per click, theme switch, lab rail) and a per-keystroke writer
  (Dictation_Window_Improved.py:602). Fix: double-checked fast path on the existing
  `_CONFIG_GENERATION` before taking the lock; coalesce the write path to one parse; debounce
  the hot writers. PRE-EXISTING amplifier.
- **21125** · Writing screen: ~45 per-op `connect_private_sqlite` sites run directly on the
  event loop (load, tree clicks, autosave — no `to_thread` anywhere under UI/Writing_Modules
  or Writing_Interop) and every connection is GC-leaked (`with conn:` is a transaction, not a
  closer). Each open pays the private-seam's ~4 artifact verifications. Fix: held thread-local
  conn + route through `to_thread` + explicit close. PRE-EXISTING.
- **21126** · Library → Search/RAG panel runs `SELECT chunk_engine_version, COUNT(DISTINCT
  media_id) ... GROUP BY` over `UnvectorizedMediaChunks` (no index; rows-per-chunk table) ON
  THE EVENT LOOP per panel mount — and the panel remounts per destination switch
  (RAG_Admin/local_rag_admin_service.py:592-596; the `_maybe_await` seam evaluates the sync
  call before any await). Fix: `to_thread` + cache per session + index or maintained count.
  REGRESSION.
- **21127** · Research runs: per-op GC-leaked connects; engine launched as a loop coroutine
  (not thread) with a 30 s lease WRITE and a 2 s `get_run` read poll on the loop while a run
  is active (local_research_service.py:99-123; Research_Window.py:594,816-831). Fix: held
  conn, `to_thread` service calls, batch keepalive with progress writes. PRE-EXISTING (likely).
- **21128** · `messages_au` FTS trigger fires on ANY `messages` UPDATE (no `OF content`
  column list, re-confirmed in the v46 SQL), so usage-only and metadata-only flushes — now
  3–4 UPDATEs per chat turn — each re-tokenize and rewrite the full assistant reply into
  `messages_fts`. Fix: ~~`AFTER UPDATE OF content`~~ **`AFTER UPDATE OF content, deleted`**
  (corrected during implementation — see correction 3 below), preserving the v46/v47 guards.
  PRE-EXISTING shape, flush-count growth post-baseline.
- **21129** · Notes-sync executor: six `list_bindings` read-all sites (no LIMIT, no
  `root_id` index), five invoked WITHOUT `to_thread` from async methods — ~3·K full owner-set
  reads per sync batch, each also paying 21101's per-op connect+census until that lands
  (notes_sync_executor.py:1144,1256,1761,1978,2260,2682). Fix: indexed predicates +
  `to_thread`. REGRESSION.
- **21130** · TTS profile v3→v4 migration snapshots the entire reference-BLOB table
  (`wav_bytes`) into memory TWICE, first snapshot still held — up to ~1 GB peak at open under
  the 512 MiB store bound (TTS/profile_schema.py:1300-1316, 1439, 1468). Fix: project without
  `wav_bytes` (sibling `profile_migration_candidate.py:320` already does) + hash compare.
  REGRESSION.
- **21131** · Notifications event-state repository: per-op GC-leaked connects, 3+ per feed
  build on a 3 s-TTL Home cache in server mode (event_state_repository.py:85-106). Fix: clone
  the sibling `client_notifications_db.py:69-108` held-conn template. PRE-EXISTING.
- **21132** · Note-folder managed-membership recursive CTE anchors on the WHOLE closure and
  filters at the end; runs twice per Notes-tree refresh (note_folder_repository.py:1831-1861;
  off-loop, latency only). Fix: seed the anchor from the requested ids. REGRESSION.
- **21133** · Dead 10 s token-count producer: its entire consumer surface was retired by
  task-17653 (`#chat-log` no longer exists; footers compose `show_token_count=False`), yet the
  app-global timer still resolves footers and runs four failing queries every 10 s
  (app.py:11746; chat_token_events.py:103-181). Fix: delete the interval + periodic path;
  keep the estimator for on-demand callers. Producer PRE-EXISTING, dead state REGRESSION.
- **21134** · Small-residue batch (each verified on the pin): setup-modal snow ticks a
  full-screen refresh at 5 Hz for every not-yet-configured user (measured 9.9% of a core —
  honor reduced-motion by default on low-core machines or drop the tick rate);
  `unicode_casefold` Python-UDF in WHERE+ORDER BY on watchlists agent-tool queries
  (Subscriptions_DB.py:2908-2985); MCP execution log does two full-file JSON parses +
  fsync-on-close per tool invocation (MCP/execution_log.py:156); re-chunk per-chunk INSERT
  loop → executemany (library_rechunk_service.py:265-271); GC-leaked `with conn:` closes in
  sync_state/event_state/writing/research/tamagotchi; `EnhancedStatusWidget` recompose-per-
  status-message during ingest (status_widget.py:82-140); media-viewer match-nav restyles the
  whole document per click (library_media_content.py:16-53 — cache match list, restyle two
  lines); trajectory brush-drag rebuilds the ledger DataTable per mouse-move
  (trajectory_screen.py:861-867 → sync `_render_ledger`); `CAPABILITY_REGISTRY` builds 1,323
  frozen dataclasses (62% server-only) + runs `validate_registry_completeness()` in
  production at every import (registry.py:1358,1414 — move validation to tests, lazy-build
  the server partition); dormant sqlite owners to verify-then-retire
  (`Sync_Interop/notes_mirror.py`, `Widgets/Tamagotchi/tamagotchi_storage.py` never imported,
  Kanban boot connect with no UI, `Third_Party/aider/repomap.py` diskcache with no prod
  caller).

---

## What the burn-down changed (added 2026-08-23 at close-out)

Fifteen merges closed every P0 and P1 in this review. Numbers below are the measurements
recorded when each PR merged; a cell reads "—" where no before/after number was measured, and
none has been inferred or estimated here.

| Task | PR / merge | Measured before → after |
|---|---|---|
| 21100 first-boot migration wall | #2001 `41a240ccd` | `TldwCli` construction @100k messages **1.248 s → 0.693 s**, with FTS fully off the boot path; backfill made SIGKILL-resumable |
| 21101 notes device-state store | #1992 `56e2de875` | receipt transaction **92× faster**; statements per read **~60 → 3** |
| 21102 Chunking import chains | #1994 `d60ebe1d0` | import closure **1831 → 1757** modules; Chunking modules at boot **43 → 0**; warm app import **~811 ms → 731 ms** (7 chains broken; the survey had found 6) |
| 21103 Persona_Buddy / PIL defer | #2002 `6c0abdba7` | **−80 modules**, **−1.28 s cold import**; 4 independent PIL chains broken (the buddy chain had masked three) |
| 21104 bs4 boot guard | #1985 `3c3c919fc` | — (outcome: a base install can import the app; new ratchet `Tests/Packaging/test_extras_import_closure.py`) |
| 21105 lazy feature databases | #2008 `92f9dba52` | boot files opened **35 → 27**; **~90 DDL statements** off boot |
| 21106 Actor_Packs recovery | #1988 `908b802da` | CSS cliff guard re-armed **green at 47 sources**; nav tests **99 red → 130 passed**; cleared ~286 pre-existing Scheduling/Watchlists reds |
| 21112 notes-sync gate + watcher backoff | #2009 `30c7e1fe9` | zero-profile boots create **no state file at all**; quiet watcher **60 → 8 scans/min** |
| 21114 transcript drag-selection | #2007 `898cd8852` | body wraps per drag **151 → 1**; mouse-move handler **32× / 213× faster** |
| 21115 CSS bundle-ride + ratchet | #2004 `82650cc1f` | modal-heavy session stylesheet sources **70 → 45** (cliff 64; ~19 headroom) — the cliff was confirmed crossed pre-fix; new-class count honestly corrected **34 → 25** |
| 21117 Inspector right-rail scroll | #2016 `7489a0ec8` | — (pure-scroll path split from the layout reconcile; no before/after number recorded) |
| 21118 keystroke workspace memo | #2010 `736359202` | registry calls per 20 keys **25 + 25 → 0 + 0**; staged-launch evidence-bundle parses **11 → ≤1** |
| 21124 `get_cli_setting` fast path | #2005 `8e949873e` | warm reads **100 → 0** lock acquisitions; worst reader stall **18.2 ms → 3.7 ms**; write-path parses **4 → 2** |
| 21160 config_profiles circular import (hotfix) | #2003 `ae018308b` | — (unmasked by 21102's facade; three edges deferred to use-site) |
| 21200 restore the 21103 boot-path guard | #2019 `99005884` | — (another session's Actor Packs activation had put PIL and `Persona_Visual` back on the boot path, undoing 21103's win; the guard existed at their merge, CI simply was not enforcing yet) |

Two defects were found and fixed **because of** this work rather than being on its list: a
latent pre-existing FTS corruption (an unguarded `messages_ad` trigger on a tombstoned hard
delete, silent doclist poisoning rather than a raise), and `v45_to_v46` missing from every
packaging list (part of TASK-19860) — both inside 21100.

### Landed after close-out

Findings that were still open when the table above was written, measured the same way.

| Task | Measured before → after |
|---|---|
| 21121 changed-files guard per run tick | Per 25 simulated run ticks with a reply streaming, the guard alone: `messages_for_session` **25 → 0**, message copies **10,025 → 0** (400-message session; **1,025 → 0** at 40), guard wall time **32.1 ms → 0.02 ms** (**2.87 ms → 0.01 ms** at 40). Identical with a marker present (10,050 → 0) and the reported scope byte-identical in every arm, before and after. |

## Corrections found during implementation (added 2026-08-23 at close-out)

Three statements in the findings above were proved wrong by the implementations they produced.
They are corrected in place and listed here so the record stays trustworthy rather than
silently edited.

1. **Finding 21119 undercounted the walks and miscounted the invocations.** The review said the
   dismissal handler runs "two" full-screen queries and is invoked on both MouseDown and Click
   of every press (~4 traversals). Measured truth: each handler invocation costs **three**
   walks — `_remove_selection_menu` runs its own query in addition to the two the review
   counted — and the handler runs **once or twice per physical press depending on who swallows
   the Click**. A composer press is 1 invocation = **3 walks**; a rail press is 2 invocations =
   **6 walks**. The finding's direction and its fix were right; its arithmetic was not.
2. **Finding 21116 listed `_apply_library_row_toggle` as a hot per-click whole-screen
   recompose site. It is a false positive.** Instrumented at the base commit,
   `_apply_library_row_toggle` performs **0 recomposes and 0 widget constructions** per click:
   the recompose statement the review cited sits inside an **exception-fallback arm** that a
   normal toggle never reaches. The other sites named in that finding stand.
3. **Finding 21128 prescribed a fix that would have made soft-deleted messages searchable.**
   The finding's diagnosis was right — the trigger really did fire on every `messages` UPDATE,
   measured at **4 index rewrites per streamed turn**, `messages_fts_data` 55 → 12,636 bytes
   for one 400-token reply. Its prescribed shape, `AFTER UPDATE OF content`, was not: soft
   delete is `UPDATE messages SET deleted = 1 …` and never names `content`, so the trigger
   would not fire and the tombstoned row would stay in the index. Measured on a scratch matrix
   before any code was written (a direct `messages_fts MATCH` returned the tombstoned rowid),
   and re-proved by mutation — that shape turns all three `messages` cases in
   `Tests/DB/test_fts_soft_delete_index_witness.py` red. **Scope it precisely:** all six
   production `messages_fts` consumers re-filter on `m.deleted = 0`
   (`ChaChaNotes_DB.py:9131, 10318, 12496, 13935`; `RAG_Search/simplified/rag_service.py:2371,
   2402`), so the bad shape would have been an **index-layer** leak — deleted text left
   tokenized in `messages_fts_data` and reachable by a direct index query — **not** a
   user-visible search leak. Still a real regression of the task-19567 guarantee, which is
   stated at the index precisely because that consumer-side filtering is what kept the
   original trigger defect invisible. The shipped column list is `content, deleted`: every
   column the index stores, plus the column that decides membership. Shipped as v48 → v49
   (task-21128; authored as v47 → v48 and renumbered when the Console Library policy step took
   48 by merging first), one line different from v47's trigger.

4. **Seven unstarted findings were re-verified before dispatch, and five needed correcting.**
   After three of this doc's findings turned out wrong at implementation time, the remaining
   queue was re-read against dev `2be18842a` before any more of it was dispatched. That pass
   changed the plan more than the three earlier corrections did:

   | finding | as filed | after re-verification |
   |---|---|---|
   | 21107 | ~44 ms; TYPE_CHECKING the import | ~19 ms warm; that fix **cannot compile** — the spec table stores the schema classes as runtime values, and a test already allowlists the import as deliberate. Needs a lazy spec table instead. |
   | 21109 | 5 s lease stall in `__init__` | **sub-millisecond** for a user with no videos; the 5 s case needs a second instance mid-save. Deferring it past first paint would let the default `session` sweep delete a video published *that session*. |
   | 21110 | medium | **highest-value item left** — ~305 modules / ~232k LOC imported on the loop after a 1.5 s splash during which nothing else is scheduled. |
   | 21120 | three per-keystroke legs | mis-stated on all three; two fixes remove ~nothing (the "skip unchanged" guard can never fire while typing). Only the ghost-scan cap survives. |
   | 21123 | ~421 recompose sites; relocate the hook | count is ~111 (98 in `library_screen.py`); **under**-billed — the widget import precedes the enabled check and drags PIL onto the loop. The relocation would **break** the enabled case: `recompose()` removes the mounted buddy and an app-level owner would miss that signal. Split: early-return shipped, relocation deferred. |
   | 21127 | three legs | all three hold, cites drifted; `_update_row` opens **three** connections per update, worse than filed. Trap: `:memory:` mode shares one connection — closing it destroys the database. |
   | 21132 | per-interaction closure walk | recursion walks **upward**, so it is bounded by managed folders × depth and is **empty for the default profile**; already off-loop. Recommend cancel. |

   The pass also surfaced a finding worth more than the task it was found under: the composer's
   cursor-blink tick calls `Static.update()`, whose `layout` parameter **defaults to `True`**, so
   it arms a layout pass every 0.53 s while the composer is merely focused and idle — directly
   contradicting its own docstring ("must not trigger a layout recompute on every blink phase").
   Filed as task-21501.

   **The method lesson**: a finding's cost estimate decays as fast as its line numbers. Re-measure
   before dispatching, not after — five of seven is not a rounding error, and two of these would
   have shipped a fix for a problem that was not there.

---

## Verified clean — do not re-fix

- **All 13 audited 2026-08-11 fix families are structurally intact** (checked one by one):
  CSS consolidation mechanism · cost-chip TokenEstimateCache + fingerprint gating · 15452
  derivation memo + push equality gate · transcript reconciler `already_in_position` guard ·
  rail-search debounce-before-query · windowing/prune invariants under the 2.5× transcript
  rewrite (incl. 16851's under-lock re-check) · fence-delta Pygments buffering · Library
  canvas seam (whole-screen sites 147→105, none hot re-added except the 9 admin flows in
  21116) · Watchlists recompose removals (zero live screen-level recompose reactives) ·
  SubscriptionsDB held conn + off-loop scheduler · all five blocking-I/O fix families
  (media hub `to_thread` leaves, notes import thread worker, dictionaries scan replaced,
  config-write sites 149→~133 net DOWN, star-toggle off-loop) · nav-overflow signature gate +
  scheduled Ollama probe · streaming invariants (O(1) chunk append, ~2 creates + 2–3
  auxiliary UPDATEs per turn — see 21128 for the FTS amplification, no per-chunk/per-tick DB
  writes, trajectory sidecar batched once per turn).
- **DB/ pragma census: every `DB/` module on the WAL+NORMAL held-conn template** (full census
  table in the review record; `Workspace_DB`/`client_notifications_db` carry the reference
  liveness-ping shape).
- **`config.py` read caching intact** (`_SETTINGS_CACHE`/`_CONFIG_CACHE` source-keyed; the
  lock-ordering defect is 21124, not a cache defect).
- **The new subsystems are boot-cost, not idle-cost**: none of Persona_Buddy, Actor_Packs,
  Persona_Visual, runtime_policy, Chunking, Research_Interop, Notifications, notes-sync (zero
  profiles), Skills/Sync/Writing/Kanban_Interop, tldw_api, Video_Generation runs a thread,
  timer, poll loop, or subscription at idle for a never-user. The feared Research `while
  True` loop is a per-run 30 s keep-alive; the Notifications event_observer is
  subscribe-model, never boot-started.
- **runtime_policy hot path**: ~5 µs per capability check, zero checks on
  keystroke/chat-send/screen-switch paths.
- **UI-responsiveness heartbeat** (1 s): trivial arithmetic, edge-triggered stall records
  drained by a daemon thread — no I/O on the measured loop.
- **Console run-tick family** (0.2 s sync / 10 s cost TTL / 0.1 s realtime): all gated and
  self-stopping. **Nav overflow tick**: signature-gated no-op. **Video preview / prompt-queue
  / background-effect timers**: scoped, gated, or default-off.
- **tldw_api lazy facade holds** except the two leaks (21106/21107); Video_Generation
  module-lazy except the `__init__` retention slice (21109).
- **Textual Pilot latency numbers are harness-inflated** — see the artifact disclosure above
  before quoting any absolute keystroke figure from this or future reviews.

## Environment notes for reproduction

Probe recipe: second worktree at the pin, own venv (`VIRTUAL_ENV=<wt>/.venv uv pip install -e
".[dev]"`), assert `tldw_chatbook.__file__` resolves into that worktree; scratch
HOME/XDG/`TLDW_CONFIG_PATH` with `[first_run] setup_completed=true`, `[splash_screen]
enabled=false`; configured-state probes need a valid-shaped `api_settings.openai.api_key`
(otherwise the Console mounts the setup modal and you measure the snow, not the app). App boot
writes to the scratch config and regenerates the worktree CSS bundle — keep probes out of the
review worktree. `pilot.pause()` costs O(mounted widgets) — use it for A/B only.
