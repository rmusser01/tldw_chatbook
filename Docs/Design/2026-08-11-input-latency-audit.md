# Input-latency audit — dev `82b595049` (2026-08-11)

**Symptom:** 1–5 s of lag between clicks/inputs on constrained hardware; no issue on fast machines.
**Method:** seven parallel read-only code audits (click handlers, keystroke paths, timers, screen-switch,
recompose, DB layer, streaming/markdown) against a pinned worktree of dev `82b595049`; top claims
re-verified first-hand by reading the code; two live probes run isolated (scratch HOME/XDG/`TLDW_CONFIG_PATH`,
no repo or live-config writes). All timings below are from a fast M-series Mac — multiply roughly 3–5×
for constrained hardware, more for anything fsync-bound.

**Verdict:** the lag is compounding, not singular. The filed backlog is **task-15450 – task-15481**.
This document is the durable evidence record those tasks cite (the backlog CLI drops custom sections,
so measurements live here).

---

## Measured evidence

### 1. The Textual CSS parse-cache cliff is crossed (task-15450)

Textual 8.2.8's stylesheet parse cache is `LRUCache(64)` (`textual/css/stylesheet.py:153`); a sequential
scan over more than 64 sources evicts 100%. Live headless Pilot tour of all 13 hotkey destinations,
reading `len(app.stylesheet.source)` after each:

| step | screen | sources |
|---|---|---|
| boot | ChatScreen (Console) | 27 |
| ⌃3 | LibraryScreen | 32 |
| ⌃5 | PersonasScreen | 62 |
| ⌃7 | SchedulesWorkbench | **65 — cliff crossed** |
| ⌃9 | MCPScreen | 74 |
| F7 | LLMScreen | 92 |
| F9 | SettingsScreen | **93 (final)** |

After the tour, three consecutive `stylesheet.parse()` calls: **350 ms, 127 ms, 125 ms** — zero cache
benefit (a warm cache is sub-millisecond). Textual re-runs this parse whenever a widget class not yet
seen this session first mounts (screen switches, modals, deferred mounts). The repo has **183
`DEFAULT_CSS` declarations**; the bundle is 21,479 lines / 546 KB / 2,694 rules. Six modal classes
declare class-level `CSS`, each of which forces a full cold reparse + whole-app restyle on first open.
Probe recipe: scratch config with `[first_run] setup_completed=true` + `[splash_screen] enabled=false`,
`app.run_test()`, press ⌃1..⌃0/F7-F9, read `len(app.stylesheet.source)`, time repeated
`stylesheet.parse()`.

### 2. Screen switches rebuild everything, by design

Screens are never cached (deliberate — caching re-mounts raced in-flight teardown into a total UI
freeze, root-caused 2026-07-11; docstring at `app.py:7436-7445`). Per-switch synchronous chain:
first-visit **module import on the UI thread** (`screen_registry.py:39-52`; chat_screen ≈ 19.9k lines,
library_screen ≈ 26k, settings_screen ≈ 18.9k; ~161 ms at 11k lines measured in July → task-15472),
fresh construction + compose + per-widget CSS apply, then **awaited teardown of the outgoing
350–500-widget tree**. Fast-hardware today: Console ≈ 1.1–1.35 s, Watchlists 0.89 s (never profiled,
no deferral — task-15462), Roleplay 0.69–0.94 s. At 3–5× on constrained hardware this alone reproduces
the reported 1–5 s.

### 3. DB pragmas and construction (tasks 15463–15466, 15480)

Probe-verified effective settings (files created in isolation, PRAGMAs read back):

| DB | connection | journaling |
|---|---|---|
| ChaChaNotes / Client_Media / Prompts / Evals / AgentRuns | held thread-local ✓ | WAL but `synchronous=FULL` — fsync per commit |
| Subscriptions | held — but **rebuilt per service op** via `db_factory` (~52-stmt schema script; 3.4 ms vs 0.04 ms held, ~85×; first build 35 ms) | **DELETE+FULL** |
| Workspace | held ✓ (missing `isolation_level=None` — task-15480) | **DELETE+FULL** |
| Library_Collections | fresh per op; write-`BEGIN` for pure reads | **DELETE+FULL** |
| RAG_Indexing / client_notifications | fresh per op, **never closed** (GC only) | DELETE+FULL |
| Library_Ingest_Jobs | held ✓ | **WAL + `synchronous=NORMAL` ✓ — the template** (`Library_Ingest_Jobs_DB.py:57-61`) |

DELETE mode makes writers exclusive-lock readers; contenders wait up to sqlite's default **5.0 s busy
timeout** — a direct multi-second-stall candidate on slow disks. Research/Writing/Mindmap/search_history/
Sync_Client are dead code (task-15481) — skip, don't churn.

---

## Findings → tasks

**Console (hottest surface).** Cost chip rebuilds cost state unconditionally on every control-bar sync
including the 0.2 s run tick — the guard at `chat_screen.py:8266` gates only the repaint, and
`build_cost_snapshot` (`console_cost_tracker.py:485-508`) re-runs `_estimate_tokens_locally` per
usage-less row per call, with the per-char fallback since tiktoken isn't in base deps → **15451**.
Every printable keystroke rebuilds the Workbench strip ungated (handler at `:18904` bypasses the
`:18039` guard; `sort_children` invalidates the screen-wide `query_one` cache even on no-op order) →
**15452**. The transcript reconciler `move_child`s every mounted row every pass (`console_transcript.py:
2314-2318`), per tick and per transcript click → **15453**. Rail conversation search runs 6–10 sync
SQLite queries per keystroke *before* its debounce, then recomposes the workspace tray ×3 instances
(equality guard deliberately reverted after a click-targeting regression — `console_workspace_context.py:
546-560`) → **15454**. Conversation load mounts the full history one awaited `mount()` at a time →
**15455**. A streaming reply that is one long code fence re-runs Pygments over the whole fence per tick
(Textual `Markdown.append` only advances past completed blocks) → **15456**.

**Library.** 147 statement-level whole-screen `refresh(recompose=True)` sites — regrown past July's
task-281 fix (124) — with per-click sites confirmed across notes/toggles/strips/chooser/RAG; the Notes
canvas conversion is blocked by a ctor param shadowing the `sync_state` hook (`:21925`) → **15457**.
The media viewer full-document re-parses + remounts per match-nav click, and content search double-parses
per keystroke → **15458**. Warm visits compose 2–3× → **15459**.

**Watchlists.** Search reactives are `recompose=True` → ~220 widgets torn down per character typed →
**15460**. Section click = 2+ whole-screen rebuilds; tree click = 4 recomposes; z/Z/[/] rebuilds all
four regions → **15461**. Heaviest never-profiled screen → **15462** (profile first — the
defer-past-first-paint series proved hidden-widget weight predicts nothing when sync/DB-bound).
**15462 is now done and this line's numbers are superseded** — see its Investigation Notes. After
15460/15461/15463/15464, live push is **0.50 s, not 0.89 s** (−44%); the screen is widget-bound, only
**3% of its tree is hidden** (no deferral possible), and one whole push runs **13 loop-thread sqlite
statements / ~10 ms of application code**. With an empty feed the screen pushes in 200 ms — the median
of every other screen — so its entire excess is the 224 widgets of the 100-item article feed
(`_ArticleRow`/`_DayHeader` each wrap a `Static`). `compose`'s inline `resolve_latest_follow_item()`,
flagged above, measures 0.1–0.3 ms and is retired as a concern.
SubscriptionsDB rebuilt per op + scheduler due-checks run sqlite + `ET.fromstring` inline on the loop,
enabled by default on any tab → **15463**. Items feed selects `content` for list rows and sorts the whole
table on a computed datetime → **15464**.

**DB layer.** WAL+NORMAL uniformly → **15465**. Held-connection ports (with the `isolation_level=None`
lesson and a pre-port DML audit) → **15466**. Eager BLOB `repr` on character-card save
(`ChaChaNotes_DB.py:5916`) + `SELECT *` dragging image BLOBs into 500-row pickers → **15474**.
Workspace_DB latent silent-rollback fuse → **15480**.

**Blocking I/O in click paths.** Media hub: `run_worker(coroutine)` ≠ thread; every search/filter/item
click runs sync SQLite on the loop (`MediaWindow_v2.py:2387/:1188`) → **15467**. Notes import loop
self-documented as running on the main loop (`note_ingest_events.py:638-641`) → **15468**. Personas
dictionaries: leading-wildcard `LIKE` full scan of conversations per click + double record load + N+1 →
**15469**. Config rewrites in click paths (dictation = 8–10 full config.toml cycles, per parsing
keystroke; sidebar `ui_state.toml` per toggle; ~10 more sites) → **15470**. Small residue (star toggle
fsync, Study writes, emoji recents, TTS export, file-picker stat storm, whole-file reads per tree click)
→ **15471**.

**Recompose/refresh storms elsewhere.** Settings screen-level recompose per rail click (twice on
Overview; "Sync preview" full-screen ×2 for three Statics), Evals rail selection, speech panel ~200
widgets per dropdown + mount/resume double-dispatch → **15475**. The undebounced picker/filter family
(character picker 500 rows, session switcher unbounded, Logs 10k lines re-rendered per char, etc.) →
**15476**.

**Timers.** Ollama probe: blocking `socket.create_connection` on the loop every 3 s on the Models
screen; nav bar full overflow re-measure every 0.5 s forever, no change detection → **15473**. The rest
of the 33-timer inventory is well-gated; screens' timers die with their screens (screens are never
cached), and the July fixes (tick TTL cache, psutil, splash frame cache) all held.

**Bugs found in passing.** Media-viewer prompt search imports a nonexistent symbol — ImportError
swallowed per keystroke, feature silently never worked → **15477**. STTS paste box queries a switch
composed nowhere (NoMatches per keystroke) → **15478**. `character_voice_widget` reactive
self-assignment can never fire → **15479**. Dead schedulers/DB modules that look alive → **15481**.

---

## Verified clean — do not re-fix

- Streaming chunk arrival is O(1); rendering coalesced to 5 Hz with re-entrancy guards; persistence is
  exactly twice per turn; anchor-based tail-follow — all July verdicts held.
- Console send path, conversation-browser FTS search (debounced + threaded), Library's `to_thread`
  service seam, file-notes workspace (~25 `to_thread` sites), Settings heavy ops (`@work(thread=True)`).
- Token-count content-hash gate intact; the per-keystroke token-count path is dead code (zero callers).
- July fixes held: Console 0.2 s tick persisted-rows TTL cache (task-251), psutil non-blocking,
  Workspace/AgentRuns held connections (3011/3012), sql-logging lazy guards (except `:5916`),
  splash frame cache, single save_state per switch.
- No global `on_key` in app.py; bindings resolution cheap; agents/evals/MCP have no polling loops.

## Environment note

The checked-out venv's editable install currently resolves `tldw_chatbook` to
`.worktrees/task-2512-mcp-unified` — not the main checkout, and not the CWD. Any probe or
`python -c` import must assert `tldw_chatbook.__file__` and pin `PYTHONPATH` to the intended tree.

## Follow-ups filed (2026-08-13)

Residuals, review-round findings, and pre-existing red tests surfaced while closing out
task-15450 - task-15481, each verified live against dev before filing (one candidate
from the original sweep, RAG top_k config-isolation, turned out already resolved by
task-15512's test fix and was not filed; the chat_screen size-ratchet breach is already
tracked as task-3070).

- **task-15764** — Watchlists: move `URLMonitor`'s HTML extraction + difflib work off the event loop (the piece task-15463 deliberately left out).
- **task-15765** — Repair the v17-to-v18 ChaChaNotes migration fixture broken by v35-to-v36's `note_folders` table (supersedes the stale "V33->V34 compaction_representation" framing, which task-15730 already fixed).
- **task-15766** — Batch of 7 pre-existing red tests found as asides (library_collections_panel, TTS Protocol-isinstance, console coalescing flake, rail_sections, citation-sources test double, console chip-swap TypeError, QwenCloud test-dict drift).
- **task-15767** — File Notes back-navigation (`test_action_library_notes_files_back_returns_to_database`) broke when task-15503's destructive-reload confirm shipped.
- **task-15768** — Media hub local-mode reading-highlight CRUD is `AttributeError` end to end (naming mismatch found by task-15467).
- **task-15769** — Character JSON backup export crashes on any image-bearing card (found by task-15474).
- **task-15770** — Watchlists unread-badge count still scans instead of using task-15464's `effective_date` index.
- **task-15771** — Sweep non-callable `reactive([])`/`reactive({})` defaults shared by identity across widget instances (proven instance: `character_voice_widget`, found by task-15479).
- **task-15772** — STTS Select widgets (import-source, audiobook provider/format) compose `options=` backwards, so set-value calls silently fail.
- **task-15773** — `ChapterEditorWidget`/`Select` mount race under high-volume DataTable population (dodged, not fixed, by task-15478).
- **task-15774** — Library media viewer search status/nav controls sit below the fold at 80x24 (task-15458's macOS re-verification).
- **task-15775** — Watchlists screen's `region_layout` reactive default disagrees with the shipped first-run default, composing then discarding the Inspector rail every visit.
- **task-15776** — Watchlists: collapse `_ArticleRow`/`_DayHeader` into one self-rendering widget (the one lever task-15462's profiling handed over, ~15-18% of a push).
- **task-15777** — Console transcript: unbounded reveal on a far jump, and a scroll-back reachability ceiling at the low prune watermark (task-15455's residuals).
- **task-15778** — Watchlists: batch the cold Read-tab region swap and drop `_build_detail_pane`'s pre-mount seeding recompose (task-15461's residuals #1/#4).
- **task-15779** — Watchlists artifacts pane: briefing selection destroys the briefings `DataTable` and its keyboard focus (task-15461's residual #3).
- **task-15780** — Verify-then-retire `CCPDictionaryEditorWidget`/`CCPPromptEditorWidget` (zero production callers; the CCP `__init__` docstring already says the chrome was retired).
- **task-15781** — Verify-then-trim `NotesSyncService`'s `SyncProfile` CRUD surface (zero production callers beyond `sync_folder`).
- **task-15782** — Repair `test_options_persist_to_config`'s stale hardcoded expectation (found by task-15470, unmasked once its `run_worker` crash fix stopped swallowing it).

## Second-wave follow-ups (2026-08-16)

Filed at the close of the follow-up burn-down (the 34 tasks above all merged). Each
candidate was re-verified live against dev `ee741cf10` before filing; three candidates
from the reviews' filing queue were dropped as already resolved or already tracked
(the stale `test_current_schema_version_is_37` contract red — fixed by `4a2d48046`;
the ChatScreen `_wait_for_production_chat_screen` gate cluster — 141/141 green at HEAD;
the Library note capability-matrix reds — already tracked as task-16480).

- **task-16835** — Wire or retire the multi-item review batch-analysis path (`app.llm_api_client` never assigned; `BatchAnalysisStartEvent` has no poster since the widget's deletion).
- **task-16836** — TTS export claims: key by path and honour them in `_discard_tts_artifact` (16199 review F2+F3, merged: one path-keyed redesign covers both).
- **task-16837** — Wire or retire the TTS export feature (`TTSExportEvent` never posted; handler is not a MessagePump; F6/F4 residuals become live if wired).
- **task-16838** — Watchlists: per-(subscription,url) in-flight guard against scheduled+manual double-reporting (15764 review finding 1).
- **task-16839** — Fix and bound `calculate_change_percentage` (autojunk-degenerate pct=1.0 on large Latin pages; quadratic to ~7 min at the 10 MB cap; 15764 review finding 5, merged as one entangled redesign).
- **task-16840** — Replace the ChaChaNotes rollback registry with bootstrap-under-patched-`_CURRENT_SCHEMA_VERSION` fixtures (15765 review F3; registry has already grown v38/v39 entries).
- **task-16841** — SiteConfigSettings `#auth-type-select` is backwards (6th instance of the class) + repo-wide backwards-Select AST sweep and permanent guard.
- **task-16842** — `stts_profile_library` flake family: five timing-sensitive focus-assertion tests, one reproducing standalone.
- **task-16843** — Extend the reactive-default guard to shared mutable instance defaults (`reactive(SomeClass())`, 5 sites confirmed at HEAD; 15771 review F2).
- **task-16844** — `FileListItemEnhanced.compose` passes `tooltip=` to `Static`: any non-empty `FileListEnhanced` crashes on compose (15771 review, live traceback).
- **task-16845** — Study: wire or remove the four undispatched buttons (add-child / create-course / generate-guide / add-milestone) and the placeholder Structured Learning pane (16195/16196 residuals).
- **task-16846** — Wire up or retire `ScraperBuilderWindow` (ADR-020 Accepted, nav-unreachable; now composes fine after task-15991).
- **task-16847** — `chat_screen.py:3054` bare `self.call_from_thread(` keeps the repo-wide TASK-929 guard red on dev.
- **task-16848** — Rewrite the stale `Docs/Features/notes_bidirectional_sync.md` (removed SyncProfile surface; two nonexistent widget files; 15781 residual).
- **task-16849** — Chapter editor in-place edits (add/split/merge/delete) never refresh the now-truthful chapter table (15773 review residual 4).
- **task-16850** — `ChapterDetector` emits an empty placeholder chapter per titled heading, ~2x rows (15773 review residual 5).
- **task-16851** — Console transcript: a head-pinned selection disables the prune while tailward hydration reveals — unbounded mount (15777 round-3 finding, plus the one-frame End-during-prune race).
- **task-16852** — Watchlists artifacts: script selection still rebuilds the scripts table inside the detail region (15779's disclosed one-level-down residual).
