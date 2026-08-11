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
