# Core-runtime code review — 2026-09-17

**Verdict:** The single biggest thing is that this codebase keeps re-implementing the same few primitives — how a timestamp is written, how text is escaped, how a worker is started — and the copies have drifted far enough to lose user data, blank out user text, and crash the app from four different buttons. The single biggest risk is that none of it is guarded: the size ratchets are red in 18 rows, and there is no check anywhere on timestamp format, markup escaping, or the worker contract that three of the four P0s violate.

## Scope and provenance

| Item | Value |
|---|---|
| Worktree | `/Users/macbook-dev/Documents/GitHub/tldw-review` (detached, created for this review; nothing committed, nothing stashed, no branch switched in the main checkout) |
| SHA | `d8fb4053f9a27a799d5cdb8ee58f7fd1de91efce` — "Merge pull request #2706 from rmusser01/codex/transcription-failure-diagnostics"; the same SHA the prompt's seed counts were measured at. `origin/dev` moved to `e89f28d751` during the run; **not** reviewed. |
| Tree clean | `git status --porcelain` → empty, before the run, after `preflight.sh`, and at the end |
| Python / Textual | 3.12.11 (the main checkout's `.venv/bin/python`, run from the worktree root with `PYTHONPATH=<worktree>`) / Textual 8.2.8 |
| Preflight | `PYTHON=.venv/bin/python ./scripts/preflight.sh` → **all derived-artifact checks passed, exit 0** (CSS bundle sync, Canvas Mermaid assets, profile-owned path census 54/51, diagnostic inventory 611 owners, 4,041 task files with no duplicate ids, chachanotes 116 tables allowlisted, 304 index plan pins). Log: `candidates/preflight.log` |
| Ruff fatal-only (`--select E9,F63,F7,F82`) | **0** in entry/config, `Chat/`, `Agents`+`Tools`+`MCP`, `DB/`, `LLM_Calls/`, `Event_Handlers/`, `Utils/`, `RAG_Search/`, the four screens, `Widgets/`; **1** in UI modules — `UI/Library_Modules/library_notes_controller.py:5009 F821 Undefined name NoteImportExecutor` (annotation-only; `from __future__ import annotations` at :501; the module imports fine — P3) |
| ADRs | 217 files in `backlog/decisions/` (list: `candidates/adr_list.txt`) |

**Two environment facts that shaped every command** (and cost the first hour):

1. Running Python with the working directory *inside* `tldw_chatbook/` silently resolves the editable install to the **main checkout**, not the worktree. Every command in this review runs from the worktree root with `PYTHONPATH=<worktree>`, and each subagent verified the resolved `__file__` before trusting a result.
2. Importing the package against the machine's real profile exits at import: `config.py:6` calls `admit_startup()` (ADR-126) which raises `SystemExit: Recovery required: recovery_scope_uncertain`. All verification runs use an isolated `HOME`/`XDG_*`/`TLDW_CONFIG_PATH` scratch profile, the same shape `Tests/conftest.py` establishes.

### Tier-1 line counts (measured in the worktree; `.py` only)

| Area | Paths | Lines | Files |
|---|---|---:|---:|
| Entry / config | `app.py`, `config.py`, `Constants.py`, `model_capabilities.py`, `Logging_Config.py`, `emergency_stop.py`, `runtime_policy/` | 40,463 | 22 |
| Chat | `Chat/` | 207,504 | 192 |
| Agents / tools | `Agents/`, `Tools/`, `MCP/` | 84,378 | 110 |
| Data | `DB/` | 70,357 | 36 |
| Providers | `LLM_Calls/` | 22,561 | 19 |
| Events | `Event_Handlers/` | 16,816 | 40 |
| Shared helpers | `Utils/` (excl. `Splash_Screens/`) | 24,868 | 64 |
| RAG | `RAG_Search/` | 27,572 | 46 |
| Main screens (4 files) | `UI/Screens/{chat,library,settings,personas}_screen.py` | 108,119 | 4 |
| UI modules | `UI/{Console,Library}_Modules/`, `UI/Navigation/`, `UI/MCP_Modules/`, `UI/Persona_Modules/` | 121,585 | 123 |
| Widgets top-level | `Widgets/*.py` | 24,763 | 66 |
| Widgets sub-packages | `Widgets/{Console,Library,Persona_Widgets,Settings_Widgets,Chat_Widgets}/` | 138,869 | 192 |
| **Tier-1 total** | | **887,855** | **914** |
| Interop cluster (reviewed as a class, §"Duplication clusters") | 31 `*_Interop/` packages (none under a Tier-1 root) + 33 `*_scope_service.py` / `server_*_service.py` elsewhere | 68,545 + 27,442 | 178 + 33 |

### Coverage

**All 29 planned slices completed: 887,855 of 887,855 Tier-1 lines (100 %).** Three session usage limits killed subagent runs mid-flight (23:50 PT, 04:50 PT, and a third earlier); from the first onward every subagent was required to write its report incrementally, so interrupted runs were resumed from their partials rather than restarted.

A note on how this run went, because it affects how the coverage should be read: the review was declared finished once at 21 slices (69 %). That was the orchestrator's error, not a budget limit, and it was corrected. The eight slices added afterwards — the `Widgets/` tree, the rest of `Chat/`, and the navigation/MCP modules — contained **three of the four P0s in this report**. Nothing about those eight slices was lower-risk; they were simply last in the queue.

| Slice | Lines | Read in full | Sampled | Mechanical only | Findings |
|---|---:|---|---|---|---|
| ENTRY-app (`app.py`) | 21,050 | 1–21050 | — | — | P1 1, P2 2, P3 8 |
| ENTRY-config (`config.py`, `Constants.py`, `model_capabilities.py`, `Logging_Config.py`, `emergency_stop.py`, `runtime_policy/`) | 19,413 | all but two string literals | `config.py` 3572–5708, `Constants.py` 195–1621 + 1632–1935 | — | P1 2, P2 4, P3 7 |
| CHAT-controller (`console_chat_controller.py`) | 29,048 | 1–29048 | — | — | P2 5, P3 6 |
| CHAT-store (`console_chat_store.py`) | 22,245 | 1–22245 | — | — | P2 3, P3 6 |
| CHAT-bridge (`console_agent_bridge/_provider_gateway/_trace_service/_runtime`) | 29,362 | all 4 files | — | — | P2 3, P3 13 |
| CHAT-rest-1 (`Chat/`, 50 files) | 43,946 | 26 files (~11,500 lines) | by candidate row | all 50 swept | P1 1, P2 5, P3 5 |
| CHAT-rest-2 (`Chat/`, 74 files) | 42,794 | ~9,000 lines read as prose | 7 files linearly | all 74, 12 sweeps + AST import resolution | P1 2, P2 3, P3 4 |
| CHAT-rest-3 (`Chat/`, 62 files incl. `citation_*`) | 40,109 | 7 files | 20 by named range (~6,900 of 22,900) | 35 | P1 1, P2 6, P3 2 |
| AGENTS (`Agents/`) | 40,468 | 14 core files incl. `agent_service.py` | 4 at candidate rows | ~20 small | P2 3, P3 9 |
| TOOLS-MCP (`Tools/`, `MCP/`) | 43,910 | 21 files (~21k) | 9 at candidate rows | 21 small | P1 2, P2 8, P3 4 |
| DB-chacha (`ChaChaNotes_DB.py`) | 24,180 | 1–24180 | — | — | P1 1, P2 4, P3 8 |
| DB-media-base (`base_db.py`, `sql_validation.py`, `Client_Media_DB_v2.py`) | 11,932 | all 3 | — | — | **P0 1**, P2 3, P3 7 |
| DB-rest (`DB/` remainder) | 34,245 | 6 large + 10 small | 6 by symbol cluster | 6 | P1 2, P2 8, P3 6 |
| LLM (`LLM_Calls/`) | 22,561 | all 19 files | — | — | P1 3, P2 6, P3 4 |
| UTILS (`Utils/` excl. `Splash_Screens/`) | 24,868 | 63 of 64 files | `Splash_Strings.py` | — | P1 1, P2 4, P3 12 |
| RAG (`RAG_Search/`) | 27,572 | 6 of 14 files >600 lines (~12,900) | ~7,400 by symbol cluster | rest | P1 2, P2 11, P3 8 |
| EVENTS (`Event_Handlers/`) | 16,816 | 18 files | 8 largest | 5 dead files | P1 2, P2 11, P3 14 |
| UI-chat (`chat_screen.py`) | 25,215 | 1–25215 | — | — | P1 3, P2 3, P3 5 |
| UI-library (`library_screen.py`) | 35,680 | 1–35680 | — | — | P1 1, P2 1, P3 8 |
| UI-settings (`settings_screen.py`) | 30,810 | 1–30810 | — | — | P1 1, P2 3, P3 8 |
| UI-personas (`personas_screen.py`) | 16,414 | 1–16414 | — | — | P2 4, P3 5 |
| UIM-console (`UI/Console_Modules/`) | 47,714 | 18 of 42 (~39,700) | 4 by candidate row | 20 small | P1 3, P2 4, P3 7 |
| UIM-library (`UI/Library_Modules/`) | 47,026 | 2 in full + the moved-body region of 4 more | 6 controllers | 34 small | P2 5, P3 3 |
| UIM-nav-mcp-persona (`UI/Navigation`, `UI/MCP_Modules`, `UI/Persona_Modules`) | 26,845 | 13 files (~18,400) | 9 by symbol | 13 small (all 35 AST-swept) | **P0 1**, P1 2, P2 4, P3 4 |
| W-top (`Widgets/*.py`) | 24,763 | 8 files (~6,900, incl. all five >800 lines) | 27 by symbol cluster | 31 | **P0 1**, P1 1, P2 5, P3 8 |
| W-console-1 (`Widgets/Console/`, 60 files) | 34,296 | 15 files incl. all 7 >1,000 lines | 24 by named range | 21 | P1 4, P2 2, P3 5 |
| W-console-2 (`Widgets/Console/`, 34 files) | 33,727 | 7 files >1,000 lines (24,537 lines) | — | 27 (all candidate rows + sweeps) | P1 2, P2 3, P3 4 |
| W-library (`Widgets/Library/`) | 40,112 | 13 largest | `library_file_notes_workspace.py` ~2,600 by symbol | 4 tiny | **P0 1**, P1 11, P2 17, P3 16 |
| W-persona-settings-chat (`Widgets/{Persona,Settings,Chat}_*`) | 30,734 | 10 files | 5 large partially + whole-file grep verification | rest | P1 1, P2 6, P3 7 |
| **Total** | **887,855** | | | | **P0 4 · P1 50 · P2 146 · P3 209** |

Every slice's own report — with its full evidence, candidate dispositions, verified-fine list and unverified table — is in `slices/<SLICE>.md`. Mechanical candidate censuses are in `candidates/`.

### Already handled (open tasks that pre-empt a recommendation)

| Task | Area it covers |
|---|---|
| task-1378 · task-31202 | Split `settings_screen.py`; its missing size-ratchet row |
| task-2542 | `_toast` duplicated between `mcp_inspector.py` and `mcp_workbench.py` |
| task-586 · task-609 | Image-gen adopts `Utils/egress.py`; consolidate SSRF layers |
| task-31502 | Quiescence registry taxes every ChaChaNotes statement with a shared lock |
| task-31572 · task-31584 | Library media focus-region helper; dead progress helper |
| task-31650 · task-32089 · task-32170 · task-32013 · task-32199 | Library decomposition hygiene (controller globals, canvas_syncs dispatchers, phase-C media extraction, exclusion debt, test health) |
| task-194 | `console_model_popover` uses the shared provider display-name catalog |
| task-19867 | `VALID_TABLES` media/prompts entries drifted with no guard test |
| task-32499 | Agent-routing helper placement (ADR-147 follow-up) |
| task-25704 · task-287 | `DEPENDENCIES_AVAILABLE` flags no probe populates (the cause of the `doctor` P1 below) |
| task-2902 · task-26834 | Console: defer hidden rail past first paint; interactive stalls |
| task-30019 | Decide legacy Collections migration or retirement |
| task-17387 | `summarize_with_kobold`/`_tabbyapi` return an error string (the P1 below adds new evidence) |
| task-1320 | Move screen mount I/O off the App message pump |
| task-27010 · 26962 · 26984 | Ruff formatter debt |


## Executive summary

**4 P0 and 50 P1 findings.** Every P0 and the headline P1s were re-run by me after their slice filed them; commands and output are in `phase4-verification.md`. One claim did not reproduce and is recorded as contested rather than dropped; one was corrected upward to P0 after I traced its caller chain; one turned out worse than filed; one of my own enumerations produced a false positive that I caught and retracted before it reached this table.

| # | Sev | Dim | Symptom | Where | Size |
|---|---|---|---|---|---|
| 1 | **P0** | D1 | A reading-list re-import leaves an implicit transaction open on the Media connection; every later Media write on that thread is rolled back at close. Proven: 2 media rows expected, 1 found; a keyword and a read-it-later row vanish | `DB/Client_Media_DB_v2.py:5530` ← `Media/local_media_reading_service.py:3525` | S / M |
| 2 | **P0** | D1 | Pressing **Escape** in the Prompts work pane kills the app: a "More actions" flag survives a recompose into a branch that never composes the region its key handler dereferences | `Widgets/Library/library_prompts_canvas.py:549` | S |
| 3 | **P0** | D1 | Clicking "Advanced…" then "Hide advanced" in the MCP inspector while the section load is in flight kills the app: the awaited load resumes into a subtree `_hide_advanced` already removed, and nothing cancels it | `UI/MCP_Modules/mcp_inspector.py:3418` | S |
| 4 | **P0** | D1 | The Audio Troubleshooting dialog passes a **sync** method to `run_worker` with no `thread=True`; Textual raises `WorkerError` and `exit_on_error` (default True) takes the app down. Reachable: route `stts` → STTS window → dictation window → dialog | `Widgets/audio_troubleshooting_dialog.py:246` | S (one kwarg) |
| 5 | P1 | D1 | **`rich.markup.escape` does not protect Textual 8.** Its pattern only escapes tags starting `[a-z#/@`, so any uppercase-initial bracket is eaten by `Content.from_markup`: a note titled `[TODO] Q3 plan` renders as ` Q3 plan` and `[IMPORTANT]` renders as an **empty row**. ~40 sites across Library rows and Console labels | `Widgets/Library/library_rail.py:377` + 27 more; 12 Console sites | M |
| 6 | P1 | D1 | A reviewed flashcard is not "due" until the UTC day *after* its due time, and study stats drop the boundary day — `next_review` is written `T…+00:00` and compared lexically to `CURRENT_TIMESTAMP` | `DB/ChaChaNotes_DB.py:21570` vs `:21593`/`:21622` | S |
| 7 | P1 | D1 | The RAG search cache leaks its byte counter on every TTL prune and then **stops caching permanently**: dead at prune cycle 42 with zero entries cached, counter pinned at 1,022 KB of a 1,024 KB cap, no recovery path | `RAG_Search/simplified/simple_cache.py:1055` | S |
| 8 | P1 | D1 | `config.get_api_key()` hands out placeholder and un-stripped credentials the shared validity rule rejects, on 5 live spend paths; a second accessor disagrees with the chat spend path about env-vs-config precedence, and **both orders are pinned by test names** | `config.py:9083`, `:9074-9087` | S (+ L for the ruling) |
| 9 | P1 | D1 | One transient read error on `mcp_permissions.json` is treated as corruption: the live policy file is renamed `.bak` and permissions reset — kill switch ON→OFF, a tool set to Off resolves Allow | `MCP/permission_store.py:747` | S |
| 10 | P1 | D1 | The always-on `CalculatorTool` evaluates model-supplied `**` and `str * int` unbounded — `'ab' * 10**7` returns a 20 MB string in 2 ms; a prompt-injected `9**9**9` pins a core for the process lifetime | `Tools/tool_executor.py:181` | S |

The other 45 P1s are in Findings. The ones a user hits soonest: the irreversible "Delete stored Full captures" confirmation **never appears** when the title contains `[/…]` (a `MarkupError` inside a dialog 45 modules import); seven of eight streaming handlers leak the HTTP response on Stop; every local LLM server is stopped without a process group, so a worker survives while the app reports "stopped" and holds the port; a settings keystroke costs 300–1,000 ms and a Speech & TTS panel open costs 65 ms of re-read config; the Console's left-rail Model section never updates; "open video externally" and the Transformers "Browse models dir" button can never work at all; Gemini and Cohere streamed turns record no token usage; `doctor` reports all 70 optional feature groups missing with torch and numpy installed; and a File Notes review dialog pauses the folder poll with nothing to resume it.

**Three defect classes produce most of this list, and all three are duplication with drift.**

*Markup escaping.* Two slices independently found that `rich.markup.escape` is the wrong tool for Textual 8, and the repo already contains the right one — `library_rag_state.py:520 _escape_all_brackets`, whose own comment names the `[TODO]` case — applied to exactly one surface. Every pinning test for the broken sites uses a lowercase tag, the one case the broken escape handles. Related P1s sit at the opposite extreme: two sites *double*-escape into a markup-off surface, so `R&D Report` reaches the user as `R&amp;D Report` and `summarize [draft]` as `summarize \[draft]`.

*Timestamps.* Twelve UTC string shapes from 55 helper copies, a thirteenth from SQLite's `CURRENT_TIMESTAMP`, and 100 lexical comparisons over the result. Three are already wrong for users (flashcard due dates, media cleanup cutoffs, a voice-promotion validator that rejects the shape its own writer produces); one was worked around with `julianday()` in conversation pagination. `console_trace_settlement.recover_open_calls` is the in-repo answer — it compares three shapes correctly through `julianday()` — and nothing else uses it.

*Transactions and workers.* The store template lets a write method assume an outer transaction; two stores have one that does DML on a held legacy-isolation connection anyway (data loss in Media, latent in Prompts). The worker contract has the mirror-image problem: a sync callable passed to `run_worker`, or an `await` that resumes into a removed subtree, both take the whole app down because `exit_on_error` defaults to True. Three of the four P0s are that one contract.

Fixing any of these at the copy level is a sweep; fixing them at the helper level is the consolidation in "Duplication clusters".


## Findings
One `###` per finding, P0 → P3, then by dimension. Each carries its slice name; that slice's report (`slices/<SLICE>.md`) has the surrounding evidence, candidate dispositions and unverified rows.
Counts across all 29 slices: **P0 4 · P1 49 · P2 146 · P3 209** (D1 104, D2 50, D3 124, D4 130). Three P0s were filed as such by their slices; the fourth was filed P1 and corrected upward here. Two P1 rows describe one defect found from both ends of its call. Every P0 and the headline P1s were re-run by me — see `phase4-verification.md`.
### P0 [D1] — A bare DML call on `MediaDatabase`'s held connection leaves an implicit transaction open; every later `transaction()` on that thread silently borrows it, nothing commits, and the writes are rolled back at close (one shipped caller does this)  ·  _slice: DB-media-base_
- Where: `tldw_chatbook/DB/Client_Media_DB_v2.py:1325-1372` (`transaction()` borrows when `conn.in_transaction` is already true and then skips `commit()`), `:1072-1083` (the connection deliberately keeps legacy `isolation_level=''` — "task-22224 EXCEPTION … flipping requires this file's own commit/write-site census first … its own task"), `:5412-5413` and `:5530-5531` (`create_document_version` / `update_keywords_for_media` "Assumes called within an existing transaction" but write DML on `self.get_connection()` with no guard). Shipped bare caller: `tldw_chatbook/Media/local_media_reading_service.py:3525` (`db.update_keywords_for_media(media_id, merged_tags)` in `_materialize_reading_import_row`, no `db.transaction()` anywhere in `import_reading_items` → `_execute_reading_import_job` → that helper: `sed -n '2848,3485p' … | rg transaction\(` hits only four unrelated functions). The other three external callers ARE wrapped (`local_media_reading_service.py:4874-4880`, `Library/meeting_speaker_rename.py:369-394`).
- Evidence: `$PY $HOME/repro/media_repro2.py` → `after bare update_keywords_for_media: in_transaction = True` … `after later add_media_with_keywords: in_transaction = True | mid2 = 2` … `after close+reopen: media rows = 1 (expected 2) | keywords linked to mid1 = ['k1'] (expected ['k1','k2']) | read_it_later rows = 0 (expected 1)`. `media_repro.py` part A: the same for a bare `create_document_version` (`DocumentVersions rows after close+reopen (expected 2 if durable): 1`). `media_repro4.py` part D (the write done on another thread, as an executor would): the main thread's next `add_media_with_keywords` → `DatabaseError: Unexpected error processing media: database is locked` (immediately — WAL read→write upgrade against a live writer does not invoke the busy handler; `PRAGMA busy_timeout` on the connection is 10000 ms, verified).
- Why it matters: after one re-import of a reading list whose URL already exists (`merge_tags` defaults to True), every Media write on that thread for the rest of the session is uncommitted — the second manifestation is every OTHER thread's Media write failing with "database is locked". `MediaReadingScopeService.import_reading_items` (`media_reading_scope_service.py:1799-1817`) runs the sync local service inline via `_maybe_await`, i.e. on the event-loop thread, so the leaked transaction would sit on the UI thread's own connection. No UI/tool surface calls `import_reading_items` today (`rg -n -i "import_reading|reading_import" UI Chat Tools MCP Agents Library Tool_Packs app.py` → 0), which is why this is P1 and not P0; it becomes P0 the day one does.
- Recommended correction: root cause is the documented, still-unfiled Media half of task-22224 (`isolation_level=None` + explicit-BEGIN-only manager; the ChaChaNotes precedent is `ChaChaNotes_DB.py:3504-3514`). Grep of `backlog/tasks` for a task naming `Client_Media_DB_v2` + `isolation_level|22224` → none: file it. Interim S fix at the shared function, not per caller: have `create_document_version` and `update_keywords_for_media` open `with self.transaction() as conn:` themselves (a nested call joins the outer transaction exactly as today, a bare caller now gets BEGIN/COMMIT) — or at minimum `if not conn.in_transaction: raise RuntimeError("caller_transaction_required")`, the guard `base_db._SemanticMutationAuthorization._authorize` already uses (`base_db.py:615-616`).
- Size: S (interim guard) · M (isolation flip + write-site census) · ADR: no · Confidence: verified
- Pinning test: none observed. `Tests/Media/test_media_reading_scope_service.py:773,2262` fake the service; `Tests/Media_DB/*` run `:memory:` databases, which cannot observe a close-time rollback.
- Already covered: none (task-22224 covered ChaChaNotes; the Media docstring defers to "its own task", which does not exist)
- **Severity corrected by the orchestrator (was P1):** the slice rated this P1 because it could not show a shipped caller without an enclosing transaction. I traced the chain — `Media/media_reading_scope_service.py:1799` → `local_media_reading_service.py:2848` → `:2873` → `:3383` (per-row loop) → `:3485` → `:3525 db.update_keywords_for_media(...)` — with no `db.transaction()` anywhere on it. Data loss on a shipped path is P0. The sibling shape in `Prompts_DB.update_keywords_for_prompt` (DB-rest P2) stays P2: its only public wrapper, `Prompts_Interop.py:230`, has no caller.
### P0 [D1] — Escape kills the app: the Prompts work pane's "More actions" flag survives a recompose into a branch that never composes the region it dereferences  ·  _slice: W-library_
- Where: `library_prompts_canvas.py:549-555 on_key` (`self.query_one("#library-prompt-more-actions-region")`, unguarded) and the same shape at `:527 _toggle_more_actions`; the flag is set at `:230`/`:525`, cleared only at `:512` and in `on_key`; `sync_state:323-357` recomposes without resetting it; the region is composed only by `_compose_editor`.
- Evidence: probe `<SCRATCH>/probe_skills_escape_import.py` (real keypress, real focus, production compose branch), **re-run by me**:
  `textual.css.query.NoMatches: No nodes match '#library-prompt-more-actions-region' on LibraryPromptWorkPane(id='library-prompt-work-pane')` raised from `library_prompts_canvas.py:554 in on_key` → `App._exception`.
  Production sequence (each step traced in the controller): prompt open in the work pane → press `#library-prompt-more-actions` (flag True) → press Import… on the Prompts list → `library_prompts_controller.py:1595-1604` sets `_library_prompts_import_open` and `_library_prompt_work_pane_kwargs` (`:985-987`) forces `mode="list"`, `import_open=True`, so only the import row composes and the flag is never in the kwargs (`grep -rn "more_actions_open" tldw_chatbook` → skills-side hits only) → the same handler focuses `#library-prompts-import-path` → **Escape** bubbles from that Input to `on_key`.
  Two neighbours in the same file (`_open_more_collections:534`, `_open_more_history:542`) already wrap their `query_one` in `try/except NoMatches`.
- Why it matters: an ordinary two-click sequence followed by Escape terminates the TUI — the defect class this repo already recorded in task-32639 ("a re-run flake was a real app-killing unguarded `query_one`").
- Recommended correction: root cause first — reset `self.more_actions_open = False` in `sync_state` before `refresh(recompose=True)` (this also fixes the quieter twin: `:1673` composes `more_region.display = self.more_actions_open`, so the menu silently reopens on the next prompt). Then guard `on_key`/`_toggle_more_actions` like their two neighbours. The skills sibling avoids the whole class by keeping the flag screen-owned (`library_screen.py:25651-25653`).
- Size: S · ADR: no · Confidence: verified (reproduced twice, independently)
- Pinning test: `Tests/UI/test_library_prompts_canvas.py:802 test_prompt_more_actions_is_inline_and_escape_restores_opener_focus` pins only the happy path (region mounted) — not a decision, a gap.
- Already covered: none
### P0 [D1] — The Audio Troubleshooting dialog passes a SYNC method to `run_worker` without `thread=True`; Textual raises inside the worker and, because `exit_on_error` defaults to True, kills the app  ·  _slice: W-top_
- Where: `tldw_chatbook/Widgets/audio_troubleshooting_dialog.py:246-248`
  ```python
  self.audio_devices = await self.run_worker(self._get_devices_safe).wait()
  ```
  `_get_devices_safe` is a plain `def` (`:275`). Reached from `_initialize_audio` (`:229`), started at `:226` by `self.run_worker(self._initialize_audio())`.
- Evidence (verified by execution):
  ```
  cd $WT && source env.sh && PYTHONPATH=$WT $PY -   # minimal Textual app, same call shape
  ... textual/worker.py:343 in _run_async
      raise WorkerError("Request to run a non-async function as an async worker")
  <full Textual crash panel printed>
  run_worker(sync_callable) without thread=True ->
     {'outcome': "WorkerFailed: Worker raised exception: WorkerError('Request to run a non-async function as an async worker')"}
  ```
  Mechanism in the installed Textual 8.2.8: `worker.py:342-343` rejects a sync callable on the async path; `worker.py:149` `exit_on_error: bool = True`; `worker.py:382-384` `if self.exit_on_error: app._handle_exception(WorkerFailed(self._error))`.
  Reachability: `grep -rn AudioTroubleshootingDialog tldw_chatbook/` → pushed at `UI/Dictation_Window_Improved.py:735` and `:1047` (`_show_troubleshooting`), and that module is imported by `UI/STTS_Window.py:105`.
- Why it matters: the dialog exists to diagnose a broken microphone; opening it takes the whole app down with a traceback panel. The local `except Exception` at `:265` catches the `WorkerFailed` that `.wait()` re-raises and writes "❌ Unexpected error" — but `app._handle_exception` has already fired from inside the worker, so the swallow does not save the app; it only hides the cause from the log.
- Recommended correction: `self.run_worker(self._get_devices_safe, thread=True, exit_on_error=False)` — device enumeration is exactly the blocking C call a thread worker is for. (`exit_on_error=False` because the method already returns `[]` on failure, `:277-281`.)
- Size: S · ADR: no · Confidence: verified (mechanism executed; end-to-end button press is in Left UNVERIFIED)
- Pinning test: none — `rg -n audio_troubleshooting Tests/` finds nothing.
- Already covered: none

---
- **Re-verified by the orchestrator:** confirmed against Textual's own source — `Worker._run_async` raises `WorkerError("Request to run a non-async function as an async worker")` for a plain callable, `Worker._run` catches it generically, and `exit_on_error` defaults True. Reachability traced: route `stts` (`screen_registry.py:194`) → `UI/Screens/stts_screen.py` → `STTSWindow` → `UI/Dictation_Window_Improved.py:732` pushes the dialog. **Correction to the slice's count:** only `:246 run_worker(self._get_devices_safe)` is this defect; `:226` passes an already-called `@work`-decorated coroutine, which my own first enumeration misread — retracted before it reached the summary.
### P0 [D1] — pressing "Hide advanced" while the MCP inspector's section load is in flight kills the whole app  ·  _slice: UIM-nav-mcp-persona_
- Where: `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py:3418` (`_load_advanced_section`, the two DOM reads after `await self._service.load_section(section)`: `query_one("#mcp-adv-content", Static)` and, on the next line, `self._refresh_advanced_actions()` which reads `#mcp-adv-action-select`/`#mcp-adv-payload`/`#mcp-adv-run`). Dispatched at `:3536` / `:3395` as `run_worker(partial(self._load_advanced_section, ...), group="mcp-adv-section", exclusive=True)` — Textual's `exit_on_error` defaults to **True**. `_hide_advanced()` (`:1639`) removes `#mcp-adv-collapsible` and every one of those widgets with it, and nothing cancels the in-flight worker.
- Evidence (real button-click path, fresh-install default of `advanced_visible=False`):
  ```
  cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$PWD $PY - <<'EOF'
  import asyncio
  from textual.app import App, ComposeResult
  from textual.widgets import Button
  import tldw_chatbook.UI.MCP_Modules.mcp_inspector as mod
  from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
  mod.get_cli_setting = lambda s,k=None,d=None: False if k in ("advanced_open","advanced_visible") else d
  mod.save_setting_to_cli_config = lambda *a, **k: True
  class SlowService:
      def __init__(self): self.gate = asyncio.Event()
      async def load_section(self, section=None):
          await self.gate.wait(); return {"source":"local","section":section or "overview"}
      def available_actions(self): return []
  class Harness(App):
      def compose(self) -> ComposeResult: yield MCPInspector(id="insp")
  async def main():
      app = Harness()
      async with app.run_test() as pilot:
          insp = app.query_one(MCPInspector); svc = SlowService()
          insp.set_service_context(svc, [("Overview","overview")], source="local")
          await pilot.pause()
          await pilot.click("#mcp-inspector-advanced-reveal")      # "Advanced…"
          for _ in range(6): await pilot.pause()
          btn = app.query_one("#mcp-inspector-advanced-reveal", Button)
          print("after reveal: label=%r disabled=%r" % (str(btn.label), btn.disabled))
          await pilot.click("#mcp-inspector-advanced-reveal")      # "Hide advanced", load still in flight
          for _ in range(6): await pilot.pause()
          svc.gate.set()
          for _ in range(20): await pilot.pause()
      print("no crash")
  try: asyncio.run(main())
  except BaseException as e: print("TOP-LEVEL RAISE:", type(e).__name__, e)
  EOF
  ```
  ->
  ```
  ...
  /Users/macbook-dev/Documents/GitHub/tldw-review/tldw_chatbook/UI/MCP_Modules/mcp_inspector.py:3418 in _load_advanced_section
    3417     payload = await self._service.load_section(section)
  ❱ 3418     self.query_one("#mcp-adv-content", Static).update(
  NoMatches: No nodes match '#mcp-adv-content' on MCPInspector(id='insp', classes='ds-inspector')
  after reveal: label='Hide advanced' disabled=False
  TOP-LEVEL RAISE: WorkerFailed Worker raised exception: NoMatches(...)
  ```
  Note the printed line: the toggle is re-enabled and relabelled "Hide advanced" BEFORE `_reveal_advanced()` calls `set_service_context()` (`mcp_inspector.py:1629-1631` vs `:1632`), so the second press is available to the user during the entire load.
- Why it matters: the whole Chatbook process exits. The window is not microseconds — `load_section` is the control-plane round trip (server source resolves its access context over multiple sequential client calls before answering), so any user who opts into Advanced and changes their mind mid-load loses the app. The same shape reaches `_refresh_advanced_actions()` on the line after.
- Recommended correction: guard the post-await DOM writes on `self._advanced_visible` (the flag `_hide_advanced` already clears synchronously at `:1668`, before its own `await`) and/or wrap the two reads in `except NoMatches: return`, matching how `_set_test_unavailable`/`show_test_preview`/`show_tool_result` in this same file already handle a panel that went away mid-flight. Dispatching the worker with `exit_on_error=False` (as `MCPWorkbench` does for every worker it owns) is the belt-and-braces half.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none. `Tests/UI/test_mcp_inspector.py` covers the reveal path and the rescheduled-reveal exclusivity, but never a hide racing a slow section load.
- Already covered: none
- **Re-verified by the orchestrator:** the load runs as `run_worker(partial(self._load_advanced_section, ...), group="mcp-adv-section", exclusive=True)` (`:3294`) — `exclusive` only cancels another worker in the *same group*, and `_hide_advanced` does `await collapsible.remove()` without cancelling it, so the await resumes into a removed subtree.
### P1 [D1] — A Search/RAG evidence row whose title carries a non-Rich-shaped bracket renders with that segment deleted  ·  _slice: W-library_
- Where: `library_search_rag_panel.py:1157-1165` (row title), `:1178-1182` (citation labels), `:1271-1276` (`results_count_line`), `:1318-1322` (empty-state copy) — all four `Static(...)` markup-ON; root at `Library/library_rag_state.py:388` (`_sanitize_display_text`'s terminal `escape_markup`), `:1138`, `:2062`.
- Evidence: `<SCRATCH>/probe_rail_title_markup2.py` / `probe_rail_widget_level.py`:
  `'[TODO] Q3 plan' -> render='1.  Q3 plan'` · `'[IMPORTANT]' -> render='1. '` · `'[ WIP ] thing' -> render='1.  thing'` · count line `"1 result for ' plan'."` for the query `[TODO] plan` · empty state `"No evidence matched ' plan'."` (the last two read from the widget's `.visual.plain`).
- Why it matters: same defect class as the row-title finding above, on the surface where the user is trying to identify which evidence matched.
- Recommended correction: same fix — `_escape_all_brackets` (`library_rag_state.py:520`) at these display sinks; the module's own comment already explains why `escape_markup` is not enough here.
- Size: M (shares the fix with the row-title P1) · ADR: no · Confidence: verified
- Pinning test: `Tests/Library/test_library_rag_state.py:591 test_result_row_display_snippet_bracketed_emphasis_stays_inert` pins the already-fixed snippet path; nothing pins titles/citations/count lines.
- Already covered: none
### P1 [D1] — A reviewed flashcard is not reported "due" until the UTC day after its due time; study stats drop the boundary day  ·  _slice: DB-chacha_
- Where: `tldw_chatbook/DB/ChaChaNotes_DB.py:21570` (writes `next_review = datetime.now(utc)+timedelta(days=interval)` via `.isoformat()` → `2026-09-19T02:13:11.350086+00:00`); compared lexically at `:21593` (`get_due_flashcards`) and `:21622` (`count_due_flashcards`) against `CURRENT_TIMESTAMP` (`2026-09-19 02:13:11`, space separator). Same shape mismatch in `get_study_stats` `:23523,:23535,:23549` (`start_date.isoformat()` bound against `reviewed_at`/`updated_at`/`started_at` columns that are `DEFAULT CURRENT_TIMESTAMP`).
- Evidence: `$PY <SCRATCH>/repro_flashcard_next_review.py` → `raw next_review: ('text', '2026-09-19T02:13:11.350086+00:00', '2026-09-18 02:13:11')`. `$PY <SCRATCH>/repro_flashcard_due_day.py` (card set due **one hour ago** in the shape the writer produces, then the same instant in the column's own shape) →
  `T-form ... next_review<=now -> 0 ; count_due=0`
  `space  ... next_review<=now -> 1 ; count_due=1`
  `review 1h inside a 30-day window, stats(days=30)['reviews'] -> {'total_reviews': 0, 'avg_rating': None}` (control with the review well inside the window → `total_reviews: 1`). Mechanism: on the due date `'T'` (0x54) sorts after `' '` (0x20), so `next_review <= CURRENT_TIMESTAMP` is false until the date component advances.
- Why it matters: `Study_Interop/local_study_service.py:88,901` feed the Library rail badge (`count_due_flashcards`) and the next-card picker (`get_due_flashcards(limit=1)`); an "Again" card (interval=1) reviewed at 09:00 does not come back until 00:00 UTC two days later instead of 09:00 next day; the stats window silently excludes up to a day of reviews.
- Recommended correction: write `next_review` in the column's own shape (`next_review.strftime("%Y-%m-%d %H:%M:%S")`, or compute in SQL `strftime('%Y-%m-%d %H:%M:%S','now','+N days')`) and bind `start_date.strftime(...)` in `get_study_stats`; or compare through `julianday()` as `get_conversations_for_character` already does (`:11456-11471`). Existing rows carrying the `T…+00:00` shape need a one-shot `UPDATE ... SET next_review = strftime('%Y-%m-%d %H:%M:%S', next_review)` (SQLite's `strftime` parses the ISO form).
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/ChaChaNotesDB/test_study_functionality.py::test_get_due_flashcards` (:274–291) and `:457` exercise only `next_review IS NULL` / empty-deck cases; no test binds a reviewed card's due instant (grep of assertions, not executed).
- Already covered: none
### P1 [D1] — Console Context rail passes conversation titles and character names through Textual markup with NO escape: `[/…]` in the string raises `MarkupError` inside `compose()`  ·  _slice: W-console-1_
- Where: `console_character_context.py:348` (`CharacterGroupButton(self._group_label(...))`), `:386`, `:437` (`CharacterConversationButton(label, ...)`), plus `header.tooltip`/`button.tooltip` set from the same raw strings at `:356`, `:400`, `:451`
- Evidence:
  ```
  $PY -c "from types import SimpleNamespace as N
  from tldw_chatbook.Widgets.Console.console_character_context import ConsoleCharacterContext as C, CharacterConversationButton as B
  for s in ['[WIP] Draft plan','Q3 [budget] review','notes a [/] b']:
      r=N(title=s,is_current=False)
      try: print(repr(s),'->',repr(str(B(C._row_label(r),row=r).label)))
      except Exception as e: print(repr(s),'-> RAISED',type(e).__name__,e)"
  '[WIP] Draft plan'   -> ' Draft plan'
  'Q3 [budget] review' -> 'Q3  review'
  'notes a [/] b'      -> RAISED MarkupError: auto closing tag ('[/]') has nothing to close
  ```
  Every sibling picker in this same package escapes first (`console_rewind_modal.py:181`, `console_scope_picker_modal.py:567`, `console_prompt_picker_modal.py:273`, `console_reaction_picker_modal.py:470`, `console_endpoint_template_modal.py:503`); this file does not, so it is the only one that can raise.
- Why it matters: `compose()` raising takes the Console screen down, and the input is a DB string — a conversation title the user typed, or a `character_label` from an imported V2/V3 character card (a third-party file). The escaped siblings degrade to silent text loss; this one degrades to a crash.
- Recommended correction: the same literal-content fix as the finding above. If only a stop-gap is wanted, `CharacterConversationButton(Text(label), ...)` / `CharacterGroupButton(Text(label))` is a two-line change that removes both the crash and the deletion at these three sites (a `Text` first arg is not re-parsed — verified).
- Size: S · ADR: no · Confidence: verified
- Pinning test: none (`Tests/UI/test_console_character_context.py` asserts nothing about markup)
- Already covered: none
### P1 [D1] — Google and Cohere streaming handlers drop the provider's usage block; the gateway records usage only from SSE chunks, so streamed Gemini/Cohere turns carry no token counts or cost  ·  _slice: LLM_
- Where: google `tldw_chatbook/LLM_Calls/LLM_API_Calls.py:3798-3921` (the SSE→OpenAI translator reads `candidates` only; `usageMetadata`, present on every Gemini stream chunk, is never emitted); cohere `:2832-2846` (`message-end` carries `delta.finish_reason` AND `usage`; only `finish_reason` is forwarded). Contrast anthropic `:1846-2006` (accumulates `usage` and emits a trailing usage chunk) and openai `:714-715` (`stream_options.include_usage`).
- Evidence: `PYTHONPATH=$WT $PY -` (see the `run("google"…)`/`run("cohere"…)` snippet in this report's shell history: mocked stream with `usageMetadata` / `message-end.usage`, `chat_api_call(streaming=True)`) → `[google stream] yielded=1 chunks; any 'usage' key forwarded: False` / `[cohere stream] yielded=2 chunks; any 'usage' key forwarded: False`. Consumer: `Chat/console_provider_gateway.py:7067-7076 _maybe_record_usage` reads `payload.get("usage")` per chunk; `rg -n "estimate_usage|usage is None" Chat/console_provider_gateway.py` → no fallback estimate.
- Why it matters: the cost ticker / usage ledger (ADR 156 live per-run usage attribution) gets nothing for streamed Gemini and Cohere turns even though the provider sent the numbers; non-streaming turns on the same providers DO report usage (`:4018-4027`, `:2975-2996`), so the ledger is inconsistent per streaming toggle.
- Recommended correction: google — when a chunk carries `usageMetadata`, emit the same `{"choices": [], "usage": {prompt_tokens, completion_tokens, total_tokens}}` trailing chunk the anthropic branch emits (cumulative, so emit once at finish); cohere — on `message-end`, map `usage.billed_units|tokens` exactly as the non-streaming branch (`:2976-2986`) and emit it. S each.
- Size: S · ADR: no · Confidence: verified (handler drop) — consumer effect read from `_maybe_record_usage`, not run through the gateway
- Pinning test: none for streamed usage on either provider (`rg -ln usageMetadata Tests/Chat/` hits `test_google_native_tools.py` and the gateway tests, neither asserts stream usage).
- Already covered: none. Related open question (Left UNVERIFIED): deepseek/groq/mistral/openrouter streams never request `stream_options.include_usage` (`rg -n include_usage LLM_Calls/` → openai, moonshot, qwencloud only), so whether THEIR streamed usage arrives depends on each provider's default.
### P1 [D1] — MCP permission store treats a transient read `OSError` as corruption: it renames the live `mcp_permissions.json` to `.bak` and resolves from fresh defaults — kill switch ON→OFF, a built-in set to Off resolves Allow  ·  _slice: TOOLS-MCP_
- Where: `tldw_chatbook/MCP/permission_store.py:747-757` (`except (OSError, ValueError, json.JSONDecodeError)` → `_backup_corrupt_file()` + `return _fresh_payload()`); same shape at `:892` (`_load_for_raw_getter`, no backup but same fresh-default fallback)
- Evidence: `cd $WT && source env.sh && PYTHONPATH=$WT $PY - <<EOF` (set kill_switch=True and `agent:builtin/calculator=deny`, then `p.chmod(0o000)` and `MCPPermissionStore(p).load()`) → `before: kill_switch= True calculator= deny` / `after chmod000 load(): kill_switch= False calculator= allow | store exists: False | .bak exists: True`; the WARNING logged says "unreadable/corrupt ([Errno 13] Permission denied ...); backing it up and resetting to defaults."
- Why it matters: EACCES/EIO/EMFILE/EINTR on read are not corruption, yet one such read moves the user's whole policy file aside; the verdict for that call is already the permissive default, and the next mutator (`_mutate_locked` → `load()` → `_save_locked`) persists the fresh payload, so every Off/Allow decision and the kill switch are lost for good. The module docstring's own contract ("uncertain native persistence failures propagate without resetting policy") is not honoured on the plain-file path. Trigger needs an I/O fault (wrong-owner file after a `sudo` run, fd exhaustion, a Windows sharing violation while the standalone `TldwMCPServer` — which opens the SAME path, `MCP/server.py:989` — writes it), so P1 not P0.
- Recommended correction: split `OSError` out of that except: re-raise (or return a deny-all payload) and only back up on `ValueError`/`JSONDecodeError`/shape mismatch, which is what spec §9 actually says ("unknown schema version -> back up"). S.
- Size: S · ADR: no · Confidence: verified
- Pinning test: SEE "Left UNVERIFIED" (whether `Tests/MCP/test_permission_store*.py` pins the OSError branch as a requirement)
- Already covered: none
### P1 [D1] — Opening the File Notes "Review pairing" dialog silently stops the folder poll for the rest of the mount  ·  _slice: W-library_
- Where: `library_file_notes_workspace.py:6707-6709` (`_review_pairing` pauses `self._poll_timer` and no path resumes it); the only `resume()` in the 8,846-line file is `:8845`, inside `_refresh_pressed` (the manual Refresh button). `grep -n "_poll_timer" library_file_notes_workspace.py` → one `.pause()` (6709), one `.resume()` (8846), the rest are create/stop/None.
- Evidence: probe `<SCRATCH>/probe_poll_pause.py` (reuses the shipped `Tests/Backup_Recovery` subprocess harness, real replica + real workspace):
  `cd <wt> && source <SCRATCH>/env.sh && PYTHONPATH=<wt> $PY -m pytest <SCRATCH>/probe_poll_pause.py -q -s`
  → `POLL ACTIVE before review: True` / `POLL ACTIVE after dialog closed: False` / `POLL ACTIVE after pressing Refresh: True`.
- Why it matters: the workspace's background file monitoring (external edits, deletions, replica reconciliation) is dead from the moment the user opens the pairing review until they press Refresh or leave and re-enter the screen. Nothing tells them.
- Recommended correction: resume in a `finally:` in `_review_pairing` (guarded on `self._poll_timer is not None and self._active`), i.e. pair the pause with the scope that needed it.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/Backup_Recovery/test_notes_recovery_controls.py` exercises exactly this flow and its `approve` branch presses Refresh at the end — which masks the paused timer rather than asserting it.
- Already covered: none
### P1 [D1] — Renaming a speaker in Library ▸ Media replaces the 3-line metadata preview pane with the entire transcript  ·  _slice: W-library_
- Where: `tldw_chatbook/Widgets/Library/library_media_canvas.py:688-696` (`_refresh_after_speaker_rename`); the pane it writes into is composed at `:1900-1904` from `canvas.preview_lines`, built at `tldw_chatbook/Library/library_media_state.py:1197-1205` and `:1573-1583` as exactly three metadata lines (title / `Type: …` / `Updated: …`).
- Evidence: probe `<SCRATCH>/probe_media_preview.py` (production-shaped `preview_lines`, real `MediaDatabase`, real canvas):
  `cd <wt> && source <SCRATCH>/env.sh && PYTHONPATH=<wt> $PY -m pytest <SCRATCH>/probe_media_preview.py -q -s`
  → BEFORE `'Meeting\nType: audio\nUpdated: today'`; AFTER `'[00:00:00] Alice: hello hello …\n[00:00:01] Speaker 2: world …'` — `AssertionError: PREVIEW PANE CLOBBERED BY FULL TRANSCRIPT`.
- Why it matters: after one inline rename the small "selected item" pane in the media list becomes the whole transcript (unbounded text in a fixed pane), and it never goes back until the next state push. The sibling reader (`library_media_viewer.py:870-899`) does the same job correctly, so this is the drifted copy.
- Recommended correction: `_refresh_after_speaker_rename` must not write `Media.content` into `#library-media-preview-lines`. The pane is metadata-only; the rename only needs the legend label patch (the second half of the method). If the canvas wants a content echo it has to ask the controller for a fresh `LibraryMediaCanvasState`.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/UI/test_library_media_speaker_rename.py:121 test_speaker_legend_submit_renames_and_refreshes_preview` — it MASKS the bug: line 143 seeds `preview_lines=tuple(row["content"].splitlines())`, a shape no production builder ever produces, then asserts `"Alice" in preview`. It does not state the production behaviour as a requirement.
- Already covered: none
### P1 [D1] — Seven of eight streaming handlers in `LLM_API_Calls.py` still `yield "data: [DONE]"` inside `finally`; a consumer Stop (`gen.close()`) raises `RuntimeError: generator ignored GeneratorExit` and the HTTP response is never closed. Only the OpenAI handler was fixed (and pinned).  ·  _slice: LLM_
- Where (yield-in-finally, ALL copies): anthropic `tldw_chatbook/LLM_Calls/LLM_API_Calls.py:2019-2022`, cohere `:2888-2901` (guarded by `not stream_properly_closed`, which is exactly the Stop case), deepseek `:3310-3313`, google `:3918-3921`, groq `:4370-4373`, mistral `:5135-5138`, openrouter `:5389-5392`. Fixed shape: openai `:916-928` (sentinel yielded AFTER the `finally`, with the comment explaining why). Clean: huggingface `:4800-4803`. Variant with no leak but the same RuntimeError: `LLM_API_Calls_Local.py:364-369` (`response.close()` runs BEFORE the sentinel yield; every local provider — llama.cpp/vllm/ollama/mlx/ooba/tabby/aphrodite/custom-openai 1+2 — routes through it).
- Evidence: `PYTHONPATH=$WT $PY $SCRATCH/llm_repro_generatorexit.py` (patch `requests.Session.post` with a 2-line mock stream, `chat_api_call(provider, streaming=True)`, `next()`, `.close()`) →
  ```
  openai      close() OK                                    response.close() called=True
  anthropic   RuntimeError: generator ignored GeneratorExit response.close() called=False
  deepseek    RuntimeError: generator ignored GeneratorExit response.close() called=False
  groq        RuntimeError: generator ignored GeneratorExit response.close() called=False
  mistral     RuntimeError: generator ignored GeneratorExit response.close() called=False
  openrouter  RuntimeError: generator ignored GeneratorExit response.close() called=False
  google      RuntimeError: generator ignored GeneratorExit response.close() called=False
  cohere      RuntimeError: generator ignored GeneratorExit response.close() called=False
  ```
  Consumer path: every stream is wrapped in `recovery_review._OpenAIStream` (`:485-539`) whose `close()` forwards to the inner generator; `Chat/console_provider_gateway.py:5904-5906 call_response_close` wraps that in `contextlib.suppress(Exception)`, so the RuntimeError is swallowed silently and the `response.close()` statement after the yield never runs.
- Why it matters: Console Stop on any non-OpenAI hosted stream leaves the streaming socket open (server keeps generating into a dead buffer up to the 180 s read timeout) until the generator frame is garbage-collected; the local-server variant turns every Stop into a swallowed RuntimeError.
- Recommended correction: same shape as the OpenAI fix — move the sentinel out of `finally` (after the try/except/finally, or a `stopped` flag set in an `except GeneratorExit`); Cohere keeps its `stream_properly_closed` sentinel but emits after `finally`. One mechanical PR across 8 sites plus the OpenAI pinning test's shape per provider.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/Chat/test_openai_streaming_usage.py::test_stopping_stream_closes_transport_without_yielding_after_generator_exit` — states the requirement for OpenAI ONLY; `rg -n "GeneratorExit" Tests/Chat/` finds no sibling test.
- Already covered: none (`rg -il "GeneratorExit" backlog/tasks/` → only task-18300, an unrelated Console inspector review).
### P1 [D1] — The Library Search/RAG semantic query runs ChromaDB's synchronous `query()` directly on the Textual event loop; the first search of a session freezes the UI for ~212 ms  ·  _slice: RAG_
- Where: `tldw_chatbook/RAG_Search/simplified/rag_service.py:1592` (`search_with_citations`) and `:1600` (`search`), both inside `async def _semantic_search` (L1552). Store side: `tldw_chatbook/RAG_Search/simplified/vector_store.py:470` `ChromaVectorStore.search` (sync) and `:575` `search_with_citations` (sync), whose `client` property (`vector_store.py:280`) lazily imports `chromadb` and opens `PersistentClient` on first use.
- Caller chain (traced, not inferred): `UI/Screens/library_screen.py:34871` `@work(exclusive=True, group="library_rag_search") async def _execute_library_rag_search` (a **coroutine** worker → Textual event loop, not a thread) → `Library/library_rag_service.py:94 run_library_rag_search` → `Library/library_local_rag_search_service.py:885` `await rag_service.search(..., search_type="semantic")` → `rag_service.search` → `_semantic_search` → the sync store call above.
- Evidence:
  - `grep -n "async def _execute_library_rag_search" -B1 tldw_chatbook/UI/Screens/library_screen.py` → `34871: @work(exclusive=True, group="library_rag_search")` / `34872: async def _execute_library_rag_search(`
  - Measured, isolated env, 20 000 chunks × dim 384 persisted Chroma collection, fresh interpreter, a 5 ms asyncio ticker running alongside:
    ```
    COLD first search_with_citations on the loop: 212 ms (20 results)
    WARM subsequent search_with_citations:        3.5 ms
    max event-loop tick lag observed:             215 ms (5 ms target tick)
    ```
    (script: build store with `ChromaVectorStore(...).add()` ×20 batches, then `asyncio.run` a ticker + two `vs.search_with_citations(q,"query",20)` calls; full command in "Left UNVERIFIED"/notes below)
- Why it matters: the whole UI (spinner, keystrokes, the search's own cancel path) is stalled for the full store call; the cold cost scales with collection size because it includes the HNSW index load, and every scoped search multiplies it — `library_local_rag_search_service.py:893-901` issues **one store query per source type** in a Python `for` loop, all of them on the loop.
- Asymmetry that shows this is an oversight rather than a decision: the sibling calls on the very same path are deliberately offloaded — `library_local_rag_search_service.py:1092 service = await asyncio.to_thread(get_shared_rag_service)` ("First-time construction … runs in `asyncio.to_thread` -- never on the UI event loop") and `:1122 stats = await asyncio.to_thread(get_stats)` ("ChromaDB-backed stats can touch disk; keep it off the event loop"). The embedding half of `_semantic_search` is offloaded too (`embeddings_wrapper.py:747 await asyncio.to_thread(native_worker(...))`). Only the query is not.
- Recommended correction: wrap the two store calls in `_semantic_search` in `await asyncio.to_thread(...)` (one `functools.partial` each, ~6 lines). That is the single choke point — every semantic caller routes through `_semantic_search`.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none found (`rg -n "to_thread" Tests/RAG*` → no test asserts the sync shape)
- Already covered: none
### P1 [D1] — The ingest chunking-template picker shows "None (manual settings)" while the form still submits a saved template  ·  _slice: W-library_
- Where: `library_ingest_canvas.py:1553-1560` (compose clamp) and `:2027-2036` (post-fetch restore).
- Evidence: `<SCRATCH>/probe_ingest_template_value.py` (canvas mounted with `form.type_options["generic"]["chunk_template"] = "big-words"` and a scope service returning it):
  `OPTIONS: ['', 'auto', 'big-words']` · `PICKER VALUE (what the user SEES): ''` · `FORM VALUE (what Start SUBMITS): 'big-words'` · `OptionValueChanged posted: []`.
  Mechanism: a fresh canvas has `_chunk_template_names = []`, so `available` at `:1556` lacks the saved name and `picker_value` is clamped to `""` — display only, the form is never corrected. `_fetch_chunk_templates` then reads `selected = picker.value` (`:2027`), now `""`, finds it in the new options and preserves it; the `else` branch at `:2033` pre-seeds `_reported_option_values` before `set_options`, so the resulting `Select.Changed` is swallowed as mount noise by `_handle_option_value_changed:2135`. Reachable on the shipped route: the canvas is rebuilt on every resolution to `ingest-media` (`library_screen.py:11898`, `:15585`) while `self._ingest_state.form` lives on the screen (`:3963`); `app.py:4275` reads `flat_opts["chunk_template"]`.
- Why it matters: the screen states one chunking policy and executes another, on the control whose entire job is to state it.
- Recommended correction: in `_fetch_chunk_templates`, prefer the form's stored value when it has become available — two lines in the existing preserve branch.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/UI/test_library_ingest_template_picker.py` pins the default (`test_picker_default_is_none_manual_settings`) and the populate path, never a persisted value.
- Already covered: none
### P1 [D1] — The irreversible "Delete stored Full captures" confirmation names the WRONG conversation when its title contains a bracketed token, and raises `MarkupError` for `[/…]`  ·  _slice: W-console-1_
- Where: `console_capture_policy_dialog.py:525-542` (`_purge_confirmation_message`, interpolates `snapshot.conversation_title`) → `:592-597` (`delete_full_captures` calls `_confirm` with it) → `:627-642` `_confirm` → `Widgets/confirmation_dialog.py:117` `yield Label(self.message, classes="dialog-message")` — **no `markup=False`**
- Evidence:
  ```
  $PY -c "import tldw_chatbook.Widgets
  from textual.widgets import Label
  from tldw_chatbook.Widgets.Console.console_capture_policy_dialog import ConsoleCapturePolicyDialog as D
  from types import SimpleNamespace as N
  for title in ('[WIP] Q3 planning','notes a [/] b'):
      snap=N(enabled=False, conversation_title=title, effective=N(detail=N(value='safe')))
      m=D._purge_confirmation_message(snap,12)
      try: print(repr(title),'->',repr(str(Label(m).render())[:90]))
      except Exception as e: print(repr(title),'-> RAISED',type(e).__name__,e)"
  '[WIP] Q3 planning' -> 'Delete 12 stored Full captures from “ Q3 planning”? This irreversible action …'
  'notes a [/] b'     -> RAISED MarkupError: auto closing tag ('[/]') has nothing to close
  ```
- Why it matters: the whole job of this string is to let the user confirm *which* conversation's stored Full captures are about to be irreversibly deleted, and the title it shows is not the title that exists. The `[/…]` case raises inside `ConfirmationDialog.compose()`, so the confirmation for a destructive action fails to appear at all. `ConfirmationDialog` is imported by 45 modules (`grep -rln "confirmation_dialog import" tldw_chatbook | wc -l` → 45), several of which interpolate user-named entities into `message`, so the same hole is open wherever a caller does.
- Recommended correction: `markup=False` on `Widgets/confirmation_dialog.py:117`'s `Label` (and `:116`'s title `Static`). **The repo already does exactly this next door**: `Widgets/Library/prompt_delete_confirmation_modal.py:144` is `yield Static(self._preview_copy(), id="prompt-delete-preview", markup=False)` — same class of dialog, same user-named entity, correct fix. It is the canonical home: one line fixes every one of the 45 callers and cannot be forgotten at a call site. `ConfirmationDialog` messages are all plain prose today, so nothing loses styling.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `ConfirmationDialog` itself has none (`find Tests -iname "*confirmation*"` → only `Tests/UI/test_prompt_delete_confirmation_modal.py`, which covers the *other* modal). That test is worth reading before fixing this: `:306 test_markup_looking_names_render_literally` asserts literal rendering using `name = "[bold magenta]not markup[/bold magenta]"` — an all-lowercase tag, the one case `rich.markup.escape` handles. The uppercase case the Console hits (`"[WIP] not markup"`) is untested there too, so the new test should use an UPPERCASE tag or it proves nothing about this defect class.
- Already covered: none
### P1 [D1] — The left rail's Model section (Temperature / Max tokens) never updates after compose: the screen's updater queries two widgets TASK-23196 deleted, raises `NoMatches` on the first, and the real rows' writes are unreachable  ·  _slice: UI-chat_
- Where: `tldw_chatbook/UI/Screens/chat_screen.py:9404-9431` (`_apply_console_settings_summary_state`); compose-time render `UI/Console_Modules/left_rail.py:2281-2320`; the rail's state is set only in `left_rail.py:445` (`__init__`) with no sync method (`grep -n "_settings_summary_state" left_rail.py` → 445, 2275 only).
- Evidence: `grep -rn "console-model-section-provider\|console-model-section-model\b" tldw_chatbook/` → only chat_screen.py:9416/9420 (no widget composes them); `Tests/UI/test_console_model_section.py:31-32` and `test_console_model_section_dedup.py:40-43` assert they are ABSENT. Probe `<SCRATCH>/probes/test_probe_model_section_stale.py` (ConsoleHarness 160×44; patched `_build_console_settings_summary_state` → `temperature="9.87"`, called `_sync_console_settings_summary()`): `PROBE before='0.60' after='0.60' provider_rows=[]` → `AssertionError: row did not update`. The `try` at 9414 dies on its first `query_one` (`#console-model-section-provider …`), so the temperature/max_tokens `.update()` calls at 9422–9429 never execute; the regex parse at 9406–9411 (`re.search(r"T ([\d.]+)"…)`) is dead too.
- Why it matters: after the user changes temperature or max_tokens (Conversation settings, Alt+M popover, `/model`), the Context rail's Model section keeps the compose-time values until the rail happens to recompose — the rail shows one temperature while sends use another.
- Recommended correction (S): delete 9404–9431's provider/model queries and regex parse; write `summary_state.temperature or "—"` / `summary_state.max_tokens or "—"` (the structured fields TASK-32338 added to `ConsoleSettingsSummaryState`, already used by `left_rail.py:2281`) into `#console-model-section-temperature/-max-tokens .console-model-section-value` under their own guarded `try`; update the two pinned rows in `Tests/Architecture/test_timer_path_static_update_inventory.py:223-229` (they inventory the dead `query_one` expressions). Better shape: a `ConsoleLeftRail.sync_settings_summary(state)` next to `sync_sections`, called from `_apply_console_settings_summary_state`, so the rail owns its rows.
- Size: S · ADR: no · Confidence: **verified** (probe)
- Pinning test: `Tests/UI/test_console_model_section.py::test_model_sync_updates_rows` — its NAME asserts the refresh but its body asserts only that the row is non-empty (the compose value satisfies it), so it does not pin the behaviour. That file is red under the isolated review env for an unrelated bootstrap reason — see UNVERIFIED.
- Already covered: none (TASK-23196 removed the rows; TASK-32338 added the structured fields to the rail's compose path only).
### P1 [D1] — The note editor's "where does this note live" row is elided to the LIST pane's width, not its own  ·  _slice: W-library_
- Where: `library_notes_canvas.py:1364 _effective_pane_width` (consumed at `:2760` compose and `:3650 _restate_note_location`); `:1387-1388` — `on_resize` returns before recording `_measured_width` for any mode but `"list"`. The work pane's `pane_width` is `reader_layout.items_width` (`library_notes_controller.py:3337-3338`), and `apply_pane_width` is only dispatched to `#library-notes-canvas` (`library_screen.py:6692`), never to `#library-note-work-pane`.
- Evidence: `<SCRATCH>/probe_notescanvas_live_width.py`, driving the repo's own harness from `Tests/UI/test_library_notes_w5_ideas.py` at its `WIDE = (235, 52)`:
  `work pane rendered width : 157` · `row rendered width : 156` · `wp.pane_width (contract) : 64` (the list pane) · `_effective_pane_width() : 64`.
  `RENDERED row: 'In a synced folder · …low-ups.md · file written 2026-09-15 01:06'` vs at the row's real width: `'In a synced folder · /var/folders/.../Obsidian/Vault/Project…atlas-follow-ups.md · file written …'`.
- Why it matters: the path is crushed past its own filename in a row 156 cells wide, and at a narrower split the `_NOTE_LOCATION_PATH_FLOOR` branch (`:265-267`) drops "file written …" entirely — a fact removed from a pane with room for it. Answering "where does this note live" is the row's whole purpose.
- Recommended correction: pass the row's own width (`self.size.width`, already 157) to `_restate_note_location`/`_compose_editor`, or let `on_resize` record `_measured_width` in every mode and stop preferring `pane_width` outside list mode.
- Size: S · ADR: no · Confidence: verified (measured against the shipped harness)
- Pinning test: `Tests/UI/test_library_notes_w5_ideas.py:300` asserts only `"Sam.md" in line` on a short tmp path; `:344` pins the pure function at hand-passed widths. Neither pins which width the editor supplies.
- Already covered: none
### P1 [D1] — User text that contains an uppercase bracketed token (`[WIP] plan`, `[Draft] notes`, `[TODO]`) is silently DELETED from Console labels, and `rich.markup.escape` does not prevent it  ·  _slice: W-console-1_
- Where (cluster, all in `tldw_chatbook/Widgets/Console/`):
  - `console_character_context.py:348` (`_group_label` → character name), `:386` and `:437` (`_row_label`/`_search_row_label` → conversation title) — **no escape at all**
  - `console_prompt_queue_modal.py:306` — `Button(f"{entry.position}. {entry.preview}")`, queued prompt text
  - `console_rewind_modal.py:181` — `escape_markup(f"{row.index_label}  {row.preview}")`, the user's own prompt text
  - `console_scope_picker_modal.py:567` (media/note title), `:589`, `:605` (tag names)
  - `console_reaction_picker_modal.py:470` (`option.display_label`)
  - `console_prompt_picker_modal.py:273` (prompt name), `console_endpoint_template_modal.py:503` (`candidate.label`), `console_provider_picker.py:242`
  - `console_inspector_section.py:975-988 _refresh_tooltip` — `self.tooltip = escape("\n".join(parts))` at `:988`, whose own docstring says *"Markup is escaped because a Textual tooltip is Rich-parsed content and row text is user-adjacent"*; the tooltip is the ONLY way to read a row's untruncated text, and a changed-file path `src/[WIP]/notes.md` renders in it as `src//notes.md` (verified: `Tooltip().update(escape(s)); str(t.render())`)
  - `console_composer_bar.py:5889` — attachment filename (see the D4a finding below)
  - `console_canvas_card.py:177` — `Static(self.presentation.label)`, where `label` is `f"{card.title} · revision {n} · {status}"` built at `console_transcript.py:237` from a user-authored Canvas document title; **no escape at all**, so this one can raise like `console_character_context.py`
  - upstream helper: `Chat/console_prompt_queue.py:155 make_prompt_preview`, whose docstring claims "Rich-markup-safe"
  - shared sink: `Widgets/confirmation_dialog.py:117` (see the purge-confirmation finding below)
- Evidence (worktree, isolated env):
  ```
  $PY -c 'import rich.markup as rm; from textual.content import Content
  for t in ["[WIP] plan","[TODO]x","[b]b[/b]","[/]"]:
      e=rm.escape(t); print(repr(t), rm.escape(t), repr(str(Content.from_markup(e))))'
  '[WIP] plan'  -> rich.escape '[WIP] plan'  -> Content ' plan'
  '[TODO]x'     -> rich.escape '[TODO]x'     -> Content 'x'
  '[b]b[/b]'    -> rich.escape '\\[b]b\\[/b]' -> Content '[b]b[/b]'   (lowercase IS escaped)
  '[WIP]a[/WIP]' -> Content 'a[/WIP]'   (the open tag vanishes, the close tag prints literally)
  ```
  `rich.markup.escape`'s regex is `(\\*)(\[[a-z#/@][^[]*?])` — it only escapes tags whose first character is `[a-z#/@]`. Textual 8's `Content.from_markup` accepts **any** tag body, uppercase included, and drops it as an unresolvable style span. `textual.markup.escape` is the identical regex, so switching helper does NOT fix it.
  Production constructors reproduced end to end:
  ```
  $PY -c "from types import SimpleNamespace as N
  from tldw_chatbook.Widgets.Console.console_character_context import ConsoleCharacterContext as C, CharacterConversationButton as B
  r=N(title='[WIP] Draft plan', is_current=False); print(repr(str(B(C._row_label(r), row=r).label)))"
  ' Draft plan'                                  # '[WIP] ' gone
  r.title='notes a [/] b'  -> MarkupError: auto closing tag ('[/]') has nothing to close
  ```
  ```
  $PY -c "from tldw_chatbook.Chat.console_prompt_queue import make_prompt_preview as m
  from textual.widgets import Button; print(repr(str(Button('1. '+m('[WIP] summarize this')).label)))"
  '1.  summarize this'
  ```
- Why it matters: a conversation titled `[WIP] Draft plan` shows in the Console Context rail as ` Draft plan`; a queued prompt `[TODO] rerun` shows as ` rerun`; the rewind list loses the same text from the prompt the user is choosing between. Bracketed prefixes are a common titling convention, so this is routine, not adversarial. The `console_character_context.py` sites are unescaped entirely, so a title/character name containing `[/…]` raises `MarkupError` **inside `compose()`** (verified above) — that is an app-level crash from a DB string, and character names can arrive from imported third-party cards.
- Recommended correction: stop routing user text through markup parsing at all — pass `Content(label)` / `Text(label)` (a `Text`/`Content` first arg is not re-parsed; verified: `str(Button(Text("plain [b]x[/b]")).label) == 'plain [b]x[/b]'`), or `markup=False` where the widget supports it. Canonical home: one helper next to the existing escape users, e.g. `Utils/text.py` `as_literal_content(str) -> Content`, then replace every `escape_markup(<user text>)` call in `Widgets/Console/` with it. Deleting the `escape_markup` calls without replacing the mechanism would make it worse (lowercase tags would then also be eaten).
- Size: M · ADR: no · Confidence: verified
- Pinning test: none. `Tests/UI/test_console_character_context.py` has no markup/bracket assertion (`grep -n "markup\|escape\|bracket" Tests/UI/test_console_character_context.py` → only `pilot.press("escape")` lines).
- Already covered: none
- **Re-verified by the orchestrator:** `rich.markup.RE_TAGS` is `((\\*)\[([a-z#/@][^[]*?)])`. Measured through `Content.from_markup`: `'[TODO] Q3 plan'` → `' Q3 plan'`, `'[WIP] Draft plan'` → `' Draft plan'`, `'[IMPORTANT]'` → `''` (blank), while `'[draft] x'` survives — which is why every existing pinning test, all of which use lowercase tags, passes.
### P1 [D1] — `CalculatorTool` (always-on built-in) evaluates model-supplied `**` and `str * int` with no bound: one call allocates a 200 MB string in 20 ms and bignum powers scale superlinearly (7**4e6 = 1.26 s, 9x the cost of 7**1e6)  ·  _slice: TOOLS-MCP_
- Where: `tldw_chatbook/Tools/tool_executor.py:181-220` (`allowed_operators` admits `ast.Pow`/`ast.Mult`; `ast.Constant` admits `str`, so `'ab' * 10**8` and `7**(4*10**6)` both evaluate)
- Evidence: `cd $WT && source env.sh && PYTHONPATH=$WT $PY - <<EOF ... asyncio.run(CalculatorTool().execute(expression="'ab' * 10**8")) ... EOF` → `str*int 200MB: 0.02s len=200000000 maxrss MB=227` (baseline 36 MB); `7**10**6: 0.14s bits=2807355`; `7**4e6: 1.26s bits=11229420`
- Why it matters: `tool_executor.py` docstring calls this one of "the two always-on built-in tools"; `risk_tags` is `()` so it inherits `allow` (`BUILTIN_DEFAULT_STATE`), and `Tool.timeout_seconds` documents that a timed-out worker THREAD is abandoned, not killed — so a prompt-injected `9**9**9` or `'a'*10**10` pins a core / exhausts RAM for the rest of the process lifetime with no user approval and no kill path.
- Recommended correction: in `safe_eval`, reject non-numeric `ast.Constant`; bound `Pow` (`abs(right) <= 1_000` and `left.bit_length()*right <= ~1<<20`) and refuse `Mult` on `str`; S.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none (`rg -n 'Pow|\*\*' Tests/Tools/test_tool_executor.py` → no hits)
- Already covered: none
### P1 [D1] — `_open_video_with_os` is a class-body function with neither `self` nor `@staticmethod`: every production call raises `TypeError`, so "open video with the OS player" silently never works  ·  _slice: UI-chat_
- Where: `chat_screen.py:19648-19658`; wired at `UI/Console_Modules/wiring.py:808` (`open_video_with_os=lambda path: screen._open_video_with_os(path)`); called at `UI/Console_Modules/video.py:978` (external open after save) and `video.py:1308` (managed-open fallback when playback tools are missing) — both inside `except Exception` that logs `"Console video operation={} failed error_type=TypeError"` and returns.
- Evidence: `$PY -c` (isolated env): `ChatScreen.__dict__['_open_video_with_os']` is a plain `function` (not staticmethod); `ChatScreen.__new__(ChatScreen)._open_video_with_os(Path('/x.mp4'))` → `TypeError: ChatScreen._open_video_with_os() takes 1 positional argument but 2 were given`. `grep -rn "_open_video_with_os" Tests/` → every test injects a stub (`open_video_with_os=lambda _path: None`, `harness._open_video_with_os = fail_open`); none exercises the real method.
- Why it matters: the user asks to open a generated video externally, nothing opens, one warning line lands in the log; the feature shipped dead and every test is green because the seam is always mocked (the `lessons-testing-evidence.md` "fakes mask real signatures" class).
- Recommended correction (S): add `@staticmethod` (the body never uses `self`); give `Tests/Chat/test_console_video_controller.py` one case that wires the REAL `ChatScreen._open_video_with_os` with `subprocess.Popen` patched so the signature is exercised.
- Size: S · ADR: no · Confidence: **verified**
- Pinning test: none (all stubbed).
- Already covered: none.
### P1 [D1] — `config.get_api_key()` returns placeholder and un-stripped credentials that the shared validity rule rejects, on 5 live spend/readiness paths  ·  _slice: ENTRY-config_
- Where: `tldw_chatbook/config.py:9083-9087` (only `!= "<API_KEY_HERE>"` is screened; value returned raw), `9078-9080` (env value returned raw). Callers: `MCP/tools.py:108`, `MCP/server.py:610` (tool spend), `UI/Console_Modules/realtime.py:616` (realtime credential), `UI/Screens/llm_screen.py:2837`, `UI/LLM_Management/vllm_connection.py:536`.
- Evidence: config `[api_settings.openai] api_key = "YOUR_KEY"` under the isolated env →
  ```
  bridge openai_api.api_key       : None
  get_api_key('openai')           : 'YOUR_KEY'
  ```
  (`$PY - <<EOF ... config.load_settings()["openai_api"]["api_key"]; config.get_api_key("openai") EOF`). `rg -n "resolve_provider_api_key|is_valid_provider_api_key" tldw_chatbook -l` → 15 importers use the shared rule; `get_api_key` is not one of them.
- Why it matters: exactly the PR-T2 "two readers disagree" split re-created one accessor over — readiness says not-ready, an ungated MCP tool or the realtime engine sends `YOUR_KEY` (or `" sk-xyz "`) as the bearer token and gets a 401.
- Recommended correction: replace both raw returns with `resolve_provider_api_key(...)` (already defined at `config.py:1400`); one-line change per branch, canonical home is the function that already exists.
- Size: S · ADR: yes (`012-provider-credential-settings-boundary.md` already frames the boundary; no new decision needed for the placeholder half) · Confidence: verified
- Pinning test: `Tests/Utils/test_config_api_key_resolution.py::test_placeholder_api_key_is_never_returned` — covers only `<API_KEY_HERE>`; its docstring states the requirement ("returning it would send a literal placeholder to a provider as a bearer token"), so the fix is consistent with the pin, not against it.
- Already covered: none (CLAUDE.md's "google env-only gap" is a different defect).
### P1 [D1] — `doctor` reports every optional feature group as "not installed" under the shipped (lazy) dependency mode, including groups that are installed  ·  _slice: UTILS_
- Where: `tldw_chatbook/Utils/doctor.py:59-75` (`check_optional_dependencies` reads `DEPENDENCIES_AVAILABLE` raw when no `available` is injected) + `tldw_chatbook/Utils/optional_deps.py:1480-1541` (registry starts all-False; populated only by `initialize_dependency_checks`, which nothing on the doctor path calls).
- Evidence: `$PY -c "from tldw_chatbook.Utils.doctor import check_optional_dependencies as c; from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE as D; r=c(); print(r.status, r.detail[:160]); import importlib.util as u; print(u.find_spec('torch') is not None, D['torch'])"` → `warn 70 optional feature group(s) not installed: PIL, aiohttp, audio_processing, av, beautifulsoup4, chatterbox, chinese_chunking, chromadb, chunker, cohere, defused…` then `True False` (torch is installed; flag is False). Same for numpy.
- Why it matters: the aggregate health surface (TASK-25906) tells a user with a full install that 70 groups are missing and to `pip install` them — a wrong answer a user acts on.
- Recommended correction: in `check_optional_dependencies`, call `optional_deps.ensure_dependencies_checked()` (or the `find_spec`-only probes) before reading the registry; or pass a `find_spec`-derived mapping. The root defect is the registry contract itself, already tracked.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/Utils/test_doctor.py` exists; doctor.py's own docstring says checks are "pure; inject the input for testability", so the injected path is what it pins (whether it also covers the no-argument path: not read — see UNVERIFIED).
- Already covered: task-25704 / task-287 (DEPENDENCIES_AVAILABLE flags never populated). This finding is the doctor-visible consequence of that task; cite, do not re-file.
### P1 [D1] — `get_detected_api_providers()` always returns `[]`; the `doctor` "providers" check always warns "no API providers are configured"  ·  _slice: ENTRY-config_
- Where: `tldw_chatbook/config.py:8795-8801` (iterates `config.items()` for keys starting with `"api_settings."` — a flat dotted key that a nested TOML load never produces; the same dead-branch shape `get_api_key`'s own comment at 9054-9063 says was fixed there). Consumer: `Utils/doctor.py:99-110`.
- Evidence: same probe as above, config had a real `[api_settings.anthropic] api_key` → `get_detected_api_providers()    : []`. `rg -n "get_detected_api_providers" tldw_chatbook Tests` → callers are `Utils/doctor.py:101` and `UI/Tools_Settings_Window.py` (DEPRECATED, nav-unreachable per CLAUDE.md); zero tests.
- Why it matters: every user running `doctor` is told to "add a key under [api_settings.<provider>]" regardless of what they have configured; the helper also uses a third placeholder rule (`startswith("<") and endswith(">")`) distinct from `PROVIDER_API_KEY_PLACEHOLDERS`.
- Recommended correction: iterate `config.get("api_settings", {})` and accept a provider when `resolve_provider_api_key(table.get("api_key"))` is truthy; add one test with a nested config.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none
### P1 [D1] — `load_imported_trace()` lets `UnicodeDecodeError` and `RecursionError` escape the import seam; the only caller catches `TrajectoryImportError` alone, so picking the wrong file in the trace-import picker raises out of a Textual action handler  ·  _slice: CHAT-rest-3_
- Where: `tldw_chatbook/Chat/trajectory_import.py:80-108` (`_read_document`: `path.read_text(encoding="utf-8")` guarded by `except OSError` only; `json.loads` guarded by `except json.JSONDecodeError` only). Caller: `tldw_chatbook/UI/Screens/trajectory_screen.py:2026-2029` (`await asyncio.to_thread(load_imported_trace, path)` / `except TrajectoryImportError`). The picker at `:2054-2060` deliberately offers an **"All Files"** filter alongside "Trace files".
- Evidence: `cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY <<'EOF' ...` (script inline in the run; writes a latin-1 `.json` and a 20000-deep `[[[...]]]` `.json`, then calls `load_imported_trace`) ->
  ```
  1) *** UNHANDLED UnicodeDecodeError : 'utf-8' codec can't decode byte 0xe9 in position 50: invalid continuation byte
  2) *** UNHANDLED RecursionError : maximum recursion depth exceeded while decoding a JSON array from a unicode string
  3) handled TrajectoryImportError: Cannot read trajectory trace file '/var/folders/...   <- the OSError path IS handled
  ```
  Row 3 is the negative control: the author handled unreadable files, and missed decode failures.
- Why it matters: `action_open_trace`'s own docstring promises "Import failures surface as an error notification carrying the actionable message from the shared validator". For any non-UTF-8 file the user picks through "All Files" (a PNG, a `.sqlite`, a latin-1 JSON) that contract is broken and the exception leaves the action handler. `app.py:19025 _handle_exception` (TASK-32533) keeps the *screen* alive only when the raise came through `_PUMP_DISPATCH_FRAMES` with `pump is not self`; otherwise it calls `super()._handle_exception` and **the app exits**. Which branch an async binding-action takes is the unverified part (see Left UNVERIFIED) — the escape itself is verified.
- Recommended correction: widen the two guards in `_read_document` to the idiom this repo already uses ~15 times — `except (OSError, UnicodeDecodeError)` on the read and `except (json.JSONDecodeError, RecursionError, ValueError)` on the decode — re-raising `TrajectoryImportError`. Exact precedent in this same slice: `Chat/library_activity.py:501`, `Chat/library_preparation.py:190`, `Chat/local_server_discovery.py:451` (`except (RecursionError, UnicodeDecodeError, ValueError)` on the same `json.loads` shape), `Chat/trajectory.py:153`.
- Size: S · ADR: no · Confidence: **verified** (escape), inferred (crash-vs-toast)
- Pinning test: none. `grep -rn "UnicodeDecodeError\|RecursionError" Tests/Chat/test_trajectory_import*.py` -> no hits.
- Already covered: none
### P1 [D1] — `summarize_with_kobold` and `summarize_with_tabbyapi` are generator functions; every `analyze("koboldcpp"|"tabbyapi")` call returns `"Error: Unexpected result type <class '…_OpenAIStream'>"` and never contacts the server (ALREADY FILED: task-17387, To Do, priority high)  ·  _slice: LLM_
- Where: `tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py:697,712,718,1190,1208,1214,1285` (bare `yield` in the function body, not in a nested generator); re-exported and dispatched via `Summarization_General_Lib.py:36-46`, `:298-308` (`_CHAT_DISPATCH_NAME_ALIASES` maps `koboldcpp`→`kobold`), `:431-453`.
- Evidence: `inspect.isgeneratorfunction(inspect.unwrap(L.summarize_with_kobold))` → True (also tabbyapi, and both re-exports); end-to-end with a mocked 200 `{"results":[{"text":"THE SUMMARY"}]}` and `load_settings` patched: `S.analyze("koboldcpp", "some text", "summarize it", api_key="k", streaming=False)` → `"Error: Unexpected result type <class 'tldw_chatbook.LLM_Calls.recovery_review._OpenAIStream'>"`; same for tabbyapi. New detail beyond the task text: `recovery_review.unqualified` (`:127-131`) sees the generator, wraps it in `_OpenAIStream`, and `analyze()`'s `consume_generator` (`:585-606`, `inspect.isgenerator` only) passes the wrapper through untouched — so the failure is a visible error string, not the "truthy generator stored as evidence" the task describes.
- Why it matters: Library ingest analysis with `[analysis_defaults] provider = koboldcpp|tabbyapi` can never produce a summary.
- Recommended correction: per task-17387 (nest the streaming bodies; re-key the diagnostic ledger). Note the ledger tests already consume these as generators (`Tests/LLM_Calls/test_summarization_diagnostic_privacy.py:506,2789,2818` — `_consume_generator(summarize_with_kobold(...))`), i.e. the tests pin the DEFECT; the task text acknowledges that.
- Size: M (governed diagnostic ledger) · ADR: no · Confidence: verified
- Pinning test: `test_summarization_diagnostic_privacy.py::_invoke_local_credential` et al. — they assert the current (broken) generator contract.
- Already covered: task-17387 (and task-17383 In Progress for the config-section half).
### P1 [D1] — the Console OS-video fallback is dead: every `_open_video_with_os` call raises `TypeError`, swallowed into a misleading notice  ·  _slice: UIM-console_
- Where: callers `UI/Console_Modules/video.py:1308` (`_play_console_video`, the fallback when ffmpeg/ffplay are absent) and `video.py:978` (after an external save); wiring `UI/Console_Modules/wiring.py:808 open_video_with_os=lambda path: screen._open_video_with_os(path)`; definition `UI/Screens/chat_screen.py:19648 def _open_video_with_os(path: Path) -> None:` — an instance method with no `self` and no `@staticmethod`.
- Evidence: `<SCRATCH>/repro/uim_console_probe4.py` →
  ```
  descriptor type: function signature: (path: pathlib.Path) -> None
  TypeError: ChatScreen._open_video_with_os() takes 1 positional argument but 2 were given
  ```
  (the probe performs the exact `wiring.py:808` call, `screen._open_video_with_os(path)`, on a real `ChatScreen`.)
- Why it matters: with ffmpeg/ffplay missing, `/generate-video` playback silently never works — the user gets "Could not open the video with the system player." (`video.py:1315`); after a successful external save the user is told "Video saved, but could not open it automatically" (`video.py:986`) on every save. Both `except Exception` blocks (`video.py:979`, `:1309`) hide a plain programming error.
- Recommended correction: add `@staticmethod` at `chat_screen.py:19647` (the body uses no instance state).
- Size: S · ADR: no · Confidence: verified
- Pinning test: none.
- Already covered: none. Independently found from the definition side as UI-chat P1; this confirms it from the caller side, which is where the two swallowing handlers are.
- **Same defect as the UI-chat P1**, found independently from the caller side. Count it once.
### P1 [D1] — the RAG search cache leaks its memory accounting on every TTL prune and eventually stops caching permanently; every search after that is a cold miss  ·  _slice: RAG_
- Where: `tldw_chatbook/RAG_Search/simplified/simple_cache.py:1055-1088` (`_prune_expired_async`) and `:1090-1126` (`prune_expired`) delete entries from `self._cache` **without** decrementing `self._current_memory_bytes`. The sync twin `_prune_expired_sync` (`:1128-1150`) does it correctly — it calls `_update_memory_sync()` at `:1147`. The counter is then the gate on `put_async`'s eviction loop (`:642-644`) and on its give-up branch (`:646-651`, `logger.warning("Entry too large for cache") ; return`).
- Live path: `RAGService.__init__` (`rag_service.py:816`) constructs `SimpleRAGCache` with the default `max_memory_mb=100.0`; the production search path uses `get_async`/`put_async` only (stated at `simple_cache.py:748-752` and re-verified there), and `get_async:378-381` is what fires `_prune_expired_async`.
- Evidence — reproduced, isolated env, `max_memory_mb=1.0`, `ttl_seconds=0.2` (only to shorten the clock; nothing else changed):
  ```
  round 1: entries in cache =  0, _current_memory_bytes =    637.5 KB (cap 1024 KB)
  round 2: entries in cache =  0, _current_memory_bytes =   1020.0 KB (cap 1024 KB)
  round 3: entries in cache =  0, _current_memory_bytes =   1020.0 KB (cap 1024 KB)
  ... (rounds 4-6 identical)

  after the drift, is 'final' actually cached? -> False
  entries now: 0  _current_memory_bytes: 1020.0 KB
  get_async('final') returns: None  <-- cache is dead
  ```
  (script: 6 rounds of 5 `put_async` + `sleep(0.25)` + one `get_async` to fire the prune, then one more `put_async`/`get_async`. The counter reports 1020 KB of resident entries while the cache holds **zero**.)
- Time to failure under the shipped defaults: one cached entry for a `top_k=10` search with ~2 KB of text per chunk measures 31.9 KB via `_deep_getsizeof`, so the 100 MB cap is reached after ~3 200 leaked entries — about **32 full-cache prune cycles**, and a prune fires at most once per `_prune_interval = min(ttl/2, 1800)` = 30 minutes. A long-lived TUI session (this app runs for days) gets there; a short one does not. There is no recovery: nothing on the async path ever recomputes `_current_memory_bytes`, so once it ratchets up it stays up for the life of the process, and it only ever goes down by an eviction's share.
- Second-order effects before total failure: the effective cache budget shrinks monotonically with every expiry (so the cache degrades progressively, not only at the end), and `get_metrics()["memory_usage_percent"]` / `log_gauge("cache_memory_estimate_mb", …)` report a number that is wrong in the same direction the whole time.
- Recommended correction: make `_prune_expired_async` and `prune_expired` decrement the counter the way `_prune_expired_sync` already does — accumulate the pruned entries' sizes and subtract, or (simplest) recompute from the surviving entries. The two public prune bodies are also byte-identical duplicates of each other (`:1055-1088` vs `:1090-1126`), so the fix belongs in one shared body, not three.
- A companion inconsistency worth fixing in the same change: the async path sizes entries with `_deep_getsizeof` (`:604` region, a real object-graph walk) while the sync path uses `_estimate_entry_size` (`:970-1002`, a flat 1 KB per result), and both mutate the same `_current_memory_bytes`. Mixed use would make the accounting drift a second way. Production only uses the async path today, so this one is latent.
- Size: S · ADR: no · Confidence: verified (reproduced)
- Pinning test: none — `grep -rn "_current_memory_bytes" Tests/` finds no assertion on the counter after a prune.
- Already covered: none. (TASK-15701 covered the sync twins' *cache key*, a different defect in the same file.)
- **Re-verified by the orchestrator, worse than filed:** driven past the counter cap the cache stopped accepting writes at prune cycle 42 with `entries=0 counter=1022 KB cap=1024 KB` and never recovered — a permanent no-op for the process, not a degradation.
### P1 [D1] — the Transformers "Browse models dir" button can never open a picker: it imports the *top-level* `textual_fspicker`, which is not installed and is not a dependency (the library is vendored)  ·  _slice: EVENTS_

- Where: `Event_Handlers/LLM_Management_Events/llm_management_events_transformers.py:152-159` — `try: from textual_fspicker import FileOpen / except ImportError: app.notify("File picker utility (textual-fspicker) not available.", severity="error"); return`. Wired at `llm_management_events_transformers.py:196` → `UI/LLM_Management_Window.py:52,528`, button composed at `UI/LLM_Management_Window.py:1395`.
- Evidence (the handler run against a fake app):
```
cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY - <<'EOF'
import asyncio, inspect
from loguru import logger
from tldw_chatbook.Event_Handlers.LLM_Management_Events import llm_management_events_transformers as tr
from tldw_chatbook.Third_Party.textual_fspicker import FileOpen as VendoredFileOpen
class FakeApp:
    loguru_logger = logger
    def __init__(self): self.notes=[]; self.pushed=[]
    def notify(self, msg, severity="information"): self.notes.append((severity,msg))
    async def push_screen(self, screen, callback=None): self.pushed.append(screen)
app = FakeApp()
asyncio.run(tr.handle_transformers_browse_models_dir_button_pressed(object(), app, None))
print("NOTIFICATIONS:", app.notes); print("PICKERS PUSHED:", app.pushed)
print("vendored FileOpen accepts select_dirs?:", "select_dirs" in inspect.signature(VendoredFileOpen.__init__).parameters)
EOF
```
→
```
ERROR … textual_fspicker not found for Transformers model dir browsing.
NOTIFICATIONS: [('error', 'File picker utility (textual-fspicker) not available.')]
PICKERS PUSHED: []
vendored FileOpen accepts select_dirs?: False
```
and the module is genuinely absent, not merely unimported:
```
$PY -c "import importlib.util as u; print(u.find_spec('textual_fspicker'))"   → None
rg -n "fspicker" pyproject.toml                                               → only the vendored package-data line (:621)
```
`rg -n "^from textual_fspicker" --glob '*.py' tldw_chatbook/` finds exactly two: this line, and `Third_Party/textual_fspicker/__main__.py:20` (the vendored library's own demo entry point, which is allowed to).
- Why it matters: the button ships, is composed, and `Tests/ProductionApp/test_llm_destination_actions.py:101,121,175` asserts it exists **and** carries the tooltip "Choose the local Transformers models root directory." — a promise the handler cannot keep. Every press produces an error toast and no picker, with no way for the user to fix it (installing `textual-fspicker` is not a documented extra). The intended fallback is wrong twice over: the vendored `FileOpen` does not accept the `select_dirs=True` the handler passes at `:182`, so even importing the right module would `TypeError`.
- Recommended correction: use what every sibling in this package already uses — `from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedSelectDirectory` and push it with `callback=_make_path_update_callback(window, app, "transformers-models-dir-path")`, exactly as `llm_management_events_vllm.py:306-318 handle_vllm_local_directory_browse_requested` does. Drop the `try/except ImportError` entirely; the vendored package is not optional.
- Size: S · ADR: no · Confidence: **verified**.
- Pinning test: `Tests/ProductionApp/test_llm_destination_actions.py` pins the button's *presence and tooltip*, not its behaviour — which is why this survived. Adding one assertion that a picker is pushed would have caught it.
- Already covered: none.
### P1 [D1] — the prompt-queue shelf renders a literal backslash in front of any `[` the user typed, and the escape eats the cell budget it was measured against  ·  _slice: CHAT-rest-2_
- Where: `tldw_chatbook/Chat/console_prompt_queue.py:171` (`make_prompt_preview` → `rich.markup.escape`), consumed at `tldw_chatbook/UI/Console_Modules/prompt_queue.py:182` → `:219` → `:350` into `tldw_chatbook/UI/Console_Modules/prompt_queue.py:320` `Static("", id="console-prompt-queue-preview", markup=False)`.
- Evidence:
  ```
  cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$PWD $PY - <<'EOF'
  from tldw_chatbook.Chat.console_prompt_queue import make_prompt_preview
  s = "summarize [draft] and TODO[1]"; print(repr(s)); print(repr(make_prompt_preview(s)))
  EOF
  ```
  → `input   : 'summarize [draft] and TODO[1]'` / `preview : 'summarize \\[draft] and TODO[1]'`.
  The consuming widget is composed with `markup=False` (`rg -n 'console-prompt-queue-preview' tldw_chatbook/UI/Console_Modules/prompt_queue.py` → `:320  yield Static("", id="console-prompt-queue-preview", markup=False)`), so that backslash is rendered verbatim.
- Why it matters: two consequences. (a) A queued prompt containing `[` shows as `Next: "summarize \[draft]"` — visible corruption of the user's own text. (b) `make_prompt_preview`'s docstring states "Rich escaping happens only after fitting, so escape syntax does not consume the visible-cell budget" — true only on a markup surface; on `markup=False` every inserted `\` *does* consume a cell, so `PROMPT_PREVIEW_CELL_BUDGET` is silently exceeded and the preview can push the Manage/Pause buttons.
- Recommended correction: pick one side. Either drop `escape_markup` from `make_prompt_preview` (the terminal-control stripping + grapheme truncation above it is what actually makes the string safe, and `markup=False` already neutralises tags), or set `markup=True` on that one `Static`. Dropping the escape is the smaller diff and keeps the byte-exact user text; it also makes the `cell_len` budget true again.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/Chat/test_console_prompt_queue.py::test_preview_is_one_line_terminal_safe_and_markup_escaped` (and `::test_preview_truncates_by_cells_without_splitting_graphemes`) — both assert through `Text.from_markup(preview).plain`, i.e. they pin the *assumption that the surface interprets markup*. They pass today and would keep passing after the widget was switched to `markup=False`; they are the reason this drifted. A fix must change these two tests to assert the string the widget actually receives.
- Already covered: none. Same defect class as CHAT-rest-1 P1 (`Chat/console_display_state.py:93`) — worth fixing as one pass over the Console's markup-off Statics.
### P1 [D1] — the realtime pre-connect credential gate is defeated by a placeholder key, and the placeholder is handed to the transport  ·  _slice: UIM-console_
- Where: `UI/Console_Modules/realtime.py:730` (`if not self._console_realtime_api_key():`) reading `realtime.py:616 get_api_key(...)`; the value is then put on the wire at `realtime.py:783-786` (`RealtimeSessionConfig(api_key=self._console_realtime_api_key(), ...)` → `_build_console_realtime_session` → `OpenAIRealtimeSession`).
- Evidence: run-1 probe (`uim_console_probe.py`): with `get_api_key` returning `"YOUR_KEY"`, `_start_console_realtime_connect` dispatched the `console-realtime-connect` worker — the gate passed. This run, `<SCRATCH>/repro/uim_console_probe5.py` →
  ```
  resolve_provider_api_key(placeholder) -> None
  api_key handed to the provider session: '<API_KEY_HERE>'
  ```
  i.e. the shipped placeholder is rejected by `config.py:1400 resolve_provider_api_key` but reaches `RealtimeSessionConfig.api_key` and the provider session constructor verbatim.
- Why it matters: the gate's own comment (`realtime.py:723-729`) states its purpose — "dispatching one anyway would spend the connect timeout to come back with whatever 401 text the provider chose, and the fallback toast would quote THAT instead of the one thing the user can act on". A placeholder in `[api_settings.openai] api_key` produces exactly the outcome the gate was written to prevent, plus a credential-shaped placeholder sent to a third party.
- Recommended correction: gate on `config.resolve_provider_api_key(...)` (or `is_valid_provider_api_key`, `config.py:1411`) instead of raw truthiness, in `_console_realtime_api_key` at `realtime.py:610-621`; the root cause (`get_api_key` not applying that check) is ENTRY-config P1.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none found.
- Already covered: none; root cause tracked as ENTRY-config P1.
### P1 [D2] — 77% of the Session Git panel's arrow-key handler is a DOM existence probe guarding 0.03 ms of work  ·  _slice: W-library_
- Where: `library_file_notes_git_panel.py:1883` — `if any(not list(self.query(selector)) for selector in selectors): return` in `_sync_action_layout`; reached per keypress from `_row_highlighted` (`:3432` → `_update_actions`), three times per `set_mutating` (`:3038`), and on every resize (`:1862`).
- Evidence: `<SCRATCH>/probe_gitpanel_hot2.py` (panel mounted 120x40, 6 ready rows), **re-run by me**:
  `existence probe only (5x self.query): 1.404 ms` / `_visible_action_cells x5: 0.027 ms` / `_action_row_width x5: 0.004 ms` / `_sync_action_layout total: 1.431 ms`.
  First run (`probe_gitpanel_hot.py`): `_update_actions 1.841 ms`, of which `_sync_action_layout 1.452 ms`.
- Why it matters: `self.query(selector)` materialises a full-subtree `DOMQuery` five times to decide whether to do 0.03 ms of measuring; it runs on every Up/Down in the changed-files list.
- Recommended correction: delete the probe and wrap the `needs_stack` computation in `try: … except NoMatches: return` — `NoMatches` is already imported (`:18`) and already used this way at `:2782`.
- Size: S · ADR: no · Confidence: verified (measured twice, different processes)
- Pinning test: `Tests/UI/test_library_file_notes_git.py::test_action_controls_fit_from_visible_label_cells_and_recompute` (`:2323`) pins the RESULT, not the probe.
- Already covered: none
### P1 [D2] — A folder import of N files freezes the UI thread for ~3.3 ms × N: 0.33 s at 100 files, 4.4 s at the 1000-file scan-limit maximum (measured on the real registry + store)  ·  _slice: ENTRY-app_
- Where: `tldw_chatbook/app.py:3322-3346` (per-file loop in `submit_library_ingest_job`, documented "UI-thread only" at 3184) → `app.py:2681` registers `app.py:2749-2787 _schedule_settled_research_source_operations` as a registry listener; the listener calls `self.library_ingest_jobs.jobs()` (a `_copy_job` deep copy of every job, `Library/library_ingest_jobs.py:1593-1608`) **before** any of its early-returns (2757-2765), and the registry fires listeners on every mutation (`grep -c '_notify_listeners()' Library/library_ingest_jobs.py` → 16 sites). Each `submit` also runs one store transaction (`Library/library_ingest_jobs.py:588-600 _persist` → `DB/Library_Ingest_Jobs_DB.py:447-451 upsert_job` = `with self.transaction()` per job). Both halves live outside this slice; the synchronous UI-thread loop and the O(n) listener are app.py's.
- Evidence: `<SCRATCH>/bench_ingest_submit.py` (real `LibraryIngestJobRegistry` + real file-backed `LibraryIngestJobsDB`, listener body = the same three statements as app.py:2757-2780) →
  ```
  n=  100 store=False listener=False  submit_total=     0.5 ms
  n=  100 store=True  listener=False  submit_total=   292.5 ms  (2.92 ms/job)
  n=  100 store=True  listener=True   submit_total=   331.0 ms  (3.31 ms/job)
  n= 1000 store=False listener=True   submit_total=  2056.0 ms  (2.06 ms/job)   <- O(n^2): 100->1000 files = 108x
  n= 1000 store=True  listener=False  submit_total=  2803.9 ms  (2.80 ms/job)   <- one COMMIT per job
  n= 1000 store=True  listener=True   submit_total=  4444.8 ms  (4.44 ms/job)
  ```
  `collect_directory_files(candidate, scan_limit)` default `library.ingest_directory_scan_limit` = 1000 (app.py:3150) — so 1000 is the designed maximum, not a pathological input. Store is attached before any user submit (`_apply_ingest_job_restore` app.py:3124-3126, from the boot restore worker).
- Why it matters: every file in a folder import pays a synchronous fsync'd commit plus a deep copy of the whole queue, on the event loop, inside one click handler; a 1000-file import wedges input for ~4.4 s and a 100-file import for a visible third of a second. The same listener then fires on each of the 1000 later `mark_parsing`/`mark_done` transitions (≈2 ms each at n=1000 → another ~2 s spread over the run).
- Recommended correction: (S, app.py) in `_schedule_settled_research_source_operations` move the `scheduler is None` / restore-in-progress returns above the `jobs()` call and drop the `intersection_update` scan when `_research_source_terminal_jobs_scheduled` is empty; (M, cross-module — hand to the LIBRARY/DB reviewer) give `LibraryIngestJobRegistry` a non-copying iterator for listeners and a batch-persist context so a folder submit is one store transaction; app.py's folder loop then wraps the loop in it. Keep the exactly-once semantics the pinning test states.
- Size: M · ADR: no · Confidence: verified (component benchmark on the worktree's own registry/store; the end-to-end freeze in the running app is inferred from it — see UNVERIFIED)
- Pinning test: `Tests/App/test_submit_library_ingest_job.py::test_startup_reconcile_terminal_listener_schedules_exactly_once` (states the listener's exactly-once scheduling as a requirement; nothing pins its cost) · `Tests/App/test_submit_library_ingest_job.py::test_required_persisted_submit_is_durable_before_listener_visibility` (durability-before-listener ordering — a batch persist must keep it)
- Already covered: none (`grep -lisE 'app\.py' backlog/tasks/*.md | xargs grep -liE 'decompos|split|god.module|size.ratchet'` → empty; no task names the ingest listener cost)
### P1 [D2] — Every Notes keystroke pays 462 µs for four whole-subtree `DOMQuery` truth-tests that `query_one` answers in 0.6 µs  ·  _slice: W-library_
- Where: `library_notes_canvas.py:3155, 3199, 3213-3215` (editor branch) and `:3122, 3129, 3132, 3139` (list branch) in `apply_compact_presentation`, called from `apply_session_state:3419`; same shape at `sync_state:1585, 1605`.
- Evidence: `<SCRATCH>/probe_notescanvas_perf.py` (real mounted editor canvas, 68 children, 35 KB body):
  `4x bool(query('#id')) 462.4 us` · `3x query_one('#id') 0.6 us` · `2x bool(query) in sync_state 230.2 us` · `apply_compact_presentation() 509.8 us` · `apply_session_state() 1002.3 us` · `len(BODY.split()) 80.3 us` (the scan this file's own design ruled out). Absent id: `bool(query) 119.7 us` vs `query_one 7.4 us`.
  Per-keystroke reachability traced: `library_notes_controller.py:3703/:3730` (`@on(Input.Changed, "#library-note-title")` / `@on(TextArea.Changed, "#library-note-body")`) → `_apply_library_note_presentation_state:1677` → `canvas.apply_session_state(...)`.
- Why it matters: 91% of `apply_compact_presentation`'s cost and 46% of every keystroke's, for existence tests — and `update_note_chrome_facts`' own docstring (`:3571-3579`) records the same anti-pattern being measured at 263.6 µs and fixed.
- Recommended correction: the sibling shape 40 lines over is right — `library_media_canvas.py:740` uses `try: self.query_one(...) except NoMatches: return`. Same swap for all seven sites plus the two in `sync_state`.
- Size: S · ADR: no · Confidence: verified (measured)
- Pinning test: none — `Tests/UI/test_library_honesty_accessibility.py:1272,1289` and `Tests/UI/test_library_multiselect_media.py:2930` drive `apply_compact_presentation` but assert labels, never lookup shape.
- Already covered: none
### P1 [D2] — In the media Reader, every `]`/`[` keypress, focus change and viewer sync runs 2–4 synchronous sqlite reads on the event loop (a full up-to-500-row review-set load each time), ~6 ms per read pair measured  ·  _slice: UI-library_
- Where: `tldw_chatbook/UI/Screens/library_screen.py:31238` `_review_set_active` (docstring: "this runs on every key resolution and footer render"), `:31098` `_active_review_progress`, `:31131` `_active_review_loaded_at_last`, `:31169` `_active_review_set_banner`, `:30987` `_walk_active_review_set_unguarded`; all call `service.get_active_review_set()` (loads header + every pinned item, `REVIEW_SET_CAP`=500) and `:30911` `_review_set_live_ids` (`media_db.execute_query("SELECT id FROM Media WHERE id IN (…)")` + `fetchall`) synchronously. Reached from `check_action` (`:25411`, `:25474` — bindings `]` `[` `R` `m`), the footer builder (`:4341`, `:4362` via `_register_footer_shortcuts`, which runs on `compose_content`, every rail switch, every `on_descendant_focus` in the viewer `:10705`), and the media controller's viewer build/sync (`library_media_controller.py:3754`, `:3913` → screen `:3124`).
- Evidence: probe under the isolated env (LibraryCollectionsDB + ReviewSetService with a 500-item active set; empty MediaDatabase):
  `get_active_review_set (500 items): median 5.064 ms, p95 6.415 ms` · `live_ids IN(500) on Media: median 0.080 ms, p95 0.116 ms` · `one _active_review_progress() equivalent (both queries + progress): median 6.218 ms, p95 8.085 ms`. Per `]` press in a 500-item set the chain is `check_action`→`_review_set_active` (≈5 ms) → `_walk…` (`get_active_review_set` + live_ids + `mark_item_done`/`set_cursor`/`refresh_completion` writes) → viewer sync → `_active_review_set_banner` (≈6 ms) → `_register_footer_shortcuts` → `_active_review_progress` (≈6 ms) + `_active_review_loaded_at_last` (≈5 ms): ≈25–30 ms of loop-blocking sqlite per keypress, scaling with set size. Probe script: `<SCRATCH>/probe_review_set` (20 lines, in the transcript).
- Why it matters: the Reader's ]/[ traversal and footer refresh stall the UI loop for tens of ms per gesture on a large review set; every off-loop sibling in the same section (`_review_these_worker`, `_review_set_picker_worker`, `_auto_resume_review_set_worker`, :31397–:31899) already routes the same reads through `_run_library_service_call(isolate_in_worker=True)`, so this is the gap, not the design.
- Recommended correction: memoise the active-set snapshot the way `_decorate_library_media_reviewed` (:16097) already does — one screen-level `_active_review_set_snapshot()` keyed on `service.revision` (bumped by every write in `ReviewSetService._write`) returning `(review_set, live_ids)`, and route the five synchronous readers through it; the write path (`_walk…`) stays synchronous but drops from 2 loads to 1. No new helper module.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/UI/test_review_set_banner.py`, `Tests/UI/test_review_set_walker.py`, `Tests/Architecture/test_library_media_wiring.py` exercise these readers (behaviour, not call-count); no test asserts the per-call DB load, so a memo does not contradict a pinned requirement.
- Already covered: none
### P1 [D2] — One keystroke in any Library/RAG editor field re-loads the active RAG profile 7× and re-validates it 4× on the event loop (measured 388-426 ms per keystroke; 1.3 s for the first)  ·  _slice: UI-settings_
- Where: every `handle_library_rag_*_changed` (`settings_screen.py:25630-25932`, 24 handlers) → `_stage_library_rag_value:9339` (→ `_library_rag_loaded_values` → `load_rag_defaults_from_active_profile`) → `_mark_library_rag_settings_staged:9353` → `_library_rag_validation_result:9355`, `_update_library_rag_preview:9366`, `_update_library_rag_validation_classes:9367`, `_update_library_rag_soft_warning:9368`, `_update_draft_status_widgets:9369` → `_update_category_state_banner`/`_category_state_banner_text:8566-8572` and `_update_guided_action_widgets` → `_guided_actions_enabled:7973` + `_guided_action_message:7899-7904`. Each of those re-derives `_library_rag_setting_values():6672` from scratch (`load_rag_defaults_from_active_profile()` + `validate_library_rag_defaults`). Compose does the same: `_render_library_rag_detail:18684` (`_library_rag_setting_values`), `:19193` (`_library_rag_soft_warnings`), `:19203` (`_library_rag_preview_rows`), `:18718` (`_library_rag_first_run_active`), `:18698` + `_render_library_rag_profile_block:18381-18382` (`active_profile_info`/`list_profiles_grouped` twice each).
- Evidence: sync-constructed `SettingsScreen(app)` (the construction `Tests/UI/test_settings_rag_profile_region.py:144` uses), `active_category = LIBRARY_RAG`, module names wrapped with counters, then `_stage_library_rag_value("default_top_k", n)` + `_mark_library_rag_settings_staged()` — the exact handler body: `PROBE keystroke#0: 1343.2 ms; calls = {'load_rag_defaults_from_active_profile': 7, 'config.get_cli_setting': 170, 'validate_library_rag_defaults': 4, 'soft_config_warnings': 1}`, `#1: 387.8 ms (7 loads, 32 get_cli_setting, 4 validations)`, `#2: 426.0 ms (same)`. Compose parts: `_library_rag_setting_values=51.9ms _library_rag_soft_warnings=42.8ms _library_rag_preview_rows=40.5ms _library_rag_first_run_active=21.2ms` (≥156 ms of reads per category open before any widget is built). Per-call costs: `load_rag_defaults_from_active_profile() median=53.1ms`, `active_profile_info() 10.0ms`, `list_profiles_grouped() 11.0ms`, `validate_library_rag_defaults 10.5ms`, `index_change_pending 9.9ms`. cProfile of the adapter load: all of it is 4× `config.get_cli_setting` → `Backup_Recovery` admission → `posix.open` (9640 opens / 5 loads).
- Why it matters: the Library/RAG category is the only Settings surface where typing stalls the TUI for ~0.4 s per character; the 7×/4× multiplier is structural (the handlers never cache the loaded defaults), so it holds even if the per-read price is lower on a settled profile (see UNVERIFIED for the env caveat).
- Recommended correction: (S) load once per event — cache `_library_rag_loaded_values()` on the screen for the lifetime of a category visit (invalidate on set-active/clone/rename/delete/save/revert, exactly the moments the file already lists for `_image_gen_raw_section_cache:6762`), and compute `_library_rag_validation_result()` once inside `_mark_library_rag_settings_staged` and pass it down; the `_guided_action_message`/`_category_state_banner_text` re-validations then read the cached result. The Image Gen block already models this cache (its docstring at `:3333-3348` gives the three invalidation points).
- Size: S · ADR: no · Confidence: verified (counts + wall time on the shipped handler body; absolute ms are from the isolated env)
- Pinning test: none for cost; `Tests/UI/test_settings_rag_profile_region.py` pins behaviour only.
- Already covered: none (task-19647 is the Backfill-control ADR-003 drift, not this)
### P1 [D2] — The 0.25 s credential-readiness poll costs ~49 ms per tick at idle: it rebuilds provider readiness outside any config-admission scope, so one tick performs six `load_settings()` admissions (~66 verified-directory opens)  ·  _slice: UI-chat_
- Where: `chat_screen.py:11966-11984` (`_poll_console_credential_readiness`), armed at 16805 `set_interval(0.25, …)` for the screen's whole life (`is_current`-gated), stopped only in `on_unmount` 16994 — NOT in `on_screen_suspend` (23476–23479 stop the other timers). The cost sits in `_provider_readiness_app_config` 7624–7659 → `load_settings()` → `Backup_Recovery.config_participants.operation` → `storage_admission.acquire_storage` → `private_paths._open_verified_parent`.
- Evidence: probe `<SCRATCH>/probes/test_probe_credential_poll_cost.py` (ConsoleHarness): `readiness_ms=41.4 poll_tick_ms=37.4 app_config_ms=11.7`. cProfile probe `test_probe_credential_poll_profile.py` over 20 ticks: `_poll_console_credential_readiness` cum 0.977 s (**49 ms/tick**); `_provider_readiness_app_config` **120 calls / 20 ticks = 6 per tick**, cum 0.953 s; `config_participants.operation` 120 calls; `storage_admission._acquire_storage` 120 calls cum 0.739 s; `private_paths._open_verified_parent` **1320 calls** (66 per tick) cum 0.361 s; `session._maybe_refresh_stale_default_console_settings` 40 calls cum 0.331 s. Snapshot was disk-loaded (`disk_loaded=True`), i.e. the production branch of 7646.
- Why it matters: with Console current and nothing happening, ≈49 ms of every 250 ms of event-loop time (≈20 %) re-derives readiness that almost never changes; the docstring's "cheap (cached)" claim for `load_settings()` is false under the backup-recovery admission wrapper (≈8 ms per call). The 0.2 s transcript tick was routed through ONE `operation(config)` per tick (`_run_console_config_sync` 21604) precisely to avoid this — the credential poll bypasses that seam.
- Recommended correction (S): (1) read `subscription_readiness_revision()` FIRST and return when it equals the cached revision — that is the only external signal this poll exists to notice; (2) when it changed, build readiness inside `with self._console_derivation_scope():` (memoises `app_config` for the pass: 6 → 1 admissions) or via `_run_console_config_sync`; (3) stop the timer in `on_screen_suspend` beside its siblings. Cross-slice: the per-call cost of `load_settings()` under `Backup_Recovery` admission belongs to ENTRY-config / Backup_Recovery — cite, don't fix here.
- Size: S · ADR: no · Confidence: **verified** (measured in harness; a real profile with keyring/network credentials can only be slower)
- Pinning test: none for the tick's cost; `Tests/Architecture/test_timer_path_static_update_inventory.py` inventories `.update()` receivers on timer paths, not this timer's body.
- Already covered: none (task-26834 covers tab-switch remount/CSS/title lookup, not this idle poll).
### P1 [D2] — The Notes editor re-renders the hidden Preview source on every keystroke, contradicting the comment two lines above it  ·  _slice: W-library_
- Where: `library_notes_canvas.py:3361` — `preview_source = render_preview_source(snapshot.body, title=snapshot.title)` computed before the `if show_preview and preview_body.source != preview_source:` gate; the comment at `:3347-3350` states the intent ("Keep the hidden Preview stale while typing … so edits cannot queue an unbounded hidden-render backlog").
- Evidence: `<SCRATCH>/probe_notescanvas_perf.py` → `render_preview_source(35KB) 177.9 us` (2.25× the 79.0 µs whole-body scan the design banned), `render_preview_source(350KB) 1859.5 us`. Three full-body passes (`_drop_leading_title_heading`'s `split("\n")`, `render_note_links`, `render_obsidian_callouts`).
- Why it matters: 1.86 ms per keystroke on a large note, in Edit mode, for output nobody can see.
- Recommended correction: move the call inside `if show_preview:`. One line.
- Size: S · ADR: no · Confidence: verified (measured)
- Pinning test: none — every test hit (`test_library_notes_w4_editor.py:433,440`, `w3_layout.py:195`, `w5_import_preview.py:85`) calls the pure function directly; none asserts it runs while hidden.
- Already covered: none
### P1 [D2] — The Skills trust header's `has_skills` predicate exists twice and the two copies disagree; the in-place path drops the recovery banner  ·  _slice: W-library_
- Where: `library_skills_canvas.py:1216-1219` (compose, includes the `trust_posture == "recovery_review"` disjunct) vs `:992-997` (`sync_state`'s `header_only` path, which does not).
- Evidence: `<SCRATCH>/probe_skills_trust_header_divergence.py`, identical inputs (`SkillsListState(rows=(), count=0, source_summary_fresh=False)`, posture `recovery_review`):
  `COMPOSE header: 1 action: 1` / `SYNC header: 0 action: 0` → `AssertionError: assert 0 == 1  # '#library-skills-trust-action'`.
- Why it matters: the `header_only` path exists precisely because "the posture read may settle after rows become interactive" (`:1213-1215`), so when the posture settles to `recovery_review` during a routine refresh the user loses the banner and the only list-level "Review restored skills" button until an unrelated input forces a full recompose.
- Recommended correction: extract the predicate beside `skill_trust_header_line` in `library_skills_state.py` and call it from both sites.
- Size: S · ADR: no · Confidence: verified (divergence); inferred (that this state pair occurs in a live session — settle by logging `(trust_posture, state.source_summary_fresh)` pairs in `_library_skills_canvas_kwargs` during a Skills refresh on a restored trust store)
- Pinning test: none (`grep -rn "recovery_review" Tests` → zero hits in the Skills canvas tests)
- Already covered: none
### P1 [D2] — The splash screen re-reads the config file 9 times to build one 9-key dict, costing ~59 ms of every app launch before first paint  ·  _slice: W-top_
- Where: `tldw_chatbook/Widgets/splash_screen.py:213-224` (`_load_splash_config`, a dict comprehension calling `get_cli_setting` once per key; called from `__init__:83`), plus a 10th read at `:280` (`_get_predefined_cards` → `splash_screen.custom_image_path`). Near-identical copy at `tldw_chatbook/Widgets/settings_splash_screen_viewer.py:61-79`.
- Evidence (verified):
  ```
  cd $WT && source env.sh && PYTHONPATH=$WT $PY -
  current per-key loop (9 keys): 58.9 ms
  one section read + .get():      6.34 ms      (~9x)
  9x get_cli_setting warm:       48.5 ms       (5.4 ms/call warm — matches the sibling's 11 ms cold figure)
  ```
- Why it matters: `SplashScreen.__init__` is on the startup path of every launch, and 9 of the 10 reads are for keys in the SAME `[splash_screen]` table — `get_cli_setting` pays a full settings resolution per call. This is not a cache-miss story: the numbers above are all warm.
- Recommended correction: one section read — `section = get_cli_setting("splash_screen", default={}) or {}`, `effects = section.get("effects", {})`, then `.get(key, default)` per key. `get_cli_setting`'s own docstring (`config.py:8447-8477`) documents the section-only call shape. Put the resulting loader in ONE place (see the D4b below) and have both callers use it.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

---
### P1 [D2] — `AgentRunsDB.reconcile_orphaned_runs` re-scans the ENTIRE terminal run history on the first construction per process, issuing one `agent_run_steps` query per run and JSON-parsing every step payload, inside a `BEGIN IMMEDIATE` write transaction — and one of the construction sites is a Textual `compose()`  ·  _slice: DB-rest_
- Where: `tldw_chatbook/DB/AgentRuns_DB.py:2340-2519`. The cost is the second half: `:2494-2497` selects **every** non-running primary/subagent row (`SELECT id, status FROM agent_runs WHERE status != 'running' AND agent_kind IN ('primary','subagent')` — no time bound, no LIMIT), then `:2508` calls the closure `run_observations(row["id"])` (`:2391-2419`) per row, which runs `SELECT payload FROM agent_run_steps WHERE run_id = ?` (`:2394-2397`) and `json.loads` on every payload. Called unconditionally from `__init__` (`:283-287`).
- Evidence (isolated env, temp-file DB, steps carrying the expected terminal `kind` so the diagnostic INSERT does *not* fire — i.e. this is the steady-state cost, not a one-off repair):
  - 5000 runs × 50 steps (250 000 step rows, 119 MB file): `first open : open+reconcile = 326.2 ms` · `second open : open+reconcile = 331.9 ms` · `guarded : open (sweep skipped) = 34.2 ms` → **~295 ms of reconcile on every process's first open**, repeated identically on the next launch.
  - Linear in history: 200 runs×20 = 46.5 ms, 1000×20 = 83.3 ms, 2000×20 = 136.8 ms (same script, `_swept_paths` cleared between runs).
- Why it matters: `AgentsSettingsPanel.__init__` calls `_derive_runs_db(app_instance)` → `AgentRunsDB(...)` (`Widgets/settings_agents_panel.py:131-146`, `:177`), and that panel is constructed inside `settings_screen.py:20834` `yield AgentsSettingsPanel(self.app_instance, id="settings-agents-panel")` — a `compose()` body, i.e. the Textual event loop. Whichever construction happens first in the process pays the full sweep; when that is Settings ▸ Agents, the UI is frozen for the duration and the `BEGIN IMMEDIATE` also holds the single SQLite write lock against any concurrent agent writer. (`Chat/console_runtime.py:3458` is the well-behaved site — its docstring says "Call via `asyncio.to_thread`".)
- Recommended correction: two independent fixes, both small. (1) Replace the per-run `run_observations` query with the class's own `_batch_hydrate_steps` (`:1687-1723`) — it exists for exactly this and already chunks at `_IN_CLAUSE_CHUNK`; the file's own docstring calls no-N+1 "this file's existing no-N+1 precedent (e.g. TASK-1972's conversation-level `change_snapshots` fetch)". (2) Bound the terminal-row scan — the lifecycle-capture backfill only needs rows that could plausibly predate the capture code, so an `AND created_at > ?` watermark (persisted like the `schema_version` audit rows) turns a full-history scan into an incremental one. Failing both, move the sweep off the event loop at the `compose()` site.
- Size: M · ADR: no · Confidence: verified (timings measured; the `compose()` reachability is a traced read of the two files named)
- Pinning test: none measuring cost. `rg -n "reconcile_orphaned_runs" Tests/` — see dispositions.
- Already covered: none
### P1 [D2] — `ConsoleWorkspaceSwitcherModal.compose()` runs a synchronous sqlite SELECT (~2 ms) per workspace row, undoing the caller's deliberate off-thread fetch  ·  _slice: W-console-2_
- **Where:** `console_workspace_switcher_modal.py:199`, inside the `for index, workspace in enumerate(self._workspaces)` loop opened at `:183` **inside `compose()`**, calling `workspace_persona_label_suffix` (`:22-65`), which at `:42` calls `registry.get_workspace(...)` and at `:56` `personas.get_persona_profile(...)`.
- **Evidence:**
  - `tldw_chatbook/Workspaces/registry_service.py:658-674` — `get_workspace` is `with self.db.connection() as conn: conn.execute("SELECT * FROM workspace_records WHERE workspace_id = ?").fetchone()`. Plain synchronous sqlite.
  - `probe_ws.py`, real `WorkspaceDB` in a tmp dir, 200 warm iterations →
    `get_workspace warm ms median=2.0116 mean=2.0416 max=2.5979`
  - The caller already fetched the same records **off the loop**: `tldw_chatbook/UI/Console_Modules/workspace.py:4259-4263`
    `workspaces = tuple(await storage_call(registry_service, "list_workspaces", include_archived=True))`, and `storage_call`
    (`tldw_chatbook/Chat/conversation_archive_actions.py:50-69`) is `await asyncio.to_thread(call, ...)`. Those exact
    `WorkspaceRecord`s reach the modal at `workspace.py:4310-4314` — and `workspace_persona_label_suffix` already accepts
    them: `:45-46` `if record is None: record = workspace`.
- **Why it matters:** the modal's first paint blocks the event loop for N × ~2 ms of sqlite plus N persona lookups (10 workspaces ≈ 20 ms, 30 ≈ 60 ms) on the very screen the caller took care to keep off it. The re-read is also strictly redundant with data already in hand.
- **Recommended correction:** use the passed-in `WorkspaceRecord`'s `assistant_defaults` (the branch the code already has as a fallback), or resolve every suffix once, off-thread, in `_open()` beside `list_workspaces` and pass them in. No per-row service call in `compose()`.
- **Size:** S · **ADR:** no · **Confidence: verified** (per-call timing measured; the N-row multiply is a direct read of the loop)
- **Pinning test:** none — `grep -rn "workspace_persona_label_suffix" Tests/` returns nothing; only the module defines it.
- **Already covered:** none
### P1 [D2] — `MCPWorkbench._collect_snapshots()` burns 20.6 ms of pure `get_cli_setting` on the event loop, and 14 call sites re-run it  ·  _slice: UIM-nav-mcp-persona_
- Where: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py:1288-1295` (the 4 reads) — re-entered from `:1082` (`reload`), `:3771` (`_switch_source`), `:3836` (`_select_server_key`), `:4043`, `:4096`, `:4178`, `:5760` (`_save_builtin_flag`), `:5812` (`_save_tool_gate`), `:5861`, `:5956`, `:6041`, `:6199`, `:6309`
- Evidence:
  ```
  cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$PWD $PY - <<'EOF'
  import asyncio, time
  import tldw_chatbook.UI.MCP_Modules.mcp_workbench as wb
  calls=[]; real=wb.get_cli_setting
  wb.get_cli_setting = lambda *a, **k: (calls.append(a[:2]), real(*a, **k))[1]
  w = wb.MCPWorkbench.__new__(wb.MCPWorkbench)
  w._source="local"; w._server_mutations_available=False; w._catalog_records={}; w._service=lambda: None
  asyncio.run(w._collect_snapshots()); calls.clear()
  t=time.perf_counter(); asyncio.run(w._collect_snapshots()); d=time.perf_counter()-t
  print(len(calls), calls); print(f"{d*1000:.1f} ms")
  EOF
  ```
  ->
  ```
  config reads per _collect_snapshots(): 4 [('mcp','enabled'),('mcp','expose_tools'),('mcp','expose_resources'),('mcp','expose_prompts')]
  wall time for one _collect_snapshots() (no service, no I/O): 20.6 ms
  ```
  Where the time goes (cProfile over 50 warm `get_cli_setting("mcp","enabled",False)` calls): every call runs `load_cli_config_and_ensure_existence` -> `Backup_Recovery/config_participants.py:400 wrapped` -> `storage_admission.acquire_storage` -> **24100 `posix.open` calls for 50 reads = 482 syscalls per config read** (~8 ms/call profiled, ~5 ms unprofiled).
- Why it matters: this 20.6 ms is paid synchronously on the asyncio loop by every rail click, every source/scope switch, every built-in-flag checkbox, every tool-gate button and every lifecycle completion in the Hub — before any of the server round-trips those paths also make. The "config reads are cache-backed" assumption does not hold at the call boundary: the cache lookup itself goes through the storage-admission scope.
- Recommended correction: hoist the 4 reads into one snapshot read per `_collect_snapshots()` (they are all `[mcp]` keys — one `get_cli_setting("mcp", ...)`-free section read, or a single cached tuple invalidated by `_save_builtin_flag`), or await them off-loop via `asyncio.to_thread` the way `_save_builtin_flag` already does for the write. The 482-syscall config read itself is `config.py`/`Backup_Recovery`'s to fix and out of this slice.
- Size: M · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none
### P1 [D2] — `PromptsDatabase.search_prompts` materialises EVERY matching prompt id in Python and binds them as one `IN (?,?,…)` list: search cost grows linearly with the match count and the call hard-fails with `too many SQL variables` past 32766 matches  ·  _slice: DB-rest_
- Where: `tldw_chatbook/DB/Prompts_DB.py:3884-3922` (the two unbounded `fetchall()`s into `matching_prompt_ids`, then `:3919-3921` `conditions.append(f"p.id IN ({id_placeholders})")`), consumed by the count at `:3935` and the page at `:3939`.
- Evidence (isolated env, temp-file DB, rows inserted directly + FTS populated so the state matches a normal library):
  - `SQLITE_LIMIT_VARIABLE_NUMBER` on this build = **32766** (`$PY -c "import sqlite3; print(sqlite3.connect(':memory:').getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER))"`).
  - 32800 prompts all matching `dragon` → `search_prompts("dragon", page=1, results_per_page=20)` → `DatabaseError: Failed to search prompts: Query execution failed: too many SQL variables`, raised from `Prompts_DB.py:3935` (`SELECT COUNT(p.id) FROM Prompts p WHERE p.deleted = 0 AND p.id IN (?,?,…`).
  - Cost below the cliff, same script at three sizes: `N=  2000 … elapsed=6.2 ms` / `N= 10000 … elapsed=29.3 ms` / `N= 32000 … elapsed=93.3 ms` — and only 20 rows are ever displayed. The work is linear in the match count, not the page size.
- Why it matters: this is the shipped prompt-search path (`UI/Console_Modules/prompts.py:1353` → `Prompts_Interop.py:475`, `Prompt_Management/prompt_scope_service.py:581`, `Library/library_local_rag_search_service.py:789`). A broad one-token query against a bulk-imported prompt library pays ~3 ms per 1000 matches on every search and stops working entirely — an error toast, not an empty result — once the library passes ~32.7k matching rows.
- Recommended correction: keep the id set in SQLite instead of round-tripping it — replace the `IN (...)` with `p.id IN (SELECT rowid FROM prompts_fts WHERE prompts_fts MATCH ?) OR p.id IN (SELECT prompt_id FROM PromptKeywordLinks WHERE keyword_id IN (SELECT rowid FROM prompt_keywords_fts WHERE prompt_keywords_fts MATCH ?))`, which is the exact shape `search_library_prompts_page` (`:3510-3519`) already uses in this same file. That makes both the count and the page O(page) and removes the parameter cliff.
- Size: M · ADR: no · Confidence: verified
- Pinning test: none — `rg -n "too many SQL variables" Tests` → no hits; no test states the `IN`-list construction as a requirement.
- Already covered: none
### P1 [D2] — `get_cli_setting("console","turn_file_cards")` runs once per turn-file-card row build at ~5 ms a call; 20 such rows turn a transcript row-build pass from 1.2 ms into 102 ms  ·  _slice: W-console-2_
- **Where:** `tldw_chatbook/Widgets/Console/console_transcript.py:7345` (`_build_message_widget`) and `:7700` (`_update_row_widget`); import at `:94`.
- **Evidence:**
  - `probe_gcs.py` (200 warm iterations) → `get_cli_setting warm ms median=4.8012 mean=4.8217 max=5.9183`
  - `probe_tfc.py` — a `ConsoleTranscript` with 20 USER + 20 TOOL rows, `_message_widgets()`, median of 8 passes:
    `with_card=False: 1.2 ms (81 widgets)` vs `with_card=True: 107.9 ms (81 widgets, 20 ConsoleTurnFileCard)`
  - `probe_tfc2.py` — same transcript, stubbing only the module-level name:
    `real get_cli_setting: 101.9 ms` → `get_cli_setting stubbed: 1.2 ms (calls per pass=20)`.
    **99 % of the pass is config I/O, exactly one read per card row.**
- **Why it matters:** `compose()` (`:3262-3270`) and `_reconcile_rows` build every row through this path, so a resumed agent session with N file-change turns pays N × ~5 ms on open. `:7700` additionally sits on the *selection* path — the card row's signature folds in `selected` (its own comment, `:7686-7695`) — so moving j/k selection onto or off a card row pays another ~5 ms per keypress.
- **Recommended correction:** the same class already has the cheap shape two methods away: `_assistant_markdown_enabled()` (`:4862`) and `_prune_watermarks()` (`:4854`) read `getattr(self.app, "app_config", None)` and hand it to a pure resolver. Mirror that, or (smallest diff that preserves the pinning test's patch point) call the module-level `get_cli_setting` **once per row-build pass** and thread the bool through `_build_row_widget`/`_update_row_widget`.
- **Size:** S · **ADR:** no · **Confidence: verified**
- **Pinning test:** `Tests/UI/test_console_turn_file_card_factory.py::test_summary_row_stays_plain_marker_when_disabled` patches `transcript_mod.get_cli_setting` (the module-level name at `:94`), so a per-pass memo of that same name keeps it green. `Tests/Chat/test_console_diff_feedback_delivery.py::test_kill_switch_off_does_not_prevent_note_delivery` patches `config_module.get_cli_setting` and never reaches this path.
- **Already covered:** none
### P1 [D2] — `looks_attachable()` re-reads the attachment config once per path, on the event loop: 98 ms measured for a 20-file clipboard paste  ·  _slice: CHAT-rest-2_
- Where: `tldw_chatbook/Chat/console_paste_attach.py:162` — `looks_attachable` ends with `any(fnmatch(name, pattern) for pattern in _supported_patterns())`, and `_supported_patterns()` (`:28`) calls `attachment_core.attachment_filter_specs()` → `supported_image_formats()` on every invocation. Callers: `UI/Screens/chat_screen.py:20298` `[p for p in grab.paths if looks_attachable(p)]` (inside `_paste_console_clipboard_image`, an `async def` run via `run_worker(coroutine)` — the event loop, not a thread; only the `grab_clipboard_image` call above it is `asyncio.to_thread`-ed) and `UI/Screens/chat_screen.py:23105` inside `on_paste`, a Textual event handler.
- Evidence:
  ```
  cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$PWD $PY - <<'EOF'
  import os, time, statistics
  from tldw_chatbook.Chat.console_paste_attach import looks_attachable
  p = os.path.expanduser("~/pics/a.png"); looks_attachable(p)
  t=[time.perf_counter() for _ in ()]
  t=[]
  for _ in range(20):
      s=time.perf_counter(); looks_attachable(p); t.append((time.perf_counter()-s)*1000)
  print(statistics.median(t))
  ps=[p]*20; s=time.perf_counter(); [q for q in ps if looks_attachable(q)]; print((time.perf_counter()-s)*1000)
  EOF
  ```
  →
  ```
  supported_image_formats    warm median=  5.232 ms
  attachment_filter_specs    warm median=  4.947 ms
  _supported_patterns        warm median=  4.832 ms
  looks_attachable ACCEPTED path warm median=  4.840 ms
  20 clipboard paths (chat_screen.py:20298 shape) =    98.1 ms
  ```
  (A *rejected* path costs 0.053 ms — `is_safe_path` short-circuits before the config read, so the cost is paid only by paths that actually attach, i.e. exactly the case the user cares about.)
- Why it matters: copying 20 images in Finder and pasting stalls the Console UI for ~98 ms on the event loop; every single drag-drop paste stalls it ~5 ms inside `on_paste`. The work is re-deriving a constant tuple of glob patterns from config.
- Recommended correction: memoize `_supported_patterns` (`@functools.lru_cache(maxsize=1)`) or, better, hoist it: `looks_attachable` should take the pattern tuple, and `chat_screen.py:20298` should compute it once outside the comprehension. The underlying `attachment_core` readers are the sibling finding (3 unmemoized readers at ~7 ms); this is the per-path amplification of it and is fixable in this file alone.
- Size: S · ADR: no · Confidence: verified (measured)
- Pinning test: none. `Tests/` has no timing assertion on this path.
- Already covered: none. Depends on the same root as the sibling slice's `attachment_core` config-read finding.
### P1 [D2] — a `workspace_records` SELECT runs on the UI thread 5×/s for the whole of every streaming run, for a title `ensure_session` throws away  ·  _slice: UIM-console_
- Where: `UI/Console_Modules/session.py:4419-4426` (`_sync_console_session_draft`); same shape at `session.py:3688` (`_replace_active_console_session_settings`). Chain: `UI/Screens/chat_screen.py:18797 set_interval(0.2, _poll_transcript)` → `:18746 await self._sync_native_console_chat_ui()` → `:18547 self._session._sync_console_session_draft()` → `self._workspace_initial_session_title(...)` → (`wiring.py:1535` lambda) `workspace.py:5248 _console_initial_session_title_for_workspace` → `workspace.py:5232 registry_service.get_workspace()` → `Workspaces/registry_service.py` sync `SELECT ... FROM workspace_records`.
- The value is discarded: `Chat/console_chat_store.py:2122-2124` — `ensure_session` returns `self._sessions[self.active_session_id]` untouched when a session is active; `title` is consumed only by `create_session`.
- Evidence: `<SCRATCH>/repro/uim_console_probe3.py` →
  ```
  'workspace-default': median=0.1us -> per-second at 5Hz = 0.00ms
  'global':            median=0.1us -> per-second at 5Hz = 0.00ms
  'w-1':               median=2296.2us -> per-second at 5Hz = 11.48ms
  ```
  `uim_console_probe2.py` on the same machine: `title-for-workspace(non-default): median=2.18ms p95=2.36ms max=2.54ms` over 200 calls against a fresh `WorkspaceDB`. Run-1 probe measured the underlying `LocalWorkspaceRegistryService.get_workspace` at median 3.63 ms / p95 5.47 ms and showed `_console_initial_session_title_for_workspace` issuing one `get_workspace` per call with no memo (10 calls → 10 queries).
- Scope: only when the active workspace is a real workspace — `workspace-default` and the `global` sentinel return early at `workspace.py:5252-5258` and cost nothing. `_poll_transcript` runs only while a run is live (`chat_screen.py:18720 _console_transcript_poll_needed`), i.e. exactly while the UI must stay responsive for streaming.
- Why it matters: ~11 ms of main-thread SQLite per second of streaming, entirely wasted, on the loop that paints the transcript.
- Recommended correction: the fix already exists three methods up — `session.py:3534-3546` carries the TASK-26839 comment and computes the title only `if creating_blank_session`. `_sync_console_session_draft` already computes `creating_blank_session` at `session.py:4418`; apply the same conditional there and at `session.py:3688`.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none found asserting the unconditional call.
- Already covered: none (task-26839 fixed the sibling call site only).
### P1 [D2] — every Permissions-matrix Space press blocks the event loop for ~56 ms in `tool_gate_breadcrumb()`, and the call site's comment claims it is "cheap"  ·  _slice: UIM-nav-mcp-persona_
- Where: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py:2657-2662` (inside `_sync_permissions_mode`, comment: *"computed fresh every pass (cheap -- the same settings-time-enumeration cost ... already pays every pass)"*) and `:1899` (`_empty_tools_diagnosis`). `_sync_permissions_mode()` has NINE standalone callers besides the full `_sync_children()` pass: `:2139`, `:2152`, `:2165` (`select_tool_policy_profile`), `:3265`, `:3436` (`on_mcp_permissions_mode_state_cycle_requested` — the Space press), `:3494` (kill-switch toggle), `:4830`, `:4895`, `:4960` (re-allow / remove-arg-rule / revoke-approval).
- Evidence:
  ```
  cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$PWD $PY - <<'EOF'
  import time, statistics
  from tldw_chatbook.Agents.builtin_tool_gate import tool_gate_breadcrumb, all_tool_gates
  for _ in range(3): tool_gate_breadcrumb()
  ts=[]
  for _ in range(10):
      t=time.perf_counter(); tool_gate_breadcrumb(); ts.append((time.perf_counter()-t)*1000)
  print("tool_gate_breadcrumb ms: median %.1f min %.1f max %.1f" % (statistics.median(ts),min(ts),max(ts)))
  ts=[]
  for _ in range(10):
      t=time.perf_counter(); all_tool_gates(); ts.append((time.perf_counter()-t)*1000)
  print("all_tool_gates ms: median %.1f min %.1f max %.1f" % (statistics.median(ts),min(ts),max(ts)))
  EOF
  ```
  ->
  ```
  tool_gate_breadcrumb ms: median 55.7  min 54.5  max 58.6
  all_tool_gates ms: median 55.2  min 54.4  max 60.9
  ```
  An instrumented count shows both make **11 `get_cli_setting` calls** (one per gate row), each ~4.8 ms (see the `_collect_snapshots` finding below for where that 4.8 ms goes: 482 `posix.open` syscalls per config read through `Backup_Recovery.storage_admission`).
  `all_tool_gates()` is also re-run on every Servers-mode detail repaint: `mcp_servers_mode.py:1322` (`_tool_gate_widgets`), reached from `show_detail()` -> `_rebuild_toggle_groups()` -> `_rebuild_tool_gate_buttons()` on EVERY `_sync_children()` pass whenever the selected row is the built-in — which is the fresh-install default (`_preselect_single_problem_on_load` lands on the lone built-in row, `mcp_workbench.py:1130-1136`).
  Aggregate for one `_sync_children()` pass on a fresh install with the built-in selected, measured piecewise: `_collect_snapshots` 20.6 ms + `_local_tools_config_values` 9.6 ms + `resolve_server_workspace_root` 4.7 ms + `tool_gate_breadcrumb` 55.7 ms + `all_tool_gates` 55.2 ms = **~146 ms of config reads alone**, on the loop, per rail click / lifecycle completion / gate toggle.
- Why it matters: Space-cycling a permission in the matrix is the mode's primary gesture (its own docstring says so) and each press stalls the Textual message pump for ~56 ms of config re-reads before the matrix repaints — held-Space repeat rate is capped at ~18/s by this alone, and it is paid on top of the store load and the server round-trips those handlers already make.
- Recommended correction: resolve the gate set ONCE per `_sync_permissions_mode()` pass and thread it, as this method already does for `effective`/`policy_inventory`/`_last_cascade`; better, memoize `all_tool_gates()` in `Agents/builtin_tool_gate.py` behind the same write path that already invalidates it (`_save_tool_gate` is the only writer in-process). Fix the comment either way — "cheap" is measurably false.
- Size: M · ADR: no · Confidence: verified
- Pinning test: none found asserting per-pass recomputation
- Already covered: none
### P1 [D2] — opening Settings ▸ Speech & TTS spends 65 ms re-reading cached config on the UI thread  ·  _slice: W-persona-settings-chat_
- Where: `Widgets/Settings_Widgets/speech_tts_settings_panel.py:878-882` (`SpeechTTSSettingsPanel.__init__`, the `restored is None` branch) → `_read_realtime_settings_draft()` (:291-320) and `_read_pipeline_voice_settings_draft()` (:322-341). The panel is constructed from a `yield` inside the settings screen's category render: `UI/Screens/settings_screen.py:20063`.
- Evidence:
```
cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$PWD $PY - <<'PY'
import time
from tldw_chatbook.Widgets.Settings_Widgets import speech_tts_settings_panel as p
from tldw_chatbook.config import get_cli_setting
get_cli_setting("dictation", "response_eagerness_ms", 0)      # warm the cache
t=time.perf_counter(); p._read_realtime_settings_draft();  print(f"realtime {(time.perf_counter()-t)*1000:.2f} ms")
t=time.perf_counter(); p._read_pipeline_voice_settings_draft(); print(f"pipeline {(time.perf_counter()-t)*1000:.2f} ms")
t=time.perf_counter()
for _ in range(20): get_cli_setting("dictation","response_eagerness_ms",0)
print(f"per get_cli_setting {(time.perf_counter()-t)*1000/20:.3f} ms")
PY
```
  → `realtime draft (warm): 54.64 ms` · `pipeline draft (warm): 10.33 ms` · `per get_cli_setting: 5.427 ms` (≈12 reads × 5.4 ms). Corroborates the sibling's "config reads are not free even warm" measurement independently.
- Why it matters: 65 ms of blocking work on the UI thread every time the Speech & TTS category is opened with no draft snapshot, purely to re-read a config the loader has already cached. The panel needs one settings snapshot, not twelve point lookups.
- Recommended correction: read the config once (`load_settings()` / `get_runtime_config_snapshot()` — the settings screen already holds one at `settings_screen.py:20054`) and pass the mapping into both draft readers, the way `_read_pipeline_voice_settings_draft` already does internally (:337 builds `config = {"dictation": section}` and hands it to `response_eagerness_ms(config)` — the helpers already accept a mapping).
- Size: S · ADR: no · Confidence: verified (measurement above)
- Pinning test: none.
- Already covered: none (task-1378/task-31202 cover `settings_screen.py` size, not this panel's config reads).
### P1 [D3] — Toggling one Notes-import review row drops keyboard focus entirely, and costs a full-page recompose measured at 120-175 ms  ·  _slice: W-library_
- Where: `library_note_import_canvas.py:597` (`sync_state` → `refresh(recompose=True)`) and `:664-667` (`_after_recompose`, no focus restore).
- Evidence: `<SCRATCH>/probe_ingest_review_focus.py` → `FOCUS BEFORE: note-import-action-item-6-skip` / `FOCUS AFTER: None` / `same-id button re-rendered: True`.
  Cost, `<SCRATCH>/probe_ingest_review_recompose_cost.py` (`app.run_test(size=(140,50))`, one item's action flipped per sample): 1 row/6 buttons → 69-94 ms; **25 rows/78 buttons → 119-174 ms** (25 is the real ceiling, `MAX_IMPORT_REVIEW_PAGE_SIZE`, `Library/library_note_import_state.py:39`); 50 rows → 250-314 ms. The ~70-90 ms floor is harness pump overhead, so the marginal row cost is ~50-80 ms per press at the ceiling.
  Shipped path: `ItemActionRequested` → `library_notes_controller.py:5534 set_item_action` → `_publish_library_snapshot` → `_sync_library_canvas(self, "notes")` (`:5036`) with **no** `then=`; the focus helper `_focus_library_note_import_control` (`:4971`) is wired at exactly one unrelated site (`:5071`).
- Why it matters: a keyboard user settling a 25-row review re-tabs from nowhere after every press, and waits >100 ms (the repo's own worker threshold) on the UI thread for it.
- Recommended correction: record `self.app.focused.id` before the recompose and re-focus it in `_after_recompose` (the canvas already carries `PostRecomposeCallback`; ids are stable — the probe shows the same-id button is re-rendered). Structurally: give the review rows their own render-from-state child, as the sibling ingest canvas already did (`LibraryIngestQueuePanel`, task-2042).
- Size: S (focus) / M (granularity) · ADR: no · Confidence: verified (harness-measured, not terminal-measured — stated)
- Pinning test: `Tests/UI/test_library_notes_w4_import_keyboard.py:321` goes review → Enter without ever toggling a row.
- Already covered: none
### P1 [D4a+D1] — Every Library list row title is escaped with `rich.markup.escape`, which does not cover `[TODO]`-shaped brackets; the repo's own full escaper is two packages away  ·  _slice: W-library_
- Where: the shared escaper `library_rail.py:367-377 _visible_row_title` = `escape_markup(_truncate_row_title(...))`, consumed by `library_media_canvas.py:204` (every media row), `library_conversations_canvas.py:317`, `library_media_trash_canvas.py:437`, and `Widgets/Home/home_rail.py:127,170`. Same narrow escaper at 27 further `escape_markup(...)` sites in `Widgets/Library/` that feed markup-ON surfaces — notably `library_notes_canvas.py:2132, 2324, 2373-2374, 2662` (note titles, folder labels), `library_skills_canvas.py:1284, 1318`, `library_prompts_canvas.py:697, 1007, 1008, 1023`, `library_entry_canvases.py:164`, `library_rail.py:1011, 1046`, `library_search_rag_panel.py:1387`.
- Evidence: `rich.markup.escape`'s pattern is `(\\*)(\[[a-z#/@][^[]*?])` — it only escapes a `[` followed by **lowercase**, `#`, `/` or `@`. Textual 8's tokenizer opens a tag on any unescaped `[`. Measured through the real helper:
  ```
  '[TODO] Q3 plan'  escaped='[TODO] Q3 plan'   rendered='▸  Q3 plan'
  '[IMPORTANT]'     escaped='[IMPORTANT]'      rendered='▸ '
  'Meeting [Call]'  escaped='Meeting [Call]'   rendered='▸ Meeting '
  '[ WIP ] thing'   escaped='[ WIP ] thing'    rendered='▸  thing'
  '[draft] notes'   escaped='\\[draft] notes'  rendered='▸ [draft] notes'   (lowercase IS covered)
  ```
  Widget level, real canvas (`<SCRATCH>/probe_media_title_markup.py`): `TITLE: '[TODO] Q3 plan'` → `RENDERED: '▸  Q3 plan\n    document · today'` → `AssertionError: row label lost the bracketed run`.
  The repo has already diagnosed this exact class and shipped the fix: `Library/library_rag_state.py:505-530` (`_escape_all_brackets`), whose own comment names the `[TODO]` shape — but it is applied only to `display_snippet` (`:1546`) and `library_rag_answer_display_text` (`:601`).
- Why it matters: `[TODO] …`, `[WIP] …`, `[2024-Q3] …` are ordinary titling conventions. A media item, conversation, note, prompt, skill or Home-rail row titled that way loses the bracketed run from its row, and a title that is only a bracketed word renders as a blank row — the user cannot see which item they are selecting.
- Recommended correction: promote `_escape_all_brackets` out of `Library/library_rag_state.py` into `tldw_chatbook/Utils/` and point `_visible_row_title` plus the 27 `escape_markup(...)` display sites at it. Canonical home: `Utils/` (a Library-private helper already has three packages of consumers). Leave `escape_markup` only where a Rich-tag-shaped escape is genuinely wanted.
- Size: M · ADR: no · Confidence: verified (helper-level and widget-level)
- Pinning test: none pins the broken shape. `Tests/Library/test_library_rag_state.py:2003 test_query_is_markup_escaped` and `:2032` only assert `[bold]…[/]` (tag-shaped, which `escape_markup` does cover); `Tests/UI/test_library_ingest_template_picker.py:143 test_picker_escapes_markup_in_labels` likewise asserts `"chapter [red] bold"`. Every existing test picks a lowercase tag name, so the gap is untested rather than decided.
- Already covered: none
- **Re-verified by the orchestrator:** `rich.markup.RE_TAGS` is `((\\*)\[([a-z#/@][^[]*?)])`. Measured through `Content.from_markup`: `'[TODO] Q3 plan'` → `' Q3 plan'`, `'[WIP] Draft plan'` → `' Draft plan'`, `'[IMPORTANT]'` → `''` (blank), while `'[draft] x'` survives — which is why every existing pinning test, all of which use lowercase tags, passes.
### P1 [D4a] — The repo already diagnosed the `rich.markup.escape` gap in writing (`console_composer_bar.py:5983-5993`) and fixed it with `Content` in ONE place; the composer's own attachment indicator in the same file still uses the broken escape  ·  _slice: W-console-1_
- Where: fix + diagnosis at `console_composer_bar.py:5971-6011` (`set_voice_status`'s docstring) and `:6039`/`:6069`/`:6139` (`chip.update(Content(...))`); unfixed sibling at `console_composer_bar.py:5889` — `indicator.update(escape(resolve_glyph_text(f"📎 {normalized}")))`
- Evidence: the docstring states the mechanism verbatim — *"a `Static` parses strings as Textual markup, and `rich.markup.escape` (which used to guard this) only escapes tags opening with `[a-z#/@]`. Whisper's own tokens are uppercase, so `[BLANK_AUDIO]` and `[Music]` survived escaping untouched and were then stripped at paint time … `Content` carries plain text with no markup semantics at all, so it fixes the swallowing and the opposite failure (`[/tmp/x]` raising `MarkupError`) in one move."* Reproduced on the indicator's exact expression:
  ```
  $PY -c "from rich.markup import escape; from textual.content import Content
  print(repr(str(Content.from_markup(escape('\U0001F4CE [WIP]report.pdf · 4 KB')))))"
  '📎 report.pdf · 4 KB'          # lowercase '[budget]' IS escaped and survives; uppercase is not
  ```
  The label is a real filename: `UI/Screens/chat_screen.py:22219` passes `pendings[0].label` ("photo.png · 240 KB") straight through.
- Why it matters: this is the same defect the file's own docstring says was fixed, still live 82 lines earlier in the same file, on a path where the user is being told which file they staged. It also means the "use `escape_markup`" pattern the rest of `Widgets/Console/` follows is documented-here as insufficient, yet 14 files in `Widgets/Console/` still follow it — 8 in this slice plus `console_session_switcher_modal`, `console_session_surface`, `console_settings_modal`, `console_setup_modal`, `console_style_picker_modal`, `console_workspace_context` in the other half (see the first finding).
- Recommended correction: `indicator.update(Content(resolve_glyph_text(f"📎 {normalized}")))` and drop the now-unused `from rich.markup import escape` at `:26` once the other `escape(...)` call sites in the file are checked. Canonical statement of the rule belongs next to the existing `set_voice_status` docstring or in `backlog/docs/lessons-textual.md` so the rest of the package stops reaching for `escape_markup`.
- Size: S · ADR: no · Confidence: verified
- Pinning test: three exist and none can catch it — `Tests/UI/test_console_composer_collapse.py:1334` (`assert "photo.png · 12 B" in str(attachment.renderable)`), `Tests/UI/test_console_voice_chip.py:308` (`assert "2 files" in _painted(indicator)`), `Tests/UI/test_console_dictation_streaming.py:450` — every fixture label is bracket-free, so the substring assertion passes either way
- Already covered: none
- **Re-verified by the orchestrator:** `rich.markup.RE_TAGS` is `((\\*)\[([a-z#/@][^[]*?)])`. Measured through `Content.from_markup`: `'[TODO] Q3 plan'` → `' Q3 plan'`, `'[WIP] Draft plan'` → `' Draft plan'`, `'[IMPORTANT]'` → `''` (blank), while `'[draft] x'` survives — which is why every existing pinning test, all of which use lowercase tags, passes.
### P1 [D4a] — `server_lifecycle.py` spawns every local LLM server without a process group and stops it with a bare `Popen.terminate()`, so a forked worker child survives and the app still reports "stopped"  ·  _slice: EVENTS_

- Where: `tldw_chatbook/Event_Handlers/LLM_Management_Events/server_lifecycle.py:482-501` (`terminate_process_bounded`) and `:559-570` (the `Popen` kwargs in `run_server_subprocess` — no `start_new_session`). Consumers of the stop path: `llm_management_events.py:546,680` (llamafile, llama.cpp), `llm_management_events_ollama.py:256`, `_onnx.py:188`, `_mlx_lm.py:143`, `_vllm.py:239,294`, `UI/Screens/llm_screen.py:3748,4150`.
- Evidence (mechanism reproduced against the shipped function):
```
cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY - <<'EOF'
import subprocess, sys, os, time
from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import terminate_process_bounded
launcher = ("import subprocess,sys,time;"
            "c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']);"
            "print(c.pid,flush=True);time.sleep(120)")
p = subprocess.Popen([sys.executable,"-c",launcher], stdout=subprocess.PIPE, text=True)
g = int(p.stdout.readline().strip())
print("child",p.pid,"grandchild",g)
print("terminate_process_bounded ->", terminate_process_bounded(p, timeout=2.0))
time.sleep(0.5)
def alive(pid):
    try: os.kill(pid,0); return True
    except OSError: return False
print("child alive:",alive(p.pid)," GRANDCHILD alive:",alive(g)); os.kill(g,9)
EOF
```
→ `child 48697 grandchild 48698` / `terminate_process_bounded -> True` / `child alive after stop: False  GRANDCHILD alive after stop: True`
- The comparison that makes this a D4a, not a design choice: **nine other subprocess sites in this repo already do the containment** — `Web_Server/artifact_share.py:150,217,226`, `Tools/git_tool_impls.py:254,316`, `Tools/workspace_tool_executor.py:180`, `Chat/console_trace_regex_worker.py:209`, `Chat/console_voice_process.py:361`, `Skills_Interop/skill_script_runner.py:425,305`, `Notes/git_process_containment.py:200,250,1172`, `Agents/run_hooks.py:361,340`, `STT/executor_process_tree.py:395,402` — all `start_new_session=True` + `os.killpg`. The one spawner of *long-running servers* is the one that skips it. `rg -n "start_new_session|killpg" --glob '*.py' tldw_chatbook/` returns zero hits inside `Event_Handlers/`.
- Why it matters: `terminate_process_bounded` returns `True`, so `stop_server_process` clears `app.<provider>_server_process`, calls `_notify_snapshot_stopped`, and notifies `"<label> stopped."` — while a forked engine/runner child still holds the listening port. The next Start then fails on a port already in use with no handle left to kill, and the app's own state says nothing is running. The launchers that fork are the shipped ones: vLLM is launched as the venv console script (`UI/LLM_Management/vllm_setup.py:714  candidate = python_path.with_name("vllm")`), which forks engine-core workers, and `ollama serve` forks a runner per loaded model.
- Recommended correction: in `run_server_subprocess`, add `start_new_session=(os.name == "posix")` (Windows: `creationflags=CREATE_NEW_PROCESS_GROUP`, the shape already written in `Notes/git_process_containment.py:840,1172`); in `terminate_process_bounded`, escalate to `os.killpg(os.getpgid(process.pid), SIGTERM)` then `SIGKILL`, exactly as `Web_Server/artifact_share.py:217-226` does. Canonical home: keep it in `server_lifecycle.py` (it is already the single lifecycle primitive for all six providers) and copy the artifact_share shape rather than inventing a third.
- Size: M · ADR: no (`rg -in "server|subprocess|process group|lifecycle" <SCRATCH>/adr_list.txt` → no ADR covers local-server subprocess containment) · Confidence: verified (mechanism); the per-launcher fork shape is the inferred half — see UNVERIFIED.
- Pinning test: none. `Tests/LLM_Management/test_server_lifecycle_resources.py` has `test_successful_stop_closes_resource_once_after_process_death` and `test_stop_failure_notification_is_actionable_without_process_id`; neither asserts anything about children, and `rg -n start_new_session Tests/LLM_Management/` is empty. Nothing pins the current behaviour.
- Already covered: none.
### P1 [D4b] — Console evidence/inspector rows HTML-entity-escape text destined for a terminal surface, so a Library title "R&D Report" reaches the user as "R&amp;D Report"; the Library surface fixed exactly this bug and Console did not  ·  _slice: CHAT-rest-1_
- Where: `tldw_chatbook/Chat/console_display_state.py:8` (`from html import escape as html_escape`) and `:93-95 _safe_display_text`. 18 call sites: `:496, 507, 508, 519, 528, 829, 849, 852, 854, 855, 901, 902, 903, 1158, 1159, 1170, 1171`. Already-fixed sibling copy: `tldw_chatbook/Library/library_rag_state.py:369-389 _sanitize_display_text` + `:392 _unescape_and_rescrub`.
- Evidence: `cd $WT && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY - <<EOF` importing both helpers →
  ```
  'R&D Report'      console='R&amp;D Report'              library='R&D Report'
  'Alice & Bob'     console='Alice &amp; Bob'             library='Alice & Bob'
  '<b>Release</b>'  console='&lt;b&gt;Release&lt;/b&gt;'  library='<b>Release</b>'
  'R&amp;D Report'  console='R&amp;amp;D Report'          library='R&D Report'   <-- Console double-escapes
  ```
- Why it matters: these rows render in a Textual/Rich `Static` with markup OFF — stated by `ConsoleStagedEvidenceRow`'s own docstring (`console_display_state.py:1096-1102`) and by `Tests/UI/test_console_staged_evidence_strip.py:156-158`. Rich never decodes HTML entities, so the escape protects nothing on this surface and corrupts the displayed string; a title that arrives already entity-encoded is escaped twice. `library_rag_state.py:392`'s docstring records the identical bug found in **live UAT on 2026-08-03 (task-15 finding 1)** — "a Note containing 'Alice & Bob' rendered as 'Alice &amp; Bob' in the evidence card" — and fixed there. The same Library note staged into Console still shows the pre-fix string.
- Recommended correction: replace `_safe_display_text`'s `html_escape` with the Library tail (`html.unescape` → re-run the dangerous-pattern scrubber → `escape_markup`), promoted out of `Library/library_rag_state.py` into a shared `Utils/` display-text helper as the canonical home; `escape_markup` is the only escape a markup-off Rich surface needs. Heed `library_rag_state.py:392`'s own warning: the un-escape must be followed by re-running the scrubber, so lift the whole tail, not just the `html.unescape`.
- Size: M · ADR: no · Confidence: verified (string drift reproduced; the on-screen render is inferred from the Library live-UAT precedent on the identical widget class — see UNVERIFIED)
- Pinning test: `Tests/UI/test_console_staged_evidence_strip.py::test_strip_state_escapes_untrusted_library_titles` asserts `row.title == "[bold]pwn[/bold] &lt;script&gt;"` — **it states the current behaviour as a requirement**, so this is a standing decision the later task-15 ruling contradicts; fixing Console means re-ruling that test, not just editing the helper. Same shape pinned at `Tests/UI/test_console_staged_context.py:369-370`, `Tests/UI/test_console_live_work_handoffs.py:1790`, `Tests/UI/test_console_internals_decomposition.py:4184-4185`.
- Already covered: none (task-15 covered the Library copy only)
### P2 [D1+D4a] — The File Notes "Use <folder>" button deletes bracketed runs from the folder name, and its idempotence guard can then never match  ·  _slice: W-library_
- Where: `library_file_notes_workspace.py:2964` builds `label=f"Use {_folder_label(sync_folder)}"` (a real directory name) and `:2839-2841` does `if str(button.label) != label: button.label = label`. The file never imports `escape_markup`; the sibling canvases in the same package do (`library_entry_canvases.py:8,164,180`, `library_conversations_canvas.py:333`, and `library_rail.py:377 _visible_row_title` = `escape_markup(_truncate_row_title(...))`).
- Evidence: Textual 8.2.8 parses markup in Button labels — `Content.from_text` calls `Content.from_markup` (`textual/content.py:258`):
  `$PY -c` probe → `Content.from_text('Use [draft] notes').plain` = `'Use  notes'`, `Content('Use [draft] notes').plain` = `'Use [draft] notes'`; `Button('[archive]')` → `''`; `Button('x [/] y')` → **raises `textual.markup.MarkupError: auto closing tag ('[/]') has nothing to close`**.
  Faithful micro-repro of the two lines above (`<SCRATCH>` one-liner): `pass 0: label attr='Use [archive]' rendered='Use ' needs_write_next_time=True` … repeated for passes 1 and 2.
- Why it matters: a notes folder named `[archive]`, `[wip]` or `Notes [old]` is offered as "Use " with the name gone; and because the read-back never equals the written string, every `_update_root_surface()` re-assigns the reactive label (a refresh per call, including on the 3 s structural-wait tick).
- Recommended correction: `label=f"Use {escape_markup(_folder_label(sync_folder))}"` — the helper is already the package convention (`rich.markup.escape`, used 4 files over). The general fix for the class is to pass `Content(...)` instead of `str` to `Button`, but the one-line escape matches what this package already does.
- Size: S · ADR: no · Confidence: verified (mechanism + micro-repro); the widget-level manifestation follows from the two quoted lines
- Pinning test: none found (`grep -rn "Use {" Tests` and `rg file-notes-use-sync-folder Tests` → only presence/visibility assertions)
- Already covered: none
### P2 [D1/D2] — `trajectory_import._read_document` reads a user-picked file with `Path.read_text()` and no size ceiling  ·  _slice: CHAT-rest-3_
- Where: `Chat/trajectory_import.py:89-91`
- Evidence: read only — no `st_size` check, no streaming; the whole file is materialised as one `str` and then again as a parsed object. `grep -n "MAX_\|st_size" trajectory_import.py` -> no hits (the only cap constants in the trio are `PREVIEW_MAX_CHARS` in `trajectory_export.py`).
- Why it matters: same picker, same "All Files" filter; a multi-GB pick is a `MemoryError` that escapes exactly as above. Compare `Chat/local_server_discovery.py:39` `MODEL_PROBE_RESPONSE_MAX_BYTES = 1MB`, which bounds a *remote* read on the same kind of JSON.
- Recommended correction: `path.stat().st_size` ceiling before the read, raising `TrajectoryImportError`; same constant style as `MODEL_PROBE_RESPONSE_MAX_BYTES`.
- Size: S · ADR: no · Confidence: inferred (command that would settle it: write a 3 GB `.json` under `<SCRATCH>` and call `load_imported_trace` on it)
- Pinning test: none
- Already covered: none
### P2 [D1/D4a] — `relative_age()` subtracts a possibly-naive datetime from an always-aware one; the hardened parser it should have used is in the same package  ·  _slice: CHAT-rest-2_
- Where: `tldw_chatbook/Chat/console_environment_state.py:245` (`seconds = max(0, int((now - then).total_seconds()))`), reached from `:784` `f"Merged {relative_age(pr.merged_at, now)}"`. `pr.merged_at` is produced by `Workspaces/environment_status.py:141 _parse_merged_at`, which is `datetime.fromisoformat(raw.replace("Z", "+00:00"))` with **no naive guard**. `now` is always aware (`chat_screen.py:8598` and `:8742` both pass `datetime.now(timezone.utc)`).
- Evidence (consequence verified, trigger inferred):
  ```
  cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$PWD $PY - <<'EOF'
  from datetime import datetime, timezone
  from tldw_chatbook.Workspaces.environment_status import _parse_merged_at
  from tldw_chatbook.Chat.console_environment_state import relative_age
  from tldw_chatbook.Chat.console_switcher_state import parse_console_switcher_instant
  naive = _parse_merged_at("2026-09-18T12:00:00")
  print(repr(naive), naive.tzinfo)
  try: relative_age(naive, datetime.now(timezone.utc))
  except TypeError as e: print("TypeError:", e)
  print(repr(parse_console_switcher_instant("2026-09-18T12:00:00")))
  EOF
  ```
  →
  ```
  _parse_merged_at('2026-09-18T12:00:00') -> datetime.datetime(2026, 9, 18, 12, 0) tzinfo: None
  relative_age(naive, aware) -> TypeError: can't subtract offset-naive and offset-aware datetimes
  hardened sibling parse_console_switcher_instant same input -> datetime.datetime(2026, 9, 18, 12, 0, tzinfo=datetime.timezone.utc)
  ```
  What is NOT verified: that `gh pr view --json mergedAt` can ever emit an offset-less value. GitHub emits `…Z` today, so this is a latent robustness gap, not a shipped crash — hence P2 and `inferred`.
- Why it matters: `project_environment_section` is a render path with no `try` at either call site (`chat_screen.py:8595` is a bare `return`, `:8741` a bare assignment), so the `TypeError` would take the Inspect rail down rather than degrade one row. This is the same shape the other slice found in `conversation_local_marks` (CHAT-rest-1 P2) — a parser that accepts a shape the consumer cannot use.
- Recommended correction: the hardened parser already exists **in this slice** and is explicitly public: `Chat/console_switcher_state.py:583 parse_console_switcher_instant` ("Public safe timestamp parser shared by bounded History adapters"), which does `if parsed.tzinfo is None: parsed = parsed.replace(tzinfo=UTC)` then `.astimezone(UTC)`. Make `_parse_merged_at` call it instead of rolling `fromisoformat`. Cheaper alternative if the cross-package import is unwanted: one `if parsed.tzinfo is None: return None` in `_parse_merged_at` — the row then reads "Merged" with no age instead of crashing.
- Size: S · ADR: no · Confidence: inferred (trigger) / verified (consequence)
- Pinning test: `Tests/Chat/test_console_environment_state.py:80-83` exercises `relative_age` only with two aware datetimes from the same `now`; it cannot go red on this.
- Already covered: none
### P2 [D1] — A failed home-server profile link is swallowed with no log line anywhere: `_run_personal_context_link` catches `Exception`, toasts, returns  ·  _slice: ENTRY-app_
- Where: `tldw_chatbook/app.py:13335-13340`
- Evidence: read — the `except Exception:` body is `self.notify("Profile linking needs attention. …", severity="error"); return` with no `logger` call; `grep -cE 'logger\.(warning|error|exception|opt)' tldw_chatbook/Personal_Context/link_service.py` → `0`, so the coordinator does not log for it either. The try covers keyring access (13211-13212, 13221), `sync_state_repository.get_personal_context_link_state` (13213), `bootstrap_personal_context_service` (13226), `coordinator.plan/apply/resume` (13257-13333) and two `push_screen_wait` dialogs.
- Why it matters: this is the only path that links Personal Context to the home server; when it fails the user sees "needs attention; retry from Settings" and the profile log records nothing — the exact undiagnosable-from-the-log failure TASK-32533 fixed for widget crashes.
- Recommended correction: `logger.opt(exception=True).warning("Personal Context link failed (stage=…)")` before the notify (type + stage only, no plan content — the toast already commits to "No profile content was shown"). S.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/UI/test_personal_context_link_app_flow.py` lines 174/242/315 assert the toast text as a requirement; none asserts the absence of a log line, so a log is compatible.
- Already covered: none
### P2 [D1] — A read failure inside `AgentService._persist` skips the terminal write and leaves the PRIMARY run row `running` until the next app launch  ·  _slice: AGENTS_
- Where: `tldw_chatbook/Agents/agent_service.py:4627-4642` (`exact_terminal_event_id`), called at 4642 and 4656; the terminal write it gates is `_set_terminal_status` at 4690.
- Evidence: `cd $WT && source $SCRATCH/env.sh && $PY $SCRATCH/agents_probe_persist.py` (stub DB whose `get_run` raises `sqlite3.OperationalError("database is locked")`) →
  ```
  RESULT: _persist RAISED OperationalError: database is locked
  db calls: [('insert_steps_at_indices', 1), ('get_run',)]
  terminal write attempted: False
  ```
  `exact_terminal_event_id` catches only `(KeyError, StopIteration, TypeError)` (4638). `AgentRunsDB.get_run` (DB/AgentRuns_DB.py:2552-2566) runs through `connection()` → `_held_connection` → `_core_access` (Backup_Recovery/participants.py:392-417), so the escaping classes are `sqlite3.OperationalError` (busy_timeout 5000 ms, AgentRuns_DB.py:313) and `RecoveryRequired(RuntimeError)` (`storage_locally_paused`, participants.py:417). Every sibling read in this module wraps the same call in `except Exception` (`_latest_durable_event_id` 4056, `_next_owner_seq` 4016, `_service_error_step` 4556, `_resolved_control_event_id` 4066). The bridge's `finally` after `service.run_turn(` (Chat/console_agent_bridge.py:7266-7300) writes no terminal status. A fleet CHILD is covered by `run_child`'s `finally` fallback (`_set_terminal_status`, agent_service.py:5747-5753); the primary has no equivalent. `AgentRunsDB.reconcile_orphaned_runs` (DB/AgentRuns_DB.py:2340) flips stranded `running` rows to `error`, but only once per DB open (docstring: "On open ... tracked via `_swept_paths`").
- Why it matters: for the rest of the session the rail shows a run that never finishes, `steer_primary`/`send_to_agent` and `list_running_run_ids` treat it as live, and the reply's `run_turn` call raises into the bridge after the answer was already produced — a transient lock during fleet concurrency (children write steps to the same file) is the realistic trigger.
- Recommended correction: widen 4638 to `except Exception` (matching 4056) so a failed read degrades to the existing diagnostic/recovery path instead of aborting `_persist`; optionally wrap the whole of `_persist`'s pre-terminal section so `_set_terminal_status` always runs.
- Size: S · ADR: no · Confidence: verified (code path, stub repro); the production trigger (real lock/pause at exactly that read) is inferred, hence P2 not P1.
- Pinning test: `Tests/Agents/test_agent_service.py::test_terminal_recovery_does_not_duplicate_lifecycle_transition` (happy path only; nothing pins a raising `get_run` — `rg -n "OperationalError|RecoveryRequired" Tests/Agents | rg -i "persist|get_run"` → 0).
- Already covered: none.
### P2 [D1] — A saved theme whose name contains a Rich markup tag can never be re-opened, and the failure is completely silent  ·  _slice: W-top_
- Where: `tldw_chatbook/Widgets/settings_theme_editor.py:297` (`theme_name = str(event.node.label)`), `:635`/`:641` (`_write_theme_file` duplicate-leaf check), `:865`/`:867` (`_delete_user_theme` node removal), and the silent half at `:339-361` (`load_user_theme`: `if theme_path.exists():` with **no** `else`).
- Evidence (runnable, verified):
  ```
  cd $WT && source env.sh && PYTHONPATH=$WT $PY -   # Tree.add_leaf -> str(node.label)
  'Solar [Flare]'   ACCEPTED by validate_filename   -> 'Solar [Flare]'  round-trips=True
  'My [bold]theme'  ACCEPTED by validate_filename   -> 'My theme'       round-trips=False
  'Solar[x]'        ACCEPTED by validate_filename   -> 'Solar'          round-trips=False
  ```
  Cause: `textual/widgets/_tree.py:858-859` `Tree.process_label` runs `Text.from_markup()` on every `str` label, so `add_leaf(theme_name)` stores markup-*parsed* text. `Utils/path_validation.validate_filename` (`:284-330`) rejects only separators / `..` / NUL — `[` is accepted at save time.
- Why it matters: save a theme as `my [x]theme` → the file `my [x]theme.toml` is written, the tree shows `my theme`, clicking that row calls `load_user_theme("my theme")`, `theme_path.exists()` is False and the method **returns with no notify and no log** — the theme silently cannot be reloaded. The same mismatch makes `_write_theme_file`'s `theme_exists` check (`:635-641`) never match, so each save appends a duplicate leaf, and `_delete_user_theme` (`:865-867`) leaves a stale row behind after deleting the file. Same defect class as the confirmed sibling P1 at `Chat/console_display_state.py:93`.
- Recommended correction: two independent one-liners. (1) Stop round-tripping identity through the label: store the name in `node.data` (already a dict-free slot carrying only `"user"`/`"catalog"` — make it `("user", theme_name)`) and read *that*, exactly as `Library/library_rag_state.py:392` documents for the same class. (2) Give `load_user_theme` an `else: self.app.notify(f"Theme file for '{theme_name}' not found", severity="warning")` — a missing file must not be a silent no-op regardless of cause.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none — `rg -n "process_label|node.label" Tests/` finds nothing for this widget.
- Already covered: none

---
### P2 [D1] — Agent worktrees are created under a fixed-name, default-mode directory in the shared system temp dir  ·  _slice: AGENTS_
- Where: `tldw_chatbook/Agents/agent_worktree.py:69-70` (`_worktrees_base` → `tempfile.gettempdir()/"tldw_agent_worktrees"`), `:126-129` (`mkdir(parents=True, exist_ok=True)` then `dest.parent.resolve(strict=True)`), consumed by `git worktree add` at 134-144.
- Evidence: read only — see UNVERIFIED. `rg -n "tldw_agent_worktrees|_worktrees_base" Tests tldw_chatbook` → every test monkeypatches `_worktrees_base` to `tmp_path`; `Tests/Agents/test_agent_worktree.py::test_checkout_created_under_symlinked_temp_base_can_be_recovered` pins that a symlinked base IS followed (stated as a requirement for app-owned aliases).
- Why it matters: on a multi-user host the first process to create `/tmp/tldw_agent_worktrees` owns it; `exist_ok=True` accepts a pre-existing directory or symlink of any owner/mode and `resolve(strict=True)` follows it, so every child checkout (user code, later merged back with `merge_agent_worktree`) lands wherever that name points. The per-run leaf name is uuid4 (unguessable) — the exposure is the parent, not the leaf. This is the classic fixed-name-in-tmp shape `tempfile.mkdtemp` exists to avoid.
- Recommended correction: root the base under the profile's data dir (`profile_paths.user_data_dir()/agent_worktrees`, app-owned, created 0o700) or `tempfile.mkdtemp(prefix="tldw-agent-")` per checkout; keep the `resolve(strict=True)` canonicalisation. The `agent_worktrees` DB records `child_path`, so no layout contract breaks.
- Size: M · ADR: no · Confidence: inferred
- Pinning test: `Tests/Agents/test_agent_worktree.py::test_checkout_created_under_symlinked_temp_base_can_be_recovered` (pins symlink-following as desired for owned aliases; does not pin the shared-tmp location).
- Already covered: none.
### P2 [D1] — Character-card import reads the user-picked file into memory with no byte cap; the two sibling importers in the same file cap at 10 MB  ·  _slice: UI-personas_
- Where: `personas_screen.py:13725-13726` (`_import_character_from_path`: `source.read_bytes()` unbounded, then the bytes go to `inspect_character_card_tts_attachment` and `import_character_card_with_outcome`). Siblings: `:14079` (world book, `stat().st_size > PERSONAS_WORLDBOOK_IMPORT_MAX_BYTES`) and `:14196` (dictionary, `PERSONAS_DICTIONARY_IMPORT_MAX_BYTES`) gate before reading; `:10520` avatar upload gates at 5 MB.
- Evidence: `grep -n "MAX_.*BYTES" Character_Chat/Character_Chat_Lib.py` → the lib caps decode **pixels** (`_MAX_CARD_DECODE_PIXELS = 50_000_000`, L148/1539) and history-export bytes (`_MAX_EXPORTED_HISTORY_FILE_BYTES`, L75) — nothing caps the card file's byte size; `grep -n "MAX\|len(" ccp_character_handler.py` → only field-length caps. The "Character Cards" picker filter (L581) accepts any `.png`/`.webp`/`.json`.
- Why it matters: a mispicked multi-GB `.png` is read whole into RAM on the `to_thread` worker before any validation runs; every other user-file import in this screen refuses first. Trust-boundary behaviour drifted across three siblings (also a D4b).
- Recommended correction: add `PERSONAS_CHARACTER_CARD_IMPORT_MAX_BYTES` (same 10 MB family) and the same `stat().st_size` gate before `read_bytes()` at 13725.
- Size: S · ADR: no · Confidence: inferred (memory consequence not reproduced)
- Pinning test: `Tests/UI/test_personas_workbench.py:3537` asserts `character_writes == [source.read_bytes()]` — pins the whole-file read as the contract; a size gate must precede it, and no test pins a card byte cap (grep `IMPORT_MAX_BYTES` in that file → only the avatar cap at :2507).
- Already covered: none (task-19558 "Security primitives … five seams" is Done and did not touch this read)
### P2 [D1] — The Notes location row is overwritten with a factually false "In the Library database only — no file on disk" before the first `apply_session_state`  ·  _slice: W-library_
- Where: `library_notes_canvas.py:1047` (`_note_location = ("", "")`), `:3649-3652`, fired by `@on(Resize) _note_chrome_follows_width:3667`. Compose renders the row from `presentation_state.location_path` (`:2757`) while `_restate_note_location` reads the `_note_location` cache that only `apply_session_state` fills — two sources for one row.
- Evidence: `<SCRATCH>/probe_notescanvas_width2.py` → `A) _note_location right after mount: ('', '')` / `row text after mount+layout: 'In the Library database only — no file on disk'`; `B) after the controller's first state apply: row text: 'In a synced folder · /…/atlas-follow-ups.md'`.
- Why it matters: the row asserts the note has no file on disk when it does.
- Recommended correction: seed the cache in `_compose_editor` next to `:2756`. One line.
- Size: S · ADR: no · Confidence: verified (mechanism); inferred (duration in production — the open-a-note path recomposes and closes the window; the exposed path is the screen's direct construction at `library_screen.py:14970` with `mode="editor"`). Settling command: `$PY -m pytest Tests/UI/test_library_notes_w5_ideas.py -q` with `await pilot.resize_terminal(...)` inserted between shell build and `_open_first_note`.
- Pinning test: none · Already covered: none
### P2 [D1] — The git panel's public render API raises `NoMatches` when detached, and the guarding convention is split between callee and caller  ·  _slice: W-library_
- Where: unguarded — `:1931 _fit_fixed_regions`, `:3073/:3086 set_last_action/clear_last_action`, `:3136/:3215 _clear_rows/_replace_rows`, `:3508 _settle_action_focus` (reads `self.screen`); guarded — `:1979` (`is_mounted`), `:2017` (`is_attached`). Callers guard at only 4 of ~25 sites (`library_file_notes_workspace.py:3623, 3692, 3715, 4125`; `:3559 mark_stale` is bare).
- Evidence: `<SCRATCH>/probe_gitpanel_unmounted.py` against an unmounted instance → `set_current_status: NoMatches`, `set_last_action: NoMatches`, `mark_stale: NoMatches`, `render_unavailable: NoMatches`, `render_untrusted: NoMatches`, `return_to_commit_list: NoMatches`, `return_to_push_list: NoMatches`; `clear_commit_availability: OK`, `clear_push_availability: OK`.
- Why it matters: two mounting predicates and two conventions across one API mean the safe set is undiscoverable; a late service callback landing during a Library recompose raises.
- Recommended correction: one rule — every public `render_*`/`set_*`/`clear_*`/`return_to_*` entry starts with `if not self.is_mounted: return`; then drop the four caller-side checks.
- Size: S · ADR: no · Confidence: verified (the raises), inferred (the live detach window)
- Pinning test: none
- Already covered: none
### P2 [D1] — The push-destination authorization dialog's Confirm bypasses the file's own double-dismiss guard  ·  _slice: W-library_
- Where: `library_file_notes_git_panel.py:4284-4286` uses `self.dismiss(True)` while every other exit in the file uses `dismiss_safe_once` (`:4180`, `:4271`, `:4275`).
- Evidence: `Widgets/modal_dismissal.py:270-282` — `dismiss_safe_once` refuses a second dismissal (`_safe_dismiss_committed`) and checks `host.app.screen is self`; Textual 8's `Screen.dismiss` has no idempotency guard (fires `_result_callbacks[-1](result)` then `pop_screen()` unconditionally). The authorize path therefore never sets `_safe_dismiss_committed` and skips the opener-focus restore at `:291-298`.
- Why it matters: this is the consent gate for first contact with a push destination; two `Button.Pressed` messages queued before the pop completes fire the authorization callback twice and pop the screen underneath.
- Recommended correction: `self.dismiss_safe_once(True)`.
- Size: S · ADR: no · Confidence: verified (the bypass), inferred (the double-press race)
- Pinning test: `Tests/UI/test_library_modal_dismissal.py::test_concrete_library_modal_public_positive_result_type` clicks ONCE and asserts one result — it pins the single-press result, not the guard.
- Already covered: none
### P2 [D1] — Three summarization stream generators yield the accumulated full text AGAIN after the deltas, doubling the summary for any consumer that joins chunks  ·  _slice: LLM_
- Where: `tldw_chatbook/LLM_Calls/Summarization_General_Lib.py:2290` (deepseek), `Local_Summarization_Lib.py:1993` (custom_openai), `:2252` (custom_openai_2). Siblings anthropic `:1217-1218`, groq `:1678-1679`, mistral `:2491-2492` carry the same line COMMENTED OUT.
- Evidence: `$SCRATCH/llm_repro_summarization.py` → `[deepseek stream] chunks=['Hello ', 'world', 'Hello world'] joined='Hello worldHello world'`
- Why it matters: doubled output whenever `analyze(streaming=True)` is used with these providers. Reachability today: every `analyze(` caller I could locate passes `streaming=False` or omits it (`Web_Scraping/WebSearch_APIs.py:1414`, `Local_Ingestion/Book_Ingestion_Lib.py:1354,1398,2029`, others default) — so LATENT, a P1 the moment a streaming caller appears.
- Recommended correction: delete the three trailing yields (S).
- Size: S · ADR: no · Confidence: verified (bug); reachability verified-latent
- Pinning test: none
- Already covered: none
### P2 [D1] — Two `run_worker` calls in the git panel keep `exit_on_error=True` with an unguarded `query_one` as the coroutine's first statement  ·  _slice: W-library_
- Where: `library_file_notes_git_panel.py:2847-2866` (`_render_commit_review_notes`, first line queries `#file-notes-git-commit-included-notes`) and `:3215-3247` (`_render_rows`, queries `#file-notes-git-rows` OUTSIDE its `try`).
- Evidence: `run_worker`'s default is `exit_on_error: bool = True` (`textual/dom.py`); `grep -rn "exit_on_error=False" tldw_chatbook | wc -l` → **268** sites opt out repo-wide, these two do not. `_render_rows` is incidentally protected by a synchronous `query_one` at `:3242` before scheduling; `_replace_commit_review_notes` has no such pre-check.
- Why it matters: an uncaught `NoMatches` in a Textual worker with `exit_on_error=True` terminates the application, and this panel is explicitly retained across parent remounts (`library_file_notes_workspace.py:1422-1425`).
- Recommended correction: `exit_on_error=False` on both plus `if not self.is_mounted: return` at the top of each coroutine.
- Size: S · ADR: no · Confidence: inferred (the detach window's reachability was not reproduced; the defaults and the missing guard are verified)
- Pinning test: none
- Already covered: none
### P2 [D1] — Two per-keystroke instant-persist writers can land an older `[permission_summary]` / `[model_catalog]` snapshot on disk after a newer one  ·  _slice: UI-settings_
- Where: `settings_screen.py:6558-6581` `_persist_permission_summary_section_values` (`@work(thread=True)`, no group/exclusive; generation token checked at `:6576` BEFORE the file lock) and `:14130-14149` `_persist_model_catalog_section_values` (`@work(thread=True)`, no group/exclusive, no token at all). Dispatchers: `:6520-6556` (bound to `Input.Changed` on the provider/model Inputs, `:27595-27603`) and `:14050-14128` (bound to `Input.Changed` on `#settings-model-catalog-stale-hours`, `:27585`; builds a FULL-section snapshot of all catalog checkboxes + the input).
- Evidence: `save_settings_to_cli_config` (config.py:8346) → `apply_settings_mutation_to_cli_config` → `_apply_literal_settings_transaction_locked` under `_CONFIG_FILE_LOCK` (config.py:6318) serialises writers but does not order them. Reproduced on the shipped worker bodies (`.__wrapped__`) with `save_settings_to_cli_config` replaced by a barrier + lock stub: `permission_summary: writers that reached the file write = ['NEW','OLD'] (both passed the generation guard); landing order = ['NEW','OLD']; ON DISK = OLD` and `model_catalog: typed 12 then 123; reached = [12, 123] (no guard at all); landing order = [123, 12]; ON DISK = 12`. A real write is ~64 ms (`save_settings_to_cli_config median=63.9ms max=73.3ms`), so two keystrokes inside that window overlap.
- Why it matters: config.toml ends up disagreeing with the widget the user is looking at (and with the cache the no-op guard compares against on the NEXT keystroke, so it self-heals only if the user types again).
- Recommended correction: check the token INSIDE the lock — `apply_settings_mutation_to_cli_config(section_values, locked_snapshot_precondition=lambda _s: generation == self._permission_summary_persist_generation)` (the seam already exposes that hook); add the same generation token to the model-catalog writer. `group=`+`exclusive=True` does NOT fix it (a started thread write cannot be cancelled).
- Size: S · ADR: no · Confidence: verified
- Pinning test: none (`grep -rln "_persist_permission_summary\|_persist_model_catalog" Tests` → nothing); `Tests/UI/test_settings_model_catalog_toggles.py` mocks `save_settings_to_cli_config` and asserts `call_args` (last call) — sequential, cannot see ordering.
- Already covered: none
### P2 [D1] — Two readers use the raw connection as a context manager, which commits any caller-owned transaction on exit  ·  _slice: DB-chacha_
- Where: `ChaChaNotes_DB.py:12752` (`get_conversation_context_summary`) and `:12802` (`get_conversation_console_project_context`): `with self.get_connection() as conn:`. `sqlite3.Connection.__exit__` calls `commit()`; every sibling reader uses `with self.transaction() as conn:` which borrows a live transaction and never commits it (`:24018-24025`, `:24159`).
- Evidence: `$PY <SCRATCH>/repro_conn_ctx_commit.py` → control (inner read via `transaction()`, caller aborts) `title after abort: t`; candidate (inner read via `get_conversation_context_summary`, caller aborts) `in_transaction after inner read: False` … `title after abort: LEAKED`. The caller's `rollback()` becomes a silent no-op.
- Why it matters: any future caller that reads the summary/project-context inside an outer `transaction(immediate=True)` gets its partial writes committed and its rollback discarded with no error. Today's two production callers (`Chat/console_chat_store.py:22119` in `_resolve_context_summary_on_resume`, `Chat/chat_persistence_service.py:2603`) are not inside a transaction (checked: no `transaction(`/`BEGIN` between the enclosing `def` at `:22083` and the call), so this is a trap, not a live loss.
- Recommended correction: replace both `with self.get_connection() as conn:` with `with self.transaction() as conn:` (the file's own idiom; 2 lines).
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/UI/test_console_resume_active_path.py:1388,1407`, `Tests/Chat/test_console_chat_store_project_instructions.py:54,69` assert returned values only, not transaction ownership.
- Already covered: none
### P2 [D1] — `BaseReranker._cache` is an unbounded dict with no size limit, TTL, or eviction, on a process-lifetime singleton  ·  _slice: RAG_
- Where: `tldw_chatbook/RAG_Search/reranker.py:188` (`self._cache = {} if config.cache_results else None`, and `RerankingConfig.cache_results` defaults to `True` at `:176`); written at `:568` (`PointwiseReranker`) and `:1282` (`CrossEncoderReranker`).
- Evidence: `grep -n "_cache" tldw_chatbook/RAG_Search/reranker.py` → 13 hits: one construction, two membership tests, two reads, two writes, and the `_get_cache_key`/`_cross_encoder_cache_key` helpers. **No eviction, no cap, no TTL, no `clear()` anywhere in the file.** The owning reranker is built once in `EnhancedRAGServiceV2._configure_reranker` (`enhanced_rag_service_v2.py:215-231`) and lives for the life of the shared service.
- Why it matters: every distinct (query, result-id-set) adds a `List[RerankingResult]` of up to `top_k_to_rerank` (default 20) entries, each of which can carry the model's `reasoning` text when `include_reasoning` is on. Nothing ever removes them. Contrast `SimpleRAGCache` in the same package, which has `max_size`, a TTL, *and* a memory cap for the same kind of payload. Reranking is opt-in (`SearchConfig.enable_reranking` defaults to `False` and a profile must carry a `reranking_config`), so this only bites users who turned it on — which is why it is P2 and not P1.
- Recommended correction: reuse the bounded cache that already exists rather than adding another one — an `OrderedDict` with `max_size` and `move_to_end`, or `functools.lru_cache`-style bounding, is a ~5-line change at `:188`/`:568`/`:1282`. (Fix `SimpleRAGCache`'s accounting first — see the P1 above — if it is to be reused directly.)
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none
### P2 [D1] — `ChromaVectorStore.search` converts every store failure into "no results"; the Library UI then renders the verified-empty-index message  ·  _slice: RAG_
- Where: `tldw_chatbook/RAG_Search/simplified/vector_store.py:568-571` (`except Exception as e: … return []`); same shape at `:1527` (stats, benign).
- Evidence: read; the `except` block logs at ERROR and `return []`. `search_with_citations` (`:602`) calls `search` and therefore inherits it. Downstream, `Library/library_local_rag_search_service.py:907` treats `not raw_results` plus a zero `get_collection_stats` count as "Index empty"; with a corrupted/locked Chroma directory `get_collection_stats` also fails → `_semantic_index_is_empty` returns False → the user gets the generic "0 results" outcome for what is actually a broken store.
- Why it matters: a broken vector store is indistinguishable from an empty corpus at the UI; the user re-indexes (or gives up) instead of seeing the real error.
- Recommended correction: let the exception propagate (the callers already have a `except Exception` → "Retrieval failed / Retry" recovery outcome at `library_rag_service.py:152`), or return a sentinel the caller can distinguish. Keep the `return []` only for the "collection does not exist yet" case.
- Size: M · ADR: no · Confidence: inferred (the swallow is verified by reading; the UI-message consequence is traced through code, not reproduced live)
- Pinning test: none
- Already covered: none
### P2 [D1] — `ConsoleCapturePolicyRepository` has three bare `except Exception: return …UNAVAILABLE` arms and the module imports no logger, so a failed write of the per-conversation capture/PII-redaction policy leaves no diagnostic at all  ·  _slice: CHAT-rest-1_
- Where: `Chat/console_capture_policy_repository.py:91-96` (read), `:157-158` (`replace_detail`), `:239-240` (`replace_privacy`). Imports at `:3-10` contain no `loguru`/`logging`.
- Evidence: `grep -n "logger\|import" tldw_chatbook/Chat/console_capture_policy_repository.py` → only `sqlite3`, `dataclasses`, `enum`, `CaptureDetail`, `CharactersRAGDB`. `grep -n -A3 "except Exception" …` → the three arms, none logging.
- Why it matters: `console_chat_controller.py:4935` and `:5081` read `status is CapturePolicyWriteStatus.UNAVAILABLE` and degrade the user to "session only". A genuine defect (a `TypeError` in `_upsert`, a schema drift, a bad `CaptureDetail`) is therefore indistinguishable from "the database was busy" and produces zero log output — the user's PII-redaction preference silently fails to persist with nothing to debug from. The write is inside `db.transaction()`, so no partial write occurs; what is lost is the diagnostic.
- Recommended correction: bind a module logger and `logger.opt(exception=True).warning(...)` in each arm (Console's own convention, e.g. `chat_persistence_service.py:118`), and narrow the catches to `sqlite3.Error` plus `(TypeError, ValueError)` so a programming error is not laundered into a storage verdict.
- Size: S · ADR: no · Confidence: verified (code + import census; not reproduced as a runtime failure)
- Pinning test: none
- Already covered: none
### P2 [D1] — `MCP/tools.py::chat_with_character` is a shipped, advertised MCP tool that can never answer: it calls a module-level stub that unconditionally raises, while the unified dispatcher it says does not exist (`Chat_Functions.chat_api_call`) is imported 20 lines away in `server.py`  ·  _slice: TOOLS-MCP_
- Where: `tldw_chatbook/MCP/tools.py:47-58` (`save_conversation_from_messages`/`chat_with_provider` stubs raise `NotImplementedError`), `:135-143` (the only call site), `:32-44` (comment claims "no unified dispatcher ... verified: neither name exists"); contrast `tldw_chatbook/MCP/server.py:588` (`from ..Chat.Chat_Functions import chat_api_call`) and `:619-628` (`chat_with_llm` uses it); the tool is registered at `server.py:640-659`
- Evidence: `rg -n 'def chat_with_provider|raise NotImplementedError|chat_api_call' tldw_chatbook/MCP/tools.py tldw_chatbook/MCP/server.py` → stub at tools.py:54/55; `chat_api_call` imported+used only in server.py:588/620. Unconditional path: with a key the call raises → `except Exception` → `{"error": ...}`; without a key `{"error": "No API key configured"}` — no branch produces a chat response.
- Why it matters: every external MCP client that lists tools sees `chat_with_character` and gets a "dead upstream reference" error on every call; `MCPTools` also constructs `SimplifiedRAGSearchService` etc. for a tool that cannot work, and the stale comment misleads the next maintainer into believing the dispatcher is gone.
- Recommended correction: route through `chat_api_call` exactly as `chat_with_llm` does (or drop the registration); delete the two stubs and the stale comment; M (persistence of the new conversation needs a real `ChaChaNotes_DB` call — `add_conversation`/`add_message` exist).
- Size: M · ADR: no · Confidence: verified (unconditional stub; not executed)
- Pinning test: `Tests/MCP/test_tools_resources_prompts_real_methods.py::test_chat_with_character_uses_the_declared_api_key_accessor` pins only the accessor (AST), not the outcome — nothing states "always errors" as a requirement
- Already covered: none
### P2 [D1] — `MediaDatabase(..., check_integrity_on_startup=True)` always fails: `__init__` calls `self.check_integrity()`, which the class never defines  ·  _slice: DB-media-base_
- Where: `tldw_chatbook/DB/Client_Media_DB_v2.py:1024-1030` (call), whole class (no `def check_integrity`; `rg -n "def check_integrity" Client_Media_DB_v2.py` → exit 1). The standalone `check_database_integrity(db_path)` at `:9103-9143` is the only integrity check that exists. Compare `base_db.py:851-874`, which `MediaDatabase` does not inherit (see the P3 D4a finding).
- Evidence: `$PY $HOME/repro/media_repro4.py` part E → `DatabaseError: Unexpected database initialization error: 'MediaDatabase' object has no attribute 'check_integrity' | cause=AttributeError(...)`.
- Why it matters: the constructor advertises the flag in its signature and docstring (`:973,982`); the first caller to pass it gets a fatal `DatabaseError` from the `except Exception` at `:1047-1056` on every open. No shipped caller passes it today (`rg "check_integrity_on_startup\s*=\s*True" tldw_chatbook` → 0).
- Recommended correction: implement `check_integrity()` as a 6-line instance method around `PRAGMA integrity_check` on `self.get_connection()` (or delete the flag and the dead branch). S.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none
### P2 [D1] — `NotesRecoveryDialog`'s 0.25 s interval never stops and its callback queries children unguarded  ·  _slice: W-library_
- Where: `Widgets/Library/notes_recovery_dialog.py:90` (`set_interval(0.25, self._check_current)`), `:92-101` (`_check_current` returns `False` intending to stop; `set_interval` ignores the return value), `:96-97` (`query_one` with no guard).
- Evidence: probe `<SCRATCH>/probe_recovery_timer.py` →
  mechanism: `MECHANISM: NoMatches No nodes match '#notes-recovery-approve' on NotesRecoveryDialog()` (children pruned, callback still queries);
  reachability: `RACE HITS: none in 40 close cycles` — Textual 8.2.8 stops the pump's timers in `MessagePump._close_messages` (`message_pump.py:533-535`) before the dismiss settles, so I could NOT reproduce a real crash. The remaining, certain defect is that once `current()` has gone False the interval keeps re-`update()`-ing `#notes-recovery-status` with the same string 4×/s for as long as the dialog stays open (a refresh per tick), because nothing stops the timer.
- Why it matters: a repaint every 250 ms behind a modal for an unbounded time, plus a query that is only safe by accident of Textual's teardown order.
- Recommended correction: keep the returned `Timer` and `.stop()` it in the two branches that return `False`; wrap the two `query_one` calls (or use `self.query(...)`) as every other timer body in this package does.
- Size: S · ADR: no · Confidence: verified (mechanism + no-stop), inferred (crash reachability — retired below)
- Pinning test: `Tests/Backup_Recovery/test_notes_recovery_controls.py` (`route`/`navigation` branches) asserts the "Selection changed" status appears; it does not assert the timer stops.
- Already covered: none
### P2 [D1] — `PromptsDatabase.update_keywords_for_prompt` executes DML on the legacy-isolation held connection with no transaction/commit; a bare caller's link changes are silently discarded at `close()` (same shape as DB-media-base's `MediaDatabase` P0, but no shipped caller reaches it bare)  ·  _slice: DB-rest_
- Where: `tldw_chatbook/DB/Prompts_DB.py:1760-1870` (method; comment at 1764 says "called within an existing transaction (e.g. from add_prompt)… don't start a new transaction here"); DML at `:1819` (DELETE links), `:1840` (INSERT OR IGNORE links), plus `_log_sync_event` INSERTs at `:1829/:1848`. Bare public wrapper: `tldw_chatbook/Prompt_Management/Prompts_Interop.py:230-233`. In-transaction callers (fine): `Prompts_DB.py:1679` (add_prompt) and `:2059` (update_prompt_by_id), both inside `with self.transaction()`.
- Evidence (reproduced, isolated env, temp-file DB):
  `PromptsDatabase(p).add_prompt(... keywords=["k1"]); db.update_keywords_for_prompt(pid, ["k2","k3"]); conn.in_transaction; db.close(); reopen; SELECT links` →
  `in_transaction after bare update_keywords_for_prompt: True` / `links seen on same conn: 2` / `links after close+reopen: ['k1']` / `keywords table rows: ['k1','k2','k3']` / `sync_log rows: 5`.
  I.e. the k2/k3 keyword ROWS persist (each `_add_keyword_full` runs its own `with self.transaction()`), but the link DELETE/INSERTs and their `sync_log` unlink/link events sit in an implicit legacy-isolation BEGIN that `close_connection()` (`Prompts_DB.py:504-517`, plain `conn.close()`) rolls back. Worse than pure loss: the store is left half-applied (new keyword rows, old membership).
  Reachability: `rg -n "update_keywords_for_prompt" tldw_chatbook Tests` → only `Prompts_Interop.py:230-233` (wrapper, itself uncalled anywhere in `tldw_chatbook/`) and 4 tests that read back on the SAME connection. No shipped path calls it bare today — hence P2, not P0.
- Why it matters: `Prompts_DB.transaction()` (`:659-693`) BORROWS when `conn.in_transaction` is already true (`in_outer=True` → no commit), so once a bare call has opened the implicit transaction, every subsequent `with db.transaction()` on that thread also stops committing — the whole thread's later prompt writes ride on the one uncommitted transaction until something calls `execute_query(commit=True)` or the app exits (then all lost). Any future UI use of the Interop wrapper (the obvious "edit keywords" affordance) turns this into a P0.
- Recommended correction: wrap the body in `with self.transaction() as conn:` (borrow semantics make the in-transaction callers a no-op change) — S. Also make `close_connection()` commit-or-warn if `conn.in_transaction` (the task-22224 flip is the real fix; this store is a documented EXCEPTION at `:404-413`).
- Size: S · ADR: no (task-22224 store-template rule; this file's exception is documented) · Confidence: verified
- Pinning test: `Tests/Prompts_DB/test_prompts_db_legacy.py::test_update_keywords_for_prompt_with_empty_list_removes_all` (:1133-1141) and `::test_update_keywords_for_prompt_is_idempotent` — both call it bare and read back on the same connection, so they PASS while the data is uncommitted; none states durability. No test closes+reopens.
- Already covered: none (task-22224 is the template rule; no task for this store's flip — the docstring says "its own task" but none exists in `backlog/tasks/` — `rg -l "Prompts_DB" backlog/tasks | rg 22224` → checked below in dispositions)
### P2 [D1] — `ReadFileTool` returns the ENTIRE file as `content` with no size cap, and both read families materialise the whole file before slicing  ·  _slice: TOOLS-MCP_
- Where: `tldw_chatbook/Tools/file_operation_tools.py:325-336` (`content = path.read_text(...)` → `{"content": content}`), `tldw_chatbook/Tools/local_tool_impls.py:345-356` (`text = target.read_text(...)` then `numbered[:MAX_READ_CHARS]`)
- Evidence: 150 MB single-line file: `ReadFileTool 150MB: 1.09s content_len=150000000 maxrss MB=396` (baseline 180); `local_tool_impls.read_file 150MB single-line: 2.81s returned=32782 chars maxrss MB=389`
- Why it matters: a model asking to read a build artefact or log in a bound workspace costs the app process (ReadFileTool runs in-process on the agent worker thread) file-size bytes of RSS and, for ReadFileTool, hands a 150 MB dict to `BuiltinToolProvider` before the runtime's 16,000-char result cap ever applies; the `fs_read` copy is bounded to the one-shot worker process but still pays 2.8 s per call.
- Recommended correction: `open(...).read(MAX_READ_BYTES + 1)` and stat-check before read in both families; S.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none found (`rg 'read_file.*(large|size|cap)' Tests/Tools` → no hits)
- Already covered: none
### P2 [D1] — `RichLogHandler.emit` silently drops DEBUG/INFO records emitted from worker threads; only WARNING+ leak to stderr (invisible under a TUI)  ·  _slice: ENTRY-config_
- Where: `tldw_chatbook/Logging_Config.py:171-183` (`asyncio.get_running_loop()` raises `RuntimeError` in any thread without a running loop; the except branch prints only `levelno >= WARNING`). Installed live: `UI/Screens/logs_screen.py:8,54` embeds `LogsWindow` (`#app-log-display` at `UI/Logs_Window.py:243`); `app.py:12170` constructs the handler; route `logs` in `UI/Navigation/screen_registry.py:188`.
- Evidence: 20-line probe constructing `RichLogHandler` with a fake widget, `start_processor()` on the loop thread, then `emit(INFO)` + `emit(WARNING)` from a plain `threading.Thread` →
  ```
  LOG_FALLBACK: 2026-09-17 19:15:12 [WARNING ] t:1    : from-worker WARNING
  widget received: ['from-loop-thread INFO']
  ```
  The worker's INFO record reached neither the widget nor stderr.
- Why it matters: every `@work(thread=True)` worker (ingest, exports, embeddings) logs INFO that the in-app Logs screen never shows; the comment at 169-170 claims the opposite ("call_soon_threadsafe ... from any thread").
- Recommended correction: capture `self._loop = loop` in `start_processor` and use `self._loop.call_soon_threadsafe(...)` in `emit` (fall back to the current behaviour only when `_loop` is None/closed).
- Size: S · ADR: no · Confidence: verified
- Pinning test: none (`rg -l RichLogHandler Tests` → harness/conftest references only)
- Already covered: none
### P2 [D1] — `TTSEventHandler._request_cooldown` is a class-body dict mutated in place, so TTS cooldown state is process-global and outlives the handler that wrote it  ·  _slice: EVENTS_

- Where: `tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py:600` (`_request_cooldown: Dict[str, float] = {}`), never reassigned in `__init__` (`:606-697`). Every access mutates the class object: `:756-760` (`del`), `:767-773` (`del`), `:1656` (`.pop`), `:1658-1659` (read), `:1680` (write).
- Evidence:
```
cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY - <<'EOF'
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSEventHandler as H
a, b = H(), H()
print("same dict object:", a._request_cooldown is b._request_cooldown is H._request_cooldown)
a._request_cooldown["msg-42"] = 1234.0
print("written on a, visible on b:", b._request_cooldown)
print("a fresh third handler starts with:", H()._request_cooldown)
EOF
```
→ `same dict object: True` / `written on a, visible on b: {'msg-42': 1234.0}` / `a fresh third handler starts with: {'msg-42': 1234.0}`
- The repo-wide check that this is the only one in the slice: an `ast` sweep of every class body under `Event_Handlers/` for a mutable literal/`dict()`/`list()`/`set()` default found three — `chat_image_events.py:38 ChatImageHandler.SUPPORTED_FORMATS` (never mutated; in fact never *read* — `rg -n SUPPORTED_FORMATS tldw_chatbook/` shows no use inside that file), `tts_events.py:483 CostTracker.DEFAULT_COSTS` (correctly `.copy()`-ed at `:492`), and this one.
- Why it matters: `app.py:18406-18424` recreates the handler whenever `_initialize_tts_service_owned` raised (it sets `self._tts_handler = None` at `:18424`, and `_ensure_tts_handler` at `:18461` then constructs a fresh one). The fresh handler inherits the old handler's cooldown map, so a message the user already asked to speak is silently refused for up to `COOLDOWN_SECONDS` with "Please wait N seconds…" against a handler that has never spoken anything. It is also why four test files (`Tests/TTS/test_console_audio_cpp_native.py:762,1128`, `test_console_speak_autoplay.py:528,550`, `test_console_speech_snapshot_admission.py:131`, `Tests/UI/test_uat_first_time_character_chat.py:900`) have to hand-reset `handler._request_cooldown = {}` — the tests are working around the leak, and that assignment shadows the class attribute rather than fixing it.
- Recommended correction: one line — move it into `__init__` beside its sibling `self._last_cooldown_cleanup = 0.0` (already correctly per-instance at `:697`): `self._request_cooldown: Dict[str, float] = {}`. Keep the class-level annotation without a value if the type is wanted.
- Size: S · ADR: no · Confidence: verified.
- Pinning test: none asserts sharing. `Tests/TTS/test_tts_improvements.py:88-114` exercises cleanup/limit on one handler and passes either way.
- Already covered: none.
### P2 [D1] — `_permission_summary_worker` reads (and can persist) the owner-thread-only `ConsoleChatStore` from a raw worker thread  ·  _slice: CHAT-controller_
- Where: `:14045-14085` (`threading.Thread(target=self._permission_summary_worker …).start()` → `_summary_tail_messages` `:14087-14106` → `self._provider_messages_for_session(session_id)` `:28061` → `self.store.messages_for_session` → `console_chat_store.py` `_materialize_stream_buffer` → `_persist_pending_message_if_ready` → `_persist_new_message`).
- Evidence: `grep -nE 'single-threaded|owns the store' console_chat_controller.py` → `4389: store mutation always runs on the thread that owns the store.`, `25107: The store is single-threaded, so the actual fold …` (the fleet-drain consumer hops onto the loop via `call_soon_threadsafe` for exactly this reason, `:25102-25129`). `console_chat_store.py::_materialize_stream_buffer` folds buffered chunks and calls `_persist_pending_message_if_ready` (→ `_persist_new_message`, a persistence write) — i.e. `messages_for_session` is not a pure read. The only tests of this lane stub the thread out: `Tests/Chat/test_permission_summary_wiring.py:43-59` (`monkeypatch.setattr(ccc.threading, "Thread", _ThreadStub)`), so the real cross-thread path is never exercised.
- Why it matters: with `[permission_summary]` active (ADR-090), every approval round folds/persists the session's streaming rows from a non-owner thread concurrently with the loop's own `messages_for_session`/`append_stream_chunk` calls — torn snapshots or `RuntimeError: list changed size` are swallowed by the `except Exception: return []` at 14103 (summary silently missing), and a persist on that thread also leaks a registered handle by the finding-1 mechanism (raw thread, no `operation_owned_connection`).
- Recommended correction: build `tail` on the UI thread inside `_maybe_fire_permission_summary` (it already runs there for the head-mount path) and pass the ready-made list into the worker; the worker should only do the network summarizer call. S.
- Size: S · ADR: no (ADR-090 covers the feature, not thread ownership) · Confidence: verified (path and ownership contract); whether `_persist_pending_message_if_ready` actually fires in a given round depends on an unflushed buffer being present — see UNVERIFIED.
- Pinning test: none (the Thread stub above means no test can go red on this).
- Already covered: none.
### P2 [D1] — `_settle_dispatch_recovery` swallows the repository's exception on the terminal-settlement write with no log line; the user sees only the generic "Durable dispatch terminal settlement failed." and the cause is unrecoverable from logs  ·  _slice: CHAT-store_
- Where: `tldw_chatbook/Chat/console_chat_store.py:3784-3818` (`except Exception:` → restore metadata → `return False`, no `logger` call); caller turns `False` into `ConsoleDispatchSettlementError("Durable dispatch terminal settlement failed.")` at `:3941-3943`. Same shape (exception → sentinel, no log) in the settings drain at `:9006-9018` and `:9116-9128` (`except Exception: result = object()`), where the failure is recorded on the session for a Retry chip but its cause is never logged. The repository itself logs nothing in `settle_with_assistant` (`Chat/console_dispatch_repository.py:915-1010`: zero `logger.` lines).
- Evidence: `PYTHONPATH=$WT $PY <SCRATCH>/repro/store_settle_swallow.py` (real temp DB, claimed RETRY owner, `settle_with_assistant` monkeypatched to raise `Boom`) → `settle_dispatch_recovery -> False`, `log records emitted during call: []`, `any record mentioning Boom/settle: []`, `status before: complete` / `live message status after failed settle: complete | receipt id in metadata: None` (fail-closed: pre-attempt metadata restored, no partial state).
- Why it matters: this is the write that makes an assistant turn durable; when it fails in the field (disk full, constraint, schema drift) there is no diagnostic at any level, so the report is "settlement failed" with nothing to act on.
- Recommended correction: `logger.opt(exception=True).warning("console_dispatch_settlement_failed")` (type-only if content-safety is a concern — the surrounding code already uses `type(exc).__name__`) before each `return False` / `result = object()`; keep the fail-closed behaviour unchanged.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/Chat/test_console_dispatch_recovery.py::test_retry_cancel_settlement_failure_retains_exact_preterminal_owner` pins the fail-closed state (a requirement); nothing asserts silence, so adding a log line does not contradict a decision
- Already covered: none
### P2 [D1] — `_sync_disabled_action_presentation` reconstructs every button's base label from the rendered one  ·  _slice: W-library_
- Where: `library_file_notes_git_panel.py:3419-3430` — `label = str(button.label)`, `base = label.removeprefix(prefix)`, `if label != rendered_label: button.label = rendered_label`, looped over `self.query(Button)` (all 24 buttons).
- Evidence: `<SCRATCH>/probe_gitpanel_label.py` (Textual 8.2.8) → `before: Content('Ref  here', spans=[Span(4, 9, style='main')])` / `after: Content('○ Ref  here')` — the span is gone after the round trip. (Consistent with my own measurement that `Button.label` markup-parses a `str`: `Content.from_text` → `Content.from_markup`, `textual/content.py:258`.)
- Why it matters: the first styled label anyone gives a Session Git button is silently stripped on the next disabled-state sync, and the loop is not opt-in.
- Recommended correction: keep the plain base label on the button (attribute or id-keyed dict) — the same "stash the raw remainder" pattern the media/conversations canvases already use (`library_media_canvas.py:1872 button._library_row_label_rest`).
- Size: M · ADR: no · Confidence: verified (loss measured); latent today — no current label in the file contains `[`
- Pinning test: marker behaviour pinned at `Tests/UI/test_library_file_notes_workspace.py:8031, :8052`; the lossiness is not pinned.
- Already covered: none
### P2 [D1] — `_trajectory_lock` and `_capture_quiescence_lock` are held across the persistence adapter's own `BEGIN IMMEDIATE`, and the UI thread acquires them *while already holding* the sqlite write lock inside `_dispatch_branch_mutation`; the two orders invert, and the losing side is a multi-minute UI-thread stall plus lost sidecar rows  ·  _slice: CHAT-store_
- Where: lock-then-DB order: `tldw_chatbook/Chat/console_chat_store.py:14357-14365` (`write_trajectory_rows`: `with self._trajectory_lock:` → adapter `writer(list(rows))` → `CharactersRAGDB.upsert_trajectory_rows` opens `self.transaction(immediate=True)` at `DB/ChaChaNotes_DB.py:15580`; and on failure `_write_capture_failed_diagnostic` at :14364 issues a SECOND adapter write still inside the lock), `:20174-20236` (`_persist_exchanges_only`: `with self._capture_quiescence_lock:` → `writer(message_id=..., rows=rows)`). DB-then-lock order: `:21971-21994` (`_dispatch_branch_mutation` holds `db.transaction(immediate=True)`) wrapping `:11054-11068` (`_create_sibling(persist=True)` → `_persist_new_message` → `:20071 _write_trajectory_row_for_message` → `write_trajectory_rows`) and `:12311-12317` (`_update_message_content` → `_persist_existing_message` → `:20605-20606 _persist_exchanges_only`).
- Evidence (mechanism, verified): `PYTHONPATH=$WT $PY <SCRATCH>/repro/store_lock_inversion.py` — thread A enters `store._dispatch_branch_mutation(session_id)` then calls `store.write_trajectory_rows`; thread B calls `store.write_trajectory_rows` 150 ms later. Output: `A(ui, inside branch mutation): {}` / `B(worker): {}` (neither call returned inside the two 60 s joins), `warnings: ['Database error upserting trajectory rows: exception_type=OperationalError' ×8, 'trajectory_rows_write_failed', …]`, `stored event kinds: []`. Busy timeout is `timeout=15` at `DB/ChaChaNotes_DB.py:3484`; the diagnostic retry inside the lock (:14364) doubles the held window.
- Why it matters: any second thread that reaches either lock while the UI thread is mid edit/regenerate/delete freezes the UI for ≥2×15 s (observed >120 s) and the worker's trajectory/exchange rows are dropped (`write_trajectory_rows` returns False; `_persist_exchanges_only` swallows to a warning). The DB layer already serialises `seq` assignment inside its own `BEGIN IMMEDIATE` (`ChaChaNotes_DB.py:3175-3176` docstring), so the store-level lock adds no correctness and only adds the inversion.
- Recommended correction: drop `_trajectory_lock` (the adapter transaction is the serialiser) or, at minimum, release it before `_write_capture_failed_diagnostic` and never take it when `db._local.transaction_depth > 0`; same for `_capture_quiescence_lock` — narrow it to the in-memory merge in `_attach_message_exchanges_locked` and call `_persist_exchanges_only` after the lock is released. Size S.
- Size: S · ADR: no (066-console-trajectory-view-and-trace-metadata.md governs the sidecar's contents, not this lock) · Confidence: **verified (mechanism) / inferred (production reachability of the second thread)** — `commit_durable_turn` runs in `asyncio.to_thread` via `_run_durable_db_call` (`console_chat_controller.py:11036`, task-22205) but does not take either lock; the only off-loop caller path I could not settle is `publish_owners` at `console_chat_controller.py:11311` (see UNVERIFIED). Rated P2 because reachability is inferred.
- Pinning test: none (no test exercises `write_trajectory_rows` from two threads; `Tests/Chat/test_console_dispatch_recovery.py::test_dispatch_retry_cas_gap_rejects_fork_until_runtime_owner_is_published` is the only two-thread store test and covers `cas_state`, not this lock)
- Already covered: none (task-22205 Done moved the commit off-loop; task-26834 "Console interactive stalls" is the nearest open umbrella — cite it there rather than filing a duplicate stall task)
### P2 [D1] — `coerce_bool_setting(None, default)` returns `None`, not `default`, despite its `-> bool` annotation; one caller already carries a hand-written workaround  ·  _slice: CHAT-rest-2_
- Where: `tldw_chatbook/config.py:1201 coerce_bool_setting` → `config.py:1010 _get_typed_value`, whose line 1019-1020 is `if value is None: return None` under the comment "If key is missing and default is None" — but the guard fires on a None **value** regardless of what `default` is.
- Evidence: the table above, last column — `config.coerce_bool_setting(None, False)` → `None`, where all seven private re-rolls return `False`. Independently corroborated in the tree: `UI/Screens/change_review_screen.py:338-342` reads
  ```
  value = get_cli_setting("change_review", "git_actions", True)
  if value is None:
      # `coerce_bool_setting(None, ...)` returns None unchanged,
      # which would read as falsy and silently disable a feature
      # that ships ON.
      return True
  ```
  i.e. somebody has already been bitten and patched their own call site rather than the helper.
- Why it matters: `-> bool` is violated, and the failure mode is "a feature that ships ON silently reads OFF" — the exact words in that comment. Unguarded sites exist: `config.py:8622` `get_rag_citation_canonical_writes_enabled() -> bool` returns `coerce_bool_setting(section.get("canonical_writes_enabled"), False)` and therefore returns `None` whenever the key is absent; `UI/Library_Modules/library_skills_controller.py:934`, `UI/Screens/settings_screen.py:6313` and `:6046` are the same shape. Every one I traced happens to pair with `default=False`, where falsy-`None` coincides with the intended answer — which is why nothing has broken yet and why the next `default=True` site will.
- Recommended correction: one line — in `coerce_bool_setting`, `if value is None: return default` before delegating (do not change `_get_typed_value`, whose None-passthrough other typed getters may rely on). Then delete the workaround at `change_review_screen.py:339-342`.
- Size: S · ADR: no · Confidence: verified (the return value); inferred (that any shipped caller currently reads wrong — I found none, and say so)
- Pinning test: none asserts `coerce_bool_setting(None, …)`. The `change_review_screen.py` comment documents the behaviour but as a defect worked around, not as a contract, so this is a bug rather than a decision.
- Already covered: none. Outside my slice (`config.py`); found while auditing this slice's `_coerce_bool` clone at `Chat/console_rail_state.py:335`.
### P2 [D1] — `conversation_local_marks.updated_at` receives two different UTC string shapes, and the table's only ORDER BY is lexical  ·  _slice: CHAT-rest-3_
- Where: writer A `Chat/conversation_local_marks_service.py:93-95` (`_now()` -> `datetime.now(timezone.utc).isoformat()`, **microsecond** precision, or *no* fraction when `microsecond == 0`); writer B `Chat/chat_persistence_service.py:453` and `:879` and `Chat/console_dispatch_repository.py:1058,1110` (`db._get_current_utc_timestamp_iso()` -> `DB/ChaChaNotes_DB.py:8700-8713`, **millisecond** precision). Both funnel into the same INSERT at `conversation_local_marks_service.py:258-268`. Lexical readers: `:427` `ORDER BY updated_at DESC`, `:455` same.
- Evidence: `cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY <SCRATCH>/repro_marks_two_shapes.py` ->
  ```
  service._now()                    -> 2026-09-18T15:40:07.404621Z
  db._get_current_utc_timestamp_iso -> 2026-09-18T15:40:07.404Z
  stored shapes in one column: [('conv-ms', '2026-09-18T12:34:56.789Z'), ('conv-us', '2026-09-18T12:34:56.789012Z')]
  ORDER BY updated_at DESC -> ['conv-ms', 'conv-us']
  expected newest-first     -> ['conv-us', 'conv-ms']  (conv-us is 12us later)
  INVERTED
  ```
  Mechanism: on a shared `SS.mmm` prefix the ms form continues with `'Z'` (0x5A) and the us form with a digit (0x30-0x39), so the **older** ms row sorts as newer.
- Why it matters: two shapes are already committed to one column. The inversion is reachable today only *within* one `mark_type`, and no `mark_type` is currently written by both helpers (`starred`/`fleet_unseen` -> writer A only via `set_mark`; `console_unseen:*`/`console_terminal_outcome:*` -> writer B only via `set_console_terminal_with_cursor`) - so this is a live shape split with a latent ordering bug one writer away, not a bug a user hits today. The `# task-15471` comment at `:65-84` asserts "every star and fleet mark in the process goes through this instance", which is true of `set_mark` but not of `set_mark_with_cursor`, the public entry the two ms-shape callers use.
- Recommended correction: make `ConversationLocalMarksService._now()` delegate to `db._get_current_utc_timestamp_iso()` (or move that helper to `Utils/` and have both call it), so the column has one shape. S, mechanical.
- Size: S · ADR: no · Confidence: **verified**
- Pinning test: none found asserting either shape; `Tests/Chat/test_conversation_local_marks_service.py` exercises `get_mark`/ordering but not the string format.
- Already covered: none (this is an instance of the review-wide timestamp class, headline finding).
### P2 [D1] — `fs_edit`, `fs_patch` and the legacy `write_file` write in place (truncate-then-write, no fsync, symlink-following) while `fs_write` in the same module is O_EXCL-temp + fsync + `os.replace` via a pinned `dir_fd`  ·  _slice: TOOLS-MCP_
- Where: `tldw_chatbook/Tools/local_tool_impls.py:856` (`target.write_bytes(data)` in `_edit_relative_file`), `tldw_chatbook/Tools/patch_tool_impls.py:509` (`target.write_bytes(data)` in `_patch_relative_file`), `tldw_chatbook/Tools/file_operation_tools.py:744-751` (`open(path, "w"/"a")`); contrast `local_tool_impls.py:577-677` (`_atomic_write_target`)
- Evidence: `rg -n 'write_bytes\(|open\(path, "[wa]"' tldw_chatbook/Tools/` → the three sites above; `local_tool_impls.py:627,656` show the sibling's `os.fsync(temp_fd)` / `os.fsync(parent_fd)`
- Why it matters: the pinned worker is killed by its parent at the 300 s deadline (`workspace_tool_executor.py:215-227` → `_settle_after_spawn`) and `fs_edit`/`fs_patch` run inside it; a kill or crash between `O_TRUNC` and the write's completion leaves the user's file empty/partial with no recovery, and the edit path re-opens by name after `_relative_target_is_safe` (TOCTOU on a swapped symlink) where `fs_write` deliberately does not.
- Recommended correction: route `_edit_relative_file`/`_patch_relative_file` through `_atomic_write_target` (already in the module; pass `expected_sha256=None`); M (has to choose whether `fs_edit` keeps mode bits / CRLF — `_atomic_write_target` already `fchmod`s the live mode).
- Size: M · ADR: no · Confidence: verified (unconditional code path; the crash window itself was not reproduced)
- Pinning test: `Tests/Tools/test_local_tool_impls.py::test_fs_edit_unencodable_new_string_preserves_file` pins encode-before-open only; nothing pins in-place vs atomic
- Already covered: none
### P2 [D1] — `hosted_chat.owned_json_post` and `qwencloud.chat_with_qwencloud` honour a provider `Retry-After` header with no upper bound and sleep it on the worker thread  ·  _slice: LLM_
- Where: `tldw_chatbook/LLM_Calls/hosted_chat.py:867-884` (`_retry_delay` returns `float(int(raw))` uncapped) → `:581 time.sleep(delay)`; consumers moonshot (`hosted_chat_request`, `moonshot.py:237`) and zai (`owned_json_post`, `zai.py:424`). Second copy: `qwencloud.py:130-156 _advance_retry_policy` (`retry_policy.get_retry_after(...)` → urllib3 returns the raw seconds) → `:1291 time.sleep(retry_sleep)`.
- Evidence: `_retry_delay(Mock(headers={"Retry-After":"99999999"}), attempt=0, retry_delay=1.0)` → `99999999.0 seconds (no cap)`. qwencloud copy: read only.
- Why it matters: `api_base_url` is user-configurable for all three providers; a misbehaving/hostile endpoint pins the gateway worker (`console_provider_gateway.py:6337 asyncio.to_thread(worker)`) in `time.sleep` for an arbitrary time — Stop cancels the task but cannot interrupt the thread.
- Recommended correction: clamp to a small cap (e.g. `min(delay, 60.0)` or `config.timeout`) in both copies; treat larger values as "fail now" (S).
- Size: S · ADR: no (ADR 062/045 cover the boundary, not the retry cap) · Confidence: verified (function) / inferred (thread pin)
- Pinning test: `Tests/Chat/test_dispatcher_status_mapping.py:66,75` exercises Retry-After parsing at the dispatcher, not this cap.
- Already covered: none
### P2 [D1] — `replace_conversation_capture_detail` leaks one registered ChaChaNotes connection per call on a file-backed DB  ·  _slice: CHAT-controller_
- Where: `tldw_chatbook/Chat/console_chat_controller.py:5028-5050` (`run_repository_call` spawned as a bare `threading.Thread(name="console-capture-policy-write")`); contrast `:10448-10456` `_run_owned_chat_db_operation` / `:10458-10480` `_run_durable_db_call`, which wrap the same class of write in `operation_owned_connection`.
- Evidence: repository-level probe (scratchpad, run under `env.sh` with cwd=worktree; `ConsoleCapturePolicyRepository(db).replace(conv, CaptureDetail.SAFE)` on a real conversation in a tmp-file `CharactersRAGDB`):
  ```
  PROBE baseline registered handles: 1
  PROBE after 3 raw threads (controller.py:5044 shape): 4
  PROBE after 3 owned threads (_run_owned_chat_db_operation shape): 4
  ```
  Same result with a bare `db.get_connection().execute("SELECT 1")` per thread (1 → 4 → 4). Registry is a strong `dict[int, sqlite3.Connection]` (`DB/base_db.py:112`), so the handle and its file descriptor survive until a backup `quiesce_connections()` → `close_registered()` (`base_db.py:253-272`) closes strays. `console_capture_policy_repository.py:114` confirms `replace` opens a thread-local handle via `self.db.transaction(immediate=True)`.
- Why it matters: every conversation-level Capture Full/Safe change by the user leaks a live sqlite connection (fd + WAL reader) for the lifetime of the process (or until the next backup pass); the codebase pins the opposite invariant elsewhere (`Tests/UI/test_console_send_refresh_scope.py:129`, `test_console_send_diagnostic_flow.py:185`, `test_console_composer_blink_wrap.py:198` assert `registered_connection_count() == 0` after a send).
- Recommended correction: wrap `run_repository_call`'s body in `operation_owned_connection(getattr(self.store.persistence, "db", None))` (the helper `_run_owned_chat_db_operation` already exists 10448); keep the hand-rolled Event/`call_soon_threadsafe` handoff if the cancel-survival semantics are wanted, or replace the thread with `asyncio.to_thread(self._run_owned_chat_db_operation, repository.replace, ...)` + the existing shield loop.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none — `Tests/Chat/test_console_chat_controller_exchanges.py::test_conversation_policy_reservation_blocks_race_and_reconciles_cancel` (and siblings at 989–1347) drive this method against a file-backed `CharactersRAGDB` (line 749) but never assert the handle count.
- Already covered: none (task-31502 is the registry's shared lock, not stray handles).
### P2 [D1] — `summarize_with_anthropic` mounts a Retry adapter on a session it never uses; the request goes through bare `requests.post`, so configured `api_retries` never apply and a 429 returns `None` on the first attempt  ·  _slice: LLM_
- Where: `tldw_chatbook/LLM_Calls/Summarization_General_Lib.py:1109-1131` (session + adapter built, then `requests.post(...)` at `:1131`), `:1253-1258` (non-200/non-500 → `return None`), `:1106` (manual loop only retries 500 / RequestException). The unused session is also never closed — one leaked `Session` per attempt.
- Evidence: `PYTHONPATH=$WT $PY $SCRATCH/llm_repro_summarization.py` (patch `requests.post` → 429, spy on `requests.Session.post`) → `[anthropic 429] result=None requests.post calls=1 Session.post calls=0`
- Why it matters: a rate-limited Anthropic summarization (Library ingest analysis) yields `None` → `analyze()` reports "Error: Summarization failed unexpectedly." with no retry despite `[anthropic_api] api_retries`; every sibling posts through the session and does get the policy.
- Recommended correction: `session.post(...)` at `:1131`; close the session in a `finally`. The diagnostic manifest (`test_summarization_diagnostic_privacy.py`) freezes this module's log call sites, so the fix must not add a log line.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none for the retry path
- Already covered: none
### P2 [D1] — `summarize_with_vllm` fails on every call that passes an explicit `api_key` (`loaded_config_data` is only bound on the no-key branch)  ·  _slice: LLM_
- Where: `tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py:1301-1305` (binds `loaded_config_data` only when `api_key` is blank) vs `:1412-1413`, `:1463-1464` (reads it unconditionally). `analyze()` callers pass `api_key` through (`Local_Ingestion/Book_Ingestion_Lib.py:1354`, `PDF_Processing_Lib.py:862`, …).
- Evidence: `L.summarize_with_vllm("k", "text", "prompt")` with `load_settings` patched to a full `vllm_api` table and a mocked 200 → `"vLLM Summarization: Unexpected error occurred: cannot access local variable 'loaded_config_data' where it is not associated with a value"`; `L.summarize_with_vllm(None, …)` → `'THE SUMMARY'`.
- Why it matters: any ingest path that resolves a vLLM credential before calling `analyze()` (the documented, gated path) gets an error string; only the key-less fallback works.
- Recommended correction: hoist `loaded_config_data = load_settings()` above the key check (S).
- Size: S · ADR: no · Confidence: verified
- Pinning test: none (`rg -n "summarize_with_vllm" Tests/` → diagnostic-privacy canaries only, which call it with `api_key=None`).
- Already covered: none found (`rg -il "summarize_with_vllm" backlog/tasks/` → none).
### P2 [D1] — `terminal.py` swallows every runtime failure with no log anywhere: nine `except Exception: return False/None` in a file that imports no logger  ·  _slice: UIM-console_
- Where: `UI/Console_Modules/terminal.py` — `request_focus:298`, `request_retry_cleanup:317`, `send_key:328`, `send_paste:345`, `request_resize:366`, `_refresh:428`, `_session_view:481`, `_selected_session_id:500`, and the nested `detach_workspace:113/137/141`. Five of these (`open_workspace:98/110`, `request_arm:152/165`, `request_new_session:189/225/230`, `request_rename:262`, `request_close:286`) at least call `self._status(...)`; the nine above return silently.
- Evidence: `grep -c "logger\|logging" tldw_chatbook/UI/Console_Modules/terminal.py` → `0`. Every other controller in this slice logs at its boundaries (`dictation.py:1223`, `video.py:130`, `image.py:461`, `retrieval.py:547`, `hands_free.py:1269`, …).
- Why it matters: a keystroke or paste dropped by a manager-side error (including an `AttributeError` from a refactor) is indistinguishable from a working terminal that ignored you; `_refresh` returning early leaves the projection frozen. Nothing reaches the log, the status line, or persistent diagnostics, so there is no way to diagnose it after the fact.
- Recommended correction: bind `loguru` in this module and add a one-line `logger.opt(exception=True).debug(...)` in each of the nine handlers (narrowing the catches would be better still, but the manager boundary has no shared error base — the same reason `character_context.py:375` gives).
- Size: S · ADR: no · Confidence: verified (grep + full read)
- Pinning test: none.
- Already covered: none.
### P2 [D1] — a loguru call uses `%s` formatting, so the field name is silently dropped from the log line  ·  _slice: EVENTS_

- Where: `Event_Handlers/LLM_Management_Events/llm_management_events.py:155` — `logger.info("File/Directory selection cancelled for #%s.", input_widget_id)`. The same function's other two calls (`:147`, `:150`) correctly use `{}`.
- Evidence:
```
cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && $PY - <<'EOF'
import sys
from loguru import logger
logger.remove(); logger.add(sys.stdout, format="{message}")
logger.info("File/Directory selection cancelled for #%s.", "llamafile-exec-path")
logger.info("Correct loguru style: {}", "llamafile-exec-path")
EOF
```
→ `File/Directory selection cancelled for #%s.` / `Correct loguru style: llamafile-exec-path`
- Why it matters: every cancelled file-picker in LLM Management logs an identical, field-less line, so the log cannot say *which* picker was cancelled — the exact thing the argument was added for. Repo-wide there are 12 more in 8 other files (`rg -c 'logger\.(debug|info|warning|error|critical|exception)\("[^"]*%[sdrf]' --glob '*.py' tldw_chatbook/` → `Embeddings/Embeddings_Lib.py:1, UI/Wizards/FirstRunSetupWizard.py:1, Chat/custom_endpoint_registry.py:2, UI/Screens/settings_screen.py:4, Chat/prompt_template_manager.py:1, UI/Screens/scheduling/conflicts_tab.py:1, Library/local_media_chunk_tool_service.py:1, TTS/kokoro_pytorch.py:1`) — a defect class, not a one-off.
- Recommended correction: `%s` → `{}` at `:155`; the repo-wide fix is the same mechanical swap plus a `ruff`/grep guard, since this is invisible in review and in tests.
- Size: S · ADR: no · Confidence: verified.
- Pinning test: none.
- Already covered: none.
### P2 [D1] — a misconfigured reranker provider reports `UnboundLocalError: cannot access local variable 'response'` instead of the real error  ·  _slice: RAG_
- Where: `tldw_chatbook/RAG_Search/reranker.py:620-634` (`PointwiseReranker._score_result`): `response = await self._call_llm(prompt)` is inside the `try`, and the `except (json.JSONDecodeError, ValueError, KeyError)` handler's log line reads `len(response)` and `content_fingerprint(response)` — both unbound when `_call_llm` itself raised.
- Evidence (isolated env, `_call_llm` patched to raise the exact error the dispatcher produces):
  ```
  $PY -c "... patch PointwiseReranker._call_llm -> raise ValueError('Unsupported API endpoint: foo') ...
           await r._score_result('q', res, 0)"
  → RAISED UnboundLocalError: cannot access local variable 'response' where it is not associated with a value
  ```
  That `ValueError` is real, not invented: `Chat/Chat_Functions.py:1073` raises `ValueError(f"Unsupported API endpoint: {endpoint_display}. Valid endpoints: …")` whenever `API_CALL_HANDLERS` has no handler for the configured `model_provider`, and `BaseReranker._call_llm` (`reranker.py:301-330`) re-raises it after exhausting `max_retries`.
- Why it matters: the result row still degrades correctly (`asyncio.gather(..., return_exceptions=True)` at `:539` catches it and the row is counted failed), but the ONLY diagnostic the user and the log get is `Failed to score result 0: cannot access local variable 'response'` — the message that would have named the misconfigured provider and listed the valid ones is destroyed. This is a setup-error path, which is exactly where a useful message matters most, and the same `reranker.py` header records TASK-17065 fixing a previous version of "reranking silently reached 0 of the 29 providers".
- Recommended correction: initialise `response = ""` before the `try`, or narrow the handler to wrap only `json.loads`/`float` (the parse errors it is for) and let a call failure propagate to `gather`, where it is already handled and logged with its own type and message.
- Size: S · ADR: no · Confidence: verified (reproduced)
- Pinning test: `Tests/RAG_Search/test_reranker_degraded_paths.py` exists and pins the degraded-rerank contract; it does not cover a raising `_call_llm` at this call site (the repro above goes red today).
- Already covered: none
### P2 [D1] — a profile switch retires the shared RAG service without ever calling `close()`, and a module global in `health_check.py` pins the retired instance past garbage collection  ·  _slice: RAG_
- Where: `tldw_chatbook/RAG_Search/ingestion_indexing.py:483-489` (`set_shared_rag_service` / `reset_shared_rag_service` — reassign the global, no `close()`), against `tldw_chatbook/RAG_Search/simplified/rag_service.py:850 init_health_checker(self)` → `simplified/health_check.py:437-440` (`_health_checker = RAGHealthChecker(rag_service)`, a module-level **strong** reference; `health_check.py:63 self.rag_service = rag_service`).
- Shipped callers of the retirement path: `UI/Screens/settings_rag_profile_adapter.py:471` (Settings ▸ RAG save) and `RAG_Search/simplified/active_config.py:373` (`set_active_profile`).
- Evidence (isolated env, `create_config_for_testing()` → mock embeddings + in-memory store):
  ```
  service 1 still alive after del + gc: True
    held by health_check._health_checker.rag_service: True
    its ThreadPoolExecutor shut down: False
  after a SECOND service is built, service 1 alive: False
    _health_checker now points at service 2: True
  ```
- Why it matters: `RAGService.close()` shuts down the per-service `ThreadPoolExecutor` (`rag_service.py:855`, up to 8 threads) and releases the embeddings handle, the Chroma client, and the DB connection pools. A profile switch drops the only *intended* reference and calls none of that, so the previous profile's loaded embedding model and open Chroma client stay resident until the next service is built (which is what finally overwrites `_health_checker` and lets GC run). The module already knows this rule and applies it on the *other* retirement path: `ingestion_indexing.py:225-250 _close_discarded_rag_service` closes a race-losing build and documents "a discarded build is therefore not actually resource-free". The main retirement path skips it.
- Recommended correction: have `set_shared_rag_service` close the instance it displaces, using the existing `_close_discarded_rag_service` helper and its documented rule (call it OUTSIDE `_shared_service_lock`, since `close()` blocks on `ThreadPoolExecutor.shutdown`). Separately, `init_health_checker` should hold a `weakref` (or the global should be dropped with the service) — nothing reads it anyway (see the next finding).
- Size: M · ADR: no · Confidence: verified
- Pinning test: none (`grep -rn "reset_shared_rag_service" Tests/` finds isolation fixtures, none asserting close-on-retire)
- Already covered: none
### P2 [D1] — a raising filesystem service leaves the Workspace Files modal permanently on "Loading folder…", with no log line anywhere in the module  ·  _slice: W-console-2_
- **Where:** `console_workspace_files_modal.py:202-214` — `_OperationLane._run`'s `except Exception: outcome = None` (`:208-209`), after which `:213` refuses to publish and the caller's `status_copy="Loading folder…"` (`:606`) is never replaced. The module imports **no logger at all** (`grep -n logger console_workspace_files_modal.py` → no hits).
- **Evidence (reproduced):** `probe_wsfiles.py` — the real modal in the real `_Host` harness (`Tests/UI/test_console_workspace_files_modal.py`) with an inspector whose `list_directory` raises `PermissionError`:
  ```
  status_copy  = 'Loading folder…'
  tree content = 'Loading folder…'
  state.status = 'Loading folder…'
  lanes active = 0
  ```
  Stuck forever, zero lanes running, nothing logged.
- **Why it matters:** an unexpected raise is indistinguishable from "still loading" for the user and invisible to an operator. Why P2 and not P1: the real `LocalWorkspaceFileInspector.list_directory` (`Workspaces/file_inspector.py:290-370`) is heavily guarded and returns explicit `DirectoryStatus.FAILED` values, so I could not show a *shipped* path that raises — `os.close(root_fd)` at `:328`, `os.fstat(directory_fd)` at `:331` and `_directory_revision_from_stat` sit outside the `try` that starts at `:343`, which is the nearest thing to one.
- **Recommended correction:** log the exception (`logger.opt(exception=True).warning(...)`) and publish a failure status (`status_copy = "Folder is unavailable."`) instead of `outcome = None`. `_publish_directory` already has copy for `DirectoryStatus.FAILED` (`:663`).
- **Size:** S · **ADR:** no · **Confidence: verified** (consequence reproduced; trigger inferred)
- **Pinning test:** none for the raising branch; `Tests/UI/test_console_workspace_files_modal.py` exercises the returning-status branches.
- **Already covered:** none
### P2 [D1] — a swallowed read error silently disables the ONLY optimistic-concurrency check on the Prompt save path, turning a conflicting save into a last-writer-wins overwrite  ·  _slice: UIM-library_
- Where: `tldw_chatbook/UI/Library_Modules/library_prompts_controller.py:3116-3146` (the pre-read and its `except Exception: fresh = None` at `:3124-3125`); contract stated by the method's own docstring at `:2946-2953`
- Evidence: read only — see UNVERIFIED. The chain is unambiguous in three lines. `_save_library_prompt`'s docstring states that the write seam "always re-derives the version to bump from a fresh read inside its own transaction, so it cannot detect 'this editor's cached version is stale' by itself. **This method does that staleness check itself**, via a fresh `get_prompt` read, BEFORE attempting the real write." That read is wrapped `try: fresh = await …get_prompt(…) except Exception: fresh = None` with **no logging at all**; `fresh_version = fresh.get("version") if isinstance(fresh, Mapping) else None` (`:3132`) is then `None`, so the guard `if fresh_version is not None and …` (`:3133-3137`) is False, `_enter_library_prompt_conflict` is skipped, and control falls straight through to the real `save_prompt` at `:3149`, which bumps whatever version it finds. The sibling pre-check at `:3079-3088` (`except Exception: candidate = None`) degrades the same way but is harmless — it only skips a name-collision pre-classification the unique-name constraint re-raises on the write.
- Why it matters: the stated design is that the editor, not the DB, detects a stale version. A transient read failure (lock/timeout/closed handle — the write is a separate call that can still succeed) therefore converts "another writer changed this prompt, show the conflict banner" into a silent overwrite of that writer's edit, logging nothing an operator could see afterwards.
- Recommended correction: on a failed pre-read, refuse the save rather than proceeding — surface the existing "Couldn't load the selected Prompt… retry" copy (`_apply_library_prompt_detail_failure`, two hundred lines up) and log with `logger.opt(exception=True).warning`. A read that did not answer must not read as "nothing changed"; that is the distinction `_load_library_note_backlinks` already makes explicitly (`library_notes_controller.py:3498-3500`: "A lookup that did not answer must not read as 'no notes link here yet'").
- Size: S · ADR: no · Confidence: inferred
- Pinning test: none found — no test in `Tests/UI/test_library_prompt*.py` forces the pre-read to raise
- Already covered: none
### P2 [D1] — server-port residue in `_load_settings_uncached`: a malformed `SINGLE_USER_FIXED_ID` env var aborts `import tldw_chatbook.config`; 8 settings keys nobody reads; 5 dead expression statements  ·  _slice: ENTRY-config_
- Where: `tldw_chatbook/config.py:2120-2125` (`int(os.getenv("SINGLE_USER_FIXED_ID", ...))` uncaught), `2126-2131` (`os.getenv("API_KEY", ...)` result discarded), `2138-2139` (two `get_toml_section(...)` results discarded), `2155-2156` (`log_level_env` computed, `_get_typed_value(...).upper()` discarded), keys `APP_MODE_STR`/`SINGLE_USER_MODE`/`PROJECT_ROOT`/`API_COMPONENT_ROOT`/`SINGLE_USER_FIXED_ID`/`SINGLE_USER_API_KEY`/`DATABASE_URL`/`USERS_DB_CONFIGURED` at 2312-2352; stale comments 2054-2061 ("config.py is in project_root/tldw_server_api/app/core/config.py").
- Evidence: `SINGLE_USER_FIXED_ID=abc $PY -c "import tldw_chatbook.config"` →
  ```
  File ".../config.py", line 2120, in _load_settings_uncached
      single_user_fixed_id = int(
  ValueError: invalid literal for int() with base 10: 'abc'
  ```
  `rg -n '"(DATABASE_URL|USERS_DB_CONFIGURED|SINGLE_USER_MODE|APP_MODE_STR|PROJECT_ROOT|API_COMPONENT_ROOT|SINGLE_USER_FIXED_ID|SINGLE_USER_API_KEY)"' tldw_chatbook Tests --glob '!config.py'` → only unrelated MCP env-literal fixtures; no reader of these settings keys. (`USERS_NAME` IS read — `Tests/Tools/test_note_tool_user_id.py` — keep it.)
- Why it matters: the whole app fails to import over an env var that nothing consumes; the rest is ~40 lines of dead server-port code in the hottest function of the module.
- Recommended correction: delete the eight residue keys, the five dead statements and the `int(...)` line; keep `USERS_NAME`.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none
### P2 [D1] — the Markdown prompt export collapses distinct prompt names to the same `.md` filename, writes duplicate entries into the ZIP, and still reports "Successfully exported N prompts"  ·  _slice: DB-rest_
- Where: `tldw_chatbook/DB/Prompts_DB.py:5091-5097` — `safe_filename = re.sub(r"[^\w\-_ \.]", "_", p_data["name"]) + ".md"`, then `open(os.path.join(temp_zip_dir, safe_filename), "w")` and `zipf.write(..., arcname=safe_filename)` inside the per-prompt loop; status line at `:5100`.
- Evidence (isolated env, temp-file DB, two prompts named `a:b` and `a?b`):
  `export_prompts_formatted(db, export_format="markdown")` → `Successfully exported 2 prompts to Markdown in a ZIP file.` / `warnings: ["Duplicate name: 'a_b.md'"]` / `namelist: ['a_b.md', 'a_b.md']` / `z.read("a_b.md") -> b'# a?b (f3ce4fda-032a'` — i.e. both prompts are in the archive under one name, the on-disk staging file was overwritten, and every normal extractor (`unzip`, Finder, `ZipFile.extractall`, `ZipFile.read`) yields only the last one. Prompt names are free text and `:` `?` `/` `|` `#` `,` all map to the same `_`, so a two-prompt collision needs no contrivance.
- Why it matters: a user exporting their prompt library silently gets fewer files than prompts, with a success message that says otherwise. Export is the backup path.
- Recommended correction: de-duplicate `arcname` per archive (append `-{p_data['id']}` or a `(2)` suffix on collision) and drop the on-disk staging entirely — `zipf.writestr(arcname, md_content)` removes both the temp directory and the overwrite.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none (`rg -n "export_prompts_formatted" Tests` — see dispositions)
- Already covered: none
### P2 [D1] — the Permissions matrix writes the permission store SYNCHRONOUSLY on the event loop (~13 ms per keypress), unlike every other write in the same file  ·  _slice: UIM-nav-mcp-persona_
- Where: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py:3345`, `:3356`, `:3387`, `:3398`, `:3417` (`on_mcp_permissions_mode_state_cycle_requested` -> `_call_profile_scoped(service.set_global_default / set_server_default / set_tool_state, ...)`) and `:3477` (`on_mcp_permissions_mode_kill_switch_toggled` -> `set_kill_switch(event.value)`). Both are `async def` handlers running on the loop; neither offloads.
- Evidence:
  - Write path traced: `unified_control_plane_service.py:5304 set_tool_state` -> `permission_store.py:1501 MCPPermissionStore.set_tool_state` -> `_mutate_locked` -> `permission_store.py:909 save()` -> `json.dump(..., indent=2, sort_keys=True)` + `os.replace` (`:953`, `:956`).
  - Same shape at `:4816` (`on_mcp_inspector_reallow_requested` -> `set_tool_state`), `:4876` (`remove_tool_arg_rule`), `:4938` (`revoke_session_approval`).
  - Measured:
    ```
    cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$PWD $PY - <<'EOF'
    import time, tempfile, pathlib, statistics
    from tldw_chatbook.MCP.permission_store import MCPPermissionStore
    d = pathlib.Path(tempfile.mkdtemp()); s = MCPPermissionStore(d/"perm.json")
    s.set_kill_switch(True); s.set_kill_switch(False)
    ts=[]
    for i in range(20):
        t=time.perf_counter(); s.set_kill_switch(i%2==0); ts.append((time.perf_counter()-t)*1000)
    print("median %.2f min %.2f max %.2f" % (statistics.median(ts),min(ts),max(ts)))
    EOF
    ```
    -> `MCPPermissionStore.set_kill_switch ms: median 13.24 min 12.71 max 16.69`
- Why it matters: this file already offloads its blocking writes -- `_save_builtin_flag` (`:5748`) and `_save_tool_gate` (`:5802`) both wrap `save_setting_to_cli_config` in `asyncio.to_thread`, with a docstring explaining why. The permission setters, the *more* frequently exercised path (Space-cycling is the mode's primary gesture), do not. Combined with the `tool_gate_breadcrumb()` finding above, one Space press costs ~13 ms (write) + ~56 ms (breadcrumb) + the profile-inventory store reads `_capture_permission_render_state` makes 2-4 times, all on the loop before the matrix repaints.
- Recommended correction: wrap the five setter calls and `set_kill_switch` the same way `_save_builtin_flag` already wraps its write (`await asyncio.to_thread(...)`), keeping the existing resync-after shape.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: the store-side write itself is named by the sibling TOOLS-MCP P2 finding ("a sync JSON write before each of 65 Hub mutation/read sites"); these UI call sites are the amplifier, not the store.
### P2 [D1] — the lasting-sync abandon path is the only `except Exception: pass` in the slice with no diagnostic at all, and the awaited call it swallows sits BEFORE the six `pop()`s that release the root's lease  ·  _slice: UIM-library_
- Where: `tldw_chatbook/UI/Library_Modules/library_notes_sync_controller.py:1175-1178` (`abandon_setup`); the swallowed body is `tldw_chatbook/Notes/notes_sync_runtime.py:2379-2400`
- Evidence: read only — see UNVERIFIED. `abandon_setup` does `try: await self._runtime.abandon_setup(root_id) except Exception: pass` — no `logger` call of any kind, unlike the other five `except …: pass` sites in this slice, each of which carries a comment naming a specific narrow exception (see Verified-fine). In the runtime, `abandon_setup` → `_release_setup_authority`, whose FIRST statement is `await self._maintenance_offload(self._coordinator.close_admission, lease, …)` (`:2392-2394`) and whose next six statements pop `self._leases`, `_admissions`, `_admission_reasons`, `_setup_reviews`, `_root_paths`, `_root_status` (`:2395-2400`). A raise out of the offload therefore skips all six pops, leaving the runtime holding a provisional lease and admission for that root, and the controller swallows the evidence.
- Why it matters: this is the "the user pressed Back out of Add-from-files" path. A failure leaves an invisible held lease on a folder the user believes they abandoned, and nothing — no log line, no notice, no state — records that it happened.
- Recommended correction: log it (`logger.opt(exception=True).warning`, metadata only — a root path must not reach the sink; the file already follows that rule elsewhere), and release the in-memory maps in `_release_setup_authority` from a `finally` so a failed coordinator call cannot strand them.
- Size: S · ADR: no · Confidence: inferred
- Pinning test: none · Already covered: none
### P2 [D1] — the three prompt-export paths write user prompt bodies to a predictable, world-readable path in the shared temp directory  ·  _slice: DB-rest_
- Where: `tldw_chatbook/DB/Prompts_DB.py:4806-4810` (`prompt_keywords_export_{timestamp}.csv`), `:4971-4973` (`prompts_export_{timestamp}.csv`), `:5008-5010` (`prompts_export_markdown_{timestamp}.zip`) — each is `os.path.join(tempfile.gettempdir(), f"...{datetime.now().strftime('%Y%m%d_%H%M%S')}...")` followed by a plain `open(..., "w")` / `zipfile.ZipFile(..., "w")`. (The fourth `tempfile` use, `tempfile.mkdtemp()` at `:5007`, is correct — 0700 — and is only the staging dir.)
- Evidence (isolated env): the same script above printed `CSV path: /var/.../T/prompts_export_20260918_074546.csv mode: 0o644` and `KW CSV path: /var/.../T/prompt_keywords_export_20260918_074546.csv mode: 0o644`. The CSV contains `System Prompt` and `User Prompt` columns in full (`:4979-4992`).
- Why it matters: two defects at once. (a) The name is derived only from a one-second-resolution timestamp, so on Linux — where `gettempdir()` is the shared, world-writable `/tmp`, unlike macOS's per-user `/var/folders/.../T` — any local user can pre-create the path as a symlink and redirect the write (CWE-377), or simply read the 0644 result (CWE-378): prompt bodies are user content and often carry credentials/system instructions. (b) Two exports inside the same second overwrite each other.
- Recommended correction: `tempfile.mkstemp(prefix="prompts_export_", suffix=".csv")` (0600, unique, O_EXCL) at all three sites, returning the fd/path — same line count, no new dependency.
- Size: S · ADR: no (no `backlog/decisions/` file covers temp-file creation; grep of `<SCRATCH>/adr_list.txt` for "temp" is in dispositions) · Confidence: verified (path + mode observed; the Linux `/tmp` half is platform reasoning, not run here)
- Pinning test: none
- Already covered: none
### P2 [D2] — A swallowed in-place patch permanently desyncs the Prompts/Skills work panes, because their change-detector reads the attribute the failed patch already wrote  ·  _slice: W-library_
- Where: `library_prompt_work_pane.py:42-46` and `library_skill_work_pane.py:176-182` (`all(getattr(self, key, object()) == value …)` → early return) against patchers that assign first and query second: `library_prompts_canvas.py:1746 sync_memberships`, `:476-482 sync_lifecycle_actions`, `library_skills_canvas.py:1086-1097`. Every caller swallows `NoMatches`/`QueryError` (`library_prompts_controller.py:1152-1157`, `:2879-2889`, `library_skills_controller.py:2349-2362`).
- Evidence: `<SCRATCH>/probe_skills_workpane_desync.py` → `summary widgets at compose: 0` / `patch raised (caller swallows this): NoMatches` / `pane.membership_state now: True` / `summary widgets after sync_state: 0` → `AssertionError: assert 0 == 1`. The next snapshot sees "unchanged" and skips the recompose; the Collections group never renders.
- Why it matters: the guard compares against INTENDED state, not RENDERED state, so any failed patch is unrecoverable instead of self-healing on the next snapshot. This is the pattern that manufactures the P1s above.
- Recommended correction: assign the attributes only after the `query_one` calls succeed (or keep a `_rendered_*` snapshot for the guard to compare against).
- Size: M · ADR: no · Confidence: verified (mechanism); inferred (that the membership variant fires in a live session)
- Pinning test: none · Already covered: none
### P2 [D2] — Every File Notes control repaint costs ~9.5 ms of config reads while no folder is linked  ·  _slice: W-library_
- Where: `library_file_notes_workspace.py:2861-2888 _configured_sync_folder` (two `get_cli_setting` calls in a `for` loop) called at `:2960` from `_update_root_surface`, which has 15 call sites plus a repeating `set_interval(STRUCTURAL_WAIT_PATIENCE_SECONDS=3.0, self._update_root_surface)` armed for every folder change (`:6909-6915`).
- Evidence: measured in the isolated env —
  `cd <wt> && source <SCRATCH>/env.sh && PYTHONPATH=<wt> $PY -c "<50-iteration timing loop over the exact two get_cli_setting calls>"` → `warm _configured_sync_folder config-read pair: median 9.45 ms, max 14.85 ms`.
  `_initialize` alone calls `_update_root_surface` twice (`:2506`, `:2510`) on top of the `on_mount` call at `:2364` → ~28 ms of config reads before the unlinked mode's first paint.
- Why it matters: the whole cost lands exactly in the state where the user is choosing a folder (the guard `None if self._root is not None` keeps it off the linked paths), and it repeats on a 3 s tick for the length of a folder change.
- Recommended correction: resolve the configured folder once per root change / per mount into an attribute and invalidate it where `_root` changes; nothing in `_update_root_surface` needs a fresh config read per repaint.
- Size: S · ADR: no · Confidence: verified (measured)
- Pinning test: none
- Already covered: none
### P2 [D2] — Every local-branch `UnifiedMCPControlPlaneService` call runs its `LocalMCPControlService` work synchronously on the event loop: `_maybe_await(sync_call())` evaluates the call before awaiting, and the section getters re-parse `server.py` with `ast` three times per call (6.3 ms measured) plus a full JSON-store load; mutations add a JSON temp-write+replace  ·  _slice: TOOLS-MCP_
- Where: `tldw_chatbook/MCP/unified_control_plane_service.py:568-571` (`_maybe_await`), `:551-565` (`load_section` local branch → `self.local_service.get_overview()/get_inventory()/...`), `:1293-1467` (`run_action` local branch: `save_external_profile`, `delete_external_profile`, `save_governance_rule`, ... all sync store writes), `:2326-2350` (`_record_local_attempt` → `store.save_profile_runtime_state` = full load + full save), `:2400-2422` (`save_local_profile`/`delete_local_profile`/`local_external_catalog` call the sync store directly); `tldw_chatbook/MCP/local_control_service.py:200-230` (`get_overview` → `get_inventory` → `manifest_provider()`), `tldw_chatbook/MCP/server.py:122-123,216-261` (`_load_server_module_ast()` = `Path(__file__).read_text()` + `ast.parse` per `_extract_registered_entries` call, ×3 per manifest, no memo)
- Evidence: `rg -c '^    async def ' MCP/local_control_service.py` → 13 vs `rg -c '^    def '` → 38 (the five section getters are sync); `PYTHONPATH=$WT $PY - <<EOF ... describe_local_mcp_capabilities() ... EOF` → `first 7.6 ms, steady 6.3 ms/call, tools=30`; `rg -n 'manifest_provider\(\)'` → 2 consumers (`local_control_service.py:224`, `local_runtime_delegate.py:673`, the latter on every `tools/list`)
- Why it matters: this service is the Hub screen's data source on the Textual loop; every section render blocks the loop for the JSON load + 3× AST parse, every profile save/delete/governance edit blocks it for a temp-file write + `replace`, and `_record_local_attempt` does that write twice per connect/test/refresh. The reference shape for the fix already exists in the tree (`Chat/chat_conversation_scope_service.py:152-215`, task-283: `if not inspect.iscoroutinefunction(fn): await asyncio.to_thread(fn, ...)`). Trace requested by the brief: `load_section("inventory")` → `local_service.get_inventory()` (sync) → `manifest_provider()` → `describe_local_mcp_capabilities()` → `_load_server_module_ast()` ×3; `load_section("external_servers")` → `get_external_servers()` → `store.get_external_catalog()` → `LocalMCPStore.load()` → `json.load(...)`. No sqlite on these paths (the only MCP sqlite is `tools.py`'s standalone-server calls; the in-process Library tools go through `asyncio.to_thread` in `local_runtime_delegate`).
- Recommended correction: (a) memoize `_load_server_module_ast()` on `(path, st_mtime_ns)` — S, removes 6.3 ms per manifest; (b) in `_maybe_await`'s callers, offload sync `local_service` methods with the task-283 shape (`to_thread` when not a coroutine function) — M.
- Size: S (a) / M (b) · ADR: no · Confidence: verified (cost measured; loop placement by reading)
- Pinning test: `Tests/MCP/test_local_control_service.py::test_local_control_service_uses_real_local_manifest_helper_by_default` pins that the real helper is used (would still pass with a memo); nothing pins the per-call re-parse
- Already covered: none
### P2 [D2] — First open of the emoji picker blocks the event loop for ~180 ms building the emoji index inside `EmojiPickerScreen.__init__`  ·  _slice: W-top_
- Where: `tldw_chatbook/Widgets/emoji_picker.py:889` (`get_emoji_data()` called from `__init__`), building via `_load_emojis()` at `:119-684`; cache at `:688-698`.
- Evidence (verified):
  ```
  cd $WT && source env.sh && PYTHONPATH=$WT $PY -
  get_emoji_data cold=179.7 ms  warm=0.0004 ms  n_emojis=5225 n_categories=9
  _filter_emojis('sm') = 0.8 ms/call over 5225 emojis      <- debounce is fine
  ```
- Why it matters: CLAUDE.md's own rule is "Workers for operations >100ms". This is 1.8x over, on the click that pushes the picker screen, once per process — the picker visibly appears late and keystrokes queue behind it. The module already has the lazy-cache half of the fix (`_EMOJI_DATA_CACHE`); it just runs the expensive half on the loop.
- Recommended correction: keep the cache, move the *first* build off the loop — `__init__` composes with an empty grid, `on_mount` runs `self.app.run_worker(get_emoji_data, thread=True, group="emoji-index")` and populates on completion. The file already uses exactly this shape for its recents write (`_save_recent_emoji_off_loop`, `:1029-1051`).
- Size: M (needs a loading state in compose) · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

---
### P2 [D2] — Four one-shot Settings actions do a 40-110 ms config write / forced full reload on the event loop  ·  _slice: UI-settings_
- Where: `:23597` `save_settings_to_cli_config(...)` + `:23614` `load_settings(force_reload=True)` inside `async def _perform_runtime_source_switch` (dispatched via `self.run_worker(coroutine)` at `:23568`, i.e. ON the loop); `:28279` `SettingsConfigAdapter().save_sections(section_values)` in the NETWORK branch of `action_settings_save_category` (sync action handler); `:10809-10821` `_run_diagnostics_validation`/`_run_diagnostics_reload` (the Validate/Reload BUTTONS at `:27810-27818`) call `adapter.validate_config_file` + `adapter.load(force_reload=True)` synchronously — while the `t` key path for the same category uses the thread worker `_diagnostics_validation_and_reload_worker:10898`.
- Evidence: `save_settings_to_cli_config (write+atomic replace+cache reload): median=63.9ms max=73.3ms`; `load_settings(force_reload=True): median=42.5ms max=49.7ms`. Every other write in this file is off-loop (15 `@work(thread=True)` writers, `asyncio.to_thread` in `_persist_briefing_schedules_gate:27499` and `WebSearchSettings.save` settings_web_search.py:331).
- Why it matters: the TUI freezes ~0.05-0.1 s per click on those paths; CLAUDE.md's own rule is "Workers for operations >100ms". Bounded to one click each, so P2.
- Recommended correction: `await asyncio.to_thread(...)` in `_perform_runtime_source_switch` (same shape as `:27502`); route the NETWORK save and the two Diagnostics buttons through the existing thread workers (`_settings_save_appearance_worker:29806` shape; `_diagnostics_validation_and_reload_worker` already exists — the buttons just don't use it).
- Size: S · ADR: no · Confidence: verified (timing) / verified (call sites)
- Pinning test: none
- Already covered: none
### P2 [D2] — Opening any enhanced file picker costs ~9.6 ms of synchronous config reads on the event loop, because `RecentLocations` was left eager when its sibling `BookmarksManager` in the same file was made lazy  ·  _slice: W-top_
- Where: `tldw_chatbook/Widgets/enhanced_file_picker.py:52-57` (`RecentLocations.__init__` → `load_from_config()`), `:1308-1316` (`EnhancedFileDialog._get_last_directory`, called from `__init__` at `:1279`), and a third read at `on_mount` → `_update_bookmarks_list` → `BookmarksManager._ensure_loaded` (`:162-165`). 37 construction sites repo-wide.
- Evidence:
  ```
  cd $WT && source env.sh && PYTHONPATH=$WT $PY -
    get_cli_setting warm x20:        5.39 ms/call
    RecentLocations() x10:           5.55 ms/construct
    BookmarksManager() x10:          0.000 ms/construct   <- task-261 made this one lazy
    EnhancedFileOpen.__init__ x10:   9.58 ms each
  grep -rn "EnhancedFileOpen(|EnhancedFileSave(|EnhancedSelectDirectory(" tldw_chatbook/ | wc -l  -> 37
  ```
- Why it matters: `BookmarksManager`'s docstring (`:117-127`) says in so many words that per-construction config I/O was a stall hazard and was deferred by task-261 — but the fix stopped at one of the two managers. Every picker open still pays two config reads inline on the click handler (a third at mount), so the deferral bought ~0 for the common path.
- Recommended correction: give `RecentLocations` the same `_ensure_loaded()` shape `BookmarksManager` already has (`self._recent: Optional[list] = None`; load on first `get_recent`/`add`), and move `_get_last_directory()` out of `__init__` to the point where `effective_location` is actually needed — or fold both into the single deferred read the file already has machinery for. Canonical home: this file; no new helper.
- Size: S · ADR: no · Confidence: verified (timings above)
- Pinning test: none found asserting eager load. `Tests/UI/test_enhanced_file_dialog_mount.py` exists but does not pin construction cost.
- Already covered: none (task-261 covered `BookmarksManager` only)

---
### P2 [D2] — The Notes-import review's in-place fast path runs on snapshots that have no collision, throws, and full-recomposes anyway  ·  _slice: W-library_
- Where: `library_note_import_canvas.py:589-596` (`collision_only` omits `collision_kind`) and `:620-657` (`_sync_collision_controls`, `except Exception: refresh(recompose=True)`).
- Evidence: `<SCRATCH>/probe_ingest_sync_state.py::test_B` (review snapshot, `collision_kind=""`, same items/page, changed status line) → `recompose calls taken via the COLLISION except path: 1`. `_sync_collision_controls`'s first statement queries `#note-import-collision-heading`, composed only when `state.collision_kind` is truthy (`:894`).
- Why it matters: the outcome matches today, but `except Exception` around 30 lines of `query_one` means a renamed id or field silently degrades the path whose whole purpose (per `sync_state`'s docstring) is not replacing an actively edited input.
- Recommended correction: add `and bool(snapshot.collision_kind)` to `collision_only` at `:589`; narrow both `except Exception` to `except (NoMatches, QueryError)`.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none
### P2 [D2] — The re-chunk thread worker keeps `exit_on_error=True` and does work outside its `try`  ·  _slice: W-library_
- Where: `library_search_rag_panel.py:283` (`@work(thread=True, group=RECHUNK_WORKER_GROUP, exclusive=False)`), body `:309`, `:337`, `:343`, `:344-348`.
- Evidence: `textual._work_decorator` default is `exit_on_error: bool = True`, and `Worker._run` does `if self.exit_on_error: app._handle_exception(WorkerFailed(...))`. `<SCRATCH>/probe_rail_detached_app.py` → `thread .app while MOUNTED: A(...)` / `after remove: _parent = None` / `thread .app after REMOVE: RAISED NoActiveAppError`. The panel is not a resident canvas (`library_canvas_sync.py:424-428` lists only media/notes; `library_screen.py:11892` rebuilds it per route and the host `remove_children`s the old one), so navigating off Search/RAG mid-run detaches it. `format_rechunk_summary(summary)` at `:344` plus the two `call_from_thread`s at `:345-348` sit outside every `try`.
- Why it matters: today a mid-run navigation silently loses the summary line and the toast; a non-dict return from `rechunk_legacy_media` on a still-mounted panel is an uncaught worker exception, i.e. `app._handle_exception`. The identical shape one file over passes `exit_on_error=False` (`library_media_viewer.py:838`).
- Recommended correction: `exit_on_error=False` on the decorator and move `:344-348` inside the `try`/`else`.
- Size: S · ADR: no · Confidence: verified (mechanism + detach behaviour); inferred (the app-exit consequence)
- Pinning test: none (`grep -rn "exit_on_error" Tests/` → no hit on this worker)
- Already covered: none
### P2 [D2] — Three coroutine workers do their file writes / sqlite reads synchronously on the event loop while their sibling paths thread the same operation  ·  _slice: UI-personas_
- Where: `personas_screen.py:5688-5692` (`_dictionary_export_worker`: `exports_dir.mkdir`, `temp.write_text(body)`, `temp.replace(target)`); `:13537-13545` (`_export_expression_set`: `mkdir`, `temp.write_bytes(blob)` of a freshly built zip, `replace`); `:2109-2114` (`_apply_pending_character_conversation_link`, runs inside the `personas_initial_load` coroutine worker: `db.get_local_authority_id()` and `db.get_character_card_by_id(...)` are called inline while lines 2125/2130 of the same function use `asyncio.to_thread`). Contrast `_lore_export_worker:6755-6756`, which threads its `write_text`/`replace`.
- Evidence: read only (structural: no `await` between the calls; `run_worker(<coroutine>)` runs on the loop — AST census shows `thread=True` on 0/64 calls). `grep -n "def get_character_card_by_id" -A40 DB/ChaChaNotes_DB.py` → `SELECT * FROM character_cards WHERE id = ? AND deleted = 0` via `execute_query` (real sqlite). Stall duration not measured — see UNVERIFIED.
- Why it matters: while the zip/JSON write or the card read runs, the app processes no input or paint; on a slow disk or a large expression set this is a visible freeze, and the deep-link read sits on the mount path.
- Recommended correction: `await asyncio.to_thread(...)` around the mkdir+write+replace triple exactly as `_lore_export_worker` already does (one `_atomic_export_write` helper on the screen serves all three — see P3 D4 below); `to_thread` the two DB reads at 2110/2114.
- Size: S · ADR: no · Confidence: verified (structure) / stall magnitude inferred
- Pinning test: `Tests/UI/test_personas_dictionaries.py::test_export_json_writes_file_and_reports_path` asserts the file lands in `exports/`, not where the write runs — stays green after the fix.
- Already covered: none (task-1320 "Move screen mount IO off the App message pump", In Progress, covers the mount pump, not these worker bodies)
### P2 [D2] — `ListDirectoryTool` neither clamps `max_depth` (its own schema says `maximum: 5`) nor caps the scan; the 100-entry cap is applied after the full recursive walk  ·  _slice: TOOLS-MCP_
- Where: `tldw_chatbook/Tools/file_operation_tools.py:413` (`max_depth = kwargs.get("max_depth", 2)` unclamped), `:469-520` (recursive walk stat()s every entry), `:550` (`entries[:100]` after the fact); contrast `local_tool_impls.py:78,272-275` (`MAX_SCAN_ENTRIES = 10_000`)
- Evidence: probe `ListDirectoryTool().execute(directory_path=d, recursive=True, max_depth=50)` → `max_depth=50 accepted -> depths seen: [0..12] total_entries 24`
- Why it matters: a model can request `recursive=True, max_depth=50` on a bound workspace root and the tool `stat()`s and appends every entry of a node_modules/build tree in-process before truncating to 100 — the exact failure `MAX_SCAN_ENTRIES` was added to the sibling family for.
- Recommended correction: `max_depth = max(1, min(int(max_depth), 5))` and a scan counter that stops at `MAX_SCAN_ENTRIES` with a truncation notice; S.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/Tools/test_local_tool_impls.py::test_list_directory_caps_the_scan` pins the SIBLING; none for this class
- Already covered: none
### P2 [D2] — `ServerUnifiedMCPService` re-runs the full access-context resolution (1 status probe + 1 bootstrap + 7 endpoint probes = 9 sequential client round-trips) and a sync JSON store write before EVERY server mutation and several plain reads — 65 call sites  ·  _slice: TOOLS-MCP_
- Where: `tldw_chatbook/MCP/server_unified_service.py:2823-2838` (`_revalidate_mutation_scope` → `resolve_access_context`), `:64-130` (`resolve_access_context`: `client.get_status()`, `_bootstrap_access_context`, `_probe_section_capabilities`), `:2852-2896` (seven awaited probes), `:111,3192-3216` (`_persist_target_status` → `target_store.update_target_status` → `mcp_sources.write_json`, synchronous file write on the event loop); reads that pay it too: `get_governance_pack_detail:2161`, `list_governance_pack_upgrade_history:2190`, `list_workspace_set_members:2567`
- Evidence: `rg -c '_revalidate_mutation_scope\(' tldw_chatbook/MCP/server_unified_service.py` → 65; the probe count is the seven `await _probe(...)` lines at `:2877-2885`
- Why it matters: one Hub click on a remote target costs 10 HTTP calls before its own request, serially; on a slow link each mutation stalls for seconds and the target-status JSON rewrite blocks the Textual loop each time. `_browse_cache` never short-circuits this path.
- Recommended correction: cache the resolved context per `(server_id, scope, scope_ref)` with a short TTL (or reuse the last `resolve_access_context` result when the scope selection is unchanged) and move `_persist_target_status` off-loop; M.
- Size: M · ADR: no (no ADR title matches `revalidat|mutation scope|unified mcp`) · Confidence: verified (unconditional code path; network cost not measured)
- Pinning test: `Tests/MCP/test_control_plane_permissions.py::test_session_approval_revalidates_profile_digest_under_fence` pins the PERMISSION revalidation, not this scope re-resolution
- Already covered: none
### P2 [D2] — `_durable_context_snapshots` performs one synchronous sqlite point read per active-path message on the event loop, on every dispatch  ·  _slice: CHAT-controller_
- Where: `:22743-22862` (`version = version_reader(persisted_id)` inside the per-message loop, `:22775`); `version_reader` = `Chat/chat_persistence_service.py:923 get_message_version` → `self.db.get_message_by_id_without_blob(message_id)` (one `SELECT … FROM messages WHERE id=?` each). Callers: `:23940` `_apply_conversation_memory_preflight` (every `_stream_assistant_response_inner` dispatch with a `ConsoleProviderResolution`, `:24545`), `:23577` `context_control_inputs` (settings UI, `prepare_speculative_voice_attempt` `:8409`, `compact_context_now` twice), `:23540` `_project_session_effective_memory` (context preview), `:23601`/`:23850`, `:23649` `undo_context_memory_reset` (loops **all** sessions).
- Evidence: `grep -n '_durable_context_snapshots(' console_chat_controller.py` → 8 call sites listed above; `get_message_version` body quoted from `chat_persistence_service.py:923-942`. Cost not measured (read only).
- Why it matters: TASK-22205 offloaded the two per-send `BEGIN IMMEDIATE` transactions because "tens of ms steady-state, up to the 15 s busy timeout under write-lock contention" — this loop re-exposes the loop to N such reads per send (N = transcript length), each subject to the same busy-timeout stall (e.g. during the messages_fts backfill window named in that task).
- Recommended correction: add a batched `get_message_versions(ids) -> dict[str,int]` to `ChatPersistenceService` (one `SELECT id, version, deleted FROM messages WHERE id IN (…)`) and call it once per snapshot; or run the existing loop through `self._run_durable_db_call` (already handles the `:memory:` case). M.
- Size: M · ADR: no · Confidence: inferred (magnitude) — literal command to settle: `cd <worktree> && source <SCRATCH>/env.sh && $PY - <<'EOF'` building a file-backed store with ~200 persisted messages and timing `controller._durable_context_snapshots(session_id)` with `time.perf_counter()` (Tests/Chat/test_console_rewind_summarize.py's fixtures build such a controller).
- Pinning test: `Tests/Chat/test_console_rewind_summarize.py` / `test_console_context_compaction.py` exercise the snapshots (behaviour), none pins the read count.
- Already covered: none.
### P2 [D2] — `_fetch_chunk_templates` is a coroutine worker, so its unbounded `SELECT *` runs on the event loop  ·  _slice: W-library_
- Where: `library_ingest_canvas.py:1960-1964` (`run_worker(self._fetch_chunk_templates(), group=…, exclusive=True)`), awaiting `list_templates(mode="local")` at `:1990`; fires from `on_show`, i.e. every entry to the Import rail path.
- Evidence: `<SCRATCH>/probe_ingest_worker_thread.py` → `EVENT LOOP THREAD: MainThread` / `DB READ RAN ON THREAD: MainThread` / `SAME THREAD: True`. Callee chain is synchronous SQLite: `RAG_Admin/rag_admin_scope_service.py:209` → the plain `def` at `RAG_Admin/local_rag_admin_service.py:221` (called before `_maybe_await` sees it) → `Chunking/chunking_interop_library.py:118-127` `conn.execute("SELECT * FROM ChunkingTemplates WHERE deleted = 0")` — whole table, no LIMIT.
- Why it matters: the one worker on this canvas gives no isolation at all, and the block scales with the template table.
- Recommended correction: hop the sync call (`asyncio.to_thread` inside the service) or make the method sync and run it with `thread=True`, marshalling back via `call_from_thread` — the shape `library_media_canvas.py:651` already uses.
- Size: S · ADR: no · Confidence: verified (thread identity); the block DURATION is unmeasured — settle by timing `get_all_templates` against a real `MediaDatabase` with N templates
- Pinning test: `Tests/UI/test_library_ingest_template_picker.py:159` pins only that the populate is off the mount path, not which thread it lands on.
- Already covered: none
### P2 [D2] — `attachment_core`'s three config readers memoize nothing and each costs ~7 ms; a single image attachment fans out to ~5 of them  ·  _slice: CHAT-rest-1_
- Where: `Chat/attachment_core.py:84-102 _chat_images_setting` (shared reader), `:105 supported_image_formats`, `:147 max_image_bytes`, `:164 image_resize_max_dimension`, `:184 attachment_filter_specs`. Per-attachment callers: `Utils/file_handlers.py:112` (`ImageFileHandler.can_handle`, once per candidate file), `:600-606`, `Event_Handlers/Chat_Events/chat_image_events.py:67, 76, 194, 206`.
- Evidence: `cd $WT && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY -c "..."` (warm, n=20 each) →
  ```
  get_cli_setting('chat','images')          7.415 ms/call
  supported_image_formats()                 7.563 ms/call
  max_image_bytes()                         6.923 ms/call
  attachment_filter_specs()                 7.457 ms/call
  ```
  Independently reproduces the sibling's "~11 ms per `get_cli_setting` even when cached" on this machine.
- Why it matters: per-attachment, not per-tick, so bounded (~35 ms for one image) — but it is 5 process-wide config handshakes to answer three questions that cannot change within one attachment operation, and `supported_image_formats()` additionally re-runs `svg_rendering_available()` (which imports `optional_deps` and probes cairosvg) each time.
- Recommended correction: not a per-call cache in `attachment_core` (config is mutable at runtime); resolve the three values once per attachment operation at the entry point (`process_attachment_path` / `chat_image_events`' entry) and pass them down. If a cache is preferred, the fix belongs in `config.get_cli_setting`, where the ~7 ms actually is — that is the sibling's finding, not this one.
- Size: M · ADR: no · Confidence: verified (cost measured; the "~5 per attachment" fan-out is counted statically from the call sites above, not instrumented)
- Pinning test: none
- Already covered: none — the brief's "config reads are cache-backed" known-deliberate item covers the *caching design*, not this call count
### P2 [D2] — `get_cli_setting` costs 12.6 ms per call because the cached read sits behind a per-call storage-admission handshake (~24 `posix.open` per read); this slice pays 3-6 reads per provider/summarization call  ·  _slice: LLM_
- Where (in-slice payers): `Summarization_General_Lib.py:841,851,905,907,919,929` (openai: 6 reads/call), `:1018,1072,1112,1114,1147` (anthropic: 5 reads, the last three INSIDE the retry loop — the excerpt's get_cli_setting_hot rows), and the same shape in every other `summarize_with_*`; `LLM_API_Calls.py:1139 _anthropic_caching_enabled` + `:1173 _cache_control_marker` (called 2-3× per Anthropic send), `:554 _openai_cache_key_enabled`. Cause (outside this slice): `config.py:8447 get_cli_setting` → `config.py:6253 load_cli_config_and_ensure_existence` → `Backup_Recovery/config_participants.py:400 wrapped` → `raw_participants.py:451 _scope` → `storage_admission.py:854 _acquire_storage` (0.279 s of 0.354 s for 20 calls) + `bootstrap.py:33 pinned_directory` / `Utils/private_paths.py:340 _open_verified_parent` (9,640 `posix.open` for 20 calls).
- Evidence: `$SCRATCH/llm_timing_probe.py` → `get_cli_setting(anthropic_api.api_retries): median=12600.4us max=26616.4us (3 reads/retry-attempt => ~37801us/attempt)`; cProfile of 20 calls (isolated env, warm cache) → top-of-stack lines listed above. Comparison: `recovery_review._ordinary_operation()+close` (the per-provider-call admission) = 5.8 ms median.
- Why it matters: the "config reads are cache-backed" known-deliberate holds for the DICT, not the cost — chunked summarization pays ~65-75 ms of config reads per chunk before any network I/O; an Anthropic Console send pays ~25-38 ms. The retry-loop placement (`:1112-1147`) itself is moot in-slice (the loop only re-enters after a 5 s sleep), so the three excerpt rows are P3 as placed; the cost is the wrapper.
- Recommended correction: outside this slice (ENTRY-config / Backup_Recovery reviewer): admit once per `load_cli_config_and_ensure_existence` cache generation, not per `get_cli_setting`. In-slice, hoisting the reads above the loop (S) is cosmetic until that lands.
- Size: S in-slice / M for the cause · ADR: ADR-126 governs startup admission; whether it intends per-read admission is the config reviewer's call · Confidence: verified (measurement, isolated HOME; production HOME not measured)
- Pinning test: none
- Already covered: not in the "already handled" list; cross-slice hand-off
### P2 [D2] — the Media Reader rebuilds the review-set banner from an unmemoised whole-set load on every viewer sync; measured 3.25 ms per call at the 500-item cap, and the cost is per-CALL not per-item, so every review-set user pays it  ·  _slice: UIM-library_
- Where: `tldw_chatbook/UI/Library_Modules/library_media_controller.py:3913` (`_sync_library_media_viewer_state`, the unchanged-compare) and `:3754` (`_build_library_media_reader`, the route build). Both call the late-bound `_active_review_set_banner`, whose body is `tldw_chatbook/UI/Screens/library_screen.py:31169-31220` → `service.get_active_review_set()` at `:31186`.
- Evidence: `cd $WT && source $SCRATCH/env.sh && PYTHONPATH=$WT $PY - <<EOF` building a real `LibraryCollectionsDB` + `ReviewSetService`, creating a 500-item and a 50-item set, and timing 60 calls each → `50 items: median 2.59 ms  p90 2.93 ms  max 3.42 ms` / `500 items: median 3.25 ms  p90 3.78 ms  max 16.86 ms`. Per-`]`-press call count, traced: `action_library_media_next_item:3241` → `_select_library_media_adjacent_item:3249` → `_walk_active_review_set` (screen `:30953`, load #1) and, when the walk does not consume the key, `_select_library_media_reader_row:2171` → `_sync_library_media_viewer_or_recompose:4295` → `_sync_library_media_viewer_state:3848` → `:3913` (**load #2 — this file's share**). `:3754` is NOT on the in-place sync path: the changed branch at `:3980-4058` patches viewer attributes and calls `viewer.refresh(recompose=True)`; the builder is reached only from compose/route-swap (`library_screen.py:15286`, `library_browse_route_swap.py:204`), so a Reader ENTRY pays a third load.
- Why it matters: the banner is rebuilt from storage on a path whose sibling on the same class is already memoised. `_decorate_with_review_state` (`library_screen.py:16100-16127`) caches `get_active_review_set()` against `service.revision` with a comment saying exactly why ("this runs at every one of the ~30 viewer-flip sync sites — a whole-set load to stamp at most a page"); `_active_review_set_banner`, 15,000 lines later, does the identical load with no cache. The measurement also corrects the mental model: the cost is dominated by `read_transaction` setup, not by the 500 rows, so a user with a 12-item set pays 2.6 ms of the 3.2 ms — this is not a large-set-only problem.
- Recommended correction: reuse the existing `(service.revision, value)` memo shape for the banner. Canonical home is `library_screen.py` beside `_review_done_map_cache`, not this slice — this file's share is the call site. **This confirms and bounds the UI-library P1**; fix it there, not twice.
- Size: S · ADR: no · Confidence: verified (measurement); the per-press call count is verified by call-graph read
- Pinning test: none
- Already covered: UI-library P1 (same defect; this is its measurement and this file's share)
### P2 [D2] — the scope picker's tag vocabulary rebuilds the whole notes keyword-usage table on every tag query, although the expensive part does not depend on the query  ·  _slice: CHAT-rest-3_
- Where: `Chat/scope_picker_listers.py:462-483` (`_notes_tag_usage`: 1 `list_keywords(limit=200)` + up to **200** `get_notes_for_keyword(kid, limit=1000)` calls, each materialising up to 1000 full note rows only to take `len()`), called from `:519-524` inside `build_keyword_tag_lister`'s `_tag_lister(query)`. The `query` is used only at `:528-537`, an in-memory substring filter over the already-built `counts` dict.
- Evidence: `cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY` with a real `CharactersRAGDB` seeded 200 keywords x 20 notes ->
  ```
  caps: vocab=200 usage_sample=1000
  200 keywords x 20 notes each -> _notes_tag_usage took 22.4 ms, 200 tags
  second identical call (no cache): 22.1 ms
  ```
  20 notes/keyword is 2% of the `_NOTES_TAG_USAGE_SAMPLE_LIMIT` ceiling; cost is linear in rows materialised, so a user at the cap pays ~50x that per query.
- Why it matters: `Widgets/Console/console_scope_picker_modal.py:739` debounces and `:749` runs it `exclusive=True, group=TAG_SEARCH_WORKER_GROUP`, so this is off-loop and de-duplicated within a keystroke burst — but every *settled* query re-pays the full scan for a result that is identical each time. The modal also calls `_tag_lister("")` at `:514` on open.
- Recommended correction: compute `counts` once per modal (or memoise it on the closure with a short TTL) and apply only the substring filter per query. The docstring at `:447-449` already names the root cause ("no single-query usage-stats seam exists on the notes side, unlike media's `get_keyword_usage_stats`") — a `get_keyword_usage_stats` equivalent on `ChaChaNotes_DB` is the other, larger fix.
- Size: S (memoise) / M (add the notes-side stats query) · ADR: no · Confidence: **verified** (measured)
- Pinning test: none found.
- Already covered: none
### P2 [D3/D4a] — all five "Security functions for input validation" in `llm_management_events_vllm.py` are dead; one of them re-rolls `Utils/input_validation.validate_port`  ·  _slice: EVENTS_

- Where: `tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_vllm.py:47-149` — `validate_python_path` (50), `validate_host` (81), `validate_port` (104), `validate_model_path` (116), `validate_additional_args` (131), under the banner comment `# Security functions for input validation`.
- Evidence: `rg -n "\bvalidate_python_path\b|\bvalidate_host\b|\bvalidate_model_path\b|\bvalidate_additional_args\b" --glob '*.py' .` → only the five `def` lines; zero callers, zero tests, including under `Tests/`. `rg -n "\bvalidate_port\b" --glob '*.py' .` → the dead one here plus the live canonical `tldw_chatbook/Utils/input_validation.py:1221`.
- Why it matters: the banner asserts the vLLM launch inputs are injection-checked. They are not checked by anything — the real vLLM input gate is `Utils/input_validation.validate_vllm_draft_input` (`input_validation.py:313`). Dead security code that reads as live is worse than none: the next reader adds a field and assumes the file's validators cover it. (The three `re.compile`-per-call sites the mechanical scan flagged at `:57/:87/:91/:122` are therefore *not* a hot-path cost — they are never called at all. Retired as a D2, see Retired.)
- Recommended correction: delete lines 47-149. If any of it is wanted, `Utils/input_validation.py` already has `validate_port` (1221) and `validate_ip_address` (1093); that is the canonical home.
- Size: S · ADR: no · Confidence: verified.
- Pinning test: none.
- Already covered: none.
### P2 [D3] — 2,349 of this slice's 40,109 lines are in modules with zero production importers; only one of the five is in the census that owns the decision  ·  _slice: CHAT-rest-3_
- Where, with the check that resolved each (production = `tldw_chatbook/`, excluding the module itself and `Tests/`):
  | module | lines | production importers | kept alive by |
  | --- | --- | --- | --- |
  | `Chat/console_visual_evaluation.py` | 1286 | **0** | `Tests/Chat/test_console_visual_evaluation.py`; also has a row in the `GatewayCallsiteRecord` census at `Chat/console_trace_provenance.py:384-391` (a *string* path, not an import) |
  | `Chat/console_visual_benchmark.py` | 242 | 1 — and that one is `console_visual_evaluation.py:39`, i.e. transitively dead | `Tests/Chat/test_console_visual_transcript.py` |
  | `Chat/document_generator.py` | 601 | **0** | 6 test files (see the finding above) |
  | `Chat/prompt_template_manager.py` | 138 | **0** — zero mentions anywhere in `tldw_chatbook/` | `Tests/Chat/test_prompt_template_manager.py`; `Docs/Development/Developer_Guide.md:49` still documents it as live |
  | `Chat/server_chat_loop_service.py` | 82 | **0 consumers of `ServerChatLoopService`**, but `Chat/__init__.py:9` re-exports it, so it is imported at **boot** (it appears in `Tests/Performance/boot_budget_snapshots/boot_import_modules.txt`) | `Tests/Chat/test_server_chat_loop_service.py` |
- Evidence: `grep -rn --include='*.py' "<module>" tldw_chatbook/ | grep -v "Chat/<module>.py:"` per row (outputs above); plus `grep -rn --include='*.py' "ServerChatLoopService" tldw_chatbook/ Tests/` -> production hits are only the `__init__.py` re-export and the class definition.
- Why it matters: `console_visual_evaluation.py:833` makes a real `stream_chat` gateway call and is registered in the production gateway-callsite census, so a derived-artifact check now pins a dead module in place. `server_chat_loop_service` costs boot-import time for a class nothing constructs.
- Recommended correction: **none from me** — `task-19571` ("Decide once what to do with 170 unreachable modules, 78 of which still carry a test suite", status `To Do`) owns this as a wire-or-retire policy decision and explicitly names `Chat/console_visual_evaluation.py` (1,285) in its subsystem table. The useful output of this review is the four rows its table does *not* name: `console_visual_benchmark.py`, `document_generator.py`, `prompt_template_manager.py`, `server_chat_loop_service.py`. `server_chat_loop_service` is the one with a cost attached today (boot import) and is a one-line `__init__.py` fix independent of the policy.
- Size: S (drop the `__init__` re-export) / L (the census decision) · ADR: no · Confidence: **verified**
- Pinning test: each row has one; that is the census's own complaint.
- Already covered: **task-19571**
### P2 [D3] — 21 of the 66 modules in `Widgets/` (7,512 of 24,763 lines — 30% of the slice) are imported by nothing in production; ~15 test files exist only to keep them alive  ·  _slice: W-top_
- Where (module, lines, test files that reference it):
  ```
  media_details_widget        1009   1 test     base_components              681   6 tests
  file_extraction_dialog       617   2 tests    voice_input_widget           617   4 tests
  activity_log                 580   2 tests    template_selector            452   1 test
  detailed_progress            354   2 tests    status_dashboard             353   1 test
  document_generation_modal    330   1 test     loading_states               326   2 tests
  dictation_performance_widget 319   1 test     file_picker_dialog           276   3 tests
  enhanced_sidebar             273   0          lazy_widgets                 272   0
  file_list_item_enhanced      263   1 test     toast_notification           247   1 test
  tooltip                      160   0(*)       feedback_dialog              151   2 tests
  chunk_preview                120   0          empty_state                  104   0(*)
  custom_list_items              8   0
  ```
- Evidence (two independent passes, not a bare grep):
  1. AST walk of every `Import`/`ImportFrom` in `tldw_chatbook/**/*.py`, resolving the last path component → these 21 have **0** importers.
  2. Per-module `rg` for the module's own public class/function names (`ast` list) excluding its own file → **0 production hits** for every one. The four that first looked live are name collisions, each checked by hand:
     - `MediaDetailsWidget` — the only prod mention is `Local_Ingestion/local_file_ingestion.py:1875`, a comment reading *"mirrors the **dead** ``MediaDetailsWidget`` writer"*.
     - `file_picker_dialog.create_filter` — `UI/Screens/chat_screen.py:20237` defines its own nested `create_filter`.
     - `tooltip.TooltipMixin` / `HelpIcon` — zero prod hits; the many `tooltip` matches are Textual's own `widget.tooltip` attribute.
     - `base_components.NavigationButton` — `UI/Navigation/main_navigation.py:204` **defines its own** `class NavigationButton(Button)`. Two classes, same name; the live screen uses its local one and `Tests/UI/test_focus_token_parity.py:22` imports the dead one.
     (*) `tooltip` and `empty_state` show many grep hits for the *words*; zero for their classes.
- Why it matters: (a) it is 30% of the package's maintenance surface — every CSS-bundle, focus-token and timer-inventory sweep pays for widgets no screen mounts (`Tests/Architecture/test_timer_path_static_update_inventory.py` already indexes three of them); (b) dead code hides live-looking defects — `dictation_performance_widget.py:172,175` carries an exact copy of the P0 `run_worker` crash above, and `base_components.NavigationButton` shadows the live one; (c) the tests give false confidence that these surfaces are covered.
- Recommended correction: delete in batches, one PR per group, with the existing precedent as the shape — `backlog/tasks/task-1280 - Delete-dead-Widgets-voice_input_button.py-zero-callers...md` did exactly this for a sibling widget. Delete the module *and* its tests together. Caveat: several carry `DEFAULT_CSS`, so run `./scripts/preflight.sh` (CSS bundle sync) in the same commit, and `Tests/UI/test_widget_css_consolidation.py` / `test_css_class_coverage_contract.py` / `test_timer_path_static_update_inventory.py` each need their rows dropped.
- Size: L (many PRs; each individually S) · ADR: no · Confidence: verified
- Pinning test: the ~15 test files above — none states the widget as a *product requirement*; they assert CSS/focus/timer hygiene of whatever exists.
- Already covered: none (task-31584 is a different helper; task-1280 is the precedent, already done)

---

---
### P2 [D3] — 25 of the 62 `Message` classes this package defines are never referenced outside their own file; 10 of those are in *live* modules  ·  _slice: EVENTS_

- Where / evidence: an `ast` pass that collects every class in the slice whose base is `Message` (or a `*Event` base), then counts bare-name references across all of `tldw_chatbook/` excluding the defining file:
```
62 Message/Event classes defined in the slice; 25 with ZERO production references outside their own file
```
Broken down: 9 live in the dead `Chat_Events/chat_messages.py` and 5 in the dead `Audio_Events/dictation_integration_events.py` (both covered by the census), leaving **11 in live modules**:
  - `Audio_Events/recording_events.py` — `AudioRecordingEvent:10`, `RecordingStartedEvent:16`, `RecordingStoppedEvent:24`, `RecordingErrorEvent:33`, `AudioDeviceChangedEvent:49`. The file's *only* live export is `AudioLevelUpdateEvent:41`; the one consumer (`Widgets/voice_input_widget.py:21-28`) imports seven names and takes exactly one from this file. Confirmed: `rg -c "\bRecordingStartedEvent\b" --glob '*.py' --glob '!Tests/**' tldw_chatbook/` → 1 file (the definition), 0 test files, 0 rows in `collect_only.txt`.
  - `Audio_Events/dictation_events.py` — `DictationEvent:10` (base), `DictationPausedEvent:35`, `DictationResumedEvent:41`, `DictationStateChangeEvent:64`.
  - `TTS_Events/tts_events.py:240 TTSStreamingEvent` — never constructed or posted in production; the only references are two **negative** assertions, `Tests/TTS/test_console_audio_cpp_native.py:835` and `:1595`, both `assert not any(isinstance(message, TTSStreamingEvent) …)`. Per the brief's rule 4 that makes its absence a stated requirement — so it is a deliberately-retired contract, not an oversight. Report only; if it is deleted the two assertions go with it.
- Why it matters: `recording_events.py` (55 lines) is 5/6 dead and `dictation_events.py` (87 lines) is 4/10 dead, both behind `from .x import *` in `Audio_Events/__init__.py:2-3` — so a star import is what keeps them nominally "exported". A reader adding a recording feature will wire into `RecordingStartedEvent` and find nothing listens.
- Recommended correction: delete the 9 unused classes in the two Audio_Events modules (`recording_events.py` collapses to the single `AudioLevelUpdateEvent`), and replace the two `import *` lines with explicit names so the next dead export is visible in review. Leave `TTSStreamingEvent` alone unless its two pinning assertions go too.
- Size: S · ADR: no · Confidence: verified.
- Pinning test: only the two negative `TTSStreamingEvent` assertions; none for the Audio_Events nine.
- Already covered: none.
### P2 [D3] — A canvas `sync_state` silently collapses every ingest queue outcome group the user opened  ·  _slice: W-library_
- Where: `library_ingest_canvas.py:514` (`self.expanded_groups: set[str] = set()`) and `:1288` (`sync_state` → `refresh(recompose=True)`).
- Evidence: `<SCRATCH>/probe_ingest_sync_state.py::test_A` → `BEFORE panel id: … expanded: {'failed:boom'}` / `AFTER panel id: … expanded: set()` / `SAME OBJECT: False`. `expanded_groups` is panel-local (writers: `library_ingest_controller.py:2841-2844`, `:2901`); nothing seeds it from render state, while the sibling disclosure flag `tooling_detail_expanded` DOES round-trip through state (`library_ingest_state.py:1364`, `:3442`). Reachable from any checkbox/select change (`library_screen.py:27896`) and the backend switch (`:27717`).
- Why it matters: "the expansion must survive" is already a pinned requirement one path over (`Tests/UI/test_library_crit9_import.py:480 test_dismissing_the_leading_member_keeps_the_group_expanded`); this is the same requirement failing at the canvas-recompose boundary.
- Recommended correction: seed from state the way `LibraryIngestPreflightSummary.__init__:311` already does — add an `expanded_queue_groups` field to the form/state and read it in `LibraryIngestQueuePanel.__init__`.
- Size: M · ADR: no · Confidence: verified
- Pinning test: the crit9 test pins the sibling path only. · Already covered: none
### P2 [D3] — A policy-denied profile loses the legacy-chunk report AND the Re-chunk control, with no log line and no notice  ·  _slice: W-library_
- Where: `library_search_rag_panel.py:142-145` — `try: payload = await get_diagnostics(mode="local") / except Exception: return`.
- Evidence: `RAG_Admin/rag_admin_scope_service.py:312` calls `self._enforce_policy(self._admin_action_id(mode, "observe"))`, which raises `PolicyDeniedError` (`runtime_policy/types.py:116`); a shipped caller reaches this on every `on_show`. The sibling worker in the same file (`:328-334`) catches `PolicyDeniedError` explicitly and notifies the user. `_apply_legacy_chunk_report:177` drives `button.display` off the report, so a denial also hides the Re-chunk control.
- Why it matters: "nothing to re-chunk" and "you are not allowed to look" paint identically, and nothing is logged to tell them apart.
- Recommended correction: `except PolicyDeniedError: logger.debug(...)` and log the broad catch too. No UI change needed.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none · Already covered: none
### P2 [D3] — God module whose size ratchet is red on dev and collected by the core-tests job: 25,215 lines / 762 methods against a 16,966 / 563 budget (and 20,099 / 633 for the task-22507 base)  ·  _slice: UI-chat_
- Where: whole file; `Tests/Architecture/test_screen_size_ratchet.py:77` and `:894`; `.github/workflows/test.yml:121`.
- Evidence: `cd <worktree> && source <SCRATCH>/env.sh && $PY -m pytest Tests/Architecture/test_screen_size_ratchet.py -q -p no:cacheprovider` → `3 failed, 2 passed` — `test_screen_does_not_grow_past_its_budget[chat_screen.py]`, `[library_screen.py]`, `test_task_22507_4_does_not_worsen_chat_screen_base` (`assert 25215 <= 20099`). `test.yml:121` core-tests: `pytest Tests --ignore=Tests/UI …` (Tests/Architecture is collected; no marker skip in the test). `grep -rl "16966\|ratchet" backlog/tasks/` → no open task names the red.
- Responsibility clusters (line ranges, all read): imports 1–696 · module constants/helpers 697–1757 (turn-undo planner 970–1093, inspector exchanges loader 1585–1696, `_ControllerState` descriptor 1698, `_active_lineage_rows` 1021) · bindings/actions/focus 1758–2160 · left-rail/terminal/inspector-section events 2173–2810 · **settings-durability + default-intent state machine 2804–3320** · **Conversation-settings modal open/suspend/return handoff 3318–4190** · **roleplay projection persistence drain 4192–4780** · workbench help/F6 panes 4782–5245 · session switcher + model popover 5245–5605 · settings submission durability + default recovery 5606–6100 · palette actions 6101–6465 · row/workspace action menus + markdown export 6465–7240 · class attrs + `__init__` 7239–7560 · config memo / provider intents / vLLM handoff 7560–8340 · context estimate / control state 8338–8500 · **Environment/Tasks section + focus restore 8512–9330** · settings summary / agent section sync 9323–9620 · provider-selection derivation + runtime-handle properties 9626–10000 · store/controller/gateway ensure + view hooks 10000–10500 · **~60 proxy properties to controllers 10528–10860** · TTS/voice/impersonate/composer menu 10854–11340 · hands-free/realtime delegations 11341–11470 · collapse toggles 11466–11620 · control state + cost chip (`_build_console_cost_state` 240 lines) + timers 11621–12000 · retrieval scope / chips / inspector push 12002–12500 · character avatar render 12502–12800 · rail prefs persistence + onboarding flags 12792–13320 · **rail state/preferences/visibility 13320–14020** · workspace context sync 14022–14180 · inspector state build 14187–14400 · dictionary/world-book appliers + 4 attach/detach workers 14390–14690 · inspector rows / live-work cards 14684–15100 · readiness copy / blockers / setup modal 15015–15650 · live-work strip/swap/center builder 15649–16070 · **`compose_content` 16071–16632 (560 lines)** · on_mount / attach reconciliation / resume startup 16633–16960 · on_unmount 16954–17010 · state serialize/restore + attachment stash 17011–17430 · handoff consumers 17433–17690 · citation counts 17683–18030 · transcript fingerprint + `_sync_native_console_transcript` 18029–18345 · run copy / mode bar / sync maintenance 18355–18500 · **`_sync_native_console_chat_ui` + tabs + 0.2 s poll 18496–18810** · send pipeline 18811–19130 · **slash commands 19131–20070** · collapse/stop/attach/paste/clipboard/chatbook 20071–20460 · change review / turn undo / approval focus 20456–20890 · canvas bridge 20892–21310 · summarize range 21327–21500 · control-bar sync / coalescing / config-sync / popup 21497–21960 · composer undo/redo + action state 21956–22270 · resize / rail-collapse notice / focus frame 22268–22450 · **`on_key` 22447–22690** · selection / side-chat / review notes / paste / mouse 22690–23300 · compact shell sync 23294–23390 · suspend/resume 23396–23660 · watchlists ops 23656–23860 · task cards / questions / park approvals / run toasts / agent chat create 23853–24460 · dead `on_button_pressed` 24462–24662 · sidebar state persistence 24664–24843 · dead `_restore_collapsible_states` 24844–24870 · live `on_button_pressed` 24893–25079 · collapsibles 25089–25206 · lazy modal loader 25209.
- Why it matters: the ratchet's own docstring says a red budget "defeats the entire mechanism"; the file grew ~8,250 lines past it. Either the Core Tests job is red on dev and PRs are landing anyway, or the check is not required — both mean the governance is not governing.
- Recommended correction (M, then per-subsystem L): re-measure and decide (the docstring forbids raising) — the honest states are "raise once with a written waiver" or "cut below budget". Extraction candidates that own no pixels and can move verbatim under `backlog/docs/library-decomposition-recipe.md` §1/§2: the settings-durability/default-intent machine (2804–3320 + 5569–6100 ≈ 1,050 lines, pure state over `app_instance` attributes), the Conversation-settings suspend/return handoff (3318–4190), the roleplay persistence drain (4192–4780), and the Environment focus-restore cluster (8512–9330). Cite the recipe's §17 size governance; do not redesign.
- Size: M/L · ADR: yes — `backlog/docs/library-decomposition-recipe.md` (§1 per-subsystem PR series, §2 field-ownership script, §17 size governance) governs the shape; no ADR in `adr_list.txt` names chat_screen's size · Confidence: **verified** (the red + collection), inferred (cluster sizes are my line counts; whether Core Tests is a *required* check is UNVERIFIED)
- Pinning test: `Tests/Architecture/test_screen_size_ratchet.py` (red).
- Already covered: none found; task-2902/task-26834 are perf tasks, not size.
### P2 [D3] — God module: 29,048 lines, 470 methods on one class, 831-line `__init__`, 1,575-line `_submit_draft_body`, 963-line `_run_agent_reply`, and no size governance  ·  _slice: CHAT-controller_
- Where: whole file; cluster map in Coverage above. Longest bodies: `_submit_draft_body` 8800–10375, `__init__` 3758–4588, `_run_agent_reply` 26259–27221, `resume_durable_postcommit` 11206–11883, `_apply_conversation_memory_preflight` 23892–24411.
- Evidence: `grep -cE '^    (async )?def '` → 470; `grep -cE '^class '` → 32; `grep -cE 'except Exception'` → 152; `ls Tests/Architecture | grep -i ratchet` → `test_library_modules_size_ratchet.py`, `test_screen_size_ratchet.py`; `grep -ln 'console_chat_controller\|Chat/' Tests/Architecture/*ratchet*.py` → (none). `grep -rliE 'console_chat_controller' backlog/tasks | xargs grep -liE 'split|decompos|size.?ratchet|god.?module'` → (none).
- Why it matters: 60 synchronous `self.store.<persisting call>` sites (`grep -cE` of append/persist/mark/create_sibling/record_trace_event/set_message_usage) and 25 `asyncio.to_thread` sites live in one file with no per-subsystem field ownership; every review (this one included) has to re-derive which of the ~70 `__init__` attributes each cluster owns.
- Recommended correction: out of scope to redesign here — apply `backlog/docs/library-decomposition-recipe.md` §1 (per-subsystem PR series), §2 (field-ownership script, ≥2-subsystems rule), §3 (monkeypatch-name routing — this file is patched by name in dozens of tests), §17 (add a `_BUDGETS` row for `Chat/console_chat_controller.py` in a new sibling ratchet, since neither existing ratchet covers `Chat/`). Candidate first extractions with the fewest shared fields: the module-level review-hook builders (1626–3040, no `self`), project-instruction authority (827–1625, no `self`), the settings rebase (12648–12885, pure), and the interrupt bridges (13382–17874, already mostly delegated to `InterruptRoundHost`).
- Size: L · ADR: yes (new — a decomposition ADR in the shape of the Library series) · Confidence: verified
- Pinning test: none (no ratchet).
- Already covered: none (task-1378/task-31202 govern `settings_screen.py`, not this file).
### P2 [D3] — God module: one class, 24,180 lines, 15 responsibility clusters  ·  _slice: DB-chacha_
- Where: `CharactersRAGDB` `:709–23951` (+ module helpers `:1–708`, `TransactionContextManager` `:23952–24180`). Clusters with ranges: (1) module-level validators/authorizations/SQL splitters `:1–708`; (2) schema + 25 migration SQL literals as class attributes `:720–3305`; (3) connection lifecycle, quiescence, local-authority, backup/integrity `:3306–3988`; (4) `execute_query`/`execute_many`/`transaction` `:3989–4197` + the context manager `:23952–24180`; (5) migration runner primitives + 69 `_migrate_from_*` steps `:4198–8420`; (6) `_initialize_schema` + per-open repair hooks `:8421–8720`; (7) character cards + expression images + FTS `:8951–10402`; (8) conversations (identity normalisation, archive, search/paging/locator, cursor/summary/project-context, delete/restore) `:10403–13165`; (9) messages (adaptive insert, continuation/generation projection, attachments/generation metadata, update/tombstone, exchanges, trajectory, annotations, variants, sync-delete proofs, FTS) `:13166–16493`; (10) generic-item CRUD + keywords/collections `:16494–17461`; (11) notes + owner proofs + links + Library note seams + organization/receipts/dispatch `:17462–19431` (with the Library conversation seams interleaved at `:18466–18874`); (12) link tables + sync_log intents/retention/prune + `backfill_messages_fts` `:19432–21214`; (13) flashcards/decks/templates/assets `:21242–22256`; (14) quizzes/questions/attempts/grading incl. a Levenshtein implementation `:22257–23478`; (15) learning paths/topics/stats + kept briefings/scripts `:23402–23951`.
- Evidence: `grep -nE '^    def ' | wc -l` ≈ 430 methods; `awk` map at review start (line numbers above).
- Why it matters: any change pays a whole-file read/edit cost (this review needed 13 chunks); clusters 13–15 share nothing with 7–12 beyond the connection.
- Recommended correction: out of scope to redesign here — a split must follow `backlog/docs/library-decomposition-recipe.md` (§1 per-subsystem PR series, §2 field-ownership script, §17 file-size governance). The obvious first PRs are the self-contained study cluster (13–15, ~2.7k lines) and the migration steps (5, ~4.2k lines, 27 copies of one 45-line try/verify block — see the P3 below).
- Size: L · ADR: yes (`backlog/docs/library-decomposition-recipe.md` governs the shape; no ADR names this file) · Confidence: verified
- Pinning test: none
- Already covered: none
### P2 [D3] — The size ratchet that governs this file is RED at the reviewed dev commit: 35680 lines / 1327 methods against a pin of 33204 / 1276, in the required CI sweep  ·  _slice: UI-library_
- Where: `Tests/Architecture/test_screen_size_ratchet.py:887` `"tldw_chatbook/UI/Screens/library_screen.py": ("LibraryScreen", 33204, 1276)`; the test's measure is `len(source.splitlines())` and `ast` method count of the class (:900-937).
- Evidence: `$PY -m pytest "Tests/Architecture/test_screen_size_ratchet.py::test_screen_does_not_grow_past_its_budget[tldw_chatbook/UI/Screens/library_screen.py]" -q` → FAILED at `assert lines <= max_lines` (35680 > 33204; methods 1327 > 1276). Provenance: at the pin commit 576daf57bb (2026-09-09) the file measured exactly 33204 lines (pin == measurement, recipe §6); 226 commits touched the file since; the first landed commit past the pin is 6b5f6ec83b "feat(library): Obsidian mode for Notes Import once (task-32129)" (33381 lines, author-dated 09-08 — a concurrent branch landing after the pin, the exact hazard the test's own docstring records). CI: `.github/workflows/test.yml:121` core-tests = `pytest Tests --ignore=Tests/UI -n auto … --num-shards=6`, no `-m` deselection, tests are `@pytest.mark.unit` → the failure is in the required sweep. (`chat_screen.py`'s row fails the same way — outside this slice, reported as a shared red.)
- Why it matters: the governance mechanism the decomposition recipe relies on (§6 "measure after final rebase, lower budgets in the landing PR", §17) is not holding: the screen grew +2476 lines / +51 methods in eight days with the ratchet red, which is precisely "the month in which library_screen.py tripled" the test was written to prevent. The test message says "Do NOT raise the budget to make this pass".
- Recommended correction: recipe §6 — the next Library PR (or a dedicated landing PR) must bring the file back under 33204/1276 or the program owner must record a deliberate re-pin with the numbers; and the PR merge gate for `library_screen.py` changes must run this test (it is in core-tests, so the red is being merged past — check the branch-protection/"cancelled checks" situation the lessons files record).
- Size: M · ADR: no (governed by `backlog/docs/library-decomposition-recipe.md` §6/§17) · Confidence: verified
- Pinning test: `Tests/Architecture/test_screen_size_ratchet.py::test_screen_does_not_grow_past_its_budget[…library_screen.py]` — states the budget as a requirement, currently red.
- Already covered: task-31202 is the settings_screen row; no open task names the library_screen breach (task-32170 / task-32013 move bodies but do not mention the red pin).
### P2 [D3] — Two methods are defined twice in the class body; the first `on_button_pressed` (200 lines, an older divergent row-open implementation) and the first `_restore_collapsible_states` are silently shadowed dead code  ·  _slice: UI-chat_
- Where: `on_button_pressed` at `chat_screen.py:24462-24662` (dead) and `24893-25079` (live); `_restore_collapsible_states` at `24844-24870` (dead) and `25089-25115` (live, byte-identical); nine blank lines 25080–25088 between them (merge artifact).
- Evidence: AST census of `ChatScreen.body` → `on_button_pressed x 2 at [24462, 24893]`, `_restore_collapsible_states x 2 at [24844, 25089]` (every other duplicate name is a `@property`/`.setter` pair). The file's own comment at 22714 names this exact trap ("The two also must not share a method NAME: the second definition would silently replace the first"). The dead copy's `console-workspace-conversation-` branch (24554–24646: task-457(b) loading flag + TASK-717 broken-row marking inline) was superseded by `_workspace.open_console_workspace_conversation` — `workspace.py:4861` carries the same loading/broken logic through wiring 1074/1081 — so behaviour is preserved by the live copy; the first copy is dead, not lost.
- Why it matters: an edit to the first copy is inert with no error (which is how a prior handler died, per the 22714 comment); `Tests/UI/test_console_button_routing.py`'s "19 branches" characterisation and the size ratchet count the wrong method; 230 dead lines on the largest file in the app.
- Recommended correction (S): delete 24462–24662, 24844–24870 and the blank block; add one assertion to `Tests/Architecture/test_screen_size_ratchet.py`'s existing AST walk — method names in the class body are unique once `.setter` pairs are excluded (ruff `F811` would also catch it, but is outside this review's allowed lint selection).
- Size: S · ADR: no · Confidence: **verified**
- Pinning test: `Tests/UI/test_console_button_routing.py` presses real buttons, i.e. characterises the LIVE copy; deleting the dead copy cannot change its result.
- Already covered: none.
### P2 [D3] — `AgentRunsDB._METADATA_COLUMNS` says it is "Every `agent_runs` column EXCEPT `steps`" and is not: the v21 routing-snapshot columns were added to the table and to `SELECT *` but not to this list, so `get_run_metadata()` silently returns a smaller dict than `get_run()`  ·  _slice: DB-rest_
- Where: `tldw_chatbook/DB/AgentRuns_DB.py:1671-1682` (the constant + its comment "Every ``agent_runs`` column EXCEPT ``steps``"); the v21 columns added at `:483-486` (DDL) and `:829-835` (guarded ALTER); readers at `:2600` (`get_run_metadata`) and `:2759` (`list_recent_metadata`-style page). `get_run` at `:2563` uses `SELECT *` and does carry them.
- Evidence (isolated env, temp-file DB, one run created with all four routing fields set):
  `agent_runs columns: 23` · `_METADATA_COLUMNS : 18` · `missing from _METADATA_COLUMNS: ['resolved_base_url', 'resolved_model', 'resolved_params_json', 'resolved_provider', 'steps']`
  `get_run keys` includes all four `resolved_*`; `get_run_metadata keys` does not.
- Why it matters: `get_run_metadata`'s own docstring (`:2568-2589`) instructs callers to "Use this instead of :meth:`get_run` wherever the caller only inspects status/budget/result/task/etc", and two call sites were already migrated on that advice. The next caller that also wants the pinned route gets `KeyError` — or, if it uses `.get()`, a silent `None` that falls back to **live re-resolution of a possibly-edited preset**, which is precisely the failure ADR-147/TASK-32477 added these columns to prevent. Nothing hits it today only because the routing snapshot has its own dedicated reader (`:2694-2707`).
- Recommended correction: either add the four columns to `_METADATA_COLUMNS` (they are small TEXT values; the constant exists to avoid fetching the big `steps` blob, not these) or change the comment to name every excluded column. Then add the drift guard: `rg -n "_METADATA_COLUMNS" Tests/` → no hits, so nothing fails when the next `ALTER TABLE agent_runs` lands. The file already has the precedent — `Tests/DB/test_agent_runs_db.py::test_schema_version_constant_agrees_with_the_version_table` exists for exactly this class of constant-vs-schema drift (class docstring, `:245-256`).
- Size: S · ADR: no (ADR-147 defines the columns, not this constant) · Confidence: verified
- Pinning test: none (`rg -n "_METADATA_COLUMNS" Tests/` → no hits)
- Already covered: none
### P2 [D3] — `Chat/console_conversation_hydration.py` reaches into a UI module for a **private** helper, inverting the very Chat←UI layering the module exists to remove  ·  _slice: CHAT-rest-1_
- Where: `Chat/console_conversation_hydration.py:137-139` (function-body import) and `:173` (call) of `_apply_console_message_attachments` from `UI/Console_Modules/message.py:186`.
- Evidence: `grep -rn "_apply_console_message_attachments" --include='*.py' .` → defined once at `UI/Console_Modules/message.py:186`; used at `UI/Console_Modules/message.py:926` and `Chat/console_conversation_hydration.py:138,173`. Nothing else.
- Why it matters: `UI/Console_Modules/message.py:611-614` states the tree walk was moved into `Chat/` precisely so "the launch wake — which has to hydrate a conversation with no screen at all — shares this policy instead of copying it". Importing back into UI for the attachment-folding half restores a screen-layer dependency on a headless path, through a leading-underscore name whose contract UI is free to change, and the function-body import hides it from any module-level dependency check.
- Recommended correction: move `_apply_console_message_attachments` to `Chat/console_chat_models.py` (which already owns `ConsoleChatMessage` and its attachments tuple) as a public name; UI imports it from there.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none
### P2 [D3] — `Chat/document_generator.py` (601 lines) has zero production callers; the dead code hides a real API misuse that would corrupt note ids the day it is wired  ·  _slice: CHAT-rest-3_
- Where: whole module `Chat/document_generator.py`; the defect is at `:595` — `self.db.add_note(title, full_content, conversation_id)` passes a conversation id into `CharactersRAGDB.add_note`'s third **positional** parameter, which is `note_id`, not a conversation field.
- Evidence:
  1. Reachability: `grep -rn --include='*.py' "document_generator" tldw_chatbook/ Tests/ | grep -i import` -> six hits, **all in `Tests/`** (`Tests/Chat/test_thinking_privacy_surfaces.py:33`, `test_provider_continuation_privacy.py:19`, `test_assistant_generation_state_roundtrip.py:19`, `Tests/Internal_Prompts/test_document_generation_migration.py:45,297`, `Tests/Utils/test_config_nested_settings.py:163`). The only production mention is a *comment* in `Internal_Prompts/document_generation_prompts.py:12-22`.
  2. The defect, run for real:
  ```
  add_note signature: (self, title: str, content: str, note_id: Optional[str] = None, *, cursor: sqlite3.Cursor | None = None) -> Optional[str]
  1st note id returned: conv-abc-123  <- equals the conversation_id, not a generated note id
  note row id: conv-abc-123 title: T1
  2nd note for same conversation RAISED: ConflictError Note with ID 'conv-abc-123' already exists. (Entity: notes, ID: conv-abc-123)
  ```
  So the first generated document silently takes the conversation's id as its note primary key, and the second document for that conversation is refused outright.
- Why it matters: 601 lines carry their own provider table (`chat_with_openai/anthropic/cohere/groq/openrouter/deepseek`, `:104-120`) that duplicates the dispatcher in `Chat/Chat_Functions.py`, three `get_cli_setting` prompt tables, and a `pyperclip` dependency — all maintained by tests alone. The `ConflictError` above is the proof that no caller has ever exercised it. Same module, `:589`: `datetime.now().isoformat()` — **naive local time** written into a user-visible `Generated: ...` note header, the only non-UTC timestamp I found in this slice (another instance of the review's timestamp class, currently unreachable).
- Recommended correction: delete the module and the six test imports, or wire it and fix `:595` to `self.db.add_note(title, full_content)` plus `:589` to `datetime.now(timezone.utc)`. A deletion needs a call, not a code review — the tests that import it assert things about *prompt migration*, not about this class.
- Size: S (delete) / M (wire + fix) · ADR: no · Confidence: **verified**
- Pinning test: `Tests/Internal_Prompts/test_document_generation_prompt_parity.py` and `test_document_generation_migration.py` pin the module's *prompt literals* as a requirement, so the file cannot be deleted without touching them. Nothing pins `create_note_with_metadata`.
- Already covered: none
### P2 [D3] — `ConsoleProviderGateway._chat_api_kwargs` has zero production callers; ten tests pin a kwargs builder the app never runs while its live twin has drifted  ·  _slice: CHAT-bridge_
- Where: `tldw_chatbook/Chat/console_provider_gateway.py:6791-6865` (dead); live twin `:6711-6788 _chat_api_kwargs_from_prepared`; third copy `:5117-5165 _auxiliary_chat_api_kwargs`
- Evidence: `rg -n "_chat_api_kwargs\(" tldw_chatbook/ | rg -v "_from_prepared\(|_auxiliary_chat_api_kwargs\(|def _chat_api_kwargs"` → no output. `rg -n "\._chat_api_kwargs\(" Tests/` → `Tests/Chat/test_console_provider_gateway.py:1666, :3645, :3673, :7195, :7209, :7219, :7234, :7239, :7269, :7270`. The docstring at :6796-6813 cites "the cache-stability test in Tests/Chat/test_console_provider_gateway.py" as the reason its system-row join is deterministic — that test exercises the dead builder. Drift (read): `_from_prepared` adds `chat_template_kwargs` (:6749-6753), local `api_key_resolved` (:6754-6756), moonshot/zai transport + `provider_continuations` (:6760-6768), vllm/local_vllm base_url pinning (:6775-6776), openai `response_format` pinning (:6781-6787); `_chat_api_kwargs` has none. `_auxiliary_chat_api_kwargs` pins `api_base_url` for EVERY provider (:5131) where `_from_prepared` pins a list (:6754-6787), and omits `prompt_caching`/`tools`/`chat_template_kwargs`.
- Why it matters: the Anthropic prompt-cache stability property and the leading-system-row extraction are asserted against code the send path never executes; a regression in `_chat_api_kwargs_from_prepared` cannot be caught by those ten tests, and the three copies already disagree on which providers receive `api_base_url`.
- Recommended correction: delete `_chat_api_kwargs`; port the ten tests to `_chat_api_kwargs_from_prepared` (through `prepare_provider_request`); fold `_auxiliary_chat_api_kwargs` into `_from_prepared` behind the existing `prepared is not None` branch at :4976 (the auxiliary path already uses `_from_prepared` when a prepared request exists).
- Size: M · ADR: no · Confidence: verified
- Pinning test: the ten above state the *dead* builder's behaviour as a requirement (the deletion is a test-port, not a silent change)
- Already covered: none
### P2 [D3] — `LibraryFileNotesWorkspace` is a 8,846-line widget whose git half (94 methods, ~3.0k lines) duplicates the responsibility of the 4,291-line git panel beside it  ·  _slice: W-library_
- Where: `library_file_notes_workspace.py` (one class, `LibraryFileNotesWorkspace`, 280 body nodes / 305 defs) + `library_file_notes_git_panel.py` (4,291).
- Evidence: `ast` bucket of the class's own methods (command in `<SCRATCH>`, run under env.sh):
  `class LibraryFileNotesWorkspace: 280 body nodes, 7246 lines of methods (file 8846)` —
  `git 46/1428`, `push 30/927`, `commit 18/630`, `root 18/563`, `path 21/400`, `editor 12/311`, `reader 5/276`, `save 12/171`, `conflict 9/153`, `tree 7/127`, `folder 5/87`, `search 5/66`, `other 80/1975`.
  Responsibilities in one widget: service/replica lifecycle + root change under a deadline, the folder tree navigator with paging, the editor + autosave state machine, conflict resolution, search, git status/stage/commit/push review flows, the structural-wait patience surface, recovery pairing, and responsive layout.
- Why it matters: 41% of the method lines are the git workflow that already has its own 4.3k-line widget; every File Notes change pays the cost of reading a file where those concerns interleave.
- Recommended correction: the per-subsystem PR series in `backlog/docs/library-decomposition-recipe.md` §1 with the field-ownership script (§2); the seam is the `_git_*`/`_commit_*`/`_push_*`/`_stage_*` method set moving next to `library_file_notes_git_panel.py`. §17's size governance is the guard afterwards. Do not redesign — follow the recipe.
- Size: L · ADR: no (recipe is the settled shape) · Confidence: verified (counts)
- Pinning test: none (no size ratchet covers `Widgets/`; `scripts/` has no widget census — `ls scripts` shows only the four preflight checks)
- Already covered: none for this file (task-1378 / task-31202 are `settings_screen.py`)
### P2 [D3] — `TTSEventHandler` is a 3,606-line, 65-method class with a single 721-line method, and neither it nor `STTSEventHandler` is on any size ratchet  ·  _slice: EVENTS_

- Where: `Event_Handlers/TTS_Events/tts_events.py:590-4196` (`TTSEventHandler`, 3,606 lines, 65 methods, 30 instance attributes set in `__init__`); `Event_Handlers/STTS_Events/stts_events.py:669-2963` (`STTSEventHandler`, 2,294 lines, 62 methods, 20 attributes).
- Evidence (`ast` measurement over the slice; the 15 largest functions):
```
 721  Event_Handlers/TTS_Events/tts_events.py:1798 _generate_tts
 386  Event_Handlers/TTS_Events/tts_events.py:3652 handle_tts_playback
 343  Event_Handlers/note_ingest_events.py:308 handle_ingest_notes_import_now_button_pressed
 245  Event_Handlers/TTS_Events/tts_events.py:2521 _stream_response_via_sink
 237  Event_Handlers/Chat_Events/chat_rag_events.py:1811 get_rag_context_capture_for_chat
 236  Event_Handlers/Chat_Events/chat_rag_events.py:956  resolve_scope_for_session
 232  Event_Handlers/STTS_Events/stts_events.py:1956 _persist_settings
 185  Event_Handlers/STTS_Events/stts_events.py:2679 handle_audiobook_generate
 170  Event_Handlers/TTS_Events/tts_events.py:2846 _play_utterance_legacy_artifact
```
(7 functions >200 lines in the slice.) And the ratchet that exists covers only two files: `Tests/Architecture/test_screen_size_ratchet.py:77` (`chat_screen.py`) and `:887` (`library_screen.py`) — `rg -ln tts_events scripts/ Tests/Architecture/` finds only `scripts/validate_live_tts.py` and the diagnostic-inventory test.
- Responsibilities inside the one class, from the method list: message-level TTS requests, guarded hands-free utterances, console speech-destination resolution and fingerprinting, global voice overrides (issue/peek/consume/prune, 4 methods), per-message cooldown rate limiting (3), generation admission + rate limit, the streaming sink, legacy clip playback and its timeout poll, artifact lifecycle (create/append/secure-delete/cache/release/discard/drain, 9 methods), playback lifecycle, user-facing error-copy mapping, and the maintenance protocol. `CostTracker` (`:480-560`) is already a separate class in the same file and has no dependency on the handler.
- Why it matters: `_generate_tts` alone carries the destination authorization closure, the outcome-reporting closure, the service call, the streaming branch, the legacy-artifact branch, and the metrics — 721 lines with one `try` covering nearly all of it. Nothing stops either file growing further, unlike the two screens that are budgeted.
- Recommended correction: **do not redesign** — follow `backlog/docs/library-decomposition-recipe.md` §1 (per-subsystem PR series) and §17 (controller-file size governance). The cheapest first step that buys the ratchet: add `tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py` and `.../STTS_Events/stts_events.py` rows to `Tests/Architecture/test_screen_size_ratchet.py`'s budget table at their measured current values, so the files can only shrink. The natural first extraction is the artifact lifecycle (9 methods, no TTS-request state) into `TTS/`.
- Size: L (needs the recipe's PR series) · ADR: no (recipe covers it) · Confidence: verified.
- Pinning test: `Tests/Architecture/test_screen_size_ratchet.py` is the mechanism, but it does not cover these files today.
- Already covered: task-1378/task-31202 are the settings_screen equivalents; nothing covers these two.
### P2 [D3] — `app.py` is a 21,050-line god module with no size governance: 25 responsibility clusters (table above), a 4,500-line ingest mixin, a 600-line `__init__`, an 800-line wiring method, a 1,100-line unmount  ·  _slice: ENTRY-app_
- Where: whole file; cluster table above gives ranges.
- Evidence: `wc -l` → 21050; `grep -cE '^    def |^    async def '` → 513; `grep -nE '^class '` → 16 classes; `grep -n 'app\.py' Tests/Architecture/test_screen_size_ratchet.py` → no row; `backlog/docs/library-decomposition-recipe.md` §17 (line 3309) governs `UI/Library_Modules/*_controller.py` only; `grep -inE 'app\.py|god|monolith|decompos' <SCRATCH>/adr_list.txt` → only `097-boot-budget-ratchets.md` (boot census, not structure).
- Why it matters: every edit anywhere in the app's lifecycle, ingest, TTS, navigation or wiring lands in one 600 KB file that whole-file readers cannot open (the console-interaction lesson, 608 KB controller), and nothing stops it growing.
- Recommended correction: follow `backlog/docs/library-decomposition-recipe.md` — §1 per-subsystem PR series in this order: (1) `LibraryIngestQueueMixin` 2585–7089 → `Library/ingest_queue_mixin.py` (it is already a mixin; keep `from tldw_chatbook.app import LibraryIngestQueueMixin` as a re-export — 5 tests import it by that path); (2) the 10 palette providers 1171–2222 → `UI/command_providers.py` (2 tests import `ThemeProvider`/`TabNavigationProvider` from `app`); (3) the `_wire_*` composition 9520–11464 → an `app_wiring.py` function set taking `app`; §2 field-ownership script before each move; §17 add an `app.py` budget row pinned at the post-move count.
- Size: L · ADR: new · Confidence: verified (counts)
- Pinning test: none for size; `Tests/test_call_from_thread_guard.py` and `Tests/Packaging/test_chat_persistence_import_closure.py` pin import-closure/marshalling properties any move must keep.
- Already covered: none
### P2 [D3] — `chat_message.py` + `chat_message_enhanced.py` (1,135 lines) have no production mount site  ·  _slice: W-persona-settings-chat_
- Where: `Widgets/Chat_Widgets/chat_message_enhanced.py` (769), `Widgets/Chat_Widgets/chat_message.py` (366); the app-side consumers `app.py:14666-14690, 14723, 14783, 14812` (`self.query(ChatMessage) + self.query(ChatMessageEnhanced)` in the TTS complete/progress handlers).
- Evidence:
  - `grep -rn "ChatMessageEnhanced(" tldw_chatbook/ Tests/` → exactly one production construction site: `UI/CCP_Modules/ccp_message_manager.py:173`. Everything else is the class definition or `Tests/`.
  - `grep -rn "CCPMessageManager\|ccp_message_manager" --include="*.py" tldw_chatbook Tests` → in `tldw_chatbook/` the name appears only as the class definition (`:26`), its own `logger.bind` (:12) and the package re-export (`UI/CCP_Modules/__init__.py:25,72`). The only construction anywhere is `Tests/UI/test_ccp_handlers.py:415`.
  - `grep -rnE "(^|[^a-zA-Z_])ChatMessage\(" tldw_chatbook --include="*.py" | grep -v ConsoleChatMessage` → only the two class definitions. Nothing mounts the legacy widget either.
  - `grep -rn "CCP_Modules" --include="*.py" tldw_chatbook | grep -v "UI/CCP_Modules/"` → `personas_screen.py:344-348` imports `ccp_character_handler`, `ccp_enhanced_handlers`, `ccp_messages`, `ccp_persona_handler` — **not** `ccp_message_manager`.
- Why it matters: 1,135 lines of widget code that reads as live (it is imported lazily in `app.py` with a `TASK-21103` boot-cost comment, and has two test suites) but can never be mounted; `chat_message_enhanced` is also the reason `PIL`/`rich_pixels`/`textual_image` are kept on a lazy-import leg. The app-side TTS branches that target it are inert.
- Caveat on the "dead" claim: `Tests/UI/test_legacy_entrypoints_retired.py:155-159` lists `ccp_message_manager.py` under `CCP_HANDLER_FILES` — "reused CCP handlers" — and asserts only that its *source text* names `PersonasScreen`. That test does not assert it is wired, and nothing wires it.
- Recommended correction: either wire `CCPMessageManager` into `PersonasScreen` (the intent the test records) or retire both widgets plus the `app.py` query branches through the same `RETIRED_FILES` mechanism `task-577` used.
- Size: M · ADR: no · Confidence: verified (greps above)
- Pinning test: `Tests/Widgets/test_chat_message_enhanced.py`, `Tests/Backup_Recovery/test_recovered_media.py:176` — both construct the widget directly, so they pass whether or not production mounts it.
- Already covered: none
### P2 [D3] — `console_transcript.py` (8,333 lines / one 5,365-line class) and `console_settings_modal.py` (7,601 / one 6,597-line class) are god modules that **no size ratchet governs**  ·  _slice: W-console-2_
- **Where / cluster map (from `ast`, `probe`-free):**
  - `console_transcript.py` — 58 module-level defs/classes, then `class ConsoleTranscript` at **2969-8333, 5,365 lines, 193 methods**. Responsibility clusters inside that one class: windowing/hydration (`3491-4197`), presentation setters (`4198-4738`), pruning (`4842-5058`), thinking/activity projection (`5085-5182`), keyboard text-selection mode (`5412-5623`), mouse drag-selection + floating menu (`5750-6460`), row planning (`6623-7030`), row build/reconcile (`7074-7742`), signatures/caching (`7744-7831`), action rows + overflow menu (`8124-8333`).
  - `console_settings_modal.py` — 18 module-level defs/classes, then `class ConsoleSettingsModal` at **1005-7601, 6,597 lines, 245 methods**: a ~970-line `compose()` (`1589-2593`), focus/scroll/layout (`2837-3253` + `3787-3828` + `4023-4139`), draft snapshot/restore (`2695-2836`), default-durability recovery (`3522-3772`), context-policy controls (`4267-4530` + `7238-7394`), provider/model rebase (`5047-5560`), model discovery + connection probes (`5560-6486`), generation test (`5928-6127`), readiness sync (`6488-6612`), provider/base-URL resolution (`6858-7208`).
- **Evidence:** `Tests/Architecture/test_screen_size_ratchet.py::_BUDGETS` holds exactly two rows — `UI/Screens/chat_screen.py (16966, 563)` and `UI/Screens/library_screen.py (33204, 1276)`. `Tests/Architecture/test_library_modules_size_ratchet.py` covers only `UI/Library_Modules/*_controller.py` (21 paths). Neither names anything under `Widgets/Console/`. `console_transcript.py` at 8,333 lines is larger than most of the 21 governed Library controllers.
- **Why it matters:** the recipe's §17 (`backlog/docs/library-decomposition-recipe.md:3309`) exists precisely because "the files we decompose INTO had no size governance at all". `Widgets/Console/` is the same gap, one package over, and already at screen scale.
- **Recommended correction:** do not redesign — follow `backlog/docs/library-decomposition-recipe.md` §17 option (a): a sibling `Tests/Architecture/test_console_widgets_size_ratchet.py` with exact per-file `_BUDGETS` rows discovered by glob over `Widgets/Console/*.py`, pinned at today's measured counts. Any actual split must follow §1 (per-subsystem PR series) and §2 (field-ownership script).
- **Size:** M for the ratchet (one new test file, no production change) · L for any split · **ADR:** no for the ratchet (§17 already rules the shape) · **Confidence: verified**
- **Pinning test:** the recommendation *is* the pinning test.
- **Already covered:** `task-1378` splits `settings_screen.py` and `task-31202` adds its ratchet row — the same pattern, a different file; neither covers `Widgets/Console/`.
### P2 [D3] — `import tldw_chatbook.RAG_Search.parallel_processor` raises ImportError from a circular import; the module is only importable because every shipped caller happens to import `simplified` first  ·  _slice: RAG_
- Where: `tldw_chatbook/RAG_Search/parallel_processor.py:28` `from .simplified.data_models import IndexingResult` → executes `simplified/__init__.py:35` `from .enhanced_rag_service_v2 import EnhancedRAGServiceV2` → `enhanced_rag_service_v2.py:30` `from ..parallel_processor import (create_embedding_processor, …)` on a partially-initialized module.
- Evidence: `cd <worktree> && source env.sh && PYTHONPATH=<worktree> $PY -c "import tldw_chatbook.RAG_Search.parallel_processor"` →
  ```
  File ".../RAG_Search/simplified/enhanced_rag_service_v2.py", line 30, in <module>
      from ..parallel_processor import (
  ImportError: cannot import name 'create_embedding_processor' from partially initialized module
  'tldw_chatbook.RAG_Search.parallel_processor' (most likely due to a circular import)
  ```
- Why it matters: this is the *third* instance of the exact cycle this package has already been burned by twice — `enhanced_rag_service_v2.py:36-55` documents task-21160 deferring `config_profiles`, and `:64-84` documents deferring `reranker`, both after `task-21102` made `RAG_Search/__init__` lazy and unmasked the latent order dependency. The remaining edge is live today; any new module that imports `parallel_processor` before `simplified` (a script, a test, a CLI entry point) fails at import.
- Recommended correction: if the module survives at all (see the previous finding, which deletes most of it), defer `IndexingResult` the same way its two siblings are deferred — `if TYPE_CHECKING:` for the annotation plus a function-body import at the one runtime use (`parallel_processor.py:480`). If the module is deleted, the cycle goes with it.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none (no test imports `parallel_processor` first — which is exactly why this survived)
- Already covered: none
### P2 [D3] — `import tldw_chatbook.config` performs filesystem work beyond the ADR-126 `admit_startup()`: creates the config file, secures directories, takes the data-root lock, reads a packaged resource, probes optional packages; five module-level aliases it computes are dead  ·  _slice: ENTRY-config_
- Where: `config.py:412` (`installation_client_id()`), `845` (`load_openai_mappings()` → `importlib.resources` read), `5717-5719` (`_default_stt_provider_for_platform()` → `find_spec` ×2), `5729` (`tomllib.loads` of the 2,100-line template), `9742-9745` (`load_cli_config_and_ensure_existence()` + `settings = load_settings()` → `create_private_text`, `secure_private_directory`, `_default_data_root_lock`, `chat_dicts_folder.mkdir` at 3423), dead aliases `9747-9759` (`default_api_endpoint`), `9767-9771` (`APP_CONFIG`, `DATABASE_CONFIG`, `RAG_SEARCH_CONFIG`), `9784` (`APP_CONFIG_GLOBAL`).
- Evidence: fresh dir, `TLDW_CONFIG_PATH=$D/cfgdir/config.toml $PY -c "import tldw_chatbook.config as c; print(c._CONFIG_GENERATION, c.first_profile_created_this_session())"` → `generation 1 first_profile True`; `find $D` before: `cfgdir` only; after: `cfgdir/config.toml`. `rg -n "\bAPP_CONFIG_GLOBAL\b|\bconfig\.APP_CONFIG\b|import APP_CONFIG\b|RAG_SEARCH_CONFIG\b|DATABASE_CONFIG\b|default_api_endpoint" tldw_chatbook Tests --glob '!config.py'` → no importer of the five aliases (the one `default_api_endpoint` hit is a QA test's own attribute). `tldw_chatbook.config.settings` IS a de-facto API: monkeypatched in `Tests/TTS/test_stts_settings_reconfiguration.py` (×8) and imported by `Tests/test_config_stt_provider_probe.py:65`.
- Why it matters: this is why every `python -c "import tldw_chatbook..."` in this repo needs an isolated HOME (the brief's `env.sh`), why `Tests/RuntimePolicy` collection once broke on an import cycle (config.py:976-989), and why the lazily-created settings locks (finding below) happen to be safe. No ADR names it: `Tests/Packaging/test_config_import_closure.py::test_config_import_stays_out_of_feature_packages` pins the import CLOSURE, ADR-097 pins module COUNTS, ADR-126 governs admission — none pins "config is loaded and written at import".
- Recommended correction: (a) delete the five dead aliases now (S); (b) record the import-time load as a decision (keep it — 8 test files and `_current_settings_view` depend on the warm `settings` object) or move the two module-scope loads behind the first `load_settings()` call — either way an ADR, not a drive-by.
- Size: S for (a), L for (b) · ADR: new · Confidence: verified
- Pinning test: `Tests/Packaging/test_config_import_closure.py` (closure only); `Tests/test_config_stt_provider_probe.py:65` (relies on `settings` existing at import)
- Already covered: none
### P2 [D3] — `mcp_workbench.py` is a 6,328-line god module with no size ratchet  ·  _slice: UIM-nav-mcp-persona_
- Where: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py` (one class, `MCPWorkbench`, 130 methods)
- Evidence: `wc -l` = 6328; `grep -rn "mcp_workbench|MCP_Modules" Tests/Architecture/test_screen_size_ratchet.py Tests/Architecture/test_library_modules_size_ratchet.py` -> no output (no ratchet row covers this package).
- Responsibilities carried by the one class: (1) triad assembly + deferred canvas mounting; (2) readiness snapshot collection and CHECKING overlay; (3) the local/server source switch and rail scope model; (4) the Tools catalog derivation (`_collect_hub_tools`, `_local_agent_hub_tools`, `_raw_shell_hub_tool`, `_empty_tools_diagnosis`); (5) the whole permission matrix derivation (`_tool_policy_inventory`, `_capture_permission_render_state`, `_build_permission_rows`, `_build_permission_preview`, `_builtin_permission_matrix_rows`, the cascade map); (6) the prepared Tool-Test admission/nonce/lease state machine (~600 lines); (7) profile CRUD + mcpServers import incl. path validation; (8) the server-mutation/credential-slot panel wiring; (9) audit log + findings; (10) the recovery-review dialog flow; (11) lifecycle dispatch and in-flight bookkeeping; (12) view-state save/restore.
- Why it matters: (5), (6) and (7) are each independently testable pure-ish derivations wedged into a widget; the file has no size governance, so it grows unchecked while `chat_screen.py`/`library_*` are ratcheted.
- Recommended correction: follow `backlog/docs/library-decomposition-recipe.md` §1 (per-subsystem PR series) — the Tool-Test admission machine and the permission-row derivation are the two clean first extractions — and add the package to the controller ratchet per §17 in the same PR that first moves code.
- Size: L · ADR: no (recipe exists) · Confidence: verified
- Pinning test: none (that is the finding)
- Already covered: none
### P2 [D3] — `parallel_processor.py` (585 lines) is a dead subsystem: the only thing it exports that runs is the `ProcessingConfig` dataclass; the pipeline it exists for is documented as "not implemented"  ·  _slice: RAG_
- Where: `tldw_chatbook/RAG_Search/parallel_processor.py` (whole file). Its two constructors are called at `simplified/enhanced_rag_service_v2.py:201` (`create_embedding_processor`) and `:204` (`create_chunking_processor`), under the default `enable_parallel_processing=True`.
- Evidence:
  - `grep -rn "embedding_processor|generate_embeddings_batch|EmbeddingBatchProcessor|process_documents_parallel" tldw_chatbook/ Tests/` → the only assignments of `self.embedding_processor` are `enhanced_rag_service_v2.py:197,201`; **no read of it anywhere**. `self.chunking_processor` is read exactly once, at `:486`, inside a branch whose body is `logger.debug("Parallel batch-indexing pipeline is not implemented; using base optimized path")`. The `generate_embeddings_batch` hits in `rag_service.py:88/1152` and `enhanced_indexing_helpers.py:196/213` resolve to `simplified/indexing_helpers.py:93`, a **different** function.
  - The module's own docstring at `enhanced_rag_service_v2.py:474-484` records that the parallel branch was removed in task-247 because it was "broken in both directions".
- Why it matters, concretely (three live costs for zero function):
  1. **Misleading INFO log on every RAG service construction**: `create_chunking_processor` → `ChunkingBatchProcessor.__init__` → `BatchProcessor._determine_worker_count()` (`parallel_processor.py:133-148`) emits `Using N workers (CPUs: M)`, then `enhanced_rag_service_v2.py:205` emits `Initialized parallel processors with N workers`. Nothing parallel ever runs.
  2. **A live circular-import edge** — see the next finding.
  3. It is untested-and-broken code that a future contributor could re-enable: `ChunkingBatchProcessor.chunk_documents_batch` (`:457`) submits a **closure** (`process_document`, defined inside the method) to `ProcessPoolExecutor.submit` (`:196`); closures are not picklable, so that path could never have run. `EmbeddingBatchProcessor.generate_embeddings_batch` (`:298`) also confuses two index spaces: on a failed batch it does `failed_indices.extend(range(i, min(i + batch_size, total)))` at `:353` where `i` is an index into `tasks` (batches), not into `texts`, and the successful embeddings of surviving batches are appended positionally, so any failure silently shifts every later embedding onto the wrong text.
- Recommended correction: delete `parallel_processor.py` except `ProcessingConfig` (its only live consumer is `config_profiles.ProfileConfig.processing_config`), move that dataclass to `RAG_Search/simplified/config.py` next to the other config dataclasses, and drop the two `create_*_processor` calls and the `enable_parallel_processing` flag from `EnhancedRAGServiceV2`. Deletion, not repair — nothing asks for the feature.
- Size: M · ADR: no · Confidence: verified
- Pinning test: none for the processors themselves; `Tests/RAG_Search/test_reranker_construction.py` patches `select_profile_for_experiment`/`record_experiment_result` on a fake manager but asserts nothing about parallel processing.
- Already covered: none
### P2 [D3] — `pipeline_integration.py` (184 lines) has zero importers anywhere, and one of its branches imports a module that no longer exists  ·  _slice: RAG_
- Where: `tldw_chatbook/RAG_Search/pipeline_integration.py` (whole file); the stale import is `:122` `from ..Event_Handlers.Chat_Events.chat_rag_events_simplified import perform_search_with_pipeline`.
- Evidence:
  - `grep -rn "pipeline_integration|PipelineManager|get_pipeline_manager|reload_all_pipelines|get_available_pipeline_ids" tldw_chatbook/ Tests/` (excluding the file itself) → **no output**. Not re-exported from `RAG_Search/__init__.py` either.
  - `$PY -c "import importlib; importlib.import_module('tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events_simplified')"` → `ModuleNotFoundError: No module named 'tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events_simplified'`. The module was renamed to `chat_rag_events.py`, which still carries the old name in its header comment (`:1`) and in `logger.bind(module="chat_rag_events_simplified")` (`:70`). `perform_search_with_pipeline` lives at `chat_rag_events.py:383`.
- Why it matters: this is the exact "function-body import of a module that no longer exists, which mocked tests never catch" shape from the review brief, and it survives only because the entire module is unreachable. The sibling modules of the same subsystem (`pipeline_builder_simple`, `pipeline_functions_simple`, `pipeline_types`, `pipeline_loader`) ARE live via `Event_Handlers/Chat_Events/chat_rag_events.py:54-55`, so this is a stranded integration layer, not a dead subsystem.
- Recommended correction: delete the file. If any of `PipelineManager`'s legacy-mode mapping is wanted, it belongs in `pipeline_loader.py` next to the loader it wraps.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none
### P2 [D3] — `pipeline_loader.py` (777 lines) is reachable only through the dead `pipeline_integration.py`; the whole legacy `perform_*` pipeline path hangs off an entry point nothing in production calls  ·  _slice: RAG_
- Where: `tldw_chatbook/RAG_Search/pipeline_loader.py` (777), `pipeline_integration.py` (184); downstream `pipeline_builder_simple.py` (766) and `pipeline_functions_simple.py` (914) are reached only via `Event_Handlers/Chat_Events/chat_rag_events.py`'s `perform_plain_rag_search` / `perform_full_rag_pipeline` / `perform_hybrid_rag_search`.
- Evidence (each a grep over `tldw_chatbook/` with the defining file excluded):
  - `PipelineLoader|get_pipeline_loader|get_pipeline_function` → 3 hits, **all** in `pipeline_integration.py` (`:13`, `:22`, `:47`), which itself has zero importers (previous finding).
  - `perform_plain_rag_search|perform_full_rag_pipeline|perform_hybrid_rag_search` → definitions at `chat_rag_events.py:177/225/279`; the only `await` call sites are `chat_rag_events.py:1959/1972/1989`, all inside `get_rag_context_capture_for_chat` (`:1811`).
  - `get_rag_context_capture_for_chat` → called only by `get_rag_context_for_chat` (`:2054`); `get_rag_context_for_chat` repo-wide (all file types, excluding `Tests/`, `Docs/`, `backlog/`) → definition plus two docstring mentions and **no caller**. Its only callers are `Tests/RAG/test_rag_ui_integration.py` and `Tests/RAG/test_rag_dependencies.py`.
  - The live Console/Library retrieval path goes elsewhere: `UI/Console_Modules/retrieval.py:36` and `Chat/console_runtime.py:2476` reach `Library/library_rag_service.run_library_rag_search` → `Library/library_local_rag_search_service` → `RAGService.search`. `capture_console_staged_evidence_for_chat` (`chat_rag_events.py:1659`) authorizes already-staged evidence and runs no pipeline.
- Limit of the method (stated honestly): `pipeline_loader` resolves pipeline functions **by name** from `tldw_chatbook/Config_Files/rag_pipelines.toml` (`:41 function = "perform_hybrid_rag_search"`), so a name-based registry is in play — but the only code that consults that registry is `pipeline_integration.py`, which nothing imports. I did not execute the app to confirm.
- Why it matters: ~2 600 lines of this slice (plus `pipeline_types.py`) exist to serve a chat-RAG entry point no shipped surface calls, while the surface that IS live uses a different engine path entirely. That is a large maintenance surface and a standing source of "which RAG path is the real one?" confusion — `pipeline_functions_simple.py` and `rag_service.py` already carry cross-referencing comments explaining that they are two implementations of the same fusion.
- A concrete cost, if the path IS live: `chat_rag_events.py:346 resolve_hybrid_alpha(hybrid_alpha)` is called with `hybrid_alpha=None` by default, and `pipeline_builder_simple.py:393 resolve_rrf_k(merge_config.get("rrf_k"))` gets `None` because `BUILTIN_PIPELINES["hybrid"]`'s merge step config sets `alpha` (pinned at `chat_rag_events.py:362-365`) but never `rrf_k`. Each `None` sends `fusion.py` into `resolve_active_rag_config()`. Measured, isolated env, warm caches:
  ```
  resolve_active_rag_config():                    10.38 ms/call (warm)
  resolve_hybrid_alpha(None)+resolve_rrf_k(None): 19.11 ms per hybrid search
  with explicit values (the rag_service path):     0.0004 ms
  ```
  `RAGService._hybrid_search` (`rag_service.py:1325-1326`) passes both explicitly and pays 0.4 µs; the pipeline path pays ~19 ms of config re-reading per query, on the event loop. This matches the sibling reviewer's finding that a cached `get_cli_setting`/`load_settings` still costs ~11 ms per call.
- Recommended correction: decide the path's fate first. If it is dead, delete `pipeline_integration.py`, `pipeline_loader.py`, `Config_Files/rag_pipelines.toml`, the three `perform_*` functions and `get_rag_context_for_chat` together (they only keep each other alive). If it is meant to stay, pin `rrf_k` into `BUILTIN_PIPELINES["hybrid"]`'s merge config beside `alpha` — a one-line fix that removes half the 19 ms.
- Size: L (a deletion this size crosses `RAG_Search` ↔ `Event_Handlers` and needs an owner decision) · ADR: no — but see `backlog/docs/library-decomposition-recipe.md` §1 for the per-subsystem PR shape if it is deleted · Confidence: verified for every grep above; **inferred** for the conclusion "unreachable", because of the name-based TOML registry.
- Pinning test: `Tests/RAG/test_rag_ui_integration.py:147-166` calls `get_rag_context_for_chat` directly and asserts its behaviour — it pins the helper, not any product path.
- Already covered: none. (`pipeline_builder_simple.py:375-386` records that TASK-3501 "intentionally retained this pipeline materializer; do not refactor it speculatively" — that ruling is about not refactoring the fusion, not about the path's reachability.)
### P2 [D3] — `simplified/health_check.py` (453 lines) is constructed on every RAG service build and read by nothing  ·  _slice: RAG_
- Where: `tldw_chatbook/RAG_Search/simplified/health_check.py` (whole file); constructed at `simplified/rag_service.py:850`; the only reader is `RAGService.get_health_status()` (`rag_service.py:4341-4348`).
- Evidence: `grep -rn "get_health_status" tldw_chatbook/ Tests/` → exactly two hits, both inside `rag_service.py` (the method definition and its one-line body). `grep -rn "RAGHealthChecker|HealthStatus|ComponentHealth" tldw_chatbook/ Tests/` outside `health_check.py` → **zero**. Not exported from `simplified/__init__.py` or `RAG_Search/__init__.py` (`grep -n health` on both → no match).
- Why it matters: 453 lines of unexercised code that nonetheless runs its constructor on every service build and installs the module global behind the leak above. Its `get_health_sync()` (`:425-430`) would also spin up a fresh event loop with `asyncio.new_event_loop()` + `run_until_complete` — calling it from the Textual event loop thread would raise; nothing calls it, so that is latent rather than live.
- Recommended correction: delete the module and `RAGService.get_health_status()`, or wire it to the RAG admin surface if the diagnostics are actually wanted. Do not leave it half-attached.
- Size: M · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none
### P2 [D3] — `speech_tts_settings_panel.py:5438` is the package's only `run_worker(exclusive=True)` with no `group=`  ·  _slice: W-persona-settings-chat_
- Where: `Widgets/Settings_Widgets/speech_tts_settings_panel.py:5438-5440` — `self.run_worker(self._rebuild_after_custom_id(axis), exclusive=True, exit_on_error=False)`
- Evidence: `grep -rn "run_worker(" Widgets/{Persona_Widgets,Settings_Widgets,Chat_Widgets}` → 28 sites; this is the only one without `group=`. The other 5 in this same file all name a group (`_AUDIO_CPP_PACKAGE_SAVE_VALIDATION_GROUP` :4131, `settings-speech-provider-leave` :5256, `settings-speech-open-lab` :5487/:5925, `_AUDIO_CPP_PACKAGE_SCAN_GROUP` :5648). Textual's default is `group="default"` (`.venv/.../textual/worker_manager.py:87`) and `exclusive` cancels every worker in that group **on the same node** (`worker_manager.py:75-76` → `cancel_group(worker.node, worker.group)`).
- What a collision does, concretely: the two axes share the group. Confirm a custom **model** id, then a custom **voice** id before the first rebuild completes, and the voice worker cancels the model worker mid-`await card.recompose()` (`_replace_card_bodies`, :2211-2247). `_rebuild_after_custom_id`'s `finally` then discards `"model"` from `_custom_id_rebuild_pending` (:5465) — the fence whose documented job (:5411-5419, :3652-3654) is to stop `_collect_visible_state` reading the stale mounted Select. With the fence gone and the Select not yet rebuilt, the next collection reads the old value over the confirmed custom model id.
- Why it matters: the file's only ungrouped exclusive worker is also the one guarding a "never lose the confirmed value" fence.
- Recommended correction: `group="settings-speech-custom-id"` (or per-axis, `group=f"settings-speech-custom-id-{axis}"`, which removes the cross-axis cancel entirely).
- Size: S · ADR: no · Confidence: inferred (the group semantics and the `finally` are verified from source; the end-to-end value loss is not reproduced — settling it needs a Textual `run_test` driving two `_custom_id_modal_result` calls back to back)
- Pinning test: none.
### P2 [D3] — a UI controller owns three hand-written schema queries over four ChaChaNotes tables  ·  _slice: UIM-console_
- Where: `UI/Console_Modules/workspace.py:5417-5442` (`_revalidate_character_conversation_target_sync`) — `SELECT local_authority_id FROM rag_identity_context`, a `conversations`⋈`character_cards` join, and `SELECT data_revision FROM character_conversation_search_revision`.
- Evidence: `grep -rln "connection.execute(\|cursor.execute(" tldw_chatbook/UI/` → exactly two files repo-wide (`UI/Console_Modules/workspace.py`, `UI/Persona_Modules/personas_conversations_controller.py`); 3 statements in this slice.
- Not a D1: the block is inside `db.transaction()`, parameterized, and reached only through `asyncio.to_thread` (`workspace.py:5395-5397`), so neither the thread-safety nor the loop-blocking rule is broken.
- Why it matters: table/column knowledge for four tables lives in a Console screen controller. A migration that renames `conversations.assistant_authority_id` or the revision singleton passes every check in `DB/` and silently breaks character-conversation activation here; `sql_validation.py`/`VALID_TABLES` governance does not see it either.
- Recommended correction: move the query into `DB/ChaChaNotes_DB.py` as one revalidation method returning the typed row the controller already consumes, and call it from the same `to_thread`.
- Size: M · ADR: no · Confidence: verified
- Pinning test: none found.
- Already covered: none.
### P2 [D3] — personas_screen.py is a 16,414-line / 461-method god module with no row in the screen size ratchet  ·  _slice: UI-personas_
- Where: whole file; `Tests/Architecture/test_screen_size_ratchet.py:76-812` `_BUDGETS` has rows only for `chat_screen.py` (16966/563) and `library_screen.py` (33204/1276); `@pytest.mark.parametrize("rel_path", sorted(_BUDGETS))` (L938/977) so an absent row is simply never measured.
- Evidence: `python - <<EOF ast …` with the test's own `_measure` semantics → `lines 16414 PersonasScreen methods 461` (97% of chat_screen's line budget, 82% of its method budget). `__init__` alone sets ~90 instance attributes (L1321-1469).
- Responsibilities (line ranges): module constants + 12 frozen snapshot dataclasses + `_drain_async`/`_drain_to_thread` + 2 lifetime decorators (1-1067); compose/state round-trip/mount/demand-mounted center views per ADR-115 (1529-2035); character-conversation deep link per ADR-120 (2077-2223); runtime-backend switch + responsive rails (2225-2397); library paging/sort/search/tag + dictionary/lore row rendering (2398-2522, 3480-4386); **Character TTS controls** per ADR-028 (2523-3475, ~950 lines); mode switching + header copy (4387-4603); **Actor Pack export/import/create** per ADR-074 (4604-5131, 7512-7920); selection (5132-5540); **dictionaries** incl. character attach (5541-6485); **lore/world books** (6485-6932); saved conversations + Console handoff + preview delegation (6933-7389); create/edit/duplicate/toggle (7390-8302); **Persona shared visual identity** (8303-8919); **Persona Visual pack authoring** incl. `_persona_visual_thread` (8920-10205); character edit + visual-identity load + avatar upload (10206-10575); LLM-assisted character generation (10576-10723); avatar/expression thumbnails (10724-11003); **Character visual identity pack** incl. `_visual_identity_thread` (11004-12353); **expression slots** upload/generate/style (12355-13342); expression-set import/export (13343-13547); **import** (character + TTS commit, lore, dictionary) (13548-14288); **export** single/bulk/JSON/PNG (14289-14638); **delete** single/bulk (14639-15108); character save (15109-15432); policy rules + persona save (15433-15703); cancel (15704-15763); `_show_center` + aggregate draft snapshot + navigation veto (15764-16058); `_run_guarded`, key bindings, focus, footer sync (16060-16414).
- Why it matters: every one of the ~10 subsystems above shares one 90-attribute `__init__`, one `_io_dialog_active` flag and one message namespace; the file grew past the point where the repo ratchets its peers, and nothing stops it growing.
- Recommended correction: (1) add the row `"tldw_chatbook/UI/Screens/personas_screen.py": ("PersonasScreen", 16414, 461)` now (one-way ratchet, Size S); (2) any split follows `backlog/docs/library-decomposition-recipe.md` §1 per-subsystem PR series with §2 field-ownership script — the natural first peels are Character TTS (~950 lines, self-contained snapshot/authority family), Actor Packs (~1000), Persona Visual (~1300), Character visual identity (~1200) — each already has its own snapshot dataclasses at 671-826 (Size L, per recipe §17, not this review's design).
- Size: S (row) / L (split) · ADR: no for the row; the split is governed by the recipe (ADR 004/007/115 cover the workbench shape, not decomposition) · Confidence: verified
- Pinning test: `Tests/Architecture/test_screen_size_ratchet.py` (the row IS the pinning test once added)
- Already covered: none (task-1378 / task-31202 are settings_screen; task-118 extracted the preview controller and is Done)
### P2 [D3] — physical trace GC/compaction is silently disabled forever by a swallowed `ImportError` of three first-party modules  ·  _slice: CHAT-bridge_
- Where: `tldw_chatbook/Chat/console_runtime.py:3277-3359` (`_schedule_legacy_trace_maintenance.run`): `from ...console_trace_maintenance import PhysicalTraceCompactor, TraceGarbageCollector`, `from ...console_trace_models import new_opaque_id`, `from tldw_chatbook.config import resolve_trace_compaction_policy` sit inside `try: … except ImportError: pass  # Narrow test doubles may provide only the legacy worker`
- Evidence: `fbimports.py console_runtime.py` → all three targets `exists=True` today; `tldw_chatbook.config` is ALREADY imported at module scope (:159), so its function-body import (:3285) can never raise ImportError in production. `rg -n "PhysicalTraceCompactor|only the legacy worker|ImportError" Tests/UI/test_console_runtime*.py Tests/Chat/test_console_runtime*.py` → no hits: nothing pins the swallow. The sibling `except Exception` two lines below DOES log; only the import failure is silent.
- Why it matters: a renamed/moved symbol in `console_trace_maintenance` or `console_trace_models` degrades to `pass` + `await asyncio.sleep(1.0)` forever — the ledger's GC and VACUUM never run again, with no log line, for every user.
- Recommended correction: hoist the three imports to the top of `run()` (`console_trace_models` is a leaf and already loaded — module scope), and give test doubles an explicit `physical_maintenance_enabled=False` seam instead of relying on ImportError.
- Size: S · ADR: no (097-boot-budget-ratchets.md governs *deferral*, not swallowing) · Confidence: verified (swallow), inferred (nobody has hit it yet)
- Pinning test: none
- Already covered: none
### P2 [D3] — quitting the app orphans every running local LLM server  ·  _slice: EVENTS_

- Where: `tldw_chatbook/app.py:19203 on_unmount` (the whole teardown, ~800 lines, tears down recovery service, backup monitor, speech, canvas control, app-owned lifecycles, responsiveness monitor, ingest pools and workers) never touches the six `*_server_process` handles declared at `app.py:7801-7806`.
- Evidence: `rg -n "terminate_process_bounded|stop_server_process" --glob '*.py' tldw_chatbook/` → 20 hits, **every one** inside `Event_Handlers/LLM_Management_Events/` or `UI/Screens/llm_screen.py` (explicit user Stop actions). Zero in `app.py`, zero in any shutdown path.
- Why it matters: `q`/quit leaves llama.cpp / vLLM / Ollama running and holding GPU memory and their port; the next app launch's readiness check sees a server it does not own and cannot stop. (Closing the terminal happens to kill them only because of the *missing* `start_new_session` above — so fixing the P1 above without also adding a shutdown sweep makes this strictly worse.)
- Recommended correction: one `await asyncio.gather(*(stop_server_process(self, p, l) ...))` sweep in `on_unmount`, before `arm_exit_watchdog`'s deadline bites — or an explicit ADR that says servers deliberately outlive the TUI. The two fixes are coupled; ship them in one PR.
- Size: M · ADR: yes if the answer is "deliberate" (new) · Confidence: verified (the grep is exhaustive over the package).
- Pinning test: none found.
- Already covered: none.
### P2 [D3] — the Prompts controller is the one Library cluster where whole-screen recomposes outnumber the canvas-scoped sync the performance audit introduced, 16 to 8  ·  _slice: UIM-library_
- Where: `tldw_chatbook/UI/Library_Modules/library_prompts_controller.py` — `self.refresh(recompose=True)` at `:1652, 2586, 3353, 3408, 3905, 4155, 4358, 4394, 4477, 4570, 4661, 4733, 4778, 4880, 4925, 4968`
- Evidence: `rg -c '^\s*self\.refresh\(recompose=True\)' tldw_chatbook/UI/Library_Modules/` → prompts 16, export 5, media 3, ingest 3, unavailable_navigation 2, notes 1, collections 1, prompt_collection_manager_modal 1. `rg -c '_sync_library_canvas\(' …` → notes 40, skills 16, media 8, conversations 8, **prompts 8**, rag 4. So notes is 40:1 and skills 16:0 in favour of the targeted seam while prompts is 8:16 against it. The seam's own docstring (`canvas_sync.py:516-537`) records what a whole-screen recompose costs — "a whole-screen remove/remount of the nav bar, footer, ~20-row rail, and 50-100-row canvas" — and cites `Docs/Design/2026-07-16-performance-audit.md §P1 B2` (file present) as why the conversion exists. Every one of the 16 sites is in a cluster the same file already syncs canvas-scoped elsewhere (`_sync_library_prompt_selection:1131`, the import open/close pair `:1600/:1622`, the detail retry `:1573`, the browse-result projection `:1351`).
- Why it matters: it is not a hot path today (each of the 16 is one user gesture — duplicate, convert, undo, receipt dismiss, conflict entry/resolution, a file-picker callback), which is why it has survived; but it is the pattern that produced the audit's finding, on the one cluster that never finished converting, and each recompose tears the rail and footer down beside an editor holding unsaved text. `:1652` (`browse_callback`) is the clearest single case: it recomposes the whole screen to store one string into `_library_prompts_import_path`.
- Recommended correction: convert the 16 to `_sync_library_canvas(self, "prompts", then=…)` in one PR, in the shape the same file's 8 existing call sites already use — what the notes and skills series did. No redesign: the helper, the follow-up ordering and the fallback all already exist.
- Size: M · ADR: no (`backlog/docs/library-decomposition-recipe.md` §1 per-subsystem PR series is the shape) · Confidence: verified
- Pinning test: none (the size ratchet measures lines, not recompose discipline)
- Already covered: none
### P2 [D3] — the switcher modal drives character activation on a bare `asyncio.create_task`, outside Textual's worker lifecycle, and never cancels it on unmount  ·  _slice: W-console-2_
- **Where:** `console_session_switcher_modal.py:2235` (`_begin_character_activation`) and `:2440` (`_recover_character_activation`). `on_unmount` (`:548-566`) stops both timers and sets the cancellation `Event` only in the `OPENING_CANCELLABLE` phase; it never touches `self._activation_task`.
- **Evidence:** `.venv/.../textual/widget.py:4849-4851` — `Widget._on_unmount` calls `self.workers.cancel_node(self)`, so every `run_worker` task on this screen is cancelled at unmount; a bare `asyncio.create_task` is not. The same class uses `run_worker(..., group=...)` for its four other async paths (`:650`, `:1326`, `:1877`, `:1907`).
- **Why it matters:** after the modal is popped, `_run_character_activation` keeps awaiting the activator and then drives the dead screen (`_show_activation_failure` → `_set_status`, or `dismiss_safe_once`); the modal cannot be collected until the adapter returns. Every reached call site is `try/except NoMatches`-guarded, so I could not produce a crash — this is lifetime/leak, not a crash.
- **Recommended correction:** `self.run_worker(self._run_character_activation(...), group="console-session-switcher-activation", exclusive=True)`, matching its own four siblings.
- **Size:** S · **ADR:** no · **Confidence: inferred**. Literal command to settle it: a probe that pushes the modal, calls `_begin_character_activation` with an activator blocking on an `asyncio.Event`, pops the screen, and asserts `modal._activation_task.cancelled()`.
- **Pinning test:** none
- **Already covered:** none
### P2 [D3] — ~610 of `note_ingest_events.py`'s 693 lines are handlers for a retired Ingest UI: every widget id they query exists nowhere, and one calls a `TldwCli` method that no longer exists  ·  _slice: EVENTS_

- Where: `Event_Handlers/note_ingest_events.py` — `_update_note_preview_display:60`, `_parse_single_note_file_for_preview:108`, `_handle_note_file_selected_callback:213`, `handle_ingest_notes_select_file_button_pressed:~260`, `handle_ingest_notes_clear_files_button_pressed:~290`, `handle_ingest_notes_import_now_button_pressed:308` (343 lines — the third-largest function in the slice). Only `_import_template_files` is live.
- Evidence, symbol by symbol (script walks every top-level def and greps all of `tldw_chatbook/` and `Tests/`):
```
_update_note_preview_display                   prod=['Event_Handlers/ingest_events.py'] tests=0
_parse_single_note_file_for_preview            prod=['Event_Handlers/ingest_events.py'] tests=2
_handle_note_file_selected_callback            prod=['Event_Handlers/ingest_events.py'] tests=0
handle_ingest_notes_select_file_button_pressed prod=['Event_Handlers/ingest_events.py'] tests=0
handle_ingest_notes_clear_files_button_pressed prod=['Event_Handlers/ingest_events.py'] tests=0
handle_ingest_notes_import_now_button_pressed  prod=['Event_Handlers/ingest_events.py'] tests=2
_import_template_files                         prod=['Backup_Recovery/async_file_participants.py'] tests=1
```
…and `ingest_events.py` itself has **zero** importers (census above), so six of the seven are reachable from nothing.
- The UI they target is gone. Every widget id in the file:
```
for wid in ingest-notes-import-status-area import-as-templates-radio chat-notes-collapsible \
           ingest-notes-preview-area ingest-notes-selected-files-list ; do
  rg -l "\"$wid\"" --glob '*.py' --glob '*.tcss' tldw_chatbook/ ; done
```
→ only one hit in total, `ingest_status_helper.py` (itself dead). `#ingest-notes-import-type` likewise. CLAUDE.md records why: "ingestion lives in the Library screen's Import rail path; `media_ingest_screen.py` was removed".
- And `note_ingest_events.py:534` calls `app.call_later(app.on_chat_notes_collapsible_toggle, …)` — `rg -c on_chat_notes_collapsible_toggle tldw_chatbook/app.py` → **0**. That is an `AttributeError`, not a `QueryError`, so the `except QueryError` two lines up would not catch it; it is unreachable only because the `query_one("#chat-notes-collapsible")` on the line above fails first. Two dead things covering for each other.
- Why it matters: 343 lines of carefully-commented concurrency work (TASK-15468's snapshot-the-list and cancel-event fix rounds) maintained on a path no user can reach, and two tests (`Tests/Event_Handlers/test_note_ingest_events.py::test_successful_note_import_refreshes_mounted_library_screen`, `…_does_not_touch_non_library_screen`) keep it green so the rot is invisible. `app.py:7792,8058` still declares `selected_note_files_for_import` for it.
- Recommended correction: delete `ingest_events.py`, `ingest_status_helper.py` and the six retired handlers, keeping `_import_template_files` (+ `ingest_utils`, which it needs) — ideally moved next to its one consumer in `Backup_Recovery/`. The `Tests/test_application_state_ownership.py:1232-1244` source-text assertions on `ingest_events.py`/`ingest_utils.py` must move or go in the same commit, and `app.py`'s `selected_note_files_for_import` with them.
- Size: M · ADR: no · Confidence: verified.
- Pinning test: `Tests/Event_Handlers/test_note_ingest_events.py` (2 collected tests) exercises the dead handler; `Tests/test_application_state_ownership.py::test_legacy_ccp_prompt_handlers_and_compatibility_exports_are_absent` reads `ingest_events.py` as text and would fail with `FileNotFoundError` on deletion.
- Already covered: none. Related: task-30019 (legacy Collections migration/retirement) is the nearest in spirit but does not cover this.
### P2 [D3] — §17 controller-file size governance is red for 16 of 27 `_BUDGETS` rows at the reviewed SHA; both >5k controllers are over their pins (notes +1065, prompts +65)  ·  _slice: UIM-library_
- Where: `Tests/Architecture/test_library_modules_size_ratchet.py:135-576` (`_BUDGETS`), measured files under `tldw_chatbook/UI/Library_Modules/`
- Evidence: `cd $WT && source $SCRATCH/env.sh && PYTHONPATH=$WT $PY -m pytest Tests/Architecture/test_library_modules_size_ratchet.py -q -p no:cacheprovider` → `16 failed, 27 passed`. Rows: notes_controller 6365 vs 5300 (+1065); prompts_controller 5063 vs 4998 (+65); ingest 3078/2721 (+357); notes_sync 2380/2024 (+356); media_browse 720/371 (+349); note_import 793/602 (+191); export 1448/1307 (+141); media 4763/4670 (+93); conversations 1800/1738 (+62); collections 1747/1689 (+58); skill_import 805/760 (+45); conversation_reader 969/943 (+26); unavailable_navigation 840/817 (+23); rag_search 1915/1898 (+17); navigation 202/198 (+4); skills 3145/3142 (+3). The test's last commit is `18a8369bf1 2026-09-11`; the notes controller was modified by 11 commits after that (wave-5 PRs #2694–#2702, 2026-09-13..15) with no re-pin. `git show c8e2a913e6:…/library_notes_controller.py | wc -l` → 5816 at the "re-pin" commit itself vs its 5300 row, and media_browse was 720 at that commit vs its 371 row — two rows were already stale when pinned. `.github/workflows/test.yml:121` runs `pytest Tests --ignore=Tests/UI`, so this test is in the core CI shards.
- Why it matters: the ratchet is the only mechanism §17 of the recipe relies on to stop the decomposed controllers regrowing into god modules; a 16-row-red ratchet that CI tolerates is inert, and `library_notes_controller.py` has regrown 20% past its pin in four days.
- Recommended correction: re-pin all 16 rows at their measured counts in one commit per the test's own dated guidance, and — since the notes controller is now the largest file in the slice at 6365 — open the §1 per-subsystem move series for it. Not a redesign; the recipe already prescribes the shape.
- Size: S (re-pin) · ADR: yes (`backlog/docs/library-decomposition-recipe.md` §17 governs; no new ADR) · Confidence: verified
- Pinning test: `Tests/Architecture/test_library_modules_size_ratchet.py::test_controller_does_not_grow_past_its_budget[...]` — states the requirement (it is the test that is red)
- Already covered: none for the re-pin; task-32199 (test-health after the notes decomposition) is adjacent — cite it there rather than filing a duplicate
### P2 [D4] (b) — Library FTS tokenisers re-roll `Utils.fts5_match_forms.build_and_match_query` with different tokenisation, a 20-token cap and no NUL guard  ·  _slice: DB-chacha_
- Where: `ChaChaNotes_DB.py:17838` `_library_note_fts_query`, `:18471` `_library_conversation_fts_query` (verbatim clones), plus `DB/Prompts_DB.py:3330` `_library_prompt_fts_query`, `DB/Client_Media_DB_v2.py:8612` `_library_fts_query` (same shape; outside this slice). Canonical: `Utils/fts5_match_forms.py:348` `build_and_match_query` (used by this file's other 6 search seams).
- Evidence: drift script (inline in this review) comparing `CharactersRAGDB._library_note_fts_query(q)` vs `build_and_match_query(q)`:
  `'foo-bar'` → library `'"foo" "bar"'` | canonical `'"foo-bar"'` DIFF · `"it's"` → `'"it" "s"'` | `'"it\'s"'` DIFF · `'a\x00b'` → `'"a" "b"'` | `''` DIFF (canonical refuses NUL; library binds it and SQLite truncates the parameter) · 22 tokens → library truncates to 20 | canonical keeps all DIFF · `'hello world'` SAME.
- Why it matters: Library ▸ Notes/Conversations search answers a different question from Console/character/flashcard search for hyphenated or apostrophe-bearing terms (`"foo-bar"` is a phrase over the runs `foo bar`; `"foo" "bar"` is AND in any order) and silently drops tokens past 20.
- Recommended correction: make the four library copies call `build_and_match_query` (canonical home already exists); if the 20-token cap is wanted, add it as a parameter there. `Tests/Utils/test_fts5_quoting_adoption_census.py` guards the quote escape only, not the tokeniser.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/DB/test_fts5_quoting_search_seams.py` covers quoting on the canonical seams; none pins the library tokeniser's split-on-punctuation.
- Already covered: none
### P2 [D4] (b) — Six timestamp shapes are written to rows; two of them have already forced `julianday()` workarounds and the un-worked-around sites sort lexically  ·  _slice: DB-chacha_
- Where (writers): (a) `_get_current_utc_timestamp_iso` `:8700` → `YYYY-MM-DDTHH:MM:SS.mmmZ` (≈45 call sites); (b) SQL `CURRENT_TIMESTAMP` → `YYYY-MM-DD HH:MM:SS`: every column DEFAULT (87 occurrences in the schema literals) plus 14 explicit Python-path writes `:13587 :21479 :21566 :21567 :21593 :21623 :21732 :21787 :21836 :21891 :21904 :22069 :22124 :23456` and `:3777`; (c) SQL `strftime('%Y-%m-%dT%H:%M:%fZ','now')` `:5773`, `:9337 :9342 :9396` and the `character_expression_images` defaults (same shape as (a) — correct, `%f` = SS.SSS); (d) `datetime('now')` in the `world_book_entries_sync_delete` trigger `:2013` (shape (b)); (e) `datetime.isoformat()` with microseconds + `+00:00` `:21570`, `:23523/:23535/:23549` (the P1 above); (f) read-side re-serialisation `:18728` `raw_timestamp.isoformat().replace("+00:00","Z")` → `…ffffffZ` (differs from (a)'s `mmmZ`); (g) `sqlite_datetime_fix.adapt_datetime` turns any `datetime` bind parameter into shape (e).
- Evidence: the file's own comment at `:17723-17734` ("`last_modified` is DATETIME DEFAULT CURRENT_TIMESTAMP, whose space-separated shape sorts against the ISO `T...Z` shape application writers stamp -- which is why the ACTIVE notes list wraps its date ordering in `julianday()` (task-32172)") and the `julianday()` wrappers at `:11456-11471`. Lexicographic `ORDER BY last_modified DESC` remains at `:11184 :11820 :11905 :17737 :18157 :18562 :19689 :19866 :22689`. `grep -rlE 'isoformat\(timespec="milliseconds"\)' tldw_chatbook` → 8 files / 13 copies of the (a) formatter (`Personal_Context/repository.py`, `Canvas/staging.py`, `Canvas/repository.py`, `Notes/note_import_executor.py`, `Notes/note_folder_repository.py`, `Notes/notes_organization_repository.py`, `Notes/Notes_Library.py`, `DB/ChaChaNotes_DB.py`); no `Utils` helper exists (`grep -rnE 'def (utc_now|now_utc|utc_iso|iso_now|utc_timestamp)'` → none). `grep -iE 'timestamp|datetime' adr_list.txt` → no ADR.
- Why it matters: rows written by a DEFAULT (or by any writer that omits the column) and rows written by (a) sort in the wrong order whenever they share a date; the flashcard/stats P1 is the same defect class reaching a comparison. That the (a)/(b) mix reaches `conversations.last_modified`/`notes.last_modified` in real profiles is **inferred** from the `:17723` comment and task-32172, not demonstrated here.
- Recommended correction: one canonical formatter in `Utils/` (e.g. `Utils/time_format.py::utc_now_iso()` returning shape (a)) adopted by the 13 copies; new columns default to `(STRFTIME('%Y-%m-%dT%H:%M:%fZ','NOW'))` as `character_expression_images` already does; date ORDER BYs over columns that can carry both shapes go through `julianday()` (the `:11456` precedent). Retiring the existing (b) DEFAULTs is a storage-format change → L.
- Size: M for the helper + writer adoption; L for the schema defaults · ADR: new (storage timestamp format) · Confidence: verified for the writer census, inferred for the mixed-row claim on conversations/notes
- Pinning test: none found for ordering across shapes.
- Already covered: task-32172 (Notes date ordering only)
### P2 [D4] — `_apply_console_message_attachments` is a byte-identical copy of `ConsoleChatStore._set_message_attachments`, and `Chat/` imports the UI copy to get it  ·  _slice: UIM-console_
- Where: `UI/Console_Modules/message.py:186-210` vs `Chat/console_chat_store.py:10708-10729`; the inverted import is `Chat/console_conversation_hydration.py:137-139` (`from tldw_chatbook.UI.Console_Modules.message import _apply_console_message_attachments`, inside `_batch_fetch_resume_attachments`).
- Evidence: both bodies read (above); the store copy is decorated `@staticmethod` at `console_chat_store.py:10708`, so it needs no instance. The UI copy's own docstring justifies itself with "outside the store, where that helper isn't reachable" — that premise is false for a staticmethod.
- Why it matters: the helper enforces a storage invariant ("every attachments mutation sets the tuple AND the scalar image fields together"). Two copies of a storage invariant drift into two different on-screen/persisted attachment states; and a `Chat/` module reaching into `UI/` inverts the layering, so the business layer cannot be used (or tested) without the UI package.
- Recommended correction: delete `message.py:186-210`, call `ConsoleChatStore._set_message_attachments` from both call sites (canonical home: `Chat/console_chat_store.py`, where the invariant is documented), and drop the function-body import in `console_conversation_hydration.py`.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none found.
- Already covered: none. Same defect reported from the store side as CHAT-store P2 — confirmed from this side, with the staticmethod fact that removes its stated justification.
### P2 [D4] — `_console_message_role_from_persisted` is duplicated verbatim across the Chat/UI boundary  ·  _slice: UIM-console_
- Where: `Chat/console_conversation_hydration.py:109-122` (module function) and `UI/Console_Modules/message.py:597-610` (staticmethod). Bodies read side by side: identical except the parameter annotation (`Mapping[str, Any]` vs `dict[str, Any]`).
- Why it matters: this maps persisted `role`/`sender` columns onto `ConsoleMessageRole`; drift between the two copies shows a resumed conversation with different message roles depending on which path hydrated it (user-visible, storage-derived).
- Recommended correction: keep the `Chat/console_conversation_hydration.py` copy as canonical and have `message.py` import it (that direction does not invert the layering).
- Size: S · ADR: no · Confidence: verified
- Pinning test: none found.
- Already covered: none.
### P2 [D4] — the Advanced runner renders un-redacted secrets and absolute paths, bypassing the two helpers the same file uses everywhere else (helper exists, ignored)  ·  _slice: UIM-nav-mcp-persona_
- Where: `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py:3882` (`if isinstance(result, dict): result = redact_mapping(result)` — a LIST result skips it), `:3878` (`result_widget.update(f"Action failed: {exc}")`), `:3873` (`f"{_ADVANCED_BLOCKED_HEADING}\n{exc}"`), `:3854` (`f"Invalid JSON payload: {exc}"`). The helpers that exist and are used by every OTHER result surface in this same file: `redact_mapping` (`MCP/redaction.py`, module docstring: *"Secret redaction applied at every MCP display and log boundary"*) and `_safe_exception_text`/`_safe_tool_test_text` (`mcp_inspector.py:192`/`:158`, which redact `api_key=`, bearer tokens, `sk-*`, and absolute paths).
- Evidence:
  ```
  cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$PWD $PY <repro_adv_redact.py>
  ```
  (harness mounts `MCPInspector`, binds a fake service, calls `_run_advanced_action()`) ->
  ```
  redact_mapping(dict) would give: {'name': 'docs', 'api_key': '***', 'env': {'TOKEN': '***'}}
  --- list result rendered into #mcp-adv-result ---
  [ { "name": "docs", "api_key": "sk-live-ABCDEF123456", "env": { "TOKEN": "t-secret-999" } } ]
  --- exception text rendered into #mcp-adv-result ---
  Action failed: connect failed: api_key=sk-live-ABCDEF123456 at /Users/rob/secret/path
  ```
  Script kept at `<SCRATCH>/repro_adv_redact.py`.
- Why it matters: the Advanced runner's whole point is dumping raw control-plane payloads (`external_servers` env/args included) — `mcp_workbench._redact_external_server_record()` exists precisely because that renderer leaked full raw records. That shim covers `load_section`; `run_action`'s own result and every exception path are not covered, so a list-shaped action result or any service exception puts credentials and absolute paths on screen in a pane users copy into bug reports.
- Recommended correction: route the result through `redact_mapping` for Mappings *inside* sequences too (or reuse `mcp_workbench._redact_external_servers_list`'s shape), and pass every `{exc}` through this module's own `_safe_exception_text()` — the same call `show_tool_result`, `show_test_unavailable`, and `_handle_test_run` already make four lines away.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none
### P2 [D4] — the Select-blank predicate exists in THREE copies and the fourth site got it wrong (no helper home, copies drifted)  ·  _slice: UIM-nav-mcp-persona_
- Where: three identical copies —
  - `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py:841-853` `_is_blank()` (`value is Select.BLANK or value is Select.NULL`, with a 12-line comment explaining the trap)
  - `tldw_chatbook/UI/MCP_Modules/mcp_server_mutations.py:42` (same one-liner)
  - `tldw_chatbook/UI/MCP_Modules/mcp_rail.py:582` (same expression, inlined)
  — and the divergent fourth: `tldw_chatbook/UI/MCP_Modules/mcp_permissions_mode.py:1041` (`if event.value is Select.BLANK: return`), which checks only the sentinel that is NOT the blank marker.
- Evidence:
  ```
  $PY -c "from textual.widgets import Select; from textual.widget import Widget; print(repr(Select.BLANK), Select.BLANK is Widget.BLANK, repr(Select.NULL))"
  ```
  -> `False True Select.NULL` (Textual 8.2.8). The real no-selection sentinel is `Select.NULL`; `Select.BLANK` resolves through the MRO to `Widget.BLANK == False`, so `event.value is Select.BLANK` can never be true for a string profile id.
- Why it matters: the guard is dead. Today it is unreachable-harmless because `#mcp-perm-tool-profile` is constructed `allow_blank=False` (verified in Textual's `Select._setup_variables_for_options`: `NULL` is only inserted when `_allow_blank`), so the value is always a real profile id. If `allow_blank` ever flips, `ToolPolicyProfileSelected(str(Select.NULL))` posts the literal string `"Select.NULL"` as a profile id. `mcp_inspector._is_blank()` (`:841`) already documents this exact trap and checks BOTH sentinels; this sibling does not use it.
- Recommended correction: one canonical `select_is_blank(value)` — the natural home is `tldw_chatbook/UI/Widgets/` (or `MCP_Modules/__init__.py`, currently 1 line) since three of the four sites are MCP canvases and `mcp_inspector` is the wrong direction for `mcp_workbench` to import from anyway (see the private-import finding). All four sites call it.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none
### P2 [D4a+b] — five near-identical list-picker modals in one directory; three of them skip the canonical dismiss mixin  ·  _slice: W-persona-settings-chat_
- Where: `Persona_Widgets/dictionary_picker.py` (134), `world_book_picker.py` (131), `dictionary_attach_picker.py` (122), `conversation_attach_picker.py` (121), `tag_filter_picker.py` (118) — 626 lines total, one widget shape.
- Evidence (clone measurement): normalising each file (drop comments/blank lines, rewrite every domain noun — dictionary/world_book/conversation/tag — to `X`) and diffing:
```
dictionary_attach_picker vs conversation_attach_picker : 18 differing lines out of 95
dictionary_picker        vs world_book_picker          : 39 differing lines out of 108
```
  Every one of the 18 differences in the first pair is the class name, the docstring, or a CSS type selector — **no logic differs at all**. `conversation_attach_picker.py`'s own docstring says so: "Generic — used by the Roleplay Lore Attachments flow (P2e); the dictionary flow keeps its own DictionaryAttachPicker."
- The drift that makes this more than tidiness: `grep -n "^class \|SafeModalDismiss\|dismiss" *picker*.py` →
  - `DictionaryPicker` (:28) and `WorldBookPicker` (:29) inherit `SafeModalDismissMixin` and cancel via `dismiss_safe_once(None)`;
  - `DictionaryAttachPicker` (:25), `ConversationAttachPicker` (:24) and `TagFilterPicker` (:27) inherit plain `ModalScreen` and cancel via raw `self.dismiss(None)`.
  `Widgets/modal_dismissal.py` has **82 importers** (`grep -rl SafeModalDismissMixin tldw_chatbook/ | wc -l`). What the three lose is not cosmetic: `dismiss_safe_once` (:270-299) is the one-shot guard (`_safe_dismiss_committed`), the topmost-screen check (`app.screen is not self`), the backdrop-click shield, **and opener-focus restoration** (`_restore_focus_after_dismissal`). A keyboard user cancelling `ConversationAttachPicker` does not get focus back on the button that opened it; cancelling `WorldBookPicker` does.
  Secondary asymmetry inside the two compliant pickers: `_confirm` still calls raw `self.dismiss(self._selected_id())` (`dictionary_picker.py:126`, `world_book_picker.py:123`) while only `_cancel` goes through the mixin — so the double-press guard covers Cancel and not Confirm.
- Recommended correction: one parameterised `ListPickerModal(rows, *, id_key, title, search_placeholder)` in `Widgets/` (next to `modal_dismissal`), inheriting `SafeModalDismissMixin`, with the five call sites passing their id key and copy. Failing that, at minimum put the mixin on the three that lack it and route `_confirm` through `dismiss_safe_once` in all five.
- Size: M · ADR: no · Confidence: verified (diffs + greps above)
- Pinning test: none found asserting focus restoration for these pickers.
- Already covered: none. (The brief's note that the 2-line `_cancel` adapters are P3 holds for the *adapters*; this is the class-level drift it exempts.)
### P2 [D4a] — 15 private `_coerce_bool` re-rolls of a public helper that has 17 importers, and they disagree on the integer `1`  ·  _slice: CHAT-rest-2_
- Where: `tldw_chatbook/Chat/console_rail_state.py:335` is this slice's copy. The public helper is `tldw_chatbook/config.py:1201 coerce_bool_setting` (17 import sites). Census: `rg -n --glob 'tldw_chatbook/**/*.py' 'def _coerce_bool\(|def _coerce_bool_option\(|def coerce_bool_setting\('` → **16 definitions**; `rg -n 'coerce_bool_setting' --glob 'tldw_chatbook/**/*.py' | rg import | wc -l` → **17**.
- Evidence — same inputs through eight of them (`FALSE` = the `default` argument, which was passed as `False`):
  ```
  helper                                              1        0        2      1.0   'TRUE'      'y'     None
  config.coerce_bool_setting                       True    False    False    False     True     True     None
  Chat/console_rail_state                          True    False     True    False     True    False    False
  Utils/adaptive_reader_state                     False    False    False    False     True    False    False
  Character_Chat/world_book_manager                True    False     True     True     True    False    False
  Image_Generation/config                         False    False    False    False     True    False    False
  UI/Screens/settings_appearance_defaults          True    False    False    False     True    False    False
  Library/library_rail_state                       True    False     True    False     True    False    False
  Home/home_rail_state                             True    False     True    False     True    False    False
  ```
  (Command: `PYTHONPATH=$PWD $PY -` importing each `_coerce_bool` and printing `fn(v, False)` for `v in (1, 0, 2, 1.0, "TRUE", "y", None)`.)
- Why it matters: this is drift that reaches **stored data**, so it is a D1 as well as a D4. `1` is exactly what SQLite hands back for a boolean column and what TOML/JSON round-trips produce for `true` in a loosely-typed section — and two of the eight read it as the caller's *fallback* rather than True. `1.0` (a JSON round-trip of a bool through a float-coercing layer) is True in exactly one of the eight. `"y"` is True only in the canonical helper. A value migrating between two of these surfaces flips meaning.
- Recommended correction: `config.coerce_bool_setting` is already the repo's named standard (`change_review_screen.py:341` calls it "the repo's standard coercion"), but it lives in `config.py`, which is why leaf modules re-roll rather than import it. Move the coercion itself to `Utils/` (a stdlib-only leaf), have `config.coerce_bool_setting` delegate, and delete the 15 copies. Settle the int/float question once in that one place and write the truth table into its docstring.
- Size: M (one PR, and a format has to be chosen: does `2` mean True, does `1.0`) · ADR: no · Confidence: verified (measured)
- Pinning test: each site has its own local tests; none of them compares across sites, which is why the drift survived.
- Already covered: none. This spans the repo; my slice contributes one copy (`console_rail_state.py:335`) and the census.
### P2 [D4a] — Every held-connection store re-rolls the PRAGMA/WAL/busy_timeout/isolation setup that `BaseDB._get_connection` could own; the undocumented rows drift  ·  _slice: DB-media-base_
- Where: `tldw_chatbook/DB/base_db.py:818-825` sets only `row_factory`. Overrides (all read):

| store (`_get_connection` / opener) | foreign_keys | journal_mode WAL | synchronous | busy_timeout | isolation_level | documented? |
|---|---|---|---|---|---|---|
| `BaseDB` 818-825 | — | — | — | sqlite3 default 5 s | legacy | — |
| `Workspace_DB.py` 353-388 | ON | file-only | NORMAL | default 5 s | None | task-3012/15480 comment |
| `AgentRuns_DB.py` 297-337 | ON | file-only | NORMAL | `PRAGMA busy_timeout=5000` set BEFORE WAL (only store that orders it) | None | comment |
| `Library_Collections_DB.py` 498-543 | ON | file-only via `_enable_wal` retry loop (only store with one) | NORMAL | default 5 s | None | task-15466 comment |
| `Subscriptions_DB.py` 614-660 | ON | file-only, non-RO | NORMAL | `BUSY_TIMEOUT_MS` | legacy | **task-22224 EXCEPTION** (docstring) |
| `Evals_DB.py` 196-236 | ON | unconditional (also `:memory:`) | NORMAL | default 5 s | legacy | **task-22224 EXCEPTION** (docstring) |
| `RAG_Indexing_DB.py` 104-144 | **absent** (declares no FKs — verified `rg -i "FOREIGN KEY\|REFERENCES"` → 0) | file-only | NORMAL | default 5 s | None | comment |
| `Library_Ingest_Jobs_DB.py` 84-110 (the template) | **absent** (no FKs) | unconditional | NORMAL | default 5 s | None | module docstring |
| `Client_Media_DB_v2.py` 1127-1152 | ON | file-only | NORMAL | connect `timeout=10` (→ `PRAGMA busy_timeout` 10000 verified) | legacy | **task-22224 EXCEPTION** (docstring) |
| `ChaChaNotes_DB.py` 3480-3514 | ON | file-only | NORMAL | `timeout=15` | None | task-22224 comment |
| `Prompts_DB.py` 462-482 | ON | file-only | NORMAL | `timeout=10` | legacy | **task-22224 EXCEPTION** (docstring) |

- Evidence: read only (table above is from the cited ranges); `media_repro4.py` PRAGMA read-back for the Media row.
- Why it matters: the documented `isolation_level` rows are P3 (and the Media one is where the P1 above lives). The undocumented drift is `busy_timeout` (five stores on the implicit 5 s default, three on 10–15 s, two on an explicit PRAGMA) and the WAL-conversion race (handled by ordering in AgentRuns, by a retry loop in Library_Collections, by nothing elsewhere) — the same cross-process first-open contention hits every file store identically. `foreign_keys` absence in RAG_Indexing/Library_Ingest_Jobs is inert today (no FKs) and stays P3.
- Recommended correction: one `configure_held_connection(conn, *, is_memory: bool, busy_timeout_ms: int)` in `DB/base_db.py` (row_factory, `busy_timeout` first, `foreign_keys`, guarded WAL, `synchronous=NORMAL`, `isolation_level=None`) that `BaseDB._get_connection` calls and the four legacy-isolation stores call with an explicit `legacy_isolation=True` until their task-22224 census lands. The template docstring in `Library_Ingest_Jobs_DB.py:1-20` then points at code instead of prose. M.
- Size: M · ADR: no (adr_list grep for connection/pragma/wal/isolation → only 114/125/157, unrelated) · Confidence: verified (table) / inferred (that the WAL race reaches the un-guarded stores in practice — literal check: run two processes opening the same fresh `.db` under `Evals_DB` concurrently)
- Pinning test: none for the shape; `Tests/DB/test_chachanotes_connection_quiescence.py` pins ChaChaNotes' registry behaviour only
- Already covered: task-15466 / task-15480 / task-21101 ported the idiom store-by-store; none owns the shared helper
### P2 [D4a] — RAG profile JSON is written with a bare truncate-then-write while `Utils/atomic_file_ops.atomic_write_json` exists and is used by 10 other modules  ·  _slice: RAG_
- Where: `tldw_chatbook/RAG_Search/config_profiles.py:652` (`_save_one`, the single choke point every profile save goes through: `save_profile` ← `clone_profile`, `create_custom_profile`, `rename_profile`, `active_config.ensure_imported_profile`, the Settings screen's save path). Same shape at `:822` (legacy-blob migration), `:1096` (experiment config), `:1234` (experiment results), and `pipeline_loader.py:746`.
- Evidence:
  - `sed -n '648,653p' config_profiles.py` → `with self._definition_write(), open(self._profile_path(profile.id), "w") as f: json.dump(profile.to_dict(), f, indent=2, default=str)`
  - `grep -rln "atomic_write_json" tldw_chatbook/` → 10 modules (`Model_Artifacts/service.py`, `Skills_Interop/skill_trust_store.py`, `Chatbooks/local_chatbook_service.py`, `UI/LLM_Management/vllm_profiles.py`, `Web_Server/artifact_share_manifest.py`, …). `Utils/atomic_file_ops.py:199 atomic_write_json` does temp-file + `os.fsync` (`:103`) + `os.replace`.
- Why it matters: a crash, a power loss, or ENOSPC between `open(..., "w")` (which truncates immediately) and the end of `json.dump` leaves a zero-length or half-written profile file. `_load_custom_profiles` (`:761`) catches the resulting `JSONDecodeError`, logs `Failed to load profile <name>`, and **skips** it — the user's saved RAG profile silently disappears from the picker, and the previous good content is already gone.
- Recommended correction: `_save_one` → `atomic_write_json(self._profile_path(profile.id), profile.to_dict(), indent=2, default=str)` (check that helper's kwargs; it may need a `json.dumps` + `atomic_write_text`). Canonical home already exists: `Utils/atomic_file_ops.py`. The three non-profile call sites are lower value (`:1096`/`:1234` write experiment artefacts from a subsystem that never runs — see the dead-subsystem finding).
- Size: S · ADR: no · Confidence: inferred (the non-atomic write and the skip-on-parse-failure loader are both verified by reading; the crash-window loss is not reproduced)
- Pinning test: `Tests/RAG/test_config_profiles.py` covers save/load round-trips but nothing about torn writes.
- Already covered: none
### P2 [D4a] — `EvalsDB._loads_json_or_default` was added to stop a NULL JSON column raising `TypeError` out of a lookup, then applied to only 4 of the 15 JSON-column reads in the same file; the 11 raw ones include three nullable columns  ·  _slice: DB-rest_
- Where: helper at `tldw_chatbook/DB/Evals_DB.py:1457-1476`. Adopted at `:1492` (`get_model`), `:1518` (`list_models`), `:1720` (`get_run`), `:1784` (`list_runs`). **Not** adopted at `:1101` `get_task`, `:1125` `list_tasks`, `:1188` `search_tasks`, `:1260` `get_dataset`, `:1279` `list_datasets`, `:1410` `search_datasets`, `:2007-2010` `get_run_results` (four columns), `:2141` `list_probe_turn_annotations`, `:2395/:2397` `get_ab_test`, `:2447/:2449` `list_ab_tests`.
- Evidence (isolated env, temp-file DB; one row inserted with `metadata` NULL, which the schema at `:298` permits — `metadata TEXT` with no NOT NULL):
  `get_dataset RAISED: TypeError the JSON object must be str, bytes or bytearray, not NoneType` / `list_datasets RAISED: TypeError …` / `search_datasets RAISED: TypeError …`, while `EvalsDB._loads_json_or_default(None, {}, column="x") -> {}`. `eval_results.logprobs/metrics/metadata` (`:355-360`) are nullable too and read raw at `:2007-2010`.
- Why it matters: the helper's own docstring says why it exists — *"Rows created without their config columns carry NULL, and `json.loads(None)` raises TypeError out of lookup APIs whose consumers include the Evals screen"* (TASK-21519). That is still true for datasets and results; only models and runs were fixed. Today's in-module writers always pass `json.dumps(x or {})`, so the NULL has to arrive from a restored/recovered database (`Evals/recovery.py` recreates these tables) or an older row — which is exactly the case TASK-21519 was filed for. The second half of the helper (raising `EvalsDBError` naming the column instead of a bare `ValueError` for corrupt JSON, PR #2634) is also missing at those 11 sites.
- Recommended correction: route the 11 remaining reads through `_loads_json_or_default` with the right default (`{}` for the config/metadata columns, `[]` for `eval_probe_turn_annotations.tags`). Mechanical.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none asserting the raw form.
- Already covered: none
### P2 [D4a] — `Utils/Utils.py:253 truncate_content()` has **zero importers** while 52 inline re-rolls of exactly its body live in the package; my slice's copy is byte-identical in behaviour  ·  _slice: CHAT-rest-3_
- Where: helper `tldw_chatbook/Utils/Utils.py:253-261` (`content[: max_length - 3] + "..."`). Copy in this slice: `Chat/provider_failures.py:50-53` (`if len(detail) > 240: detail = detail[:237] + "..."`). Nearest neighbours outside this slice: `Chat/console_generate_image.py:230`, `Chat/console_fleet_wake.py:132`, `UI/Research_Modules/bundle_rendering.py:99`, `Tools/code_audit_tool.py:286,429,487,531`, `UI/STTS_Window.py:568,679,842,1428`, `UI/Evals/bench_editor.py:249`, `UI/Evals/inspector.py:213`, `UI/MCP_Modules/mcp_inspector.py:458`.
- Evidence:
  - `grep -rn --include='*.py' "truncate_content" tldw_chatbook/ Tests/ | grep -v "Utils/Utils.py:"` -> **no output**: the helper is dead.
  - `grep -rn --include='*.py' -E '\[: *[A-Za-z_0-9]+ *(- *[0-9]+)? *\] *\+ *"(\.\.\.|…)"' tldw_chatbook/ | wc -l` -> **52**
  - Behavioural identity for this slice's copy: `truncate_content(s, 240) == s[:237] + "..."` -> `True`, `len == 240`.
- Why it matters: this is the brief's "dead helper with >=10 re-rolls" shape exactly. The drift is real and user-visible in the copies: some use `"..."`, some `"…"` (one codepoint vs three), some truncate to `limit` and some to `limit - 1`, and several (`UI/STTS_Window.py:679,842`) append the ellipsis *unconditionally*, so a 40-character string is shown as `"short text..."` with nothing elided.
- Recommended correction: import `Utils.Utils.truncate_content` at the copies whose semantics match (the `-3` family), or move a single `truncate(text, limit, *, marker="…")` into `Utils/Utils.py` beside `elide_path_middle` and retire the rest. Canonical home: `Utils/Utils.py`, where the helper already is.
- Size: M (52 call sites, and the `"…"` vs `"..."` marker has to be chosen) · ADR: no · Confidence: **verified**
- Pinning test: none on `truncate_content` (`grep` above returns nothing from `Tests/` either).
- Already covered: none. NOTE: this is a repo-wide class; a sibling slice reviewing `UI/` or `Tools/` will see the same helper. Reported here because the census is in my evidence and `provider_failures.py:53` is my row.
### P2 [D4a] — `Widgets/base_components.py` (681 lines, 5 widget classes + 2 factories) has zero production importers; its `create_form_field` duplicates the live one in `form_components.py`  ·  _slice: W-top_
- Where: `tldw_chatbook/Widgets/base_components.py` — whole file. Duplicate surface: `base_components.create_form_field:652` vs `form_components.create_form_field:22`; `base_components.create_button_row:635` vs `form_components.create_button_group:131`; `base_components.FormField:41` (a dataclass) vs `form_components.FormField:175` (a Container).
- Evidence (AST, not grep):
  ```
  cd $WT && $PY -  # ast.walk every tldw_chatbook/**/*.py for Import/ImportFrom of base_components
  AST ImportFrom/Import of base_components in tldw_chatbook/: 0
  grep -rn base_components Tests/ -> 3 files, all keeping the dead code alive:
    Tests/UI/test_focus_token_parity.py:22        from ...base_components import NavigationButton
    Tests/UI/test_widget_css_consolidation.py:1120-1124  5 (file, class, "DEFAULT_CSS") rows
    Tests/UI/test_non_obscuring_focus_contract.py:17     BASE_COMPONENTS = ROOT / ".../base_components.py"
  ```
- Why it matters: 681 lines of widget code that nothing composes, plus three CSS/focus-contract tests whose only job is to police it — every future focus-token or CSS-bundle sweep pays for a file no screen renders. It is also the second `create_form_field` in the same package, which is what sends the next author to the wrong one.
- Recommended correction: delete the module and the three test references (the two CSS tests lose 6 rows; `test_focus_token_parity` needs a live widget instead). Canonical home for anything a future caller wants: `Widgets/form_components.py` (4 importers, live).
- Size: S · ADR: no · Confidence: verified
- Pinning test: the three above — none states the behaviour as a *requirement*, they assert CSS-token hygiene of whatever classes exist.
- Already covered: none

---
### P2 [D4a] — `_apply_console_settings_summary_state` re-derives temperature/max_tokens by regex from a formatted display string although the summary state carries them as structured fields  ·  _slice: UI-chat_
- Where: `chat_screen.py:9404-9411` (`re.search(r"T ([\d.]+)"…)`, `re.search(r"max_tokens (\d+)"…)` over `summary_state.sampling_row`) vs `Chat/console_session_settings.py:774-775` (`temperature: str`, `max_tokens: str` — "TASK-32338: structured fields … replace regex-parsing of the formatted sampling_row").
- Evidence: `grep -n "summary_state.temperature" UI/Console_Modules/left_rail.py` → 2281 uses the structured field; the screen path still parses. Same defect family as finding 1 (its dead half), listed separately because it is a helper-exists-ignored duplication that would survive a naive fix of finding 1.
- Why it matters: two derivations of one value; the regex silently yields "—" the moment the copy changes — exactly the failure TASK-32338 documented.
- Recommended correction (S): use the fields; delete the two `re.search` calls. Canonical home: `ConsoleSettingsSummaryState` (exists).
- Size: S · ADR: no · Confidence: verified (read + finding 1's probe) · Pinning test: none · Already covered: none.
### P2 [D4a] — `config.coerce_bool_setting` (24 importers) is re-rolled 21 times, and the canonical copy has the strangest vocabulary  ·  _slice: UTILS_
- Where: canonical `config.py:1201 coerce_bool_setting` → `_get_typed_value` (`config.py:1023-1027`: `str(value).lower() in ["true","1","t","y","yes"]` — so `"on"` → False and int `2` → False). Copies: `rg -n "^\s*def _?(coerce|to|as|parse|normalize|read)_?bool\w*\("` → 21 (`Character_Chat/world_book_manager.py:48`, `Character_Chat_Lib.py:158`, `Chat_Dictionary_Lib.py:95`, `world_book_import.py:34`, `world_info_processor.py:20`, `Home/home_rail_state.py:24`, `Chat/console_rail_state.py:335`, `Library/library_rail_state.py:42`, `Chunking/engine/option_utils.py:10`, `Chunking/engine/templates.py:46`, `Chunking/lab_state.py:198`, `Video_Generation/config.py:333`, `Image_Generation/config.py:466`, `Chat/message_metadata.py:510`, `MCP/unified_control_models.py:501`, `UI/Screens/settings_appearance_defaults.py:91`, `LLM_Management/snapshot_admission.py:458`, and in this slice `Utils/adaptive_reader_state.py:154` and `Utils/console_background_effects.py:43`).
- Evidence: bodies read for five: Character_Chat maps ints/floats by `!= 0` and accepts `on/off`; `home_rail_state` int `!= 0` + `_TRUE_STRINGS`; `option_utils` falls to `bool(value)` (so `[]`→False, `"maybe"`→`is_truthy`); the two Utils copies return `default` for any non-bool non-str (int `1` → default); config's canonical rejects `on`. Two more vocabularies live in this slice: `egress._config_enabled` (`egress.py:113`: anything not in `false/0/no/off` is True) and `tls_trust._TRUE_STRINGS/_FALSE_STRINGS` (`tls_trust.py:36-37`: `"yes"` is neither, so it is treated as a CA-bundle PATH and logged as an error).
- Why it matters: the same TOML value (`enabled = "on"`, `enabled = 1`) is True in one section and False/default in another; three of the readers are config-persisted preference loaders.
- Recommended correction: a stdlib-only leaf `Utils/coerce.py:coerce_bool(value, default)` with the union vocabulary (`true/1/t/y/yes/on` · `false/0/f/n/no/off` · ints by `!= 0`), `config.coerce_bool_setting` delegating to it (config.py already imports `Utils.adaptive_reader_state`/`console_background_effects` at module top — `config.py:75,81` — so the leaf must not import config, which is the same constraint those two files document).
- Size: M · ADR: no · Confidence: verified (definitions and vocabularies); inferred (which persisted keys are actually read through a divergent copy)
- Pinning test: `Tests/Utils/test_config_nested_settings.py` (config side, not read); none for the copies as a set.
- Already covered: none
### P2 [D4a] — `thaw_json` (Chat/console_prepared_request.py:110) is re-rolled five times under private names; one copy drifts; two of the re-rolling modules already import from `console_prepared_request`  ·  _slice: CHAT-bridge_
- Where: `console_trace_service.py:6565 _thaw`, `console_trace_final_values.py:779 _thaw`, `console_voice_trace_gateway.py:351 _thaw`, `console_provider_gateway.py:1745 _thaw_auxiliary_value` (all byte-identical modulo name), `console_runtime.py:586-591` nested `thaw` (DRIFTS: does not `str()` mapping keys); canonical `console_prepared_request.py:110 thaw_json`
- Evidence: `sed -n 100,130p console_prepared_request.py; sed -n 6565,6571p console_trace_service.py; sed -n 351,356p console_voice_trace_gateway.py; sed -n 779,784p console_trace_final_values.py; sed -n 1745,1752p console_provider_gateway.py; sed -n 586,591p console_runtime.py`. `console_trace_service.py:16` and `console_provider_gateway.py:65-76` already import from `console_prepared_request` (the gateway even imports `thaw_json` itself at :75 and still defines `_thaw_auxiliary_value`). `rg "_thaw\b|_thaw_auxiliary_value" Tests/` → 0 hits.
- Why it matters: `_artifact_bytes` (trace_service:6573) — the byte-identity oracle every trace verification compares against — depends on the private copy; an edit to one copy (e.g. handling `list`/`set`) silently changes which artifacts match in one verifier and not the others. The runtime copy already differs.
- Recommended correction: delete the five private copies; `from tldw_chatbook.Chat.console_prepared_request import thaw_json`. Canonical home: `Chat/console_prepared_request.py` (existing).
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none
### P2 [D4a] — `truncate_content` (0 importers) beside 24 inline `[:n] + "..."` truncations and ~20 private `_truncate*/_ellipsize` helpers with drifting ellipsis glyphs  ·  _slice: UTILS_
- Where: helper `Utils/Utils.py:253`; inline sites listed in `<SCRATCH>/candidates/patterns/inline_truncate.tsv` (24 rows: `Chat/console_fleet_wake.py:132`, `Chat/provider_failures.py:53`, `MCP/server.py:762`, `MCP/tools.py:252`, `RAG_Search/simplified/vector_store.py:642,1394`, `Tools/code_audit_tool.py:286,429,487,531`, `Widgets/tool_message_widgets.py:101,177,185,195`, … ); named helpers: `rg -n "^\s*def _?(truncate|elide|shorten|ellips)\w*\("` → 25 definitions (e.g. `Agents/run_hooks.py:193 _truncate`, `Event_Handlers/ingest_utils.py:30 _truncate_text`, `Home/dashboard_state.py:71`, `Widgets/Console/console_style_picker_modal.py:126`, `Chat/console_environment_state.py:529 _ellipsize`, `UI/MCP_Modules/mcp_tools_mode.py:110 _ellipsize`, `Chat/console_fleet_wake.py:88 _truncated`, …).
- Evidence: `rg -n "\btruncate_content\b" tldw_chatbook Helper_Scripts scripts Tests` → only the definition; `rg truncate_content <SCRATCH>/phase2/utils_collect.txt` → 0. Inline TSV shows 20 sites use `"..."` and 4 use `"…"` (`review_selection.py:460`, `console_transcript.py:1112`, `personas_character_editor_widget.py:1691`, `personas_preview_pane.py:267`).
- Why it matters: user-visible strings truncate with two different glyphs (`...` is 3 cells, `…` is 1) inside the same screens; the shared helper that would settle it is dead.
- Recommended correction: adopt — one public `truncate(text, budget, ellipsis="…")` in `Utils/text.py` (or rename `truncate_content`), mechanical swap at the 24 inline sites; leave the cell-width-aware helpers (`truncate_console_row_cells`, `ellipsize_note_title_cells`, `elide_path_middle`) alone — they are a different contract. If nobody will do the sweep, delete `truncate_content` instead of keeping a dead helper.
- Size: M · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none
### P2 [D4a] — the MLX-LM server is launched with a bare `"python"` resolved from `PATH`, and its failure is invisible because the subprocess's stderr goes to `DEVNULL`  ·  _slice: EVENTS_

- Where: `Event_Handlers/LLM_Management_Events/llm_management_events_mlx_lm.py:88-98` — `command = ["python", "-m", "mlx_lm.server", "--model", …]`.
- Evidence: the three sibling providers do not do this. ONNX takes an explicit interpreter from the form (`llm_management_events_onnx.py:138  command = [python_path, script_path]`, fed by `#onnx-python-path`); vLLM makes the user pick one (`#vllm-python-path`, `llm_management_events_vllm.py:175-190`, and `UI/LLM_Management/vllm_setup.py:714  python_path.with_name("vllm")`). MLX has no interpreter control at all — `rg -n "mlx-python" tldw_chatbook/UI/LLM_Management_Window.py` → nothing; the only MLX path input is `#mlx-model-path:1432`. And the repo already has the pinned-interpreter idiom in six places, documented at `Notes/file_notes_git_network.py:1440` ("the running interpreter (`sys.executable`) is pinned") — `Tools/file_operation_tools.py:1437`, `Tools/workspace_tool_executor.py:172`, `Audio/system_audio_tap.py:166,204`, `Audio/diarizer_local.py:296`.
- Why it matters: macOS — the only platform MLX runs on — ships no `/usr/bin/python`; `python` exists only if a venv is active in the shell that launched the TUI or the user installed one. A pipx/uv-tool install therefore gets `FileNotFoundError`, and a system-python hit gets `No module named mlx_lm`. Either way `run_server_subprocess` (`server_lifecycle.py:559-562`) sets `stdout=DEVNULL, stderr=DEVNULL`, so the only user-visible trace is the optimistic `app.notify("MLX-LM server starting…")` at `:121` followed by silence — the actual reason never reaches the UI or the log. (On this review box `command -v python` happens to resolve into the checkout's venv, which is exactly why this would pass a developer's manual test.)
- Recommended correction: `sys.executable` in place of `"python"` (one word, matching the six precedents), or add an `#mlx-python-path` input like ONNX's. Separately worth considering: `run_server_subprocess` capturing the child's first N stderr bytes for the destination log — today a launch failure of *any* provider is indistinguishable from a silent one.
- Size: S · ADR: no · Confidence: verified (the code fact and the DEVNULL consequence); **inferred** for "the user hits it", since that depends on their `PATH` — see UNVERIFIED for the check.
- Pinning test: none asserts the argv's first element.
- Already covered: none.
### P2 [D4a] — the RAG embeddings wrapper re-rolls ONE row of `Embeddings_Lib`'s bare-id→HF-path table; the other 13 rows (including `RAGConfig`'s own default model and the Settings placeholder) go to the Hub unqualified  ·  _slice: RAG_
- Where: `tldw_chatbook/RAG_Search/simplified/embeddings_wrapper.py:141-143` (`_BARE_HF_MODEL_ID_ALIASES`, one entry) applied at `:432` inside `_build_config`. The canonical table it duplicates a row of is `tldw_chatbook/Embeddings/Embeddings_Lib.py:936-1000` (`get_common_embedding_models()`), plus `:923-930` (`get_default_embedding_config()`).
- Evidence (isolated env):
  ```
  canonical table rows whose ID is bare but whose HF path is org-prefixed: 14
    'mxbai-embed-large-v1' -> 'mixedbread-ai/mxbai-embed-large-v1'   covered by RAG alias table: False
    'e5-small-v2'          -> 'intfloat/e5-small-v2'                 covered by RAG alias table: False
    'all-MiniLM-L6-v2'     -> 'sentence-transformers/all-MiniLM-L6-v2'  covered by RAG alias table: True
    'bge-base-en-v1.5'     -> 'BAAI/bge-base-en-v1.5'                covered by RAG alias table: False
    ... (14 rows, 1 covered)
  ```
  and the mechanism, calling `_build_config` directly:
  ```
  'mxbai-embed-large-v1'   -> model_name_or_path='mxbai-embed-large-v1'
  'all-MiniLM-L6-v2'       -> model_name_or_path='sentence-transformers/all-MiniLM-L6-v2'
  'bge-base-en-v1.5'       -> model_name_or_path='bge-base-en-v1.5'
  RAGConfig() default embedding model: mxbai-embed-large-v1
  ```
- Why it matters: `embeddings_wrapper.py`'s own comment states the consequence for exactly this shape — a bare id "silently 404-ing into the dim=768 default". The uncovered ids are not hypothetical: `simplified/config.py:290 EmbeddingConfig.model` defaults to `"mxbai-embed-large-v1"`, `UI/Screens/settings_library_rag_defaults.py:61` carries the same default, and `UI/Screens/settings_screen.py:19019` shows `placeholder="e.g. mxbai-embed-large-v1"` — so the string a user is most likely to type into the embedding-model field is one of the 13 the alias table does not cover, while `Embeddings_Lib` has known its correct HF path all along. Every shipped built-in profile overrides the model with an id that happens to be covered or already prefixed (`config_profiles.py:260,282,304,326,348`), which is why this has not been hit in the default flow.
- Recommended correction: D4 sub-case (a) — the helper exists and is ignored. Replace `_BARE_HF_MODEL_ID_ALIASES` with a lookup into `Embeddings_Lib.get_common_embedding_models()` (`model_name_or_path` of the matching row), keeping the existing rule that only the HTTP-facing id is rewritten and never `self.model_name` (which determines the collection fingerprint — see `collection_fingerprint._index_fields`). Canonical home: `Embeddings/Embeddings_Lib.py`.
- Size: S · ADR: no · Confidence: verified for the divergence and for the unqualified id `_build_config` emits; **inferred** for the 404 itself (no network in this review).
- Pinning test: none found for the alias table's coverage.
- Already covered: none (task-640 AC#7 added the one-row table; nothing covers the other 13)
### P2 [D4a] — the Settings "Test" button in `server_switch_modal.py` sends the user's API token to a user-entered URL without the app's egress gate  ·  _slice: W-persona-settings-chat_
- Where: `Widgets/Settings_Widgets/server_switch_modal.py:207-232` (`_run_connection_test`): `async with httpx.AsyncClient(timeout=5.0) as client: reach = await client.get(f"{url}/docs")`, then `await client.post(f"{url}/api/v1/sync/send", headers={"X-API-KEY": token}, json={})`.
- The helper that exists and is ignored: `Utils/egress.py` (`check_url_or_raise_async(url, trusted_origins=origin_set(origin))`), whose module docstring states the exact rule for this case — "OR its hostname is in `trusted_origins` (a host the USER explicitly typed/configured) … Metadata endpoints are stricter: blocked even for trusted origins". Its adopter count is 20+ modules; the **sibling settings probe for the same kind of action** does it correctly: `UI/Screens/settings_endpoint_probe.py:43-47, 513-522` calls `check_url_or_raise_async(..., trusted_origins=origin_set(endpoint.origin))` and constructs the client with `follow_redirects=False` (:526-529).
- Evidence: `grep -rln "Utils.egress" tldw_chatbook/` → 20+ modules, `server_switch_modal.py` not among them; `grep -n "egress\|guarded_fetch\|trusted_origin" Widgets/Settings_Widgets/server_switch_modal.py` → no hits.
- What the modal *does* validate (`_validated_root_url`, :162-192): scheme ∈ {http, https}, non-empty netloc, no path/query/fragment, then `Utils/input_validation.validate_url` — which by its own docstring validates *shape* only ("long TLDs, IPv6 literals, IDN/Unicode hosts, IPs, and localhost all validate"). It has no IP-class or metadata-endpoint rule; that is what `egress` is for.
- Why it matters: this is a credential-bearing outbound request (`X-API-KEY: <the user's token>`) to a host the modal never runs past the app's own network policy, from a settings surface whose direct sibling does. Mitigating: httpx defaults to `follow_redirects=False`, so a redirect cannot relocate the token.
- Recommended correction: one `await check_url_or_raise_async(url, trusted_origins=origin_set(origin_of(url)))` before the client block, and construct the client `follow_redirects=False` explicitly, matching `settings_endpoint_probe`.
- Size: S · ADR: no · Confidence: verified (mechanism; no exploit attempted)
- Pinning test: none in `Tests/UI/test_settings_endpoint_probe.py` covers this modal.
- Already covered: task-586 (image-gen) and task-609 (skill remote fetch) name other egress adopters, not this one.
### P2 [D4a] — the chat-attachment image decode path does not escalate PIL's decompression-bomb warning, while five other modules in this repo do — including the one that re-validates the same payload  ·  _slice: EVENTS_

- Where: `Event_Handlers/Chat_Events/chat_image_events.py:204-238` (`ChatImageHandler.prepare_image_payload`: `PILImage.open` → `thumbnail()` → `save()`), reached from `process_image_file:88,94` on every attached image.
- Evidence (reproduced; `MAX_IMAGE_PIXELS` lowered so a harmless 900×900 lands in PIL's warn-only band rather than allocating anything):
```
cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY - <<'EOF'
import asyncio, io, warnings
from PIL import Image as PILImage
from tldw_chatbook.Event_Handlers.Chat_Events.chat_image_events import ChatImageHandler
print("PIL default MAX_IMAGE_PIXELS:", PILImage.MAX_IMAGE_PIXELS)
buf = io.BytesIO(); PILImage.new("RGB",(900,900),"red").save(buf,format="PNG"); src = buf.getvalue()
PILImage.MAX_IMAGE_PIXELS = 700_000          # 810k px = between 1x and 2x -> warn-only band
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    out, mime = asyncio.run(ChatImageHandler.prepare_image_payload(src, ".png"))
print("RETURNED normally:", len(out), mime)
print("warnings seen, not escalated:", [w.category.__name__ for w in caught])
EOF
```
→ `PIL default MAX_IMAGE_PIXELS: 89478485` / `source PNG bytes: 4360` / `RETURNED normally: 4360 image/png` / `warnings seen, not escalated: ['RequestsDependencyWarning', 'DecompressionBombWarning']`
- The helper that is ignored: `Chat/console_chat_fork.py:798-805` does exactly the right thing for the *same* payload class — `with warnings.catch_warnings(): warnings.simplefilter("error", PILImage.DecompressionBombWarning)` around `open`/`verify`/`load` — and it imports `PAYLOAD_FORMAT_MIME` from this very module. Four more precedents: `Character_Chat/visual_identity.py:1891,1927-1928`, `Character_Chat/expression_set_io.py:563`, `Actor_Packs/contracts.py:713-739`, `Petdex/sources.py:195`. (`Tools/web_tool_impls.py:503-508` deliberately leaves the default *because that path never decodes pixels* — the opposite of this one.)
- Why it matters: PIL raises `DecompressionBombError` only past **2×** `MAX_IMAGE_PIXELS`; between 1× and 2× it emits a warning and decodes anyway. A flat-colour PNG of 12,000×8,000 (96 MP) is a few hundred KB — comfortably inside the 10 MB source cap enforced at `:75-81` — and decodes to ~288 MB before `thumbnail()` and `save()` add their own buffers. The attachment path is the first thing that touches a file the user did not create (downloaded, shared, pasted). The "Processed image too large" check at `:107` runs *after* the allocation and cannot prevent it.
- Recommended correction: wrap the `PILImage.open` … `save` block in `chat_image_events.prepare_image_payload` in the identical two lines `console_chat_fork.py:798-799` already uses, and let the existing `except Exception` at `process_image_file:97` turn it into the same user-visible rejection as any other processing failure. Canonical home: leave it inline (five sites now, all two-liners), or lift a `Utils` `bomb_guard()` context manager if a sixth appears.
- Size: S · ADR: no · Confidence: verified.
- Pinning test: none. `Tests/Event_Handlers/Chat_Events/test_chat_image_properties.py` uses Hypothesis on small images and would stay green.
- Already covered: none.
### P2 [D4a] — two contradictory, both-pinned credential-precedence decisions: chat spend is modern-config-first, `get_api_key` is env-first, and the shipped defaults make every bridged table carry an `api_key_env_var`  ·  _slice: ENTRY-config_
- Where: `tldw_chatbook/config.py:1573-1746` (`_normalize_legacy_provider_api_key`: modern `api_settings` > env > legacy `[API]`) vs `config.py:9074-9087` (`get_api_key`: `api_key_env_var` env > stored `api_key` > legacy > `<NAME>_API_KEY`). Template `[api_settings.<provider>]` tables at config.py 4140-4290 all ship `api_key_env_var`.
- Evidence: config `[api_settings.anthropic] api_key = " sk-ant-modern "` with `ANTHROPIC_API_KEY=sk-ant-from-env` →
  ```
  bridge anthropic_api.api_key    : 'sk-ant-modern'
  get_api_key('anthropic')        : 'sk-ant-from-env'
  ```
  Pins: `Tests/Chat/test_provider_readiness.py::test_modern_api_settings_key_outranks_the_env_var_for_the_spending_path` AND `Tests/Utils/test_config_api_key_resolution.py::test_env_var_named_by_config_wins_over_the_stored_key` (docstring: "Precedence WITHIN the branch is unchanged").
- Why it matters: with both a Settings-entered key and an env var set, `chat_api_call` spends with the Settings key while MCP tools / Console realtime spend with the env key — the same "readiness and spend disagree" class PR-T2 was opened to close, now between two config accessors rather than two readers. Because both orders are asserted by test NAME, this is a decision, not a bug fix.
- Recommended correction: one ADR ruling (extend ADR-012 §"Precedence") and then make `get_api_key` delegate to `_normalize_legacy_provider_api_key`'s order (or the reverse) — after the placeholder fix in finding 1, the only remaining difference is the order.
- Size: L · ADR: yes (new, or an amendment to `012-provider-credential-settings-boundary.md`) · Confidence: verified
- Pinning test: both named above; each states its own order as a requirement
- Already covered: none
### P2 [D4b] — "cap this text and append a truncation marker" is re-rolled 14 times with three different markers and two different budget semantics, and the drift reaches user-visible strings and tool results  ·  _slice: W-console-1_
- Where (this slice owns 2): `console_selection.py:13-14,137-140` (`SELECTION_QUOTE_CAP=4000`, marker `"\n… [truncated]"`, budget-inclusive) and `console_feedback_comment_modal.py:28-37` (`PREVIEW_QUOTE_CAP=600`, marker `"… [truncated]"` — **no leading newline**, budget-inclusive). The other 12: `Agents/run_hooks.py:181,197` (`"…[truncated]"` — no space), `Tools/watchlists_tool_service.py:63` (same no-space form), `Tools/web_tool_impls.py:1171`, `Tools/local_tool_impls.py:355,561`, `Agents/virtual_cli_provider.py:120`, `Agents/local_tool_provider.py:435,521`, `MCP/unified_control_plane_service.py:3840`, `UI/Research_Modules/bundle_rendering.py:131`, `Library/library_rag_answer_service.py:362`.
- Evidence:
  ```
  grep -rhon "…\s*\[truncated\]" tldw_chatbook --include='*.py' | sed 's/^[0-9]*://' | sort | uniq -c
       10 … [truncated]
        2 …[truncated]
  grep -rn "\[truncated\]" tldw_chatbook --include='*.py' | wc -l   → 14
  grep -rn "len(_TRUNCATION_MARKER)\|len(PREVIEW_TRUNCATION_MARKER)\|len(_TRUNCATION_SUFFIX)" … → 4 sites
  ```
  Two distinct budget semantics: 4 sites slice at `CAP - len(marker)` so the **result** is ≤ CAP; the other 10 slice at CAP and then append, so the result is `CAP + len(marker)`.
- Why it matters: this is not cosmetic in every copy. `Tools/local_tool_impls.py:355,561`, `Agents/local_tool_provider.py:435,521`, `Agents/virtual_cli_provider.py:120` and `MCP/unified_control_plane_service.py:3840` truncate **tool results that go on the wire to a model** against a byte budget the appended marker then exceeds — a cap that is documented as N and is actually N+15. The two in this slice are user-visible instead: the same quoted selection shows with a leading newline in one surface and without it in the other.
- Recommended correction: one `truncate_with_marker(text, *, budget)` in `Utils/` (no such helper exists — `ls tldw_chatbook/Utils/` has no text-truncation module), with the budget-inclusive semantics the 4 careful sites already use, and one marker constant. Migrate the 10 append-after-slice sites deliberately, since each one's cap becomes 15 characters tighter.
- Size: M · ADR: no · Confidence: verified
- Pinning test: none found for the marker text
- Already covered: none
### P2 [D4b] — 16 byte-size formatters, no shared public helper, drifting user-visible output  ·  _slice: UTILS_
- Where (all copies): `Utils/Utils.py:729 _format_size_bytes` (private; the only one with a `Utils` home), `Utils/file_handlers.py:574`, `Chat/attachment_core.py:64`, `Widgets/Console/console_transcript.py:639` (documented dup — P3 per brief), `Widgets/Console/console_video_capacity_modal.py:23`, `Widgets/Coding_Widgets/repo_tree_widgets.py:201`, `Tools/web_tool_impls.py:463`, `UI/CodeRepoCopyPasteWindow.py:1209`, `UI/Tools_Settings_Window.py:6421`, `Widgets/NewIngest/SmartFileDropZone.py:97`, `UI/Screens/settings_privacy_security.py:364`, `Widgets/enhanced_file_picker.py:431`, `Widgets/file_list_item_enhanced.py:72` and `:233` (verbatim twins in one file), `Library/library_ingest_state.py:609`. (`Third_Party/textual_fspicker` excluded.)
- Evidence: `rg -n "def _?(format|human|humanize|fmt)_?(file_?)?(size|bytes)\w*\(|def _?bytes_to_human|def _?human_readable_size|def _?format_byte" tldw_chatbook --glob '!Tests/**' --glob '!Third_Party/**'` → 16 definitions (listed above). Bodies read; drift table:

  | copy | "B" rendering | KB precision | labels | max unit | threshold/divisor |
  |---|---|---|---|---|---|
  | Utils.py `_format_size_bytes` | `512 B` | `.1f` | KB/MB/GB | GB | 1024/1024 |
  | file_handlers `_format_size` | `512.0B` (no space) | `.1f` | KB/MB/GB/TB | TB | 1024/1024 |
  | attachment_core / console_transcript | `512 B` | `.0f` | KB/MB | **MB (no GB)** | 1024/1024 |
  | console_video_capacity_modal | always `x.x MiB` | – | MiB only | MiB | – |
  | repo_tree / CodeRepoCopyPaste / Tools_Settings | `512.0 B` | `.1f` | KB..TB | TB | 1024/1024 |
  | SmartFileDropZone | `512.0 B` | `.1f` | KB/MB/GB | GB | **decimal thresholds (1_000_000) with binary divisors** — verified: 1,000,000 B → `1.0 MB`, 999,999 B → `976.6 KB` |
  | settings_privacy_security | `512 B` | `.1f` | **KiB/MiB/GiB/TiB** | TiB | 1024/1024 |
  | enhanced_file_picker | `512 B` | trailing zeros dropped | KB..YB | YB | 1024/1024 |
  | file_list_item_enhanced ×2 | `512.0 B` | `.1f` | KB..PB | PB | 1024/1024 |
  | library_ingest_state | `512 B` | `.1f` | KB..PB | PB | 1024/1024 |
  | web_tool_impls | `512 B` | `.1f` | KB..TB | TB | 1024/1024 |
- Why it matters: the same file is shown as `1.0KB`, `1.0 KB`, `1 KB`, `1.0 KiB` in different panes; SmartFileDropZone's mixed-base thresholds mislabel sizes between 1,000,000 and 1,048,575 bytes.
- Recommended correction: make `Utils/Utils._format_size_bytes` public (`format_size_bytes(size, *, precision=1, iec=False)`) and adopt it at the 15 sites; keep `console_transcript` as documented if its widget-dependency argument still holds.
- Size: M · ADR: no · Confidence: verified (definitions and outputs), inferred (that every copy is on a user-visible path — 2 are in dead/deprecated UI: `Tools_Settings_Window` is `DEPRECATED (TASK-1346)`, `CodeRepoCopyPasteWindow` not checked)
- Pinning test: none found for a shared formatter; `Tests/Utils/test_db_status_manager.py` pins `get_formatted_file_size`/`get_formatted_db_size_with_wal` output shape (so the Utils.py copy's format is a stated requirement).
- Already covered: none
### P2 [D4b] — Seven filename sanitizers / validators with different safety envelopes; two reach disk unvalidated  ·  _slice: UTILS_
- Where: `Utils/text.py:47 sanitize_filename` (2 importers), `Utils/Utils.py:511 normalize_title` (0 importers), `Utils/file_extraction.py:614 _sanitize_filename` + `:467 _is_valid_filename`, `Utils/path_validation.py:284 validate_filename` (8 importers; raises), `Utils/input_validation.py:1396 validate_filename` (0 importers; returns bool; second copy of the 22-name Windows reserved table at `path_validation.py:332-355`), `DB/Prompts_DB.py:5091` inline `re.sub(r"[^\w\-_ \.]", "_", …)`, `Canvas/models.py:53 _UNSAFE_FILENAME_CHARACTER`.
- Evidence: `rg -n "def _?(sanitize|safe|clean|normalize)_?(file)?_?(name|filename|title)\w*\("` + bodies read. Drift table (what each REJECTS or STRIPS):

  | copy | `/` `\` | `..` | NUL / control | Windows reserved | leading `.` | length cap | consumer re-validates? |
  |---|---|---|---|---|---|---|---|
  | text.sanitize_filename | strips | **keeps** | **keeps** | **keeps** | keeps | none | chatbook_creator: yes (`validate_filename` at :1175/:1184); audio_processing:1257/1262: **no** (used as the download filename) |
  | Utils.normalize_title | → `_` | collapses to `_` via `[^\w\-.]`→`_`… `..` survives | strips (non-ASCII/NFKD) | keeps | strips `_` only | none | dead |
  | file_extraction._sanitize_filename | strips | **keeps** | strips `\n\r\t` only | keeps | keeps | 100 | yes — dialog calls `path_validation.validate_filename` (`Widgets/file_extraction_dialog.py:547`) |
  | path_validation.validate_filename | raises | raises | NUL raises | raises | allowed | none | n/a (validator) |
  | input_validation.validate_filename | False | False | **NUL not checked** | False | allowed | 255 | dead |
  | Prompts_DB:5091 | → `_` | **keeps** (`.` allowed) | → `_` | keeps | keeps | none | **no** — becomes the zip `arcname` |
  | Canvas/models regex | not covered | not covered | rejects | keeps | keeps | – | (Canvas has its own 255-byte check in JS) |
- Why it matters: `Local_Ingestion/audio_processing.py:1257` and `DB/Prompts_DB.py:5091` write names to disk/archive with a sanitizer that keeps `..` and control characters (drift that reaches storage — D1 aspect); the two `validate_filename`s with the same name have opposite contracts (raise vs bool) and duplicated reserved-name tables that will drift.
- Recommended correction: one `Utils/path_validation.sanitize_filename(name, *, max_len=255)` (strip separators/control/NUL, collapse `..`, reject reserved, cap) that `text.sanitize_filename` and `file_extraction._sanitize_filename` delegate to; delete `input_validation.validate_filename` and `Utils.normalize_title`; adopt at Prompts_DB:5091 and audio_processing:1257.
- Size: M · ADR: no · Confidence: verified (bodies, importers, consumer re-validation at the two dialog/export seams); inferred (that a hostile `..` name actually reaches a path-join in audio_processing — the call site returns the name to a downloader; not traced further)
- Pinning test: `Tests/Utils/test_path_validation.py` (validate_filename); `Tests/Utils/test_file_extraction.py` (extractor); none for text.sanitize_filename.
- Already covered: none
### P2 [D4b] — The "borrow-and-retire a native sqlite connection on the worker thread" block exists three times with behavioural drift; two copies live in this screen  ·  _slice: UI-personas_
- Where: `personas_screen.py:9186-9217` (`_persona_visual_thread.work`), `:12031-12057` (`_visual_identity_thread.work`), and `tldw_chatbook/Backup_Recovery/dictionary_source_job.py:172-195`. All three: capture `getattr(db._local, "conn", None)` before the call; in `finally`, if it was `None` → `db.close_connection()`, assert `db._local.conn is None` and `threading.current_thread() not in _repository_participant(db).retiring_threads`, else raise `<domain>_native_not_retired`.
- Evidence: `sed -n '160,200p' Backup_Recovery/dictionary_source_job.py` (third copy); `grep -rn "db\._local\b" tldw_chatbook | grep -v tldw_chatbook/DB/` → personas_screen is the only non-DB/non-Backup_Recovery module touching `_local.conn` (Actor_Packs/Persona_Visual read `transaction_depth` only); `grep -rn "_repository_participant(" tldw_chatbook | grep -v Backup_Recovery/` → personas_screen ×2 + `Sync_Interop/sync_state_repository.py:205`.
- Drift: dictionary_source_job raises `bootstrap.RecoveryRequired` (durable recovery path); both screen copies raise plain `RuntimeError` and hang recovery state on the exception object (`error.result` at 9216; `error.result`, `.source_error`, `.cleanup_candidate_relpath`, `._visual_identity_retirement_error` at 12053-12056), which the callers then pick apart (12059-12063, 9219-9232). The visual-identity copy additionally skips the retiring-threads check for `db.is_memory_db` (12050); the other two do not. That reaches storage (which retirement failures are retained for later cleanup) → also a D1-class drift.
- Recommended correction: one `retiring_native_borrow(db, *, domain)` context manager in `Backup_Recovery/participants.py` (it already owns `_repository_participant` and the third copy), returning a structured outcome instead of attribute-stuffed exceptions; the screen's two `work()` closures call it.
- Size: M · ADR: yes (`126-complete-local-backup-and-recovery.md` L317 "Existing native borrowers retain their actual owner-thread lifetime" governs the behaviour; consolidation needs no new decision) · Confidence: verified (copies read side by side)
- Pinning test: none found for the retirement block itself (`grep -rn "native_not_retired" Tests/` not run — see UNVERIFIED)
- Already covered: none
### P2 [D4b] — The Media speaker-rename flow exists twice and the canvas copy is missing all four hardenings the reader copy documents  ·  _slice: W-library_
- Where: `library_media_canvas.py:629-706` vs `library_media_viewer.py:806-899` (same four methods, same names, same message flow).
- Evidence: `sed -n '629,706p' library_media_canvas.py` / `sed -n '806,899p' library_media_viewer.py`. Drift, reader→canvas:
  1. reader captures `media_id` at submit time ("a selection change mid-rename must not retarget the write"); the canvas reads `self.speaker_rename_media_id` *inside the thread worker* (:659) and again in the repaint (:691) — a selection change between submit and worker start retargets the write to another media item;
  2. reader passes `exclusive=True` ("keeps two fast submits from piling up"); the canvas (:651-656) does not;
  3. reader does both post-rename DB reads on the worker thread; the canvas does `get_media_by_id` + `_meeting_speaker_legend_rows` on the UI thread inside `call_from_thread` (:691, :699) — two sqlite reads, one of them the whole content blob, on the event loop;
  4. reader maps `outcome.reason` through `_RENAME_REFUSAL_COPY`; the canvas shows the raw reason string (:684).
- Why it matters: the same user gesture behaves differently in the list and in the reader, and #1 can write a rename onto the wrong item.
- Recommended correction: one shared `_submit_speaker_rename(...)` helper next to `Library/meeting_speaker_rename.py` (which already owns the persistence half) taking `(db, media_id, cluster_id, name)` and returning `(outcome, content, rows)`; both widgets keep only their own repaint. Canonical home: `tldw_chatbook/Library/meeting_speaker_rename.py`.
- Size: M · ADR: no · Confidence: verified (drift), inferred (the #1 race window)
- Pinning test: `Tests/UI/test_library_media_speaker_rename.py` and `Tests/UI/test_library_media_viewer_speaker_rename.py` — two parallel suites, neither cross-checks the other's behaviour.
- Already covered: none
### P2 [D4b] — The per-run approval stamp store is hand-rolled in five providers plus the built-in gate, with documented behavioural drift between copies  ·  _slice: AGENTS_
- Where (all copies): `builtin_tool_gate.py:141-233` (`begin_turn`/`stamp_scope`/`stamped`/`_stamp_detail`), `mcp_tool_provider.py:612-766` (`apply_batch_decisions`/`stamped_decision`/`_stamped_decision_detail`/`stamp_scope`), `local_tool_provider.py:1232-1296`, `virtual_cli_provider.py:365-418` (`apply_batch_decisions`/`_pop_stamp`/`_pop_stamp_detail`/`stamp_scope`), `raw_shell_tool_provider.py:345-459` (same + `authority_generation`); the `_root_is_valid` twin at `virtual_cli_provider.py:594` / `local_tool_provider.py:2456` and `_kill_switch_engaged` ×4 belong to the same provider-mirror family.
- Evidence: `rg -n "^    def (stamp_scope|apply_batch_decisions|stamped|stamped_decision|_stamp_detail|_stamped_decision_detail|_pop_stamp|_pop_stamp_detail)\(" tldw_chatbook/Agents/*.py` → 20 defs across 6 files. Drift, by reading: (a) `stamp_scope` CLEARS the run's slice on entry in local/virtual_cli/raw_shell (local_tool_provider.py:1265 calls this "a deliberate divergence from a pure snapshot") but NOT in mcp/builtin_gate (mcp:751-766, builtin:198-211); (b) peek semantics (`stamped`) in builtin/local/mcp vs pop-on-read (`_pop_stamp`) in virtual_cli/raw_shell; (c) stamps keyed by tool NAME in local/mcp (`for name, verdict in decisions`) vs `row.call_id or row.tool_name` in virtual_cli/raw_shell; (d) raw_shell alone guards restore with `authority_generation` (raw_shell:442-459).
- Why it matters: four copies of a lock-guarded dict with three different clear/peek/key policies is exactly the place the next "child clobbered the parent's verdict" regression (PR2a Task 5's original bug, documented at builtin_tool_gate.py:100-109) re-enters; each fix so far (per-run keying, F1 peek, call-id keying, generation guard) landed in a subset of the copies.
- Recommended correction: one `RunScopedStamps` in `Agents/approval_provenance.py` (already the shared home of `ApprovalStamp`/`approval_stamp`/`approval_key_unanswered`, 42 lines): `replace_run(run_id, decisions, *, key=...)`, `peek(run_id, key)`, `pop(run_id, key, fallback=None)`, `scope(run_id, *, clear_on_enter: bool)`, `clear_run(run_id)`, optional generation guard. Providers keep their public method names as one-line delegations so `Tests/Agents/test_gate_run_scoping.py` and the parity tests stay green.
- Size: M · ADR: yes — `032-local-agent-tool-permission-boundary.md` mandates the mirroring ("Mirrors MCPToolProvider's approval discipline"), not the copy; no ADR forbids a shared store. · Confidence: verified (reading; drift points cited by line)
- Pinning test: `Tests/Agents/test_gate_run_scoping.py` (per-run keying), `Tests/Agents/test_local_tool_provider.py::test_approve_once_stamp_does_not_persist` / `::test_persist_failure_does_not_block_execution`, `Tests/Agents/test_virtual_cli_provider.py:846` (all state the current per-provider behaviour as requirements — the consolidation must preserve each copy's policy, not unify it).
- Already covered: none.
### P2 [D4b] — Three cutoff-timestamp formats are compared against one column written in a fourth; rows soft-deleted on the cutoff's calendar date are skipped by the hard-delete cleanup  ·  _slice: DB-media-base_
- Where: writer `tldw_chatbook/DB/Client_Media_DB_v2.py:2210-2218` (`%Y-%m-%dT%H:%M:%S.mmmZ`, every `last_modified`/`trash_date` in the file); readers `:4351-4354` `hard_delete_old_media` and `:4457-4458` `get_deletion_candidates` compare `last_modified < '%Y-%m-%d %H:%M:%S'` (space separator, no ms/Z; `:4457` also uses `datetime.utcnow()`, deprecated in 3.12); `:9255-9257` `empty_trash` compares `trash_date <= '%Y-%m-%dT%H:%M:%SZ'` (no ms). SQLite compares TEXT lexically: `'T'` (0x54) > `' '` (0x20), so on the cutoff date every stored value sorts AFTER the cutoff.
- Evidence: `$PY $HOME/repro/media_repro3.py` → `cutoff (module format): 2026-08-19 02:27:11 | stored: 2026-08-19T00:00:01.000Z | stored is older by 2:27:10` … `get_deletion_candidates(days_old=30) -> 0 row(s) [expected 1]` … `hard_delete_old_media(days_old=30) -> 0 deleted [expected 1]` … `same predicate, cutoff in the STORED format -> 1 row(s)` … `lexical proof: '2026-08-19T00:00:01.000Z' < '2026-08-19 02:27:11' = False`.
- Why it matters: the shipped cleanup (`app.py:19860,19883` → `run_cleanup_method(db.get_deletion_candidates / db.hard_delete_old_media, cleanup_days)`) lags by up to 24 h and its candidate count disagrees with a same-format query; `empty_trash`'s no-ms form is off by ≤1 s in the other direction. Not user-visible today, but it is storage-reaching drift and the `utcnow()` line will start warning.
- Recommended correction: derive every cutoff through the one writer — e.g. a `_utc_timestamp_str(dt)` classmethod used by `_get_current_utc_timestamp_str()` and the three cutoffs — so there is exactly one format. Canonical home: this file's existing helper (or `Utils/` if ChaChaNotes/Prompts share the `%Y-%m-%dT%H:%M:%S.%f`+`Z` form — out of my slice). S.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/Media_DB/test_media_db_v2.py:1077` and `Tests/RAG/test_ingestion_indexing.py:351` call `hard_delete_old_media(days_old=-1)` — a negative age that never lands on the boundary date, so they do not pin (or contradict) this.
- Already covered: none
### P2 [D4b] — `_CHOICE_LABELS` is duplicated byte-for-byte across the canvas and its state module, and the two copies are compared against each other  ·  _slice: W-library_
- Where: `library_notes_add_from_files_canvas.py:63` and `Library/library_notes_lasting_sync_state.py:144`.
- Evidence: both dicts printed side by side → `IDENTICAL: True`. The coupling is live: the state module builds `row.selected_label` from ITS copy (`:278`, `:863`) while the canvas decides the "✓" tick and `is-selected` class from ITS copy (`library_notes_add_from_files_canvas.py:756`, `:1316`).
- Why it matters: a rename in one file silently stops the selected choice ever ticking.
- Recommended correction: export the state module's map (it already owns the enum) and delete the canvas copy; key `_CHOICE_SLUGS`/`_CHOICE_EFFECTS` off it too. Canonical home: `Library/library_notes_lasting_sync_state.py`.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none
### P2 [D4b] — `_datetime_to_iso` is 4 verbatim copies inside MCP (7 counting `_iso_utc_now`, `_format_utc`, `runtime_policy/source_state.py:133`), the `replace("+00:00","Z")` idiom is re-rolled 44× in 38 files repo-wide with no Utils helper, and THREE timestamp formats reach the MCP stores/wire  ·  _slice: TOOLS-MCP_
- Where (copies): `MCP/unified_context_store.py:78`, `MCP/server_target_store.py:372`, `MCP/local_store.py:140`, `MCP/unified_control_models.py:15` (byte-identical), `MCP/permission_store.py:224` (`_iso_utc_now`), `Tools/watchlists_tool_service.py:2203` (`_format_utc`), `runtime_policy/source_state.py:133`
- Where (drift reaching storage/wire): `MCP/unified_control_plane_service.py:2332` writes `datetime.now(timezone.utc).isoformat()` (`+00:00` form) into `profile_runtime_state.last_attempt_at/last_ok_at` of `local_mcp_store.json`, stored raw (`local_store.py:933 dict(record)`) beside the store's own `Z`-form `updated_at`; `MCP/execution_log.py:144` writes `+00:00` form to `mcp_execution_log.jsonl`; `MCP/client.py:1253` writes NAIVE LOCAL `datetime.now().isoformat()` as `connected_at`, which `local_control_service._describe_profile` (`:894`) persists via `save_discovery_snapshot` and `describe_server` (`client.py:1612`) returns to callers; `MCP/server.py:731` returns naive local `created` to MCP clients; `local_control_service.py:1099` writes `+00:00` form but round-trips through `_iso_to_datetime`/`_datetime_to_iso` so it self-heals to `Z`.
- Evidence: `rg -n 'replace\("\+00:00", ?"Z"\)' tldw_chatbook -g '*.py' -g '!Third_Party/**' | wc -l` → 44 (38 files); `rg -n '^def .*(iso|utc_now|timestamp).*\(' tldw_chatbook/Utils/*.py` → no matches; format census above from `rg -n 'isoformat\(|_datetime_to_iso\(|_iso_utc_now\(' tldw_chatbook/MCP/*.py`
- Why it matters: `server_target_store` already imports `unified_control_models` and still carries its own copy; a consumer comparing/sorting `last_attempt_at` against `updated_at` as strings compares `+00:00` against `Z`, and `connected_at` carries no zone at all (wrong by the host's UTC offset when read on another machine or after a DST change).
- Recommended correction: one `Utils/time_format.py::utc_iso_z(dt: datetime | None) -> str | None` (new; the four MCP bodies are already identical) and use it at the three drifting write sites; S per site, the census is a follow-up sweep.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none found for the format of `last_attempt_at`/`connected_at`
- Already covered: none
### P2 [D4b] — `_manual_json_digest` is a verbatim third copy of a storage-bound digest  ·  _slice: CHAT-controller_
- Where: `:22946-22954` (`ConsoleChatController._manual_json_digest`), `Chat/console_context_repository.py:1876-1883` (`_digest_json`), `Chat/console_context_compaction.py:2657-2664` (`_digest_json`). Controller callers: `:22957` `_repository_prefix_digest`, `:23091` `content_digest=` on `PersistedLineageFenceRow`, `:23143` `legacy_summary_digest=` on `MemorySelectionFence`.
- Evidence: `sed -n` of all three bodies — byte-identical (`json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")` → `sha256().hexdigest()`). `grep -rnE 'sort_keys=True' tldw_chatbook/Utils` → no shared helper exists. Repository writers use its own copy at `console_context_repository.py:1455,1590,1686`; the controller's copy produces the `expected_*` fences the repository compares against (`append_current_branch_reset_if_current`, `BranchMemoryCommit.durable_lineage`).
- Why it matters: the controller-side and repository-side digests must agree byte-for-byte or every branch-memory admission fails "conversation changed" — the drift would reach storage comparisons silently (no test compares the two implementations).
- Recommended correction: export one `digest_json` from `Chat/console_context_repository.py` (the storage owner) and import it in the controller and `console_context_compaction.py`; delete the two private copies.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/Chat/test_console_rewind_summarize.py` exercises the fences end-to-end (would go red on drift, not on duplication).
- Already covered: none.
### P2 [D4b] — `_maybe_await` is re-implemented 66 times across the package with no shared home  ·  _slice: W-console-1_
- Where (this slice): `console_prompts_modal.py:153`. Repo-wide: 66 definitions (`grep -rn "async def _maybe_await" tldw_chatbook --include='*.py' | wc -l` → 66).
- Evidence: normalizing each body and de-duplicating shows **no behavioural drift** — every copy is `await value if inspect.isawaitable(value) else value` in one of two spellings (ternary vs `if/return`), some as a `@staticmethod`, some module-level.
- Why it matters: 66 copies of a 2-line helper is the largest single re-roll in the tier; the reason to consolidate is not correctness (there is none to fix) but that the next variant will be the one that drifts — this is exactly the D2 "a pattern that will produce a P1" shape.
- Recommended correction: one `async def maybe_await(value)` in a new `Utils/async_helpers.py` (no such module exists today — `ls tldw_chatbook/Utils/` has no async helper), imported by all 66. Mechanical, no behaviour change.
- Size: M (66 files, one-line each) · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none
### P2 [D4b] — `_set_message_attachments` is byte-identical to `UI/Console_Modules/message.py::_apply_console_message_attachments`, and the Chat layer reaches *back into the UI package* for the copy with a cycle-breaking lazy import  ·  _slice: CHAT-store_
- Where: `tldw_chatbook/Chat/console_chat_store.py:10708-10729` (staticmethod, 9 in-store call sites); `tldw_chatbook/UI/Console_Modules/message.py:186-211` (module function, 1 UI caller at :926); `tldw_chatbook/Chat/console_conversation_hydration.py:138-140` does `from tldw_chatbook.UI.Console_Modules.message import _apply_console_message_attachments` inside a function body and calls it at :173.
- Evidence: `diff <(sed -n '10719,10729p' …console_chat_store.py | sed 's/^        //') <(sed -n '200,210p' …message.py | sed 's/^    //')` → identical (the only diff lines are the leading `rebased = tuple(` I clipped from the store slice and two trailing blanks). `grep -rn "_apply_console_message_attachments" tldw_chatbook --include='*.py'` → the three sites above. message.py's own docstring: "Mirrors ConsoleChatStore._set_message_attachments's invariant … outside the store, where that helper isn't reachable."
- Why it matters: the invariant ("every attachments mutation sets the tuple AND the three scalar mirrors together") now lives in two packages; a fix to one (e.g. a future `position` rebase rule) silently misses the other, and `Chat → UI` is an inverted dependency that only works because the import is deferred.
- Recommended correction: one module-level `apply_message_attachments(message, attachments)` in `tldw_chatbook/Chat/console_chat_models.py` (where `ConsoleChatMessage`/`MessageAttachment` live; no UI or store dependency); `ConsoleChatStore._set_message_attachments` and message.py's copy become one-line delegates or are deleted and the three callers import the model helper. Size S.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none (`grep -rln "_apply_console_message_attachments" Tests/` → no files)
- Already covered: none
### P2 [D4b] — `conversation_local_marks` accepts two UTC timestamp shapes, and the voice-promotion reconciler's validator rejects the shape the table's own public writer produces  ·  _slice: CHAT-rest-1_
- Where: writer A `Chat/conversation_local_marks_service.py:94-95 _now()` (`datetime.now(timezone.utc).isoformat().replace("+00:00","Z")`), used by the public `set_mark()` at `:233`. Writer B `chat_persistence_service.py:453` and `:879`, `console_dispatch_repository.py:1058` — `db._get_current_utc_timestamp_iso()` (`DB/ChaChaNotes_DB.py:8700`, `timespec="milliseconds"`). Validator: `chat_persistence_service.py:195-212 _is_canonical_utc_timestamp`, used at `:700` and `:767`; failure raises `RuntimeError("Voice promotion persistence conflict.")` at `:723`.
- Evidence: `cd $WT && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY -c "..."` →
  ```
  marks_service._now()                   len=27 value='2026-09-18T14:46:16.004249Z' accepted=False
  db._get_current_utc_timestamp_iso()    len=24 value='2026-09-18T14:46:16.004Z'    accepted=True
  marks _now() at microsecond==0         len=20 value='2026-09-18T12:00:00Z'        accepted=False
  set_mark() accepts receipt mark type -> 'console_unseen:11111111-1111-4111-8111-111111111111'
  set_mark() accepts outcome mark type -> 'console_terminal_outcome:<uuid>:complete'
  ```
- Why it matters: `_is_canonical_utc_timestamp` hard-codes "length 24 ending in Z" or "length 19"; the marks service's own `_now()` produces neither (27 normally, 20 at microsecond==0). Today every *receipt* mark happens to be written through the cursor-scoped path that uses the DB helper, so nothing breaks — but `set_mark()`'s `_mark_type()` validates and **accepts** exactly the receipt mark types the validator later inspects (reproduced above), so one call through the public API turns an idempotent voice-promotion retry into `RuntimeError` on an already-committed row. Same class the CHAT-bridge sibling rated D4b, with a concrete single-table consumer attached.
- Recommended correction: delete `ConversationLocalMarksService._now()` and use `self.db._get_current_utc_timestamp_iso()` (already the canonical helper and already used by the other three writers); alternatively make `_is_canonical_utc_timestamp` parse rather than length-match. Canonical home: `DB/ChaChaNotes_DB._get_current_utc_timestamp_iso`.
- Size: S · ADR: no · Confidence: verified for the shape mismatch and the API acceptance; the *reachable* failure is inferred (no shipped caller passes a receipt mark to `set_mark()` today)
- Pinning test: none (`rg _is_canonical_utc_timestamp Tests/` → no hits)
- Already covered: none
### P2 [D4b] — strict-JSON loading (`object_pairs_hook` duplicate-key reject + `parse_constant` NaN/Infinity reject) is re-rolled with 15 `_reject_*constant` functions and 40 `parse_constant=` call sites repo-wide, two of them in this slice  ·  _slice: TOOLS-MCP_
- Where (this slice): `tldw_chatbook/MCP/permission_store.py:388-399` (`_reject_duplicate_keys` + `_reject_json_constant`), `tldw_chatbook/Tools/workspace_tool_protocol.py:407-417` (`_reject_duplicate_keys` + `_reject_non_finite`), `tldw_chatbook/Tools/watchlists_tool_service.py:1257` (`_unique_json_object`, no constant hook); repo census: `Petdex/sources.py:68`, `Actor_Packs/export.py:718`, `LLM_Calls/hosted_chat.py:633`, `LLM_Calls/qwencloud.py:301`, `LLM_Calls/qwencloud_streaming.py:61`, `Persona_Visual/{validation:336,repository:1368,publication:1306,importer:1090}`, `DB/private_sqlite_protocol.py:351`, `Audio/voice_process_protocol.py:556`, `Character_Chat/visual_identity.py:1658`, `TTS/audio_cpp_contract.py:87`
- Evidence: `rg -n 'parse_constant=' tldw_chatbook -g '*.py' -g '!Third_Party/**' | wc -l` → 40; `rg -n 'def _reject_(json_constant|non_finite|nan|constant)'` → 15 defs; `rg -ln 'object_pairs_hook' tldw_chatbook/Utils` → none
- Why it matters: each copy raises a DIFFERENT exception type (`PermissionStoreSnapshotError("invalid_json")`, `WorkspaceProtocolError`, `ValueError`), so the same malformed input is classified differently per boundary, and `watchlists_tool_service` rejects duplicate keys but still accepts `NaN` cursors (`json.loads` default) — the drift the hook exists to prevent.
- Recommended correction: `Utils/strict_json.py::loads_strict(raw, *, error: type[Exception])` wrapping `json.loads(..., object_pairs_hook=..., parse_constant=...)`; S per site.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none for the shared shape
- Already covered: none
### P2 [D4b] — the "mount a prepared thumbnail renderable" block is copy-pasted 3× in this slice (4× in the repo) with no helper  ·  _slice: W-persona-settings-chat_
- Where: `Persona_Widgets/personas_inspector_pane.py:1194-1236` (`set_avatar_thumbnail`), `Persona_Widgets/personas_character_editor_widget.py:1266-1310` (`set_avatar_thumbnail`) and `:1350-1392` (`set_expression_thumbnail`). The fourth is `UI/Screens/chat_screen.py:12749-12785` (`_build_character_avatar_widget`), which the other three name in their own comments ("same fallback as `ChatScreen._build_character_avatar_widget`").
- Evidence: `grep -rn "explicit_cell_size" tldw_chatbook/ | grep -v Utils/mosaic_render.py` → 4 distinct call sites (`chat_screen.py:12777`, `personas_inspector_pane.py:1226`, `personas_character_editor_widget.py:1297` and `:1379`). Each is preceded by the same `holder.remove_children()` → `isinstance(renderable, Widget)` short-circuit → `Static(renderable)` → `explicit_cell_size(...) or (BOX_COLS, BOX_LINES)` sequence, and three of them carry a verbatim copy of the same 4-line comment ("Per explicit_cell_size's documented contract, fall back to the box dimensions…").
- Behavioural drift to note: the two in `personas_character_editor_widget.py` re-import `Widget` and `Static` **inside the function** (`:1284-1285`, `:1368-1369`) even though both are already imported at module scope (`:22`, `:23`); the inspector copy re-imports only `Widget` (:1214) and uses the module-level `Static`. Same behaviour, three spellings.
- Recommended correction: one `mount_thumbnail(holder, renderable, *, box_cols, box_lines)` in `Utils/mosaic_render.py`, which already owns `explicit_cell_size` and its documented `None` contract. All four call sites pass only the holder and the box constants.
- Size: S · ADR: no · Confidence: verified
- Already covered: none.
### P2 [D4b] — the Ollama handlers hand-write the same async-staleness guard 28 times across 9 handlers (~750 of the file's 1,053 lines)  ·  _slice: EVENTS_

- Where: `Event_Handlers/LLM_Management_Events/llm_management_events_ollama.py:273-1053` — `handle_ollama_{list_models,show_model,delete_model,copy_model,pull_model,create_model,push_model,embeddings,ps}_button_pressed`.
- Evidence (counts over the one file):
```
rg -c '_begin_async_presentation'  → 9
rg -c '_owns_async_presentation'   → 28
rg -c '"generation" in locals'     → 18
rg -c 'asyncio.to_thread'          → 9
```
and repo-wide `rg -c "_owns_async_presentation" --glob '*.py' tldw_chatbook/` → this file 28, `llm_management_events_transformers.py` 2, `UI/LLM_Management_Window.py` 1 (the definition, `:2082`).
- The shape repeated verbatim in every one of the nine: `query_one` the inputs → validate-empty + `focus()` + `notify` → `clear()` → `generation = window._begin_async_presentation(channel)` → `del` the widget refs → `await asyncio.to_thread(<ollama api fn>, …)` → `if not window._owns_async_presentation(channel, generation): return` → re-`query_one` → `if error / elif data / else` → **two** `except` blocks (`QueryError`, `Exception`) each re-testing `if "generation" in locals() and not window._owns_async_presentation(channel, generation): return` before logging a category and notifying.
- Why it matters: the guard is the mechanism that stops a stale API reply from writing into a re-navigated destination. Hand-written 28 times, one miss is a silent wrong-pane write, and `"generation" in locals()` is exactly the idiom that breaks quietly when someone reorders a line. `_begin_async_presentation`/`_owns_async_presentation` (`UI/LLM_Management_Window.py:2075-2088`) are two lines each and already the right primitives — what is missing is the wrapper that uses them.
- Recommended correction: one `@asynccontextmanager async def _owned_presentation(window, channel)` (or an `async def _run_ollama_action(window, app, channel, fn, /, **kwargs)`) beside them in `UI/LLM_Management_Window.py`, yielding the generation and swallowing the stale case; each handler keeps only its input reads and its three result branches. Follow `backlog/docs/library-decomposition-recipe.md` §1 (per-subsystem PR) if this is split out.
- Size: M · ADR: no · Confidence: verified.
- Pinning test: `Tests/ProductionApp/test_llm_destination_actions.py` covers the destination-ownership behaviour these guards implement; a shared helper is the safe refactor target.
- Already covered: none.
### P2 [D4b] — the SQL-LIKE metacharacter escape is re-rolled 9 times with no shared helper, and one copy uses a different escape character  ·  _slice: DB-rest_
- Where (all copies, `rg -n 'replace\("%", "\\\\%"\)|replace\("%", "!%"\)' --type py tldw_chatbook`):
  - named `@staticmethod` helpers, byte-identical body + byte-identical docstring: `DB/Prompts_DB.py:3325-3327` `_escape_library_prompt_like`, `DB/Client_Media_DB_v2.py:8607-8609` `_escape_library_like`, `DB/ChaChaNotes_DB.py:17833-17835` `_escape_library_note_like`, `DB/ChaChaNotes_DB.py:18466-18468` `_escape_library_conversation_like`, `DB/character_conversation_search.py:1417-1420` `_escape_like_query`, `Library/library_collections_service.py:672-674` `_escape_collection_like`
  - inline, no helper: `DB/Subscriptions_DB.py:3146` and `:3236`
  - a long-form copy: `Web_Scraping/cookie_scraping/cookie_cloner.py:83-92` `escape_sql_like_pattern`
  - **the drifted one**: `TTS/profile_repository.py:1197-1198` `_escape_like_literal` escapes with `!` (`value.replace("!", "!!").replace("%", "!%").replace("_", "!_")`), i.e. a different ESCAPE character, so the two families are not interchangeable and a copy-paste between them silently stops escaping.
- Evidence: the grep above (10 hits, 9 of them the escape itself); `rg -n "LIKE" tldw_chatbook/DB/sql_validation.py tldw_chatbook/Utils/*.py` → no output, i.e. **no shared helper exists** — this is the D4(b) sub-case, not "helper ignored".
- Why it matters: nine copies of a security-adjacent string transform that must stay in lockstep with its `ESCAPE '\'` clause; the `!` variant proves the drift is already real, and the next re-roll is the one that forgets `replace("\\", "\\\\")` (escaping `%`/`_` while leaving the escape char unescaped lets a literal `\` in user text swallow the following character).
- Recommended correction: one `escape_like(value: str, escape: str = "\\") -> str` in `DB/sql_validation.py` (already the home for SQL-identifier safety and already imported by every DB module here), the 9 sites call it, the TTS caller passes `escape="!"`.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none found that states the duplication as a requirement.
- Already covered: none
### P2 [D4b] — the `maintenance_drain` polling loop is hand-rolled in 10 participants (5 byte-identical), with the poll interval already drifted 0.02 vs 0.01  ·  _slice: EVENTS_

- Where (my slice's two): `Event_Handlers/STTS_Events/stts_events.py:709`, `Event_Handlers/TTS_Events/tts_events.py:718`. The other eight: `Chat/console_chat_controller.py:5550`, `Chat/console_prompt_queue_coordinator.py:137` (partial), `STT/dispatch_coordinator.py:247`, `TTS/adapter_registry.py:194`, `TTS/audio_cpp_artifact_dependencies.py:583`, `TTS/TTS_Generation.py:1127`, `UI/Console_Modules/dictation.py:893`, `UI/Navigation/audio_cpp_model_handoff.py:169`.
- Evidence: an `ast` sweep that normalises away docstrings and string literals and hashes the remaining source of every `async def maintenance_drain` in `tldw_chatbook/`:
```
cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && $PY - <<'EOF'
import ast, pathlib, hashlib, re
root = pathlib.Path("/Users/macbook-dev/Documents/GitHub/tldw-review/tldw_chatbook")
bodies = {}
for p in root.rglob("*.py"):
    try: src = p.read_text(); tree = ast.parse(src)
    except Exception: continue
    for n in ast.walk(tree):
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "maintenance_drain":
            seg = ast.get_source_segment(src, n)
            norm = re.sub(r'"[^"]*"', '"S"', re.sub(r"'''.*?'''|\"\"\".*?\"\"\"", "", seg, flags=re.S))
            bodies.setdefault(hashlib.sha1(re.sub(r"\s+"," ",norm).strip().encode()).hexdigest()[:8], []).append(f"{p.relative_to(root.parent)}:{n.lineno}")
for h, locs in sorted(bodies.items(), key=lambda kv: -len(kv[1])): print(f"[{h}] {len(locs)}"); [print("   ",l) for l in locs]
EOF
```
→ `[17034996] 5 copies` (adapter_registry:194, TTS_Generation:1127, dictation:893, stts_events:709, tts_events:718) plus four one-off variants.
- The drift: `await asyncio.sleep(min(remaining, 0.02))` in seven copies, **`0.01`** in `TTS/audio_cpp_artifact_dependencies.py:583` and `UI/Navigation/audio_cpp_model_handoff.py:169`; the not-paused guard raises `RuntimeError` in eight and `AudioCppArtifactDependencyError` in one; `STT/dispatch_coordinator.py:247` alone takes `self._lock` around the guard. `maintenance_close_admission` and `maintenance_resume` are copy-pasted alongside it in all ten — the "copy-pasted three-method block" shape verbatim.
- Why it matters: the consumer, `Backup_Recovery/runtime_maintenance.py:199 RuntimeMaintenance`, binds ~25 participants by duck-typing and declares **no `Protocol` or ABC** (`rg -n "Protocol|class " Backup_Recovery/runtime_maintenance.py` → only `_Hook` and `RuntimeMaintenance`). A participant that mis-implements the contract fails silently at backup/quiesce time, and the next drain-loop author picks whichever interval they happened to copy.
- Recommended correction: one `async def drain_until(ready: Callable[[], bool], deadline: float, *, interval: float = 0.02) -> bool` plus a `MaintenanceParticipant` `Protocol`, both in `Backup_Recovery/runtime_maintenance.py` (the module that already owns the contract and imports every participant by dotted name). Each site keeps only its own `maintenance_ready` predicate and its own guard message.
- Size: M · ADR: no · Confidence: verified.
- Pinning test: none defines the protocol; the participants' individual drain tests would keep passing through the refactor.
- Already covered: none.
### P2 [D4b] — the compaction admission fence's canonical-JSON digest exists as three byte-identical private copies, and a fourth strictness variant of the same serializer sits beside them  ·  _slice: CHAT-rest-1_
- Where — the digest triple (identical): `Chat/console_context_compaction.py:2657 _digest_json`, `Chat/console_context_repository.py:1876 _digest_json`, `Chat/console_chat_controller.py:22946 ConsoleChatController._manual_json_digest`. The canonical-serializer triple, also identical but **stricter** (`allow_nan=False`): `citation_trace_repository.py:265`, `citation_legacy_migration.py:111`, `citation_source_locators.py:728`. A seventh variant: `citation_trace_models.py:976 _canonical_json_bytes`. ~20 such helpers repo-wide.
- Evidence: `cd $WT && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY -c "..."` →
  ```
  digest triple identical: True c9dad5b6d97afe79
  canonical triple identical: True
  citation _canonical_json   NaN -> ValueError(Out of range float values are not JSON compliant: nan)
  console _digest_json       NaN -> 'c238cb6e9ac5407491b0988102620ab9445290d2'
  ```
- Why it matters: the three `_digest_json`/`_manual_json_digest` copies compute the value `_manual_admission_matches`/`_automatic_admission_matches` (`console_context_compaction.py:2563, 2611`) compare **across module boundaries** to decide whether a compaction may commit. Three private copies in three files means a one-line change to any one (adding `default=str`, dropping `sort_keys`) silently breaks the fence, and no test compares copy against copy. The NaN strictness split is already live drift inside the same serializer family.
- Recommended correction: one `Utils/` canonical-JSON module exporting `canonical_json(value) -> str` and `canonical_json_digest(value) -> str`, strict (`allow_nan=False`) as the single contract; the Chat copies import it. `Skills_Interop/skill_trust_crypto.canonical_json` and `Actor_Packs/contracts.canonical_json_bytes` are existing candidates to promote rather than writing a new one.
- Size: M · ADR: no · Confidence: verified
- Pinning test: none compares the copies
- Already covered: none
### P2 [D4b] — the copy-pasted held-connection store template drifted: 7 stores' `transaction()` catches `Exception`, so a `BaseException` inside the block skips the rollback and WEDGES that thread's connection for the rest of the process; 4 stores already catch `BaseException` and one of them documents exactly why  ·  _slice: DB-rest_
- Where (the template, `_held_connection` / `close` / `transaction`, copied verbatim): `DB/Workspace_DB.py:392/:464/:434`, `DB/AgentRuns_DB.py:341/:387/:403`, `DB/RAG_Indexing_DB.py:155/:242/:214`, `DB/Library_Collections_DB.py:550/:683/:651`.
  Handler sweep over every `transaction()`/`read_transaction()` that issues a BEGIN (`ast` scan of `tldw_chatbook/DB/*.py`):
  `except Exception` → `AgentRuns_DB.py:403`, `Library_Ingest_Jobs_DB.py:65`, `Prompts_DB.py:659`, `RAG_Indexing_DB.py:214`, `Subscriptions_DB.py:1987`, `Workspace_DB.py:434`, `Client_Media_DB_v2.py:1325`.
  `except BaseException` → `Library_Collections_DB.py:601` (`read_transaction`) and `:651` (`transaction`), `Workflows_DB.py:50`, `automatic_work.py:60`.
- Evidence (isolated env, temp-file DBs, `KeyboardInterrupt` raised inside the `with` body):
  `RAG: conn.in_transaction after interruption = True` → `RAG: next transaction() RAISED OperationalError: cannot start a transaction within a transaction`
  `LibCollections: conn.in_transaction after interruption = False` → `LibCollections: next transaction() OK`
  Also verified by AST diff that the three `_held_connection` bodies and three `close` bodies are byte-identical across Workspace/AgentRuns/RAG_Indexing, and that the only differences in `transaction()` are the dropped `immediate` parameter and this handler.
- Why it matters: these are *held* connections in autocommit mode (`isolation_level = None`) with an explicit `BEGIN IMMEDIATE`. Missing the rollback leaves the transaction open on the thread-local connection, so (a) the uncommitted writes are silently discarded at close, and (b) **every subsequent `transaction()` on that thread raises** until the process ends — the store is bricked, not merely degraded. `Library_Collections_DB.transaction`'s docstring already names the case: *"Re-raised after rolling back, on any error **or interruption** inside the `with` block."* One store's author saw this; the fix was never propagated back to the template's other copies.
- Reachability, stated honestly: I could not find a shipped caller that raises a `BaseException` inside one of these blocks — an AST sweep of every `with <x>.transaction()` block in `tldw_chatbook/` found **0** containing an `await` (so `asyncio.CancelledError`, a `BaseException` since 3.8, cannot land mid-block today). The live triggers are therefore `KeyboardInterrupt` and `SystemExit` at interpreter shutdown, and any future `await`/cancellation inside such a block. That is what keeps this P2 rather than P1 — the *mechanism* is verified, the *shipped trigger* is not.
- Wider than `transaction()`: an AST sweep for every `try` block that contains a `ROLLBACK`/`rollback` next to a `BEGIN IMMEDIATE` and whose handlers are `Exception`-only found **18 sites** across 8 modules — `AgentRuns_DB.py:423`, `Chunking_Lab_DB.py:98/:414/:525/:590/:628` (this store issues `BEGIN IMMEDIATE`/`COMMIT`/`ROLLBACK` by hand, same shape), `Prompts_DB.py:673`, `RAG_Indexing_DB.py:234`, `Subscriptions_DB.py:1196/:1422/:2051`, `Workspace_DB.py:456/:731/:741/:751/:761`, plus `ChaChaNotes_DB.py:24028` and `Client_Media_DB_v2.py:1349` outside this slice.
- Recommended correction: change the `except Exception:` guards to `except BaseException:` — the same one-word edit `Library_Collections_DB`, `Workflows_DB` and `automatic_work` already carry. The deeper fix is that this template is copy-pasted at all: `DB/base_db.py` is the shared base every one of these classes already subclasses, and `_held_connection`/`close`/`transaction` belong there once (the three-store byte-identical bodies prove they can be).
- Size: S for the handler fix, M to hoist the template into `base_db.py` · ADR: no · Confidence: verified
- Pinning test: none — no test raises a `BaseException` inside a `transaction()` block.
- Already covered: none
### P2 [D4b] — the private-SQLite artifact validator (≈440 lines of TOCTOU-hardened `openat`/`fstat` checks) exists as two copies that have already drifted, and the "delegation" the docstring promises never happens  ·  _slice: DB-rest_
- Where: `tldw_chatbook/DB/private_sqlite.py:765-1246` and `tldw_chatbook/DB/private_sqlite_files.py:31-503` — 9 same-named module-level functions: `_failure`, `_open_artifact_fd`, `_artifact_postcondition_holds`, `_path_error_from_oserror`, `_optional_sidecar_restart_or_absent`, `_prepare_posix_artifact_generation` (285L / 273L), `_prepare_posix_artifact`, `_prepare_windows_artifact`, `_prepare_artifact`. `private_sqlite_files.py`'s module docstring says *"The local seam temporarily delegates here during the staged migration."*
- Evidence:
  - AST comparison of every shared function (`ast.unparse` bodies): **4 identical** (`_failure`, `_artifact_postcondition_holds`, `_path_error_from_oserror`, `_optional_sidecar_restart_or_absent`) and **5 already divergent** — `_open_artifact_fd` (13 changed lines: the `private_sqlite` copy carries a `_NativeOpenOutcome` probe the files copy lacks), `_prepare_posix_artifact_generation` (42 changed lines: `_pin_job` preflight seam + `preflight_body_errors` bookkeeping vs `open_artifact_fd`/`postcondition_holds`/`identity_out` injection seams), `_prepare_posix_artifact`/`_prepare_artifact` (signature plumbing), `_prepare_windows_artifact` (`os.stat(selected, follow_symlinks=False)` vs `selected.lstat()`). The *validation predicates* are still equal today; only the seams differ.
  - The claimed delegation does not exist: `ast` scan of `private_sqlite.py` → `Name-node uses of 'private_sqlite_files': 0`, `raw substring count: 1` (the import at `:22` itself). **`private_sqlite.py:22` is a dead import.**
  - The in-module copy is **Windows-only**, hence dead on every platform this app actually ships on: `_prepare_artifact` is called at `private_sqlite.py:1866/:1872` (inside `if private_paths._WINDOWS_PLATFORM:` at `:1864`; the `else` is `prepare_in_helper`) and at `:2409/:2417` inside `_prepare_source_artifacts`, whose only caller is `_pin_sqlite_source:2525` — reached only after `if not private_paths._WINDOWS_PLATFORM: … return` at `:2514-2521`. Verified empirically on this macOS box: wrapping `private_sqlite._prepare_artifact` with a counter and running `connect_private_sqlite("db.prompts.primary", <tmp>/x.db)` → `in-process _prepare_artifact calls during a POSIX open: 0`, `_WINDOWS_PLATFORM: False`, `_posix_guards_available: True`. POSIX opens go `_connect_registered_sqlite` → `prepare_in_helper` (`:1878`) → helper process → `private_sqlite_files.prepare_batch` (`private_sqlite_helper.py:272`).
- Why it matters: this is the code that decides whether a SQLite file is a safe private artifact (O_NOFOLLOW|O_EXCL open, `st_nlink == 1`, uid, 0600, re-stat postcondition). Every macOS/Linux user runs only the `private_sqlite_files` copy; the 440 lines in `private_sqlite.py` are exercised on Windows alone and by nothing in the POSIX test runs. A hardening fix applied to the copy a reader happens to open leaves the other platform unprotected, and the five already-divergent bodies are the mechanism by which that happens.
- Recommended correction: delete the `private_sqlite.py` copies and import the `private_sqlite_files` ones, passing `_pin_job`'s preflight callables through the injection seams that module already has (`open_artifact_fd`/`postcondition_holds`/`identity_out`) — that is exactly the "staged migration" its docstring describes, and it makes the `:22` import live. If the seams cannot express `preflight_body_errors`, add one more parameter rather than a second copy.
- Size: M · ADR: no (no `backlog/decisions/` entry covers this split; `private_sqlite_files.py`'s own docstring already states the intended end state) · Confidence: verified
- Pinning test: `Tests/DB/test_private_sqlite.py` patches `private_sqlite_files._open_artifact_fd` in 14 places to drive the TOCTOU races — i.e. the race tests exercise the **helper** copy only; nothing patches `private_sqlite._open_artifact_fd`. `Tests/Packaging/test_private_sqlite_helper_distribution.py:28` pins `private_sqlite_files.py` as a shipped helper file, so the module must stay — only the duplicate must go.
- Already covered: none
### P2 [D4b] — three drifted inline tail-truncates in this slice; the shared helper `Utils.Utils.truncate_content` has ZERO importers repo-wide  ·  _slice: W-persona-settings-chat_
- Where (my slice): `Persona_Widgets/personas_character_editor_widget.py:1691` (`first[:60] + "…"`), `Persona_Widgets/personas_lore_detail.py:324` (`[:57] + "..."`), `Persona_Widgets/personas_preview_pane.py:267` (`[:39] + "…"`).
- Evidence:
  - `grep -rn "truncate_content" . --exclude-dir=.git` → **one hit**, its own definition at `tldw_chatbook/Utils/Utils.py:253`. No importer in `tldw_chatbook/` or `Tests/`; `Utils/Utils.py` has no `__all__` and `Utils/__init__.py` has no star-import, so nothing can be reaching it indirectly.
  - `grep -rEn "\[: *[0-9]+ *\] *\+ *(\"|')(…|\.\.\.)" tldw_chatbook` → **38** inline re-rolls across the tree.
- The drift: the three copies in my slice disagree on the ellipsis character (`…` vs `...`), on whether the budget includes the ellipsis (`truncate_content` reserves 3 chars, `[:57] + "..."` makes 60, `[:60] + "…"` makes 61), and on the `len <= max` short-circuit. Two of the three are user-visible list previews sitting next to each other in the same workbench.
- Why it matters: the rubric's "dead helper with ≥10 re-rolls" case, at 38. This is a repo-wide finding — I am reporting the three copies inside my slice and the census; whoever owns `Utils/` should decide whether the helper is adopted or deleted.
- Recommended correction: canonical home is the existing `Utils/Utils.py:truncate_content` (fix its off-by-one if the `…` single-char form is preferred), or delete it and stop pretending a helper exists.
- Size: M (repo-wide) · ADR: no · Confidence: verified
- Already covered: none.
### P2 [D4b→D1] — Strict-JSON parsing exists in three drifted families; the wire family accepts duplicate keys that the storage family rejects, so a tool-call argument string accepted from a provider is refused when its continuation checkpoint is built  ·  _slice: LLM_
- Where (ALL copies):
  - Family A (depth+node cap, finite floats, str keys, NO duplicate-key rejection): `LLM_Calls/hosted_chat.py:633-678` (`_reject_json_constant`, `_json_shape_is_safe`, `_strict_json_loads`; depth 128 / 1,000,000 nodes / exact `dict`/`list` types) ≡ `LLM_Calls/qwencloud_streaming.py:61-109` (byte-identical modulo annotations); `LLM_Calls/moonshot.py:920-948 _json_shape_is_bounded` ≡ `LLM_Calls/zai.py:913-941` (depth 64 / 50,000 nodes / `Mapping`/`Sequence` / 16 MiB string cap — the excerpt's dup_verbatim row, confirmed).
  - Family B (duplicate-key rejection via `object_pairs_hook`, non-finite rejection, NO depth/node cap — bounded only by Python's recursion limit, which both modules catch at their public boundaries: `thinking_blocks.py:328,363,380,405`, `provider_continuation.py:450,573,584,601,689`): `Chat/thinking_blocks.py:221-235`, `Chat/provider_continuation.py:204-219`.
  - Family C (parse_constant-only, no shape check): `qwencloud.py:301-302,498,864`; `Tools/workspace_tool_protocol.py:394`; `Chat/library_activity.py:498`; `Chat/console_dispatch_checkpoint.py:288`; `Chat/library_preparation.py:187`; `Chat/console_generation_settings_metadata.py:381,387`; `Actor_Packs/export.py:360,371,718`; `Workflows/document_service.py:212`.
  - `Utils/input_validation.py:483 _validate_strict_json_value` (recursive, cycle-detect, finite floats, str keys, no dup-key since it validates decoded objects): private, 1 importer via `validate_tool_arguments` ← `MCP/hub_test_execution.py:22`.
- Which behaviour reaches the wire/storage: A guards provider→app; B guards the persisted continuation/thinking JSON. Drift consequence, verified: `hosted_chat._strict_json_loads('{"a":1,"a":2}')` → `{'a': 2}` (accepted, last-wins); `hosted_chat._normalize_tool_calls([...arguments='{"a":1,"a":2}'])` → accepted; `moonshot._moonshot_continuation_candidate(turn_with_that_call, …)` → `ContinuationValidationError: Invalid continuation data.` (raised by the `parse(dump(candidate))` round-trip at `moonshot.py:403`; `provider_continuation._strict_json_loads` rejects dup keys with `_InvalidContinuation`). `ContinuationValidationError` is not a `HostedChatProtocolError`, so `chat_with_moonshot:250` / `chat_with_zai:448` do not catch it — it escapes as an unexpected exception from the handler (non-streaming) or from `MoonshotStream.provider_continuation` (streaming).
- Why it matters: a provider emitting a duplicate key in tool arguments (unusual but legal JSON) turns a successful tool-call turn into an uncaught `ValueError` subclass at the Console boundary; more generally three definitions of "strict JSON" with different caps means the same payload can pass A and fail B.
- Recommended correction: one public `strict_json_loads(text, *, max_depth, max_nodes, reject_duplicate_keys=True)` in `Utils/input_validation.py` (the canonical home the brief names; today's private `_validate_strict_json_value` covers only the decoded-object half) combining A's caps with B's dup-key hook; A and B become thin wrappers; either make A reject dup keys (then the checkpoint round-trip cannot disagree) or make B tolerate them — pick one and pin it. ADR check: 062 (hosted boundary), 045 (qwencloud), 063 (durable tool continuation) define the boundaries but none prescribes the JSON acceptance rule — new decision.
- Size: M (one helper + wrappers) · ADR: new (acceptance rule shared by wire and storage) · Confidence: verified
- Pinning test: none for dup-key handling on either side (`rg -n '"a":1,"a"' Tests/` → none)
- Already covered: none
### P2 [D4b→D1] — Video Gen save persists "Clear" deletions as separate non-atomic writes; the Image Gen twin it "mirrors" does one atomic mutation  ·  _slice: UI-settings_
- Where: `:7812-7827` `_settings_save_video_gen_worker` (`adapter.save_sections(sections)` then a `for section, keys in deletions: adapter.delete_values(section, keys)` loop) vs `:7223-7235` `_settings_save_image_gen_worker` (`apply_settings_mutation_to_cli_config(sections, delete_keys=deletions)`). The Video block's own header (`:7475-7481`) says it is "mirroring the Image Gen block's idioms".
- Evidence: `SettingsConfigAdapter.delete_values` → `delete_settings_from_cli_config` (config.py) → its OWN `apply_settings_mutation_to_cli_config({}, delete_keys=...)` — 1 + N independent atomic file replacements (~64 ms each) with no all-or-nothing guarantee. Mechanical pair diff (ast, prefix-normalised, difflib): 17 paired methods, 7 with similarity ≥ 0.90 (4 at 1.00), 365 vs 348 lines; `_settings_save_x_gen_worker` similarity 0.54 is exactly this divergence.
- Why it matters: a Video Gen save that both edits a field and clears a saved secret can land the edit and NOT the clear (or vice-versa) when a later replacement fails; the user is told "Failed to save" while half the change is on disk — and the cleared-secret case is the one `delete_values`' docstring says must never be written back.
- Recommended correction: use the image block's single `apply_settings_mutation_to_cli_config(sections, delete_keys=deletions)` in the video worker (S). The larger D4(b) — one parameterised backend-draft block (prefix, BACKEND_IDS, FIELD_SCHEMA, raw-section name) — is the shape task-1378's split would produce; cite, don't redesign.
- Size: S (atomicity) / M (shared block) · ADR: no · Confidence: verified (seam traced; not reproduced with an injected failure)
- Pinning test: none for atomicity.
- Already covered: task-1378 (block extraction only)

### P3 findings (209)

One line each; full text in the named slice report.

| Dim | Finding | Slice |
|---|---|---|
| D1 | , cross-ref ENTRY-config] — `_provider_api_key_value` reads the raw `api_settings.<p>.api_key` without the `resolve_provider_api_key` validity check the readiness path applies | UI-settings |
| D1 | -shaped, retired to P3] — `validate_host` is annotated `-> bool` but returns a `re.Match` | EVENTS |
| D1 | `_retire_agent_worktree` failures are swallowed with a bare `pass` on two teardown paths | AGENTS |
| D1 | `except BaseException: pass` ×6 around diagnostics on the webhook delivery worker | AGENTS |
| D1 | a failing `conversation_id_for_session` lookup makes the shutdown fleet fence key on the session id instead of the conversation id, silently | CHAT-bridge |
| D1 | silent `except Exception: pass` (no log) on four data-adjacent paths | CHAT-controller |
| D1 | Message-variant methods collapse `ConflictError`/typed errors into a generic `CharactersRAGDBError` | DB-chacha |
| D1 | `_deserialize_row_fields` logs the first 100 chars of a malformed JSON field's content | DB-chacha |
| D1 | `BaseDB.vacuum` / `check_integrity` leak the connection on exception and `check_integrity` returns a non-bool | DB-media-base |
| D1 | `PromptsDatabase.soft_delete_keyword` raises `ConflictError` with the table name as the message and the row id as `entity` | DB-rest |
| D1 | `EvalsDB.search_tasks`'s LIKE branch does not escape `%`/`_`, and that branch is selected precisely *because* the query contains punctuation | DB-rest |
| D1 | the "Chats with unavailable characters" filter lowercases one side in Python (`str.casefold`, full Unicode) and the other in SQLite (`LOWER()`, ASCII-only), so a non-ASCII search term matches nothing | DB-rest |
| D1 | `RAGIndexingDB.needs_reindexing` raises `TypeError` on a naive `datetime`, the same shape its own writer silently accepts | DB-rest |
| D1 | Mutable class attributes on `TldwCli`: one is mutated in place downstream (shadowed today), one guard is dead because of them, one is unused | ENTRY-app |
| D1 | `Filters(…)` is constructed and thrown away, so the Llamafile executable picker has no filter while its Llama.cpp twin does | EVENTS |
| D1 | Malformed-SSE warnings log the raw provider line regardless of `is_sensitive_llm_request()` | LLM |
| D1 | assorted small correctness gaps (each verified by reading; none reproduced) | TOOLS-MCP |
| D1 | `_active_lineage_rows` swallows a failed `get_conversation_by_id` and silently exports EVERY branch of the conversation | UI-chat |
| D1 | Two persistence workers swallow every failure of `save_setting_to_cli_config` with no log line (rail-section prefs, search history) | UI-library |
| D1 | 26 `query_one` calls in the slice sit outside any `try`, against a package idiom of `except (NoMatches, QueryError)` | UIM-console |
| D1 | 13 HTTP call-site modules use neither `Utils/egress` nor `Utils/tls_trust` (documented decision for egress; ADR-deferred long tail for TLS) | UTILS |
| D1 | `ConsoleAutoSpeakCoordinator._observed_completion_generations` grows for the life of the screen and is never pruned | W-console-1 |
| D1 | The one seam every Library canvas' post-recompose follow-up runs through swallows its exception into a message-less DEBUG line | W-library |
| D1 | `ChatMessageEnhanced.watch_tts_state` calls `self.refresh()` where its own comment says "recompose" | W-persona-settings-chat |
| D1 | `summarize_arguments` silently skips redaction when `arguments` is not a mapping | W-persona-settings-chat |
| D1 | every personal-data failure path in the My Profile surfaces is caught, toasted, and never logged | W-persona-settings-chat |
| D1 | `_update_breadcrumbs` wraps its whole body in `except Exception: pass`, so any breadcrumb-build failure silently yields an empty breadcrumb bar | W-top |
| D2 | provider readiness is offloaded on the generic path but runs inline on the event loop on the llama.cpp path | CHAT-bridge |
| D2 | the no-sink trace settlement fallback runs SQLite on the calling event loop | CHAT-bridge |
| D2 | streamed tool-call `arguments` are accumulated with string `+=` per SSE fragment | CHAT-bridge |
| D2 | first `ensure_chat_store` runs ledger recovery and media-reference retry synchronously on whichever loop asks (the UI loop on first Console mount) | CHAT-bridge |
| D2 | Synchronous sqlite on the event-loop thread remains the default for every non-send write (edit, delete, rename, system prompt, pinned prefill, speech prefs, thinking policy, cursor writes) | CHAT-store |
| D2 | Bounded synchronous SQLite on the event loop from timer callbacks and `run_worker` coroutines (four sites, each one-shot or ≤50 rows) | ENTRY-app |
| D2 | `ModelCapabilities(section)` is constructed (and every provider regex recompiled) on each Settings context-window resolution; `_compile_patterns` itself is not the problem | ENTRY-config |
| D2 | Synchronous sqlite / file I/O on the event loop from press handlers (bounded, one-shot) | UI-library |
| D2 | Whole-screen `refresh(recompose=True)` on every external-preparation status flip | UI-library |
| D2 | Workspaces compose does sqlite + filesystem reads on the loop (N+1 per workspace row) | UI-settings |
| D2 | the Buddy scope reconciler is an app-lifetime 0.5 s interval that is never stopped | UIM-nav-mcp-persona |
| D2 | `optional_deps.py` imports `config` at module scope, and adoption of `optional_deps` would not fix the guarded-import cost | UTILS |
| D2 | `ConsoleProjectInstructionContextPanel.sync_preview` recomposes unconditionally; its sibling `sync_state` seven lines below has the equality guard | W-console-1 |
| D2 | Importing any one Library widget executes the whole package: 27 sibling modules for a 134-line leaf | W-library |
| D2 | `_review_row_summary` is computed twice per Notes-import review row | W-library |
| D3 | Module-scope `try/except` guarding an INTERNAL import in `run_webhooks.py` | AGENTS |
| D3 | `ensure_activity_receipt_service` documents "call via asyncio.to_thread"; its only caller is synchronous and runs from the UI | CHAT-bridge |
| D3 | two dead module-private helpers and one dead alias in console_trace_service.py | CHAT-bridge |
| D3 | broad `except Exception` wrapping first-party imports (masks ImportError; the import failure degrades silently) | CHAT-bridge |
| D3 | redundant function-body imports (target already bound at module scope, or already loaded before the importing module finishes); zero import cycles in all three files; the bridge's justification comment is false | CHAT-bridge |
| D3 | private helpers imported across modules/packages | CHAT-bridge |
| D3 | raw `connection.execute` SELECT outside `db.transaction()`/`execute_query` in `prepare_speculative_voice_attempt` | CHAT-controller |
| D3 | `Chat_Functions.py` uses stdlib `logging` for ~100 calls and loguru for the rest, in one file | CHAT-rest-1 |
| D3 | `UI/Console_Modules/message.py:597 _console_message_role_from_persisted` is a dead verbatim copy of `Chat/console_conversation_hydration.py:109` | CHAT-rest-1 |
| D3 | every non-attachable clipboard path logs `WARNING Path traversal attempt detected` + `ERROR Path validation error` for an ordinary user action | CHAT-rest-2 |
| D3 | `ConversationLocalMarksService.get_mark()` has no production caller; its docstring documents a delivery feature that was never wired | CHAT-rest-3 |
| D3 | God module: 22,245 lines, one class of 20,600 lines with ≥12 independent responsibility clusters and 12 store-level locks | CHAT-store |
| D3 | The one raw SQL statement in the Chat layer duplicates a check the DB layer owns | CHAT-store |
| D3 | Redundant function-body import of a module-level name | CHAT-store |
| D3 | loguru and stdlib `logging` both used; one live stdlib call sits on the message-update write path | DB-chacha |
| D3 | Dead per-call imports of already-imported stdlib names, and one sibling-inconsistent in-body import | DB-chacha |
| D3 | `search_flashcards` has no production caller and no `LIMIT` | DB-chacha |
| D3 | loguru and stdlib `logging` used side by side in Client_Media_DB_v2 (124 stdlib calls, 221 loguru), with one method shadowing the module logger | DB-media-base |
| D3 | `base_db.py` hard-codes its own subclasses and needs 7 function-body imports to dodge the resulting cycle | DB-media-base |
| D3 | Dead-in-prod helpers that materialise every media row WITH `content`, and three `not implemented` stubs | DB-media-base |
| D3 | Function-body imports of modules already imported at module scope, one on the per-call connection hot path | DB-media-base |
| D3 | loguru and stdlib `logging` emit side by side: 38 `logging.<level>(…)` message sites vs 366 loguru sites in the same file | ENTRY-app |
| D3 | Dead code cluster: a handler whose only poster is the DEPRECATED Tools_Settings_Window, a window-hiding no-op, a placeholder widget, two never-called log helpers, an unused provider map plus its 19 module-scope imports, and six write-only attributes | ENTRY-app |
| D3 | 208 function-body imports; 20 are redundant re-imports of names already imported at module scope, and 9 lazy-import a module that is resident before `on_mount` anyway | ENTRY-app |
| D3 | Four copies of the "find the ChatMessage/ChatMessageEnhanced widget by `message_id_internal`" loop in the TTS handlers | ENTRY-app |
| D3 | `config.py` is a 9,788-line god module (≈8,000 executable lines); responsibility clusters | ENTRY-config |
| D3 | dead `css_content` (1,427-line string) in Constants.py, documented dead in its own body | ENTRY-config |
| D3 | small consistency defects (six, one line each) | ENTRY-config |
| D3 | 17 function-body imports in `tts_events.py` for two modules that are neither optional nor cycle-breaking | EVENTS |
| D3 | `ChatImageHandler.SUPPORTED_FORMATS` is a dead constant | EVENTS |
| D3 | `RAG_Search/pipeline_integration.py:122` imports `chat_rag_events_simplified`, a module that does not exist (in a module with zero importers) | EVENTS |
| D3 | `VLLM_BUTTON_HANDLERS` is a permanently empty dict that is still imported, splatted, and pinned empty by two tests | EVENTS |
| D3 | `notes_events.py` does file I/O, JSON parsing and a config-path lookup at module import, and re-imports `pathlib.Path` at line 138 | EVENTS |
| D3 | Assorted confirmed smells (one line each, all read-verified) | LLM |
| D3 | the RAG A/B-testing subsystem has no production entry point | RAG |
| D3 | `parallel_processor.py:31` guards an INTERNAL import with `try/except ImportError` and installs a silent no-op stub | RAG |
| D3 | `config_profiles.quick_profile` is dead | RAG |
| D3 | `EmbeddingsServiceWrapper.create_embeddings` hashes the whole batch to probe a cache whose key space it cannot match; `embeddings_cache_hit` can never fire | RAG |
| D3 | `simple_cache._make_key` advertises xxhash, which is not a dependency of this project and is not installed; every key pays a failed import instead | RAG |
| D3 | Five function-body imports re-import names already bound at module scope | UI-chat |
| D3 | Boilerplate handler pairs (20 shape clones) are exact 3-line delegations with no behavioural drift | UI-chat |
| D3 | `asyncio.to_thread(lambda: asyncio.run(process_attachment_*(…)))` spins a fresh event loop in a worker thread to call an async helper | UI-chat |
| D3 | Same rail-switch coroutine scheduled with and without an exclusive group | UI-library |
| D3 | Three redundant function-body imports of modules already imported at top level | UI-library |
| D3 | Import-time monkeypatch of two controller classes from this module | UI-library |
| D3 | The screen body still owns single-subsystem business logic the recipe says belongs in controllers (size-ratchet context) | UI-library |
| D3 | Private helpers reached across packages and across widget boundaries (14 distinct names, 40 sites) | UI-personas |
| D3 | Nine `_…_generation` fence counters, bumped inline at 27 sites; two have helpers, seven do not | UI-personas |
| D3 | Post-attach editor re-sync swallows its DB read and editor lookup silently, leaving the editor's optimistic-lock version stale | UI-personas |
| D3 | God module: 28,140-line `SettingsScreen` (996 methods), no size-ratchet row exists, task-1378 still To Do | UI-settings |
| D3 | 8 function-body imports re-import a module already imported at module scope; `assign_select_value` is body-imported 4× for no cycle | UI-settings |
| D3 | Validators return prose; the screen maps message PREFIXES back to field keys by `startswith`, and that already produced a silent-failure incident once | UI-settings |
| D3 | `on_unmount` swallows `BaseException` from the audio.cpp handoff cleanup, contradicting the cleanup's "without hiding failures" contract | UI-settings |
| D3 | Comment rot / dead defensiveness (four small items) | UI-settings |
| D3 | six sites reach into another object's private state across a package boundary | UIM-console |
| D3 | the notes controller's `TYPE_CHECKING` block lost one import in the decomposition move: `NoteImportExecutor` is an undefined name in a return annotation — the one ruff fatal in the whole Tier-1 baseline | UIM-library |
| D3 | a two-press destructive confirmation is branched on the rendered Button label, with the authoritative flag 200 lines up in the same file | UIM-library |
| D3 | nine cross-module imports of underscore-private names, one of them a security helper from another package | UIM-nav-mcp-persona |
| D3 | the Persona controllers reach into 8-12 private members of `PersonasScreen` each, and write two of them | UIM-nav-mcp-persona |
| D3 | `_SCREEN_ROUTES["customize"]` targets a module that no longer exists | UIM-nav-mcp-persona |
| D3 | `Utils/paths.py` imports a symbol that has never existed in `Utils.Utils`; the fallback branch is the only branch that ever runs, and the four helpers built on it raise | UTILS |
| D3 | `Utils/Utils.py` re-exports stdlib `logging`; three modules import their logger from it (274 stdlib log calls), and the file mixes loguru + stdlib | UTILS |
| D3 | `Utils/Utils.py`: 25 of ~32 top-level symbols are dead (0 importers, 0 collected tests); two are test-only | UTILS |
| D3 | Six dead modules in the shared-helper layer (plus one census false negative) | UTILS |
| D3 | `Utils/text.py`: 5 of 7 functions dead, one a verbatim copy of `Utils.py:84` | UTILS |
| D3 | `secure_temp_files.py` is security theatre over stdlib guarantees; its "secure delete" has one caller path | UTILS |
| D3 | `input_validation.sanitize_string` and `log_sanitizer.sanitize_string` share a name and opposite contracts | UTILS |
| D3 | `custom_tokenizers.py`: hard-coded profile path, import-time warning, non-atomic mapping write | UTILS |
| D3 | `Utils/Utils.ensure_directory_exists` (0 importers) vs 22 raw `mkdir(parents=True, exist_ok=True)` sites | UTILS |
| D3 | `ConsoleComposerBar` is a 6,098-line / 191-method single class, and unlike both decomposed screens it is under no size ratchet | W-console-1 |
| D3 | 13 `query_one` guards in this slice catch bare `Exception` where the package's own idiom (76 sites) is `except NoMatches` | W-console-1 |
| D3 | `console_turn_file_card.py:1319` re-imports `rich.text.Text` inside `_styled_diff`, shadowing the module-level import at `:19` | W-console-2 |
| D3 | `console_transcript.py` reaches into another class's private attribute and its `_TranscriptRow.kind` Literal is missing a live value | W-console-2 |
| D3 | Three copy-pasted deferred-focus helpers in the git panel, one of which lost its attachment guard | W-library |
| D3 | `_sync_commit_footer_layout` sizes the commit footer from hardcoded label literals that omit one of the labels it sizes for | W-library |
| D3 | Two copies of "flatten + ellipsize an untrusted artifact name" in the same delete flow, with different limits | W-library |
| D3 | `_compose_pager` is copied between the two near-sibling canvases and has already drifted three ways | W-library |
| D3 | Near-miss id pair between two simultaneously mounted Skills widgets | W-library |
| D3 | The three `str(button.label) != …` guards in the Notes canvas are redundant | W-library |
| D3 | Two spellings of the same `self.app.size.width` guard in the Notes canvas | W-library |
| D3 | `chat_message_enhanced.py` uses stdlib `logging` and an `except (ImportError, TypeError, Exception)` optional-dep guard | W-persona-settings-chat |
| D3 | `speech_tts_settings_panel.py` is a 6,014-line god module outside every size ratchet | W-persona-settings-chat |
| D3 | four function-body imports of modules already imported at module scope | W-persona-settings-chat |
| D3 | the runtime imports a Textual widget module for a policy constant | W-persona-settings-chat |
| D3 | `SettingsThemeEditor.__init__` does filesystem I/O (`mkdir`) on a widget that is re-constructed on every Settings recompose | W-top |
| D3 | `config._get_effective_config_path` is a private helper used at 53 sites across 20+ packages | W-top |
| D3 | `Widgets/dictation_performance_widget.py` (319 lines) has zero users and carries the same `run_worker` defect twice | W-top |
| D4 | 27 migration steps are the same 45-line try/execute/verify/except block; two steps re-roll helpers the file already has | DB-chacha |
| D4 | Five identical LIKE-escape one-liners across four DB modules; no shared helper | DB-chacha |
| D4 | `ConflictError.__str__` is defined three times with an attribute-name drift | DB-chacha |
| D4 | Paired Library keyword fetchers and the `IN (...)` chunking convention are applied inconsistently | DB-chacha |
| D4 | `CharactersRAGDB` re-implements `BaseDB`'s path/`:memory:` handling instead of inheriting it | DB-chacha |
| D4 | `_utf8_prefix` exists twice, verbatim | UIM-console |
| D4 | `_blank_console_session_settings` + `_console_new_chat_default_generation` copied between the session and workspace controllers | UIM-console |
| D4 | `list_image_models_for_catalog` lazy-import shim copied into two screens | UIM-console |
| D4 | the `maintenance_ready`/`maintenance_drain`/`maintenance_resume` poll loop is re-rolled three times while a helper for the shape exists and is used in this same slice | UIM-console |
| D4 | `_maybe_await` is defined 66 times in the tree; one copy is in this slice | UIM-console |
| D4 | `_next_request_token` and the modal-dismissal clone are re-rolls of surfaces that already have a canonical home; the skills/prompt browse-controller pair is one generic controller written twice | UIM-library |
| D4 | `wrap_console_plain_text_uncapped` is a hand-copy of `wrap_console_conversation_title`'s wrap loop with the 2-line cap removed | W-console-2 |
| D4 | documented duplication, reported not recommended (per brief) | W-console-2 |
| D4 | `_write_spill` re-implements `Utils/atomic_file_ops.atomic_write_text` | AGENTS |
| D4 | `citation_artifact_ownership.py` rolls its own SQL-identifier validator instead of `DB/sql_validation.validate_identifier` | CHAT-rest-1 |
| D4 | three helpers are re-rolled privately in modules that already import the module (or class) holding the public original | CHAT-rest-2 |
| D4 | Three sites assign `session.persisted_conversation_id` directly, bypassing the two store helpers that own the binding-revision/lifecycle bookkeeping | CHAT-store |
| D4 | `MediaDatabase` does not subclass `BaseDB` and re-implements its path handling, `vacuum`, and `close` | DB-media-base |
| D4 | The `agent_lessons_seed_state` "is this seed still unknown?" SELECT is re-rolled inline in `app.py` while the service constructed two lines earlier owns the identical query | ENTRY-app |
| D4 | `emergency_stop._write` hand-rolls the atomic write that `Utils/atomic_file_ops.atomic_write_json` already provides (8 importers), and skips the `fsync` the helper does | ENTRY-config |
| D4 | `get_tts_profiles_db_path` re-rolls the TTS-specific path rule that `Backup_Recovery/profile_paths.database_path` already owns, bypassing the `_get_custom_database_path` helper its 13 sibling getters use | ENTRY-config |
| D4 | 22 sites in this package log raw exception text/filenames by f-string while the rest of the package logs bounded `category=` codes | EVENTS |
| D4 | `_sensitive_fetchall` reaches around `execute_query` to avoid parameter logging that `ChaChaNotes_DB` already solved with `redact_params=True` | EVENTS |
| D4 | `notes_events._parse_note_from_file_content` is a superseded 85-line parser kept alive only by its own test file | EVENTS |
| D4 | the `[:200] + "..."` note-preview truncation is inlined 3× in MCP while `Utils/Utils.py:253 truncate_content(content, max_length=200)` exists | TOOLS-MCP |
| D4 | `_safe_text` is the 12th local re-roll of a one-line wrapper over `Utils.input_validation.sanitize_string` | UI-chat |
| D4 | Dead shared helper with one re-roll: `Utils/Utils.py:253 truncate_content` | UI-library |
| D4 | `atomic_file_ops` (8 importers) ignored by three plain temp-write-then-`os.replace` re-rolls, none of which fsync; `emergency_stop` calls itself "durable" | UTILS |
| D4 | Four `len(text) // 4` token estimates beside `Utils/token_counter.estimate_tokens` (memoized, CJK-weighted, +20 % headroom) | UTILS |
| D4 | `github_api_client.py` builds its `httpx.AsyncClient` directly and guards only half its fetches | UTILS |
| D4 | 4 hand-rolled `[:N] + "..."` truncations beside `Utils/Utils.truncate_content`, which is a drop-in | W-top |
| D4 | , documented] — `_split_leading_token` exists three times, by explicit decision | CHAT-rest-1 |
| D4 | `_now_iso`/`_utc_now` are re-declared in four places in this slice (five with `DB/AgentRuns_DB.py`); no drift reaches storage, but no shared helper exists | AGENTS |
| D4 | `_identity()` is byte-identical in three activation modules | AGENTS |
| D4 | Two hand-rolled bounded `run_git` subprocess runners (three git runners counting `Workspaces/git_workspace._run_user_git`) | AGENTS |
| D4 | `cap-with-marker` truncation micro-helper duplicated across Agents and Widgets/Console | AGENTS |
| D4 | `get_internal_prompt` boot-leg shim duplicated verbatim | AGENTS |
| D4 | seven "UTC now" re-rolls in `Chat/` produce three incompatible timestamp formats; the trace repository's parser assumes one of them | CHAT-bridge |
| D4 | `_mapping_value` verbatim ×2 plus a shape clone; no shared helper | CHAT-bridge |
| D4 | three lazy-delegate wrappers with an identical double-checked `_get_delegate` | CHAT-bridge |
| D4 | `get_internal_prompt` lazy wrapper duplicated in two modules that already import each other | CHAT-controller |
| D4 | `_empty_profile_context_snapshot` duplicated between controller and `console_chat_models` | CHAT-controller |
| D4 | chat-create confirm is the one interrupt kind hand-rolled outside `InterruptRoundHost`, and it has drifted from the five host kinds | CHAT-controller |
| D4 | documented deliberate duplicate `_split_skill_command_word` | CHAT-controller |
| D4 | seven copies of the "reject duplicate JSON keys" `object_pairs_hook`, no shared factory | CHAT-rest-2 |
| D4 | `_mapping_value` / `_metadata_object` clone pairs | CHAT-rest-2 |
| D4 | the strict-JSON parser (duplicate-key rejection + constant rejection) is hand-rolled 49 times across the package, twice inside this slice as 30-line structural clones, with no `Utils/` home | CHAT-rest-3 |
| D4 | Conversation/message metadata JSON is emitted by five differently-configured `json.dumps` calls; all decode identically but the bytes drift | CHAT-store |
| D4 | `_library_fts_query` / `_library_keywords_for_*` / `_escape_library_like` are four-way shape clones across the three FTS stores | DB-media-base |
| D4 | three divergent filename sanitisers, and the *shared* one is the unsafe one (keeps NUL and control characters) | DB-rest |
| D4 | `EvalsDB` writes two incompatible textual timestamp shapes into the same `updated_at` columns; no reader compares them today, so this is the pattern one query away from the sibling P1s, not a live defect | DB-rest |
| D4 | 29 verbatim copies of `try: X.from_config(self.app_config, policy_enforcer=…) except ValueError: X(client=None, policy_enforcer=…)`; no helper exists | ENTRY-app |
| D4 | `_datetime_to_iso` is copied verbatim into 5 modules; no shared time-format helper exists | ENTRY-config |
| D4 | `ChatImageHandler.MAX_IMAGE_SIZE` is a dead constant that `attachment_core.MAX_IMAGE_BYTES` claims to mirror, plus three dead public methods | EVENTS |
| D4 | `handle_start_{llamafile,llamacpp}_server_button_pressed` are ~70-line copies that have drifted in widget type and default port | EVENTS |
| D4 | `media_events.py` writes `self.record_id = record_id if record_id is not None else media_id` in 12 of its 13 message classes | EVENTS |
| D4 | `stts_events._terminate_conversion_process` ends with an unbounded `await process.wait()` where the package's own `terminate_process_bounded` bounds both waits | EVENTS |
| D4 | `summarize_with_*` re-rolls the same "session + Retry adapter + post + stream_generator" block 16 times across two modules with behavioural drift (some close the response on abandon, most do not; some `raise_for_status`, some check status by hand; one hardcodes its URL) | LLM |
| D4 | `moonshot.py` and `zai.py` duplicate ~14 pure helpers verbatim (only the provider label in error strings differs) | LLM |
| D4 | process-RSS-in-MB is re-rolled in 5 places with 3 different divisor spellings; two of the copies are a byte-identical 5-second-TTL cache | RAG |
| D4 | two `# Exponential backoff` comments in this slice sit over linear sleeps | RAG |
| D4 | "strip to a non-empty string or None" is re-rolled 5 times under 5 names across 4 packages | RAG |
| D4 | `_identity()` (pid, thread, current task) is byte-identical in `MCP/activation.py:25`, `Agents/activation.py:24`, `RAG_Search/activation.py:141`; `_require_field`/`_require_payload_field` are identical in `MCP/unified_control_plane_service.py:2309` and `MCP/server_unified_service.py:3144`; `_text_or_none`/`_normalize_optional_text`/`_clean_text`/`_optional_text` (`str(v).strip() or None`) ×5 with `server_target_store.py:391` re-rolling a helper from a module it already imports | TOOLS-MCP |
| D4 | `web_tool_impls._format_size` is one of 15 human-readable-size formatters (no shared helper); the SSRF classifier `_is_public_ip`/`validate_outbound_url` is a second implementation beside `Utils/egress.py` (47 importers), which `egress.is_public_http_url`'s own docstring acknowledges | TOOLS-MCP |
| D4 | Four attach/detach picker workers are a 4-copy shape clone (dictionary attach/detach, world-book attach/detach); no drift found | UI-chat |
| D4 | Intra-file copies with mild drift (no shared helper exists) | UI-library |
| D4 | Lazy image-generation import shims copied into four UI modules | UI-personas |
| D4 | Three exports-dir writers in one file drifted (stamp precision, threading, error surface) | UI-personas |
| D4 | 15 methods carry the same draft-staging shape (setdefault SettingsDraft / set_value / pop-if-clean), 222 lines | UI-settings |
| D4 | Two verbatim 3-way "Save / Discard / Cancel" leave modals; three verbatim workspace-create continuation wrappers; three verbatim screen-signal loops | UI-settings |
| D4 | the graphics→pixels image fallback is duplicated in three Console widgets and one of the three swallows the failure silently | W-console-1 |
| D4 | Four Library canvases each re-roll the same two-line `sync_state`; a `reactive(recompose=True)` replaces all four | W-library |
| D4 | `_repository_path_for_display` is a fourth, weaker copy of the control-character display sanitizer | W-library |
| D4 | A third cell-aware middle-elide implementation, and `Utils/` still has none | W-library |
| D4 | `_widget_id` can collide and mount duplicate ids in the Collections capture reader | W-library |
| D4 | The Parakeet install label hard-codes a size that has a source of truth | W-library |
| D4 | `_toggle_label` has no production caller and is the 4th copy of a two-line glyph formatter | W-library |
| D4 | Two copies of the splash config loader + defaults table, no shared home | W-top |
| D4 | Byte-identical credential-provenance strings in two settings panels, with a third spelling in the settings screen | W-top |
| D4 | Three byte-size formatters in this slice alone | W-top |


## Duplication clusters (D4)

The prompt's seed table was measured, extended, and in two places corrected. Copy counts are AST definition counts over the whole package (`candidates/dup_by_name.tsv`, `dup_verbatim.tsv`, `dup_shape.tsv`); "distinct bodies" is after stripping docstrings.

| Cluster | Copies (files) | Helper exists? (importers) | Drift | Outward / downward LOC | Canonical home | Size | Rec |
|---|---|---|---|---|---|---|---|
| **UTC-now → string** (`_utc_now` 23, `_now` 15, `_now_iso` 7, `_utc_now_iso` 3, `_datetime_to_iso` 6, `_utcnow` 1) | 55 defs / ~50 files | **no** — `Utils/` has no time module; `runtime_policy/source_state.py:133 _datetime_to_iso` is private with 5 verbatim copies | **12 output shapes**: `+00:00` vs `Z`; s/ms/µs/mixed precision (`.isoformat()` omits the fraction when `microsecond==0`, so one writer emits two widths); `strftime("%…%SZ")`; `strftime("%…%f")[:-3]+"Z"`; returns `datetime`; `time_ns()`. SQLite adds a 13th (`CURRENT_TIMESTAMP`, space separator) on 31 ChaChaNotes tables | outward ≈ 55 one-liners; downward: stdlib only | new `Utils/timestamps.py` (`utc_now_iso(timespec)`, `to_iso_z`, `SQLITE_TS_FMT`) | M — a format must be chosen per store | ✅ |
| **Scope-service scaffold** (`_maybe_await` 65, `_enforce_policy` 51 (43 byte-identical), `_require_client` 47, `_normalize_mode` 46, `_action_id`, `_with_record_id`) | ~45 services (33 Interop + 12 core) | partial — `runtime_policy/enforcement.py ServicePolicyEnforcer` (4 importers) is the only shared piece | helper bodies are identical; the **local-dispatch shape** drifts. `await self._maybe_await(local_sync_call())` runs sync sqlite on the event loop. task-283 fixed exactly one service (`Chat/chat_conversation_scope_service.py:152-215`, thread file-backed, keep `:memory:` inline); ~40 others still have `to_thread=0` | outward ≈ 2,000 lines; downward: a base class needs only `inspect` + the 119-line `enforcement.py` | new `runtime_policy/scope_service.py` | L — cross-module interface, ADR | ✅ |
| **Bool/int coercion** (`_coerce_bool` 13 / 7 distinct bodies, `_coerce_int` 10 / 5) | 23 defs, incl. a byte-identical rail-state triple (`Chat/console_rail_state.py:335` = `Home/home_rail_state.py:24` = `Library/library_rail_state.py:42`) | **yes** — `config.py:1201 coerce_bool_setting` (25 importers), `:1214 coerce_int_setting` (10) | the canonical copy has the *strangest* vocabulary: `{true,1,t,y,yes}` else False, rejects `"on"`, returns the default only for `None`; copies accept `on/off/enabled/disabled`, map ints by `!= 0`, or fall back to `bool(value)` | outward ≈ 250 lines | a stdlib-only `Utils/coerce.py` leaf that `config` delegates to (config's import-time work makes the reverse direction a cycle) | M — the unrecognised-string behaviour is a decision | ✅ |
| **Byte-size formatters** (`_format_size` 5, `_format_file_size` 4, `_human_size` 2, `_format_bytes` 2, +3) | 16 defs / 15 files | **yes, private** — `Utils/Utils.py:729 _format_size_bytes` (2 internal callers) | KB vs KiB; `.0f` vs `.1f`; ladders ending GB / TB / PB; `"B"` as int vs float; `Widgets/NewIngest/SmartFileDropZone.py:97` mixes decimal thresholds with binary divisors, so 1,000,000 B renders "1.0 MB"; `Widgets/Console/console_transcript.py:639` documents its copy on purpose | outward ≈ 120 lines | make `_format_size_bytes` public | S | ✅ |
| **Strict JSON** (`_reject_json_constant` 9 (7 identical), `_strict_json_loads` 4, `_json_shape_is_safe` 2) | 15 defs across LLM_Calls, Chat, MCP, Actor_Packs, Character_Chat, Persona_Visual, TTS | partial — `Utils/input_validation.py:483 _validate_strict_json_value` is private, 1 importer | **two families that disagree**: the wire family (`hosted_chat`, `qwencloud`) caps depth/node count and *accepts* duplicate keys; the storage family (`thinking_blocks`, `provider_continuation`) *rejects* duplicate keys with no depth cap. A moonshot/zai tool-call argument accepted on the wire can raise an uncaught `ContinuationValidationError` on the way to storage | outward ≈ 150 lines | `Utils/input_validation.py`, public `strict_json_loads(text, *, max_depth, max_nodes, reject_duplicate_keys)` | M | ✅ |
| **Filename sanitizers** | 7 defs (`Utils/text.py:47`, `Utils/path_validation.py:236`, `Utils/input_validation.validate_filename`, `Utils/file_extraction.py:614`, `DB/Prompts_DB.py:5091` inline, 2 `_safe_filename`) | **yes** — `Utils/path_validation.py` (111 importers) | different safety envelopes: `text.sanitize_filename('../../etc/passwd\x00\x07x.txt')` → `'....etcpasswd\x00\x07x.txt'` (NUL and control chars kept) while `path_validation.validate_filename` raises on the same input; two same-named `validate_filename`s have opposite contracts; two copies reach disk/zip unvalidated | outward ≈ 80 lines | `Utils/path_validation.py` | M — security envelope is a decision | ✅ |
| **Per-store connection setup** (`_get_connection` overrides ×8 in `DB/`) | 8 | partial — `BaseDB._get_connection` sets only `row_factory` | PRAGMA `foreign_keys`, `journal_mode`, `synchronous`, `busy_timeout`, `isolation_level` all drift; four rows document the drift as a "task-22224 EXCEPTION", the others do not | outward ≈ 200 lines | `DB/base_db.py` | M | ✅ |
| **Atomic write** (`os.replace` hand-rolled) | 9 sites (`MCP/permission_store.py:956`, `emergency_stop.py:92`, `Agents/local_tool_provider.py:466`, `Chat/trajectory_export.py:1319`, `RAG_Search/config_profiles.py:833`, `Tools/local_tool_impls.py:649`, `UI/Console_Modules/video.py:700`, `Utils/tls_trust.py:200`, Agents `_write_spill`) | **yes** — `Utils/atomic_file_ops.py` (9 importers) | three re-rolls skip `fsync`; `emergency_stop` calls itself durable | outward ≈ 90 lines | `Utils/atomic_file_ops.py` | S | ✅ |
| **Approval-stamp store** (`stamp_scope` / `apply_batch_decisions` / `stamped*`) | 20 defs / 6 `Agents/` files | no | `stamp_scope` clears the run's slice on entry in three providers and not in the other two — one copy calls this "a deliberate divergence" | ≈ 400 lines | `Agents/` shared `ApprovalStampStore` | M — ADR-032 mandates mirroring, not copying | ✅ |
| **Small verbatim pairs** (`_manual_json_digest` ×3, `_identity` ×3 across MCP/Agents/RAG_Search, `_thaw`/`thaw_json` ×5, `get_internal_prompt` shim ×2, `_empty_profile_context_snapshot` ×2, `_mapping_value` ×2, `_utf8_prefix` ×2, `_set_message_attachments` ≡ a UI private the Chat layer imports backwards) | ~20 | no | none (verbatim) except the `thaw` key-coercion copy | ≈ 150 lines | `Chat/console_chat_models.py`, `Agents/activation.py`, `Chat/console_prepared_request.py` | S — one batch | ✅ |
| **UI cancel handlers** (`_cancel` 53 — 23+8+8 byte-identical, `_perform_safe_cancel` 45) | ~60 modals | **yes** — `Widgets/modal_dismissal.SafeModalDismissMixin` (79 importers) | none; these are 2-line adapters and `_perform_safe_cancel` is the mixin's template hook | ≈ 120 lines | a default `@on(Button.Pressed, "#cancel")` on the mixin | S | ➖ |
| `_set_status` | 22 defs / 21 files, 22 distinct bodies | no (`Widgets/base_components.py` is dead) | three shapes: bare `query_one().update()` (raises if unmounted), `is_mounted` guard, `try/except: pass` | ≈ 150 lines | — | S | ➖ |
| **Truncation** (`truncate_content` + 24 inline `[:n] + "..."` + ~20 `_truncate*`/`_ellipsize`) | ~45 sites | **yes, dead** — `Utils/Utils.py:253` (0 importers) | `…` vs `...` for the same 120-char budget | ≈ 60 lines | `Utils/text.py` — or delete the helper | S | ➖ |
| Token estimate `len(text)//4` | 4 sites | **yes** — `Utils/token_counter.estimate_tokens` (memoized, CJK-weighted, +20 % headroom) | flat `/4` vs weighted | trivial | `Utils/token_counter.py` | S | ✅ |
| **The 31 `*_Interop/` packages as "one template"** | — | — | **the prompt's premise is half wrong.** Normalising the enum name away, `Feedback_Interop/server_feedback_service.py` (158 lines) and `Kanban_Interop/server_kanban_service.py` (888) differ by 891 diff lines — the business bodies are not clones. Only the scope-service scaffold above is verbatim | — | — | — | ❌ as stated; the scaffold row above is the real cluster |

## Dead or under-adopted shared helpers

Zero-importer claims were re-verified independently of the slice that raised them: exact dotted-path `rg` over `tldw_chatbook/`, `Helper_Scripts/`, `scripts/` excluding the defining module, plus a `rg` over a full `pytest --collect-only -q` inventory (97,863 collected tests) to see through re-exports.

| Helper | Importers (pkg, non-Tests) | Hand-rolled equivalents | Rec |
|---|---:|---|---|
| `Utils/ui_helpers.py` (`UIHelpers`) | 0 | — (it targets `#chat-api-model` selects that no longer exist) | **delete** |
| `Utils/pagination.py` (`PaginatedResult`, `LazyPaginator`) | 0 | Media/Evals services page by hand in different shapes | **delete** |
| `Widgets/base_components.py` | 0 (3 CSS-census tests name the file) | `Widgets/form_components.py` (3 importers) covers the same ground | **delete** |
| `Utils/cost_estimation.py`, `debug_helpers.py`, `ingestion_preferences.py`, `splash_animations.py` | 0 each | — | **delete** (cost_estimation is an orphan of the removed `cost_estimation_widget`) |
| `Utils/Utils.py` — 25 of ~32 top-level symbols, incl. `ensure_directory_exists`, `truncate_content`, `is_valid_url` | 0 | 22 raw `mkdir(parents=True, exist_ok=True)`; 24 inline truncations | **delete the 25**; `truncate_content` adopt-or-delete. (The raw `mkdir` idiom is correct — `private_paths.secure_private_directory`, 32 importers, is the hardened one) |
| `Utils/text.py` — 5 of 7 functions | 0 | `extract_text_from_segments` is a verbatim twin of `Utils.py:84` | **delete** |
| `Utils/paths.py` project-* helpers | 0 | — | **delete** — they import a symbol that has never existed, so the only branch that runs always raises |
| `Utils/input_validation` dead validators (`validate_email`, `validate_username`, `validate_ip_address`, `validate_port`, `validate_and_raise`, `validate_filename`) | 0 | `validate_port` re-rolled at `llm_management_events_vllm.py:104` | **delete** |
| `Utils/Utils.py:729 _format_size_bytes` | 2 internal | 15 copies | **adopt** (make public) |
| `config.coerce_bool_setting` / `coerce_int_setting` | 25 / 10 | 21 + 10 private copies | **adopt** via a stdlib-only `Utils/coerce.py` leaf |
| `Utils/atomic_file_ops.py` | 9 (+4 compat aliases, 0 users) | 3 fsync-less re-rolls + 1 partial | **adopt** at the 3 plain sites |
| `Utils/token_counter.estimate_tokens` | 5 | 4 `len//4` | **adopt** |
| `Utils/path_validation.py` | 111 | 1 inline `is_relative_to` (`Agents/agent_worktree_git.py:46`) + 7 filename sanitizers | **adopt** for the sanitizers |
| `Utils/secure_temp_files.py` | 7 | 18 raw `tempfile` sites | **keep, do not push adoption** — stdlib already gives 0600/0700 and the "secure delete" is theatre |
| `Utils/optional_deps.py` | 41 | 11 module-scope `try: import` files | **keep** — adoption does not fix the import cost |
| `Utils/widget_helpers.py` (1, live via the `stts` route), `Utils/Splash.py` (1), `Utils/ui_responsiveness_artifacts.py` (harness-only) | 1 / 1 / 0 | — | **keep** |
| `Utils/egress.py` | — | 13 modules make HTTP calls through neither `egress` nor `tls_trust` | cite task-586 / task-609; the egress half is a stated decision, the TLS half is ADR-079's deferred tail |

## Legacy reachability

| Symbol / route | Marker | Importers | Reachable from entry? | Pinning tests | Verdict |
|---|---|---|---|---|---|
| `UI/Tools_Settings_Window.py` (6,926 lines) | `DEPRECATED (TASK-1346)` | 1 code importer (`UI/Screens/tools_settings_screen.py:15`) + 9 comment mentions | **No** — route `tools_settings` resolves to `MCPScreen` (`screen_registry.py:176`); the wrapper is lazily mapped in `UI/Screens/__init__.py:16` with no route | `Tests/UI/test_tools_settings_window.py`, `Tests/UI/test_settings_tools_se*`, `Tests/ProductionApp/test_tools_settings_backup.py`, `Tests/Packaging/test_raw_cli_import_closure.py` (asserts it stays *out* of the raw-cli closure), `Tests/find_call_from_thread.py`, `Tests/test_call_from_thread_guard.py` | **delete** (window + wrapper + suites). It is also the only caller of the broken `get_detected_api_providers` besides `doctor` — M |
| `UI/Screens/tools_settings_screen.py` | `DEPRECATED (TASK-1346)` | 0 (lazy map only) | No | as above | **delete** with the window |
| `UI/Screens/schedules_screen.py` | `DEPRECATED: superseded by scheduling/schedules_workbench.py` | 0; no route | No | `Tests/test_application_state_ownership.py` (name only) | **delete** — S |
| `Event_Handlers/` deletion candidates: `app_lifecycle`, `ingest_events`, `ingest_status_helper`, `tab_events`, `Chat_Events/chat_messages`, `Audio_Events/dictation_integration_events`, `Media_Creation_Events/swarmui_events`, and two 0-byte `llm_management_events_{llamacpp,llamafile}` | — | 0 outside the package | No | — | **unknown → verify**: the census is complete but the confirming greps were cut off by the usage limit; the EVENTS slice is re-running |
| `_SCREEN_ALIASES` (`screen_registry.py:231-299`): `subscriptions`/`subscription`→`watchlists_collections`; `notes`/`prompts`/`skills`/`ingest`/`search`/`media`→`library`; `coding`→`chat`; `customize`→`settings` | alias | registry | Yes — they resolve to live screens | registry tests | **keep** — route ids are a compatibility surface for deep links and the palette |
| `Widgets/Chat_Widgets/chat_approval_card.py` | docstring names the retired `Chat_Window_Enhanced` | 11 importers (`chat_screen.py:529`, `console_status_chips`, `chat_task_cards`, `buddy_conversation_modal`, `permission_summary_service`, the MCP/local tool providers) | Yes, via `set_batch` | 16 test files | **keep**; fix the docstring — P3 |
| `UI/Chat_Window_Enhanced.py` | removed in task-649 | 0 | — | 3 test files mention the name | **doc drift**: `CLAUDE.md:47/51` still describes it, and `AGENTS.md`'s "Main Windows" list names **seven** modules that no longer exist (`Chat_Window_Enhanced`, `Conv_Char_Window`, `Notes_Window`, `SearchRAGWindow`, `Evals_Window_v3`, `IngestTldwApiWindow`, `MediaWindow`) — P3 |
| `Constants.py` `css_content` (1,427-line string) | documented dead in its own body | 0 | — | — | **delete** — S |
| `app.py` dead cluster (a handler whose only poster is `Tools_Settings_Window`, a window-hiding no-op, a placeholder widget, two never-posted messages) | — | 0 | — | — | **delete** with the window |
| `ConsoleProviderGateway._chat_api_kwargs` | — | **0 production callers, 10 test callers** | No | the Anthropic cache-stability tests pin it | **delete or repoint** — the tests pin a builder the send path never runs, and its live twin has drifted (P2 in the findings) |

## Verified-fine (do not "fix")

Confirmed correct, with the evidence. The seed list from the prompt held up except where noted.

- **The prompt's "known-deliberate" list is all still true** where it was checked: the transcript reconciler is incremental; Library has no per-keystroke DB *search*; the browser-search debounce carries a cancellation token; the subscriptions scheduler is `thread=True`; the screen registry is lazy; `Local_Ingestion`'s PEP 562 lazy `__init__`; `load_settings`' defense-in-depth decrypt; `UI/Logs_Window.py:459 _compile_pattern` memoizes; the hidden-column `fts.messages_fts MATCH` form is required; `Logging_Config.py` legitimately bridges both loggers; the `_GATEABLE_BUILTINS` row copy is the registration contract; `Widgets/Console/console_transcript.py:639` documents its duplicate on purpose.
- **`config.py:6 admit_startup()` at import** is ADR-126 and correct. It is also why every `python -c "import tldw_chatbook…"` in this repo needs an isolated HOME.
- **SQLite `strftime('%H:%M:%f')`** is not a missing-seconds bug: in SQLite `%f` is `SS.SSS`. Both sites are correct.
- **`get_cli_setting("a.b", key)` dotted-section lookups resolve.** Proven on a nested config: `("chat.images","save_location","~/Downloads")` → the configured value, `("dictation.spoken_feedback", False)` → `True`, and the 1-arg dotted form too. TASK-1771 fixed this; all 32 Tier-1 call sites use a safe shape.
- **ChaChaNotes `fetchall()` without `LIMIT`** is not an unbounded-UI-list hazard: 42 non-migration methods were examined one by one; every one is page-windowed, id-chunked, per-parent-key, or a documented export.
- **`str(params)` on hot queries** is already lazy and redacted at both ChaChaNotes sites (`logger.opt(lazy=True)` + `preview_params`).
- **`app.py`'s seven `compose()` config reads** run once; its three mutable class attributes are shadowed by `__init__` (latent, P3); every `exclusive=True` worker has a `group=`.
- **All seven `id()`-keyed maps in `console_trace_service.py`** (plus the gateway's) hold their key objects for the map's lifetime and re-check identity — id reuse is structurally impossible there. Same for `base_db.py:157`.
- **`Chat/` has zero module-scope import cycles** in its three largest files; ~90 of the function-body imports are deliberate ADR-097 startup deferrals.
- **No `except Exception` in `console_chat_store.py` swallows a committed write**; every failed commit fails closed.
- **The six `Scheduling/db/migrations/v*_to_v*.py` `_get_connection` copies** the prompt flagged are Protocol stubs (`def _get_connection(self) -> Any: ...`), not implementations.
- **`run_worker(exclusive=True)` without `group=`** is one site repo-wide (`Widgets/Settings_Widgets/speech_tts_settings_panel.py:5438`), not a pattern.
- **`Utils/secure_temp_files`** is not under-adopted in any way worth fixing: stdlib `tempfile` already gives 0600/0700.

## Retired / contested

Symptoms that were real and causes that were not, plus one measurement that did not reproduce. Reporting these is the point — the previous review of this repo had ~40 % of its causal attributions retired after tracing.

| Claim | Retired by |
|---|---|
| "Dotted-section `get_cli_setting` is silently broken" (prompt seed, and a long-standing repo belief) | Proven working on a nested config; fixed by TASK-1771 |
| "`%Y-%m-%dT%H:%M:%fZ` drops seconds" (2 sites) | SQLite `strftime`, where `%f` = `SS.SSS` |
| "31 Interop packages are one verbatim template" | 891 diff lines between two normalised server services; only the scope-service scaffold is verbatim |
| "6 Scheduling migrations hand-roll `_get_connection`" | Protocol stubs |
| "`fetchall()` with no LIMIT feeds unbounded UI lists" (ChaChaNotes) | 42 methods examined; all bounded |
| "`str(params)` reaches logs on hot queries" | Both sites already lazy + redacted |
| "Mutable class attrs on `TldwCli` are shared across instances" | All three shadowed by `__init__` — latent, P3 |
| "`Utils/Utils.py:490 is_valid_url` recompiles its regex per call" | Moot: the function is dead (0 importers) |
| "Locks in `console_chat_controller.py` guard sqlite" | None of the 7 wrap SQL |
| "Send-price computation runs sqlite per keystroke" | Tooltip-only, by its own docstring |
| "27 function-body imports in `Agents/` reference modules that no longer exist" | A `sys.path[0]` harness artefact; all resolve with `PYTHONPATH=<worktree>` |
| "`console_chat_store.py:10410` silently swallows" | Falls through to a pinned warning |
| "The 12 lazy `console_voice_promotion` imports are dead deferrals" | Pinned by the boot ratchet (`test_ui_ready_module_census.py:157`) |
| "Five `fetchall_no_limit` rows in `CHAT-bridge`" | The mechanical census attached the wrong statement text; the real queries are LIMIT-bounded |
| **Contested:** "the folder-import freeze is one fsync'd commit per job (2.80 ms/job)" | The O(n²) listener half reproduces exactly (1.65 s per 1,000 files); the store half measured 0.03 ms/job in my rerun of the same script. The finding stands on the listener; the commit-cost half is environmental and is stated as contested in the finding |
| **Corrected upward:** the Media transaction leak was filed P1 | I traced the caller chain — `media_reading_scope_service.import_reading_items` → `local_media_reading_service:2848` → `:2873` → `:3383` (per-row loop) → `:3485` → `:3525` — and found no `db.transaction()` anywhere on it. Data loss on a shipped path is P0 |
| **Prior-audit entry now wrong:** "config reads are cache-backed" (listed as verified-fine in the 2026-07 performance audit) | Still true that the *value* is cached, but each read pays a storage-admission handshake: **11.1 ms per `get_cli_setting` call**, warm, measured over 200 calls. Three separate P1s in this review are consequences |


## Left UNVERIFIED

Claims that survived review but could not be settled inside the rules (no app runs, no full suite, no second OS user, no populated real profile). Each carries the literal command that would settle it. Grouped by slice; the same tables appear in `slices/<SLICE>.md`.

The four that matter most:

| Claim | Why not verified | Check to run |
|---|---|---|
| The folder-import freeze is what a user actually sees end-to-end (the component benchmark is 1.7 s per 1,000 files) | the live app also runs a research-source catalogue pass the benchmark omits; running it is forbidden here | drive the real app per `.claude/skills/verify/SKILL.md`: `tmux -L verify new-session -d -x 235 -y 52 '.venv/bin/python -m tldw_chatbook.app'`, import a 1,000-file folder, capture pane latency |
| Whether `conversations.last_modified` / `notes.last_modified` carry **both** timestamp shapes in a real user profile — the lexicographic-`ORDER BY` half of the timestamp cluster | needs a populated profile DB; the isolated scratch profile is empty | `sqlite3 ~/.local/share/tldw_cli/*.db "SELECT DISTINCT length(last_modified), substr(last_modified,11,1) FROM conversations;"` |
| Whether a second local user can pre-create `/tmp/tldw_agent_worktrees` and redirect agent checkouts | needs a second OS user on this host | `sudo -u <other> ln -s /some/dir /tmp/tldw_agent_worktrees`, then run an agent worktree checkout |
| Whether the red size-ratchet rows are in a *required* CI check (they are collected by the core-tests job) | `gh api` not run | `gh api repos/rmusser01/tldw_chatbook/branches/dev/protection/required_status_checks` |


### AGENTS
| claim | why not verified | literal command to run |
|---|---|---|
| A pre-existing `/tmp/tldw_agent_worktrees` (or symlink) owned by another local user redirects agent checkouts | needs a second OS user on this host; no test fixture models it | `sudo -u <other> ln -s /some/dir /tmp/tldw_agent_worktrees; cd $WT && source $SCRATCH/env.sh && $PY - <<'EOF'\nfrom pathlib import Path\nfrom tldw_chatbook.Agents.agent_worktree import create_agent_worktree\nprint(create_agent_worktree(Path("$WT"), "probe-run-1"))\nEOF` then `readlink -f /tmp/tldw_agent_worktrees` |
| The `_persist` read gap fires under REAL sqlite contention (not a stub) | reproducing a 5 s `BEGIN IMMEDIATE` hold exactly across `_persist`'s `get_run` needs a second process pinned to the timing; out of scope for a read-only pass | `cd $WT && source $SCRATCH/env.sh && $PY -m pytest Tests/Agents/test_agent_service.py -q` after adding a test that monkeypatches `AgentRunsDB.get_run` to raise `sqlite3.OperationalError` once inside `_persist` and asserts `db.get_run(run_id)["status"] != "running"` |
| `reconcile_orphaned_runs` actually repairs a row stranded by the gap on the next open | read from the docstring only | `cd $WT && source $SCRATCH/env.sh && $PY - <<'EOF'\nfrom tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB\np="$SCRATCH/home/runs-probe.db"\ndb=AgentRunsDB(p); rid=db.create_run(conversation_id="c", agent_kind="primary"); db.close()\ndb2=AgentRunsDB(p); print(db2.get_run(rid)["status"])\nEOF` (expect `error`) |
| The 3 `except_exception_return` rows in agent_lesson_promotion / automatic_work_runtime / model_retry and `persona_policy.py:67` | mechanical scan only (see dispositions) | commands given in the dispositions table |
### CHAT-bridge
| claim | why not verified | literal command to run |
|---|---|---|
| first `ensure_chat_store` (recovery + media-reference retry) costs a measurable UI stall on a real ChaChaNotes DB | needs a populated user DB; the isolated profile is empty | `cd $WT && source $SCRATCH/env.sh && PYTHONPATH=$WT $PY - <<'EOF'`<br>`import time; from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB; from tldw_chatbook.Chat.console_runtime import recover_console_trace_calls`<br>`db = CharactersRAGDB("<copy of a real ChaChaNotes.db>", "review"); t=time.perf_counter(); recover_console_trace_calls(db); print("%.1f ms" % ((time.perf_counter()-t)*1000))`<br>`EOF` |
| `get_provider_readiness` on the llama path blocks the loop under a real keyring backend | isolated profile forces `PYTHON_KEYRING_BACKEND=null` (0.01 ms measured) | same timing loop as above with `PYTHON_KEYRING_BACKEND` unset and a real `~/.config/tldw_cli/config.toml` (do not run against the live profile from this worktree — ADR-126 `Recovery required`) |
| `settle_response` can drop a seal with no queue entry when `submit` raises before `_settle_claimed` | fault injection against a real DB needed; reading shows `_prepare_safely` degrades rather than raises | `cd $WT && source $SCRATCH/env.sh && $PY -m pytest Tests/Chat/test_console_trace_call_lifecycle.py Tests/Chat/test_console_trace_settlement.py -q` then inject `monkeypatch.setattr(coordinator, "_prepare_safely", raising)` in a copy of the lifecycle test |
| the `[:-1] + "+00:00"` parser at `console_trace_repository.py:553-555` ever sees a form-3 (`+00:00`) `settled_at` | no writer produces one today (all trace timestamps come from form 1/2 factories) | `rg -n "settled_at=" tldw_chatbook/ | rg -v "occurred_at|_utc_now"` — any hit that is not a `_utc_now()`/`occurred_at` source is a new writer to check |
### CHAT-controller
| claim | why not verified | literal command to run |
|---|---|---|
| Finding 2's persist branch (`_persist_pending_message_if_ready`) actually fires during a real approval round (needs an unflushed stream buffer on a pending row at the moment the summary thread starts) | requires a live run with `[permission_summary]` active; app not run per brief | `.claude/skills/verify/SKILL.md` recipe: `tmux -L verify` launch with `[permission_summary] mode="always"` and an MCP "ask" tool, send a turn that triggers an approval card, then `capture-pane` the Logs (F3) for `exchange_attach_failed`/`RuntimeError` lines and check `db.registered_connection_count()` via the diagnostics inventory |
| Magnitude of finding 3 (N point reads per send) | not measured | `cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && $PY -m pytest Tests/Chat/test_console_rewind_summarize.py -q` to confirm the fixture, then time `controller._durable_context_snapshots(session_id)` with `time.perf_counter()` on a 200-message file-backed session |
| Whether a backup quiescence pass actually reclaims the finding-1 handles in production (only read `close_registered`) | not exercised | `$PY -c` building the probe DB, leaking 3 handles, then `with db.quiesce_connections(timeout_seconds=5): pass` and printing `db.registered_connection_count()` |
### CHAT-store
| claim | why not verified | literal command to run |
|---|---|---|
| A production thread other than the UI thread reaches `write_trajectory_rows`/`_persist_exchanges_only` while the UI thread is inside `_dispatch_branch_mutation` (turns the P2 inversion into a user-hittable P1) | `commit_durable_turn` is off-loop (`_run_durable_db_call` → `asyncio.to_thread`, controller :11036) but does not take the locks; `publish_owners` (controller :11311, which does reach `_trajectory_lock` via `_hydrate_durable_turn_owner_messages` :6815) — I did not resolve how it is invoked; the agent bridge marshals through `call_from_thread` (`console_agent_bridge.py:2336`) | `cd $WT && rg -n "publish_owners|_run_durable_db_call\(|call_from_thread\(|to_thread\(" tldw_chatbook/Chat/console_chat_controller.py \| rg -n "11[23][0-9][0-9]:"` then, live: `.claude/skills/verify/SKILL.md` recipe — start an agent turn in tab A, regenerate in tab B during the tool loop, `py-spy dump --pid <app>` and look for `write_trajectory_rows` + `_dispatch_branch_mutation` on two threads |
| `rollback_transient_send` (:3093) leaves a stale `settings_persistence_failures` entry a settings Retry could target | consequence traced by reading only | `cd $WT && PYTHONPATH=$WT $PY -c "…create session; persist_session_if_needed; _record_console_settings_failure(...); rollback_transient_send(...persisted_conversation_id=None); print(session.settings_persistence_failures)"` |
| `_persistence_accepts_kwarg` (`inspect.signature`) cost per persist | not measured | `cd $WT && PYTHONPATH=$WT $PY -m cProfile -s cumtime <SCRATCH>/repro/store_settle_swallow.py 2>/dev/null \| rg "_persistence_accepts_kwarg\|signature"` |
### DB-chacha
| claim | why not verified | literal command to run |
|---|---|---|
| `conversations.last_modified` / `notes.last_modified` actually carry both timestamp shapes in real profiles (basis of the lexicographic-ORDER-BY half of the P2 timestamp finding) | needs a real profile DB; only the file's own `:17723` comment + task-32172 are cited | `cd <worktree> && source <SCRATCH>/env.sh && $PY -c "import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); print(c.execute(\"SELECT substr(last_modified,11,1) sep, COUNT(*) FROM notes GROUP BY sep UNION ALL SELECT substr(last_modified,11,1), COUNT(*) FROM conversations GROUP BY 1\").fetchall())" <path-to-a-copied-user-ChaChaNotes.db>` |
| `get_sync_log_entries` is ever called with `limit=None` (unbounded) | callers not traced | `grep -rnE 'get_sync_log_entries\(' --include='*.py' tldw_chatbook \| grep -v 'limit='` |
| `execute_query`'s per-call `log_histogram`/`log_counter` (`:4056-4066`) cost on query-heavy paths | not measured; `Metrics/metrics_logger` not read | `cd <worktree> && source <SCRATCH>/env.sh && $PY -c "import timeit; from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB as C; db=C(':memory:','t'); print(timeit.timeit(lambda: db.execute_query('SELECT 1').fetchone(), number=2000)/2000*1e6, 'us/query')"` |
| `legacy_markers_per_file` excerpt row (22 markers) | marker definition not in the excerpt | `grep -niE 'legacy' tldw_chatbook/DB/ChaChaNotes_DB.py \| wc -l` and read each |
### DB-media-base
| claim | why not verified | literal command to run |
|---|---|---|
| `Tests/DB/test_sql_validation.py::TestValidateTableName::test_chunking_templates_columns_accepted_and_live` fails ONLY because of the isolated env (`RecoveryRequired('raw_source_selection_changed')` raised from `Backup_Recovery/raw_participants.py:_participant_state` when `MediaDatabase(tmp_path/'cols.db')` opens under pytest's conftest + scratch `TLDW_CONFIG_PATH`), not because of a `ChunkingTemplates` column drift | the brief forbids running against the real profile; my own file-backed `MediaDatabase` under `$HOME/repro/` opened fine, which points at the conftest/config binding rather than the schema, but that is inference | from the user's normal dev shell (main checkout): `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook && .venv/bin/python -m pytest Tests/DB/test_sql_validation.py -q` — expect 27 passed |
| The WAL first-conversion race that `AgentRuns_DB` orders around and `Library_Collections_DB` retries around actually bites the stores that do neither (`Evals_DB`, `Workspace_DB`, `RAG_Indexing_DB`, `Library_Ingest_Jobs_DB`, `Subscriptions_DB`) | needs two processes opening the same fresh file at once; not attempted | `cd $WT && source <SCRATCH>/env.sh && for i in 1 2; do $PY -c "from tldw_chatbook.DB.Evals_DB import EvalsDB; EvalsDB('$HOME/repro/race.db')" & done; wait` — repeat ~20×, look for `OperationalError: database is locked` |
| `MediaReadingScopeService.import_reading_items` running the sync import inline on the event loop (`media_reading_scope_service.py:1799-1817` via `_maybe_await`) is the general local-mode pattern rather than an oversight (a sibling `_is_memory_backed` helper at `:131` suggests other calls go through `to_thread`) | outside my slice; only the two ranges above were read | `rg -n "to_thread|_is_memory_backed" tldw_chatbook/Media/media_reading_scope_service.py` and compare against the `import_reading_items` body |
### DB-rest
### ENTRY-app
| claim | why not verified | literal command to run |
|---|---|---|
| The 4.4 s / 0.33 s folder-submit freeze is what a user sees in the running app (P1 end-to-end) | measured on the registry+store components with a listener of the same shape; the live app also runs `_dispatch_research_source_catalog_job`→`_top_up_ingest_parse_pool` per file (≈0.15 ms each in the benchmark's "top-up reads" column) and canvas repaints, so the real number is ≥ the benchmark; not run because the brief forbids running the app | `.claude/skills/verify/SKILL.md` recipe: `tmux -L verify new -d -s v 'TLDW_CONFIG_PATH=<scratch profile> python -m tldw_chatbook.app'`; navigate Library ▸ Import, submit a 100-file folder (`mkdir /tmp/f100 && for i in $(seq 100); do echo x > /tmp/f100/$i.txt; done`), then `tmux -L verify capture-pane -p` every 100 ms and count frames until the queue rows appear; repeat with 1000 files |
| Adding `logger.opt(exception=True)` in `_run_personal_context_link` leaks no plan content | the log-file sink runs with `diagnose=True` per comments at 15893/15954, so `.opt(exception=True)` may dump frame locals including `plan` | `grep -n 'diagnose' tldw_chatbook/Logging_Config.py` then, if `diagnose=True` on the file sink, use `logger.warning("… (exception_category={})", type(exc).__name__)` instead |
| The P3 `initialize_agent_lessons_folder` timer write measurably stalls the loop on a large ChaChaNotes DB | not timed | `cd <worktree> && source <SCRATCH>/env.sh && PYTHONPATH=$PWD .venv/bin/python -c "import time; from tldw_chatbook.config import get_chachanotes_db_lazy; from tldw_chatbook.Notes.agent_lessons import initialize_agent_lessons_folder; db=get_chachanotes_db_lazy(); t=time.perf_counter(); initialize_agent_lessons_folder(db, scope_mode='local_only', profile_id='local', dataset_id='local'); print(time.perf_counter()-t)"` |
### ENTRY-config
| claim | why not verified | literal command to run |
|---|---|---|
| Console realtime actually puts the `get_api_key("openai")` value on the wire (finding 1's user-visible consequence) | code path read to `realtime.py:616` (returns it as the credential); no live provider send in a read-only review | `.claude/skills/verify/SKILL.md` recipe: `tmux -L verify` launch with `[api_settings.openai] api_key = "YOUR_KEY"`, open Console ▸ realtime, `capture-pane` the auth error |
| Worker-thread INFO drop shows in the real Logs screen (finding 3 in situ) | proven at the handler level only; the app was not run | `tmux -L verify` launch, trigger any `@work(thread=True)` job (e.g. an ingest), open Logs (F3), `capture-pane` and compare against the file log at `get_cli_log_file_path()` |
| `get_model_cache_dir` (config.py 9506-9512) accepts a config-sourced relative/`..` path with no `validate_path_simple` while every DB getter validates | read only; a config value is inside the user's trust boundary so severity is at most P3 | `printf '[embedding_config]\nmodel_cache_dir = "../../escape"\n' > $CFG; TLDW_CONFIG_PATH=$CFG $PY -c "from tldw_chatbook.config import get_model_cache_dir; print(get_model_cache_dir())"` |
| Extraction of the Canvas / Console-trace clusters out of config.py is cycle-safe (god-module finding) | not attempted (read-only) | `cd $WT && $PY -m pytest Tests/Packaging/test_config_import_closure.py -q` after the move, plus `$PY -m pytest Tests/RuntimePolicy --collect-only -q` (the collection that broke last time) |
### EVENTS
(in progress)
### LLM
| claim | why not verified | literal command to run |
|---|---|---|
| `summarize_with_google` posts to `https://generativelanguage.googleapis.com/v1beta/openai/` (base path, no `/chat/completions`) and therefore 404s on the live API; the diagnostic-privacy test pins that exact URL (`Tests/LLM_Calls/test_summarization_diagnostic_privacy.py:5883`, inside `test_google_success_hides_credential_input_prompt_and_response`) | needs a network call (forbidden); the pinning test asserts the URL as current behaviour, so a change is a decision, not a fix | `cd $WT && source $SCRATCH/env.sh && GOOGLE_API_KEY=… $PY -c "import requests; print(requests.post('https://generativelanguage.googleapis.com/v1beta/openai/', json={}, timeout=10).status_code)"` (expect 404 vs 4xx-with-body on `/chat/completions`) |
| `summarize_with_huggingface` targets the retired `api-inference.huggingface.co/models/<id>` `inputs` API and cannot succeed against current HF | network | same shape as above against `https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2` |
| deepseek/groq/mistral/openrouter streams omit usage because `stream_options.include_usage` is never requested (`rg -n include_usage LLM_Calls/` → openai, moonshot, qwencloud only) | provider default behaviour needs a live stream | `rg -n "include_usage" LLM_Calls/LLM_API_Calls.py` (shows the gap) + one live streamed request per provider inspecting the final chunk for `usage` |
| Per-call `requests.Session`/`HTTPAdapter` construction (no pool reuse) is a measurable cost for N-chunk summarization | not timed | `$PY - <<EOF` timing 50 × `create_default_session(); session.mount(...); session.close()` vs one reused session against a local `http.server` |
| Leak-on-abandon in the 14 summarization stream generators lacking `response.close()` | not exercised | mock `iter_lines`, `next(gen)`, `del gen`, `gc.collect()`, assert `response.close.called` per generator |
| The `_maybe_record_usage` consumer produces an EMPTY ledger row (vs an estimate) for a streamed Gemini turn in the real Console | gateway not run | `cd $WT && source $SCRATCH/env.sh && $PY -m pytest Tests/Chat/test_console_provider_gateway.py -q` after adding a case that streams a `usageMetadata` chunk through `_stream_generic_chat` and asserts `signals.usage_snapshot()` |
| `get_cli_setting` costs 12.6 ms against a REAL profile (measured only in the isolated scratch HOME) | brief forbids the real profile (ADR-126 recovery gate) | rerun `$SCRATCH/llm_timing_probe.py` without `env.sh` on a machine whose profile is not in recovery |
| `chat_with_openai` overlays an unchecked `api_settings.openai.api_key` placeholder (`:640-642`) onto the wire | needs a config with `api_settings.openai.api_key = "<API_KEY_HERE>"` and a mocked post | `$PY -` with `load_settings` patched to that table, patch `requests.Session.post`, call `chat_with_openai(...)`, inspect `headers["Authorization"]` |
### RAG
### TOOLS-MCP
| claim | why not verified | literal command to run |
|---|---|---|
| Whether `Tests/MCP/test_permission_store*.py` pins the OSError→backup+reset branch as a REQUIREMENT (would turn the P1 into a decision) | grep found only `corrupt`/`unknown-version` pins (`:3,201`) and a `PermissionError` raised inside a test double at `:149` (context not read) | `cd $WT && source $SCRATCH/env.sh && $PY -m pytest Tests/MCP/test_permission_store.py -q` then `rg -n 'PermissionError|OSError|chmod' Tests/MCP/test_permission_store*.py -B3 -A12` |
| Import cost of `tldw_chatbook.MCP.server` when `mcp_unified` IS installed (module-scope `from mcp_unified.gateway import serve_stdio` at `server.py:71`, pulled by `local_runtime_delegate.py:10` and `local_control_service.py:133`) | `mcp_unified` absent in the review venv | `cd $WT && source $SCRATCH/env.sh && $PY -X importtime -c 'import tldw_chatbook.MCP.server' 2>&1 \| rg 'mcp_unified\|MCP.server' \| tail -5` (after `uv pip install -e ".[mcp]"` in a scratch venv) |
| Wall-clock cost of `_revalidate_mutation_scope` per Hub mutation against a real tldw_server (the 10-round-trip count is verified; latency is not) | needs a live server | `.claude/skills/verify/SKILL.md` recipe: `tmux -L verify` → MCP Hub → Servers ▸ any mutation, with `[logging] level = DEBUG` and `rg 'Unified MCP endpoint probe' ~/.local/share/tldw_cli/logs/*.log \| wc -l` before/after one click |
| `fs_edit`/`fs_patch` truncate-window data loss under a mid-write kill | crash window not reproduced (unconditional code path only) | `cd $WT && source $SCRATCH/env.sh && PYTHONPATH=$WT $PY - <<'EOF'` — monkeypatch `Path.write_bytes` to raise after `open(...,"wb")` truncates, call `_edit_relative_file`, assert the file is empty `EOF` |
| `ServerUnifiedMCPService._client_cache` keeps a stale client after an auth-only target edit | needs a client_factory that reflects auth; not exercised | `rg -n '_client_cache' tldw_chatbook/MCP/server_unified_service.py` (expect only `.get`/assignment, no invalidation) + `rg -n 'invalidate_cache\|_client_cache' tldw_chatbook/UI/MCP_Modules/*.py` |
| The 9 `except Exception: return` sites in `Tools/watchlists_command_service.py` and 2 in `Tools/raw_cli_executor.py` | files not read (mechanical only) | `rg -n 'except Exception' -A3 tldw_chatbook/Tools/watchlists_command_service.py tldw_chatbook/Tools/raw_cli_executor.py` |
| ruff fatal baseline for the slice | not run | `cd $WT && .venv/bin/ruff check --select E9,F63,F7,F82 tldw_chatbook/Tools tldw_chatbook/MCP` |
### UI-chat
| claim | why not verified | literal command to run |
|---|---|---|
| The `Core Tests` job (`test.yml:53`, which collects `Tests/Architecture/`) is a *required* branch-protection check — i.e. dev is landing PRs over a red ratchet rather than the job being advisory | branch protection is not readable from the tree; I only verified collection | `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook && gh api repos/rmusser01/tldw_chatbook/branches/dev/protection --jq '.required_status_checks.contexts'` then `gh run list --workflow=test.yml --branch dev --limit 3 --json conclusion,headSha` |
| `Tests/UI/test_console_model_section.py` is green on dev (it is red under the isolated review HOME with `Backup_Recovery.bootstrap.RecoveryRequired: raw_source_selection_changed`; my probes using the same `ConsoleHarness` from a scratchpad rootdir pass, so the failure is `Tests/conftest.py` profile handling under the review env, not the product) | could not run against the real profile (ADR-126 recovery gate) | `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook && .venv/bin/python -m pytest Tests/UI/test_console_model_section.py -q -p no:cacheprovider` (main checkout, real profile) |
| Live surface of finding 1: change temperature via Alt+M and observe the Context rail's Model ▸ Temperature row not updating | do-not-run-the-app rule | `.claude/skills/verify/SKILL.md` recipe (main checkout): `tmux -L verify new-session -d -x 160 -y 44 'python3 -m tldw_chatbook.app'`; open Context rail ▸ Model (F6 → left rail, expand "Model"), note Temperature; `alt+m`, set temperature 0.99, Enter; `tmux -L verify capture-pane -p` and compare the Temperature row |
| Finding 2's magnitude on a REAL profile (keyring-backed credentials) — harness measured 37–49 ms/tick with the null keyring | harness only | `cd <worktree> && source <SCRATCH>/env.sh && PYTHONPATH=<worktree> $PY -m pytest <SCRATCH>/probes/test_probe_credential_poll_profile.py -q -s --rootdir=<worktree> -c <worktree>/pyproject.toml`, then repeat with `PYTHON_KEYRING_BACKEND` unset against a scratch profile holding an anthropic key |
| Cost of the two on-loop PK reads at row-menu open (6619/6997) on a large user DB | not timed | `cd <worktree> && source <SCRATCH>/env.sh && $PY - <<'EOF'` building a file-backed ChaChaNotes DB with ~10k conversations (see `Tests/DB/` fixtures) and timing `db.get_conversation_by_id(cid)` + `db.get_messages_for_conversation(cid, limit=1)` with `time.perf_counter()` |
### UI-library
| claim | why not verified | literal command to run |
|---|---|---|
| The whole-surface recompose ratchet (`LIBRARY_WHOLE_SCREEN_RECOMPOSE_MAX = 63`) is green/red at this commit | every test in the file ERRORs at fixture setup under the isolated env (`_disable_model_catalog_refresh`, pytest-asyncio strict-mode async fixture from `Tests/UI/conftest.py`) — environment, not the ratchet | `cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && $PY -m pytest Tests/UI/test_library_recompose_ratchet.py -q -p no:cacheprovider -p asyncio --asyncio-mode=auto` (or run it in the main checkout's normal pytest config) |
| Live cost of a `]` press in the Reader with a large review set (the P1 measured the DB reads in isolation, not the end-to-end keypress) | do not run the app (brief rule) | main checkout: `tmux -L verify new-session -d -x 235 -y 52 '.venv/bin/python -m tldw_chatbook.app'; sleep 12; tmux -L verify send-keys C-p; tmux -L verify send-keys -l 'Library'; tmux -L verify send-keys Down Enter;` open Browse Media → a row → "Review these" (needs ≥100 items) → `tmux -L verify send-keys ']'` repeatedly while `TEXTUAL_LOG`/`--durations` captures frame time; `tmux -L verify send-keys C-q; tmux -L verify kill-server` |
| The red size ratchet is also red on the CI runs for the PRs that landed since 2026-09-09 (vs. merged past a cancelled check) | needs GitHub CI history, outside the worktree | `gh run list --workflow=test.yml --branch dev --limit 20` then `gh run view <id> --log-failed \| grep test_screen_size_ratchet` |
| `legacy_markers_per_file` (29 rows) | not examined individually | `grep -n "legacy\|LEGACY\|STALE" tldw_chatbook/UI/Screens/library_screen.py` and read each |
| The `_notify_*_warning` consolidation is acceptable under the byte-for-byte canon | canon says moved bodies are not edited mid-series; whether a shared helper may be introduced at the SCREEN layer first needs the program owner | `grep -n "byte-for-byte" backlog/docs/library-decomposition-recipe.md` (§1 "The byte-for-byte canon") |
### UI-personas
### UI-settings
| claim | why not verified | literal command to run |
|---|---|---|
| The absolute per-keystroke cost (388-426 ms) holds on a real, settled user profile — in the isolated scratch profile every `get_cli_setting` costs 10-14 ms and 482 `posix.open` calls (`config.py:8447` → `Backup_Recovery` admission → `storage_admission.py:835/854`), which inflates every adapter read. The 7×/4×/32× call multiplier is structural and env-independent; the ms are not. ENTRY-config owns the config-side cost (their report §"ADR-126 admission", not the per-call price). | Brief forbids running against the real profile (config.py exits with `Recovery required`); no settled profile is available in this env. | On a machine with a settled profile: `cd $WT && $PY - <<'EOF'` → `import time; from tldw_chatbook.config import get_cli_setting; t=time.perf_counter(); [get_cli_setting("console","max_parallel_runs",1) for _ in range(100)]; print((time.perf_counter()-t)*10, "ms/call")` — if ≪ 1 ms, re-run the keystroke probe in this report; the finding stays P1 only if the keystroke is still ≥ 100 ms, else downgrade to P2 (still 7 loads per keystroke). |
| Library/RAG category open visibly stalls (compose ≥ 156 ms of reads + widget build) | live surface; app not run | `.claude/skills/verify/SKILL.md` recipe: `tmux -L verify new -d -x 190 -y 55 'cd $WT && $PY -m tldw_chatbook.app'`, navigate Settings → RAG, then `tmux -L verify send-keys` a digit into "Default results" and `capture-pane` at 100 ms intervals — the rail should visibly freeze between keystrokes. |
| Video Gen half-applied save (edit landed, Clear not) on a failed second replacement | needs an injected failure in `delete_settings_from_cli_config` mid-loop | `cd $WT && source env.sh && $PY -m pytest Tests/UI/ -q --collect-only \| rg video_gen` to find the save test, then monkeypatch `SettingsConfigAdapter.delete_values` to return False after `save_sections` succeeded and assert config.toml still carries the edit. |
| Stuck `AUDIO_CPP_MODEL_LIBRARY_RESULT` claim after a swallowed release failure in `on_unmount` | consequence not reproduced | see the P3 finding's command. |
| Workspaces compose cost at realistic N | not timed against a populated `WorkspaceDB` | see the P3 D2 finding's command. |
| `_provider_current_credential_source("google")` returns `"stored"` on the shipped template | needs a sync-built screen with `app_config` = the shipped template | `cd $WT && source env.sh && $PY - <<'EOF'` → build `SettingsScreen(SimpleNamespace(app_config=load_settings()))`, monkeypatch `SettingsScreen.app` to a stub (as `Tests/UI/test_settings_rag_profile_region.py::fake_app` does), call `screen._provider_current_credential_source("google")`. |
### UIM-console
### UTILS

| claim | why not verified | literal command to run |
|---|---|---|
| `Chat/usage_recorder.estimate_tokens` (`len//4`) numbers are persisted to a usage ledger (would raise the token-estimate finding from P3 to P2 per the "drift reaches storage" rule) | traced only to `record_usage` in the same file; the ledger write path was not read | `rg -n "record_usage\|_prompt_tokens\|ledger" tldw_chatbook/Chat/usage_recorder.py tldw_chatbook/Chat/usage_ledger*.py tldw_chatbook/DB/*usage* 2>/dev/null` |
| ADR-079 accepts the TLS long tail as permanent (vs. "to be threaded later") — decides whether the 13-module list is a finding for other slices or out of scope | only grep hits `:29,:67-68` read, not the ADR body | `sed -n 20,40p backlog/decisions/079-network-tls-trust-policy.md; sed -n 60,80p backlog/decisions/079-network-tls-trust-policy.md` |
| `Tests/Utils/test_doctor.py` exercises `check_optional_dependencies()` with no argument (i.e. pins the wrong production answer as a requirement) | test file not read | `rg -n "check_optional_dependencies\(\)\|run_doctor\(" Tests/Utils/test_doctor.py` |
| SmartFileDropZone's `1,000,000 B → "1.0 MB"` is visible on a shipped path | computed from the function; the drop zone's liveness in the current Library import rail not checked | `rg -n "SmartFileDropZone" tldw_chatbook --glob '!Tests/**' -l` then `.claude/skills/verify/SKILL.md` recipe: `tmux -L verify …` drop a 1,000,000-byte file on Library ▸ Import and `capture-pane` |
| `github_api_client.py:488/568/636` in-function `from ..Utils.egress import …` can be hoisted (no import cycle through `config`) | not attempted | `cd $WT && source <SCRATCH>/env.sh && $PY -c "import tldw_chatbook.Utils.egress, tldw_chatbook.Utils.github_api_client; print('ok')"` |
| In-app marginal import cost of `Utils.Utils` (cold-process cumulative was 61.8 ms, dominated by first-import of loguru) | only cold `-X importtime` measured | `$PY -X importtime -c "import loguru, hashlib, unicodedata; import tldw_chatbook.Utils.Utils" 2>&1 \| rg "Utils\.Utils"` |
| `Utils/text.sanitize_filename` output at `Local_Ingestion/audio_processing.py:1257/1262` reaches a path join without a later `validate_filename` | call site returns the name to a downloader; not traced | `rg -n "_derive_filename\|sanitize_filename" tldw_chatbook/Local_Ingestion/audio_processing.py` then read the consumer of that return value |

### Dead or under-adopted shared helpers

| helper | importers (pkg, non-Tests) | hand-rolled equivalents (count, source) | dead-check evidence | Rec |
|---|---|---|---|---|
| `Utils/ui_helpers.py` (`UIHelpers`) | 0 | none needed (targets `#chat-api-model` selects that no longer exist) | `rg -n "Utils\.ui_helpers\b\|from \.\.Utils\.ui_helpers" tldw_chatbook Helper_Scripts scripts Tests` → 0; `rg ui_helpers utils_collect.txt` → 0 | delete |
| `Utils/pagination.py` (`PaginatedResult`, `paginated_fetch*`, `LazyPaginator`) | 0 | Media/Evals services page by hand (not counted; different shapes) | exact-path rg → 0; collect-only → 0 (`rg -ln "Utils\.pagination" Tests` → none) | delete |
| `Utils/widget_helpers.py` | 1 (`UI/Voice_Cloning_Window.py:276`, reachable via the registered `stts` screen) + 2 test files | – | live | keep |
| `Utils/Utils.ensure_directory_exists` | 0 | 22 `mkdir(parents=True, exist_ok=True)` outside Utils (`raw_mkdir.tsv`, 30 rows − 8 Utils) | rg → definition only; collect → 0 | delete (the raw idiom is the right one; `private_paths.secure_private_directory` is the hardened helper, 32 importers) |
| `Utils/Utils.truncate_content` | 0 | 24 inline `[:n] + "..."` (`inline_truncate.tsv`) + 25 named `_truncate*/_ellipsize` defs | rg → definition only; collect → 0 | adopt (one public `truncate(text, budget, ellipsis="…")`) — or delete if the sweep is not done |
| `Utils/secure_temp_files.py` | 7 | 18 raw `tempfile` sites outside Utils (`tempfile_no_secure.tsv`, 24 − 6 Utils) | live | keep; do not push adoption (stdlib already gives 0600/0700; overwrite-delete is theatre) |
| `Utils/atomic_file_ops.py` | 8 (+ 4 backward-compat aliases, 0 users) | 3 plain re-rolls without fsync (`local_tool_provider:466`, `trajectory_export:1319`, `emergency_stop:92`) + 1 partial (`permission_store:956`); 3 dir_fd-relative sites justified | live | adopt at the 3 plain sites |
| `Utils/optional_deps.py` | 41 | 11 files with module-scope `try: import X except ImportError` and no optional_deps (`try_import_guard.tsv` filtered) | live | keep; adoption is not the fix for the D2 cost (`check_dependency` imports eagerly too) |
| `Utils/token_counter.estimate_tokens` (`_chars_estimate` is private) | 5 | 4 `len//4` re-rolls (`usage_recorder:41`, `world_info_processor:681`, `Subscriptions/token_manager:113`, `chunk_preview:98`) | live | adopt |
| `Utils/Utils._format_size_bytes` (private) | 2 internal | 15 public/private copies across UI/Widgets/Chat/Tools/Library (bodies tabled above) | live (via `get_formatted_db_size_with_wal`) | adopt: make public, one implementation |
| filename sanitizers (`text.sanitize_filename` 2 · `path_validation.validate_filename` 8 · `input_validation.validate_filename` 0 · `Utils.normalize_title` 0 · `file_extraction._sanitize_filename` internal) | see left | `Prompts_DB.py:5091` inline, `Canvas/models.py:53`, `comfyui _safe_filename`, `snapshot_models._safe_filename` (strict validators, fine) | `input_validation.validate_filename` and `normalize_title`: rg → 0, collect → 0 | consolidate into one `path_validation.sanitize_filename`; delete the two dead ones |
| `config.coerce_bool_setting` | 24 | 21 private `_coerce_bool`-family defs (+ `egress._config_enabled`, `tls_trust._TRUE/_FALSE_STRINGS`) | live | adopt via a stdlib-only `Utils/coerce.py` leaf that config delegates to (cycle constraint) |
| `Widgets/base_components.py` (parent seed, outside slice) | 0 | `Widgets/form_components.py` (3 importers) covers the same ground | `rg -n "Widgets\.base_components\|from \.base_components" tldw_chatbook Helper_Scripts scripts` → 0; only CSS-census tests name the file | delete |
| `Utils/text.py` ×5 (`format_metadata_as_text`, `format_transcription`, `extract_text_from_segments`, `format_text_with_line_breaks`, `format_transcript`) | 0 | `extract_text_from_segments` is a verbatim twin of `Utils.py:84` (live) | rg → definitions only | delete |
| `Utils/cost_estimation.py` | 0 | – | rg → 0; orphan of removed `cost_estimation_widget` (`test_evals_deletion_guard`) | delete |
| `Utils/debug_helpers.py` | 0 | – | rg → 0 | delete |
| `Utils/ingestion_preferences.py` | 0 | – | rg → 0 | delete |
| `Utils/splash_animations.py` | 0 | – | rg → 0 | delete |
| `Utils/Splash.py` | 1 (`Splash_Screens/card_definitions.py`) | – | live; `print_tldw_ascii`, `get_splash_card_config` → 0 | keep module; delete the two functions |
| `Utils/paths.py` project-* (4 fns + 4 constants) | 0 | – | rg → 0; the import they depend on always fails (probe above) | delete; keep `get_user_data_dir` |
| `Utils/input_validation` dead validators (`validate_email`, `validate_username`, `validate_ip_address`, `validate_port`, `validate_and_raise`, `validate_filename`) | 0 | `validate_port` re-rolled at `llm_management_events_vllm.py:104` | rg → 0; collect → 0 | delete |
| `Utils/Utils.py` 25 dead symbols (listed in the P3 finding) | 0 | – | per-symbol rg → 0; collect → 0 | delete |
| `Utils/db_status_manager._get_db_status_widget` | 0 | – | rg → def only | delete |
| `Utils/secure_temp_files.cleanup_all_temp_files` | 0 (`cleanup_all` only from `__del__`) | – | rg → 0 | delete the wrapper; keep the manager |
| `Utils/ui_responsiveness_artifacts.py` | 0 in package; used by `Tests/UI/run_workbench_soak.py` + its test | – | harness-only by design | keep |


## Method appendix

**Phases.** 0 provenance and baseline → 1 mechanical candidate generation → 2 per-slice reads by 29 read-only subagents (≤6 concurrent) → 3 cross-module consolidation → 4 verification of every P0/P1 → 5 this report. `progress.md` carries the resumable state; it was written after every slice because the run outlived several context windows and two usage limits.

**Scripts** (copies in `candidates/`, originals in the session scratchpad):

| Script | What it produced |
|---|---|
| `dup_census.py` | `dup_by_name.tsv` (same name, ≥3 files), `dup_verbatim.tsv` (byte-identical bodies, docstring stripped), `dup_shape.tsv` (identical after anonymising names/args/constants) |
| `helper_adoption.py` | `helper_adoption.tsv` — every public symbol in `Utils/*.py`, `DB/base_db.py`, `DB/sql_validation.py`, `Widgets/{form,base}_components.py` with its in-package importer count, resolving relative imports and module aliases |
| `pattern_greps.py` | `patterns/*.tsv` — 27 AST-based pattern censuses (not greps: `re.compile` *inside a def*, `get_cli_setting` *inside compose/loop/retry*, `fetchall` whose preceding `execute` has no LIMIT, `query_one` in a timer callback without an enclosing `try`, and so on) |
| `variants.py`, `dump_clusters.py` | per-cluster variant tables: every definition of a seed name, grouped by exact body and by anonymised shape, with the source of each distinct variant |
| `make_excerpts.py` | the 29 per-slice candidate excerpts each subagent was given |

**Counts from Phase 1** (Tier-1 files unless noted):

| Census | Count |
|---|---|
| Duplicate definitions, Tier-1 (1,058 files) | 519 same-name rows (≥3 files), 92 verbatim-clone groups, 247 shape-clone groups |
| Duplicate definitions, whole package (2,499 files) | 1,931 / 267 / 709 — the prompt measured 1,823 / 244 / 665 at this SHA, about 6 % lower, same order; the difference is exclusion sets |
| Shared-helper adoption | 453 public symbols; **207 with zero in-package importers**; 10 modules with zero |
| Pattern censuses (rows / files) | `fetchall` no LIMIT 210/32 · `fetchall` dynamic SQL 107/11 · `except Exception: pass` 285/119 · `except Exception: return` 507/173 · `run_worker(coroutine)` 498/77 · function-body imports 1,943/292 · `try/except ImportError` guards 218/109 · `query_one` in a timer without try 37/10 · `re.compile` in a def 19/13 · `get_cli_setting` in compose/loop/retry 17/7 · loguru+logging in one file 8 · `threading.Lock` + `.execute(` 16 · raw `1024*1024` 116/54 · `tempfile` without `secure_temp_files` 24/17 · `mkdir(parents=True…)` 30/22 · `os.replace` without `atomic_file_ops` 13/10 · inline truncation 24/17 · `strftime` 45/33 (17 distinct formats) · dotted `get_cli_setting` 32/12 · mutable class attrs 24/9 · `id()`-keyed dicts 23/7 · `.plain` read-back 55/28 · `run_worker(exclusive=True)` without `group=` **1** |

**Subagents.** One per slice, read-only, each given §0/§2/§4 of the prompt, its own candidate excerpt, the "already handled" list, and the output contract. Each returned files-read-in-full versus sampled, findings in contract format, and a disposition (`confirmed` / `retired` / `unverified`) for every candidate row in its excerpt. After the first usage limit, every brief required incremental report writes; that is why the five interrupted runs still produced usable partials.

| Slice | Lines | Slice | Lines | Slice | Lines |
|---|---:|---|---:|---|---:|
| ENTRY-app | 21,050 | LLM | 22,561 | UI-personas | 16,414 |
| ENTRY-config | 19,413 | UTILS | 24,868 | DB-rest (partial) | 34,245 |
| CHAT-controller | 29,048 | UI-chat | 25,215 | RAG (partial) | 27,572 |
| CHAT-store | 22,245 | UI-library | 35,680 | EVENTS (partial) | 16,816 |
| CHAT-bridge | 29,362 | UI-settings | 30,810 | UIM-console (partial) | 47,714 |
| AGENTS | 40,468 | DB-chacha | 24,180 | UIM-library (partial) | 47,026 |
| TOOLS-MCP | 43,910 | DB-media-base | 11,932 | | |

**Verification (Phase 4).** Every P0/P1 from a completed slice was re-run by me from the worktree root under the isolated profile; the commands and their output are in `phase4-verification.md`. One claim did not reproduce and is recorded as contested; one was corrected upward to P0 after tracing its caller chain.

**What would finish this review.** Eight slices were never started (`Chat/` remainder 126,849 lines, `UI/Navigation`+`MCP_Modules`+`Persona_Modules`, and the four `Widgets/` slices — 317,326 lines in total) and five are partial. The same prompt, pointed at those slices with the excerpts already in `candidates/`, resumes cleanly: `progress.md` names each one's state.
