# progress.md — core-runtime code review 2026-09-17

Resumable state. One line per package per phase. Worktree: `/Users/macbook-dev/Documents/GitHub/tldw-review` @ origin/dev d8fb4053f9 (detached, clean).
Interpreter: main checkout `.venv/bin/python` (3.12.11, Textual 8.2.8) run from the WORKTREE ROOT (running from inside `tldw_chatbook/` resolves the editable install to the main checkout — caught 19:00).
Any python import of the package must use the isolated env in scratchpad `env.sh` (HOME/XDG/TLDW_CONFIG_PATH → scratch); against the real profile `config.py:6 admit_startup()` exits with `Recovery required: recovery_scope_uncertain` (ADR-126, see report).

## Phase 0 — DONE
- preflight: all checks pass (exit 0), tree still clean after.
- ruff fatal-only (E9,F63,F7,F82): 0 in every Tier-1 package except UI modules = 1 (`UI/Library_Modules/library_notes_controller.py:5009 F821 NoteImportExecutor` — annotation-only, `from __future__ import annotations` at line 501, module imports; P3).
- ADRs: 217 files in backlog/decisions (list in candidates/adr_list.txt).
- Already-handled open tasks: see report §Scope.

## Phase 1 — DONE (counts)
- dup census Tier-1 (1058 files): 519 same-name rows (≥3 files), 92 verbatim-clone groups, 247 shape-clone groups.
- dup census whole package (2499 files, Splash_Screens excluded): 1931 / 267 / 709 (prompt measured 1823/244/665 — +6%, same order; prompt's run likely had a different exclusion set).
- helper adoption: 453 public symbols across Utils/*.py + base_db + sql_validation + form_components + base_components; 207 have zero in-package importers; modules with 0 importers: Utils.{Splash,Splash_Strings,cost_estimation,debug_helpers,ingestion_preferences,pagination,splash_animations,ui_helpers,ui_responsiveness_artifacts}, Widgets.base_components.
- pattern greps over 962 Tier-1 files (rows / files): re_compile_in_def 19/13 · get_cli_setting_hot 17/7 · fetchall_no_limit 210/32 · fetchall_dynamic_sql 107/11 · query_one_in_timer_no_try 37/10 · run_worker_exclusive_no_group 1/1 · run_worker_coroutine 498/77 · except_exception_pass 285/119 · except_exception_return 507/173 · lock_and_execute 16 files · try_import_guard 218/109 · loguru_and_logging 8 files · strftime 45/33 (17 distinct formats) · inline_truncate 24/17 · raw_mkdir 30/22 · os_replace_no_atomic 13/10 · tempfile_no_secure 24/17 · legacy_markers 1330/234 · dotted_section_setting 32/12 · mutable_class_attr 24/9 · id_keyed_dict 23/7 · plain_readback 55/28 · function_body_import 1943/292 · raw_1024x1024 116/54 · token_est_len_div4 1/1 · inline_path_check_no_pv 1/1 · sys_path_mutation 1/1.

## Phase 2 — per-package reads (subagent | lines | status | files full/sampled | findings)
(filled as each returns; reports in scratchpad phase2/<name>.md)

### Wave 1 dispatched 19:20 — ENTRY-app, ENTRY-config, CHAT-controller, CHAT-store, DB-chacha, UTILS (running)
### Phase 3 pre-work done while waiting: cluster dumps in scratchpad clusters/ (variants_{time,json,bytes,coerce,cred,db,interop,ui}.md); findings so far noted in report drafting notes (scratchpad/phase3_notes.md)
| ENTRY-config | 19,413 | DONE 19:33 | all full except config.py 3572-5708 (TOML template) + Constants literal blocks | P1×2 P2×4 P3×7 (report: scratchpad phase2/ENTRY-config.md) |
### Wave 2 started 19:34: DB-media-base dispatched (slot from ENTRY-config)
| ENTRY-app | 21,050 | DONE 19:37 | read in full 1-21050 | P1×1 P2×2 P3×8 (phase2/ENTRY-app.md) |
### 19:38 AGENTS dispatched (slot from ENTRY-app). Running: CHAT-controller, CHAT-store, DB-chacha, UTILS, DB-media-base, AGENTS
| DB-chacha | 24,180 | DONE 19:42 | read in full (SQL literal block 720-3306 collapsed per trigger family) + base_db, migrations/README, sqlite_datetime_fix | P1×1 P2×4 P3×8 (phase2/DB-chacha.md) |
### 19:43 TOOLS-MCP dispatched. Running: CHAT-controller, CHAT-store, UTILS, DB-media-base, AGENTS, TOOLS-MCP
| CHAT-controller | 29,048 | DONE 19:50 | read in full 1-29048 | P2×5 P3×6 (phase2/CHAT-controller.md) |
### 19:51 CHAT-bridge dispatched. Running: CHAT-store, UTILS, DB-media-base, AGENTS, TOOLS-MCP, CHAT-bridge
| UTILS | 24,868 | DONE 20:05 | 63/64 read in full (Splash_Strings sampled) | P1×1 P2×4 P3×12 (phase2/UTILS.md) |
### 20:06 LLM dispatched. Running: CHAT-store, DB-media-base, AGENTS, TOOLS-MCP, CHAT-bridge, LLM
| CHAT-store | 22,245 | DONE 20:10 | read in full 1-22245 | P2×3 P3×6 (phase2/CHAT-store.md) |
### 20:11 UI-library dispatched. Running: DB-media-base, AGENTS, TOOLS-MCP, CHAT-bridge, LLM, UI-library
| DB-media-base | 11,932 | DONE (report complete despite the agent being killed by the 23:50 usage-limit) | see report coverage table | 11 findings (phase2/DB-media-base.md) |
| AGENTS | 40,468 | DONE (same) | see report coverage table | 12 findings (phase2/AGENTS.md) |
### 23:50 PT usage limit: TOOLS-MCP, CHAT-bridge, LLM, UI-library killed before writing (UI-library had read the whole file). Limit reset; 23:55 re-dispatched all four with INCREMENTAL report writes + dispatched UI-chat, UI-settings. Running: TOOLS-MCP, CHAT-bridge, LLM, UI-library, UI-chat, UI-settings. Not started: CHAT-rest-1/2/3, DB-rest, EVENTS, RAG, UI-personas, UIM-console, UIM-library, UIM-nav-mcp-persona, W-top, W-console-1/2, W-library, W-persona-settings-chat (15).
| LLM | 22,561 | DONE 00:30 (retry) | all 19 files in full | P1×3 P2×6 P3×4 (phase2/LLM.md) |
| TOOLS-MCP | 43,910 | DONE 00:32 (retry) | 7 files >1.5k + 14 more in full (~21k); 9 sampled; 21 small mechanical | P1×2 P2×8 P3×4 (phase2/TOOLS-MCP.md) |
### 00:33 DB-rest + RAG dispatched. Running: CHAT-bridge, UI-library, UI-chat, UI-settings, DB-rest, RAG. Not started: CHAT-rest-1/2/3, EVENTS, UI-personas, UIM-console, UIM-library, UIM-nav-mcp-persona, W-top, W-console-1/2, W-library, W-persona-settings-chat (13)
| UI-chat | 25,215 | DONE 00:40 | read in full 1-25215 | P1×3 P2×3 P3×5 (phase2/UI-chat.md) |
| CHAT-bridge | 29,362 | DONE 00:41 (retry) | all 4 files in full | P2×3 P3×13 (phase2/CHAT-bridge.md) |
| UI-library | 35,680 | DONE 00:43 (retry) | read in full 1-35680 | P1×1 P2×1 P3×8 (phase2/UI-library.md) |
### 00:45 EVENTS, UI-personas, UIM-console dispatched. Running: UI-settings, DB-rest, RAG, EVENTS, UI-personas, UIM-console. Not started: CHAT-rest-1/2/3, UIM-library, UIM-nav-mcp-persona, W-top, W-console-1/2, W-library, W-persona-settings-chat (10)
| UI-settings | 30,810 | DONE 01:08 | read in full 1-30810 | P1×1 P2×3 P3×8 (phase2/UI-settings.md) |
### 01:09 UIM-library dispatched. Running: DB-rest, RAG, EVENTS, UI-personas, UIM-console, UIM-library. Not started: CHAT-rest-1/2/3, UIM-nav-mcp-persona, W-top, W-console-1/2, W-library, W-persona-settings-chat (9)
| UI-personas | 16,414 | DONE (report complete despite the 04:50 usage-limit kill) | read in full 1-16414 | P2×4 P3×5 (phase2/UI-personas.md) |
### 04:50 PT SECOND usage limit: DB-rest, RAG, EVENTS, UIM-console, UIM-library killed — all left USABLE partials (the incremental-write rule worked): DB-rest (Subscriptions_DB full + 1 P2), RAG (rag_service + vector_store full), EVENTS (full reachability census), UIM-console (7 files full + probes + 1 candidate P1), UIM-library (mechanical sweeps + 1 P2). UI-personas finished before the kill.
### 07:35 re-dispatched the 5 partials as RESUMES + CHAT-rest-1. Running: DB-rest, RAG, EVENTS, UIM-console, UIM-library, CHAT-rest-1. Not started (8): CHAT-rest-2, CHAT-rest-3, UIM-nav-mcp-persona, W-top, W-console-1, W-console-2, W-library, W-persona-settings-chat.

## Phase 3/4/5 — DONE 07:50 (first full pass)
- Phase 3 consolidation: cluster variant tables built for every seed name (scratchpad `clusters/variants_*.md`); 15 clusters tabled in report.md with canonical homes, outward/downward LOC, sizes and recs. Two prompt seeds corrected (Interop "one template" retired; `_get_connection` Scheduling copies are Protocol stubs).
- Phase 4: every P0/P1 from a completed slice re-run by me → `phase4-verification.md`. 1 promoted (Media transaction leak, P1→P0, caller chain traced), 1 contested (folder-import per-job commit cost), 1 prior-audit entry retired (config reads are cached but cost 11.1 ms each).
- Phase 5 deliverables written: `report.md` (1,112 lines), `gap-candidates.md` (8 batches + 6 recommend-against), `phase4-verification.md`, `slices/*.md` (21 per-slice reports), `candidates/*.tsv` + the 5 census scripts.
- STILL RUNNING (resumed 07:35, will be folded into report.md as they land): DB-rest, RAG, EVENTS, UIM-console, UIM-library, CHAT-rest-1.
- NOT STARTED (7 slices, 317k lines — named in report.md's coverage table as unknown, not clean): CHAT-rest-2, CHAT-rest-3, UIM-nav-mcp-persona, W-top, W-console-1, W-console-2, W-library, W-persona-settings-chat.

## RUN COMPLETE — 2026-09-18 08:40
All 21 dispatched slices finished (the 5 killed at 04:50 were RESUMED from their partials, not restarted; CHAT-rest-1 added). Final: **614,475 of 887,855 Tier-1 lines reviewed (69 %)**, P0 1 · P1 25 · P2 100 · P3 159.
Wave-2 P1s all re-verified by the orchestrator (see phase4-verification.md "Wave 2"): RAG cache permanent no-op (worse than filed), fspicker always errors, no process group on LLM servers, Console double-escape, prompts IN() limit, AgentRuns reconcile in a compose path, RAG sync ChromaDB on the loop.
Deliverables in this directory: report.md (2,030 lines), gap-candidates.md, phase4-verification.md, progress.md, slices/*.md (21), candidates/*.tsv + 6 scripts.
NOT STARTED (8 slices, 273,380 lines): CHAT-rest-2, CHAT-rest-3, UIM-nav-mcp-persona, W-top, W-console-1, W-console-2, W-library, W-persona-settings-chat. Their candidate excerpts already exist in the session scratchpad; re-running this prompt against them resumes cleanly.
Worktree /Users/macbook-dev/Documents/GitHub/tldw-review @ d8fb4053f9 left clean and in place; remove with `git worktree remove /Users/macbook-dev/Documents/GitHub/tldw-review` when done.

## RUN RESUMED 2026-09-18 08:5x — finishing the remaining 8 slices
Stopping at 21/29 was the orchestrator's error, not a budget limit: the usage window had reset and the finding rate was high. Dispatched the rest.
Wave A (6): CHAT-rest-2, CHAT-rest-3, W-console-1, W-console-2, W-library, W-top.
Wave B (2, queued): UIM-nav-mcp-persona, W-persona-settings-chat.
All deliverables will be regenerated from all 29 slices when these land.

## RUN ACTUALLY COMPLETE — 2026-09-18, all 29 slices
887,855 / 887,855 Tier-1 lines (100 %). Final: **P0 4 · P1 49 · P2 146 · P3 209** (408 findings).
The 8 slices added after the premature stop contained 3 of the 4 P0s. Deliverables regenerated from all 29 slices.
Verified by the orchestrator in wave 3: the 3 new P0s (all three crash paths traced to source + Textual's worker contract), and the systemic `rich.markup.escape` failure (RE_TAGS only covers [a-z#/@ → '[IMPORTANT]' renders blank).
One self-correction logged: my own run_worker enumeration produced a false positive (`:226` passes an already-called @work coroutine); caught via a decorator check before it reached the report.
