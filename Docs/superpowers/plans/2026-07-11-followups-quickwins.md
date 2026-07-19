# Follow-ups quick-wins batch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Eight independent, low-risk follow-ups from backlog tasks 152–171. Branch `claude/followups-quickwins` off dev 050aed94. Anchors exact at branch point; grep symbols, lines drift.

**Goal:** Clear eight mechanical/low-risk backlog follow-ups in one reviewed batch: two chatbook-export fixes, three latent-bug fixes, a badge-drift fix, a logging cleanup, a Home control fix, an ingest→notes refresh, and a legacy-CSS prune.

**Global Constraints:** explicit-path staging (NEVER `git add -A`); Fable 5 co-author line; RED-first for behavior changes; `escape_markup` for user text in labels; parameterized SQL only; CSS via source module + `./build_css.sh` (commit both); venv pytest with isolated HOME. Test command: `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <files> -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread`.

## Cluster 1 — mechanical code fixes (RED-first)

### T170 — console-rail badge ignores the empty staged summary (task-170)
`Tests/Chat/test_console_rail_state.py::test_console_context_rail_badge_ignores_empty_staged_summary` fails: `'staged' == 'session'`. Root cause found: `ConsoleStagedContextState.empty().summary == "No sources attached."`, but `console_rail_state.py:34 _INACTIVE_STAGED_SUMMARIES` only lists `"no live work item is staged"`/`"no staged work"`, so `_has_active_staged_summary` treats the empty-state string as active. Fix: add the normalized empty-state summary to the set. `_normalized_inactive_text` lowercases + strips trailing `.`, so the entry is `"no sources attached"`. The failing test IS the RED; add a drift-guard assertion that `_normalized_inactive_text(ConsoleStagedContextState.empty().summary) in _INACTIVE_STAGED_SUMMARIES` so a future empty-copy change can't silently reintroduce it (import ConsoleStagedContextState in the test, not the module, to avoid a new import edge).

### T155 — chatbook export drops media metadata (task-155)
`ChatbookCreator._collect_media` (chatbook_creator.py, method ~:756) reads `media_type`/`created_at`/`updated_at` from the Media row dict, but the real columns are `type`/`ingestion_date`/`last_modified` — all resolve to None, silently losing media type + timestamps on every export. Fix the key reads to the real columns (grep the Media schema / `get_media_by_id` result shape to confirm exact names; `summary`/`prompt`/`media_keywords` live in join/analysis tables — leave those as best-effort/None, out of scope). RED: extend the F4 round-trip test (`Tests/Library/test_library_export_roundtrip.py`) to assert the exported media entry carries the correct `type` and a non-None timestamp, then re-import and assert type survives.

### T158 — export success message carries item counts (task-158)
The success completion (`library_screen.py` `_apply_library_export_success` ~:3495-3515) reports only the path; the creator's own `outcome["message"]` (item counts, deleted-mid-export skips) is discarded. Append the creator message (or its count portion) to the success notification. Keep the existing "N characters auto-included" dependency-info suffix and the registry-failure warning. RED/anchor: the execution test asserting the success notify text — extend it to assert the counts/message are included.

### T168 — three latent unreached-path bugs (task-168)
(a) `PDF_Processing_Lib.py:666` calls undefined `analyze_media_content` — resolve to the correct symbol (grep for the real analysis entry, likely `analyze`/`summarize_*`) or remove the dead branch. (b) `OCR_Backends.py` `DocextOCRBackend.cleanup` references an unimported `torch` — add the guarded import or drop the reference. (c) `PDF_Processing_Lib` `process_pdf` result check tests `'error' in result` (always present, None default) instead of truthiness — change to `result.get('error')`. Each is on an unreached path today; add a focused unit test per fix where cheap (the process_pdf error-key one is directly testable with a stub result).

### T171 — simplify redundant opt(exception=False) sites (task-171)
`grep -rn "opt(exception=False)" tldw_chatbook/ --include='*.py'` → 13 sites. `logger.opt(exception=False).<level>(...)` == `logger.<level>(...)`. Rewrite each to the plain call. No behavior change (verify none pass a real exception). Anchor: `python -c "import tldw_chatbook.app"` + a grep asserting zero `opt(exception=False)` remain.

**Cluster 1 commit:** `fix(chatbooks,ingestion,console,logging): batch quick-win follow-ups (155,158,168,170,171)`

## Cluster 2 — light-judgment fixes

### T154 — Home Pause control has no ingest semantics (task-154)
The generic Home Pause control renders for local ingest items (they enter Running) but has no wired pause behavior. Decision: SUPPRESS it for ingest-kind items (no queue-pause action exists; wiring one is a separate feature). Find where Pause is emitted in `Home/dashboard_state.py`'s control builder and gate it out when the selected item is an ingest job (item_id starts `local:ingest:` or the equivalent kind marker). RED: a dashboard-state unit — a selected ingest item's controls do NOT include the pause control; a non-ingest running item still does.

### T167 — wire post-ingest Library notes refresh (task-167)
`Event_Handlers/note_ingest_events.py::handle_ingest_notes_import_now_button_pressed` imports notes but never refreshes the Library notes canvas. Since Library composes fresh per visit (freeze fix), the only gap is when the Library notes canvas is ALREADY mounted at import time. Fix: after a successful ingest-notes import, if the active screen is a mounted LibraryScreen, trigger its local-source snapshot refresh (grep the screen for the snapshot-refresh entry, e.g. `_refresh_local_source_snapshot`/the worker that repopulates notes). Guard defensively (screen may not be Library). RED: a pilot/unit — with a mounted Library screen on the notes canvas, an ingest-notes import causes a notes snapshot refresh (assert the refresh method is invoked, or the new note appears).

### T169 — prune legacy notes-window CSS + build_css warning (task-169)
Zero-survivor discipline (F2 lesson): `.notes-content-header`/`.notes-content-label` (source in `css/features/_notes.tcss` and/or `_sidebars.tcss`; generated into `tldw_cli_modular.tcss:549`) and the rest of the pre-workbench notes-window CSS. For EACH selector, grep its id/class across `tldw_chatbook/*.py` — delete only zero-hit selectors; keep any live one and note why. `#notes-window` is still named in `app.py:1968 ALL_MAIN_WINDOW_IDS` — adjudicate: if no `notes-window` widget is composed anywhere (grep), remove that id from the list; if something still mounts it, keep and document. Rebuild `./build_css.sh` (commit source + generated). Also resolve the `build_css.py` "Missing module: features/_evaluation_v2.tcss" warning (a stale module reference in the build manifest — remove the dangling entry or restore the module; grep `_evaluation_v2` in build_css.py). Gate: `grep -rn "notes-content-header\|notes-content-label" tldw_chatbook/` prints nothing after; `./build_css.sh` warns no more about _evaluation_v2.

**Cluster 2 commit:** `chore(home,notes,css): suppress ingest Pause, wire notes refresh, prune legacy notes CSS (154,167,169)`

## Verification & gate

Combined gate: `Tests/Chat/test_console_rail_state.py Tests/Library/ Tests/Chatbooks/ Tests/Home/ Tests/UI/test_library_shell.py Tests/UI/test_home_screen.py Tests/Local_Ingestion/` + `python -c "import tldw_chatbook.app"` + `python -m compileall tldw_chatbook -q`. No visual QA (no user-facing pixel change except suppressing a control — capture the Home ingest-item canvas if cheap). PR to dev; merge only on explicit user authorization. Mark tasks 154/155/158/167/168/169/170/171 Done via `backlog task edit`.
