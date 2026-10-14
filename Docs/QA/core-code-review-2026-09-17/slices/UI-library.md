# UI-library — tldw_chatbook/UI/Screens/library_screen.py, 35680 lines

## Coverage
| file | lines | read in full / sampled (which ranges) / mechanical only |
|---|---|---|
| tldw_chatbook/UI/Screens/library_screen.py | 35680 | **read in full**, 1-35680, in ~1200-line chunks (all 30 chunks); plus mechanical scans (59 `run_worker(` sites, 72 indented import lines, 5 `set_timer` sites, 0 `reactive(`, class attrs, `id(` sites, `.plain`/`str(label)`). |
| tldw_chatbook/UI/Library_Modules/library_media_controller.py | — | sampled: `_library_media_request_matches_current_authority` (:1383), `_library_media_item_traversal_active` (~:3232), banner call sites (:716 :1021 :3754 :3913) — evidence for items (11) and the P1 only. |
| tldw_chatbook/Notes/notes_scope_service.py, tldw_chatbook/Workspaces/registry_service.py, tldw_chatbook/Library/review_set_service.py, tldw_chatbook/config.py (`save_setting_to_cli_config`), Tests/Architecture/test_screen_size_ratchet.py, .github/workflows/test.yml | — | sampled: only the seams named in findings (to resolve "is this sqlite on the loop?" / "does CI run the ratchet?"). |

Interpreter/env: every probe and pytest run used `cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY ...`. `ruff check --select E9,F63,F7,F82` on the file: All checks passed.

## Findings

### P1 [D2] — In the media Reader, every `]`/`[` keypress, focus change and viewer sync runs 2–4 synchronous sqlite reads on the event loop (a full up-to-500-row review-set load each time), ~6 ms per read pair measured
- Where: `tldw_chatbook/UI/Screens/library_screen.py:31238` `_review_set_active` (docstring: "this runs on every key resolution and footer render"), `:31098` `_active_review_progress`, `:31131` `_active_review_loaded_at_last`, `:31169` `_active_review_set_banner`, `:30987` `_walk_active_review_set_unguarded`; all call `service.get_active_review_set()` (loads header + every pinned item, `REVIEW_SET_CAP`=500) and `:30911` `_review_set_live_ids` (`media_db.execute_query("SELECT id FROM Media WHERE id IN (…)")` + `fetchall`) synchronously. Reached from `check_action` (`:25411`, `:25474` — bindings `]` `[` `R` `m`), the footer builder (`:4341`, `:4362` via `_register_footer_shortcuts`, which runs on `compose_content`, every rail switch, every `on_descendant_focus` in the viewer `:10705`), and the media controller's viewer build/sync (`library_media_controller.py:3754`, `:3913` → screen `:3124`).
- Evidence: probe under the isolated env (LibraryCollectionsDB + ReviewSetService with a 500-item active set; empty MediaDatabase):
  `get_active_review_set (500 items): median 5.064 ms, p95 6.415 ms` · `live_ids IN(500) on Media: median 0.080 ms, p95 0.116 ms` · `one _active_review_progress() equivalent (both queries + progress): median 6.218 ms, p95 8.085 ms`. Per `]` press in a 500-item set the chain is `check_action`→`_review_set_active` (≈5 ms) → `_walk…` (`get_active_review_set` + live_ids + `mark_item_done`/`set_cursor`/`refresh_completion` writes) → viewer sync → `_active_review_set_banner` (≈6 ms) → `_register_footer_shortcuts` → `_active_review_progress` (≈6 ms) + `_active_review_loaded_at_last` (≈5 ms): ≈25–30 ms of loop-blocking sqlite per keypress, scaling with set size. Probe script: `<SCRATCH>/probe_review_set` (20 lines, in the transcript).
- Why it matters: the Reader's ]/[ traversal and footer refresh stall the UI loop for tens of ms per gesture on a large review set; every off-loop sibling in the same section (`_review_these_worker`, `_review_set_picker_worker`, `_auto_resume_review_set_worker`, :31397–:31899) already routes the same reads through `_run_library_service_call(isolate_in_worker=True)`, so this is the gap, not the design.
- Recommended correction: memoise the active-set snapshot the way `_decorate_library_media_reviewed` (:16097) already does — one screen-level `_active_review_set_snapshot()` keyed on `service.revision` (bumped by every write in `ReviewSetService._write`) returning `(review_set, live_ids)`, and route the five synchronous readers through it; the write path (`_walk…`) stays synchronous but drops from 2 loads to 1. No new helper module.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/UI/test_review_set_banner.py`, `Tests/UI/test_review_set_walker.py`, `Tests/Architecture/test_library_media_wiring.py` exercise these readers (behaviour, not call-count); no test asserts the per-call DB load, so a memo does not contradict a pinned requirement.
- Already covered: none

### P2 [D3] — The size ratchet that governs this file is RED at the reviewed dev commit: 35680 lines / 1327 methods against a pin of 33204 / 1276, in the required CI sweep
- Where: `Tests/Architecture/test_screen_size_ratchet.py:887` `"tldw_chatbook/UI/Screens/library_screen.py": ("LibraryScreen", 33204, 1276)`; the test's measure is `len(source.splitlines())` and `ast` method count of the class (:900-937).
- Evidence: `$PY -m pytest "Tests/Architecture/test_screen_size_ratchet.py::test_screen_does_not_grow_past_its_budget[tldw_chatbook/UI/Screens/library_screen.py]" -q` → FAILED at `assert lines <= max_lines` (35680 > 33204; methods 1327 > 1276). Provenance: at the pin commit 576daf57bb (2026-09-09) the file measured exactly 33204 lines (pin == measurement, recipe §6); 226 commits touched the file since; the first landed commit past the pin is 6b5f6ec83b "feat(library): Obsidian mode for Notes Import once (task-32129)" (33381 lines, author-dated 09-08 — a concurrent branch landing after the pin, the exact hazard the test's own docstring records). CI: `.github/workflows/test.yml:121` core-tests = `pytest Tests --ignore=Tests/UI -n auto … --num-shards=6`, no `-m` deselection, tests are `@pytest.mark.unit` → the failure is in the required sweep. (`chat_screen.py`'s row fails the same way — outside this slice, reported as a shared red.)
- Why it matters: the governance mechanism the decomposition recipe relies on (§6 "measure after final rebase, lower budgets in the landing PR", §17) is not holding: the screen grew +2476 lines / +51 methods in eight days with the ratchet red, which is precisely "the month in which library_screen.py tripled" the test was written to prevent. The test message says "Do NOT raise the budget to make this pass".
- Recommended correction: recipe §6 — the next Library PR (or a dedicated landing PR) must bring the file back under 33204/1276 or the program owner must record a deliberate re-pin with the numbers; and the PR merge gate for `library_screen.py` changes must run this test (it is in core-tests, so the red is being merged past — check the branch-protection/"cancelled checks" situation the lessons files record).
- Size: M · ADR: no (governed by `backlog/docs/library-decomposition-recipe.md` §6/§17) · Confidence: verified
- Pinning test: `Tests/Architecture/test_screen_size_ratchet.py::test_screen_does_not_grow_past_its_budget[…library_screen.py]` — states the budget as a requirement, currently red.
- Already covered: task-31202 is the settings_screen row; no open task names the library_screen breach (task-32170 / task-32013 move bodies but do not mention the red pin).

### P3 [D1] — Two persistence workers swallow every failure of `save_setting_to_cli_config` with no log line (rail-section prefs, search history)
- Where: `library_screen.py:21546-21552` `_save_library_search_history` (`@work(thread=True)`, `except Exception: pass`) and `:21602-21608` `_save_library_rail_preferences` (same). Compare the sibling lifecycle drainer `:21415-21435` (turns the same failure into a visible "the choice may not be remembered" line), `:24949-24964` / `:26579-26594` (toast), `:27719-27729` / `:29177-29195` (`logger.error`).
- Evidence: read only — the four persistence workers for `[library.*]` settings use three different failure grammars; these two are silent (excerpt rows `except_exception_pass :21551`, `:21607` confirmed).
- Why it matters: a read-only config dir / disk-full / permission error silently drops the preference AND leaves nothing in the log (consequence is low: a UI preference, re-derivable).
- Recommended correction: `logger.warning("Failed to persist %s", key)` in both `except` blocks (Size S); optionally the same warning line the lifecycle drainer uses.
- Size: S · ADR: no · Confidence: inferred (no probe; the swallow is textual)
- Pinning test: `Tests/UI/test_library_crit8_collections_row.py`, `Tests/Architecture/test_library_cluster_membership_census.py` reference the worker names (wiring/census), none asserts silence.
- Already covered: none

### P3 [D2] — Synchronous sqlite / file I/O on the event loop from press handlers (bounded, one-shot)
- Where: `:6446` `db.get_media_by_id` in `@on(LibraryMediaViewer.SpeakerRenamed)`; `:13726` `registry.unlink_membership`, `:13777-13804` `get_active_workspace`/`get_item_memberships`/`link_membership` (`Workspaces/registry_service.py:1114` → `with self.db.transaction() as conn: conn.execute(...)`, confirmed sqlite); `:16120` `service.get_active_review_set()` in `_decorate_library_media_reviewed` (memoised on `revision`); `:16207` `can_rename_meeting_speakers` (DB + `exists()`, memoised per id); `:29318`/`:29328` `media_db.get_media_by_url`/`get_media_by_hash` in `_open_job_in_library`; `:21001` and `:27153` `validated_path.write_text(...)` (note/prompt export, docstrings acknowledge the sync write); `:32608` `_meeting_speaker_legend_rows` (parses `transcript.jsonl`, on memo miss only).
- Evidence: read only; each is a point read or one small write per explicit gesture; the rest of the file routes DB through `_run_library_service_call` (`:12930`: `asyncio.to_thread(run_finite_local_worker, …)`).
- Why it matters: consistency with the file's own rule ("blocking — worker thread only", `:27827`); none is a hot path.
- Recommended correction: none required individually; if the P1 memo helper lands, route `:16120` through it too. Leave the rest.
- Size: S · ADR: no · Confidence: inferred
- Pinning test: none
- Already covered: none

### P3 [D2] — Whole-screen `refresh(recompose=True)` on every external-preparation status flip
- Where: `:28329-28340` `_set_library_external_status` (`if changed and _is_mounted: self.refresh(recompose=True)`), called ≥8 times along the Parakeet/VAD preparation path (`:28637`, `:28768`, `:28784`, `:28804`, `:28813`, `:28861`, `:28874`, `:28901`, `:28908`, `:29049`, `:29113`, `:29128`).
- Evidence: read only; 28 explicit `self.refresh(recompose=True)` statements in the file; the whole-surface pin is `LIBRARY_WHOLE_SCREEN_RECOMPOSE_MAX = 63` in `Tests/UI/test_library_recompose_ratchet.py` (could not be run here — see UNVERIFIED).
- Why it matters: a status-line text change is the textbook targeted-update case (`_update_library_ingest_gate` :20240 already patches the ingest canvas in place); each flip tears down the rail/nav/footer during a multi-step install flow.
- Recommended correction: patch `LibraryIngestCanvas`'s `external_status`/`external_busy` in place (the canvas already receives them as constructor kwargs, :15585-15590) and let the ratchet drop by ~1 site.
- Size: S · ADR: no · Confidence: inferred
- Pinning test: `Tests/UI/test_library_recompose_ratchet.py::test_library_screen_whole_screen_recompose_count_is_ratcheted` (ceiling, states the count as a requirement)
- Already covered: none (task-32170/task-32013 move handler bodies; the ratchet counts statements)

### P3 [D3] — Same rail-switch coroutine scheduled with and without an exclusive group
- Where: `:33870-33873` and `:33891-33894` `run_worker(self._select_library_rail_row(...), group="library_get_started_step")` (not exclusive) vs `:8897-8901`, `:8917-8921` `run_worker(…, exclusive=True, group="library_rail_row_switch")` for the same coroutine.
- Evidence: read only. Every other `exclusive=True` in the file carries a `group=` (59 `run_worker` sites + 8 `@work` decorators audited: 0 ungrouped exclusives).
- Why it matters: a double-press of a Get-started step runs two `_select_library_rail_row` coroutines concurrently; the seam is idempotent for the same row but does its flush/admission dance twice.
- Recommended correction: use `exclusive=True, group="library_rail_row_switch"` at both sites.
- Size: S · ADR: no · Confidence: inferred
- Pinning test: none
- Already covered: none

### P3 [D3] — Three redundant function-body imports of modules already imported at top level
- Where: `:21558` `from dataclasses import replace as dataclass_replace` (`dataclasses` imported at :6, `dataclasses.replace` used ~20× elsewhere); `:31517` `from rich.markup import escape` (top level `:35` `escape as escape_markup`); `:10224` `from ...Utils.input_validation import validate_conversation_archive_scope` (module imported at `:392`).
- Evidence: `ast` + `importlib.util.find_spec`/`hasattr` script under the isolated env → 64 function-body import statements, 37 distinct modules, **0 unresolvable**; the three above are the only ones whose module is also a top-level import. All others are documented lazy imports (pre-import payload ratchet `Tests/Performance/test_screen_preimport_payload_budget.py`, `# noqa: PLC0415` markers, "Lazy: review sets are one Reader mode").
- Why it matters: none of the 64 targets is dead; this is cosmetic.
- Recommended correction: use the top-level names.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P3 [D3] — Import-time monkeypatch of two controller classes from this module
- Where: `:35595` `LibraryConversationsController._safe_text = staticmethod(LibraryScreen._safe_text)`, `:35605` `LibraryExportController._safe_text = …` (a 40-line comment explains why: classmethod dispatch + circular import).
- Evidence: read only.
- Why it matters: a controller constructed in a test without importing `library_screen` has no `_safe_text` (the comment accepts this). It is the recipe's documented "class-level rebinding" shape.
- Recommended correction: out of scope by `backlog/docs/library-decomposition-recipe.md` §3 (monkeypatch-name routing); report only.
- Size: — · ADR: no (recipe) · Confidence: verified
- Pinning test: `Tests/Architecture/test_screen_size_ratchet.py` comments reference the binding
- Already covered: recipe §3

### P3 [D4b] — Intra-file copies with mild drift (no shared helper exists)
- Where (all in `library_screen.py`):
  (a) `:7593-7818` `_toggle_library_media_reader_pane`: seven per-destination blocks (collections/conversations/notes_files/notes/prompts/skills/media) of the same 8-step sequence; only notes/notes_files carry a real delta (work-session expand, `ignore_previous`); the sibling `_persist_library_reader_preference` (:6883) already dispatches the same 7 destinations through dict tables.
  (b) `canvas_kind → widget` builder ladder ×3: `_build_library_entry_active_child` (:11841), `_reconcile_library_entry_state` (:12499-12564), `compose_content` (:15340-15616); `isinstance(widget) → kind` ladder ×2 (:11918-11938, :12191-12203).
  (c) `:12969-12971` and `:13000-13002` the `<`/`>`/`javascript:`/`onclick=`/`onerror=` strip loop (`Utils/input_validation.py:1484` holds the same list but only DETECTS).
  (d) `:24460-24483` / `:24507-24530` byte-identical 24-line `browse_callback` closures (FileOpen vs SelectDirectory).
  (e) `:24949-24964` / `:26579-26594` `_persist_library_skill_editor_mode` / `_persist_library_prompt_editor_mode` (15 lines, key + noun differ).
  (f) `:34004-34057` three conversations pager handlers each carrying the same 8-line `_library_unavailable_browse_scope` pre-branch.
  (g) browse-location persistence: `:27426` `_persist_library_note_import_location`, `:27806` `_persist_library_ingest_location`, `library_notes_controller.py:5167` `_persist_library_notes_sync_location`, `library_file_notes_workspace.py:6808` `_persist_file_notes_browse_location` — four verbatim copies of `claim_browse_directory` + `run_worker(lambda: remember_browse_directory(...), thread=True)` differing only in the context key (excerpt `dup_shape` row confirmed).
  (h) `_handle_workspace_create_result` ×3 verbatim (screen `:35388`, `settings_screen.py:23967`, `Console_Modules/workspace.py:4739`) — the "offer profile interview then continue" chain (excerpt `dup_shape` row confirmed).
- Evidence: read; (g)/(h) bodies printed side by side in the transcript.
- Why it matters: (g) and (h) are cross-file and will drift the next time the interview/continuation contract changes; the rest are same-file consistency.
- Recommended correction: (g) one `persist_browse_location(node, context, key, path)` in `Library/library_browse_location.py` (already the home of the three primitives); (h) one `run_workspace_create_result(app, result, continue_fn)` in `Workspaces/` next to `interview_launch`; (a)-(f) table-driven where the recipe's next cleanup PR touches them.
- Size: S each · ADR: no · Confidence: verified (bodies compared)
- Pinning test: none
- Already covered: none

### P3 [D4a] — Dead shared helper with one re-roll: `Utils/Utils.py:253 truncate_content`
- Where: helper has **0 importers** outside `Utils/Utils.py`; `library_screen.py:13832-13844` `_hub_table_cell` re-rolls an ellipsis truncator (word-boundary aware, so output differs).
- Evidence: `grep -rl "\btruncate_content\b" tldw_chatbook --include='*.py' | grep -v Utils/Utils.py | wc -l` → 0 (a bare-name grep; the helper is not re-exported anywhere — `grep -rn "truncate_content" tldw_chatbook` shows only its definition).
- Why it matters: below the brief's P2 threshold (≥10 re-rolls); reported as a dead helper.
- Recommended correction: delete `truncate_content` or adopt it (the screen's word-boundary variant is the better one).
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P3 [D3] — The screen body still owns single-subsystem business logic the recipe says belongs in controllers (size-ratchet context)
- Where: 627 of the 1327 class methods are one-line delegators (`return self._<x>_controller.<m>(…)`) = the recipe §3 shim layer awaiting cleanup PRs; full bodies still screen-resident include Notes-only policy `_library_notes_footer_shortcuts` (:8244-8434, handed BACK to `LibraryNotesController` as an accessor at :3519), the Media bulk-delete/undo/trash/analysis bodies (:23042-23850, :32093-32457, :33169-33760), the Notes tree/mutation machinery (:17038-18700, :29963-30682), and the review-set section (:30885-31957). Cross-subsystem shell logic that legitimately stays: `_library_route_shortcuts_for_current_state` (:4173, 330 lines), `check_action` (:25050, 430 lines), `_select_library_rail_row_after_source_admission` (:22040, 357 lines), `compose_content` (:14808, 810 lines).
- Evidence: counts by grep (`grep -c "^        return self\._[a-z_]*controller\."` → 627); the ratchet numbers above.
- Why it matters: this is the remaining mass behind the red ratchet.
- Recommended correction: none new — task-32170 (phase-C media extraction of the 20 canvas-origin handler bodies), task-32013 (media controller exclusion debt), task-31650 (controller-globals bypass census), task-32089 (canvas_syncs dispatchers / state accessors) already own this; cite, do not re-split.
- Size: — · ADR: no (recipe §1/§2/§17) · Confidence: verified
- Pinning test: `Tests/Architecture/test_screen_size_ratchet.py`, `Tests/Architecture/test_library_modules_size_ratchet.py`
- Already covered: task-32170, task-32013, task-31650, task-32089

## Candidate dispositions
| candidate (file:line pattern) | confirmed / retired (why) / unverified (check) |
|---|---|
| dotted_section_setting :623 `library.ingest_options.generic` | retired — 3-arg form with explicit `None` default (:630/:635), resolves since TASK-1771; memoised on the app (:658-689) |
| dotted_section_setting :651 `transcription.transcribe_cpp` | retired — same, read once per config identity |
| dotted_section_setting :21058, :21452/:21468/:21471 `library.rail_state` | retired — 1-arg dotted read of the sub-dict; write-back `save_setting_to_cli_config("library.rail_state", …)` handles nested sections (config.py:8386) and the in-memory `app_config` is mirrored first (:21564-21574); read-back verified consistent |
| dotted_section_setting :21488/:21521/:21528 `library.search` | retired — same shape; history entries re-sanitised through `_safe_text` on read |
| dotted_section_setting :27422 `library.notes_import`, :27802 `library.ingest` | retired — `get_cli_setting("<section>", "last_directory", None)` 3-arg form; value validated by `browse_start_directory` |
| dotted_section_setting :28466 `library.ingest_directory_scan_limit` | retired — 2-arg dotted/default form, evaluated on the preflight worker THREAD |
| dup_shape `handle_library_media_rail_return` @7177 (114-copy cluster) | retired — `event.stop()` + one call; shape-only |
| dup_shape `finish_create_projection` @30677 (25-copy cluster) | retired — 2-line closure; shape-only |
| dup_shape `_explore_library_rail`/`_explore_library_landing` @21661/21667 (20-copy cluster) | retired — 2-line async handlers both calling `_explore_library_tools`; shape-only |
| dup_shape `_collapse_library_rail`/`_expand_library_rail` @21631/21637 (17-copy cluster) | retired — 2-line handlers; shape-only |
| dup_shape `_advance_library_stage_interaction` @5590 (10-copy generation bump) | retired — the `n += 1; return n` idiom; not a helper candidate |
| dup_shape / dup_verbatim `_notify_library_ingest_warning` @28510, `_notify_library_media_edit_warning` @32251, `_notify_library_media_delete_warning` @32459, `_notify_library_media_analysis_warning` @33311 (+ 4 controller copies) | confirmed — four verbatim 3-line `getattr(app_instance,"notify")` + `severity="warning"` helpers in this file (8 across 3 files); `Utils/NotificationHelper.show_notification` exists with 1 importer (`Notifications/notification_dispatch_service.py`) but is a different contract (it takes an app/severity mapping, not the `app_instance` duck), and the recipe's byte-for-byte canon moved the controller copies deliberately. **Disposition: P3 documented duplication** — one `_notify_library_warning(message)` on `BaseAppScreen` would replace all 8 (Size S); not raised to a finding because the canon (§1 "byte-for-byte") forbids editing moved bodies mid-series |
| dup_shape `refresh_library_details_sizes_for_diagnostics` @22476 (3-copy) | retired — `event.stop(); run_worker(...)` shape-only |
| dup_shape `_persist_library_note_import_location` @27426 (3-copy) | confirmed — see P3 [D4b] (g); actually 4 copies (`:27806` too) |
| dup_shape `_handle_workspace_create_result` @35388 (3-copy) | confirmed — see P3 [D4b] (h), bodies verbatim |
| dup_verbatim `handle_library_media_empty_import` @22704 / `open_workspace_import_sources` @35346 / `open_import_export_from_library_rag` (controller) | retired — 2-line `event.stop(); await self._select_library_rail_row(LIBRARY_ROW_INGEST_MEDIA)`; each is an `@on` target for a distinct button id, which is the contract; collapsing them needs a multi-selector `@on`, a cosmetic change |
| except_exception_pass :687 | retired — wraps `setattr(app, cache_attr, …)`, a cache write |
| except_exception_pass :19505 | retired — `self.app.call_from_thread(...)` teardown guard on a worker THREAD (NoApp is an `Exception`), documented |
| except_exception_pass :19944 | retired — same guard in the export progress callback |
| except_exception_pass :21551 | **confirmed** — P3 [D1] (silent search-history persistence failure) |
| except_exception_pass :21607 | **confirmed** — P3 [D1] (silent rail-preference persistence failure) |
| except_exception_pass :27736 | retired — `call_from_thread` teardown guard; the write above it logs `logger.error` |
| except_exception_pass :28307 | retired — `worker.cancel()` on an already-finished worker |
| except_exception_return_per_file (8) | examined — :6447 (rename patch read, DB closed = no patch), :12007 (bounded DOM repair, logged), :13160 (decorative count, logged debug), :19188 (progress read, logged), :30946 (liveness fail-open, documented), :33296 (decoration read, documented), :34847 (console-lock pre-check, logged debug), :31232 (banner, logged). None on a persistence path → retired |
| fetchall_no_limit :30945 `_review_set_live_ids` | retired as UNBOUNDED (bounded by `IN(…)` of ≤900 ids per batch, `_REVIEW_SET_LIVENESS_BATCH`) — but the CALL SITE is the loop-blocking one → folded into P1 [D2] |
| function_body_import_per_file (62) | confirmed count 64 statements / 37 modules; 0 dead; 3 redundant → P3 [D3]; rest cycle/payload-breaking by design |
| get_cli_setting_hot :630 / :635 (loops) | retired — the loop lives in `_read_library_ingest_options_from_config`, memoised on the App by `_library_ingest_options_for` keyed on `current_config_identity()`; uncached only for an app-less screen |
| legacy_markers_per_file (29) | unverified — not examined individually (task-id/"legacy" comment markers; none read as dead code during the full read; one stray `# STALE` at :925) |
| raw_1024x1024 :3801 :3802 :28435 | retired — `16 * 1024 * 1024` etc.; no shared MiB constant exists in `Constants.py`/`Utils/` (grep: every module spells its own) — repo-wide style, not a slice finding |
| run_worker_coroutine_per_file (54) | confirmed count 59 `run_worker(` sites + 8 `@work`; every coroutine that touches DB/file routes through `_run_library_service_call` (`asyncio.to_thread`) or an `async` service method that offloads (`notes_scope_service._run_folder_repository` → `to_thread`, verified :487) or `asyncio.to_thread` directly (:6851, :6959, :21422, :24952, :26283, :26582, :33400); 0 coroutine workers block on sync sqlite. The loop-blocking sqlite is in SYNC handlers/actions (P1, P3) |
| seed_name `_safe_text` :12965 | confirmed present; a staticmethod re-bound onto two controllers at :35595/:35605 (P3 [D3] monkeypatch row); 4 accessor lambdas hand it to other controllers (:2689 :2860 :2990 :3169) |
| try_import_guard :31185 (`_active_review_set_banner`) | retired — the `try` guards a storage read, not an import; the inner `from …review_set_state import` is the documented lazy import |
| try_import_guard :31567 (`_review_set_picker_worker`) | retired — same; the `except Exception` logs with traceback + toast (task-31220) |

Items from the task brief, dispositions:
| item | result |
|---|---|
| (1) 54 `run_worker(<coroutine>)` with sync sqlite/file I/O inside | none found in coroutine workers (see row above); sync sqlite is in press handlers/actions/footer → P1 + P3 [D2] |
| (2) `get_cli_setting` loops :630/:635 + rail/search write-back | loops memoised; write-back persists correctly (mirror in-memory then thread write, nested-section-aware saver) — verified-fine |
| (3) `_notify_*_warning` ×8 / shared helper | confirmed duplication; `NotificationHelper.show_notification` (1 importer) is not the same contract → documented P3, canon forbids mid-series edits |
| (4) `except Exception: pass` ×7 | 2 confirmed (P3 [D1]), 5 retired |
| (5) 62 function-body imports | 64 verified resolvable, 0 dead, 3 redundant |
| (6) `plain_readback` | not in this excerpt; file has 0 `.plain` attribute reads; `getattr(renderable, "plain", …)` at :12114/:12485 is an equality gate before `Static.update`, never a re-render of user text → retired |
| (7) `handle_library_media_empty_import` ×3 | retired (2-line `@on` targets) |
| (8) timer callbacks with unguarded `query_one` | all 5 `set_timer` targets checked (:9430 :10477 :10526 :10970 :20587) — every `query_one` inside is wrapped in `except (NoMatches, QueryError)` or the callback touches no DOM → verified-fine |
| (9) `recompose=True` reactives | 0 `reactive(` declarations in the file; whole-screen recomposes are explicit `self.refresh(recompose=True)` ×28, governed by `Tests/UI/test_library_recompose_ratchet.py` (63 ceiling) → P3 [D2] on the one status-flip site |
| (10) mutable class attributes | `CSS_PATH`/`BINDINGS` lists (Textual contract, never mutated), every `LIBRARY_*_SHORTCUTS` / `_*_WORKBENCH_FOCUS_TARGETS` a tuple, `_LIBRARY_*` frozensets/str — none mutated per instance → verified-fine |
| (11) `id()`-keyed caches across recompose | `:6138`/`:6157-6159` `id(owner/shell/items_host)` on `_LibraryMediaReturnSettlement` are AND-ed with `compose_generation`, `lifecycle_generation`, `presentation_epoch`, `receipt is`, `focus_anchor is`, `current_owner is` in `library_media_controller.py:1383` → a recycled id cannot false-match; `:21069` `id(app_config)` is an admission fence; `_library_layout_ref_cache` is selector-keyed and validated by `_library_ref_is_live`; viewer/match memos are `is`-identity on immutable detail dicts → verified-fine |

## Verified-fine
- `_run_library_service_call` (:12930) offloads every sync service callable via `asyncio.to_thread(run_finite_local_worker, …)`; coroutine callables await on the loop. `notes_scope_service` folder/placement/search methods (33 of them) go through `_run_folder_repository` = `asyncio.to_thread` (verified :487, and for `page_note_placements`, `locate_note_tree_placement`, `load_note_tree_mutation_context`, `attach_note_to_folder`, `detach_note_from_folder`, `create_note_folder`).
- Reader-preference persistence (:6844-:7069): `asyncio.to_thread(read_cli_config_serialized)` / `asyncio.to_thread(save_setting_to_cli_config, …)`, per-authority `asyncio.Lock`, generation-fenced, rollback + toast on failure.
- Local source snapshot (:13167-13471): all list calls off-loop, `asyncio.wait_for(LIBRARY_SOURCE_SNAPSHOT_TIMEOUT_SECONDS)`, collections count concurrent with its own deadline, failure text through `_retry_failure_reason` (no URL/path leak).
- Negative-predicate thread offload at :13155 and :19425/:19734 (`is_memory_db → inline`): the unknown/default shape is THREADED and only the thread-local `:memory:` case runs inline — the safe direction, documented against `LibraryLocalRagSearchService._search_conversations`.
- All 5 `set_timer` callbacks guarded (item 8); `on_screen_suspend` stops every retained timer (:9187-9217).
- 59 `run_worker` + 8 `@work` sites: every `exclusive=True` carries a `group=` except none (the only inconsistency is the non-exclusive pair in P3 [D3]).
- Path boundaries: every user-chosen path goes through `Utils/path_validation.validate_path_simple` (:20987, :26220, :27133, :27789, :28430, :28552); URLs through `validate_url` (:28545); `_safe_text`/`_sanitize_media_field` built on `Utils/input_validation` (:12965-13003).
- Media return-settlement id() fences (item 11) — see dispositions.
- `_review_set_live_ids` `IN(…)` batched at 900 < SQLite's 999 default.
- Boundary hygiene on failures: exception TYPE only in logs across the Notes/Media/Prompts mutation paths (`type(exc).__name__`), copy never carries a path; one exception at :19822 (`f"Export failed: {exc}"`, user-facing status may carry the chosen destination path — not a secret).
- The two `Static.renderable` `"plain"` reads (:12114, :12485) are equality gates, not re-renders.
- `truncate`/`elide` helper check: `Utils/Utils.py:264 elide_path_middle` has 4 importers and is not re-rolled here.

## Retired
- `fetchall_no_limit :30945` as an unbounded read — bounded by the 900-id batch (kept as the P1's call site).
- `get_cli_setting_hot :630/:635` — memoised on the App keyed on `current_config_identity()` (:658-689); the docstring at :29203-29210 records the 43-reads-per-visit incident this memo closed.
- `except Exception: pass` :687 :19505 :19944 :27736 :28307 — cache write / teardown guards / cancel guard, not data paths.
- `raw_1024x1024` — no shared constant exists to adopt.
- `try_import_guard` :31185/:31567 — storage guards around documented lazy imports.
- `plain_readback` — no un-escaping read-back in this file.
- `dotted_section_setting` ×12 — all resolve (TASK-1771); write-back verified.
- dup_shape rows @7177 @30677 @21661/21667 @21631/21637 @5590 @22476 and the `handle_library_media_empty_import` verbatim trio — shape-only matches of 2-line handlers.
- "coroutine workers block on sqlite" (task-brief hypothesis) — no instance; the loop-blocking reads are in sync actions/footer code (P1).
- Notes tree paging/mutation `await service.<method>(...)` (:17321 :17728 :17952 :18541 :18553 :18601 :24020) — retired as loop-blocking after confirming the service offloads.

## Left UNVERIFIED
| claim | why not verified | literal command to run |
|---|---|---|
| The whole-surface recompose ratchet (`LIBRARY_WHOLE_SCREEN_RECOMPOSE_MAX = 63`) is green/red at this commit | every test in the file ERRORs at fixture setup under the isolated env (`_disable_model_catalog_refresh`, pytest-asyncio strict-mode async fixture from `Tests/UI/conftest.py`) — environment, not the ratchet | `cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && $PY -m pytest Tests/UI/test_library_recompose_ratchet.py -q -p no:cacheprovider -p asyncio --asyncio-mode=auto` (or run it in the main checkout's normal pytest config) |
| Live cost of a `]` press in the Reader with a large review set (the P1 measured the DB reads in isolation, not the end-to-end keypress) | do not run the app (brief rule) | main checkout: `tmux -L verify new-session -d -x 235 -y 52 '.venv/bin/python -m tldw_chatbook.app'; sleep 12; tmux -L verify send-keys C-p; tmux -L verify send-keys -l 'Library'; tmux -L verify send-keys Down Enter;` open Browse Media → a row → "Review these" (needs ≥100 items) → `tmux -L verify send-keys ']'` repeatedly while `TEXTUAL_LOG`/`--durations` captures frame time; `tmux -L verify send-keys C-q; tmux -L verify kill-server` |
| The red size ratchet is also red on the CI runs for the PRs that landed since 2026-09-09 (vs. merged past a cancelled check) | needs GitHub CI history, outside the worktree | `gh run list --workflow=test.yml --branch dev --limit 20` then `gh run view <id> --log-failed \| grep test_screen_size_ratchet` |
| `legacy_markers_per_file` (29 rows) | not examined individually | `grep -n "legacy\|LEGACY\|STALE" tldw_chatbook/UI/Screens/library_screen.py` and read each |
| The `_notify_*_warning` consolidation is acceptable under the byte-for-byte canon | canon says moved bodies are not edited mid-series; whether a shared helper may be introduced at the SCREEN layer first needs the program owner | `grep -n "byte-for-byte" backlog/docs/library-decomposition-recipe.md` (§1 "The byte-for-byte canon") |
