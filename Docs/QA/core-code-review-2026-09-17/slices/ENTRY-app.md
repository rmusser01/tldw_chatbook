# ENTRY-app — tldw_chatbook/app.py, 21050 lines

Worktree `/Users/macbook-dev/Documents/GitHub/tldw-review` @ d8fb4053f9. All commands below were run from the worktree root; Python commands with `source <SCRATCH>/env.sh` and `PYTHONPATH=<worktree>` (a first pass without `PYTHONPATH` silently resolved `tldw_chatbook` to the main checkout because the script's own directory became `sys.path[0]` — those numbers were discarded and re-measured; the discrepancy is itself recorded under Retired).

## Coverage
| file | lines | read in full / sampled (which ranges) / mechanical only |
|---|---|---|
| tldw_chatbook/app.py | 21050 | **read in full**, 1–21050, in 1000-line chunks (`Read` offset/limit; 700-line `sed -n` for 1–700). Nothing sampled, nothing mechanical-only. |
| Library/library_ingest_jobs.py, DB/Library_Ingest_Jobs_DB.py, Notes/agent_lessons.py, Sync_Interop/notes_organization_sync_service.py, Notes/notes_organization_repository.py, Scheduling/scheduler/loop.py, Research_Workspace/source_association.py, Research_Workspace/local_adapter.py, Persona_Buddy/controller.py, Sync_Interop/sync_state_repository.py, UI/Console_Modules/archive.py, Event_Handlers/note_ingest_events.py, config.py | — | **evidence-only spot reads** (the specific functions app.py calls), not reviewed |

### Responsibility clusters of `app.py` (the god-module map; line ranges from the symbol map `grep -nE '^class |^    def |^def '`)
| lines | cluster |
|---|---|
| 1–1170 | ~1,000 lines of imports (≈170 `from … import` blocks), spawn-child stderr guard, boot constants (pre-import pacing tables 950–1010) |
| 1171–2222 | 10 command-palette `Provider` classes (Theme, TabNavigation, LLMProvider, QuickActions, Settings, Character, Media, LibraryIngest, SetupWizard, Developer) |
| 2231–2583 | ingest helper functions (error sanitizers, cookies-file validation, done-progress dict, real-stderr fd, keyword sniffing) |
| 2585–7089 | `LibraryIngestQueueMixin` — **4,500 lines**: submission + Research-source linkage (2656–3892), parse-pool sizing/lifecycle (3895–4172), job→options (4173–4656), local-STT dispatch (4658–5290), top-up/backpressure (5291–5637), pool retirement/breakage (5664–6262), remote poller (6264–6786), writer thread (6789–7089) |
| 7092–7538 | Notes-sync deferred facades, collections-capture reference resolvers |
| 7540–8417 | `TldwCli` header (CSS/BINDINGS/COMMANDS/class attrs) + **600-line `__init__`** (7814–8417) |
| 8418–8950 | lazy-owner properties (terminal manager, notes-sync runtime, RAG admin trio, persona buddy, skills stack, credential store) |
| 8952–9519 | navigation/handoff entry points, Home control actions |
| 9520–11464 | `_wire_*` service composition (persona/actor packs, tool packs, workspace, chat, writing, collections, research, prompts/chatbooks, evals, study; **800-line `_wire_watchlists_and_notifications_services`** 10667–11464) |
| 11466–11720 | scheduler handler glue, FTS backfill workers, parity-state repos |
| 11721–11880 | runtime-policy / backend switch |
| 11882–12195 | parallel `_init_*` initializers, logging setup |
| 12196–12360 | `compose`, UI-responsiveness monitor |
| 12361–13016 | persona-buddy overlay, screen-owned CSS, reusable-screen cache, Roleplay→Console activation |
| 13017–13366 | Personal Context interview/link |
| 13367–13514 | research-workspace screen factory, startup route |
| 13515–14480 | focus mode + **1,000-line screen-navigation engine** (lock, dispatch, overlay dismissal, complete) |
| 14481–15405 | TTS/STTS event handlers, profile-repository / voice-bundle lifecycle |
| 15406–15502 | watchlists command service, llamacpp snapshots |
| 15503–17060 | `on_mount`, model-catalog refresh/consent, first-run wizard/recovery/skills offer, initial screen push, `_post_mount_setup` |
| 17061–18072 | deferred startup work, staggered boot fleet, citation reconcile/migration, footer timers, screen pre-importer, TTS init scheduling |
| 18073–18484 | backup/recovery service, speech delivery/initialization admission gates |
| 18485–19590 | shutdown/unmount lifecycle (**1,100 lines**), `_handle_exception` |
| 19607–20250 | splash-closed, worker-state hook, media cleanup, help/actions, quit flow |
| 20256–21050 | module entry: early logging, CSS staleness, arg parser, `__main__`, `get_app`, `main_cli_runner` |

Totals: 16 classes, 513 methods/functions; `import tldw_chatbook.app` = 0.87 s / 669 package modules resident (`python -c "import tldw_chatbook.app"` timed in the isolated env).

## Findings

### P1 [D2] — A folder import of N files freezes the UI thread for ~3.3 ms × N: 0.33 s at 100 files, 4.4 s at the 1000-file scan-limit maximum (measured on the real registry + store)
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

### P2 [D1] — A failed home-server profile link is swallowed with no log line anywhere: `_run_personal_context_link` catches `Exception`, toasts, returns
- Where: `tldw_chatbook/app.py:13335-13340`
- Evidence: read — the `except Exception:` body is `self.notify("Profile linking needs attention. …", severity="error"); return` with no `logger` call; `grep -cE 'logger\.(warning|error|exception|opt)' tldw_chatbook/Personal_Context/link_service.py` → `0`, so the coordinator does not log for it either. The try covers keyring access (13211-13212, 13221), `sync_state_repository.get_personal_context_link_state` (13213), `bootstrap_personal_context_service` (13226), `coordinator.plan/apply/resume` (13257-13333) and two `push_screen_wait` dialogs.
- Why it matters: this is the only path that links Personal Context to the home server; when it fails the user sees "needs attention; retry from Settings" and the profile log records nothing — the exact undiagnosable-from-the-log failure TASK-32533 fixed for widget crashes.
- Recommended correction: `logger.opt(exception=True).warning("Personal Context link failed (stage=…)")` before the notify (type + stage only, no plan content — the toast already commits to "No profile content was shown"). S.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/UI/test_personal_context_link_app_flow.py` lines 174/242/315 assert the toast text as a requirement; none asserts the absence of a log line, so a log is compatible.
- Already covered: none

### P2 [D3] — `app.py` is a 21,050-line god module with no size governance: 25 responsibility clusters (table above), a 4,500-line ingest mixin, a 600-line `__init__`, an 800-line wiring method, a 1,100-line unmount
- Where: whole file; cluster table above gives ranges.
- Evidence: `wc -l` → 21050; `grep -cE '^    def |^    async def '` → 513; `grep -nE '^class '` → 16 classes; `grep -n 'app\.py' Tests/Architecture/test_screen_size_ratchet.py` → no row; `backlog/docs/library-decomposition-recipe.md` §17 (line 3309) governs `UI/Library_Modules/*_controller.py` only; `grep -inE 'app\.py|god|monolith|decompos' <SCRATCH>/adr_list.txt` → only `097-boot-budget-ratchets.md` (boot census, not structure).
- Why it matters: every edit anywhere in the app's lifecycle, ingest, TTS, navigation or wiring lands in one 600 KB file that whole-file readers cannot open (the console-interaction lesson, 608 KB controller), and nothing stops it growing.
- Recommended correction: follow `backlog/docs/library-decomposition-recipe.md` — §1 per-subsystem PR series in this order: (1) `LibraryIngestQueueMixin` 2585–7089 → `Library/ingest_queue_mixin.py` (it is already a mixin; keep `from tldw_chatbook.app import LibraryIngestQueueMixin` as a re-export — 5 tests import it by that path); (2) the 10 palette providers 1171–2222 → `UI/command_providers.py` (2 tests import `ThemeProvider`/`TabNavigationProvider` from `app`); (3) the `_wire_*` composition 9520–11464 → an `app_wiring.py` function set taking `app`; §2 field-ownership script before each move; §17 add an `app.py` budget row pinned at the post-move count.
- Size: L · ADR: new · Confidence: verified (counts)
- Pinning test: none for size; `Tests/test_call_from_thread_guard.py` and `Tests/Packaging/test_chat_persistence_import_closure.py` pin import-closure/marshalling properties any move must keep.
- Already covered: none

### P3 [D1] — Mutable class attributes on `TldwCli`: one is mutated in place downstream (shadowed today), one guard is dead because of them, one is unused
- Where: `tldw_chatbook/app.py:7787` `_media_types_for_ui: List[str] = []`, `:7790` `media_types_for_ui: List[str] = []`, `:7793` `parsed_notes_for_preview: List[Dict[str, Any]] = []`; dead guard `:8184-8185`; in-place mutators `tldw_chatbook/Event_Handlers/note_ingest_events.py:233` (`.extend`) and `:283` (`.clear`).
- Evidence: `grep -rnE '(_media_types_for_ui|media_types_for_ui|parsed_notes_for_preview)\s*(\.append|\.extend|\.clear|…)'` → only the two `parsed_notes_for_preview` sites; `__init__` assigns `self.parsed_notes_for_preview = []` at 8059 and `_init_media_db` assigns `self._media_types_for_ui` on every branch (11982/11987/11991), so in production the class lists are never touched. `media_types_for_ui` has 0 references outside its declaration (package + Tests). `hasattr(self, "_media_types_for_ui")` at 8184 is always True because of the class attr → the `["Error: Media DB not loaded"]` fallback there is unreachable. Latent: 5 tests build the app with `object.__new__(TldwCli)` (`grep -rn '__new__(TldwCli)' Tests/`), where the class list IS the instance's.
- Why it matters: not a live shared-state bug (instance shadowing), but the class-level list is one deleted `__init__` line away from becoming one, and two of the three declarations are dead weight.
- Recommended correction: annotation-only declarations (`parsed_notes_for_preview: List[Dict[str, Any]]`), delete `media_types_for_ui` and the 8184 guard. S.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P3 [D2] — Bounded synchronous SQLite on the event loop from timer callbacks and `run_worker` coroutines (four sites, each one-shot or ≤50 rows)
- Where: (a) `app.py:17177-17180` timer → `_deferred_wire_notes_sync_services` → `_wire_notes_sync_services` 7300 `initialize_agent_lessons_folder` (write transaction, `Notes/agent_lessons.py:266`) and 7353-7370 one SELECT per sync profile; (b) `app.py:2944-2946` `_reconcile_research_source_held_jobs` → `release_dispatch_hold(require_persisted=True)` → `_persist_required` → `upsert_job` (sync, ≤ `limit=50`); (c) `app.py:13158`/`13213` `get_personal_context_link_state` (`Sync_Interop/sync_state_repository.py:2259 with self.transaction()`) and `13239 get_personal_context_service()` (DB open + keyring) inside the `run_worker` coroutine `_run_personal_context_link`; (d) `app.py:3665` `operation_store.get` — **documented deliberate** (3662-3664, ordering fence).
- Evidence: read; run_worker-coroutine census below (Candidate dispositions, `run_worker_coroutine_per_file`).
- Why it matters: each is a few ms at most once per boot / per click, so no user-visible stall today; listed because the file's own comments (17042-17046, 18622-18626) treat "sync on the loop" as a defect class.
- Recommended correction: `asyncio.to_thread` for (a)-(c); leave (d). S.
- Size: S · ADR: no · Confidence: verified (read; not timed)
- Pinning test: `Tests/Sync_Interop/test_notes_organization_app_wiring.py` (wiring semantics, not thread)
- Already covered: none

### P3 [D3] — loguru and stdlib `logging` emit side by side: 38 `logging.<level>(…)` message sites vs 366 loguru sites in the same file
- Where: `grep -nE '^\s*logging\.(info|debug|warning|error|critical|exception)\('` → 38 sites, e.g. 1130, 8162, 8258-8287, 12164-12237 (`compose`), 13463, 18520, 19216, 19432-19448, 20275, 20594-20674, 20930-20976.
- Evidence: counts above. `logging.getLogger().addHandler(...)` at 12108/12180 and the handler sweep at 19440-19448 are legitimate handler wiring and excluded from the count.
- Why it matters: two emit pipelines with different formatting/redaction paths; `Logging_Config.py` is the one sanctioned bridge.
- Recommended correction: mechanical swap of the 38 emits to `logger` (loguru). S.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P3 [D3] — Dead code cluster: a handler whose only poster is the DEPRECATED Tools_Settings_Window, a window-hiding no-op, a placeholder widget, two never-called log helpers, an unused provider map plus its 19 module-scope imports, and six write-only attributes
- Where / evidence (package refs exclude app.py and Tests; Tests refs counted from Tests content, which is stronger than `--collect-only` names for a symbol grep):
  - `handle_ingest_ui_style_changed` 14827-14842: `IngestUiStyleChanged(` is posted only from `UI/Tools_Settings_Window.py:3656` (DEPRECATED TASK-1346, nav-unreachable per CLAUDE.md) and its target `#ingest-window` is composed nowhere (`grep -rn 'ingest-window'` → CSS files + Constants.py only) → always the `QueryError` branch.
  - `hide_inactive_windows` 16725-16738 (called once, 16849): `.window`/`.placeholder-window` exist only in `.tcss`/Constants.py → `self.query(...)` is empty → no-op.
  - `ALL_MAIN_WINDOW_IDS` 7733 (0/0), `TabDropdown` 2224 (0 pkg; 1 Tests mention is a docstring), `_display_buffered_logs` 12125 (0/0; its own docstring says "currently has no callers"), `_log_view_dimensions` 19591 (0/0).
  - `API_FUNCTION_MAP`/`ALL_API_MODELS` 1109-1140 (0/0) and `API_IMPORTS_SUCCESSFUL` (always `True`; else-branch dead; 1 Tests fixture sets it on a mock, nothing reads it) — the 19 `chat_with_*` imports at ~608-630 feed only this dict; they cost nothing today because `LLM_Calls.LLM_API_Calls(_Local)` is resident via `Chat/Chat_Functions.py` anyway (`resident after app import: True`), so this is dead names, not boot cost.
  - Write-only attrs: `_default_rag_expansion_provider` 7761 (0/0), `note_import_success_handler`/`note_import_failure_handler` 7796-7797 (0/0), `_prompt_search_timer` 7799/8180 (0 pkg; the 2 Tests hits are the different name `_chat_sidebar_prompt_search_timer`), `_initialized_tabs` 8827 (0/0), `media_types_for_ui` 7790 (0/0).
  - `_save_shutdown_caches_with_timeout` 20222 is a `logger.debug` no-op that `_run_blocking_quit_persistence` 20231-20241 still spawns a thread and joins for.
- Why it matters: ~150 lines and one always-failing handler that a reader has to disprove before trusting; the retired-window ids mislead ("ingest-window" reads as live).
- Recommended correction: delete; keep `AVAILABLE_PROVIDERS` (read internally at 1521). S.
- Size: S · ADR: no · Confidence: verified (grep counts as listed; each name also grepped in string form, which catches `getattr(app, "name")`)
- Pinning test: none (`Tests/fixtures/event_handler_mocks.py:45` sets `API_IMPORTS_SUCCESSFUL` on a mock — remove with it)
- Already covered: none

### P3 [D3] — 208 function-body imports; 20 are redundant re-imports of names already imported at module scope, and 9 lazy-import a module that is resident before `on_mount` anyway
- Where: redundant (AST, `<SCRATCH>/fn_imports_app.py` against the worktree): 1270 `save_setting_to_cli_config`, 1395 `get_shell_destination` (classmethod called ~2× per tab per palette keystroke), 6346 `event_principal_id_from_active_context`, 12046 `logging`, 13441 `SHELL_DESTINATION_ORDER`, 16316 `get_cli_setting`/`get_user_data_dir`, 19205 `asyncio`, 19452-19453 `threading`/`subprocess`, 19477 `textual.worker.work`, 20259 `configure_application_logging`, 20434 `hashlib`, 20839/20889 `logging`, 20854-20855 `Path`/`sys`, 20867/20961 `subprocess`, 20890 `os`. Residency: `tldw_chatbook.Backup_Recovery.activation` is imported function-locally at 11540, 15784, 15902, 15974, 17047, 17245, 18364, 19744, 19815, 19838 yet `resident after app import: True` (via `Backup_Recovery.storage_admission` at line 2), so the deferral buys nothing against the ADR-097 census.
- Evidence: script output `UNRESOLVED modules: []` (every one of the 208 targets resolves in the worktree — no imports of retired modules) and the REDUNDANT list above; residency probe output quoted above (`TTS.profile_source` is NOT resident → those four deferrals are real and stay).
- Why it matters: noise, and a reader takes the function-local `activation` imports as a boot-budget rule that is not in force for that module.
- Recommended correction: hoist the 20 redundant ones; hoist `Backup_Recovery.activation` to module scope (re-run `Tests/Performance/test_ui_ready_module_census.py` to confirm the count is unchanged). S.
- Size: S · ADR: yes (`097-boot-budget-ratchets.md` governs the genuine deferrals, which are verified-fine) · Confidence: verified
- Pinning test: `Tests/Performance/test_ui_ready_module_census.py` (module count ratchet — must stay green)
- Already covered: none

### P3 [D3] — Four copies of the "find the ChatMessage/ChatMessageEnhanced widget by `message_id_internal`" loop in the TTS handlers
- Where: `app.py:14666-14681`, `14722-14743`, `14782-14793`, `14811-14823`
- Evidence: `grep -n 'list(self.query(ChatMessage)) + list(' app.py` → 4 sites; the three in `_deliver_tts_complete_event` re-query the DOM (`self.query` × 2 each) up to three times for one event.
- Why it matters: one query helper would halve the DOM walks on every TTS completion and stop the four copies drifting (they already differ in whether `query_one(".message-text")` is guarded: 14677 unguarded, 14737 guarded).
- Recommended correction: `_find_message_widget(message_id) -> Widget | None` used by all four. S.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P3 [D4b] — 29 verbatim copies of `try: X.from_config(self.app_config, policy_enforcer=…) except ValueError: X(client=None, policy_enforcer=…)`; no helper exists
- Where: `_wire_watchlists_and_notifications_services` 10717-11450 (most), `_wire_evaluation_services` 10503, `_wire_study_services` 10549-10568, `_wire_research_services` 10601-10631, `_build_rag_admin_services` 8592-8601.
- Evidence: `grep -c '\.from_config(' app.py` → 29; `grep -c 'client=None,'` → 30; `grep -rnE 'def [a-z_]*(from_config|offline|or_null)…' runtime_policy/ Utils/` → only `build_runtime_api_client_from_config` (different concern). No behavioural drift between copies (all identical shape; the `ServerNotificationsService` copy at 10727-10738 uses `from_server_context_provider` and is the one intentional variant).
- Why it matters: 29 sites to touch if the offline fallback ever needs a reason code or a log line.
- Recommended correction: `runtime_policy/service_factories.py::server_service_or_offline(cls, app_config, *, policy_enforcer)`; `_wire_watchlists_and_notifications_services` shrinks by ~250 lines. S.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/…` patches `tldw_chatbook.app.ServerNotesWorkspaceService.from_config` / `ServerCharacterPersonaService.from_config` (3 hits each in the app-import grep) — a helper must keep calling `cls.from_config` so those patches still land.
- Already covered: none

### P3 [D4a] — The `agent_lessons_seed_state` "is this seed still unknown?" SELECT is re-rolled inline in `app.py` while the service constructed two lines earlier owns the identical query
- Where: `app.py:7361-7366` vs `Sync_Interop/notes_organization_sync_service.py:799-806 NotesOrganizationSyncService._agent_lessons_seed_is_unknown` (same SQL text, same args; app's `seed is not None and seed["state"] != "unknown"` is its exact complement). Third relative: `Notes/notes_organization_repository.py:282` (different WHERE, different purpose — not a copy).
- Evidence: `grep -rn 'agent_lessons_seed_state' tldw_chatbook/ --include='*.py'` output above; helper importer count: 0 external (it is private).
- Why it matters: the only raw `execute` in app.py (the `lock_and_execute` row) exists to duplicate a one-line service method; a future column/CHECK change (the table has a monotonic-update trigger, `DB/recovery_core_schema.py:360`) must be made twice.
- Recommended correction: make `_agent_lessons_seed_is_unknown` public and call `organization_service.agent_lessons_seed_is_unknown(server_profile_id=…, dataset_id=…)` at 7361. S.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/Sync_Interop/test_notes_organization_app_wiring.py`, `Tests/Notes/test_agent_lessons_seed.py` (seed semantics; unaffected by the call-site change)
- Already covered: none

## Candidate dispositions
| candidate (file:line pattern) | confirmed / retired (why) / unverified (check) |
|---|---|
| dotted_section_setting app.py:3150 `library.ingest_directory_scan_limit` | retired — brief: dotted lookups resolve since TASK-1771 (config.py:8447); read as a plain cached read |
| dotted_section_setting app.py:3911 `library.ingest_parse_workers` | retired — same |
| dotted_section_setting app.py:3929 `library.ingest_heavy_lane_max_workers` | retired — same |
| dotted_section_setting app.py:4590 `transcription.transcribe_cpp.model_path` | retired — same |
| dotted_section_setting app.py:6292 `library.ingest` | retired — same |
| dup_shape `_start_deferred_audio_service_initialization@17693` (114 copies) | retired — the "shape" is two sequential method calls; nothing shared to extract |
| dup_shape `local_skill_trust_service@8864` (4 copies) | retired — lazy-property idiom (`if None: build; return`); not duplication of logic |
| dup_shape `_wire_llamacpp_snapshot_service@15462` (4 copies) | retired — two `= None` slot reservations |
| except_exception_pass app.py:54 | retired — spawn-child `loguru.remove()` guard (task-2016/2041 comments 21-49) |
| except_exception_pass app.py:12086, 12096 | retired — `logging.Handler.emit` must not raise; widget-unmounted guard |
| except_exception_pass app.py:13982, 13998, 14039 | retired — `self.notify` guard after a warning was already logged (13971/13988/14029) |
| except_exception_pass app.py:14741 | retired — cosmetic `remove_class` on a possibly-absent child; note the unguarded sibling at 14677 (folded into the TTS copy-paste P3) |
| except_exception_pass app.py:15027 | retired — done-callback `completed.exception()` retrieval |
| except_exception_pass app.py:15539, 15566, 19239, 19709 | retired — `persist_event` diagnostics guards; documented (15533-15536, 19231-19236, 19710-19712) |
| except_exception_pass app.py:17415 | retired — `timer.stop()` on teardown (noqa'd) |
| except_exception_pass app.py:19104, 19161 | retired — `_handle_exception` diagnostics/notify guards, documented 19105/19162 |
| except_exception_pass app.py:20045, 20068, 20108 | retired — `notify` guards on the quit path after a warning was logged |
| except_exception_return_per_file (15) | **confirmed ×1** at 13335-13340 (P2 above — no log); the others examined are logged (`9369`, `9407`, `9430`, `12668`, `13506`, `14124`, `16301`, `16336`, `19185`, `19195`) or return a typed outcome after `logger.opt(exception=True)` (`12817`, `12917`) — retired |
| function_body_import_per_file (167; AST counts 208 statements) | **confirmed** as the P3 redundant/residency finding; the remaining ~180 are ADR-097 deferrals and all resolve (`UNRESOLVED modules: []`) — verified-fine |
| get_cli_setting_hot app.py:12205/12208/12212/12215/12216/12217/12219 (compose) | retired — `App.compose` runs once per process (`grep -c 'def compose'` → 1, App singleton); reads are cache-backed (`config.py:1748 load_settings` `_SETTINGS_CACHE`, brief known-deliberate) |
| legacy_markers_per_file (50) | not examined as a finding — a comment-marker census with no code claim; every "legacy" region was read as part of the full pass |
| lock_and_execute app.py (locks=9, executes=1) | retired as a thread-safety finding — the one execute (7361) is a SELECT on `notes_db.get_connection()` (thread-local); no write outside `transaction()`; none of the 9 locks (814, 2695, 2707, 7219, 7907, 7934, 8071, 8348, 8360) guards SQL. The SELECT itself is the D4a P3 |
| loguru_and_logging app.py | **confirmed** — P3 above (38 vs 366) |
| mutable_class_attr app.py:7787 `_media_types_for_ui` | retired as a shared-state bug (only ever reassigned) → folded into the P3 (dead `hasattr` guard) |
| mutable_class_attr app.py:7790 `media_types_for_ui` | **confirmed dead** (0 references) → P3 |
| mutable_class_attr app.py:7793 `parsed_notes_for_preview` | **confirmed pattern, no live bug** — mutated in place at `note_ingest_events.py:233/283` but through the instance attr set at 8059 → P3 latent |
| run_worker_coroutine_per_file (13) | examined all: 2781 `_resume_settled_research_source_operation` (awaits scheduler; `to_thread` for staging) fine · 2829 `_reconcile_research_source_held_jobs` (`to_thread` reads; sync `release_dispatch_hold(require_persisted=True)` ≤50 rows) → P3(b) · 2835 `scheduler.resume_startup()` (`source_association.py:539 to_thread`) fine · 2841 `_sweep_research_paste_staging` (`to_thread`) fine · 3640 `_retry_research_source_catalog_job` (sync `operation_store.get`, documented 3662) → P3(d) · 9048 `request_conversation_resume` (`storage_call`, archive.py:48) fine · 9723 `controller.migrate_legacy_selection` (`to_thread`, controller.py:518/533) fine · 13134 `_run_personal_context_link` → P3(c) + P2 · 13694 navigation (screen construction is sync by design) fine · 14706 `_offer_tts_global_override` (dialog) fine · 15573 `_reconcile_research_quick_notes_startup` (`local_adapter.py:297 to_thread`) fine · 16355 `_apply_first_run_recovery_result` (`to_thread`) fine · 16665 nav+schedule fine · 17050 `scheduler_loop.run()` — coroutine by documented design (17042-17046), DB via `_offload` (loop.py:235/334); one sync `record_task_failure` on a rare preflight-failure path (loop.py:964) is outside this slice · 18132/19977 quit flows (`to_thread` for persistence) fine |
| try_import_guard app.py:50 (module) | retired — spawn-child guard, deliberate |
| try_import_guard app.py:1265, 2477, 9656, 9810, 10347, 11673, 11967, 13493, 15550, 15910, 15986, 16022, 16040, 16107, 16315, 16708, 17449, 19451, 19458, 19805 | retired — ADR `097-boot-budget-ratchets.md` lazy imports; all targets resolve (AST `find_spec` against the worktree); the `except` clauses guard the *work*, not the import |
| try_import_guard app.py:20601 (module, `__main__`) | retired — TASK-26040 migration guard, deliberate |

## Verified-fine
- **All `run_worker(exclusive=True)` calls carry `group=`**: AST scan over every `run_worker`/`work` call in app.py → `exclusive=True without group: []`.
- **All 9 `threading.Lock/RLock`s** are lazy-init double-checked or set/slot guards; no SQL under any of them. One note: `_persona_buddy_controller_lock` is held across a config-file write from a worker thread (9700-9713 `persist_persona_buddy_preferences`), so a UI-thread `persona_buddy_controller` property read can block for that write — bounded to one small TOML save.
- **`load_settings()` ×3 at boot** (module 1167, `__init__` 7901, `initialize_early_logging` 20264) — cache-backed (`config.py:1748`, `_SETTINGS_CACHE`); `get_cli_setting` inside the top-up `while` (5391) is ≤ worker_count iterations of a cached read.
- **Timer-callback `query_one`s are all guarded**: 12168 (`QueryError`), 13124 (`QueryError`), 14150 (`except Exception`), 17084 (`ScreenStackError, QueryError`), 17670 (`QueryError`).
- **`_dismiss_navigation_overlays` / navigation engine** (13593-14480): FIFO lock + worker dispatch documented and pinned (`test_overlapping_navigate_requests_complete_in_fifo_order` cited at 13612); read in full, nothing to add.
- **Ingest pool shutdown ordering** (6044-6262, 19255-19284): terminate/join off-loop, `_ingest_shutdown` flag first, callbacks marshal-guarded — matches its own deadlock analysis; read in full.
- **`_handle_exception` keep-alive** (19025-19167): both pump-filter clauses present as the comment requires.
- **`_media_types_for_ui` / `providers_models` `hasattr` guards**: 8154 is live (no class attr; `_init_providers_models` sets it on both branches anyway); 8184 is dead (class attr) — in the P3.
- **Splash-message list at 15612-15696** is built once inside `on_mount`'s splash branch; contains 3 duplicate strings — trivia, not reported.
- **`TldwCli._method(self, …)` call style** (8312, 10114, 10167, 11546, 14620, 15254, 16478, 18915…): deliberate bypass of instance overrides for lifecycle steps; consistent; not a finding.
- **Per-call `get_shell_destination` import at 1395** is a `sys.modules` hit (~µs) even though it runs ~24× per palette keystroke — listed only under the redundant-import P3.

## Retired
- **"N SQLite commits per folder submit are free"** — the first benchmark pass (script dir as `sys.path[0]` → main checkout's registry) showed 0.03 ms/job with the store attached; re-run with `PYTHONPATH=<worktree>` (`REGISTRY FROM: …/tldw-review/…/library_ingest_jobs.py`) showed **2.8 ms/job** — the worktree's `upsert_job` runs `with self.transaction()` per job. So this was NOT retired; it is half of the P1. Recorded here because the wrong first number would have retired it.
- **"27 function-body imports target modules that no longer exist"** — same `sys.path[0]` artefact (main checkout lacks `Backup_Recovery/activation.py` etc.); against the worktree `UNRESOLVED modules: []` and `ls` confirms every flagged file exists. Symptom real in the first run, cause was the harness, retired with evidence.
- **Mutable class attrs as a live shared-state bug** — retired (instance shadowing at 8059 / 11982-11991); kept as P3 latent.
- **`get_cli_setting` in `compose()` as a hot-path cost** — retired (once per process, cache-backed).
- **`hide_inactive_windows` as a bug** — retired as a bug (harmless no-op), kept in the dead-code P3.
- **`chat_wrapper` as dead** — 7 package refs + 10 Tests refs; live.
- **dotted-section rows ×5, dup_shape ×3, try_import_guard ×22, except-pass ×18** — retired individually above.

## Left UNVERIFIED
| claim | why not verified | literal command to run |
|---|---|---|
| The 4.4 s / 0.33 s folder-submit freeze is what a user sees in the running app (P1 end-to-end) | measured on the registry+store components with a listener of the same shape; the live app also runs `_dispatch_research_source_catalog_job`→`_top_up_ingest_parse_pool` per file (≈0.15 ms each in the benchmark's "top-up reads" column) and canvas repaints, so the real number is ≥ the benchmark; not run because the brief forbids running the app | `.claude/skills/verify/SKILL.md` recipe: `tmux -L verify new -d -s v 'TLDW_CONFIG_PATH=<scratch profile> python -m tldw_chatbook.app'`; navigate Library ▸ Import, submit a 100-file folder (`mkdir /tmp/f100 && for i in $(seq 100); do echo x > /tmp/f100/$i.txt; done`), then `tmux -L verify capture-pane -p` every 100 ms and count frames until the queue rows appear; repeat with 1000 files |
| Adding `logger.opt(exception=True)` in `_run_personal_context_link` leaks no plan content | the log-file sink runs with `diagnose=True` per comments at 15893/15954, so `.opt(exception=True)` may dump frame locals including `plan` | `grep -n 'diagnose' tldw_chatbook/Logging_Config.py` then, if `diagnose=True` on the file sink, use `logger.warning("… (exception_category={})", type(exc).__name__)` instead |
| The P3 `initialize_agent_lessons_folder` timer write measurably stalls the loop on a large ChaChaNotes DB | not timed | `cd <worktree> && source <SCRATCH>/env.sh && PYTHONPATH=$PWD .venv/bin/python -c "import time; from tldw_chatbook.config import get_chachanotes_db_lazy; from tldw_chatbook.Notes.agent_lessons import initialize_agent_lessons_folder; db=get_chachanotes_db_lazy(); t=time.perf_counter(); initialize_agent_lessons_folder(db, scope_mode='local_only', profile_id='local', dataset_id='local'); print(time.perf_counter()-t)"` |
