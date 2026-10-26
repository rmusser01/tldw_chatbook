# Verification ledger — architecture review 2026-10-25

Base: `origin/dev` @ `7a5e2513c5`. Each first-pass claim was re-checked by an **independent adversarial verifier**. There were four, one per slice, all read-only over a clean `git archive` export. Each was told to refute, not confirm. Runtime claims were reproduced in **sandboxed probes**: scratch `HOME`/`XDG_*`, `TLDW_CONFIG_PATH` unset, and `sandbox-exec` denying writes to the real profile. After every probe, `find ~/.config/tldw_cli ~/.local/share/tldw_cli -mmin -N` returned 0 files.

Verdicts: **CONFIRMED**, **CORRECTED** (true in substance; the number or location shown is the verified one), **REFUTED**. The report (`report.md`) uses only verified values.

## Self-correction: first-pass finding #1 REFUTED
- **Claim:** the agent note tools always fail in production because `config.chachanotes_db` is never set.
- **Refuted:** `config.get_chachanotes_db_lazy()` (`config.py:11873`) assigns the global through `global chachanotes_db`, and `app.py:2084` calls it during startup. `Tools/note_management_tools.py:69` imports the global at call time, so inside the running TUI it is populated.
- **Cause of the error:** the assignment grep listed `config.py:11886/11904`, which sit inside the lazy getter, but they were attributed to `initialize_all_databases`. The standalone probe had no app boot.
- **What stands:** in a fresh process where nothing has called the getter, `create_note` returns `{'error': 'Failed to create note: No unified database template for NotesInteropService.'}`, and `search_notes` fails the same way. That is temporal coupling through a lazily-populated global; it was folded into report F9.

## PR review corrections (2026-10-25)
A local review of the filing PR re-checked about 40 claims against the base. Its confirmed corrections:
- **F5 cause:** several import chains, not one. See the `agent_runtime` row below.
- **Lazy DB getters:** they self-initialize; only the note tools read the bare global.
- **Executable SCC:** 1,217 modules / 67 packages.
- **Executable upward edges:** 80 / 335 / 89.
- **Closers table:** the second one has 19 keys, 15 with a closer.
- **Getter lines:** `:11873` / `:11908` / `:11937`.
- **Widgets→`UI/Screens` targets:** `model_browser_state` ×3, `settings_speech_tts` ×2, video and image defaults, advanced config, web search and `provider_model_resolution`.
- **F11:** ADR-012's 2026-09-19 precedence amendment governs.

Each was re-verified in source before the fix.

## Runtime probes
| Probe | Result |
|---|---|
| Import `tldw_chatbook.Agents.agent_runtime` in a fresh interpreter | Loads `sqlite3`, `httpx`, `textual`, `tldw_chatbook.DB.ChaChaNotes_DB`, `tldw_chatbook.Chat`, `tldw_chatbook.config`; 207 tldw_chatbook modules. Contradicts `agent_runtime.py:400-404` ("pure module today"). |
| `SubscriptionsDB.transaction(immediate=True)` with `KeyboardInterrupt` raised in the body | Afterwards `conn.in_transaction = True`, depth `0`. The next `transaction(immediate=True)` raises `OperationalError: cannot start a transaction within a transaction`. The verifier read the source: with legacy implicit BEGIN (`:647`), the next *plain* `transaction()` commits the abandoned writes. |
| Bind `Chat/document_generator.py`'s call shape to every handler in its table | Every entry the probe matched (21 key/function pairs) fails (`missing a required argument: 'input_data'`); the table has 25 keys over 20 functions, and none of them accepts the call shape. Owned by TASK-32807.5. |
| `CreateNoteTool` / `SearchNotesTool` in a fresh process | Error dict as above (see self-correction). |

## Scripted measurements (reproducible with `scripts/`)
- `graph.py <export> graph.json` gives 2,820 modules and 30,674 internal edges, classed as top / lazy / TYPE_CHECKING.
- `analyze.py graph.json` gives:
  - import-time SCCs: 2, both in the app shell, sizes 3 and 2;
  - an SCC of 1,217 modules over 67 packages from executable (top-level plus function-level) edges, or 1,496 over 93 when TYPE_CHECKING-only edges are included;
  - a 91-package SCC at package level from top-level edges only;
  - 55 package pairs that import each other at top level, or 129 counting every import kind.
- `layers.py graph.json` gives the upward edges:
  - domain → presentation: 83, of which 42 top-level;
  - DB → feature: 358, of which 169 top-level;
  - kernel → higher: 99.
- `classes.py graph.json classes|funcs|app` gives:
  - 102 classes over 2,000 lines and 50 functions with complexity over 100;
  - 397 `TldwCli` attributes (an inline script over `app.py` plus `app_*.py`).
- `appparam.py graph.json` gives 352 non-UI functions in 71 modules taking the app, reading 233 attributes, 222 of them through `getattr` string lookups.
- `deleg.py <file>:<Class> …` gives delegator and largest-method counts per class.

## Ratchet coverage (F16)
Exact `_BUDGETS` keys were read with the AST from `test_module_size_ratchet.py`, `test_screen_size_ratchet.py` and `test_library_modules_size_ratchet.py`. These modules of 4,000+ lines have no row:

| # | Lines | Module (no row in any size ratchet) |
|---|---|---|
| 1 | 25,565 | `tldw_chatbook/DB/ChaChaNotes_DB.py` |
| 2 | 12,258 | `tldw_chatbook/config.py` |
| 3 | 12,093 | `tldw_chatbook/Chat/console_agent_bridge.py` |
| 4 | 11,400 | `tldw_chatbook/Notes/file_notes_git_service.py` |
| 5 | 9,629 | `tldw_chatbook/Agents/agent_service.py` |
| 6 | 9,532 | `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py` |
| 7 | 9,426 | `tldw_chatbook/DB/Client_Media_DB_v2.py` |
| 8 | 8,522 | `tldw_chatbook/UI/Console_Modules/workspace.py` |
| 9 | 8,184 | `tldw_chatbook/Chat/console_provider_gateway.py` |
| 10 | 7,000 | `tldw_chatbook/UI/Console_Modules/session.py` |
| 11 | 6,987 | `tldw_chatbook/DB/Subscriptions_DB.py` |
| 12 | 6,958 | `tldw_chatbook/Widgets/Console/console_composer_bar.py` |
| 13 | 6,911 | `tldw_chatbook/Media/local_media_reading_service.py` |
| 14 | 6,686 | `tldw_chatbook/Chat/console_runtime.py` |
| 15 | 6,674 | `tldw_chatbook/Chat/console_trace_service.py` |
| 16 | 6,108 | `tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py` |
| 17 | 6,007 | `tldw_chatbook/Notes/notes_sync_executor.py` |
| 18 | 5,917 | `tldw_chatbook/MCP/unified_control_plane_service.py` |
| 19 | 5,749 | `tldw_chatbook/Agents/local_tool_provider.py` |
| 20 | 5,706 | `tldw_chatbook/Tools/remote_worker_bundle.py` |
| 21 | 5,536 | `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py` |
| 22 | 5,186 | `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` |
| 23 | 5,167 | `tldw_chatbook/Notes/notes_sync_runtime.py` |
| 24 | 4,997 | `tldw_chatbook/Personal_Context/repository.py` |
| 25 | 4,968 | `tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py` |
| 26 | 4,853 | `tldw_chatbook/Model_Artifacts/service.py` |
| 27 | 4,775 | `tldw_chatbook/Character_Chat/Character_Chat_Lib.py` |
| 28 | 4,760 | `tldw_chatbook/DB/Prompts_DB.py` |
| 29 | 4,719 | `tldw_chatbook/Chat/citation_trace_repository.py` |
| 30 | 4,611 | `tldw_chatbook/Web_Scraping/WebSearch_APIs.py` |
| 31 | 4,507 | `tldw_chatbook/RAG_Search/simplified/rag_service.py` |
| 32 | 4,494 | `tldw_chatbook/Local_Ingestion/transcription_service.py` |
| 33 | 4,463 | `tldw_chatbook/Widgets/Library/library_notes_canvas.py` |
| 34 | 4,263 | `tldw_chatbook/Chat/chat_persistence_service.py` |
| 35 | 4,236 | `tldw_chatbook/UI/Screens/evals_screen.py` |
| 36 | 4,232 | `tldw_chatbook/Workspaces/registry_service.py` |
| 37 | 4,207 | `tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py` |
| 38 | 4,174 | `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py` |
| 39 | 4,061 | `tldw_chatbook/Dreams/track_service.py` |
| 40 | 4,030 | `tldw_chatbook/DB/private_sqlite.py` |

## Verifier C — Console / Agents / Chat (14 claims)
| Claim | Verdict | Verified value / evidence |
|---|---|---|
| Controller size and `__init__` | CONFIRMED | Class 3623–28971 (25,349 lines, 508 methods); `__init__` 3702–4996 (1,295 lines, 51 parameters, 195 attributes, 191 lambdas) |
| Host keyword counts | CORRECTED | `InterruptRoundHost` 122 keywords (121 lambdas: 86 `read_controller_*`, 33 `read_global_*`); `ConsoleCompactionPreflight` 67 (26 `read_controller_*`, 41 `read_global_*`, i.e. injected module globals). Mutation through getters at `console_interrupt_rounds.py:2971-2978` |
| Controller `self.app` | CORRECTED | `:4590`; assigned at `chat_screen.py:9256` and `console_launch_wake.py:223`; ≥18 app attributes; whole app forwarded to 5 helpers |
| 78-keyword reply call | CONFIRMED | `self._run_maintenance_agent_call(self._agent_bridge.run_reply, …)` at `:26835-27014`; `run_reply` takes 78 parameters, `AgentService.__init__` 59, `LoopDeps` has 61 fields |
| Store and bridge are framework-free | CONFIRMED | 19 / 16 constructor parameters; no Textual; no app |
| ConsoleRuntime private screen reads | CORRECTED | `screen_stack` walk at `:1558` (a ModalScreen check only); private `_console_hook_review_projection` reads at `:4981` and `:5296-5297` via `self.view.app` |
| `_run_one` | CONFIRMED | 5006–9168 (4,163 lines), 45 nested defs; `spawn` 551, `_launch_fleet_child` 490, `_resume_retained_child` 324 |
| `run_agent_loop` | CORRECTED | 1073–3523; `while True` 1729–3523; per-call loop 2637–3506 (870 lines) with 20 `==` and 3 `in` comparisons on `call.name`; dict dispatch at `:3103-3108` |
| `agent_runtime` purity | CORRECTED (by the PR review) | The `:22` chain is real (`console_project_instructions` imports httpx at `:12`; `Chat/__init__.py:6` → `chat_conversation_service:17` → ChaChaNotes_DB), but it is not the only cause. `:23` and `agent_models.py:24/:28` import Chat provider modules, and `project_instruction_runtime.py:40` imports `tool_catalog`, which reaches config, DB and UI. The static closure is 188 modules even with `Chat/__init__` empty. No purity test exists for `agent_runtime`; `agent_models`' test at `:409` scans source text only |
| Chat composition | CORRECTED | 250 files / 247,053 lines; 167 `console_*` files / 194,740 lines (78.8%); **42** files import DB; `console.md:120` claim confirmed false |
| Private cross-imports | CORRECTED | Agents→Chat private names: 2 (`agent_routing.py:208`, `:407`); Chat→Agents private: 5 names / 8 sites; `ChatProviderError` in 12 LLM_Calls files; `Chat_Deps.py` has no imports |
| `chat_models.py` dead | CONFIRMED | `textual.worker` at `:7`; only `Tests/UI/test_console_internals_decomposition.py:36` |
| Eight modules without ratchet rows | CONFIRMED | No row in any of the three ratchets (superseded by the 40-module table above) |
| `chat_rag_events` | CONFIRMED | 36 functions, 22 taking `app: "TldwCli"`; `pipeline_loader.py:23` imports it at top level |

## Verifier P — persistence / config / backup (15 claims)
| Claim | Verdict | Verified value / evidence |
|---|---|---|
| CharactersRAGDB clusters | CONFIRMED (counts rough) | 437 methods (233 public); learning 63; sync ~13–15 (not 22); notes ~26 public / 38; kept briefings/scripts 13; 4 of 5 repositories call `db.transaction()` |
| Semantic-revision cycle | CONFIRMED | 4× `_ensure_initial_semantic_revision` + 8× `_coordinate_semantic_mutation`; lazy imports `:14212` / `:14238`; `console_semantic_revision.py:33` top-level import |
| Sync hashing in delete | CONFIRMED | `soft_delete_message` `:16521`, transaction `:16556`, capture `:16581` → `:16907` → `canonical_payload_hash` `:16826` |
| Private `_local.transaction_depth` reads | CONFIRMED | 10 sites / 5 files |
| Learning call sites | CORRECTED | Only ~45 hit CharactersRAGDB (`local_study_service.py` 31, `local_quiz_service.py` 14); the other ~80 same-name calls are scope services, the server client or the UI |
| `transaction_observer` | CONFIRMED | 1 producer, 3 consumers |
| config.py spans | CORRECTED | Spans confirmed (template 4078–6661; `_load_settings_uncached` 2272–3943; encryption 10159–10854; canvas 165–446; capture/trace 9312–9735). DB globals are **also** assigned in the `get_*_db_lazy` getters (`:11873`, `:11908`, `:11937`); they create the DB when the global is None, so their callers need no prior set-up |
| `get_cli_setting` census | CONFIRMED | 356 calls / 104 modules / 30 packages; ~240 keys / ~46 sections; import-time reads incl. `RAG_Search/simplified/rag_service.py:96` and `app.py:693` |
| Participants registration | CORRECTED (worse) | 24 types (13 eager / 11 gated); **two disagreeing closers tables** (`:66-78` 11 entries vs `:125-145` 19 keys, 15 with a closer); `policy_id` chain `:531-553`; table rebuilt ≥3× per `get_connection`; **66** production importers |
| `unsaved_editors` | CONFIRMED | 36 `_exact()`; lazy imports reach 10 modules; ChatScreen privates at `:584-592`; `staging.py:75-82` imports 8 feature-private classes |
| Transaction managers | CONFIRMED | ~26 managers; 6 kernel subclasses; store-kernel ADR-228 (`228-sqlite-store-kernel.md`) Proposed |
| Subscriptions rollback | CORRECTED (worse) | Implicit BEGIN means a later plain `transaction()` commits the abandoned writes; Kanban and Scheduling are mitigated (connection closed in `finally`) |
| Duplicate exception classes | CONFIRMED | Prompts `:82-94`, Media `:90-102`, Evals `:131-143`, ChaChaNotes `:442-454` |
| Utils composition | CONFIRMED | 161 files, 96 Splash; `input_validation` 2,457 lines; all upward imports lazy |
| Decomposition-candidates doc | CORRECTED | The doc's sentence at `:19-20` already carries a 2026-09-21 correction; ChaChaNotes_DB is a listed candidate (`:117`) with no row; config.py is not on that list |

## Verifier R — providers / integration (11 claims)
| Claim | Verdict | Verified value / evidence |
|---|---|---|
| Adding commandcode | CORRECTED | 13 sites in 9 files; ~8 repeat ProviderRecord data; ADR-179 `:38-39` promise confirmed; TASK-33510/33511 exist (Done; hand-list drift) |
| Provider-key branches | CORRECTED | Strict: 170 in 69 files (settings_screen 17, LLM_Management 9, gateway 4); any left side: 320 in 98 files |
| Kwargs-builder drift | CORRECTED | The drift is real but `_chat_api_kwargs` has zero production callers (23 test calls); owned by TASK-32807.5 |
| Credential resolvers | CORRECTED | ≥6 real resolvers. **Precedence disagrees:** engine explicit→env→stored; moonshot/zai/qwencloud explicit→stored→env (moonshot re-read `:626-651`); custom endpoint env→stored. CLAUDE.md states the stored key outranks env |
| `chat_api_call` surface | CORRECTED | 39 parameters; 62 handler keys (33 engine); `PROVIDER_PARAM_MAP` literal 342–969 |
| `document_generator` | CONFIRMED | No handler accepts its call shape; TypeError re-raised as `ChatAPIError` (`:547-549`) |
| TLDWAPIClient size | CONFIRMED | 1209–16790; 1,220 methods (1,193 public); 47 `server_*_service` files; median 13 |
| Interop helper copies | CORRECTED | `_require_client` 44 defs (42 identical); `_enforce` 49 (44); `_maybe_await` 61; ADR-177 Proposed; owned by TASK-32898 |
| Research private transport | CONFIRMED (worse) | Probes for non-existent `run_paper_search` / `get_paper_search_detail` / `run_paper_search_ingest` (`:142/:161/:186`); the client has `paper_search*` (`:8709`), so the `client._request` fallback is always live (re-read `:136-150`) |
| Best in-repo patterns | CONFIRMED | `ToolProvider` `:1013-1020`; capability Protocols `:1023` / `:1043`; `register_provider` `:1887`; image dotted-path table `adapter_registry.py:30-40` |
| Doc drift | CONFIRMED | CLAUDE.md `:118-121`; `llm-providers.md` mentions neither the registry nor the presets |

## Verifier S — screens / misplaced code (14 claims)
| Claim | Verdict | Verified value / evidence |
|---|---|---|
| LibraryScreen size and `__init__` | CONFIRMED | Class 1006–37416 (36,411 lines, 1,354 methods); 653 single-call bodies (631 controller); `__init__` 2313–4195 |
| Controller construction in `__init__` | CONFIRMED | 21 controllers, 436 keywords, 413 lambdas |
| Program close numbers | CONFIRMED | `recipe.md:17` "COMPLETE"; `:7228` 32230/1258; spec `:64` ~4–6k target |
| Test coupling | CORRECTED | 370 real unbound calls in 46 files; 37 `object.__new__` in 8 files; **66** files reach controllers through `screen.`; only `LibraryIngestController` built directly (2 files) |
| Media / Notes controller bindings | CONFIRMED | Media `:3097` 92 lambdas; Notes `:3391` 95 |
| NotesSync controller | CONFIRMED (nominal) | 4 typed dependencies, 6 test files, but 2 dependencies are bound screen methods |
| SettingsScreen shape | CORRECTED | 3724–35602, 1,167 methods, **0** controller delegators; `_render_detail_pane` 22657–23490 (23 arms + membership + else); save 33004–33960; PROVIDERS_MODELS arm 33085–33664 |
| Category comparisons | CONFIRMED | 571 references; 239 comparisons in 65 methods; PROVIDERS_MODELS in 59 methods |
| Speech contract; TASK-1378 | CONFIRMED | `:3918` / `:4049`; TASK-1378 To Do since 2026-08-06 (~10.8k → 35.6k) |
| App string lookups in UI | CORRECTED | 1,176 lookups, 247 names, 71 private names (135 private sites); BaseAppScreen at `UI/Navigation/base_app_screen.py:101`; `library_screen.py:11828` writes an undeclared private app attribute |
| Widgets→UI imports | CORRECTED (framing) | 46 statements / 129 names / 35 modules exact; only 10 of 50 target `UI/Screens`; 12 land in Textual UI modules; theme editor is `Widgets/settings_theme_editor.py:2182` |
| Domain→presentation cases | CONFIRMED (all five) | (e) is 635 lines with a TYPE_CHECKING import at `:45` |
| Identity checks | CONFIRMED | `:1160-1183`; builder via `owner._build_console_turn_execution_context` `:1204`; 51 `__code__` refs in 20 files |
| Ratchet trend | CONFIRMED | Re-pin `:943-947`; ruling quoted `:116/:949/:1025`; row `:1048` 37542/1354; settings comment `:192` 33,607, row `:332` 35602 |

## Existing tasks that already own a concern (not re-filed)
TASK-32807.5, TASK-19571, TASK-32898 (restates TASK-32808.6), TASK-34905, TASK-33132, TASK-34906, TASK-1378 (parent of new subtasks), TASK-32463, TASK-35309, TASK-32809.2.
