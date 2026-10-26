# Architecture review — 2026-10-25

**Verdict:** The import graph is clean at import time but not structurally: lazy imports hide a 1,496-module cycle, and nothing checks layer direction globally. Most of the coupling runs through one object. `TldwCli` holds 397 attributes, and code in every layer reaches it with `getattr(app, "...")` strings. The god classes and god functions sit on the busiest paths, and their size ratchets now raise their rows instead of holding them. The repo's own decomposition doctrine works where it was run to completion (the store and bridge, `ToolProvider`, `LoopDeps`). Where it stopped halfway (Library controllers bound by lambdas, Console hosts, `_run_one`), the result is a file split, not a seam.

## Scope and provenance

- **Base:** `origin/dev` @ `7a5e2513c5` (2026-10-25 16:15 -0700), 2,820 modules, 2,008,669 lines of Python.
- **Method:** an AST import graph and class/function metrics (`scripts/`). Four read-only deep dives (screens, Console/Chat/Agents, providers/integration, persistence/config). Then **four independent adversarial verifiers**, one per slice, plus sandboxed runtime probes (scratch HOME, real profile write-denied, zero real-profile files touched).
- **Evidence:** every number below is the **verified** value. Corrections against the first pass are listed in `verification.md`, claim by claim.
- **Severity calibration:** the prompt's rubric marks any class over 500 lines as Critical. 507 classes qualify, so severity here is ranked by blast radius × change frequency instead.
- **Not applicable:** the prompt's e-commerce sections (channels, inventory, multi-tenancy) don't fit a single-user TUI. "Channel adapters" is mapped to LLM providers and the tldw_server `*_Interop` packages.

## Correction against the first-pass report

The session report's first finding, "agent note tools always fail in production", **was wrong**.

- `config.get_chachanotes_db_lazy()` (`config.py:11873`) assigns the module global, and `app.py:2084` calls it during startup.
- `Tools/note_management_tools.py` reads `config.chachanotes_db` at call time, so the tools work inside the running TUI.
- What is true: the tools work only after *something in the process* has called the lazy getter. A fresh process reproduces `"No unified database template for NotesInteropService."` That is temporal coupling through a lazily-populated singleton. It is now part of **F9**.

## Scorecard

| Dimension | Verdict |
|---|---|
| Import-time cycles | ✅ Only 2 strongly-connected components, both inside the app shell (3 and 2 modules) |
| Structural dependency graph | ❌ With lazy imports: one SCC of **1,496 modules / 93 packages**. Package level, top-level imports only: 91 packages in one SCC; 55 mutually-importing package pairs |
| Layer enforcement | ⚠️ 72 architecture tests, but forbidden-import checks are per feature (8 files); no global layer-direction rule |
| God objects | ❌ 102 classes over 2,000 lines; 50 functions with approximate cyclomatic complexity over 100 |
| Dependency inversion | ⚠️ The core seams are good. Elsewhere dependencies come from the app object or from config module globals |
| Testability | ⚠️ 18,487 `monkeypatch.setattr` calls. `test_app_extracted_patch_targets.py` exists because patches silently stopped reaching moved code |

---

## Findings

Each finding lists the backlog tasks filed for it. Task IDs are `TASK-<stream>.<n>`; see `Docs/superpowers/plans/2026-10-25-architecture-review-roadmap.md` for sequencing.

### F1 — The application object is a service locator · Critical · Pattern/DIP · Entire codebase
- **Evidence:**
  - `app.py` plus the `app_*.py` mixins assign **397 distinct attributes** on `TldwCli`. About 199 are named like services (`local_X_service` / `server_X_service` / `X_scope_service` for ~30 domains).
  - Outside `UI/`, `Widgets/` and `css/`, **352 functions in 71 modules** take `app`/`app_instance`. They read 233 distinct attributes from it, **222 via `getattr(app, "<literal>")`**.
  - Inside `UI/`, there are **1,176** such string lookups across **247 names**, 71 of them private (135 private call sites). This holds even though `UI/Navigation/base_app_screen.py:101` types `app_instance` as `TldwCli`.
  - `library_screen.py:11828` writes an undeclared private app attribute (`_library_source_snapshot_cache`).
  - `ConsoleChatController` holds a settable `self.app` (`console_chat_controller.py:4590`, assigned at `chat_screen.py:9256` and `console_launch_wake.py:223`). It reads ≥18 app attributes and forwards the whole app into 5 helpers (`:8886`, `:20998`, `:21010-21014`, `:21070`, `:21126-21129`).
  - `ConsoleRuntime` reads a screen's private `_console_hook_review_projection` through `view.app.screen` / `screen_stack` (`console_runtime.py:4981`, `:5296-5297`).
  - `Event_Handlers/Chat_Events/chat_rag_events.py` is a RAG application service: 22 of its 36 functions take `app: "TldwCli"`.
  - Other modules build their own DBs: `MCP/server.py:604`, `Web_Server/mobile_service.py:420/832`, and config's lazy singletons (F9).
- **Recommended structure:** a frozen, typed application-services bundle built once in `app_service_wiring`. Each consumer receives its slice through its constructor. No container, no registry. ADR-036 deferred exactly this "until more than one real composition root" exists, and the evidence above suggests that trigger has been reached. That is the owner's call (TASK-36947.1).
- **Strategy:**
  1. Ratchet string lookups first (TASK-36942.4).
  2. Record the decision.
  3. Introduce the bundle with the old attributes as aliases.
  4. Migrate consumer clusters one PR at a time: RAG first, the Console core last.
- **Tests affected:** 162 test files build a fake `app_instance`; `Tests/Architecture/test_app_extracted_patch_targets.py`.
- **Effort:** High (incremental). **Risk:** a consumer that used to get `None` quietly now fails loudly. Aliases keep each PR revertible.
- **Payoff:** visible, typed dependencies; tests stop faking the app.
- **Tasks:** TASK-36942.4, TASK-36947.1–.5.

### F2 — LibraryScreen keeps growing; its controllers are a file split, not a seam · Critical · Coupling/SRP · Module (busiest)
- **Size:** `LibraryScreen` is lines 1006–37416 (36,411 lines, 1,354 methods); the file is 37,542 lines. The decomposition recipe declares the program COMPLETE (`library-decomposition-recipe.md:17`) at **32,230 / 1,258** (`:7228`). That is **+5,312 lines / +96 methods** since close. The spec targeted a ~4–6k-line shell (spec `:64`).
- **What the class holds:**
  - 653 single-call bodies, 631 of them controller delegators. Everything else is real logic.
  - `__init__` (2313–4195, 1,883 lines) builds **21 controllers with 436 keyword arguments, 413 of them lambdas**.
- **How the controllers bind:**
  - `LibraryMediaController` (`:3097`) and `LibraryNotesController` (`:3391`) take `screen` plus 92 and 95 lambdas.
  - Tests:
    - 370 real unbound `LibraryScreen._x(fake_self, …)` calls in 46 test files;
    - 37 `object.__new__(LibraryScreen)`;
    - **66 test files** reach `screen._*_controller`.
  - Only `LibraryIngestController` is ever constructed directly (2 files). Media, Notes, Prompts, Skills, Conversations, Export, RAG and Collections: 0.
- **Partial counter-example:** `LibraryNotesSyncController` (`:3892-3899`) has 4 typed dependencies, but 2 of them are bound screen methods.
- **Recommended structure:** each controller owns its state dataclass and typed dependencies, with no screen reference. The screen keeps compose, routing and delegators.
- **Strategy:** "Phase A2", media first:
  1. Ratchet the lambda count and the unbound-call count.
  2. A test-only PR retargets media fixtures, byte-for-byte assertions.
  3. Move PRs delete one callback argument and one delegator per moved method.
- **Tests affected:** the 46 files with unbound calls, the 66 controller-reach files, the screen and library-modules size ratchets.
- **Effort:** High. **Risk:** click-freeze regressions (recipe §24 measurement discipline).
- **Payoff:** controllers testable alone; fewer merge conflicts.
- **Tasks:** TASK-36952.1–.3. Related: TASK-32463 (over-budget decision), TASK-35309 (controllers over budget).

### F3 — SettingsScreen is closed to extension · High · SOLID-OCP · Module
- **Evidence:**
  - **Size:** `SettingsScreen` is lines 3724–35602 (31,879 lines, 1,167 methods) with **0** controller delegators.
  - **Branching on category:** 571 references to `SettingsCategoryId` members, including **239 comparisons across 65 methods**.
  - **`_render_detail_pane`** (22657–23490, 834 lines) is 23 `category is X` arms plus one membership arm and an else.
  - **`action_settings_save_category`** (33004–33960, 957 lines) has a 580-line PROVIDERS_MODELS arm inline (33085–33664). PROVIDERS_MODELS appears in 59 methods; the contract-driven categories (SKILLS, PERSONAS, ARTIFACTS) appear in 2.
  - **A working contract already exists:** `speech_tts_settings_panel.py:3918` / `:4049` (`has_unsaved_changes` / `request_save`).
  - **Status:** TASK-1378 (split Settings) is still To Do. It was filed at ~10.8k lines; the file is now 35.6k.
- **Recommended structure:** a `SettingsCategoryId`-keyed panel registry generalizing the Speech contract (render, save, unsaved, revert, scope text, ADR-033 commit model).
- **Strategy:** NETWORK first as the proof, then PROVIDERS_MODELS; ratchet the comparison count.
- **Tests affected:** 125 test files reference `SettingsScreen`.
- **Effort:** Medium–High. **Risk:** ADR-033 save ordering; pin it per category before each move.
- **Tasks:** TASK-1378.1, TASK-1378.2.

### F4 — Console controller collaborators are bound through lambdas; replies take 78 keywords · High · Coupling · Module
- **The controller:** `ConsoleChatController` is lines 3623–28971 (25,349 lines, 508 methods). Its `__init__` (3702–4996, 1,295 lines) has 51 parameters, 195 attributes and 191 lambdas.
- **InterruptRoundHost:** `InterruptRoundHost(...)` (`:4025`) takes **122** keywords, 121 of them lambdas (86 `read_controller_*`, 33 `read_global_*`). The host mutates controller dicts through those getters (`console_interrupt_rounds.py:2971-2978`).
- **ConsoleCompactionPreflight:** `ConsoleCompactionPreflight(...)` (`:4886`) takes 67 lambdas. 41 are `read_global_*`, injected module globals and types, so the coupling there is mostly patch-target preservation, not shared state.
- **The reply call:** a maintenance call (`:26835-27014`) passes **78 keywords** to `ConsoleAgentBridge.run_reply` (78 parameters). `AgentService.__init__` takes 59, and `LoopDeps` has 61 fields.
- **Done well:** `ConsoleChatStore` (19 parameters) and `ConsoleAgentBridge` (16) are free of Textual and of the app.
- **Recommended structure:** hosts own the state they read; the preflight imports its own collaborators; a frozen `ReplyRequest`.
- **Tests affected:** 208 files reference `ConsoleChatController`.
- **Effort:** High. **Risk:** in-turn ordering; the `test_console_*_ownership` tests are the net.
- **Tasks:** TASK-36951.1–.3.

### F5 — The agent loop is two god functions; the "pure" runtime isn't · High · SRP · Module
- **Evidence:**
  - **`_run_one`:** `AgentService._run_one` (`agent_service.py:5006-9168`) is **4,163 lines** with 45 nested functions. The biggest are `spawn` 551 (`:6281`), `_launch_fleet_child` 490 (`:5707`) and `_resume_retained_child` 324 (`:7078`).
  - **`run_agent_loop`** (`agent_runtime.py:1073-3523`): its `while True` (1729–3523) is 1,795 lines. The per-call loop (2637–3506) is 870 lines with 20 `call.name ==` and 3 `call.name in` branches. A dict dispatch already exists at `:3103-3108`.
  - **Not pure:** `agent_runtime.py:400-404` says the module is "a pure module today (no Textual, app, DB, or I/O imports)". A sandboxed probe shows importing it loads `sqlite3`, `httpx`, **`textual`**, `DB.ChaChaNotes_DB`, `Chat` and `config` (207 tldw_chatbook modules). The cause is `:22` importing `Chat.console_project_instructions` (which imports httpx) through `Chat/__init__` (which imports the DB).
  - **No guard:** `agent_models` has a purity test (`Tests/Agents/test_agent_models.py:409`); `agent_runtime` has none.
- **Recommended structure:** a special-tool dispatch table, then phase functions over a `LoopState`; `_run_one`'s fleet closures become a per-run class.
- **Tests affected:** 144 test files reference `AgentService`; 54 reference `run_agent_loop`.
- **Effort:** High. **Risk:** very high if rushed; pure moves only.
- **Tasks:** TASK-36942.2 (purity guard), TASK-36950.1–.4.

### F6 — No enforced layer map · High · Layering · Entire codebase
- **Evidence:**

| Upward direction | Edges | Of which top-level | Source modules |
|---|---|---|---|
| Domain → presentation | 83 | 42 | ~45 |
| DB → feature packages | 358 | 169 | — |
| Kernel (Utils/config/runtime_policy) → higher | 99 | — | — |

  - DB importing domain `*_models` modules is the correct direction; DB importing services is not.
- **Recommended structure:** the layer ranks kernel < DB < domain < presentation, enforced by one stdlib AST test (`scripts/graph.py` is the starting point) with a frozen allowlist that may only shrink.
- **Effort:** Low (guard) / Medium (burn-down). **Payoff:** decay stops being silent.
- **Tasks:** TASK-36942.3.

### F7 — Chat/ is not a bounded context; shared provider types have no neutral home · High · Modularity · Entire codebase
- **What Chat/ holds:**
  - 250 files, 247,053 lines. 167 `console_*` files hold 194,740 lines (78.8%).
  - **42** Chat files import `tldw_chatbook.DB`, though `Docs/Architecture/console.md:120` says "`Chat/` never imports the DB layer".
- **Cycles through Chat:**
  - Agents↔Chat and Chat↔LLM_Calls import each other.
  - `ChatProviderError` is imported from `Chat.Chat_Deps` by 12 LLM_Calls files. `Chat_Deps.py` has no imports, but importing it runs `Chat/__init__`, which loads `ChaChaNotes_DB`.
  - Agents imports 2 Chat-private names (`agent_routing.py:208`, `:407`); Chat imports 5 Agents-private names.
- **Recommended structure:** a dependency-free provider-contract package that Chat, Agents and LLM_Calls depend on, with re-export shims, plus a boundary test.
- **Tasks:** TASK-36944.1, TASK-36944.2.

### F8 — CharactersRAGDB fuses aggregates; persistence calls domain code · High · SRP/Layering · Entire codebase
- **The class:** `CharactersRAGDB` (`ChaChaNotes_DB.py:794-25030`) has 437 methods (233 public). Approximate clusters: schema/migrations ~102; learning exactly 63; messages ~64; conversations ~35; characters ~34; keywords ~32; notes ~26 public / 38 total; sync ~13–15; kept briefings/scripts 13. World books, canvas, folders, traces and citations already live in repositories that join its transaction.
- **Persistence calling domain code:**
  - 12 write sites go through `_ensure_initial_semantic_revision` ×4 and `_coordinate_semantic_mutation` ×8, which lazily import `Chat.console_semantic_revision` (`:14212`, `:14238`). That module imports `CharactersRAGDB` at top level (`:33`), so it is a graph cycle.
  - `soft_delete_message` (`:16521`; capture `:16581`) computes Sync v2 hashes through `Sync_Interop.hashing.canonical_payload_hash` (`:16826`).
- **Private reach-ins:** 10 sites in 5 files read `db._local.transaction_depth`.
- **The learning split is cheap:** only ~45 call sites actually hit it (`Study_Interop/local_study_service.py` 31, `local_quiz_service.py` 14).
- **No size ratchet row** for this 25,565-line file.
- **Recommended structure:** one repository per aggregate over the same connection and transaction. Inject the semantic coordinator and the chat-sync adapter (the `ChatSyncIntentSource` Protocol already exists, `Sync_Interop/chat_outbox_producer.py:46-66`).
- **Tests affected:** 675 test files mention `CharactersRAGDB`; they stay green while delegators remain.
- **Tasks:** TASK-36949.1–.5. Related: TASK-34906 (declarative migration registry) owns the migrations cluster.

### F9 — Config is a global service locator, a DB factory and a god module · High · DIP/Pattern · Entire codebase
- **What's in the module:** `config.py` is 12,258 lines with 220 top-level functions:
  - `CONFIG_TOML_CONTENT` (4078–6661, **2,584 lines**);
  - `_load_settings_uncached` (2272–3943, 1,672 lines);
  - encryption (10159–10854);
  - canvas policy (165–446);
  - Console capture/trace policy (9312–9735; lazy Chat imports);
  - top-level DB class imports (`:70-72`).
- **DB singletons with hidden init order:** `get_chachanotes_db_lazy` / `get_prompts_db_lazy` / `get_media_db_lazy` (`:11873`, `:11908`, `:11938`) fill module globals.
  - **Consumers depend on someone having called the getter first:** `app.py:750/:2084`, `app_feature_glue.py:291`, `UI/CCP_Modules/ccp_character_handler.py:46`, `Tools/document_expansion_tool.py:173/:215`, `Utils/doctor.py:143`, `MCP/local_runtime_delegate.py:672` and `Character_Chat/Character_Chat_Lib.py:854`.
  - **Note tools:** `Tools/note_management_tools.py:69` reads the global itself; a fresh process reproduces the failure.
  - **Dead initializer:** `initialize_all_databases` (`:11827`) has no production caller.
- **Settings reads:**
  - `get_cli_setting` is called from 356 sites in 104 modules across 30 packages, reading ~240 keys in ~46 sections.
  - Import-time reads freeze values against hot reload: `Local_Ingestion/Document_Processing_Lib.py:79-86`, `PDF_Processing_Lib.py:57-62`, `Utils/optional_deps.py:1540`, `RAG_Search/simplified/rag_service.py:96`, `app.py:693`.
  - Tests patch `get_cli_setting` ~246 times.
- **Recommended structure:** config loads, merges and saves. The template is package data, policies live with their features, DB instances come from composition roots, and readers use typed section accessors.
- **Tasks:** TASK-36945.1–.5.

### F10 — Backup_Recovery hard-codes every participant · Medium-High · OCP/DIP · Entire codebase
- **Evidence:**
  - **Hard-coded registration:**
    - `_repository_types()` (`participants.py:293-366`) lists 24 exact types (13 eager, 11 gated on `sys.modules`).
    - There are **two closers tables that disagree** (`:66-78`, 11 entries; `:125-145`, 15 entries) and a `policy_id` chain (`:531-553`).
    - The type table is rebuilt ≥3 times per `CharactersRAGDB.get_connection`.
  - **66 production modules** import participants; stores import its private `_core_*` names at top level (`ChaChaNotes_DB.py:119-125`).
  - **`unsaved_editors.py`:** 36 `_exact()` checks, reads of ChatScreen private fields (`:584-592`), and lazy imports of 10 modules.
  - **Feature-private imports:** `staging.py:75-82` imports 8 feature-private classes.
- **Recommended structure:** stores call `participants.declare(ExactType, owner_id=, closer=, policy=)`; screens expose `unsaved_editor_state()`.
- **Tasks:** TASK-36948.1–.3.

### F11 — Provider knowledge is duplicated outside the registry; credential precedence disagrees · Medium · OCP · Module
- **Wiring duplication:**
  - The newest preset, `commandcode`, needed **13 edit sites in 9 files**, ~8 of which repeat `ProviderRecord` data. ADR-179 lines 38–39 promise "one registry record + one dispatch registration line".
  - TASK-33510 and TASK-33511 were hand-list drift bugs of exactly this kind.
  - Provider-key comparisons: 170 in 69 files (strict) or 320 in 98 files (any left side). `API_CALL_HANDLERS` has 62 keys, 33 of them engine handlers. The `PROVIDER_PARAM_MAP` literal is 628 lines (342–969).
- **Credential precedence disagrees across resolvers (verified in source):**

| Resolver | Order |
|---|---|
| Engine (`hosted_provider_engine.py:601`; 33 presets) | explicit → **env** → stored |
| `moonshot.py:626`, `zai.py:595`, `qwencloud.py:365` | explicit → **stored** → env |
| Custom-endpoint registry (`:495`) | env → stored |

  CLAUDE.md states the stored `api_settings` key outranks the env var; the engine's own comment states the opposite rule. A user with both set gets a different key depending on the provider family.
- **Kwargs-builder drift:** the drifted builder is the dead `_chat_api_kwargs` (zero production callers, 23 test calls), already owned by **TASK-32807.5**. Not re-filed.
- **Tasks:** TASK-36941.3 (precedence; needs an owner decision), TASK-36946.1–.2.

### F12 — Shared code lives in widget/screen files; Chat identity-checks UI internals · Medium · Layering · Module
- **Evidence:** pure helpers sit in Textual modules:
  - approval formatters (`permission_summary_service.py:15` → `chat_approval_card.py`, which imports Textual at `:39`);
  - inspector state dataclasses (`console_environment_state.py:253` → `console_inspector_section.py:145/:219`);
  - the scope-picker `SourceLister` port (`scope_picker_listers.py:41` → `console_scope_picker_modal.py:122`; the modal imports Chat back at `:89/:91`, a cycle);
  - a text sanitizer (`UI/character_display_text.py`, imported by 4 Chat modules);
  - a 635-line DB+git provider (`change_review_screen.py:252-886`, lazily imported by `console_agent_bridge.py:11227`).
- **Widgets→UI:** 46 runtime statements import 129 names from 35 widget modules. Only 10 of them target `UI/Screens`, and 12 land in Textual UI modules. There are two function-local circular-import workarounds (`Widgets/settings_theme_editor.py:2182`, `personas_character_editor_widget.py:1453`).
- **The one real upward coupling:** `console_configuration_preparation.py:1160-1183` imports UI-private symbols and compares `__code__` / `__defaults__` identity. Chat has 51 `__code__` references in 20 files.
- **Tasks:** TASK-36944.3–.9.

### F13 — 26 transaction implementations; Subscriptions can commit abandoned writes · Medium (defect: High) · Pattern · Module
- **Transaction managers:** ~26 SQLite transaction managers, 6 on the kernel. The store-kernel ADR-228 is still Proposed. The same three exception classes are defined in 4 DB modules (Prompts `:82-94`, Media `:90-102`, Evals `:131-143`, ChaChaNotes `:442-454`).
- **Defect, reproduced:** `Subscriptions_DB.transaction()` (`:2104-2116`) rolls back only on `Exception`, on a reused thread-local connection with legacy implicit BEGIN (`:647`). After a `BaseException` (KeyboardInterrupt, SystemExit, GeneratorExit) the depth resets while the transaction stays open. The probe:

  ```
  after BaseException: conn.in_transaction = True | depth = 0
  next transaction(immediate=True) raised: OperationalError cannot start a transaction within a transaction
  ```

  The next plain `transaction()` on that thread commits the abandoned partial writes. Kanban and Scheduling are mitigated because they close the connection in `finally`.
- **Tasks:** TASK-36941.1, TASK-36949.5. Related: TASK-34905, TASK-33132 (consolidation).

### F14 — TLDWAPIClient is a fat interface; one Interop service always uses the private transport · Medium-Low · ISP · Module
- **The client:** `TLDWAPIClient` (`tldw_api/client.py:1209-16790`) has 1,220 methods (1,193 public). Across 47 `server_*_service.py` files the median uses 13.
- **Interop copy-paste:** `_require_client` has 44 definitions (42 identical), `_enforce` 49 (44 identical), and `_maybe_await` 61 copies. TASK-32898 / ADR-177 own the helper consolidation (Proposed).
- **Defect:** `Research_Interop/server_research_search_service.py:142/:161/:186` probe `run_paper_search`, `get_paper_search_detail` and `run_paper_search_ingest`. None exist; the client has `paper_search` (`:8709`), `paper_search_detail` and `paper_search_ingest`. So the private `client._request` fallback (`:145/:164/:189`) is **always** the live path.
- **Tasks:** TASK-36941.2, TASK-36953.1.

### F15 — Package placement is type-based and smeared · Low-Medium · Modularity · Entire codebase
- **Smeared features:** the Console spans Chat (167 files), Widgets (112) and UI (68). Library spans UI (67), Library (61), Widgets (43) and Chat (9).
- **Misfiled RAG service:** `RAG_Search/pipeline_loader.py:23` imports `Event_Handlers.Chat_Events.chat_rag_events` at top level. That module is the RAG service from F1.
- **Utils:** 96 of `Utils/`'s 161 files are `Splash_Screens/`. `Utils/input_validation.py` (2,457 lines) lazily imports Agents, Chat and Terminal.
- **Tasks:** TASK-36947.3 (RAG service), TASK-36953.2–.3.

### F16 — 40 modules of 4,000+ lines have no size ratchet row · Low · Modularity · Entire codebase
- **Evidence (exact `_BUDGETS` keys via AST):** the largest unbudgeted modules are `DB/ChaChaNotes_DB.py` 25,565, `config.py` 12,258, `Chat/console_agent_bridge.py` 12,093, `Notes/file_notes_git_service.py` 11,400, `Agents/agent_service.py` 9,629 and `Widgets/Library/library_file_notes_workspace.py` 9,532 … down to `DB/private_sqlite.py` 4,030 (full list in `verification.md`).
- **Trend under the 2026-10-04 ruling:** the Library screen row rose 35,793 → 37,542 and the Settings module row 33,607 → 35,602 since their pins. Reported, not judged; that ruling is the owner's.
- **Related:** TASK-32809.2 (its named list), TASK-32463 (over-budget decision).
- **Tasks:** TASK-36942.1.

### F17 — Dead code, doc drift, ADR number collisions · Low · Pattern · Single module
- **Dead code:**
  - `Chat/chat_models.py` imports `textual.worker` (`:7`) and is used only by `Tests/UI/test_console_internals_decomposition.py:36`.
  - `Chat/document_generator.py`: all 20 handler functions in its 25-key table fail signature binding (probe). Already **TASK-32807.5**.
- **Doc drift:**
  - `console.md:120`;
  - CLAUDE.md's Chat description and "New LLM Provider" steps (`:118-121`), which name neither the registry nor the presets;
  - `llm-providers.md:36`, which mentions neither the registry nor the presets;
  - `size-decomposition-candidates-2026-09-18.md`, which lists ChaChaNotes_DB (`:117`), and that file has no row.
- **ADR numbering:** **50 ADR numbers are used by two files**, including 228 (store kernel and config-encryption memoization).
- **Tasks:** TASK-36941.4, TASK-36943.1–.2.

---

## Done well (keep)
- **Import-time cycles:** essentially none.
- **Agents layer:** imports nothing from UI/Widgets.
- **Narrow seams:** `LoopDeps`; the `ToolProvider` Protocol (`tool_catalog.py:1013`) with runtime-checkable capability Protocols; a framework-free store and bridge.
- **Provider registry:** frozen, stdlib-only records with parity tests; 33 presets carry no code of their own.
- **One server transport,** with redirect refusal.
- **SQL safety:** parameterized SQL, optimistic locking, the quiescence registry, the fail-closed mutation authorizer, and `transaction_observer.py`.
- **Guardrail culture:** a ratchet and arch-test culture with dated evidence ledgers, and almost no service construction inside screens.

## Already owned elsewhere (not re-filed)
| Concern | Owner |
|---|---|
| `Chat/document_generator.py` and four other dead Chat modules; the dead `_chat_api_kwargs` builder | TASK-32807.5 |
| Wire-or-retire policy for unreachable modules | TASK-19571 |
| Scope-service helper consolidation (ADR first) | TASK-32898 (restates TASK-32808.6) |
| Transaction-manager consolidation / store kernel | TASK-34905, TASK-33132 |
| Declarative ChaChaNotes migration registry | TASK-34906 |
| Settings split (parent of the new subtasks) | TASK-1378 |
| Over-budget screens decision | TASK-32463 |
| Library controllers over budget | TASK-35309 |
