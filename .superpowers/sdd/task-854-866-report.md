# TASK-854 / 855 / 858 / 865 / 866 -- path-accessor sweep report

**Worktree:** `/Users/macbook-dev/Documents/GitHub/wt-path-accessors` (branch `fix/path-accessor-sweep`, cut from `origin/dev`)
**Interpreter:** `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/wt-path-accessors /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`, verified via `tldw_chatbook.__file__` before every probe.
**Theme:** derive paths from the accessors the app actually uses instead of re-spelling their values -- the class of bug behind the `mcp_permissions.json` denylist miss the originating audit (TASK-846) found.

---

## TASK-854 -- MCP server media DB lookup

**Bug:** `MCP/server.py`'s `_init_databases()` called `get_cli_setting("database", "media_db", "media_library.db")`. The key `"media_db"` is declared nowhere; the real key is `media_db_path`, accessor `get_media_db_path()`. Because the key never matched, the lookup always fell through to the CWD-relative literal.

**Before/after (sandboxed HOME, no `TLDW_CONFIG_PATH`):**

```
get_cli_setting("database","media_db","media_library.db") -> 'media_library.db'
resolved (relative to CWD)  : /Users/macbook-dev/Documents/GitHub/wt-path-accessors/media_library.db
get_media_db_path()         : /var/folders/.../task854_.../.local/share/tldw_cli/default_user/tldw_chatbook_media_v2.db
MATCH before fix: False
```

After the fix, `_init_databases()`'s `self.media_db.db_path` equals `get_media_db_path()` exactly (verified by a real, unmodified call to `_init_databases()` in `Tests/MCP/test_server_media_db_path.py`).

**Fix:** replaced the bad `get_cli_setting` call with `get_media_db_path()`.

**Adjacent bug found and fixed (required to make the module constructible at all, hence in-scope for AC #3):** `MediaDatabase(media_db_path)` was called with **no `client_id`** -- a required positional argument -- so `_init_databases()` has apparently never completed successfully; this always raised `TypeError`. Fixed to `MediaDatabase(db_path=media_db_path, client_id=CLI_APP_CLIENT_ID)`, matching every other `MediaDatabase` call site in the app.

**Adjacent bug found and NOT fixed (separate, out of scope):** the same method also constructs `NotesInteropService(self.chachanotes_db)` (wrong positional arg -- `NotesInteropService.__init__` requires `(base_db_directory, api_client_id, ...)`) and `CharacterInteropService(self.chachanotes_db)`, where **`CharacterInteropService` does not exist anywhere in the codebase** (`grep -rn "class CharacterInteropService"` -> zero hits). This means `_init_databases()` has never run to completion in production. Recommend filing a follow-up task. The new regression test stubs both collaborators (permissive fakes swapped into their real source modules) purely to drive the real, unmodified method far enough to construct `self.media_db` -- it does not fix or mask this separate defect.

**AC #2 grep result (report on findings):** wrote an AST-based (not line-regex) scan of every `get_cli_setting("database", ...)` call site in `tldw_chatbook/`, requiring the key to either match `[a-z0-9_]+_db_path` or be one of the known non-path `[database]` settings (`check_integrity_on_startup`, `integrity_check_timeout`, `USER_DB_BASE_DIR`). Result: **zero other offenders** -- every remaining call site (all 14, all in `config.py`, plus one already-fixed `TASK-658` call in `Local_Ingestion/local_file_ingestion.py`) uses a declared `*_db_path` key. This scan is now `Tests/MCP/test_server_media_db_path.py::test_no_other_undeclared_database_config_keys`, so it re-runs on every future change.

**Files:** `tldw_chatbook/MCP/server.py`; `Tests/MCP/test_server_media_db_path.py` (new).

---

## TASK-855 -- MCP store module defaults

**Bug:** `MCP/local_store.py`, `unified_context_store.py`, `server_target_store.py` all defaulted to `DEFAULT_CONFIG_PATH.parent / <name>` (i.e. `~/.config/tldw_cli/`). Every real construction site (`app.py`) always passes an explicit `get_user_data_dir() / <name>` path, so this was latent, not live -- but the permission store and execution log are both *derived* from a `LocalMCPStore`'s own `.path` via `Path(store.path).with_name(...)`, so a store built with no argument anywhere would place both outside `Utils.sensitive_paths`' denylist coverage.

**Decision: derive lazily, don't require an explicit path.** Considered the two options the task offered. Nothing in the codebase (production or the ~90 test call sites across `Tests/MCP/`/`Tests/RuntimePolicy/`) ever constructs these classes with no argument, so "require an explicit path" would have been equally safe today -- but it would take away a legitimate future escape hatch for no real benefit. Instead each module now computes its default **lazily**, inside `__init__`, via a private `_default_*_path()` helper that calls `get_user_data_dir()` at call time, and the three eager `DEFAULT_*_PATH` module constants were removed (grepped first: nothing outside their own module referenced them). Lazy, not an eager module-level constant, for two reasons:
1. `get_user_data_dir()` has side effects (reads live config, creates the directory) that an eager import-time constant would trigger merely by importing the module.
2. An eager constant bakes in whichever profile/HOME was active the first time the module was imported in a process -- exactly the staleness class `Utils/sensitive_paths.py`'s own lazy `_sensitive_db_paths()`/`_sensitive_single_file_paths()` helpers exist to avoid.

**Files:** `tldw_chatbook/MCP/local_store.py`, `unified_context_store.py`, `server_target_store.py`; `Tests/MCP/test_store_default_paths.py` (new, 6 tests: default-path derivation for all three stores, a TLDW_CONFIG_PATH-retargeting proof that the default is resolved at call time not cached, denylist coverage of the default and its permission-store/execution-log derivatives, and explicit-path-sites-unaffected).

---

## TASK-858 -- evals/prompts/media/rag/subscriptions DB path maps

**Already done on this branch's base (verified, not redone):** the Settings screen's maintenance DB-path map (AC #2) and its six-database parity test (AC #4) were already fixed by an earlier, already-merged task (in-code comments reference `TASK-899`): `UI/Tools_Settings_Window.py`'s `_DB_PATH_RESOLVERS` dict (lines ~6823-6830) already delegates to `get_chachanotes_db_path`/`get_media_db_path`/`get_prompts_db_path`/`get_evals_db_path`/`get_rag_indexing_db_path`/`get_subscriptions_db_path`, and `Tests/UI/test_tools_settings_window.py` already has `test_get_database_path_resolves_via_config_resolvers_and_honours_profile` plus per-DB parity tests and a parametrized backup/restore round-trip test over all six databases. Re-ran all of these after my `config.py` change -- still 9/9 pass.

**Genuinely still broken (fixed):** `Event_Handlers/eval_db_operations.py:28` still hardcoded `Path.home() / ".config" / "tldw_cli" / "evals.db"`. Fixed to call `config.get_evals_db_path()` -- the same accessor `Evals/eval_orchestrator.py`'s `EvaluationOrchestrator` delegates to for its own default case, so both agree.

**AC #3 -- declare or delete `evals_db_path` / `rag_db_path` / `subscriptions_db_path`:** none were dead (all three accessors -- note the real, only key ever read is `rag_indexing_db_path`; `rag_db_path` does not appear anywhere in the codebase -- have live callers), so declared all three as real `[database]` config defaults in `config.py`'s TOML template, following the exact sentinel-literal convention the three existing entries (`chachanotes_db_path`/`prompts_db_path`/`media_db_path`) already use, with their correct real fallback filenames (`evals.db`, `rag_indexing.db`, `tldw_chatbook_subscriptions.db`). This is functionally a no-op for existing users -- the accessor's custom-path branch only fires when a value differs from the template's sentinel, and an unset key already fell through to the correct `get_user_data_dir()`-based path before this change -- but closes the "declared nowhere" gap and documents the override for new installs.

**Files:** `tldw_chatbook/Event_Handlers/eval_db_operations.py`, `tldw_chatbook/config.py`; `Tests/Event_Handlers/test_eval_db_operations_path.py` (new).

---

## TASK-865 -- sweep hardcoded `~/.config/tldw_cli` / `~/.local/share/tldw_cli` sites

Status left as **In Progress** (AC #1/#2 not checked off) -- see rationale below.

**100% complete: every site the task named with an explicit file:line reference.**
- Config-dir group: `UI/Screens/chat_screen.py` (both `ui_state.toml` sites, `_load_sidebar_state`/`_save_sidebar_state`), `Event_Handlers/notes_events.py` + `note_ingest_events.py` (`note_templates.json`), `Subscriptions/website_monitor.py` (`feed_cache/`).
- Data-dir group: `Chatbooks/chatbook_importer.py` (the highest-value fix -- `temp_dir` now `get_user_data_dir() / "temp" / "imports"`, matching `chatbook_creator.py`'s sibling `get_user_data_dir() / "temp" / "chatbooks"`), `Chatbooks/local_chatbook_service.py`'s `_default_registry_path()` fallback branch, `Character_Chat/Character_Chat_Lib.py` (all 3 `base_directory` sites), `Event_Handlers/conv_char_events.py` (all 3 `export_dir` sites).

**Best-effort additional fixes (not individually named in the task, done as bonus):** `Widgets/emoji_picker.py` (converted the eager `RECENT_EMOJIS_FILE` module constant to a lazy `_recent_emojis_path()` -- an eager constant here would have the same staleness problem TASK-855 fixed for the MCP stores), `Widgets/settings_theme_editor.py`, `Notes/sync_service.py`, `Config_Files/create_custom_template.py` (standalone dev script), `RAG_Search/pipeline_loader.py` + `pipeline_builder_simple.py`.

**Why AC #1/#2 are NOT checked off:** the task also references "~25 lower-value" config-dir sites and "~18" data-dir sites only in aggregate, without file:line references. A full sweep of that remainder was not completed given its size, and I did not want to overclaim.

**A separate, adjacent, larger finding surfaced during the broader grep -- recommend its own follow-up task, deliberately NOT fixed here:** `UI/ChatbookCreationWindow.py`, `UI/ChatbookExportManagementWindow.py`, `UI/Wizards/ChatbookCreationWizard.py` and `UI/Wizards/ChatbookImportWizard.py` each build their own ad-hoc `db_paths` dict straight from `self.app.config_data.get("database", {})` with hardcoded, wrong, non-user-folder fallback literals (e.g. `"~/.local/share/tldw_cli/tldw_prompts_db.db"`) instead of calling `get_prompts_db_path()`/`get_media_db_path()` -- these bypass the `get_*_db_path()` accessors entirely, unlike everything TASK-858/899 reconciled. This is a distinct defect class (wrong-accessor-bypass, not a `Path.home()`-literal sweep site), so it was left alone rather than folded in here.

**Deliberately excluded as a scope decision (not an oversight):** TTS/UI model-weight and voice-cache paths under `~/.config/tldw_cli/models/...` and `.../*_voices` (`STTS_Window.py`, `Dictation_Window.py`, `TTS/backends/kokoro.py`, `TTS/kokoro_pytorch.py`, `TTS/utils/download_models.py`). These are large, shared binary caches/exports, not per-profile config/state -- making them profile-relative would force re-downloading multi-hundred-MB models on every profile switch, which is very unlikely to be intended and is exactly the kind of live-file-relocation the task's hard constraints said to stop and report on rather than silently do.

**AC #3/#4 -- fully satisfied, with tests:** `Tests/Chatbooks/test_chatbook_importer.py` (new tests: `temp_dir == get_user_data_dir() / "temp" / "imports"`, and shares a parent with `ChatbookCreator`'s `temp_dir`). `Tests/UI/test_chat_screen_ui_state_path.py` (new: `TLDW_CONFIG_PATH` retargeted to a scratch profile -> `_save_sidebar_state()` writes `ui_state.toml` under THAT profile's directory; two different profiles do not collide).

**Files:** `tldw_chatbook/UI/Screens/chat_screen.py`, `Event_Handlers/{notes_events,note_ingest_events,conv_char_events}.py`, `Subscriptions/website_monitor.py`, `Chatbooks/{chatbook_importer,local_chatbook_service}.py`, `Character_Chat/Character_Chat_Lib.py`, `Widgets/{emoji_picker,settings_theme_editor}.py`, `Notes/sync_service.py`, `Config_Files/create_custom_template.py`, `RAG_Search/{pipeline_loader,pipeline_builder_simple}.py`; `Tests/Chatbooks/test_chatbook_importer.py`, `Tests/UI/test_chat_screen_ui_state_path.py` (new).

---

## TASK-866 -- sensitive-path and skills-fixture tests must re-derive, not re-spell

**`Tests/Utils/test_sensitive_paths.py`:** its two MCP-store tests re-typed `"mcp_permissions.json"` / `Path(store.path).with_name(...)` the same way `unified_control_plane_service.py`'s `permission_store`/`execution_log` properties do internally, instead of reading the paths off a live instance. Rewrote both to build a real `UnifiedMCPControlPlaneService` (the `SimpleNamespace(store=...)` idiom already used in `Tests/MCP/test_control_plane_permissions.py`) with a default-path `LocalMCPStore()` (TASK-855 made this default itself derive from `get_user_data_dir()`, so no literal filename is spelled anywhere now) and assert directly on `service.permission_store.path` / `service.execution_log.path` / `service.local_service.store.path`.

**`Tests/conftest.py` + `Tests/Skills/test_skills_library_flow.py`:** `make_trust_service` and the two `_real_*_trust_service` builders hardcoded `tmp_path / "skills"` and `tmp_path / "trust"` -- matching, by re-spelling, `LocalSkillsService`'s private `_SKILLS_DIRNAME` and `skill_trust_store`'s private `_TRUST_DIRNAME` constants. Rewrote both to derive `skills_dir` from `LocalSkillsService(store_dir=tmp_path).skills_dir` (a real, side-effect-free constructor call solely to read its computed attribute) and `trust_dir` from the already-public `default_trust_store_dir(tmp_path)` -- the exact function `app.py` itself calls.

**AC #4 verified concretely, not just asserted (and reverted cleanly afterward -- `git diff` empty on every production file touched):**
1. MCP-store test: temporarily changed `unified_control_plane_service.py`'s `permission_store` property to nest the file under an extra subdirectory. The updated test correctly **FAILED** (`is_sensitive_path` no longer covers the new location).
2. Skills fixture: temporarily renamed `local_skills_service.py`'s `_SKILLS_DIRNAME` and `skill_trust_store.py`'s `_TRUST_DIRNAME`. The NEW fixture code still passed all 52 tests (correctly re-derives regardless of the constant's name). Then reverted ONLY the test file back to the old re-spelled-literal style (keeping the renamed production constants) -- reproduced **3 real failures**, proving the old style really would have gone silently stale.

**Files:** `Tests/Utils/test_sensitive_paths.py`, `Tests/conftest.py`, `Tests/Skills/test_skills_library_flow.py`.

---

## Test commands run (all foreground, exact commands and results)

```
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/wt-path-accessors /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/MCP/test_server_media_db_path.py -v
  -> 4 passed

PYTHONPATH=... python -m pytest Tests/MCP/test_store_default_paths.py -v
  -> 6 passed

PYTHONPATH=... python -m pytest Tests/MCP/test_local_store.py Tests/MCP/test_unified_context_store.py Tests/MCP/test_server_target_store.py Tests/MCP/test_server_target_store_lane_b.py Tests/MCP/test_control_plane_lifecycle.py Tests/MCP/test_control_plane_permissions.py -q
  -> 77 passed

PYTHONPATH=... python -m pytest Tests/Event_Handlers/test_eval_db_operations_path.py Tests/Evals/test_eval_orchestrator_db_path.py -v
  -> 11 passed

PYTHONPATH=... python -m pytest Tests/UI/test_tools_settings_window.py -k "database_path or evals_db_path or rag_indexing_db_path or backup_then_restore" -v
  -> 9 passed

PYTHONPATH=... python -m pytest Tests/UI/test_tools_settings_window.py -q
  -> 6 failed (pre-existing test_chat_api_key_* baseline, confirmed identical via git stash), 25 passed, 16 skipped

PYTHONPATH=... python -m pytest Tests/Utils/test_sensitive_paths.py -v
  -> 31 passed

PYTHONPATH=... python -m pytest Tests/Skills/test_skills_library_flow.py Tests/Library/test_skill_script_grant_panel.py Tests/Skills/test_skill_script_grants.py -q
  -> 52 passed

PYTHONPATH=... python -m pytest Tests/Chatbooks/test_chatbook_importer.py Tests/Chatbooks/test_chatbook_creator.py -q
  -> 28 passed

PYTHONPATH=... python -m pytest Tests/UI/test_chat_screen_ui_state_path.py -v
  -> 3 passed

PYTHONPATH=... python -m pytest Tests/UI/test_chat_screen_worker_groups.py Tests/UI/test_chat_screen_state.py Tests/UI/test_chat_screen_suspend.py Tests/UI/test_chat_screen_context_modal.py Tests/UI/test_chat_screen_ui_state_path.py -q
  -> 22 passed

PYTHONPATH=... python -m pytest Tests/Utils/ -q
  -> 460 passed

PYTHONPATH=... python -m pytest Tests/Skills/ -q
  -> 375 passed

PYTHONPATH=... python -m pytest Tests/MCP/ -q
  -> 351 passed

PYTHONPATH=... python -m pytest Tests/Evals/ -q
  -> 436 passed, 13 skipped (optional-dep-gated)

PYTHONPATH=... python -m pytest Tests/Character_Chat/ Tests/Chatbooks/ -q
  -> 662 passed, 1 skipped

PYTHONPATH=... python -m pytest Tests/Event_Handlers/ -q
  -> 3 failed (Tests/Event_Handlers/test_worker_local_citation_capture.py -- confirmed pre-existing/unrelated via git stash, identical 3 failures with my changes removed), 47 passed, 1 skipped

PYTHONPATH=... python -m pytest Tests/RAG_Admin/ Tests/Subscriptions/ -q
  -> 139 passed

PYTHONPATH=... python -m pytest Tests/UI/test_settings_theme_editor.py Tests/Notes/test_library_notes_sync_integration.py -q
  -> 8 passed

PYTHONPATH=... python -m pytest Tests/RAG/test_fusion.py Tests/RAG/test_scope_pipeline_enforcement.py Tests/RAG/test_local_citation_capture.py Tests/RAG/test_semantic_honest_states.py -q
  -> 275 passed
```

No test run modified real user config/data -- every ad-hoc probe redirected `HOME`/`TLDW_CONFIG_PATH` to a scratch directory first, and one accidental unsandboxed probe (checking `DEFAULT_CONFIG_FROM_TOML` right after the `config.py` edit) only performed an idempotent `mkdir(exist_ok=True)` on a directory that already existed since 2025-08-14 -- confirmed via `stat`, no new file/directory created.
