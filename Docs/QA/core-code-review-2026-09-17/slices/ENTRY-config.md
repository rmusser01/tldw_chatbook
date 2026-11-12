# ENTRY-config — tldw_chatbook/config.py, Constants.py, model_capabilities.py, Logging_Config.py, emergency_stop.py, runtime_policy/ (16 files), 19,413 lines

Worktree `/Users/macbook-dev/Documents/GitHub/tldw-review` @ d8fb4053f9. All `$PY` runs used `source <SCRATCH>/env.sh` with `TLDW_CONFIG_PATH` pointed at scratch configs under `<SCRATCH>/entry-config/` (never the shared scratch profile). Ruff `--select E9,F63,F7,F82` over the whole slice: `All checks passed!`.

## Coverage
| file | lines | read in full / sampled (which ranges) / mechanical only |
|---|---|---|
| config.py | 9,788 | read in full EXCEPT 3572–5708 (`CONFIG_TOML_CONTENT`, one string literal) which was scanned mechanically for `[api_settings.*]`, `[database]`, `api_key`/`api_key_env_var` lines and the `"""` terminator |
| Constants.py | 1,953 | read in full; 195–1621 (`css_content`) and 1632–1935 (three help-text strings) are string literals, skimmed |
| model_capabilities.py | 1,156 | read in full |
| Logging_Config.py | 798 | read in full |
| emergency_stop.py | 104 | read in full |
| runtime_policy/registry.py | 1,421 | read in full; the seed table 257–1318 is declarative data, skimmed |
| runtime_policy/server_context.py | 1,141 | read in full |
| runtime_policy/server_credentials.py | 616 | read in full |
| runtime_policy/bootstrap.py | 484 | read in full |
| runtime_policy/domain_edge_contracts.py | 337 | read in full |
| runtime_policy/server_parity_models.py | 279 | read in full |
| runtime_policy/server_capabilities.py | 240 | read in full |
| runtime_policy/source_state.py | 235 | read in full |
| runtime_policy/__init__.py, types.py, enforcement.py, engine.py, unsupported_capabilities.py, server_parity_state.py, server_event_scope.py, recovery.py | 199/140/119/115/147/75/46/20 | read in full |

Files outside the slice opened only as evidence: `Utils/atomic_file_ops.py` (42–236), `Utils/config_encryption.py` (`_decrypt_config`), `Backup_Recovery/profile_paths.py` (90–225), `Utils/doctor.py` (92–118), `UI/Screens/settings_context_memory.py` (268–318), `UI/Screens/settings_screen.py` (11228–11248), `UI/Screens/logs_screen.py`, `backlog/decisions/012-*.md`, ADR-097/126 grep.

## Findings

### P1 [D1] — `config.get_api_key()` returns placeholder and un-stripped credentials that the shared validity rule rejects, on 5 live spend/readiness paths
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

### P1 [D1] — `get_detected_api_providers()` always returns `[]`; the `doctor` "providers" check always warns "no API providers are configured"
- Where: `tldw_chatbook/config.py:8795-8801` (iterates `config.items()` for keys starting with `"api_settings."` — a flat dotted key that a nested TOML load never produces; the same dead-branch shape `get_api_key`'s own comment at 9054-9063 says was fixed there). Consumer: `Utils/doctor.py:99-110`.
- Evidence: same probe as above, config had a real `[api_settings.anthropic] api_key` → `get_detected_api_providers()    : []`. `rg -n "get_detected_api_providers" tldw_chatbook Tests` → callers are `Utils/doctor.py:101` and `UI/Tools_Settings_Window.py` (DEPRECATED, nav-unreachable per CLAUDE.md); zero tests.
- Why it matters: every user running `doctor` is told to "add a key under [api_settings.<provider>]" regardless of what they have configured; the helper also uses a third placeholder rule (`startswith("<") and endswith(">")`) distinct from `PROVIDER_API_KEY_PLACEHOLDERS`.
- Recommended correction: iterate `config.get("api_settings", {})` and accept a provider when `resolve_provider_api_key(table.get("api_key"))` is truthy; add one test with a nested config.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P2 [D1] — `RichLogHandler.emit` silently drops DEBUG/INFO records emitted from worker threads; only WARNING+ leak to stderr (invisible under a TUI)
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

### P2 [D1] — server-port residue in `_load_settings_uncached`: a malformed `SINGLE_USER_FIXED_ID` env var aborts `import tldw_chatbook.config`; 8 settings keys nobody reads; 5 dead expression statements
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

### P2 [D4a] — two contradictory, both-pinned credential-precedence decisions: chat spend is modern-config-first, `get_api_key` is env-first, and the shipped defaults make every bridged table carry an `api_key_env_var`
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

### P2 [D3] — `import tldw_chatbook.config` performs filesystem work beyond the ADR-126 `admit_startup()`: creates the config file, secures directories, takes the data-root lock, reads a packaged resource, probes optional packages; five module-level aliases it computes are dead
- Where: `config.py:412` (`installation_client_id()`), `845` (`load_openai_mappings()` → `importlib.resources` read), `5717-5719` (`_default_stt_provider_for_platform()` → `find_spec` ×2), `5729` (`tomllib.loads` of the 2,100-line template), `9742-9745` (`load_cli_config_and_ensure_existence()` + `settings = load_settings()` → `create_private_text`, `secure_private_directory`, `_default_data_root_lock`, `chat_dicts_folder.mkdir` at 3423), dead aliases `9747-9759` (`default_api_endpoint`), `9767-9771` (`APP_CONFIG`, `DATABASE_CONFIG`, `RAG_SEARCH_CONFIG`), `9784` (`APP_CONFIG_GLOBAL`).
- Evidence: fresh dir, `TLDW_CONFIG_PATH=$D/cfgdir/config.toml $PY -c "import tldw_chatbook.config as c; print(c._CONFIG_GENERATION, c.first_profile_created_this_session())"` → `generation 1 first_profile True`; `find $D` before: `cfgdir` only; after: `cfgdir/config.toml`. `rg -n "\bAPP_CONFIG_GLOBAL\b|\bconfig\.APP_CONFIG\b|import APP_CONFIG\b|RAG_SEARCH_CONFIG\b|DATABASE_CONFIG\b|default_api_endpoint" tldw_chatbook Tests --glob '!config.py'` → no importer of the five aliases (the one `default_api_endpoint` hit is a QA test's own attribute). `tldw_chatbook.config.settings` IS a de-facto API: monkeypatched in `Tests/TTS/test_stts_settings_reconfiguration.py` (×8) and imported by `Tests/test_config_stt_provider_probe.py:65`.
- Why it matters: this is why every `python -c "import tldw_chatbook..."` in this repo needs an isolated HOME (the brief's `env.sh`), why `Tests/RuntimePolicy` collection once broke on an import cycle (config.py:976-989), and why the lazily-created settings locks (finding below) happen to be safe. No ADR names it: `Tests/Packaging/test_config_import_closure.py::test_config_import_stays_out_of_feature_packages` pins the import CLOSURE, ADR-097 pins module COUNTS, ADR-126 governs admission — none pins "config is loaded and written at import".
- Recommended correction: (a) delete the five dead aliases now (S); (b) record the import-time load as a decision (keep it — 8 test files and `_current_settings_view` depend on the warm `settings` object) or move the two module-scope loads behind the first `load_settings()` call — either way an ADR, not a drive-by.
- Size: S for (a), L for (b) · ADR: new · Confidence: verified
- Pinning test: `Tests/Packaging/test_config_import_closure.py` (closure only); `Tests/test_config_stt_provider_probe.py:65` (relies on `settings` existing at import)
- Already covered: none

### P3 [D2] — `ModelCapabilities(section)` is constructed (and every provider regex recompiled) on each Settings context-window resolution; `_compile_patterns` itself is not the problem
- Where: `UI/Screens/settings_context_memory.py:278` and `:312` (`_capability_or_table_window`), reached from `UI/Screens/settings_screen.py:11241` (`_provider_model_context_window`). The excerpt row `model_capabilities.py:953 _compile_patterns` is called exactly once per instance from `__init__` (`model_capabilities.py:930`); the global instance is a lazy singleton (`1107-1117`) with `lru_cache` on `is_vision_capable`.
- Evidence: `rg -n "ModelCapabilities\(" tldw_chatbook` → three sites: the singleton (1116) and the two per-call constructions above; `rg -n "resolve_model_context_window\(" tldw_chatbook` → one caller in settings_screen (a Settings-screen resolve, not a per-send or compose path).
- Why it matters: ~30 `re.compile` calls per Settings resolve; bounded and off the send path, hence P3.
- Recommended correction: memoise the per-section instance on `(current_config_identity(), id-or-hash of the staged section)` or accept the cost with a comment; do not touch `_compile_patterns`.
- Size: S · ADR: no · Confidence: verified (call graph), cost inferred (not timed)
- Pinning test: none
- Already covered: none

### P3 [D4a] — `emergency_stop._write` hand-rolls the atomic write that `Utils/atomic_file_ops.atomic_write_json` already provides (8 importers), and skips the `fsync` the helper does
- Where: `tldw_chatbook/emergency_stop.py:84-98` (mkdir + `mkstemp` + write + `os.replace`, no `fsync`); helper `Utils/atomic_file_ops.py:199-236` (`atomic_write_json` → `atomic_write_text`: mkdir, mkstemp, `os.fsync`, chmod, `os.replace`).
- Evidence: `rg -l atomic_file_ops tldw_chatbook --glob '!Third_Party/**' | wc -l` → 8; `sed -n '80,138p' Utils/atomic_file_ops.py | rg "mkdir|fsync|chmod|replace"` → all four present. `Tests/Chat/test_emergency_stop.py` (7 tests) pins round-trip/fail-safe semantics, not the write mechanics.
- Why it matters: the module's own docstring promises "survives a restart"; without `fsync` a power loss between `replace` and the journal flush can lose the stop. `mkstemp`'s 0600 is actually tighter than the helper's default — pass `mode=0o600`.
- Recommended correction: `atomic_write_json(path, {"active": ..., "reason": ...}, mode=0o600, indent=None)`; delete `_write`.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/Chat/test_emergency_stop.py::test_set_and_read_round_trip_survives_a_reread` (behaviour, stays green)
- Already covered: none

### P3 [D4b] — `_datetime_to_iso` is copied verbatim into 5 modules; no shared time-format helper exists
- Where: `runtime_policy/source_state.py:133`, `MCP/unified_context_store.py:78`, `MCP/server_target_store.py:372`, `MCP/local_store.py:140`, `MCP/unified_control_models.py:15` (all 6 lines identical, checked by `awk '/^def _datetime_to_iso/,/^$/'` over the five files).
- Evidence: `ls tldw_chatbook/Utils | rg -i "time|date|clock"` → only `tiktoken_runtime.py`; `rg -n "def .*(to_iso|isoformat_utc|utc_iso|iso_timestamp|_utc_now)" tldw_chatbook/Utils tldw_chatbook/DB/base_db.py` → none.
- Why it matters: no drift today (verbatim), but the same "Z-suffixed UTC isoformat" rule is what the `_utc_now ×23 / 7 formats` seed in the review prompt is about; a sixth copy will drift.
- Recommended correction: one `Utils/time_format.py` (`utc_iso_z(dt)`), imported by the five; coordinate with whichever reviewer owns the `_utc_now` seed so both land in the same module.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none (seed-level, cross-slice)

### P3 [D4a] — `get_tts_profiles_db_path` re-rolls the TTS-specific path rule that `Backup_Recovery/profile_paths.database_path` already owns, bypassing the `_get_custom_database_path` helper its 13 sibling getters use
- Where: `config.py:9203-9215` (own `".." in parts` check + `validate_path_simple(...).resolve()`); rule's owner `Backup_Recovery/profile_paths.py:213-219` (same `..` check, same `.resolve()` special case); shared helper `config.py:9151-9178` used by `get_chachanotes_db_path` … `get_scheduled_tasks_db_path`.
- Evidence: `rg -n "tts_profiles_db_path" Backup_Recovery/profile_paths.py` → 161 (row), 213, 217 (special case); template `[database]` at config.py 3967-3991 ships `tts_profiles_db_path` only as a commented-out override, so the profile-default branch is what runs today (cross-profile-leak hypothesis retired, see below).
- Why it matters: three homes for one rule; the config.py copy also skips `custom_database_input`'s "equals shipped default → not custom" screen.
- Recommended correction: `return _get_custom_database_path("tts_profiles_db_path") or get_user_data_dir() / profile_paths.database_leaf(...)` and let `profile_paths` keep the `.resolve()` special case (it already does).
- Size: S · ADR: no (ADR-029 `local-private-data-boundary` is respected by the shared helper) · Confidence: verified
- Pinning test: `Tests/TTS/test_tts_app_ownership.py` (names `get_tts_profiles_db_path` at 122/158/171 as an ownership seam; no test pins `.resolve()`)
- Already covered: none

### P3 [D3] — `config.py` is a 9,788-line god module (≈8,000 executable lines); responsibility clusters
- Where (line ranges): 110–406 Canvas execution/remote-access policy (`CanvasConfigPolicy`, Canvas-only); 408–508 client id, paths, encryption globals; 509–853 default dicts (TTS, RAG, media ingestion, OCR, diarization, OpenAI TTS mappings) + `deep_merge_dicts`; 865–1010 encryption password/decrypt/encrypt; 1010–1310 typed coercion helpers + Console agent budget constants (Console-only); 1310–1746 settings cache locks, tldw_api placeholders, provider-key validity, `provider_settings_for_key`, legacy bridge; 1748–3438 `load_settings` / `_load_settings_uncached` (one 1,690-line function); 3441–3565 `API_MODELS_BY_PROVIDER`/`LOCAL_PROVIDERS` literals — DEAD, reassigned from the template at 9697–9736; 3572–5735 the TOML template + parse; 5737–6090 cache globals, key validation (TASK-26039), schema migration (TASK-26040), corrupt-aside, load-failure record; 6092–6540 bootstrap load + three lock layers + external-edit stamp; 6540–7230 raw write/publish/snapshot/serialized replace/backup/shutdown persist; 7229–7920 mutation primitives (revisioned sections, literal mutations, transactions); 7918–8345 Console trace-capture policy (`RuntimeCapturePolicy`, `TraceRolloutSettings`, `apply_console_capture_settings` — Console-only); 8346–8410 save/delete wrappers; 8410–8720 getters; 8717–8920 provider tables + encryption enable/disable/change; 8920–9530 data root lock, `get_api_key`, 15 DB-path getters, notes-sync knobs, log path, model cache; 9530–9690 DB singletons + lazy getters + seeding; 9690–9788 import-time load + dead aliases.
- Evidence: symbol map `grep -n "^def \|^class \|^[A-Z_]* = " config.py` (≈300 top-level symbols); `API_MODELS_BY_PROVIDER` assigned at 3443 and again at 9697 (`rg -n "^API_MODELS_BY_PROVIDER" config.py`).
- Why it matters: the Canvas (110–406), Console-budget (1072–1198) and Console-trace (7918–8345) clusters are feature policy that only lives here because config.py cannot import feature packages (its own 809–817 comment); they are the three cleanest first PRs.
- Recommended correction: per `backlog/docs/library-decomposition-recipe.md` §1 (per-subsystem PR series) and §17 (size governance): extract Canvas policy → `Canvas/config_policy.py`, Console trace policy → `Chat/console_capture_policy.py`, the encryption trio → `Utils/config_encryption_ops.py`; delete the dead 3443–3565 literals first.
- Size: L · ADR: no existing (cite the recipe, not a redesign) · Confidence: verified (ranges), inferred (extraction safety — the config-import-closure test at `Tests/Packaging/test_config_import_closure.py` will fence it)
- Pinning test: `Tests/Packaging/test_config_import_closure.py::test_config_import_stays_out_of_feature_packages`
- Already covered: none (task-1378 / task-31202 govern settings_screen, not config.py)

### P3 [D3] — dead `css_content` (1,427-line string) in Constants.py, documented dead in its own body
- Where: `tldw_chatbook/Constants.py:195-1621`; the note at 1415-1418 says production loads `css/tldw_cli_modular.tcss`.
- Evidence: `rg -n "css_content" tldw_chatbook Tests --glob '!Constants.py'` → one comment in `Widgets/AppFooterStatus.py:113` saying NOT to use it, and unrelated local variables in `Tests/UI/test_focus_accessibility.py`.
- Why it matters: 73% of Constants.py is a stale CSS copy that a reader has to rule out.
- Recommended correction: delete 195–1621.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P3 [D3] — small consistency defects (six, one line each)
- `config.py:1335,1775,1837` — `import threading` inside three function bodies while `import threading as _threading` sits at module scope (6316); `_SETTINGS_CACHE_LOCK`/`_SETTINGS_REBUILD_LOCK` are created lazily although the comment at 6313-6315 explains why lazy lock creation is unsafe — safe today only because the import-time `load_settings()` (finding above) creates them single-threaded. Size S.
- `config.py:6034` vs `7212` — two backup-filename timestamp formats in one module (`%Y%m%dT%H%M%S` UTC for `.corrupt-*`, `%Y%m%d_%H%M%S` local for `config_backup_*`). Size S.
- `config.py:2167` — inner closure `get_api_key(toml_key, env_var, section)` shadows the module-level `get_api_key(api_name)` (9036). Size S.
- `runtime_policy/bootstrap.py:30` and `:118` — `_VALID_RUNTIME_SOURCES` defined twice. Size S.
- `runtime_policy/server_credentials.py:480-483` — production `_delete_index_record` branches on a `values` dict attribute that only the test fakes have (`Tests/RuntimePolicy/test_server_credentials.py:27`, `test_server_credentials_lane_a.py:17`). Size S (delete the branch; the fakes implement `delete_password`).
- `Logging_Config.py:218` and `:229` — `_harden_existing_generations(selected)` runs twice in `PrivateRotatingFileHandler.__init__`; `Constants.py:1937/1946` — one comment block interleaved with another. Size S.
- Evidence: read only (line numbers above); the keyring one confirmed by `rg -n "self\.values\b" Tests/RuntimePolicy`.
- ADR: no · Confidence: verified (read) · Pinning test: none · Already covered: none

## Candidate dispositions
| candidate (file:line pattern) | confirmed / retired (why) / unverified (check) |
|---|---|
| dotted_section_setting config.py:8454 | retired — proven working: 3-arg dotted → `'/tmp/probe-images'`, 2-arg dotted+default → `True`, 1-arg dotted → value, missing → default/`None`, flat control OK (`probe_get_cli_setting.py`); pinned by `Tests/Utils/test_config_nested_settings.py` |
| dup_shape/dup_verbatim `_datetime_to_iso` ×5 | confirmed — P3 [D4b] above (verbatim, no helper) |
| dup_shape clear_cache@model_capabilities.py:1091 / close@server_parity_state.py:49 / close@private_paths.py:692 | retired — three unrelated two-line delegations (cache clear, two repo closes, stream+lease close); no shared abstraction exists or should |
| except_exception_pass Logging_Config.py:136 | retired — cleanup drain inside a `CancelledError` handler on a log queue; not a data path |
| except_exception_pass Logging_Config.py:461 | retired — `persist_event` diagnostics must not fail the sink install (comment 462); not a data path |
| except_exception_pass Logging_Config.py:654 | retired — closing handlers being removed; not a data path |
| except_exception_return model_capabilities.py:853 | retired — logs a warning and returns the documented default; a budget gate that must not crash a call |
| except_exception_return server_context.py:714 | retired — `_character_authority_capture_matches` proof fails CLOSED (returns False → identity unavailable); the safe direction |
| function_body_import config.py (18) | confirmed in part — 3× `import threading` (P3 misc above); the other 15 are documented cycle-avoidance lazy imports of `Chat.*`/`Canvas.*`/`Backup_Recovery.*` (config is the dependency root, comment 976-989) — retired |
| function_body_import Logging_Config.py:662, emergency_stop.py:102, model_capabilities.py:845/877, bootstrap.py:50/195/275, enforcement.py:71, server_context.py:72/913/948, server_credentials.py:387/409, types.py:75/82/88 | retired — every one is a cycle-avoidance or startup-cost import with a comment naming the reason (task-285 for `tldw_api`, config cycle for the rest); `keyring` import is deferred keyring backend discovery (TASK-21111(b)) |
| legacy_markers_per_file (config 39, server_context 15, server_credentials 11, model_capabilities 7, Logging_Config 2) | examined mechanically — all are task-id references in comments/docstrings explaining a decision; no dead legacy code path found behind them beyond the server-port residue reported above |
| loguru_and_logging Logging_Config.py | retired — the one legitimate bridge (`_forward_loguru_to_standard` at 467; `diagnose=False` at 628 is security-load-bearing per task-2119) |
| os_replace_no_atomic emergency_stop.py:92 | confirmed — P3 [D4a] above |
| raw_1024x1024 config.py:1077,8076,8081,8106,9228 | retired — size constants (`MAX_..._BYTES = 1024 * 1024`, `64 * 1024 * 1024`); literal arithmetic, not a helper re-roll |
| raw_mkdir config.py:3423, 9521 | retired — both wrapped in `_config_participants.operation(...)` (ADR-126 admission) and target application-secured dirs |
| raw_mkdir emergency_stop.py:86 | confirmed as part of the P3 [D4a] swap (the helper does the mkdir) |
| re_compile_in_def model_capabilities.py:953 | retired as stated — called once per instance from `__init__:930`; global instance lazy singleton; the real cost is the per-call construction in `settings_context_memory.py:278/312` (P3 [D2] above) |
| seed_name `_datetime_to_iso` source_state.py:133 | confirmed — see D4b |
| strftime config.py:6034 / 7212 | confirmed as P3 misc (two formats, one module) |
| tempfile_no_secure emergency_stop.py:88 | retired — `tempfile.mkstemp` creates 0600 in the target dir; that IS the secure primitive |
| try_import_guard config.py:6134 | retired — not an import guard; it is the bootstrap `try` around the TOML read (`FileNotFoundError`/`PrivatePathError`/`RecoveryRequired`/`TOMLDecodeError`) |
| try_import_guard config.py:9545, 9551 | retired — deliberate "bundled content cannot prevent boot" seeding (`# noqa: BLE001` comments); logs exception category |
| try_import_guard model_capabilities.py:876 | retired — `LLM_Provider_Catalog/models_dev_catalog.py` exists and exports `models_dev_entry` (line 239); the guard is the documented TASK-26023 gap-fill |

## Verified-fine
- `get_cli_setting` dotted forms (brief's NOTE re-confirmed by execution, output above).
- `Logging_Config.py` loguru↔logging bridge: `loguru_logger.remove()` + single forwarding sink with `diagnose=False`; stdlib root handlers rebuilt after; the `print(..., file=sys.stderr)` lines are gated by `startup_stderr_is_quiet()`.
- `_load_cli_config_bootstrap` lock-free fast path (config.py 6423-6502): the identity re-check argument in its docstring holds because every publisher installs a brand-new dict (`deep_merge_dicts`/`deepcopy`); `_install_bootstrap_cache_from_raw`'s implicit pre-clear coupling is documented at 6666-6674.
- `registry.py:1421 validate_registry_completeness()` at import — pure validation that raises on registry inconsistency; fail-fast by design.
- `bootstrap.default_runtime_policy_path()` resolves per call (task-701 cross-profile leak fix) — not a frozen module constant.
- `_CONFIG_PERSISTENCE_ERROR` — set at 6724 and read by Backup_Recovery participants/tests (`Tests/Backup_Recovery/test_context_policy_config_lifetime.py:42,46,100`), not dead.
- Credential precedence for the 9 bridged providers is applied uniformly through ONE call site (`config.py:2181-2199`, dict comprehension over `_LEGACY_PROVIDER_API_KEY_BRIDGE`) and ONE validity rule (`resolve_provider_api_key`, 15 importers incl. `Chat/provider_readiness.py`, `LLM_Calls/{LLM_API_Calls,moonshot,qwencloud,zai}.py`). Non-bridged chat providers (moonshot/qwencloud/zai) validate via the same function inside their handlers; their env-vs-config ORDER is handler-local (LLM_Calls slice, not examined here).
- `emergency_stop.read_emergency_stop` fail-safe branches pinned by `test_corrupt_file_fails_SAFE_to_stopped` / `test_unreadable_path_fails_SAFE_to_stopped`.
- `server_credentials._RECOVERY_SCOPE_LOCK` (module-level `RLock`) guards keyring index read-modify-write, not SQL — the brief's "module lock + raw SQL" smell does not apply.

## Retired
- **TTS DB path cross-profile leak hypothesis** (raised while reading `get_tts_profiles_db_path`'s raw `custom_path` truthiness): retired — the template `[database]` block (config.py 3967-3991) ships `tts_profiles_db_path` only as a commented-out override, so `get_cli_setting(...)` returns `None` on every unmodified profile and the profile-default branch runs. Symptom impossible today; the re-roll itself stays as P3 [D4a].
- **`disable_config_encryption`/`change_encryption_password` write `enc:` values without a verifier** (config.py 8859-8861, 8903-8908 use the non-strict `decrypt_config_section`): retired — `Utils/config_encryption._decrypt_config` (non-strict) only leaves a value encrypted when it ALREADY failed to decrypt under the verified password (warning at line 238), so no new loss is introduced; the pre-existing warning is the only signal. Not a finding.
- **Lazy `_SETTINGS_CACHE_LOCK` creation race**: retired as a live race — created single-threaded by the import-time `load_settings()`; recorded as P3 consistency only.
- **`load_openai_mappings()` swallowing everything** (config.py 841-843): retired — falls back to the built-in mapping with an INFO line; TTS voice aliases, not user data.

## Left UNVERIFIED
| claim | why not verified | literal command to run |
|---|---|---|
| Console realtime actually puts the `get_api_key("openai")` value on the wire (finding 1's user-visible consequence) | code path read to `realtime.py:616` (returns it as the credential); no live provider send in a read-only review | `.claude/skills/verify/SKILL.md` recipe: `tmux -L verify` launch with `[api_settings.openai] api_key = "YOUR_KEY"`, open Console ▸ realtime, `capture-pane` the auth error |
| Worker-thread INFO drop shows in the real Logs screen (finding 3 in situ) | proven at the handler level only; the app was not run | `tmux -L verify` launch, trigger any `@work(thread=True)` job (e.g. an ingest), open Logs (F3), `capture-pane` and compare against the file log at `get_cli_log_file_path()` |
| `get_model_cache_dir` (config.py 9506-9512) accepts a config-sourced relative/`..` path with no `validate_path_simple` while every DB getter validates | read only; a config value is inside the user's trust boundary so severity is at most P3 | `printf '[embedding_config]\nmodel_cache_dir = "../../escape"\n' > $CFG; TLDW_CONFIG_PATH=$CFG $PY -c "from tldw_chatbook.config import get_model_cache_dir; print(get_model_cache_dir())"` |
| Extraction of the Canvas / Console-trace clusters out of config.py is cycle-safe (god-module finding) | not attempted (read-only) | `cd $WT && $PY -m pytest Tests/Packaging/test_config_import_closure.py -q` after the move, plus `$PY -m pytest Tests/RuntimePolicy --collect-only -q` (the collection that broke last time) |
