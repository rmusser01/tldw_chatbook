# UTILS — tldw_chatbook/Utils/ (excluding Splash_Screens/), 64 files, 24,868 lines

Worktree `/Users/macbook-dev/Documents/GitHub/tldw-review` @ d8fb4053f9. All commands below were run with cwd = worktree root and `source <SCRATCH>/env.sh` where Python was involved. `<SCRATCH>` = `/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/5cdb48ca-db0f-47a4-930a-9ef5b33bceed/scratchpad`. Collected-test evidence comes from one `pytest --collect-only -q Tests` run saved at `<SCRATCH>/phase2/utils_collect.txt` (97,863 tests collected in 120.8 s, exit 0). `ruff check --select E9,F63,F7,F82 tldw_chatbook/Utils` → `All checks passed!`.

## Coverage

| file | lines | read in full / sampled (which ranges) / mechanical only |
|---|---|---|
| Utils/Utils.py | 878 | read in full |
| Utils/input_validation.py | 1797 | read in full |
| Utils/path_validation.py | 617 | read in full |
| Utils/log_sanitizer.py | 807 | read in full |
| Utils/private_paths.py | 2016 | read in full |
| Utils/atomic_file_ops.py | 303 | read in full |
| Utils/secure_temp_files.py | 349 | read in full |
| Utils/optional_deps.py | 1541 | read in full |
| Utils/token_counter.py | 621 | read in full |
| Utils/text.py | 162 | read in full |
| Utils/file_extraction.py | 943 | read in full |
| Utils/egress.py | 1216 | read in full |
| Utils/fd_protection.py | 209 | read in full |
| Utils/log_widget_manager.py | 72 | read in full |
| Utils/tls_trust.py | 275 | read in full |
| Utils/paths.py | 195 | read in full |
| Utils/widget_helpers.py | 325 | read in full |
| Utils/ui_helpers.py | 165 | read in full |
| Utils/pagination.py | 222 | read in full |
| Utils/file_handlers.py | 672 | read in full |
| Utils/sensitive_paths.py | 1218 | read in full |
| Utils/app_shutdown.py | 554 | read in full |
| Utils/adaptive_reader_state.py | 537 | read in full |
| Utils/custom_tokenizers.py | 506 | read in full |
| Utils/windows_files.py | 1117 | read in full |
| Utils/github_api_client.py | 803 | read in full |
| Utils/boot_worker_policy.py | 374 | read in full |
| Utils/fts5_match_forms.py | 391 | read in full |
| Utils/config_encryption.py | 329 | read in full |
| Utils/terminal_utils.py | 330 | read in full |
| Utils/note_importers.py | 339 | read in full |
| Utils/persistent_diagnostics.py | 339 | read in full |
| Utils/mosaic_render.py | 306 | read in full |
| Utils/ui_responsiveness.py | 290 | read in full |
| Utils/textual_css_fastpath.py | 277 | read in full |
| Utils/Splash.py | 266 | read in full |
| Utils/sensitive_llm_logging.py | 247 | read in full |
| Utils/db_status_manager.py | 227 | read in full |
| Utils/text_wrap_index.py | 227 | read in full |
| Utils/doctor.py | 208 | read in full |
| Utils/local_stt_providers.py | 158 | read in full |
| Utils/tiktoken_runtime.py | 164 | read in full |
| Utils/reasoning_config.py | 162 | read in full |
| Utils/Emoji_Handling.py | 155 | read in full |
| Utils/sensitive_config_keys.py | 146 | read in full |
| Utils/library_rail_width.py | 144 | read in full |
| Utils/text_selection_crash_guard.py | 138 | read in full |
| Utils/debug_helpers.py | 136 | read in full |
| Utils/instance_lock.py | 120 | read in full |
| Utils/install_clipboard.py | 109 | read in full |
| Utils/markdown_parsing.py | 103 | read in full |
| Utils/ui_responsiveness_artifacts.py | 94 | read in full |
| Utils/console_background_effects.py | 85 | read in full |
| Utils/db_upgrade_notice.py | 85 | read in full |
| Utils/filesystem_identity.py | 82 | read in full |
| Utils/ingestion_preferences.py | 81 | read in full |
| Utils/cost_estimation.py | 66 | read in full |
| Utils/NotificationHelper.py | 62 | read in full |
| Utils/startup_logging.py | 57 | read in full |
| Utils/about_text.py | 52 | read in full |
| Utils/splash_animations.py | 26 | read in full |
| Utils/platform_files.py | 15 | read in full |
| Utils/Splash_Strings.py | 358 | sampled: lines 1-60 and 190-210 (a single `splashscreen_messages` list + one `random` import; 2 top-level symbols by `rg '^[A-Za-z_]+ *=\|^def '`) |
| Utils/__init__.py | 0 | empty |

Evidence files consulted (not reviewed): `Tests/Utils/test_egress_adoption_census.py` (docstring), `Tests/Utils/test_db_status_manager.py:20-23`, `Tests/Utils/test_config_import_hygiene.py:341-416`, `tldw_chatbook/Logging_Config.py:318-352, 560-720`, `backlog/decisions/079-network-tls-trust-policy.md` (grep only), the seven non-Utils `os.replace` sites, the 16 byte-size formatter bodies, the 14 filename-sanitizer bodies, and 5 `_coerce_bool` bodies (all quoted below where load-bearing).

## Findings

### P1 [D1] — `doctor` reports every optional feature group as "not installed" under the shipped (lazy) dependency mode, including groups that are installed
- Where: `tldw_chatbook/Utils/doctor.py:59-75` (`check_optional_dependencies` reads `DEPENDENCIES_AVAILABLE` raw when no `available` is injected) + `tldw_chatbook/Utils/optional_deps.py:1480-1541` (registry starts all-False; populated only by `initialize_dependency_checks`, which nothing on the doctor path calls).
- Evidence: `$PY -c "from tldw_chatbook.Utils.doctor import check_optional_dependencies as c; from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE as D; r=c(); print(r.status, r.detail[:160]); import importlib.util as u; print(u.find_spec('torch') is not None, D['torch'])"` → `warn 70 optional feature group(s) not installed: PIL, aiohttp, audio_processing, av, beautifulsoup4, chatterbox, chinese_chunking, chromadb, chunker, cohere, defused…` then `True False` (torch is installed; flag is False). Same for numpy.
- Why it matters: the aggregate health surface (TASK-25906) tells a user with a full install that 70 groups are missing and to `pip install` them — a wrong answer a user acts on.
- Recommended correction: in `check_optional_dependencies`, call `optional_deps.ensure_dependencies_checked()` (or the `find_spec`-only probes) before reading the registry; or pass a `find_spec`-derived mapping. The root defect is the registry contract itself, already tracked.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/Utils/test_doctor.py` exists; doctor.py's own docstring says checks are "pure; inject the input for testability", so the injected path is what it pins (whether it also covers the no-argument path: not read — see UNVERIFIED).
- Already covered: task-25704 / task-287 (DEPENDENCIES_AVAILABLE flags never populated). This finding is the doctor-visible consequence of that task; cite, do not re-file.

### P2 [D4b] — 16 byte-size formatters, no shared public helper, drifting user-visible output
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

### P2 [D4b] — Seven filename sanitizers / validators with different safety envelopes; two reach disk unvalidated
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

### P2 [D4a] — `truncate_content` (0 importers) beside 24 inline `[:n] + "..."` truncations and ~20 private `_truncate*/_ellipsize` helpers with drifting ellipsis glyphs
- Where: helper `Utils/Utils.py:253`; inline sites listed in `<SCRATCH>/candidates/patterns/inline_truncate.tsv` (24 rows: `Chat/console_fleet_wake.py:132`, `Chat/provider_failures.py:53`, `MCP/server.py:762`, `MCP/tools.py:252`, `RAG_Search/simplified/vector_store.py:642,1394`, `Tools/code_audit_tool.py:286,429,487,531`, `Widgets/tool_message_widgets.py:101,177,185,195`, … ); named helpers: `rg -n "^\s*def _?(truncate|elide|shorten|ellips)\w*\("` → 25 definitions (e.g. `Agents/run_hooks.py:193 _truncate`, `Event_Handlers/ingest_utils.py:30 _truncate_text`, `Home/dashboard_state.py:71`, `Widgets/Console/console_style_picker_modal.py:126`, `Chat/console_environment_state.py:529 _ellipsize`, `UI/MCP_Modules/mcp_tools_mode.py:110 _ellipsize`, `Chat/console_fleet_wake.py:88 _truncated`, …).
- Evidence: `rg -n "\btruncate_content\b" tldw_chatbook Helper_Scripts scripts Tests` → only the definition; `rg truncate_content <SCRATCH>/phase2/utils_collect.txt` → 0. Inline TSV shows 20 sites use `"..."` and 4 use `"…"` (`review_selection.py:460`, `console_transcript.py:1112`, `personas_character_editor_widget.py:1691`, `personas_preview_pane.py:267`).
- Why it matters: user-visible strings truncate with two different glyphs (`...` is 3 cells, `…` is 1) inside the same screens; the shared helper that would settle it is dead.
- Recommended correction: adopt — one public `truncate(text, budget, ellipsis="…")` in `Utils/text.py` (or rename `truncate_content`), mechanical swap at the 24 inline sites; leave the cell-width-aware helpers (`truncate_console_row_cells`, `ellipsize_note_title_cells`, `elide_path_middle`) alone — they are a different contract. If nobody will do the sweep, delete `truncate_content` instead of keeping a dead helper.
- Size: M · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P2 [D4a] — `config.coerce_bool_setting` (24 importers) is re-rolled 21 times, and the canonical copy has the strangest vocabulary
- Where: canonical `config.py:1201 coerce_bool_setting` → `_get_typed_value` (`config.py:1023-1027`: `str(value).lower() in ["true","1","t","y","yes"]` — so `"on"` → False and int `2` → False). Copies: `rg -n "^\s*def _?(coerce|to|as|parse|normalize|read)_?bool\w*\("` → 21 (`Character_Chat/world_book_manager.py:48`, `Character_Chat_Lib.py:158`, `Chat_Dictionary_Lib.py:95`, `world_book_import.py:34`, `world_info_processor.py:20`, `Home/home_rail_state.py:24`, `Chat/console_rail_state.py:335`, `Library/library_rail_state.py:42`, `Chunking/engine/option_utils.py:10`, `Chunking/engine/templates.py:46`, `Chunking/lab_state.py:198`, `Video_Generation/config.py:333`, `Image_Generation/config.py:466`, `Chat/message_metadata.py:510`, `MCP/unified_control_models.py:501`, `UI/Screens/settings_appearance_defaults.py:91`, `LLM_Management/snapshot_admission.py:458`, and in this slice `Utils/adaptive_reader_state.py:154` and `Utils/console_background_effects.py:43`).
- Evidence: bodies read for five: Character_Chat maps ints/floats by `!= 0` and accepts `on/off`; `home_rail_state` int `!= 0` + `_TRUE_STRINGS`; `option_utils` falls to `bool(value)` (so `[]`→False, `"maybe"`→`is_truthy`); the two Utils copies return `default` for any non-bool non-str (int `1` → default); config's canonical rejects `on`. Two more vocabularies live in this slice: `egress._config_enabled` (`egress.py:113`: anything not in `false/0/no/off` is True) and `tls_trust._TRUE_STRINGS/_FALSE_STRINGS` (`tls_trust.py:36-37`: `"yes"` is neither, so it is treated as a CA-bundle PATH and logged as an error).
- Why it matters: the same TOML value (`enabled = "on"`, `enabled = 1`) is True in one section and False/default in another; three of the readers are config-persisted preference loaders.
- Recommended correction: a stdlib-only leaf `Utils/coerce.py:coerce_bool(value, default)` with the union vocabulary (`true/1/t/y/yes/on` · `false/0/f/n/no/off` · ints by `!= 0`), `config.coerce_bool_setting` delegating to it (config.py already imports `Utils.adaptive_reader_state`/`console_background_effects` at module top — `config.py:75,81` — so the leaf must not import config, which is the same constraint those two files document).
- Size: M · ADR: no · Confidence: verified (definitions and vocabularies); inferred (which persisted keys are actually read through a divergent copy)
- Pinning test: `Tests/Utils/test_config_nested_settings.py` (config side, not read); none for the copies as a set.
- Already covered: none

### P3 [D3] — `Utils/paths.py` imports a symbol that has never existed in `Utils.Utils`; the fallback branch is the only branch that ever runs, and the four helpers built on it raise
- Where: `Utils/paths.py:15-27` (`from ..Utils.Utils import (PROJECT_DATABASES_DIR, log, PROJECT_ROOT_DIR, CONFIG_FILE_PATH)` → `except ImportError` sets all four to `None`/`logging`); consumers `get_project_databases_dir:47`, `get_project_database_path:68`, `get_project_relative_path:97`, `get_project_root:181`.
- Evidence: `$PY -c "import tldw_chatbook.Utils.paths as p; print(p.PROJECT_DATABASES_DIR, p.log is __import__('logging')); import tldw_chatbook.Utils.Utils as U; print(hasattr(U,'log')); p.get_project_databases_dir()"` → `None True` / `False` / `AttributeError 'NoneType' object has no attribute 'mkdir'`. Callers: `rg -n "get_project_databases_dir|get_project_database_path|get_project_root\b|get_project_relative_path" tldw_chatbook Helper_Scripts scripts --glob '!Tests/**'` → none outside paths.py. The failing import still executes `Utils.Utils` (`'tldw_chatbook.Utils.Utils' in sys.modules` → True after `import tldw_chatbook.Utils.paths`), so all 17 `get_user_data_dir` importers (incl. `tls_trust.py:34`, `config.py:3411`) pull `Utils.Utils` + `secure_temp_files` + `Metrics.metrics_logger` for a branch that always falls through.
- Why it matters: a dead, always-raising API kept alive by a silent `except ImportError`; the same "function-body import of a symbol that no longer exists" class the brief names (mocked tests never catch it).
- Recommended correction: delete lines 15-27, the four `PROJECT_*` helpers, and the `__main__` block; keep `get_user_data_dir`. Then `Utils.Utils` no longer rides in on `paths`.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P3 [D3] — `Utils/Utils.py` re-exports stdlib `logging`; three modules import their logger from it (274 stdlib log calls), and the file mixes loguru + stdlib
- Where: `Utils/Utils.py:35,51`; consumers `LLM_Calls/LLM_API_Calls_Local.py:26` (`from tldw_chatbook.Utils.Utils import logging`, 41 `logging.*` calls), `LLM_Calls/Local_Summarization_Lib.py:33` (229 calls), `Local_Ingestion/XML_Ingestion.py:14` (4 calls).
- Evidence: `rg -n "Utils\.Utils import" tldw_chatbook --glob '!Tests/**'` → the three lines above; per-file `rg -c '\blogging\.'` → 41/229/4, `rg -c 'from loguru import'` → 0 in all three. Redaction is NOT bypassed: `Logging_Config.py:350` applies `redact_log_line` in the root-handler formatter and `configure_application_logging` (`:643-712`) installs those handlers on the stdlib root, which is where these records land.
- Why it matters: `import logging` in Utils.py is load-bearing for three other modules — removing it (the obvious cleanup of the loguru+logging co-import) breaks them at import; and those three modules log through stdlib while the rest of the package logs through loguru.
- Recommended correction: in the three consumers `from loguru import logger` and `s/logging\./logger./`; then drop `import logging` from Utils.py (its own uses at `:114,360,398-439,571-608,870` are in dead functions — see next finding).
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P3 [D3] — `Utils/Utils.py`: 25 of ~32 top-level symbols are dead (0 importers, 0 collected tests); two are test-only
- Where: dead — `ensure_directory_exists:118`, `global_api_endpoints:123`, `global_search_engines:144`, `openai_tts_voices:156`, `format_api_name:171`, `convert_to_seconds:223`, `truncate_content:253`, `save_to_file:357`, `save_segments_to_json:363`, `generate_unique_identifier:464`, `is_valid_url:489`, `verify_checksum:503`, `normalize_title:511`, `clean_youtube_url:543`, `sanitize_user_input:553`, `format_file_path:569`, `safe_float:583`, `safe_int:596`, `save_temp_file:623`, `cleanup_temp_files:637`, `generate_unique_id:643`, `extract_media_id_from_result_string:798`, `get_api_name:852`, plus `UTILS_FILE_PATH/LIBS_DIR/PROJECT_ROOT_DIR/CONFIG_FILENAME/CONFIG_FILE_PATH/PROJECT_DB_DIR_NAME/PROJECT_DATABASES_DIR:67-78` (only reader is the failing import in paths.py). Test-only — `safe_read_file:387`, `FileProcessor:647` (`Tests/Utils/test_config_import_hygiene.py:377-416`, which pins that chardet stays lazy). Live — `extract_text_from_segments`, `generate_unique_filename`, `elide_path_middle`, `fold_path_lines`, `_format_size_bytes`, `get_formatted_file_size` (test-only via `test_db_status_manager.py:22`), `get_formatted_db_size_with_wal`, and the `logging` re-export.
- Evidence: per-symbol `rg -n "\b<sym>\b" tldw_chatbook Helper_Scripts scripts --glob '!Tests/**' -l | grep -v Utils/Utils.py` → 0 for each listed dead symbol (the 4 hits for `save_to_file` are unrelated methods/params in `Models/evaluation_state.py:561`, `Audio/*`; the 1 hit for `is_valid_url` is a nested def in `Web_Scraping/Article_Extractor_Lib.py:1809`; `cleanup_temp_files` hit is a TOML key at `config.py:5290`); `rg` over `utils_collect.txt` → 0 for each.
- Why it matters: `is_valid_url` (the `re_compile_in_def` candidate at `:490`) and `normalize_title` are dead — the D2 candidate is moot; the module is in the boot import closure (`Tests/Performance/boot_budget_snapshots/boot_import_modules.txt:582`) carrying ~650 dead lines, and `secure_temp_files` is imported at module top (`:38`) only to serve two dead functions.
- Recommended correction: delete the dead symbols and the `from .secure_temp_files import get_temp_manager` import; keep the seven live ones.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/Utils/test_config_import_hygiene.py::test_*` states "plain `import tldw_chatbook.Utils.Utils` must not load chardet" as a requirement — deleting `safe_read_file`/`FileProcessor` would need those tests retargeted.
- Already covered: none

### P3 [D3] — Six dead modules in the shared-helper layer (plus one census false negative)
- Where / evidence (each: exact-path `rg` over `tldw_chatbook Helper_Scripts scripts Tests` + `rg <mod> utils_collect.txt`):
  - `Utils/ui_helpers.py` (165) — 0 importers, 0 tests. Its `update_model_select` even documents that its targets "do not exist in the master-shell UI".
  - `Utils/pagination.py` (222) — 0 importers, 0 tests (`rg pagination utils_collect.txt` → 322 hits, all Media/Evals paging tests that do not import this module: `rg -ln "Utils\.pagination|Utils/pagination" Tests` → none).
  - `Utils/cost_estimation.py` (66) — 0 importers, 0 tests; orphan of the removed `Widgets/Evals/cost_estimation_widget.py` (`Tests/UI/test_evals_deletion_guard.py::test_removed_module_file_is_absent[...cost_estimation_widget.py]`).
  - `Utils/debug_helpers.py` (136) — 0 importers, 0 tests (dev snippets returning code strings).
  - `Utils/ingestion_preferences.py` (81) — 0 importers, 0 tests.
  - `Utils/splash_animations.py` (26) — 0 importers, 0 tests (compat shim exporting two regex constants).
  - Census false negative: `Utils/Splash.py` IS live — `Utils/Splash_Screens/card_definitions.py:4 from ..Splash import get_ascii_art` (6 call sites); only `print_tldw_ascii:194` and `get_splash_card_config:245` are dead (`rg` → 0). Keep the module.
  - Also in `Utils/input_validation.py`: `validate_email:1024`, `validate_username:1045`, `validate_ip_address:1093`, `validate_port:1221`, `validate_and_raise:1794`, `validate_filename:1396` — 0 importers (`rg -n "input_validation import[^\n]*(validate_email|validate_username|validate_ip_address|validate_and_raise)\b"` → none; `validate_port` has its own copy at `Event_Handlers/LLM_Management_Events/llm_management_events_vllm.py:104`). The `re_compile_in_def` candidates at `:1034` and `:1061` are therefore dead code.
  - `Utils/db_status_manager.py:187 _get_db_status_widget` — 0 callers (`rg -n _get_db_status_widget tldw_chatbook Tests | grep -v "def "` → none); its `TYPE_CHECKING` import names the retired `AppFooterStatus`.
- Why it matters: ~700 lines of helper surface that new code can "reuse" but that has no consumer and no test; two of them (`ui_helpers`, `ingestion_preferences`) target UI that no longer exists.
- Recommended correction: delete all six modules, the two dead Splash functions, the six dead validators, and `_get_db_status_widget`. (`Widgets/base_components.py` — outside this slice, seeded by the parent — 0 importers; only `Tests/UI/test_widget_css_consolidation.py`/focus-contract censuses list it: same recommendation.)
- Size: S · ADR: no · Confidence: verified
- Pinning test: none (the CSS-consolidation censuses enumerate files, they do not exercise them)
- Already covered: none

### P3 [D3] — `Utils/text.py`: 5 of 7 functions dead, one a verbatim copy of `Utils.py:84`
- Where: `format_metadata_as_text:17`, `format_transcription:63`, `extract_text_from_segments:90` (verbatim twin of `Utils/Utils.py:84`, `dup_verbatim` row), `format_text_with_line_breaks:124`, `format_transcript:132`. Live: `sanitize_filename:47` (2), `slugify:144` (2 — and a second, differently-behaved `slugify` lives at `Local_Ingestion/Book_Ingestion_Lib.py:462` for markdown anchors).
- Evidence: `rg -n "\b(format_metadata_as_text|format_transcription|format_text_with_line_breaks|format_transcript)\b" tldw_chatbook Helper_Scripts scripts Tests` → only the definitions (+ a commented example in Utils.py:220); `extract_text_from_segments` consumers all import the `Utils.Utils` copy (`Local_Summarization_Lib.py:33`).
- Recommended correction: delete the five; `text.py` keeps `sanitize_filename` + `slugify` (or absorbs the truncation/size helpers above and becomes the real text-helper home).
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

### P3 [D4a] — `atomic_file_ops` (8 importers) ignored by three plain temp-write-then-`os.replace` re-rolls, none of which fsync; `emergency_stop` calls itself "durable"
- Where: `Agents/local_tool_provider.py:457-466` (`_write_spill`: mkstemp → write → chmod 0600 → replace, no fsync), `Chat/trajectory_export.py:1313-1319` (mkstemp → write → replace, no fsync, no chmod → export file stays 0600 from mkstemp), `emergency_stop.py:88-92` (`_write`: docstring "Activate the stop durably (atomic write)", no fsync), `MCP/permission_store.py:942-956` (has fsync + `mcp_sources` bookkeeping; partial re-roll). Justified (dir_fd-relative, identity-checked, not expressible with the path-based helper): `Utils/private_paths.py:1156`, `Tools/local_tool_impls.py:649`, `UI/Console_Modules/video.py:700`. Not a temp-write at all: `RAG_Search/config_profiles.py:833` (renames a legacy blob to `.migrated`). In-slice: `private_paths.py:1000` is the Windows-only `operation is None` branch (mkstemp+replace, no fsync — `atomic_write_bytes` would apply 0o644 where this wants 0o600, so a direct swap is wrong; the branch is UNVERIFIED_PLATFORM by design), `tls_trust.py:196-203` (cache file; mkstemp → write → replace; acceptable for a regenerable cache but the `finally: if exists(tmp): unlink` runs after a successful replace on a path that no longer exists — harmless).
- Evidence: `<SCRATCH>/candidates/patterns/os_replace_no_atomic.tsv` (13 rows) + `sed` context of each; `rg -c fsync` per file → local_tool_provider 0, trajectory_export 0, emergency_stop 0; importers `rg -n "atomic_file_ops import" tldw_chatbook --glob '!Tests/**'` → 8 files.
- Why it matters: three files claim atomic/durable writes and get atomic-but-not-durable (a crash after `os.replace` before writeback can leave an empty or torn file on some filesystems); < 10 re-rolls so P3.
- Recommended correction: `atomic_write_text(path, text, mode=0o600)` at local_tool_provider, `atomic_write_json` at trajectory_export and emergency_stop.
- Size: S · ADR: no · Confidence: verified · Pinning test: `Tests/Utils/test_atomic_file_ops.py` (helper only) · Already covered: none

### P3 [D4a] — Four `len(text) // 4` token estimates beside `Utils/token_counter.estimate_tokens` (memoized, CJK-weighted, +20 % headroom)
- Where: `Chat/usage_recorder.py:29-41` (`_CHARS_PER_TOKEN = 4`; feeds `record_exchange` → `record_usage`), `Character_Chat/world_info_processor.py:681`, `Subscriptions/token_manager.py:113` (averaged with a word count), `Widgets/chunk_preview.py:98`. (`TTS/backends/kokoro.py:130,954` are audio-sample counts, not tokens — not copies.)
- Evidence: `rg -n "len\([^)]*\)\s*(//|/)\s*4\b|_chars_estimate|CHARS_PER_TOKEN" tldw_chatbook --glob '!Tests/**' --glob '!Third_Party/**' | grep -v token_counter` → the rows above; `estimate_tokens` importers = 5 (census).
- Why it matters: for the same text the ledger records `len//4` while the context-budget code sees `len*0.25*1.2` (20 % apart; ~4× apart for CJK); if the usage ledger persists these numbers the drift reaches storage (see UNVERIFIED).
- Recommended correction: `from tldw_chatbook.Utils.token_counter import estimate_tokens` at the four sites (usage_recorder keeps its "empty text = 1" rule as a wrapper).
- Size: S · ADR: no · Confidence: verified (copies), inferred (storage reach) · Pinning test: none · Already covered: task-18602 (memoization; not adoption)

### P3 [D3] — `secure_temp_files.py` is security theatre over stdlib guarantees; its "secure delete" has one caller path
- Where: `secure_temp_files.py:57,111,181,273` (`os.chmod(..., 0o600/0o700)` after `NamedTemporaryFile`/`mkstemp`/`mkdtemp`), `:199-240 secure_delete_file` (zero-overwrite then unlink), `:166` (`content_size = len(content) if isinstance(content, str) else len(content)` — both arms identical), `:329-335 __del__` cleanup, `:347 cleanup_all_temp_files`.
- Evidence: `$PY -c "import tempfile,os,stat; ..."` → `NamedTemporaryFile mode=600 mkstemp mode=600 mkdtemp mode=700` (the chmod is a no-op); `rg -n "cleanup_all_temp_files|\.cleanup_all\(\)" tldw_chatbook --glob '!Tests/**' | grep -v secure_temp_files.py | grep -v Utils/Utils.py` → none (the manager only cleans up from `__del__`); importers = 7 (`Article_Extractor_Lib`, `TTS/pcm_playback`, `speech_playback_mixin`, `tts_events`, `Chat_Functions`, `stts_events`, `Utils/Utils.py`).
- Why it matters: 18 raw `tempfile` call sites outside Utils (`tempfile_no_secure.tsv`) are NOT a gap — the helper adds metrics and a zero-overwrite that gives no guarantee on journaling/CoW/SSD storage. Pushing adoption would add cost, not safety.
- Recommended correction: keep; do not adopt further; drop the redundant chmods and the no-op at `:166`; if "secure delete" is meant as a property, say in the docstring that it is best-effort.
- Size: S · ADR: no · Confidence: verified · Pinning test: `Tests/Utils/test_security_enhancements.py` (not read; name suggests it pins the chmod) · Already covered: none

### P3 [D3] — `input_validation.sanitize_string` and `log_sanitizer.sanitize_string` share a name and opposite contracts
- Where: `Utils/input_validation.py:1664` (strip control chars, truncate — an INPUT sanitizer, 23 importers by that name) vs `Utils/log_sanitizer.py:445` (credential/PII redaction — a LOG sanitizer, 0 importers by that name; callers use `redact_log_line`/`redact_user_paths`).
- Evidence: `rg -l 'input_validation import[^\n]*sanitize_string|input_validation\.sanitize_string'` → 23; same for log_sanitizer → 0; files importing both modules: `DB/ChaChaNotes_DB.py`, `UI/Screens/change_review_screen.py`.
- Why it matters: a future `from ...log_sanitizer import sanitize_string` at an input boundary (or vice versa) type-checks and runs — it just does the wrong thing silently.
- Recommended correction: rename the log one to `redact_text` (its callers already use the `redact_*` family).
- Size: S · ADR: no · Confidence: verified · Pinning test: `Tests/Utils/test_log_sanitizer.py` (names `sanitize_string` — would need the rename) · Already covered: none

### P3 [D3] — `custom_tokenizers.py`: hard-coded profile path, import-time warning, non-atomic mapping write
- Where: `:50-52` (`os.path.expanduser("~/.config/tldw_cli/tokenizers")` — bypasses `config.get_user_data_dir()`/`TLDW_CONFIG_PATH`, the exact literal-path failure mode `sensitive_paths.py:254-265` documents), `:25-33` (`logger.warning("tokenizers library not available…")` at import; `token_counter.py:35-43` does the same for tiktoken — every boot without those extras logs a warning), `:100-109 save_mappings` (plain `open(w)` + `json.dump`, not `atomic_write_json`).
- Evidence: read; `rg -n '~/\.config/tldw_cli' tldw_chatbook --glob '!Tests/**'` shows the same literal in `UI/Voice_Cloning_Window.py:300,310`, `TTS/recovery.py:160-161`, `TTS/TTS_Backends.py:299`, `TTS/backends/higgs.py:168` (out of slice; listed for the owning reviewers).
- Recommended correction: resolve the directory from config lazily; downgrade the two import-time warnings to `debug` (or emit them from the first real use); `atomic_write_json` for `mappings.json`.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

### P3 [D4a] — `github_api_client.py` builds its `httpx.AsyncClient` directly and guards only half its fetches
- Where: `:179 _build_client` → `httpx.AsyncClient(headers=…, timeout=30.0)` (not `tls_trust.build_httpx_async_client`, so `[network] ssl_verify` custom-CA is ignored — ADR-079 says new outbound code should use the factories, `079-network-tls-trust-policy.md:67-68`); `:326 get_repository_info`, `:378 get_branches`, `:611 get_rate_limit` use raw `self.client.get` while `:494,:517,:574,:644` use `guarded_fetch_httpx_async`; `:386-388 get_branches` swallows every exception into `["main","master"]` (an auth/rate-limit failure becomes a silent wrong branch list).
- Evidence: read; `rg -n "httpx\.(get|post|Client|AsyncClient)\(" tldw_chatbook/Utils/github_api_client.py` → `:179`.
- Recommended correction: `build_httpx_async_client(headers=…, timeout=30.0)`; route the three raw gets through the guarded helper (fixed host, so SSRF risk is nil — consistency only); let `get_branches` raise `GitHubAPIError` like its siblings.
- Size: S · ADR: yes (079-network-tls-trust-policy.md — long tail explicitly deferred at `:29`; this is one of it) · Confidence: verified · Pinning test: `Tests/Utils/test_github_api_client.py` (not read) · Already covered: none

### P3 [D3] — `Utils/Utils.ensure_directory_exists` (0 importers) vs 22 raw `mkdir(parents=True, exist_ok=True)` sites
- Where: helper `Utils/Utils.py:118` (one-line `os.makedirs` wrapper); raw sites in `<SCRATCH>/candidates/patterns/raw_mkdir.tsv` (30 rows, 8 inside Utils).
- Evidence: `rg -n "\bensure_directory_exists\b" tldw_chatbook Helper_Scripts scripts Tests` → definition only.
- Why it matters: nothing — `Path.mkdir(parents=True, exist_ok=True)` IS the idiom; the hardened case already has a real helper (`private_paths.secure_private_directory`, 32 importers).
- Recommended correction: delete the wrapper; do not adopt.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

### P3 [D1] — 13 HTTP call-site modules use neither `Utils/egress` nor `Utils/tls_trust` (documented decision for egress; ADR-deferred long tail for TLS)
- Where (after removing 4 false positives that were dict `.get(request_id)` calls and one `.md`): `Audio/diarizer_engine_onnx.py:441`, `LLM_Management/snapshot_client.py:88`, `Local_Ingestion/transcription_service.py:3211`, `Research_Interop/academic_providers.py:203,360,529,602`, `Skills_Interop/skill_remote_fetch.py:261` (→ task-609), `tldw_api/client.py:1204`, `TTS/adapters/audio_cpp.py:421,440`, `TTS/backends/kokoro.py:171`, `TTS/utils/download_models.py:92` (`follow_redirects=True` model download), `UI/LLM_Management/vllm_connection.py:614`, `UI/Wizards/first_run_voice_step_state.py:269`, `UI/Wizards/FirstRunSetupWizard.py:925`, `Widgets/Settings_Widgets/server_switch_modal.py:209`, plus in-slice `Utils/github_api_client.py:179` (above). Full list: `<SCRATCH>/phase2/utils_egress_bypass.txt`.
- Evidence: `comm -23 <(rg -l 'httpx\.(get|post|Client|AsyncClient)\(|requests\.(get|post)\(|aiohttp\.ClientSession\(' tldw_chatbook --glob '!Third_Party/**' --glob '!Tests/**' | sort) <(rg -l 'Utils\.egress|Utils import egress|from \.+egress import|Utils\.tls_trust|from \.+tls_trust import' tldw_chatbook --glob '!Third_Party/**' | sort)` → 19 files (42 HTTP sites total; 47 egress importers). `Tests/Utils/test_egress_adoption_census.py` docstring: the census "deliberately does NOT census bare `requests`/`httpx` calls: this app has dozens of legitimate ones to fixed, non-user-supplied API endpoints" — the egress half is a stated decision, and most of these sites talk to user-configured local servers or fixed vendor hosts.
- Why it matters: the TLS half is not covered by that decision — a corporate-CA user (ADR-079's own motivating case) gets verification failures from model downloads (`download_models.py`, `kokoro.py`), the tldw server client, snapshot client, vLLM probe and both first-run probes.
- Recommended correction: none here beyond citing — egress: task-586 / task-609 and the census docstring; TLS: ADR-079 §"long tail threads" — hand the list to the owning slices for `build_httpx_*`/`build_requests_session` adoption.
- Size: M · ADR: yes (079-network-tls-trust-policy.md) · Confidence: verified (list), inferred (that each site is on a TLS-verified path a user can hit)
- Pinning test: `Tests/Utils/test_egress_adoption_census.py` (urllib/yt_dlp only — states the bare-client exemption as a requirement) · Already covered: task-586, task-609

### P3 [D2] — `optional_deps.py` imports `config` at module scope, and adoption of `optional_deps` would not fix the guarded-import cost
- Where: `optional_deps.py:1519-1531` (`from ..config import get_cli_setting` inside a module-level `try`, executed at import unless `PYTEST_CURRENT_TEST`); `check_dependency:629` does `__import__(module_name)` — a real import — so the 11 modules that keep module-scope `try: import X except ImportError` without `optional_deps` (`<SCRATCH>/candidates/patterns/try_import_guard.tsv` filtered: `LLM_Calls/Summarization_General_Lib.py`, `MCP/server.py`, `RAG_Search/parallel_processor.py`, `RAG_Search/simplified/{embeddings_wrapper,enhanced_indexing_helpers,indexing_helpers,rag_service,vector_store}.py`, `Widgets/emoji_picker.py`, `Widgets/Persona_Widgets/actor_pack_import_review.py`, `Widgets/Tamagotchi/tamagotchi_storage.py`) would pay exactly the same cost through `check_dependency`.
- Evidence: read; `rg -l 'Utils\.optional_deps|from \.+optional_deps import' tldw_chatbook --glob '!Tests/**' | wc -l` → 41; `awk` over the TSV → 11 files.
- Recommended correction: none for adoption (keep). The `find_spec` probes (`embeddings_rag_deps_installed`, `parakeet_onnx_deps_installed`, `local_stt_providers.module_installed`) are the cheap pattern; the eager `check_dependency` family is the expensive one — the D2 "guarded-but-not-lazy" concern belongs to whichever slice owns the 11 files, and `optional_deps` is not the fix.
- Size: – · ADR: no · Confidence: verified · Pinning test: `Tests/Utils/test_optional_import_deferral.py` (census of deferral) · Already covered: task-25704 / task-287 (registry), task-21731 (boot import closure)

## Candidate dispositions

| candidate (file:line pattern) | confirmed / retired (why) / unverified (check) |
|---|---|
| dup_shape `clear_cache@model_capabilities.py:1091` / `close@server_parity_state.py:49` / `close@private_paths.py:692` | retired — three-line `try: x.close() except: pass` shapes with unrelated semantics (`_AdmittedStream.close` releases a storage lease); not a duplication |
| dup_shape `cleanup_temp_files@Utils.py:637` / `reload_all_pipelines` / `simple_tamagotchi.main` | retired — two-line delegation shape; `cleanup_temp_files` itself is dead (finding above) |
| dup_verbatim `extract_text_from_segments@Utils.py:88` / `text.py:94` | confirmed — verbatim twin; text.py copy has 0 importers (P3 text.py finding) |
| except_exception_pass app_shutdown.py:322/330/334 | retired — `_report_hard_exit` runs from a daemon thread racing interpreter finalization; every channel is best-effort by documented design |
| except_exception_pass atomic_file_ops.py:130/193/293 | retired — temp-file unlink in the error path of a write that is about to re-raise |
| except_exception_pass doctor.py:159 | retired — `_default_path_postures` is best-effort posture gathering; `run_doctor` isolates failures by design (AC#6) |
| except_exception_pass egress.py:635 | retired — `GuardedResponse.text` tolerating a headers object without `.get`; falls back to utf-8 |
| except_exception_pass fd_protection.py:194/199 | retired — closing wrappers this function created, inside `finally`, after restoring the real streams (task-641 design) |
| except_exception_pass note_importers.py:195 | confirmed (P3, not filed separately) — `datetime.fromisoformat` failure swallowed; should be `except ValueError`; returns None → timestamps silently dropped on import |
| except_exception_pass private_paths.py:702 | retired — `_AdmittedStream.__del__` |
| except_exception_pass secure_temp_files.py:333 | retired — `__del__` during interpreter shutdown, documented |
| except_exception_pass startup_logging.py:54 | retired — loguru may be absent/torn down before the app imports; stdlib capping still applies |
| except_exception_pass terminal_utils.py:94 | retired — introspection-only protocol name for a log line |
| except_exception_pass text_selection_crash_guard.py:131 | retired — clearing `_select_state` on a Textual-internal attribute after a matched crash signature |
| except_exception_return app_shutdown.py:353 | retired — reachable from a signal handler; config failure must degrade to the default (documented) |
| except_exception_return db_upgrade_notice.py:82 | retired — courtesy pre-boot probe, documented "never let the notice break boot" |
| except_exception_return doctor.py:84 | retired — per-check isolation is the module's AC |
| except_exception_return file_extraction.py:917/928 | retired — `toml.loads` failure → validation error string is the contract |
| except_exception_return file_handlers.py:379 | retired — CSV preview error string; user-visible `[Error reading CSV: …]` |
| except_exception_return github_api_client.py:221 | retired — done-callback reading `fut.exception()` on a possibly-cancelled future |
| function_body_import Utils.py:416/656 (chardet) | retired — deliberate lazy import pinned by `test_config_import_hygiene.py` |
| function_body_import atomic_file_ops.py:222 (json) | confirmed P3 (trivial) — `json` is stdlib and cheap; move to module top |
| function_body_import db_status_manager.py:55/72/144/149 | retired — cycle-avoidance (`Backup_Recovery.bootstrap`, `config`) documented at the site |
| function_body_import debug_helpers.py:126 (loguru) | retired — module is dead (delete) |
| function_body_import doctor.py:45/62/101/132 | retired — doctor must run even when config load failed; lazy by design |
| function_body_import egress.py:1033/1060/1133 | retired — `tls_trust` (cycle: tls_trust imports egress-adjacent config), `requests` re-import inside a function that already has it at module top (`:40`) — `:1060` is redundant but harmless (P3 nit), `multidict` optional |
| function_body_import file_extraction.py:549/550/696/697 (csv, io) | confirmed P3 (trivial) — stdlib, hoist |
| function_body_import file_handlers.py:63/112/598 | retired — documented cycle (`attachment_core` imports this module) |
| function_body_import file_handlers.py:358 (csv) | confirmed P3 (trivial) — stdlib, hoist |
| function_body_import github_api_client.py:488/568/636 (`..Utils.egress`) | retired — egress imports `config`; github client is imported by UI before config is fully up; but the same import appears three times — hoist to one module-level import is safe if `config` import order allows; unverified (see table) |
| function_body_import input_validation.py:547/792/843/852/1556/1600 | retired — documented "keep validation imports acyclic" (`:546`), `Terminal.contracts`, `Chat.provider_readiness` |
| function_body_import input_validation.py:1189/1190/1192 (os, pathlib, .path_validation) | confirmed P3 (trivial) — `os`/`pathlib` stdlib; `.path_validation` is a sibling leaf with no cycle |
| function_body_import markdown_parsing.py:100/101 | retired — optional dep, gated by `check_dependency` |
| function_body_import note_importers.py:243 (csv) | confirmed P3 (trivial) — stdlib, hoist |
| function_body_import path_validation.py:19/25 | retired — documented: metrics must not bootstrap during recovery path parsing |
| function_body_import paths.py:128 (`..config`) | retired — documented cycle |
| function_body_import private_paths.py:611/763/802 | retired — dependency-leaf module by design; `storage_admission` imports it |
| function_body_import sensitive_paths.py:351-610 (9) | retired — documented at `:267-272` (config is heavy; most callers never need it) |
| function_body_import terminal_utils.py:59 | retired — `.optional_deps` imports `config` at module scope (P3 finding above) |
| function_body_import textual_css_fastpath.py:119 | retired — `textual.css.model` inside a hot path but module-scope import would be identical cost; used once per rule then cached |
| function_body_import tls_trust.py:129/178 (certifi) | retired — only needed for the custom-CA branch |
| function_body_import token_counter.py:530 | retired — `model_capabilities` is heavy; only on the resolve path |
| function_body_import widget_helpers.py:117-323 (8) | retired — `.optional_deps` (config at import) and `..Library.ingest_capabilities` (Library package init) — both documented heavy |
| function_body_import windows_files.py:212 (msvcrt) / :975 (time) | retired — platform-gated / trivial |
| id_keyed_dict textual_css_fastpath.py:232 | retired — keyed by `id(rule)` with a strong reference to the owning `rules_map` held in the same cache tuple (`:90-97`), so ids cannot be reused while cached |
| legacy_markers (27 rows across 14 files) | retired — all are prose ("legacy", "retired", "deprecated") in docstrings/comments explaining history; the only code-level legacy surface is `Splash.print_tldw_ascii` (dead — filed) and `about_text`'s note about the deprecated ToolsSettingsWindow (live consumer `settings_screen.py`) |
| loguru_and_logging Utils.py:0 | confirmed — filed (P3 D3); load-bearing for 3 importers |
| mutable_class_attr windows_files.py:57-114 (`_fields_` ×8) | retired — `ctypes.Structure._fields_` is the ctypes ABI contract, never mutated |
| os_replace_no_atomic atomic_file_ops.py:110/184/284 | retired — this IS the helper |
| os_replace_no_atomic private_paths.py:1000 | confirmed (P3, in the atomic_file_ops finding) — Windows `operation is None` branch, no fsync; not a drop-in for the helper because of the 0o600 contract |
| os_replace_no_atomic private_paths.py:1156 | retired — dir_fd-relative rename with identity precondition + parent fsync; the helper cannot express it |
| os_replace_no_atomic tls_trust.py:200 | confirmed (P3, noted) — regenerable cache; acceptable; dead `finally` unlink after success |
| plain_readback mosaic_render.py:109/113, text_wrap_index.py:75 | retired — `.plain` on a `rich.text.Text` the module itself built from pixels / from `Text(line).wrap`; no user markup is round-tripped |
| raw_1024x1024 egress.py:57-60 | retired — named module constants (`MAX_FETCH_BYTES_*`) |
| raw_1024x1024 file_extraction.py:639, file_handlers.py:148/491, input_validation.py:1762 | retired — named class constants / default args |
| raw_1024x1024 github_api_client.py:431/442 | retired — inline MB conversion for a rough cache-size estimate (`sys.getsizeof(str(data))` is itself a guess) |
| raw_mkdir atomic_file_ops.py:84/161/263 | retired — the helper's own parent-dir creation |
| raw_mkdir paths.py:53 | confirmed — inside the always-broken `get_project_databases_dir` (P3 paths finding) |
| raw_mkdir private_paths.py:1312/1634 | retired — Windows UNVERIFIED_PLATFORM branches by design |
| raw_mkdir tls_trust.py:183, ui_responsiveness_artifacts.py:70 | retired — the idiom; `ensure_directory_exists` is the thing to delete, not adopt |
| re_compile_in_def Utils.py:490 `is_valid_url` | retired — dead code (0 importers; `Article_Extractor_Lib.py:1809` has its own nested `is_valid_url`); listed under the Utils.py dead-symbols finding |
| re_compile_in_def input_validation.py:1034 `validate_email` / :1061 `validate_username` | retired — dead code (0 importers) |
| re_compile_in_def log_sanitizer.py:541 `_home_literal_pattern` | retired — `@lru_cache(maxsize=8)`; compiled once per distinct home tuple |
| seed_name__coerce_bool adaptive_reader_state.py:154, console_background_effects.py:43 | confirmed — part of the 21-copy cluster (P2 D4a finding); these two cannot import config (config imports them at `config.py:75,81`) |
| seed_name__format_size file_handlers.py:574 | confirmed — part of the 16-copy cluster (P2 D4b finding); this copy has no space before the unit |
| strftime Utils.py:700 `%b %d, %Y` | retired — display-only title date inside dead-ish `FileProcessor.process_filename_to_title` (test-only class) |
| tempfile_no_secure atomic_file_ops.py:94/169/271 | retired — same-directory `mkstemp` is required for an atomic rename; `secure_temp_files` would put it in `/tmp` |
| tempfile_no_secure private_paths.py:970, tls_trust.py:196 | retired — same-directory mkstemp for rename; and see the secure_temp_files finding (the helper adds nothing) |
| tempfile_no_secure ui_responsiveness_artifacts.py:24 | retired — `tempfile.gettempdir()` used as an allowed ROOT, not a temp file |
| try_import_guard app_shutdown.py:346 | retired — signal-handler-reachable config read, documented |
| try_import_guard custom_tokenizers.py:25 / token_counter.py:16,35 | confirmed (P3 custom_tokenizers finding) — module-scope guards WITH an import-time `logger.warning` |
| try_import_guard custom_tokenizers.py:395, file_extraction.py:913/922, file_handlers.py:118 | retired — not import guards; broad `except Exception` around real work, each producing a user-visible error string or False |
| try_import_guard db_upgrade_notice.py:48 | retired — documented failure-proof probe |
| try_import_guard doctor.py:154 | retired — see doctor.py:159 above |
| try_import_guard optional_deps.py:811/839/1288/1521 | retired — this module IS the guard layer; `:1521` is the module-scope config import (P3 filed) |
| try_import_guard paths.py:15 | confirmed — the always-failing import (P3 paths finding) |
| try_import_guard secure_temp_files.py:126 | retired — `import shutil` inside `finally` (stdlib; hoist is a nit) |
| try_import_guard startup_logging.py:45 | retired — documented (loguru may be absent) |
| try_import_guard terminal_utils.py:90/261/297/310 | retired — optional `textual_image`/`rich_pixels`/`PIL` probes; `:261,297,310` do real imports where `find_spec` would do (P3 nit, same class as optional_deps) |
| try_import_guard text_wrap_index.py:15 | retired — pinned fallback for a private Rich API (`test_text_wrap_index.py`) |
| try_import_guard tiktoken_runtime.py:145 | retired — optional tiktoken |
| try_import_guard ui_responsiveness.py:123 | retired — diagnostics drain thread must never raise (documented) |
| seeds from the parent: `Utils/ui_helpers.py`, `pagination.py`, `widget_helpers.py`, `ensure_directory_exists`, `truncate_content`, `secure_temp_files`, `atomic_file_ops`, `optional_deps`, `_chars_estimate`, `Widgets/base_components.py`, filename sanitizers, byte-size formatters, loguru+logging, `is_valid_url`/`validate_email`/`validate_username` re-compile, `private_paths:1000/1156`, `tls_trust:200`, egress bypass list | all examined — see findings and the helper table below |

## Verified-fine

- `secure_temp_files` chmod redundancy is a NO-OP, not a hole: `NamedTemporaryFile`/`mkstemp` create 0600 and `mkdtemp` 0700 (measured above). The 18 raw `tempfile` sites outside Utils are therefore not "insecure temp files".
- `log_sanitizer` regexes are all module-level compiled; the one function-level `re.compile` (`:541`) is `lru_cache`d. `redact_log_line` is applied by the stdlib root formatter (`Logging_Config.py:350`), so the three `Utils.Utils`-logging modules are still redacted.
- `token_counter`: `_CJK_RE` compiled once from the same `_CJK_RANGES` tuple as `_is_cjk`; the estimate cache write is locked, the read is a single dict `get` (documented and correct under the GIL); `ESTIMATE_CACHE_MAX_ENTRIES` bounds it (task-18602).
- `fts5_match_forms.quote_fts5_token` is the one FTS5 escape; `Tests/Utils/test_fts5_quoting_adoption_census.py` pins that no other `.replace('"', '""')` exists.
- `sensitive_paths._resolved` broad `except Exception` is the documented fail-closed contract (TASK-847: NUL bytes raise `ValueError`, not `OSError`).
- `private_paths.atomic_private_write_bytes` (POSIX path): dir_fd-relative create → write → `fchmod` → `fsync(file)` → identity re-check → `os.replace(src_dir_fd, dst_dir_fd)` → `fsync(parent)` → postcondition. This is the hardened writer; `atomic_file_ops` is the general one; both are justified as separate helpers.
- `egress.guarded_fetch_*`: per-hop policy re-check, credential stripping via allowlist (task-19733), client-level `auth=None` on cross-origin hops, byte caps, redirect caps — consistent across httpx/httpx-async/requests/aiohttp.
- `textual_css_fastpath` `id(rule)` cache holds a strong reference to the `rules_map` whose identity invalidates it — not a freed-id aliasing hazard.
- `app_shutdown` watchdog: deadline-based (not `is_alive`), start inside the lock, tighter-arm-wins — the review notes at `:208-254` are accurate to the code.
- `fd_protection`: process-wide lock held across the whole protected region, `closefd=False` wrappers, restores TRUE originals; the cross-feature serialization cost is documented as accepted (task-640).
- `file_extraction._is_valid_filename` accepting `..`-prefixed hints is harmless: the only save path re-validates with `path_validation.validate_filename` (`Widgets/file_extraction_dialog.py:547`).
- `db_status_manager.update_db_sizes` runs the stat loop via `asyncio.to_thread` and shields the task (task-22220) — not sqlite-on-the-loop.
- `optional_deps.DEPENDENCIES_AVAILABLE` reset copies from `_INITIAL_DEPENDENCIES_AVAILABLE` (the stale-duplicate bug it describes is fixed).
- `widget_helpers.py` is live: `UI/Voice_Cloning_Window.py:276` → imported by `UI/STTS_Window.py`, whose screen is registered (`UI/Navigation/screen_registry.py:194 "stts"`); plus `Tests/UI/test_install_command_clipboard.py`.
- `Utils/Splash.py` is live via `Splash_Screens/card_definitions.py:4` (census showed 0 because it excluded `Splash_Screens/`).
- `ui_responsiveness_artifacts.py` is harness-only by design (`Tests/UI/run_workbench_soak.py`, `Tests/UI/test_ui_responsiveness_artifacts.py`) — keep.

## Retired

- "`is_valid_url` compiles a regex per call (D2)" — symptom real in the source, cause moot: the function has 0 importers (`rg -n "\bis_valid_url\b" tldw_chatbook --glob '!Tests/**' | grep -v Utils.py` → only `Web_Scraping/Article_Extractor_Lib.py:1809` which defines its own). Filed as dead code instead.
- "`validate_email`/`validate_username` compile per call (D2)" — same: 0 importers, 0 tests. Dead.
- "`secure_temp_files` under-adopted (7 vs 24 raw sites) — D4a" — retired: the helper's only additions over stdlib are a redundant chmod, metrics, and a zero-overwrite; adoption would add cost and no guarantee. Filed as D3 over-engineering instead.
- "`optional_deps` under-adopted (43 vs 109 try/except ImportError)" — retired as a D4 gap: 41 importers; only 11 files carry module-scope `ImportError` guards without it, and `check_dependency` imports eagerly too, so adoption changes nothing for the D2 concern. Filed as a note.
- "egress bypassed by 19 modules (D1)" — 4 rows were dict `.get()` false positives and 1 was a `.md`; the remaining 13 are covered by the census docstring's explicit exemption + task-586/609 for egress; re-filed as the TLS-policy long tail (ADR-079), P3.
- "`Utils/Splash.py` dead (census 0 importers)" — false negative; live via `Splash_Screens`. Only two functions dead.
- "`private_paths.py:1000/1156` and `tls_trust.py:200` hand-roll `os.replace` beside `atomic_file_ops` (D4a)" — `:1156` is dir_fd-relative with fsync/identity checks (not expressible with the helper); `:1000` wants 0o600 (helper defaults 0o644) and is Windows-only; `:200` writes a regenerable cache. Justified; only the parent's three non-Utils plain re-rolls remain (filed P3).
- "`mosaic_render`/`text_wrap_index` `.plain` read-back un-escapes user text" — the `Text` objects are built by the module from pixels / from `Text(line).wrap`; no markup round-trip.
- "`windows_files._fields_` mutable class attributes" — ctypes ABI declarations.
- "`Utils.py:700 strftime`" — display format in a test-only class.
- "`tls_trust.py:200` leaves the temp file on failure" — no: `finally` unlinks if it still exists; after a successful `os.replace` the `exists` check is simply False. Harmless.

## Left UNVERIFIED

| claim | why not verified | literal command to run |
|---|---|---|
| `Chat/usage_recorder.estimate_tokens` (`len//4`) numbers are persisted to a usage ledger (would raise the token-estimate finding from P3 to P2 per the "drift reaches storage" rule) | traced only to `record_usage` in the same file; the ledger write path was not read | `rg -n "record_usage\|_prompt_tokens\|ledger" tldw_chatbook/Chat/usage_recorder.py tldw_chatbook/Chat/usage_ledger*.py tldw_chatbook/DB/*usage* 2>/dev/null` |
| ADR-079 accepts the TLS long tail as permanent (vs. "to be threaded later") — decides whether the 13-module list is a finding for other slices or out of scope | only grep hits `:29,:67-68` read, not the ADR body | `sed -n 20,40p backlog/decisions/079-network-tls-trust-policy.md; sed -n 60,80p backlog/decisions/079-network-tls-trust-policy.md` |
| `Tests/Utils/test_doctor.py` exercises `check_optional_dependencies()` with no argument (i.e. pins the wrong production answer as a requirement) | test file not read | `rg -n "check_optional_dependencies\(\)\|run_doctor\(" Tests/Utils/test_doctor.py` |
| SmartFileDropZone's `1,000,000 B → "1.0 MB"` is visible on a shipped path | computed from the function; the drop zone's liveness in the current Library import rail not checked | `rg -n "SmartFileDropZone" tldw_chatbook --glob '!Tests/**' -l` then `.claude/skills/verify/SKILL.md` recipe: `tmux -L verify …` drop a 1,000,000-byte file on Library ▸ Import and `capture-pane` |
| `github_api_client.py:488/568/636` in-function `from ..Utils.egress import …` can be hoisted (no import cycle through `config`) | not attempted | `cd $WT && source <SCRATCH>/env.sh && $PY -c "import tldw_chatbook.Utils.egress, tldw_chatbook.Utils.github_api_client; print('ok')"` |
| In-app marginal import cost of `Utils.Utils` (cold-process cumulative was 61.8 ms, dominated by first-import of loguru) | only cold `-X importtime` measured | `$PY -X importtime -c "import loguru, hashlib, unicodedata; import tldw_chatbook.Utils.Utils" 2>&1 \| rg "Utils\.Utils"` |
| `Utils/text.sanitize_filename` output at `Local_Ingestion/audio_processing.py:1257/1262` reaches a path join without a later `validate_filename` | call site returns the name to a downloader; not traced | `rg -n "_derive_filename\|sanitize_filename" tldw_chatbook/Local_Ingestion/audio_processing.py` then read the consumer of that return value |

## Dead or under-adopted shared helpers

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
