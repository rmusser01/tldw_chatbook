# gap-candidates — core-runtime review 2026-09-17

Triaged work list. **Nothing here has been filed**; these are proposed batches, one per canonical home so each is a single PR. ✅ = file it · ➖ = file only for breadth, low value on its own · ❌ = recommend against, with the design reason.

Coverage: **all 29 slices, 887,855 of 887,855 Tier-1 lines (100 %)**. Sampling within a slice is stated per slice in `report.md`'s coverage table.

## Already handled — check these before filing anything below

task-1378 · task-31202 (settings_screen split + ratchet row) · task-2542 (`_toast` dup) · task-586 · task-609 (egress/SSRF consolidation) · task-31502 (quiescence lock) · task-31572 · task-31584 (Library media helpers) · task-31650 · task-32089 · task-32170 · task-32013 · task-32199 (Library decomposition hygiene) · task-194 (provider display-name catalog) · task-19867 (`VALID_TABLES` drift) · task-32499 (agent-routing helper placement) · task-25704 · task-287 (`DEPENDENCIES_AVAILABLE` never populated — the `doctor` P1 is its visible face) · task-2902 · task-26834 (Console stalls) · task-30019 (legacy Collections) · task-17387 (kobold/tabbyapi generator functions) · task-1320 (screen mount I/O off the pump) · task-27010/26962/26984 (ruff debt).

---

## Batch 0 — The four crashes and the blanked text (P0 + the systemic P1) ✅

Do these first; each is S and independent.

| # | Change | Where | Size |
|---|---|---|---|
| 0.1 | Reset `more_actions_open` in `sync_state` so the flag cannot survive a recompose into a branch that never composes the region — **Escape currently kills the app** | `Widgets/Library/library_prompts_canvas.py:549-555` | S |
| 0.2 | Guard the post-`await` `query_one`, or cancel the section worker in `_hide_advanced` — **"Advanced…" then "Hide advanced" kills the app**. `exclusive=True` does not help: it only cancels another worker in the same group | `UI/MCP_Modules/mcp_inspector.py:3418`, `:1639` | S |
| 0.3 | Add `thread=True` — a sync callable passed to `run_worker` raises `WorkerError` and `exit_on_error` (default True) **kills the app**. Reachable via route `stts` → STTS window → dictation window | `Widgets/audio_troubleshooting_dialog.py:246` | S (one kwarg) |
| 0.4 | Replace `rich.markup.escape` with the repo's own `_escape_all_brackets` (`Library/library_rag_state.py:520`, whose comment already names the `[TODO]` case) at ~40 sites. `rich`'s pattern only escapes `[a-z#/@` tags, so `[TODO] Q3 plan` renders as ` Q3 plan` and `[IMPORTANT]` renders **blank**. Re-point the pinning tests, which all use lowercase tags — the one case the broken escape handles | `Widgets/Library/library_rail.py:377` + 27 sites; 12 Console sites | M |
| 0.5 | Add `markup=False` to the shared confirmation dialog — a title containing `[/…]` raises `MarkupError` inside its `compose()`, so the irreversible "Delete stored Full captures" confirmation **never appears**. 45 modules import this dialog; the correct pattern already exists at `Widgets/Library/prompt_delete_confirmation_modal.py:144` | `Widgets/confirmation_dialog.py:117` | S |
| 0.6 | Stop double-escaping into markup-off surfaces: `R&D Report` reaches the user as `R&amp;D Report`, and `summarize [draft]` as `summarize \[draft]` | `Chat/console_display_state.py:93`, `Chat/console_prompt_queue.py:171` | S |

**Then add the guard this class has never had.** Three of the four P0s violate one contract (a worker must be a coroutine or declare `thread=True`; an `await` must not resume into a subtree that may have been removed). An AST check over `run_worker(`/`@work` call sites — non-coroutine target without `thread=True`, and `query_one` after an `await` with no enclosing `try` — would have caught all three, and belongs next to the existing derived-artifact checks in `scripts/`.

## Batch 1 — Stop the data loss (P0 + P1, all S) ✅

One PR each; they are independent.

| # | Change | Where | Size | ADR |
|---|---|---|---|---|
| 1.1 | Give `create_document_version` and `update_keywords_for_media` their own `with self.transaction() as conn:` (a nested call joins the outer transaction, so the in-transaction callers keep working). **Then** file the Media half of task-22224 (`isolation_level=None` + explicit-BEGIN-only manager + a write-site census) — the file's own docstring defers to a task that does not exist | `DB/Client_Media_DB_v2.py:5412`, `:5530` | S now, M for the isolation flip | no |
| 1.2 | Write `next_review` in the column's own shape (or compare through `julianday()` as conversation pagination already does) and bind `start_date.strftime(...)` in `get_study_stats`; one-shot `UPDATE` for existing rows | `DB/ChaChaNotes_DB.py:21570`, `:21593`, `:21622`, `:23523-23549` | S | no |
| 1.3 | Route both `get_api_key` branches through the existing `resolve_provider_api_key` (`config.py:1400`) so a placeholder or padded key can never be returned | `config.py:9078-9087` | S | ADR-012 frames it; no new decision for the placeholder half |
| 1.4 | Fix `get_detected_api_providers` to iterate `config.get("api_settings", {})` instead of flat dotted keys; add one nested-config test | `config.py:8795` | S | no |
| 1.5 | Split `OSError` out of the corruption branch: re-raise (or return deny-all) and back up only on `ValueError`/`JSONDecodeError`/shape mismatch, which is what the spec actually says | `MCP/permission_store.py:747`, same shape at `:892` | S | no |
| 1.6 | Bound `CalculatorTool`: refuse `str` constants, cap `Pow` operands, and reject before evaluation rather than timing out into an abandoned thread | `Tools/tool_executor.py:181-220` | S | no |
| 1.7 | Move the `yield "data: [DONE]"` out of `finally` in the seven unfixed streaming handlers (OpenAI is already correct — copy it) and pin the fix for all eight | `LLM_Calls/LLM_API_Calls.py` | S | no |
| 1.8 | Forward the provider usage block on the Gemini and Cohere streaming paths, or give the gateway a final-usage fallback | `LLM_Calls/LLM_API_Calls.py:2832`, the google translator | S | no |
| 1.9 | Point the rail updater at the ids that exist (or delete the dead update), and add `@staticmethod` to `_open_video_with_os` | `UI/Screens/chat_screen.py:9404`, `:19648` | S | no |
| 1.10 | Decrement `_current_memory_bytes` in `_prune_expired_async` as every other removal path already does. Without it the RAG cache reaches its cap with **zero entries cached** and stops caching for the process lifetime (reproduced dead at prune cycle 42) | `RAG_Search/simplified/simple_cache.py:1055-1087` | S | no |
| 1.11 | Spawn local LLM servers with `start_new_session=True` and stop them with `killpg`, matching the nine other subprocess sites in this repo; add an `on_unmount` stop | `Event_Handlers/LLM_Management_Events/server_lifecycle.py:488`, `:570` | S | no |
| 1.12 | Stop HTML-escaping text bound for a markup-off Rich surface — `Library/library_rag_state.py:392` documents this exact bug and its fix from live UAT (task-15); the Console still ships the pre-fix behaviour | `Chat/console_display_state.py:93` | S | no |
| 1.13 | Import the vendored picker, or gate the button — `textual_fspicker` is neither installed nor a dependency, so the Transformers "Browse models dir" button always errors | `Event_Handlers/LLM_Management_Events/llm_management_events_transformers.py:153` | S | no |
| 1.14 | Run the realtime credential gate through `resolve_provider_api_key` instead of raw truthiness, so a placeholder key cannot reach the provider session | `UI/Console_Modules/realtime.py:730` | S | no |

## Batch 2 — The event loop (P1/P2, D2) ✅

| # | Change | Where | Size |
|---|---|---|---|
| 2.1 | Make `get_cli_setting`/`load_settings` cheap again: the value is cached but every read pays a storage-admission handshake (**11.1 ms measured, warm**). Hold one admission scope per derivation, or make the cached read skip it. This is the root of 2.2-2.4 and of several P2s | `config.py:6253` + `Backup_Recovery/config_participants.py` | M | 
| 2.2 | Gate the Library/RAG keystroke handler on a revision token instead of re-loading the active profile 7× and re-validating 4× per keypress | `UI/Screens/settings_screen.py` | M |
| 2.3 | Give the credential-readiness poll a revision-first gate and one admission scope (30–49 ms per 0.25 s tick at idle) | `UI/Screens/chat_screen.py` | M |
| 2.4 | Memoise the active review-set snapshot on `service.revision` (the file already does exactly this at `:16097` for a sibling) and route the five synchronous readers through it | `UI/Screens/library_screen.py:30987-31238` | S |
| 2.5 | Offload the Library semantic query: the ChromaDB call is synchronous on the Textual loop (212 ms cold) while every sibling call around it is already `asyncio.to_thread`'d | `RAG_Search/simplified/rag_service.py:1592` | S |
| 2.6 | Derive the workspace title only for the creation case — the fix already sits three methods up at `session.py:3534`; `:4419` and `:3688` never got it (a `workspace_records` SELECT 5×/s for the whole of every streaming run) | `UI/Console_Modules/session.py:4419`, `:3688` | S |
| 2.7 | Give `search_prompts` the subquery shape its sibling `search_library_prompts_page` already uses instead of binding every match as one `IN (?,?,…)` list (hard-fails past 32,766 matches) | `DB/Prompts_DB.py` | S |
| 2.8 | Batch the orphan reconcile through the class's own `_batch_hydrate_steps` instead of one query per run inside `BEGIN IMMEDIATE` at first construction, reachable from a Settings compose path | `DB/AgentRuns_DB.py:285`, `:2340` | S |
| 2.9 | In `_schedule_settled_research_source_operations`, move the early-returns above the `jobs()` deep copy and drop the `intersection_update` when the scheduled set is empty (O(n²): 1.65 s per 1,000 files); separately give the registry a non-copying iterator and a batch-persist context | `app.py:2749-2787`, `Library/library_ingest_jobs.py:1593` | S + M |

## Batch 3 — One timestamp helper ✅ (the review's main structural recommendation)

Twelve output shapes across 55 helper copies, a thirteenth from SQLite's `CURRENT_TIMESTAMP`, 100 lexical comparisons over the result, two confirmed user-visible defects (1.2 above, and the media cleanup cutoff below) and one existing `julianday()` workaround.

1. Add `Utils/timestamps.py`: `utc_now_iso(timespec=...)`, `to_iso_z(dt)`, `SQLITE_TS_FMT`, and a `parse_any(text)` that accepts every shape now in the wild (needed for migration-free reads). **New module** — `Utils/` has no time module and `runtime_policy/source_state.py:133` is private with 5 verbatim copies.
2. Adopt it writer-by-writer, one PR per package, storage writers first (`DB/`, `Chat/`), display last.
3. Fix the media cleanup cutoffs in the same pass — `hard_delete_old_media`/`get_deletion_candidates` build the cutoff in a different shape from the writer, so rows are skipped for up to 24 h (`DB/Client_Media_DB_v2.py:4351`, `:4457`, `:9255`; `:4457` also uses `datetime.utcnow()`, deprecated in 3.12).
4. Add the guard the repo lacks: a test that every `*_at`/`last_modified`/`timestamp` column in `VALID_TABLES` is written by the shared helper, mirroring the index-plan-pin census pattern in `scripts/`.

Size: L overall, M per package. ADR: yes — new, storage format across modules.

## Batch 4 — Adopt or delete the shared-helper layer ✅

| # | Change | Size |
|---|---|---|
| 4.1 | **Delete** the dead helpers: `Utils/{ui_helpers,pagination,cost_estimation,debug_helpers,ingestion_preferences,splash_animations}.py`, `Widgets/base_components.py`, the `Utils/paths.py` project-* helpers (their import target has never existed, so the only branch that runs always raises), 5 of 7 `Utils/text.py` functions, 6 `input_validation` validators, and 25 of ~32 `Utils/Utils.py` symbols. All re-verified at 0 importers against a 97,863-test collect-only inventory | M (one sweep) |
| 4.2 | Make `Utils/Utils.py:729 _format_size_bytes` public and collapse the 15 other byte-size formatters into it; fix `SmartFileDropZone`'s decimal-thresholds-with-binary-divisors bug on the way | S |
| 4.3 | Add a stdlib-only `Utils/coerce.py` that `config.coerce_bool_setting`/`coerce_int_setting` delegate to (config's import-time work makes the reverse a cycle), then retire the 23 private copies. Decide the unrecognised-string behaviour explicitly — the canonical copy is the odd one out (rejects `"on"`) | M |
| 4.4 | Adopt `Utils/atomic_file_ops` at the 3 fsync-less re-rolls (`Agents/local_tool_provider.py:466`, `Chat/trajectory_export.py:1319`, `emergency_stop.py:92` — which calls itself durable) | S |
| 4.5 | Fold the 7 filename sanitizers into `Utils/path_validation.py` and choose one safety envelope; `Utils/text.sanitize_filename` currently keeps NUL and control characters and two copies write to disk/zip unvalidated | M |
| 4.6 | Adopt `Utils/token_counter.estimate_tokens` at the 4 `len//4` sites | S |
| 4.7 | Publish `strict_json_loads(text, *, max_depth, max_nodes, reject_duplicate_keys)` from `Utils/input_validation.py` and retire the 15 copies — the wire family and the storage family currently disagree about duplicate keys, which can raise an uncaught error mid-round-trip | M |

## Batch 5 — Scope-service scaffold ✅ (L, needs an ADR)

`_maybe_await` ×65, `_enforce_policy` ×51, `_require_client` ×47, `_normalize_mode` ×46 across ~45 services. The helper bodies are harmless; the **call shape** is not: `await self._maybe_await(local_sync_call())` runs sync sqlite on the event loop, and task-283's fix (thread file-backed connections, keep `:memory:` inline) exists in exactly one service. Propose `runtime_policy/scope_service.py` with a `ScopeServiceBase` carrying the four helpers plus a `run_local(fn, *a, **kw)` that applies that rule once. Deletes ~2,000 lines and closes the class. ADR: yes — cross-module interface; ADR-036 governs composition only, so it does not conflict.

## Batch 6 — Delete the legacy surface ✅

The `Event_Handlers/` deletion candidates are now verified individually (dotted-path `rg`, relative-import forms, the collect-only inventory, and a real `importlib.import_module` each): `app_lifecycle` (141 lines), `ingest_events` (28, a re-export shim), `ingest_status_helper` (123), `tab_events` (48 — ADR-014's stated reason for keeping it is stale, `app.py` has zero references), `Chat_Events/chat_messages` (445), `Audio_Events/dictation_integration_events` (106), `Media_Creation_Events/swarmui_events` (236 + its `__init__`, which is not merely dead but **unimportable**: `ImportError: cannot import name 'GenerationResult'`), the two 0-byte `llm_management_events_{llamacpp,llamafile}` files, and `eval_db_operations` (336, dead but for its own test). Deleting them needs matching rows pulled from `Docs/security/production-diagnostic-inventory.json` (a preflight gate) and one text assertion moved in `test_application_state_ownership.py:1232`.


| # | Change | Size |
|---|---|---|
| 6.1 | Delete `UI/Tools_Settings_Window.py` (6,926 lines, `DEPRECATED (TASK-1346)`, nav-unreachable), its wrapper `UI/Screens/tools_settings_screen.py`, the `app.py` dead cluster that only it reaches, and the 4 suites that pin them. Keep `Tests/Packaging/test_raw_cli_import_closure.py`'s assertion that it stays out of the closure — re-target it | M |
| 6.2 | Delete `UI/Screens/schedules_screen.py` (superseded, no route, no importers) and `Constants.py`'s dead 1,427-line `css_content` | S |
| 6.3 | Delete or re-point `ConsoleProviderGateway._chat_api_kwargs` — 0 production callers, 10 test callers, and its live twin has drifted; the Anthropic cache-stability tests currently pin a builder the send path never runs | S |
| 6.4 | Fix the doc drift: `CLAUDE.md:47/51` and `AGENTS.md`'s "Main Windows" list name seven modules that no longer exist; `Widgets/Chat_Widgets/chat_approval_card.py:22` names one of them | S |
| 6.5 | Verify and then delete the `Event_Handlers/` deletion candidates (8 modules + two 0-byte files) — the census is complete, the confirming greps were cut off | S |

## Batch 7 — Correctness follow-ups (P2) ✅

Independent, one PR each: the two `with self.get_connection()` DML readers that commit a caller's transaction (`ChaChaNotes_DB.py:12752`, `:12802`); the `AgentService._persist` read gap that leaves a run `running` until the next launch; the store lock-order inversion (`console_chat_store.py:14357`, `:20174`); the swallowed terminal-settlement write (`:3812`); the per-keystroke settings writers that check their generation token before taking the file lock; `MediaDatabase(check_integrity_on_startup=True)`, which always raises because the method does not exist; `ReadFileTool`'s uncapped `content`; `fs_edit`/`fs_patch` writing non-atomically beside an atomic `fs_write`; `ListDirectoryTool` ignoring its own `max_depth ≤ 5`; `MCP/tools.py::chat_with_character`, a shipped tool that always errors; `summarize_with_vllm`'s `UnboundLocalError` on every explicit-key call; the uncapped `Retry-After` sleep; `RichLogHandler` dropping worker-thread INFO from the in-app Logs screen; the malformed `SINGLE_USER_FIXED_ID` env var that aborts `import tldw_chatbook.config`.

## Batch 8 — Size governance ➖

The ratchets are red (chat_screen, library_screen, 16 of 43 Library controller rows) and absent for `settings_screen.py` (task-31202, To Do), `personas_screen.py`, `app.py` (21,050 lines), `console_chat_controller.py` (29,048) and `console_chat_store.py` (22,245). Re-pinning the red rows is mechanical and belongs to whoever lands next in those files. Adding rows for the five unbudgeted god modules is worth one small PR. **Do not** open a decomposition for `app.py` or the Console controller off the back of this review: the recipe (`backlog/docs/library-decomposition-recipe.md`) requires a per-subsystem PR series with field-ownership scripts, and the Library series shows what that costs.

## Recommended against ❌

| Proposal | Why not |
|---|---|
| Consolidate the 31 `*_Interop/` packages as one template | The premise does not hold: two normalised server services differ by 891 diff lines. Only the scope-service scaffold is shared, and Batch 5 covers it |
| Push `Utils/secure_temp_files` adoption at the 18 raw `tempfile` sites | stdlib `tempfile` already creates 0600/0700; the module's "secure delete" is theatre over a modern filesystem. Keep the module for its manager, do not sweep |
| Adopt `Utils/optional_deps` at the 11 module-scope `try: import` sites | It does not fix the cost — `check_dependency` imports eagerly too. The real fix is lazy access, which is a different change |
| Collapse the 53 `_cancel` handlers | They are 2-line adapters over `SafeModalDismissMixin`, which is already the canonical home with 79 importers. A mixin-level default `@on(Button.Pressed, "#cancel")` would delete them, but the value is cosmetic — do it opportunistically |
| Replace the raw `mkdir(parents=True, exist_ok=True)` idiom with `ensure_directory_exists` | The raw idiom is correct and `ensure_directory_exists` is dead. Delete the helper (4.1); the hardened path is `private_paths.secure_private_directory`, which already has 32 importers |
| "Fix" the `julianday()` ORDER BY in conversation pagination | It defeats the index but it is the correct workaround for mixed shapes and is bounded by one character's conversations. It becomes removable only after Batch 3 |
