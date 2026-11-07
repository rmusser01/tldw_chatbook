# UI-settings — tldw_chatbook/UI/Screens/settings_screen.py, 30810 lines

Worktree: /Users/macbook-dev/Documents/GitHub/tldw-review @ d8fb4053f9 (read-only). All measurements under `env.sh` (isolated scratch profile, 35 KB config.toml); the app was never run.

## Coverage
| file | lines | read in full / sampled (which ranges) / mechanical only |
|---|---|---|
| tldw_chatbook/UI/Screens/settings_screen.py | 30810 | **Read in full, 1-30810** (sequential 600-1000-line chunks). Plus mechanical passes over the whole file: ast resolution of all 93 function-body import statements (module AND name level); 20 `run_worker(` + 44 `@work` sites with every coroutine/worker body read; the one `set_interval` callback; every `except Exception`/`BaseException` site; `asyncio.run`; every config write seam; class-level mutable attrs (ast); `recompose=`/`id(`/`.plain`/`re.compile`/`threading.Lock`/`get_cli_setting`-in-compose greps. Reproductions run against the shipped code under the isolated env (details per finding). |
| (evidence only) `UI/Screens/settings_config_adapter.py`, `config.py` seams, `Workspaces/registry_service.py`, `Workspaces/change_tracking.py`, `LLM_Management/snapshot_settings.py`, `Tests/UI/test_settings_*` | — | grep/targeted reads only, to trace seams and pinning tests — not reviewed. |

## Findings

### P1 [D2] — One keystroke in any Library/RAG editor field re-loads the active RAG profile 7× and re-validates it 4× on the event loop (measured 388-426 ms per keystroke; 1.3 s for the first)
- Where: every `handle_library_rag_*_changed` (`settings_screen.py:25630-25932`, 24 handlers) → `_stage_library_rag_value:9339` (→ `_library_rag_loaded_values` → `load_rag_defaults_from_active_profile`) → `_mark_library_rag_settings_staged:9353` → `_library_rag_validation_result:9355`, `_update_library_rag_preview:9366`, `_update_library_rag_validation_classes:9367`, `_update_library_rag_soft_warning:9368`, `_update_draft_status_widgets:9369` → `_update_category_state_banner`/`_category_state_banner_text:8566-8572` and `_update_guided_action_widgets` → `_guided_actions_enabled:7973` + `_guided_action_message:7899-7904`. Each of those re-derives `_library_rag_setting_values():6672` from scratch (`load_rag_defaults_from_active_profile()` + `validate_library_rag_defaults`). Compose does the same: `_render_library_rag_detail:18684` (`_library_rag_setting_values`), `:19193` (`_library_rag_soft_warnings`), `:19203` (`_library_rag_preview_rows`), `:18718` (`_library_rag_first_run_active`), `:18698` + `_render_library_rag_profile_block:18381-18382` (`active_profile_info`/`list_profiles_grouped` twice each).
- Evidence: sync-constructed `SettingsScreen(app)` (the construction `Tests/UI/test_settings_rag_profile_region.py:144` uses), `active_category = LIBRARY_RAG`, module names wrapped with counters, then `_stage_library_rag_value("default_top_k", n)` + `_mark_library_rag_settings_staged()` — the exact handler body: `PROBE keystroke#0: 1343.2 ms; calls = {'load_rag_defaults_from_active_profile': 7, 'config.get_cli_setting': 170, 'validate_library_rag_defaults': 4, 'soft_config_warnings': 1}`, `#1: 387.8 ms (7 loads, 32 get_cli_setting, 4 validations)`, `#2: 426.0 ms (same)`. Compose parts: `_library_rag_setting_values=51.9ms _library_rag_soft_warnings=42.8ms _library_rag_preview_rows=40.5ms _library_rag_first_run_active=21.2ms` (≥156 ms of reads per category open before any widget is built). Per-call costs: `load_rag_defaults_from_active_profile() median=53.1ms`, `active_profile_info() 10.0ms`, `list_profiles_grouped() 11.0ms`, `validate_library_rag_defaults 10.5ms`, `index_change_pending 9.9ms`. cProfile of the adapter load: all of it is 4× `config.get_cli_setting` → `Backup_Recovery` admission → `posix.open` (9640 opens / 5 loads).
- Why it matters: the Library/RAG category is the only Settings surface where typing stalls the TUI for ~0.4 s per character; the 7×/4× multiplier is structural (the handlers never cache the loaded defaults), so it holds even if the per-read price is lower on a settled profile (see UNVERIFIED for the env caveat).
- Recommended correction: (S) load once per event — cache `_library_rag_loaded_values()` on the screen for the lifetime of a category visit (invalidate on set-active/clone/rename/delete/save/revert, exactly the moments the file already lists for `_image_gen_raw_section_cache:6762`), and compute `_library_rag_validation_result()` once inside `_mark_library_rag_settings_staged` and pass it down; the `_guided_action_message`/`_category_state_banner_text` re-validations then read the cached result. The Image Gen block already models this cache (its docstring at `:3333-3348` gives the three invalidation points).
- Size: S · ADR: no · Confidence: verified (counts + wall time on the shipped handler body; absolute ms are from the isolated env)
- Pinning test: none for cost; `Tests/UI/test_settings_rag_profile_region.py` pins behaviour only.
- Already covered: none (task-19647 is the Backfill-control ADR-003 drift, not this)

### P2 [D1] — Two per-keystroke instant-persist writers can land an older `[permission_summary]` / `[model_catalog]` snapshot on disk after a newer one
- Where: `settings_screen.py:6558-6581` `_persist_permission_summary_section_values` (`@work(thread=True)`, no group/exclusive; generation token checked at `:6576` BEFORE the file lock) and `:14130-14149` `_persist_model_catalog_section_values` (`@work(thread=True)`, no group/exclusive, no token at all). Dispatchers: `:6520-6556` (bound to `Input.Changed` on the provider/model Inputs, `:27595-27603`) and `:14050-14128` (bound to `Input.Changed` on `#settings-model-catalog-stale-hours`, `:27585`; builds a FULL-section snapshot of all catalog checkboxes + the input).
- Evidence: `save_settings_to_cli_config` (config.py:8346) → `apply_settings_mutation_to_cli_config` → `_apply_literal_settings_transaction_locked` under `_CONFIG_FILE_LOCK` (config.py:6318) serialises writers but does not order them. Reproduced on the shipped worker bodies (`.__wrapped__`) with `save_settings_to_cli_config` replaced by a barrier + lock stub: `permission_summary: writers that reached the file write = ['NEW','OLD'] (both passed the generation guard); landing order = ['NEW','OLD']; ON DISK = OLD` and `model_catalog: typed 12 then 123; reached = [12, 123] (no guard at all); landing order = [123, 12]; ON DISK = 12`. A real write is ~64 ms (`save_settings_to_cli_config median=63.9ms max=73.3ms`), so two keystrokes inside that window overlap.
- Why it matters: config.toml ends up disagreeing with the widget the user is looking at (and with the cache the no-op guard compares against on the NEXT keystroke, so it self-heals only if the user types again).
- Recommended correction: check the token INSIDE the lock — `apply_settings_mutation_to_cli_config(section_values, locked_snapshot_precondition=lambda _s: generation == self._permission_summary_persist_generation)` (the seam already exposes that hook); add the same generation token to the model-catalog writer. `group=`+`exclusive=True` does NOT fix it (a started thread write cannot be cancelled).
- Size: S · ADR: no · Confidence: verified
- Pinning test: none (`grep -rln "_persist_permission_summary\|_persist_model_catalog" Tests` → nothing); `Tests/UI/test_settings_model_catalog_toggles.py` mocks `save_settings_to_cli_config` and asserts `call_args` (last call) — sequential, cannot see ordering.
- Already covered: none

### P2 [D2] — Four one-shot Settings actions do a 40-110 ms config write / forced full reload on the event loop
- Where: `:23597` `save_settings_to_cli_config(...)` + `:23614` `load_settings(force_reload=True)` inside `async def _perform_runtime_source_switch` (dispatched via `self.run_worker(coroutine)` at `:23568`, i.e. ON the loop); `:28279` `SettingsConfigAdapter().save_sections(section_values)` in the NETWORK branch of `action_settings_save_category` (sync action handler); `:10809-10821` `_run_diagnostics_validation`/`_run_diagnostics_reload` (the Validate/Reload BUTTONS at `:27810-27818`) call `adapter.validate_config_file` + `adapter.load(force_reload=True)` synchronously — while the `t` key path for the same category uses the thread worker `_diagnostics_validation_and_reload_worker:10898`.
- Evidence: `save_settings_to_cli_config (write+atomic replace+cache reload): median=63.9ms max=73.3ms`; `load_settings(force_reload=True): median=42.5ms max=49.7ms`. Every other write in this file is off-loop (15 `@work(thread=True)` writers, `asyncio.to_thread` in `_persist_briefing_schedules_gate:27499` and `WebSearchSettings.save` settings_web_search.py:331).
- Why it matters: the TUI freezes ~0.05-0.1 s per click on those paths; CLAUDE.md's own rule is "Workers for operations >100ms". Bounded to one click each, so P2.
- Recommended correction: `await asyncio.to_thread(...)` in `_perform_runtime_source_switch` (same shape as `:27502`); route the NETWORK save and the two Diagnostics buttons through the existing thread workers (`_settings_save_appearance_worker:29806` shape; `_diagnostics_validation_and_reload_worker` already exists — the buttons just don't use it).
- Size: S · ADR: no · Confidence: verified (timing) / verified (call sites)
- Pinning test: none
- Already covered: none

### P2 [D4b→D1] — Video Gen save persists "Clear" deletions as separate non-atomic writes; the Image Gen twin it "mirrors" does one atomic mutation
- Where: `:7812-7827` `_settings_save_video_gen_worker` (`adapter.save_sections(sections)` then a `for section, keys in deletions: adapter.delete_values(section, keys)` loop) vs `:7223-7235` `_settings_save_image_gen_worker` (`apply_settings_mutation_to_cli_config(sections, delete_keys=deletions)`). The Video block's own header (`:7475-7481`) says it is "mirroring the Image Gen block's idioms".
- Evidence: `SettingsConfigAdapter.delete_values` → `delete_settings_from_cli_config` (config.py) → its OWN `apply_settings_mutation_to_cli_config({}, delete_keys=...)` — 1 + N independent atomic file replacements (~64 ms each) with no all-or-nothing guarantee. Mechanical pair diff (ast, prefix-normalised, difflib): 17 paired methods, 7 with similarity ≥ 0.90 (4 at 1.00), 365 vs 348 lines; `_settings_save_x_gen_worker` similarity 0.54 is exactly this divergence.
- Why it matters: a Video Gen save that both edits a field and clears a saved secret can land the edit and NOT the clear (or vice-versa) when a later replacement fails; the user is told "Failed to save" while half the change is on disk — and the cleared-secret case is the one `delete_values`' docstring says must never be written back.
- Recommended correction: use the image block's single `apply_settings_mutation_to_cli_config(sections, delete_keys=deletions)` in the video worker (S). The larger D4(b) — one parameterised backend-draft block (prefix, BACKEND_IDS, FIELD_SCHEMA, raw-section name) — is the shape task-1378's split would produce; cite, don't redesign.
- Size: S (atomicity) / M (shared block) · ADR: no · Confidence: verified (seam traced; not reproduced with an injected failure)
- Pinning test: none for atomicity.
- Already covered: task-1378 (block extraction only)

### P3 [D3] — God module: 28,140-line `SettingsScreen` (996 methods), no size-ratchet row exists, task-1378 still To Do
- Where: `settings_screen.py:2671-30810`. Cluster map (ast, methods grouped by name prefix; lines = method bodies):
  | cluster | methods | lines | span |
  |---|---|---|---|
  | category chrome / panes / inspector / search / focus / footer | 109 | 5468 | 2715-29459 |
  | provider / model / credential / discovery / endpoint probe / custom endpoints | 212 | 5365 | 5663-27801 |
  | library_rag / profiles / backfill | 158 | 3389 | 4034-30215 |
  | console_behavior / capture / agent budget / thinking / background effects | 136 | 3098 | 5404-30595 |
  | (other: loaders, coercers, config-path helpers) | 100 | 1994 | 2854-30810 |
  | workspaces | 37 | 1502 | 4038-24834 |
  | audio_cpp / vllm handoff transactions | 29 | 1390 | 11946-27485 |
  | privacy / raw_cli / canvas / terminal | 47 | 837 | 5412-28251 |
  | sync / manual sync / runtime source | 30 | 784 | 3552-23875 |
  | appearance / theme / splash | 42 | 771 | 6633-30719 |
  | image_gen | 26 | 505 | 6762-7322 |
  | tool_profiles / tool packs | 15 | 437 | 4301-4779 |
  | video_gen | 20 | 382 | 7483-7889 |
  | speech_tts | 9 | 260 | 8432-23143 |
  | diagnostics / advanced_config / internal_prompts | 12 | 171 | 4276-23211 |
  | network | 7 | 97 | 23426-23531 |
  | personal_context | 5 | 36 | 3485-29524 |
  | web_search | 2 | 19 | 20004-20023 |
- Evidence: ast census above (26,505 method lines); `backlog/tasks/task-1378` status To Do (filed at "~10.8k lines"); `backlog/tasks/task-31202` status To Do ("15,922 lines at the 2026-08-02 baseline" — the file has since nearly doubled); `Tests/UI/test_screen_size_ratchet.py` does not exist and `grep -rl ratchet Tests --include='*.py' | xargs grep -l settings_screen` finds only `Tests/Terminal/test_dependency_qualification.py` (unrelated). Every cluster spans the whole file (e.g. provider 5663→27801), i.e. the clusters interleave — the §2 field-ownership script in `backlog/docs/library-decomposition-recipe.md` is the right first move, not a by-line-range cut.
- Why it matters: 6 clusters ≥ 800 lines each; the P1 above and the Video/Image drift are both symptoms of category logic that cannot see its siblings.
- Recommended correction: none new — task-1378 (split per §1 per-subsystem PR series) and task-31202 (ratchet row at the measured 30,810 / 996-method values, mutation-checked). Current values for that row: 30,810 lines, 996 methods, 44 `@work`, 20 `run_worker`.
- Size: L · ADR: yes (recipe `backlog/docs/library-decomposition-recipe.md` §1/§2/§17) · Confidence: verified
- Pinning test: none (that is task-31202's AC)
- Already covered: task-1378, task-31202

### P3 [D4b] — 15 methods carry the same draft-staging shape (setdefault SettingsDraft / set_value / pop-if-clean), 222 lines
- Where: `_stage_agent_budget_value:5881`, `_stage_console_background_effect_value:6223`, `_image_gen_stage:6808`, `_video_gen_stage:7547`, `_stage_appearance_value:8723`, `_stage_console_large_paste_value:8932`, `_stage_console_default_value:8945`, `_stage_console_paste_threshold_value:8979`, `_stage_console_max_parallel_runs_value:9007`, `_stage_tool_result_display_chars_value:9044`, `_stage_library_rag_value:9339`, `_stage_storage_value:9645`, `_stage_provider_api_mode:11041`, `_stage_provider_value:11928`, `handle_speech_tts_draft_modified:23122`.
- Evidence: ast scan for methods containing `SettingsDraft(category` + `.set_value(` + `_settings_drafts.pop(category` → 15 hits, 222 lines. Drift is only whether `_update_draft_status_widgets` is called inside.
- Why it matters: every new category re-types the idiom; `SettingsDraft` already owns `set_value`/`is_dirty`.
- Recommended correction: one `_stage_draft_value(category, key, loaded, value)` (or `SettingsDraft.stage(...)` in `UI/Screens/settings_config_models.py`); callers become one line.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none needed
- Already covered: task-1378 (falls out of the split)

### P3 [D3] — 8 function-body imports re-import a module already imported at module scope; `assign_select_value` is body-imported 4× for no cycle
- Where: `:815 _theme_save_target` (config), `:5681`/`:5699 _current_reasoning_target` (Chat.console_provider_endpoints / Chat.console_session_settings), `:6958 handle_image_gen_checkbox_changed` (Widgets.settings_image_gen_panel), `:7639`/`:7788` (UI.Screens.settings_video_gen_defaults), `:12483 _provider_current_credential_source` (Chat.provider_readiness), `:23589 _perform_runtime_source_switch` (Utils.input_validation `validate_url` — also a top-level name at `:195`, a shadowing re-import). `Widgets.select_values.assign_select_value` at `:12969`, `:21582`, `:26849`, `:29261` — the module is 53 lines and imports only `textual.widgets.Select`.
- Evidence: ast cross-check of every body `ImportFrom` against the module's top-level import set → the 8 rows above. All 93 body imports resolve at module AND name level (the `LLM_Management.snapshot_settings` one resolves as a submodule import). 1 body import breaks a documented real cycle (`Widgets.workspace_create_modal`, `:447`); 13 are the ADR-097 `settings_rag_profile_adapter` seams; the rest are ADR-097 boot-budget lazies (documented at `:453-482`, `:19608`, `:20814`).
- Why it matters: a per-call `import` of an already-loaded module is pure noise, and it makes the cycle-breaking imports indistinguishable from the accidental ones.
- Recommended correction: hoist the 8 (+ `assign_select_value`) to the top; leave the ADR-097 ones and the documented cycle.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P3 [D4b] — Two verbatim 3-way "Save / Discard / Cancel" leave modals; three verbatim workspace-create continuation wrappers; three verbatim screen-signal loops
- Where: (a) `:2572-2614 RagProfileSwitchConfirmModal` (`_handle_cancel/_handle_discard/_handle_save`) ↔ `Widgets/Settings_Widgets/speech_tts_settings_panel.py:~585-620` (`handle_cancel/handle_discard/handle_save`) — identical bodies; `Widgets/confirmation_dialog.ConfirmationDialog` is 2-way only (`confirm_label`/`cancel_label`), so no 3-way helper exists. (b) `_handle_workspace_create_result` ×3: `:23967`, `UI/Screens/library_screen.py:35388`, `UI/Console_Modules/workspace.py:4739` — identical except the class name in the `continuation=` lambda; the `_continue_*` bodies are legitimately surface-specific. (c) `_signal_console_identity_refresh:29587`, `_signal_console_appearance_refresh:29621`, `_signal_library_reader_layout_refresh:29750` — the same 30-line "bump `app_instance._<x>_generation`, walk both screen stacks deduped by `id()`, call hook" loop three times (only the attribute/hook names differ).
- Evidence: read side-by-side; dup_verbatim rows 2550/2602/2607/2612 in the excerpt confirmed as family (a). The 10-copy `event.stop(); self.dismiss(None)` `_handle_cancel` family across Persona/Console modals is the same P3 — a shared base cannot delete an `@on(Button.Pressed, "#specific-id")` handler unless ids are standardised; the saving is ~4 lines per modal.
- Recommended correction: (a) `UnsavedChangesDialog(ModalScreen[str])` beside `ConfirmationDialog` returning "save"/"discard"/"cancel"; (b) fold the interview chaining into `Personal_Context/interview_launch.py` (`launch_or_continue(app, result, continuation)`); (c) one `_signal_screens(generation_attr, hook_name)` helper.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P3 [D3] — Validators return prose; the screen maps message PREFIXES back to field keys by `startswith`, and that already produced a silent-failure incident once
- Where: `_appearance_invalid_field_key:8736-8780` (17 `startswith` branches), `_library_rag_invalid_field_key:9506-9563`, `_storage_invalid_field_key:9670-9678`; the validators live in `settings_appearance_defaults.py` / `settings_library_rag_defaults.py` / `settings_storage_defaults.py` and return a `ValidationResult(valid, message)` only.
- Evidence: the in-file incident record at `:9537-9553`: "The four startswith() checks this replaced ... never matched that wording, so a hard error on any of these fields blocked Save without ever highlighting the field red" (M3, SP3 final review) — patched with case-insensitive substring matches on RAGConfig's literal wording, which is the same fragility one layer down.
- Why it matters: a copy edit to any validator message silently un-highlights its field; there is no test that can notice because the mapping is by string.
- Recommended correction: (M) add `field: str | None` to the shared `ValidationResult` in the three `settings_*_defaults.py` modules and have the screen read `validation.field`; delete the three prefix tables.
- Size: M · ADR: no · Confidence: verified (incident documented in-file)
- Pinning test: `Tests/UI/test_settings_rag_profile_region.py` pins behaviour for the RAG branch only.
- Already covered: none

### P3 [D3] — `on_unmount` swallows `BaseException` from the audio.cpp handoff cleanup, contradicting the cleanup's "without hiding failures" contract
- Where: `:3919-3926` (`except BaseException: pass` ×2 around `_retry_audio_cpp_staged_request_cleanup` / `_retry_audio_cpp_result_cleanup(force_overlap=True)`); the callee's docstring at `:22465` is "Retry the one retained owner-thread rollback without hiding failures" and it raises `_AudioCppResultTransactionError` (a `RuntimeError`) at `:22481`, `:22496`, `:22527`, `:22531`.
- Evidence: read only; the remaining `_audio_cpp_result_cleanup` claim stays retained in `PendingHandoffStore` when the release at `:22530` fails, and nothing logs it.
- Why it matters: `BaseException` also swallows `CancelledError`/`SystemExit` during teardown; a failed release leaves a claim on `HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT` that the next Settings visit cannot claim (`store.claim` returns None) — a stuck Model-Library return with no diagnostic.
- Recommended correction: `except _AudioCppResultTransactionError as exc: logger.warning(..., type(exc).__name__)` (the same shape the file uses at `:12254` and `:21985`).
- Size: S · ADR: no · Confidence: inferred (stuck-claim consequence not reproduced) — command: `cd $WT && source env.sh && $PY -m pytest Tests/UI/test_settings_speech_audio_cpp_handoff*.py -q` and then a unit that makes `PendingHandoffStore.release` return False before `on_unmount`.
- Pinning test: none found for the unmount path.
- Already covered: none

### P3 [D3] — Comment rot / dead defensiveness (four small items)
- Where: (1) `:18258`, `:18288`, `:18303`, `:18315` cite `SearchRAGWindow._run_index_backfill` as the pattern being mirrored; `UI/SearchRAGWindow.py` does not exist (`ls` → no such file; repo grep finds one more stale mention in `watchlists_collections_screen.py:13825`). (2) `:8874-8877` `try: from tldw_chatbook.css.Themes.themes import ALL_THEMES except (ImportError, ModuleNotFoundError)` — a hard in-repo module (`css/Themes/themes.py` exists, imported by app.py); the except is unreachable. (3) `:17104-17113` `legacy_detail` — a `Select` composed with `display = False` whose only readers are `:24862` (apply) and `:30291` (sync); the user can never change it, so the apply path always re-sends the loaded value. (4) `_video_gen_raw_section:7486` and `_video_gen_select_suppress_queues:7504` use `getattr(self, ..., None)` for attributes never declared in `__init__` (the image twins are declared at `:3317`/`:3348`).
- Evidence: commands above; `asyncio.run` at `:18326` is inside `@work(exclusive=True, thread=True, ...)` `:18246` — a fresh worker thread with no running loop — so the pattern the stale comment describes is still correct (see Retired).
- Recommended correction: reword the four comments to describe the pattern without the dead reference; drop the theme try/except; either delete `legacy_detail` and read the detail from `self._console_capture_policy.detail` at `:24862`, or say why a hidden control must carry it; declare the two video attrs in `__init__`.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P3 [D2] — Workspaces compose does sqlite + filesystem reads on the loop (N+1 per workspace row)
- Where: `_render_workspaces_detail:19316-19361` — `registry.get_active_workspace()`, `registry.list_workspaces(...)`, then `registry.list_folder_bindings(record.workspace_id)` PER ROW (`:19338`) inside `compose`; `_render_workspace_card` → `get_workspace`, `_render_workspace_folder_bindings:19796` (`list_folder_bindings` "recomputed from disk" — `registry_service.py:2490` doc: status recomputed by stat), `_render_workspace_change_review:19468` (`ShadowRepoService().available` = `Path.exists`/`shutil.which`, `change_tracking.py:186-190`) + `service.status(workspace_id)`, `_render_workspace_default_assistant:19644` (`personas.list_persona_profiles()`), `:19687` (`store.list_profiles()`), `:19721` (`store.load()`).
- Evidence: read only; the sibling categories that touch stores (RAG index status `:18200`, tool profiles `:4340`, sync rows `:10242`) all fetch off-thread and compose placeholders.
- Why it matters: bounded by workspace count (small), so P3; but it is the one category whose compose is O(workspaces × bindings) of sqlite+stat on the loop, and `_refresh_settings_workspaces_pane:19852` re-runs it on every row click / toggle / archive.
- Recommended correction: fetch the listing + bindings in a `@work(thread=True)` and compose from the cached tuple (the `_tool_profiles_listing` pattern at `:4340-4385`).
- Size: M · ADR: no · Confidence: inferred (not timed against a populated Workspace_DB) — command: `cd $WT && source env.sh && $PY - <<'EOF'` building `LocalWorkspaceRegistryService` on a tmp `WorkspaceDB` with 20 workspaces × 5 bindings and timing `list(screen._render_workspaces_detail())`.
- Pinning test: `Tests/UI/test_settings_workspaces*.py` (behaviour only)
- Already covered: none

### P3 [D1, cross-ref ENTRY-config] — `_provider_api_key_value` reads the raw `api_settings.<p>.api_key` without the `resolve_provider_api_key` validity check the readiness path applies
- Where: `:12646-12648` (`str(api_key or "").strip()`); consumed by `_provider_current_credential_source:12520` (→ `"stored"`), `_stage_provider_value:11938` (the api_key "original"), and `_provider_saved_api_key_present:12658` does NOT use it (it goes through `get_provider_readiness`, which validates).
- Evidence: read only. The shipped template ships `[api_settings.google] api_key = "<API_KEY_HERE>"` (CLAUDE.md), so on an unmodified profile `_provider_current_credential_source("google")` returns `"stored"` while `get_provider_readiness` says not ready — the two disagree on the same screen.
- Why it matters: only the ProviderTestEvidence identity (`credential_source`) is fed from the raw read, so the user-visible status rows stay correct; a placeholder key can still make evidence identity say "stored". ENTRY-config P1/P2 own the placeholder/precedence defect; this is the third raw reader.
- Recommended correction: derive `credential_source` from `get_provider_readiness(...).api_key_source` (already computed two lines later at `:12522`) and delete the raw branch.
- Size: S · ADR: no · Confidence: inferred — command: `cd $WT && source env.sh && $PY -c "...screen._provider_current_credential_source('google')"` on the shipped template.
- Pinning test: `Tests/Chat/test_provider_test_evidence.py` (evidence identity) — check before changing.
- Already covered: ENTRY-config P1/P2 (the config-side defect)

## Candidate dispositions
| candidate (file:line pattern) | confirmed / retired (why) / unverified (check) |
|---|---|
| dup_shape [49 files, 114] `_start_deferred_audio_service_initialization@app.py …` | not examined — the excerpt row is truncated and names no settings_screen member |
| dup_shape [12 files, 25] `finish_create_projection … _handle_cancel:2602 _handle_discard:2607 _handle_save:2612` | confirmed P3 (D4b) — the 3-way modal family; see finding; a shared base deletes ~4 lines/modal only |
| dup_shape [9 files, 22] `handle_category_search_submitted:24846 handle_provider_search_changed:26841 handle_provider_conflict_return:27234` | retired — 2-line `event.stop(); self._x(event.value)` delegators; shape-only match, nothing to share |
| dup_shape [14 files, 16] `_handle_cancel:2550 …` | confirmed P3 — same modal family as above |
| dup_shape [5 files, 11] `handle_open_backup_restore:23885 …` | retired — 2-line delegator to `app.action_backup_restore()` |
| dup_shape [4 files, 6] `_cancel_return:27267 …` | retired — 2-line nested closure toggling one flag |
| dup_shape [4 files, 4] `_provider_config:12619 …` | retired — 2-line tuple-unpacking delegator; the real logic is `_provider_config_entry:12571` |
| dup_shape [3 files, 3] `_handle_workspace_create_result:23967 …` | confirmed P3 (D4b) — verbatim wrapper ×3; see finding |
| dup_verbatim [9 files, 10] `_handle_cancel:2550 …` | confirmed P3 (same family) |
| dup_verbatim [3 files, 3] `_handle_discard:2607 …` | confirmed P3 (same family) |
| dup_verbatim [2 files, 2] `get_image_generation_config:464` ↔ personas_screen:380 | retired — deliberate per-module ADR-097 lazy seam (6 lines, docstrings differ, each patchable by its own tests) |
| dup_verbatim [2 files, 2] `_handle_cancel:2602` ↔ speech panel `handle_cancel` | confirmed P3 (same family) |
| dup_verbatim [2 files, 2] `_handle_save:2612` ↔ speech panel `handle_save` | confirmed P3 (same family) |
| except_exception_pass :3921, :3925 | confirmed P3 — `except BaseException: pass` in `on_unmount`; see finding |
| except_exception_pass :4329 | retired — awaiting a composition worker; the fallthrough re-lists and the listing "exposes stable failure state" (comment at :4329, verified by reading `_load_tool_profiles_worker:4340`) |
| except_exception_pass :22836 | retired — cosmetic `scroll_visible` in `_reveal_active_button`; benign (tighter `except QueryError` would do) |
| except_exception_pass :24727 | retired — status-text embellishment after a SUCCESSFUL apply (`:24717-24728`); nothing lost |
| except_exception_return_per_file: 9 | retired — :10862/:10914/:10924/:10930 return redacted failure TEXT (correct); :19749/:19773 documented degrade-to-None/[]; :23047 UI scroll; :23507 UI update (should be `QueryError`, P3 consistency only) |
| function_body_import_per_file: 91 | confirmed P3 — 93 statements, ALL resolve (module+name); 8 redundant + 4× `assign_select_value`; see finding |
| legacy_markers_per_file: 13 | retired — 12 are docstring/comment words ("legacy alias" labels, loader fallback notes); the one code item is `legacy_detail:17104` (P3 dead-control note in the comment-rot finding) |
| raw_1024x1024 :20671, :20677 | retired — deliberate exact-unit "N MiB" display of `CanvasLimits`; `Chat/attachment_core._format_size` / `Utils/Utils._format_size_bytes` produce approximate human strings and would change copy |
| run_worker_coroutine_per_file: 20 | confirmed (1 of 20) — `_perform_runtime_source_switch` does sync file I/O on the loop (P2 finding); the other 19 are UI-only, `storage_call`/`asyncio.to_thread`-backed, or await async services (`discover_models`, `probe_settings_endpoint`, `control.run_once`, `service.list_profiles`) |
| try_import_guard :8874 `_appearance_theme_options` | confirmed P3 — dead `except ImportError` on a hard in-repo module (comment-rot finding) |
| (brief item 4) `SearchRAGWindow._run_index_backfill` refs :18258-18315, `asyncio.run` in a screen | retired as a defect — `asyncio.run` at :18326 runs inside `@work(thread=True)` (`:18246`), a worker thread with no running loop; confirmed only as stale comment text (P3) |
| (brief item 6) `set_interval` callbacks with unguarded `query_one` | retired — the one timer (`:3873`, 0.25 s `_poll_subscription_readiness`) writes only via `_set_static_text:10791` (guarded) and reads via `_provider_widget_value:12948` (guarded); `get_provider_readiness` per tick measured at 5-10 µs |
| (brief item 8) `.plain` read-backs / mutable class attrs / `id()` caches / whole-screen `recompose=True` | retired — `grep '\.plain\b'` → 0; class-level mutables are `BINDINGS`/`CSS_PATH` (Textual contract) + two read-only lookup dicts (`TEST_ACTION_LABELS:2763`, `_IMAGE_GEN_INT_GLOBAL_KEYS:6991`, `.get` only); `id(screen)` at :29605/:29640/:29767 is a per-call dedup set, not a cache; no `recompose=True` reactive remains (task-15475), rebuilds are region-scoped `SettingsRegion.refresh(recompose=True)` |
| (brief item 1) settings WRITE paths | retired as D1 — no hand-rolled TOML/`os.replace`; every write goes through `save_settings_to_cli_config` / `apply_settings_mutation_to_cli_config` / `SettingsConfigAdapter.save_sections` (= `save_settings_to_cli_config`, adapter:189-191) / `persist_provider_settings_atomic` / `apply_console_capture_settings` — all sparse locked read-modify-write, sibling keys preserved. The two residual write defects are the ordering race (P2) and the Video Gen non-atomic pair (P2) |

## Verified-fine
- `_poll_subscription_readiness` 4 Hz timer (`:3873`): `get_provider_readiness(...)` measured `openai median=0.010ms`, `anthropic 0.010ms`, `local_ollama 0.008ms`; all `query_one`s guarded; short-circuits on an unchanged `(category, provider, status)` tuple.
- `get_image_generation_config(reload=True)` / `get_video_generation_config(reload=True)` (`:6782`, `:7092`, `:7132`, `:7528`): 10.3 / 11.0 ms, called on category open, save/revert, and Test click — never per keystroke.
- `load_snapshot_preferences()` in the compose path (`:15711`): cache-backed (`get_cli_setting`, snapshot_settings.py:29-30), no disk read of its own.
- `run_worker(exclusive=True)` always carries `group=` (all 20 sites read); the two `exclusive=False` no-group calls (`:29151`, `:29170`) are fine by definition.
- All 44 `@work` writers/probes that touch config, Chroma, sqlite, HTTP, or the filesystem are `thread=True` (15 writers, backfill, index status, storage/privacy/diagnostics checks, custom endpoints ×4, RAG CRUD, audio.cpp review) — the shipped norm the P2 D2 finding's four sites deviate from.
- `_image_gen_raw_section_cache` (`:3333-3348`, `:6762`): the per-visit cache with three documented invalidation points — the model the P1 fix should copy.
- `ShadowRepoService().available` in compose (`:19468`): `Path.exists`/`shutil.which` only (change_tracking.py:186-190), no subprocess.
- `SettingsConfigAdapter.save_sections` (adapter:189-191) is a one-line delegate to `save_settings_to_cli_config` — one seam, not a parallel writer.
- Secrets: every `logger.*` on provider/backfill/discovery failure paths logs `type(exc).__name__` only (`:7176`, `:13838`, `:13983`, `:18339`); `_mask_url_userinfo:2399` covers the `user:pass@host` gap `redact_secret_text` misses; `env_present` reporting at `:14333-14337` emits `<redacted>` for custom-named env vars.

## Retired
- "`asyncio.run` inside a Textual screen raises RuntimeError" — retired: `:18326` executes on the `@work(thread=True)` worker thread (`:18246`); only the docstring's `SearchRAGWindow` reference is stale (P3 comment rot).
- "`legacy_detail` hidden Select is dead code" — partly retired: it IS read at `:24862`/`:30291`, so deleting it needs the apply path to read the policy object instead; kept as a P3 note, not a dead-code finding.
- "`_persist_remote_images_toggle` / `_persist_status_row_position` race like the permission-summary writer" — retired: button-driven (one write per click, no per-keystroke overlap window), and they write a single key each; ordering cannot produce a stale full-section snapshot.
- "Class-level `_permission_summary_persist_generation: int = 0`" — retired: int is immutable; per-instance `+= 1` rebinds on the instance.
- "`_workspace_worker_connection` leaks a thread-local sqlite connection" — retired: it closes only a connection the worker itself opened (`owned` predicate at `:2659-2663`).

## Left UNVERIFIED
| claim | why not verified | literal command to run |
|---|---|---|
| The absolute per-keystroke cost (388-426 ms) holds on a real, settled user profile — in the isolated scratch profile every `get_cli_setting` costs 10-14 ms and 482 `posix.open` calls (`config.py:8447` → `Backup_Recovery` admission → `storage_admission.py:835/854`), which inflates every adapter read. The 7×/4×/32× call multiplier is structural and env-independent; the ms are not. ENTRY-config owns the config-side cost (their report §"ADR-126 admission", not the per-call price). | Brief forbids running against the real profile (config.py exits with `Recovery required`); no settled profile is available in this env. | On a machine with a settled profile: `cd $WT && $PY - <<'EOF'` → `import time; from tldw_chatbook.config import get_cli_setting; t=time.perf_counter(); [get_cli_setting("console","max_parallel_runs",1) for _ in range(100)]; print((time.perf_counter()-t)*10, "ms/call")` — if ≪ 1 ms, re-run the keystroke probe in this report; the finding stays P1 only if the keystroke is still ≥ 100 ms, else downgrade to P2 (still 7 loads per keystroke). |
| Library/RAG category open visibly stalls (compose ≥ 156 ms of reads + widget build) | live surface; app not run | `.claude/skills/verify/SKILL.md` recipe: `tmux -L verify new -d -x 190 -y 55 'cd $WT && $PY -m tldw_chatbook.app'`, navigate Settings → RAG, then `tmux -L verify send-keys` a digit into "Default results" and `capture-pane` at 100 ms intervals — the rail should visibly freeze between keystrokes. |
| Video Gen half-applied save (edit landed, Clear not) on a failed second replacement | needs an injected failure in `delete_settings_from_cli_config` mid-loop | `cd $WT && source env.sh && $PY -m pytest Tests/UI/ -q --collect-only \| rg video_gen` to find the save test, then monkeypatch `SettingsConfigAdapter.delete_values` to return False after `save_sections` succeeded and assert config.toml still carries the edit. |
| Stuck `AUDIO_CPP_MODEL_LIBRARY_RESULT` claim after a swallowed release failure in `on_unmount` | consequence not reproduced | see the P3 finding's command. |
| Workspaces compose cost at realistic N | not timed against a populated `WorkspaceDB` | see the P3 D2 finding's command. |
| `_provider_current_credential_source("google")` returns `"stored"` on the shipped template | needs a sync-built screen with `app_config` = the shipped template | `cd $WT && source env.sh && $PY - <<'EOF'` → build `SettingsScreen(SimpleNamespace(app_config=load_settings()))`, monkeypatch `SettingsScreen.app` to a stub (as `Tests/UI/test_settings_rag_profile_region.py::fake_app` does), call `screen._provider_current_credential_source("google")`. |
