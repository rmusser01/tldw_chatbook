# Plan: S1 — A discovered model you choose becomes the default, and saved models stay saved and findable

> **Owner rulings, 2026-10-14.**
> - "Pick = use as default" (owner choice): a discovered model you choose becomes the default, and it joins the provider's saved list; Save selected becomes the secondary "Add to model list".
> - OD-1: **(a)**, a frozen set of the 13 legacy sample IDs, removed on the next Settings write and never rewritten at startup. This was the recommendation; the owner was told and did not object.
> - Sequencing: this stream starts after PR #3057 merges (auto-merge armed 2026-10-14 by owner go-ahead).

Source: the Settings UX review of 2026-10-14 (critique snapshot `.impeccable/critique/2026-10-14T15-26-24Z__tldw-chatbook-ui-screens-settings-screen-py.md`), reviewed on origin/dev 1ef4fcf4ae.

Tasks: TASK-35010 (Saved model lists keep what the config file holds, and shipped sample model ids never pose as saved models or defaults), TASK-35011 (Choosing a discovered model as the default adds it to the provider's saved model list when Settings saves), TASK-35012 (Switch model lists each provider's saved default among its first rows).

Source: the verified Settings UX review of origin/dev 1ef4fcf4ae (2026-10-14). Its output is `wf-result.json`, holding assessments A/B/C, their adversarial verdicts and the completeness critic. Only CONFIRMED findings are used, each with its verifier's corrected cause. A2 is the exception: it was rated BY-DESIGN, and the owner has now changed that design.

Builds on: PR #3057 (Phase 7, TASK-33007; head 29ca19f10d, still OPEN at planning time). Its plan is backlog/docs/plan-2026-10-03-model-config-p7.md.

Owner decision already made (2026-10-14), "Pick = use as default": choosing a discovered model makes it the default model, as a draft protected by P7's quit guard, and also adds it to the provider's saved model list. "Save selected" becomes a secondary **Add to model list** action for adding several at once.

## Goal

After S1, saved model lists can be trusted:

- **The file wins.** Every Settings write to a provider's saved list starts from what the config file holds at that moment. It never starts from the startup snapshot, so it cannot drop later edits or bring back deleted models (C3).
- **No fake models.** The shipped sample ids (vllm-model-z, tabby-model, aphrodite-engine, custom-model-alpha and the rest) never pose as saved models or defaults (A-pm-design-M2).
- **One save does both.** A user who discovers models and chooses one ends up, after one `s`, with that model as the default for new chats and in the provider's saved list.
- **Findable.** That model shows in Switch model without typing (C-M1).

## Assumptions & sequencing

- **#3057 merges first, as reviewed.** Re-anchor by name if its later rounds move these symbols. On `origin/feat/model-config-p7-settings-pm` (29ca19f10d), S1 builds on:
  - `UI/Settings_Modules/providers_models_card.py`:
    - `DefaultModelPicker`;
    - `SAVED_MODELS_DISCLOSURE_ID`, `ADVANCED_DISCLOSURES`, `advanced_disclosure` and `refresh_advanced`;
    - `saved_models_summary`, plus `saved_models_not_listed` / `SAVED_NOT_LISTED_ID`;
    - in `compose_providers_models_card`, the `with advanced_disclosure(screen, SAVED_MODELS_DISCLOSURE_ID)` block: `#settings-discover-provider-models`, `#settings-save-discovered-provider-models`, `#settings-clear-discovered-provider-models` and `#settings-discovered-models-list`;
    - in `compose_providers_models_inspector`, the Key block with `#settings-test-provider-guidance`.
  - `UI/Settings_Modules/settings_quit_guard.py`: `confirm_settings_quit`, `KeptSettingsDraft` and `kept_settings_draft`.
  - `SettingsScreen`, model choice and draft state:
    - `on_model_search_picker_model_selected`, `on_model_search_picker_model_value_changed`, `handle_model_value_changed`, `_set_model_field_value`, `_sync_model_picker`;
    - `_provider_saved_model_ids`, `_discovered_model_is_saved`, `_provider_catalog_model_default`, `_provider_model_default`, `_provider_form_values_from_widgets`;
    - `save_state` and `restore_state`.
  - `SettingsScreen`, the Save selected flow: `_refresh_model_discovery_widgets`, `_save_selected_discovered_provider_models` (and its worker and `_discovered_models_save_lock`), `_append_saved_discovered_models`, `_model_to_activate_after_save`, `_activate_saved_model_if_field_empty`, `_report_discovered_models_save_failure`.
  - `SettingsScreen`, saving and the Inspector:
    - the `PROVIDERS_MODELS` branch of `action_settings_save_category`, which calls `persist_provider_settings_atomic` and then `_apply_sparse_config_mutation_in_memory`;
    - `_provider_field_guidance_rows`, `_provider_field_guidance_rows_base`, `_with_save_behavior_row` and `_guided_field_id`;
    - `_raw_config_model` and `_reload_current_config`.
  - The constant `MODEL_DISCOVERY_CAPABILITY_WARNING`.
- **Shared code S1 uses, also checked at 29ca19f10d:**
  - `Chat/provider_setup_persistence.py`: `persist_provider_settings_atomic(..., locked_snapshot_precondition=)` (the parameter exists; Settings does not pass it today), `_validate_combined_provider_settings_mutation` and `resolve_remembered_provider_model`.
  - `LLM_Provider_Catalog/model_discovery_persistence.py`: `persist_discovered_models_to_settings`, `_default_save_callback` and `append_models_to_provider_list`.
  - `LocalLLMProviderCatalogService`: `persist_discovered_models_to_settings` and `refresh_stale_configured_providers`.
  - `config.py`: `get_atomic_config_snapshot`, `apply_literal_settings_transaction_to_cli_config` with `LiteralSettingsMutation` (precedent: the hooks writer in config.py and `Chat/console_settings_defaults.py`), `get_cli_providers_and_models` and `CONFIG_TOML_CONTENT`.
  - `UI/Screens/provider_model_resolution.py`: `MODEL_CAPABILITY_UNKNOWN_WARNING`.
  - `Chat/console_session_settings.py`: `provider_left_at_shipped_default`, `configured_provider_model` and `resolve_effective_chat_configuration`.
- **Phase 8, used by TASK-35012** (local branch `feat/model-config-p8-first-run`, not yet restacked on #3057). In `ConsoleModelPopover`:
  - `_returned_saved_model`, which resolves through `resolve_effective_chat_configuration(self._app_config, provider=key).model`;
  - `_models_for`, `_saved_models` and `_capped`;
  - `TOP_MODELS_PER_PROVIDER`;
  - the ready-provider branch ("ADR-020: the active model stays listed, first").
- **Size facts at #3057's head:**
  - `settings_screen.py` (35,767) and `providers_models_card.py` (2,468) each sit exactly on their row in `Tests/Architecture/test_module_size_ratchet.py`. A row more than 50 lines above its file also fails (`test_budget_is_not_left_slack`).
  - `app.py` already measures 5,824 against its 5,712 row on dev and on P7, and `app_feature_glue.py` sits exactly on its 742 row, so S1 edits neither file.
  - `console_model_popover.py` and `config.py` have no row.
  - P8 and P9 restacks change these numbers, so re-measure at each branch point.
- **Order:**
  - TASK-35010 branches from dev after #3057 merges.
  - TASK-35011 follows TASK-35010, because it uses T1's refresh and write rules.
  - TASK-35012 follows TASK-35010 (the sample-id set) and Phase 8 (TASK-33008), because the popover holds +451 lines of P8 work.
  - T2 and T3 touch disjoint files and can run in parallel.
- **Phase 9 (TASK-33641, local branch `feat/model-config-p9-riders`) is not a dependency.** It shares these seams:
  - Task 7 (TASK-33005.8): the `s` save's success branch, which T2 extends.
  - Task 16 (TASK-33005.12, TASK-33641.9): `console_model_popover.py`, beside T3.
  - Task 14: edits `.github/workflows/derived-artifacts.yml`, which S1 does not touch.

  Whichever PR lands second rebases. No rule from either side changes.
- **No dev drift.** origin/dev (67136d5dd7 at planning) has no change since 1ef4fcf4ae in any file S1 touches.
- Merge mechanics follow CLAUDE.md "Merging into dev" (queue on: arm only after Qodo has reviewed the head and every thread is resolved).

## Scope

- **TASK-35010** (persistence and config, no new UI):
  - Save selected and the startup refresh write to the list as the file holds it, read under the config write lock (C3).
  - `app.providers_models` follows the file after every Settings list write, Advanced Config save and Config reload (C3, corrected cause).
  - The shipped sample ids leave the template. Existing files are cleaned in two ways: their sample ids are ignored when read, and dropped on the next Settings write (M2).
- **TASK-35011** (the owner decision, plus its controls):
  - A pick joins the list on `s`, and it survives leaving Settings.
  - **Add to model list**: label, instant-apply hint and receipts (A2, C4 residual).
  - Tab order (B6), Inspector rows (B19) and the capability line (C-M2).
  - The ADR-002 amendment.
- **TASK-35012:** Switch model shows each provider's saved default first (C-M1).

## Out of scope (and why)

- **Closed by P7:**
  - C1/A1/B8 (rows never read "saved"): `_discovered_model_is_saved` reads the saved list.
  - C2 (draft lost on quit): `settings_quit_guard.py`. Its `KeptSettingsDraft` also covers quitting from Console, the verifier's corrected path.
  - C5/A3 (free-text Model): `DefaultModelPicker` with "Saved fallback".
  - C4's contradictory toast: P7's one-line receipt.
  - A5, A6, A-pm-design-M3/M4: P7.
- **B19's Test (t) part, closed by P7.** `#settings-test-provider` has `can_focus = False` (not a Tab stop; `t` runs it), and the Inspector's Key block always shows `PROVIDER_TEST_GUIDANCE`.
- **C6** (Chat settings and the wizard never persist a list). It is BY-DESIGN P3, a copy-only change in surfaces other streams own:
  - `console_settings_modal.py` has zero headroom, and P9 Task 6 (TASK-33006.19) reworks its served listings.
  - P8 rewrites the wizard model step (`first_run_model_discovery.py`).
  - Revisit after P8 and P9 land.
- **A7 and A-pm-design-M1** (search index): the findability stream, overlapping TASK-33003.14. S1 keeps every control id.
- **A4/B7/D3** (page order): P7's Advanced disclosures plus the shell-IA stream.
- **The send path.** `configured_provider_model` (Console sends, agent routing, the gateway) and `resolve_remembered_provider_model`'s default behaviour (Dreams stories, briefings) still return a provider table's model even when it is a template sample.
  - Some local servers ignore the model name, so the template value can be what makes a send work today.
  - Dreams raises "provider/model unavailable" on no model.
  - Readiness for template samples is TASK-34100.2 AC#9.
- **Startup auto-refresh rebinding.** `app._init_providers_models` rebinds `app.providers_models` after a refresh write. An open Settings card keeps its pre-refresh Saved fallback until the card rebuilds. This affects cloud providers only, at startup only, and fixing it needs an edit to `app.py`, which is over its size row.
- **TASK-34943 / TASK-34700** (the config cache misses in-place external edits): not fixed here. S1's writes read the file under the write lock, so they no longer depend on that cache. With no in-app write, the in-memory lists stay as stale as the cache.
- **Bulk adds stay behind '… N more'.** A model added only through Add to model list, and not made the default, still sits under '… N more' in Switch model. Find still reaches it, and the ruling makes Add the secondary path.
- **Side observations, not findings:**
  - the legacy `mistral` alias has no `[providers]` key;
  - `s` normalises an untouched `api_url`;
  - live cloud discovery needs keys. Tests stub the transport.

## Owner decisions needed

**OD-1. How sample ids already written into existing config files are recognised.**

- (a) A frozen set of the 13 legacy ids (vLLM ×4, TabbyAPI ×3, Aphrodite ×2, Custom ×2, Custom_2 ×2). The saved-list loader and Settings' default choice ignore them, and Settings drops them from a list the next time it writes that list. Nothing is written at startup.
- (b) A one-time startup migration that rewrites the user's file.
- (c) Fix new profiles only; existing files keep the samples.

Recommendation: **(a)**. It needs a ruling because TASK-34100.2 AC#9 says template samples are recognised "by an explicit template-sample marker, not by comparing values". Points for (a):
- A marker cannot identify ids that existing files already hold, since they carry none.
- The set is frozen. These ids leave the template in TASK-35010, so the set can never drift from it.
- It applies only to model lists and Settings' default choice. AC#9's readiness half and its marker stay with TASK-34100.2.

Known limit of (a): a server that really serves a model with exactly one of these ids cannot have it listed.

## Task-by-task design

### TASK-35010 — Saved model lists keep what the config file holds, and sample ids never pose as saved models or defaults

Closes: C3 (with its corrected reach: deletions are resurrected, and Console and the provider-switch fallback stay stale) and A-pm-design-M2.

**C3: the write reads the file under the lock.** Callers stay unchanged.

1. **`model_discovery_persistence.persist_discovered_models_to_settings`.** When no `save_callback` is injected (the production path for both Save selected and the startup refresh), it does one read-modify-write under the config write lock through `apply_literal_settings_transaction_to_cli_config`:
   - The builder takes the provider's list from the locked snapshot's `effective_values["providers"][list_key]` and appends the selected ids through `append_models_to_provider_list`.
   - The ids actually added, and the "already saved" answer, are decided against that locked list, not against the `providers_config` passed in. `providers_config` is still used to resolve the list key (`resolve_provider_list_key`).
   - There is no read-then-write window, so no conflict path is needed. Because the read is raw under the lock, it does not depend on TASK-34943's cache.
   - A failed write keeps today's `status="error"` result and message.
   - Injected callbacks keep today's contract. `Tests/LLM_Provider_Catalog/*` and the auto-refresh tests inject one.
2. **`refresh_providers_models(app)`.** A new helper in the same module that re-reads `config.get_cli_providers_and_models()` (config imported inside the function) and updates `app.providers_models` **in place**: clear, then update, or assign when it is not a dict.
   - Why in place: `DefaultModelPicker` keeps the dict object it was composed with (`_initial_providers_models`), so rebinding the attribute would leave the picker stale.
   - After an in-app write the writer has republished the caches, so the loader sees the file.
3. **Settings call sites,** all importing the helper inside functions:
   - In `_save_selected_discovered_provider_models`, the `saved` branch calls `refresh_providers_models` instead of `_append_saved_discovered_models`, which is deleted.
   - The two `applied` callbacks built in `_raw_config_model` (Advanced Config save) and the `app_config` assignment in `_reload_current_config` (Config reload) also refresh.
   - Then `_sync_model_picker(provider, reload=True)` and `refresh_advanced` re-read the lists, as the Save selected flow already does.

   Net lines in `settings_screen.py` go down.

**M2: sample ids.**

4. **`config.py`:**
   - `CONFIG_TOML_CONTENT`'s `[providers]` ships `vLLM`, `Custom`, `Custom_2`, `TabbyAPI` and `Aphrodite` as `[]` with a "fills via Discover models" comment. That matches the discovery-first cloud entries, and every reader treats `[]` and `["None"]` alike.
   - The first `LOCAL_PROVIDERS` literal near the top of `config.py` is dead code (rebound from `DEFAULT_CONFIG_FROM_TOML` near the module end, and it spells `Custom-2`). Mirror the change there, so a reader is not misled.
   - Add `SHIPPED_SAMPLE_MODEL_IDS: frozenset[str]` (the 13 ids) beside `get_cli_providers_and_models`.
   - `get_cli_providers_and_models()` drops those ids from every list and keeps every key. An emptied list stays an existing key, so discovery and key resolution are unchanged.

   This loader is the single source of `app.providers_models`. It covers Console Alt+M, the picker's Saved fallback and counts, the wizard's list (`first_run_model_step`), `compact_model_bar` and `Plugins/host_references`.
5. **Settings' default choice.**
   - `resolve_remembered_provider_model` gains a keyword, off by default, that treats a sample id as no remembered model.
   - Only Settings' two callers pass it: `_provider_model_default` (the provider switch) and the empty-Model branch of `_provider_form_values_from_widgets`.
   - `_provider_catalog_model_default` already reads the filtered `app.providers_models`.
   - Dreams (`story_service`) and briefings (`briefing_service`) keep calling it without the keyword, so sends are unchanged.
   - Do not filter in `_safe_model`, which `persist_provider_settings_atomic` uses for validation.
6. **Self-heal on write.**
   - `append_models_to_provider_list` drops sample ids from the one list it rewrites. Both Settings writers (Save selected here, and `s` in T2) go through it. Nothing rewrites the file at startup (OD-1 (a)).
   - The startup refresh also uses this function. Its provider set (`AUTO_REFRESH_PROVIDER_LIST_KEYS`) is cloud-only, and those lists never hold these ids, so ADR-020's append-only write-through is unaffected in practice.
7. **Deliberately untouched:** the template's `[api_settings.*].model` values (`aphrodite-engine`, `tabby-model`, `custom-model-alpha`, `custom-model-gamma`). `provider_left_at_shipped_default` compares each provider table with the template. Changing the template would make every existing untouched table look user-configured and move refusing local defaults from NOT RUNNING to NEEDS SETUP (TASK-33005.6 AC#3).

Files:
- `tldw_chatbook/LLM_Provider_Catalog/model_discovery_persistence.py`
- `tldw_chatbook/config.py`
- `tldw_chatbook/Chat/provider_setup_persistence.py`
- `tldw_chatbook/UI/Screens/settings_screen.py`
- `Tests/Architecture/test_module_size_ratchet.py` (lower the `settings_screen.py` row to its measured size)
- `scripts/ui_pr_gate_census.txt` and `scripts/check_ui_pr_gate_census.py` (`MINIMUM_FILES`)
- `Docs/User_Guide/settings.md`

### TASK-35011 — Choosing a discovered model as the default adds it to the saved list when Settings saves

Closes A2 (the owner decision replaces its by-design verdict), B6, B19 (minus Test, which P7 closed), C-M2 and the C4 residual. Implements the 2026-10-14 ruling and supersedes TASK-33007.3 AC#4. P7's other Saved model list rule stays: AC#9, the TASK-369 fill of an empty Model by Add to model list.

**When the list append happens.** Never at pick time. It happens inside the draft save (`s`), in the same mutation as `chat_defaults` / `api_settings.<p>`.

**How the pick is remembered.** Today `on_model_search_picker_model_selected` only stages `model`. That staging stays, and it now also records a pending addition (provider key and model id). It records one only when the id is in the current listing (`screen._model_discovery_models`) and not in the provider's saved list. Any other pick clears it. Rules:
- **Why record it.** `save_state` does not keep the discovery listing, so a rule derived at save time from the listing would lose the addition after leaving Settings. Leaving and returning is a normal path: `KeptSettingsDraft`'s Keep editing navigates back to a fresh Settings.
- **It survives leaving Settings.** `save_state` and `restore_state` carry the record, following the precedent of `provider_return_continuation`.
- **When it is used.** `s` uses it only while it still matches the staged provider and model.
- **When it is dropped.** On a successful save, and whenever the Providers & Models draft is dropped (Revert, Discard).
- **It is not a draft key,** so the unsaved count stays one per pick, and Revert, Discard and the quit guard need no new handling.
- **No new decision.** The rule is the ruling's wording ("choosing a discovered model") plus ADR-002's "write only selected raw endpoint model IDs". Catalog picks and Custom IDs do not join.

**Mechanism.**

1. **New module `tldw_chatbook/UI/Settings_Modules/saved_model_list.py`.**
   - Imports: the card imports it at module level, because `ADVANCED_DISCLOSURES` references its summary and the card is itself imported lazily. `settings_screen.py` imports it only inside functions.
   - What moves in, verbatim first and then changed: the Saved model list compose block, `saved_models_summary` and `saved_models_not_listed` (from the card); `_activate_saved_model_if_field_empty` (no test names it).
   - What it adds: the pending-addition helpers, the receipt and retry copy for Add to model list, the `s` suffix, the conflict copy, and the Inspector rows (item 6).
   - What stays put: `_save_selected_discovered_provider_models` and `_model_to_activate_after_save`, because tests name them.
2. **The `s` save** (the `PROVIDERS_MODELS` branch of `action_settings_save_category`). When a pending addition matches:
   - **Prepare the addition:**
     - Read the provider's list from `config.get_atomic_config_snapshot()` (under the lock, no cache) and resolve the list key with `resolve_provider_list_key`.
     - If the model is already there, add nothing.
     - Otherwise build the new list with `append_models_to_provider_list`, which also drops sample ids (T1).
     - Put it in `section_values["providers"] = {list_key: new_list}`.
   - **Guard the write.** Pass `persist_provider_settings_atomic(..., locked_snapshot_precondition=...)`, comparing the locked snapshot's list for that key with the one just read. Both sides use the same effective view; never compare with the loader-filtered list.
   - **On `fully_applied`:**
     - `_apply_sparse_config_mutation_in_memory` already applies the section.
     - Then run `refresh_providers_models` (T1), `_sync_model_picker(provider, reload=True)` and `_refresh_model_discovery_widgets()`, so the row reads "saved" and Saved fallback lists the model.
     - Append the suffix to the existing toast, e.g. "· added fake-beta to llama.cpp's model list". Pass `markup=False`, since it interpolates a model id.
   - **On `mutation_result.conflict`:** say "The saved model list changed while saving; nothing was written. Press s again." and keep the draft.
   - **When the key no longer resolves:** save the default and say why the model was not added.
3. **Validator.** `_validate_combined_provider_settings_mutation` gains one `providers` branch, admitted only when all of these hold:
   - `model` is non-empty;
   - the section has exactly one key;
   - that key's owner (`_ownership_for`) is the mutation's own provider;
   - the value is a list of strings that pass `_safe_model` and contains `model`.

   Everything else stays refused. This follows the `auth_source` precedent: `test_combined_boundary_writes_a_known_anthropic_sign_in_choice` / `..._refuses_other_sign_in_writes`.
4. **The Saved model list block:**
   - **Tab order (B6).** Discover models gets its own `settings-input-row`, then `SelectionList#settings-discovered-models-list`, then a row with Add to model list and Clear. Tab becomes Discover → list → Add → Clear.
   - **Ids and CSS.** All ids are kept. The buttons' edges are id-based CSS (`#settings-save-discovered-provider-models` …), and `#settings-advanced-saved-models .settings-input-row` already styles both rows, so no CSS change is expected. If one turns out to be needed, edit `css/features/_settings.tcss` (no new bare-type subject) and rebuild with `python tldw_chatbook/css/build_css.py`.
   - **Label and hint (A2).** The label becomes "Add to model list", with a reworded tooltip. Add a `Static(INSTANT_APPLY_BEHAVIOR_COPY, classes="settings-instant-apply-hint")`, the ADR-033 labelled exception; the class is a standalone rule in `_settings.tcss`.
   - **Title.** `saved_models_summary` appends "· 1 joins on save" while a pending addition matches. It reads only in-memory state, because it runs on every keystroke through `refresh_advanced`.
5. **Receipts (C4 residual).**
   - Add names the models added, or says they were already listed and that a model becomes the default by choosing it in Default model.
   - When it fills an empty Default model (TASK-369, kept), it says that `s` still saves that default.
   - Retry steps say "press Add to model list".
   - Service messages stay.
6. **Inspector (B19).** `_provider_field_guidance_rows_base` gains a branch for `settings-discover-provider-models`, `settings-discovered-models-list`, `settings-save-discovered-provider-models` and `settings-clear-discovered-provider-models`. The rows come from the new module.
   - **Row shape.** Each branch returns the same four content rows as every other branch (Focused setting, Purpose, Saved as, Validation). `_with_save_behavior_row` adds the fifth, Save. That shape is pinned by `test_guidance_row_builders_keep_a_uniform_row_count`.
   - **Save row.** It reads `INSTANT_APPLY_BEHAVIOR_COPY` for Add, and a "saves nothing" line for Discover, the list and Clear.
   - **Focused setting.** `_guided_field_id` then names these controls without further change.
7. **Capability line (C-M2).** Delete Settings' duplicate `MODEL_DISCOVERY_CAPABILITY_WARNING`. The card imports the shared `MODEL_CAPABILITY_UNKNOWN_WARNING` from `provider_model_resolution.py`, which `settings_screen.py` already imports at module scope, so there is no import cost. Reword it once, for example "Capabilities unknown for models no catalog lists; text chat is assumed." It shows on Settings' discovery card and as a per-model warning in Console's model options (Chat settings has no picker since TASK-33006.4).
8. **Size ratchets.** Net lines in `settings_screen.py` and `providers_models_card.py` go down. Lower both rows to their measured sizes in the same commit. Add a birth row for `saved_model_list.py`, following the TASK-33007.8 precedent.
9. **ADR-002 amendment** ("Amendment 2026-10-14, owner decision"):
   - choosing a discovered model as the default persists exactly that id on `s`;
   - Add to model list stays the bulk path;
   - nothing is saved without a user choice;
   - list writes start from the file under the write lock.

   Also note P7 parent AC#9 and TASK-33007.6 AC#3 ("Discover, Save selected and Clear behave as before"): the ruling changes their label and Tab order.

Files:
- `tldw_chatbook/UI/Settings_Modules/saved_model_list.py` (new)
- `tldw_chatbook/UI/Settings_Modules/providers_models_card.py`
- `tldw_chatbook/UI/Screens/settings_screen.py`
- `tldw_chatbook/UI/Screens/provider_model_resolution.py`
- `tldw_chatbook/Chat/provider_setup_persistence.py`
- `Tests/Architecture/test_module_size_ratchet.py`
- `scripts/ui_pr_gate_census.txt` and `scripts/check_ui_pr_gate_census.py`
- `scripts/markup_interpolation_census.tsv`, if a pinned notify or tooltip row moves files
- `backlog/decisions/002-openai-compatible-model-discovery.md`
- `Docs/User_Guide/settings.md`

### TASK-35012 — Switch model lists each provider's saved default among its first rows

Closes C-M1. Verified cause:
- `append_models_to_provider_list` appends at the end;
- `ConsoleModelPopover` shows `visible[:TOP_MODELS_PER_PROVIDER]` (3) in list order and lifts only the current model.

The Settings-side half needs nothing more. After T2 the pick is the remembered default (`build_provider_setup_mutation` writes `api_settings.<p>.model`), which `_provider_model_default` reads before the list order, and T1 removed the samples that led the lists.

**Mechanism** (on P8's popover), in the ready-provider branch:
- After the "ADR-020: the active model stays listed, first" lift, move the provider's saved default to the front.
  - If the current model is on the same provider, the default goes second.
  - The default comes from P8's resolver, `resolve_effective_chat_configuration(self._app_config, provider=key).model`, skipping `SHIPPED_SAMPLE_MODEL_IDS` (T1). Cache it per open, beside `_saved`.
- **Reorder only.** It never adds a model the group's rows do not hold. That leaves:
  - P8's return-from-Settings row (`_returned_saved_model`'s append), which stays as is;
  - TASK-33005.12's probe filtering (P9 Task 16), which decides which rows exist before this ordering runs.
- Typed Find, the cap and the '… N more' count are unchanged.

Files:
- `tldw_chatbook/Widgets/Console/console_model_popover.py`
- `Docs/User_Guide/console.md` (the READY PROVIDERS bullet: "the first three models of each provider…")

## Pinned tests to update (rewrite on purpose, and name each in the PR)

Line numbers are indicative, at P7 29ca19f10d.

**TASK-35010.** After T1, the in-memory lists are re-read from the file after a save, and a fake persist never writes the file:
- `Tests/UI/test_settings_model_discovery_journeys.py`: `test_discovery_keyboard_selection_survives_rebuild_and_save` (~:164, ~:176), `test_late_discovery_cannot_publish_into_a_changed_form` (~:259-260) and `test_discovery_operations_report_failure_and_allow_retry` (~:297). Each asserts `app.providers_models` after a fake persist. Seed the file and persist through the real writer, or assert the persist call and the receipt. This file is outside the census and failed 26/26 at setup with `RecoveryRequired('raw_source_selection_changed')` for both reviewers, so compare failure names against the merge base.
- `Tests/UI/test_settings_configuration_hub.py::test_settings_provider_model_discovery_saves_selected_runtime_models` (~:11590, `app.providers_models["openai"]` after a fake persist).
- `Tests/UI/test_settings_save_selected_not_cancelled.py::test_save_selected_is_not_cancelled_by_discover_or_clear` (~:173, `_SERVED in app.providers_models["OpenAI"]` after `_HeldSaveScope`).

Run these, and edit only if red:
- `Tests/UI/test_settings_advanced_disclosures.py::test_saved_model_list_counts_and_each_row_says_selected_and_saved` (~:234; `saved` is captured after the refresh);
- `Tests/LLM_Provider_Catalog/test_model_discovery_cache_merge_persistence.py`, `test_local_llm_provider_catalog_service.py`, `test_model_auto_refresh.py` and `test_llm_provider_catalog_scope_service.py` (they inject callbacks, whose contract is unchanged);
- the tests that drive `_raw_config_model` or `_reload_current_config`: `test_settings_raw_draft.py`, `test_settings_configuration_hub.py`, `test_settings_capture_journeys.py`, `test_settings_console_storage_journeys.py` and `test_settings_thinking_visibility_journeys.py`;
- `Tests/UI/test_console_switcher_local_probe.py` (`SHIPPED_MODELS` reaches the popover directly; `_with_shipped` copies the template's `api_settings`, which T1 leaves alone);
- `Tests/UI/test_console_session_settings.py::test_console_settings_save_clears_stale_terminal_run_status` (hand-seeded `custom-model-*`);
- `Tests/Wizards/test_first_run_setup_wizard.py` and `Tests/UI/test_first_run_wizard_live_contract.py` (they call `get_cli_providers_and_models`);
- `Tests/Chat/test_provider_setup_persistence.py` (the default path of `resolve_remembered_provider_model` is unchanged);
- `Tests/Architecture/test_module_size_ratchet.py`.

**TASK-35011:**
- `Tests/UI/test_settings_default_model_picker.py::test_choosing_a_discovered_model_replaces_a_set_default_and_saves_with_s`. It asserts that no `providers` section is in the mutation and that `app.providers_models` is unchanged. Under the ruling, the mutation carries the pick appended to the file's list.
- `Tests/UI/test_settings_default_model_picker.py::test_a_discovered_model_becomes_the_saved_default_without_joining_the_list` (TASK-33007.3 AC#10, the real-service test). Rename it `…_and_joins_the_list`. The file's `[providers].OpenAI` must equal the list before plus the served id, and `app.providers_models` must equal the file.
- `Tests/UI/test_settings_advanced_disclosures.py`: `test_save_selected_names_the_provider_in_its_receipt` (the parametrize at ~:902-904) and `test_save_selected_that_saves_nothing_reports_on_its_line_only` (~:951). Move both to the Add vocabulary.
- `Tests/UI/test_settings_save_selected_not_cancelled.py`: the receipt (~:177), `_RESELECT` and the provider-change step ("press Save selected"), the `"Try Save again"` param (~:238) and the composed failure toast (~:291).
- `Tests/UI/test_settings_configuration_hub.py`: the receipt (~:11566), "Try Save again" (~:11692), and `test_settings_provider_model_discovery_controls_render_for_eligible_provider` (the capability copy, ~:11495).
- `Tests/UI/test_console_session_settings.py::test_console_model_resolution_includes_runtime_discovered_models` (capability copy, ~:6065).
- `Tests/UI/test_settings_save_commit_models.py::test_guidance_row_builders_keep_a_uniform_row_count`: add the four control ids to its `provider_field_ids`.
- `Tests/UI/test_settings_provider_keyboard_journeys.py`: its `_tab_to` comment says that reaching Save selected "from the discovered list wraps the whole screen", which is no longer true after B6. Re-run and update the comment; the loop guard stays at 150.

Run these, and edit only if red:
- `Tests/UI/test_settings_model_picker_sync_teardown.py`;
- `Tests/UI/test_settings_advanced_disclosures.py::test_the_cards_buttons_draw_an_edge` (ids unchanged);
- `Tests/UI/test_settings_custom_endpoint_default.py::test_providers_models_locks_a_registry_default_and_links_to_its_editor` (`_REGISTRY_LOCKED_FIELDS`);
- `Tests/UI/test_settings_search_index.py` (`NON_SETTING_CONTROLS` keeps `settings-discovered-models-list`);
- `Tests/UI/test_settings_provider_test_draft.py::test_model_to_activate_after_save_prefers_first_saved_when_field_empty` (the method stays on the screen);
- `Tests/Provider/test_provider_model_resolution.py` (asserts the constant, not its text);
- the combined-boundary tests in `Tests/Chat/test_provider_setup_persistence.py`;
- `python scripts/check_markup_interpolation.py`;
- `Tests/Architecture/test_module_size_ratchet.py`.

**TASK-35012:**
- `Tests/UI/test_console_model_switcher.py::test_switcher_lists_four_groups_of_one_line_pairs` and `::test_typed_rows_are_headed_as_matches_not_top_3_each`. Their `app_config` holds no remembered defaults, so they should stay green; run them.
- P8's `::test_a_ready_returned_provider_lists_the_model_just_saved` (typed path, unchanged) and `Tests/UI/test_console_provider_return.py::test_a_model_saved_in_the_default_model_field_is_the_reopened_highlight`.
- `Tests/UI/test_console_switcher_local_probe.py`: its `chat_defaults` has `tabby-model` (~:439), a sample that is never lifted.
- `Tests/Chat/test_model_switcher_recents.py`.

## New tests (and their lane)

No workflow file is edited: P9 Task 14 edits `derived-artifacts.yml`, and a second branch editing it gets evicted from the merge queue (`evict-workflow-file`). Because of that:
- Every new node goes in a `Tests/UI/` file and joins the UI fast lane through `scripts/ui_pr_gate_census.txt`. `check_ui_pr_gate_census.py` accepts only `Tests/UI/` entries, so non-UI pins such as `Tests/LLM_Provider_Catalog/*` and `Tests/Chat/*` run in no PR lane. That is why the new real-writer cases sit in UI files.
- Each new census entry passes 3 of 3 serial runs on its own (ruling C16), goes at the census tail, and raises `MINIMUM_FILES` by one.
- Every mounted test carries `@private_profile_test`.

**TASK-35010: `Tests/UI/test_settings_saved_lists_follow_file.py`.** It uses the real `LocalLLMProviderCatalogService` and scope service with only the transport stubbed (`Tests/UI/test_settings_provider_key_check.py::_Provider` / `_route_discovery_to`), at 211x44 with the real stylesheet. Configuration is written to the file through the production writer, never by hand-seeding `app.providers_models`. Cases:
1. A model added after start keeps its place through Save selected, in both variants: another process replaces the file through the app's writer, and an in-place edit keeps `st_ino` equal. The negative control is red on #3057's head.
2. A model deleted through Advanced Config stays deleted after the next Save selected. After that Advanced Config save, the picker's Saved fallback, the Saved model list count and a popover built from `app.providers_models` no longer list it, without a recompose (this proves the in-place update).
3. A Config reload refreshes the lists the same way.
4. The startup refresh write-through, through the real writer, keeps an external addition and appends new ids.
5. On an old-sample file, written explicitly because a fresh profile cannot represent an old one:
   - the loader drops the 13 ids and keeps a real id saved beside them;
   - switching to vLLM or Aphrodite leaves Model "not set", and Save selected then fills it;
   - the next Settings write removes the samples from the file;
   - an untouched template `api_settings` stays under NOT RUNNING;
   - `resolve_remembered_provider_model(..)` without the keyword still returns `aphrodite-engine` (the send path is unchanged).
6. `CONFIG_TOML_CONTENT`'s five lists parse as empty.

**TASK-35011: `Tests/UI/test_settings_pick_joins_saved_list.py`.** Same harness rules. Cases:
1. Pick plus `s` adds the model, for llama.cpp (local) and OpenAI (cloud), and the receipt names it.
2. Pick then Revert, Discard, or quit-Discard leaves the file byte-identical.
3. Pick, leave to Console, Ctrl+Q, Keep editing, then `s` still adds the model (`KeptSettingsDraft`).
4. An already-saved id, a catalog id the server did not list, or a Custom ID leaves `[providers]` untouched.
5. A list key removed from the file before `s` still saves the default and gives the "not added" receipt.
6. A list changed between the read and the locked write is refused with the draft and the file intact. Patch the read helper to return a stale list.
7. The "· 1 joins on save" title.
8. The Add to model list label, the instant hint, its Inspector Save row and its receipts, including the already-listed and filled-empty-default receipts.
9. Tab order Discover → list → Add → Clear.
10. Inspector rows for the four controls.
11. The capability line on Settings and Console.
12. The validator's `providers` branch admits only the mutation's own key and a list holding the model, and refuses another provider's key, a list without the model and a non-list (direct function cases).

**TASK-35012:** new nodes in `Tests/UI/test_console_model_switcher.py`, which is already in the census:
1. A default saved last of four shows in the top three.
2. With the current model on the same provider, the default is second.
3. A sample-id default is never lifted.
4. Rows are only reordered, and the "… N more" count holds.
5. Typed ranking is unchanged.

Commands: run the covering files once at HEAD, always through the scratch-HOME wrapper, with xdist (`-n 6 --dist loadfile`). `Tests/Architecture` runs serially. Then run `PYTHON=<venv> ./scripts/preflight.sh` (rc 0, not piped through tail). Measure `Tests/Performance/test_ui_ready_module_census.py` and `test_screen_preimport_payload_budget.py` with `PYTHONPATH=<worktree>`.

## Docs

`Docs/User_Guide/settings.md`, content only:
- **(T1)** The vLLM, TabbyAPI, Aphrodite, Custom and Custom 2 lists start empty and fill through **Discover models**. Old sample ids are ignored, and removed the next time Settings saves that list. **Save selected** / **Add to model list** adds to the list as the file holds it, so a model added or removed elsewhere is respected.
- **(T2) Default model row.** Choosing a discovered model stages it as the default, and **s** saves it and adds it to the provider's saved list. This replaces "it does not add the model to the provider's saved list — only Save selected does that".
- **(T2) Saved model list paragraph:**
  - **Discover models**, then the list, then **Add to model list** (applies immediately; adds several at once) and **Clear**, in that Tab order;
  - the "joins on save" title;
  - the new receipts;
  - the empty-default fill.

`Docs/User_Guide/console.md` (T3): READY PROVIDERS lists each provider's saved default first, after the open chat's current model.

ADR-002 amendment (T2).

No "Verified against" paragraphs. What was verified, on which branch and date, goes in each task's Implementation Notes.

## Live verification script

**Sandbox.**
- Use a scratch dir named for the task and sha, e.g. `$SP/s1-t<N>-live-<sha>`, with scratch HOME and XDG dirs. Leave `TLDW_CONFIG_PATH` unset: the app's config is HOME-based at `$V/home/.config/tldw_cli/config.toml`, as both reviewers found.
- Run under `sandbox-exec` with writes denied on `~/.config/tldw_cli`, `~/.local/share/tldw_cli` and `~/.config/tldw-live`.
- Take a `touch $V/baseline` first.

**Fake server.**
- A loopback `/v1/models` server on 127.0.0.1:0 that answers GET only and logs requests, listing `fake-alpha`, `fake-beta` and `fake-gamma`.

**First boot.**
- Boot once to let the app write its config, then quit.
- Make every config edit with a TOML-aware script (`toml` is a dependency). Validate that the file parses with `tomllib` right after each edit and right after the next boot (lessons-live-verification, on config rewrite on boot).

**Driving the app.**
- `tmux -L s1live new-session -d -x 211 -y 44`. Repeat the key frames at **235x52**.
- Send one key per `send-keys`, waiting for the screen to change between keys. A burst reorders bound keys, and plain capture cannot tell a placeholder from a value (lessons-live-verification).
- At each step, take `capture-pane -p` and `-e` captures and a semantic TOML diff of the config.

0. **Baseline, dev after #3057** (the run the critic found missing). Seed the shipped samples and `Llama_cpp = ["None","m-one","m-two","m-three"]`. Capture:
   - Settings ▸ provider vLLM: Model shows `vllm-model-z`;
   - Alt+M: the samples read "Ready · not tested", and a model saved last sits under "… N more".
1. **T1 / C3:**
   - Point llama.cpp's endpoint at the fake server. Advanced ▸ Saved model list ▸ Discover, tick `fake-beta`, Save selected.
   - While the app runs, a second process rewrites the file in place (same inode), adding `hand-added` to `Llama_cpp`. Tick `fake-gamma` and Save selected: the file keeps `hand-added`.
   - Advanced Config: delete `m-two` and save. Alt+M no longer lists `m-two`. Save selected `fake-alpha`: `m-two` does not come back.
2. **T1 / M2, on the old-sample profile:**
   - Provider vLLM: Model reads "not set".
   - Alt+M shows no `vllm-model-*`, `tabby-model*` or `aphrodite-engine*`.
   - Discover, Save selected `fake-alpha`: the file's `vLLM` is `["fake-alpha"]` and Model is filled as a draft.
   - TabbyAPI and Aphrodite stay quietly under NOT RUNNING.
3. **T2, pick then save:**
   - On llama.cpp after Discover, Default model: type `fake-beta`, Enter.
   - Capture Source "edited \*" and the "· 1 joins on save" title. Config diff: none.
   - Press `s`. Capture the toast naming the addition. The config diff shows `chat_defaults.model`, `api_settings.llama_cpp.model` and `[providers].Llama_cpp` + `fake-beta` in one write.
   - Relaunch: Settings shows `fake-beta` ● CURRENT and its row reads "saved".
4. **T2, leave and return:**
   - Pick `fake-gamma`, go to Console, Ctrl+Q, Keep editing. Back in Settings, press `s`. The list gains `fake-gamma`.
   - Separately, pick, Ctrl+Q, Discard: the file is byte-identical.
5. **T2, keyboard and Inspector:**
   - Tab from Discover: list → Add to model list → Clear.
   - Capture the Inspector Focused setting and Save rows for each, and the instant hint.
6. **T3:**
   - With `Llama_cpp = ["None","m-one","m-two","m-three","fake-beta"]` and `fake-beta` the saved llama.cpp default, open Alt+M from an OpenAI chat that holds a message. The llama.cpp group leads with `fake-beta`.

**Cleanup and checks.**
- Kill the fake server by its pid and run `tmux -L s1live kill-server`. Remove only `$V`.
- During the run, `lsof -p <app pid>` shows no handle under the real profile. Afterwards, `find ~/.config/tldw_cli ~/.local/share/tldw_cli ~/.config/tldw-live -newer $V/baseline` prints nothing.
- Keep the captures in the session scratchpad and describe the procedure in the PR. Commit no probe scripts.

## Risks & traps

- **Identity of `app.providers_models`.** `DefaultModelPicker` keeps the dict object it was composed with, so refresh in place and never rebind. The startup refresh still rebinds through `app._init_providers_models`; see Out of scope.
- **Exact size ratchets.** `settings_screen.py` and `providers_models_card.py` sit exactly on their rows, and more than 50 lines of slack also fails. Move code out, then lower the rows in the same commit. `app.py` is already over its row on dev, and `app_feature_glue.py` sits exactly on its row, so edit neither.
- **Moved code** (lessons-testing-evidence, "Tests that read a moved file's source run outside the PR gate").
  - Grep `Tests/` for every moved symbol and for source-reading and AST tests, and run every hit on both trees.
  - Rows in `scripts/markup_interpolation_census.tsv` are keyed by file, so a moved notify or tooltip site moves its row.
- **Imports.**
  - `saved_model_list.py` is imported by the card, which is itself lazy, and only inside functions in `settings_screen.py`. `model_discovery_persistence.py` keeps importing `config` inside functions.
  - Measure the boot census with `PYTHONPATH=<worktree>` (lessons-testing-evidence, "A boot census run from a worktree can measure the main checkout instead").
- **T2's precondition.** Compare the effective view from `get_atomic_config_snapshot` with the writer's locked snapshot, which comes from the same function. Never compare against T1's loader-filtered list, which would refuse every write to a list still holding samples.
- **Template `api_settings`.** Leave its model values alone: `provider_left_at_shipped_default` compares the tables with the template.
- **The send path stays unchanged on purpose.** That is why the Settings-only keyword exists. Do not filter inside `configured_provider_model` or in the default path of `resolve_remembered_provider_model`.
- **Template defaults cannot be mutation-tested through a fresh config** (lessons-testing-evidence, same title). Write the old-sample file in the test.
- **Fakes and synthetic configs.** See lessons-testing-evidence: "A fake written to match your call site validates the mistake", "A synthetic test config lets tests pin states no user can reach", and "Persist test configuration through the same durable seam production reloads".
  - Once T1 lands, a test that hand-seeds `app.providers_models` sees the file's lists after any refresh, so seed the file instead.
  - Persistence claims need the real service and the real writer.
- **`RecoveryRequired` at setup** in the journeys file is a profile-selection trip (lessons-testing-evidence, "Tests/UI `RecoveryRequired` at setup…"). Compare failure names against the branch's merge base, not counts and not whatever `origin/dev` is now.
- **Findings may already be fixed on your base** (lessons-testing-evidence, "A finding verified on the review's commit may already be fixed on your base"). Re-check each finding at the branch point, because P8 and P9 restack in between.
- **Guidance rows** keep the four-plus-one row shape, or `_with_save_behavior_row`'s positional refresh misaligns.
- **Markup.** Receipts that interpolate model ids or provider names use `notify(..., markup=False)`. Tooltips stay static. Run `scripts/check_markup_interpolation.py`.
- **Workflows.** Editing `.github/workflows/*` in parallel with P9 Task 14 triggers `evict-workflow-file`, so use the UI census instead.
- **Config IO on the UI loop.** T2's locked read and T1's locked transaction do guarded IO on the UI loop, as the writers already do. P9 Task 4 (TASK-33641.6) scopes raw reads per task. If it has not landed, expect the same `RecoveryRequired` the writer itself would hit; this is not a new failure mode.
- **Validator hole.** The `providers` branch must admit only the mutation's own key and a list holding the saved model. Test each refusal.
- **Frozen sample set.** It cannot list a real model whose id is exactly one of the 13 (OD-1). Mark it in code with a `ponytail:` comment naming that ceiling.
- **zsh.** `git show "$ref:path"` applies the `:t` modifier; quote it as `"${ref}:path"`.
- **Live.** Never write the real profile. Verify with `lsof` and the `-newer` check, and validate the TOML after each edit and each boot.

## Rollback

Each task is one PR and reverts on its own; nothing needs a schema or data migration.
- **Reverting T1:** writes go back to the snapshot base, and the samples return for new profiles. Lists that were already written or cleaned stay valid data.
- **Reverting T2:** a pick stops joining the list, and the button reads Save selected again. Lists already written stay valid.
- **Reverting T3:** Switch model's groups go back to list order.
