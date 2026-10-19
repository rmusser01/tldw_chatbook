# Plan: S1 — A discovered model you choose becomes the default, and saved models stay saved and findable

> **Owner rulings, 2026-10-14.**
> - "Pick = use as default" (owner choice): a discovered model you choose becomes the default, and it joins the provider's saved list; Save selected becomes the secondary "Add to model list".
> - OD-1: **(a)**, a frozen set of the 13 legacy sample IDs, removed on the next Settings write and never rewritten at startup. This was the recommendation; the owner was told and did not object.
> - Sequencing: this stream starts after PR #3057 merges (owner ruling: land #3057 first). Done: #3057 merged as aabc8db355. S1 now waits for Phase 9 (see Builds on).
>
> **Owner rulings, 2026-10-19** (asked directly, after the plan was re-validated against Phase 9's head e8f475a219):
> - NO-1: **(a)**. In TASK-35010, Switch model also probes URL-based providers that are `model_missing` and still at the shipped default, and lists a refused probe under NOT RUNNING (`console_model_popover.py`, which joins T1's files). AC#9 gets a test case: vLLM with an empty list, one refused server and one running server. If (a) turns out larger than a predicate change plus one branch, the implementer stops and brings option (c) (accept vLLM under NEEDS SETUP and amend AC#9) back to the owner.
> - NO-1 scope (ruled later on 2026-10-19, after the re-review): the probe rule covers **all five** URL-based providers that are `model_missing` at the shipped default on a fresh profile: vllm, llama_cpp, koboldcpp, oobabooga and local_llamafile. On the template config, `resolve_effective_chat_configuration` returns model=None and `provider_left_at_shipped_default` returns True for each of them. Today the four that are not vLLM show under NEEDS SETUP and are never probed. AC#9 is extended to name all five, and its test cases add an untouched llama.cpp and an untouched koboldcpp.
> - NO-2: **(a)**, if at T2's branch point no open or armed PR touches `.github/workflows/derived-artifacts.yml`: a `saved_model_list.py` birth size row plus a PR Fast Lane node (`derived-artifacts.yml` and `FAST_LANE_TARGETS`), as ruling C11 did. Otherwise **(b)**: the row with no PR lane, said in the PR.
> - NO-3: **no gate.** Switch model and Chat settings' pick-mode model picker both get T3's reorder. TASK-35012 AC#1, AC#5 and console.md name both surfaces, and one pick-mode test node pins it.
> - NO-4: C6 stays out of S1. Its copy follow-up is filed as TASK-35578.
> - Implementer defaults D-a, D-b and D-c stand (the owner raised no objection). They are recorded under Decisions.

Source: the Settings UX review of 2026-10-14 (critique snapshot `.impeccable/critique/2026-10-14T15-26-24Z__tldw-chatbook-ui-screens-settings-screen-py.md`), reviewed on origin/dev 1ef4fcf4ae.

Tasks: TASK-35010 (Saved model lists keep what the config file holds, and shipped sample model ids never pose as saved models or defaults), TASK-35011 (Choosing a discovered model as the default adds it to the provider's saved model list when Settings saves), TASK-35012 (Switch model lists each provider's saved default among its first rows).

Source: the verified Settings UX review of origin/dev 1ef4fcf4ae (2026-10-14). Its output is `wf-result.json`, holding assessments A/B/C, their adversarial verdicts and the completeness critic. Only CONFIRMED findings are used, each with its verifier's corrected cause. A2 is the exception: it was rated BY-DESIGN, and the owner has now changed that design.

Builds on: Phase 9 (PR #3169, TASK-33641; head e8f475a219, armed in the merge queue on 2026-10-19). #3057 (Phase 7, TASK-33007) merged as aabc8db355, and Phase 8 (#3144, TASK-33008) as 111f8e48fc. S1 branches from dev after #3169 merges. The plans are backlog/docs/plan-2026-10-03-model-config-p7.md and backlog/docs/plan-2026-10-09-model-config-p9.md. This revision applies the 2026-10-19 re-validation against e8f475a219 (83 plan items checked).

Owner decision already made (2026-10-14), "Pick = use as default": choosing a discovered model makes it the default model, as a draft protected by P7's quit guard, and also adds it to the provider's saved model list. "Save selected" becomes a secondary **Add to model list** action for adding several at once.

## Goal

After S1, saved model lists can be trusted:

- **The file wins.** Every Settings write to a provider's saved list starts from what the config file holds at that moment. It never starts from the startup snapshot, so it cannot drop later edits or bring back deleted models (C3).
- **No fake models.** The shipped sample ids (vllm-model-z, tabby-model, aphrodite-engine, custom-model-alpha and the rest) never pose as saved models or defaults (A-pm-design-M2).
- **One save does both.** A user who discovers models and chooses one ends up, after one `s`, with that model as the default for new chats and in the provider's saved list.
- **Findable.** That model shows in Switch model without typing, when the provider's rows include it (C-M1).

## Assumptions & sequencing

- **Base: dev after #3169 (Phase 9) merges.** #3057 and Phase 8 are merged. The lines below are at e8f475a219. Re-anchor every symbol by name at the branch point: armed #3140 (+130/-129 in `settings_screen.py`, net +1; re-anchor after it) and #3167 (the census floor) may land first. S1 builds on:
  - `UI/Settings_Modules/providers_models_card.py`:
    - `DefaultModelPicker` (:317);
    - `SAVED_MODELS_DISCLOSURE_ID` (:203), `ADVANCED_DISCLOSURES` (:1290), `advanced_disclosure` (:1314) and `refresh_advanced` (:1378);
    - `saved_models_summary` (:1156), plus `saved_models_not_listed` (:1184) / `SAVED_NOT_LISTED_ID` (:204);
    - in `compose_providers_models_card` (:1519), the `with advanced_disclosure(screen, SAVED_MODELS_DISCLOSURE_ID)` block (:2073-2125): `#settings-discover-provider-models`, `#settings-save-discovered-provider-models`, `#settings-clear-discovered-provider-models` and `#settings-discovered-models-list`;
    - in `compose_providers_models_inspector` (:1415), the Key block with `#settings-test-provider-guidance` (:1509).
  - `UI/Settings_Modules/settings_quit_guard.py`: `confirm_settings_quit` (:134), `KeptSettingsDraft` (:159) and `kept_settings_draft` (:184).
  - `SettingsScreen`, model choice and draft state:
    - `on_model_search_picker_model_selected` (:30712), `on_model_search_picker_model_value_changed` (:30726), `handle_model_value_changed` (:30738), `_set_model_field_value` (:16687), `_sync_model_picker` (:16714);
    - `_provider_saved_model_ids` (:15610), `_discovered_model_is_saved` (:15639), `_provider_catalog_model_default` (:15607), `_provider_model_default` (:15653), `_provider_form_values_from_widgets` (:14247);
    - `save_state` (:4446) and `restore_state` (:4505).
  - `SettingsScreen`, the Save selected flow: `_refresh_model_discovery_widgets` (:16517), `_save_selected_discovered_provider_models` (:16780; its worker, and `_discovered_models_save_lock` at :4017), `_append_saved_discovered_models` (:16557), `_model_to_activate_after_save` (:16747), `_activate_saved_model_if_field_empty` (:16765), `_report_discovered_models_save_failure` (:16903).
  - `SettingsScreen`, saving and the Inspector:
    - the `PROVIDERS_MODELS` branch of `action_settings_save_category`, which starts at :33237 and calls `persist_provider_settings_atomic` and then `_apply_sparse_config_mutation_in_memory`:
      - The persist call is at :33672-33678. It passes no `locked_snapshot_precondition`, and a `ValueError` is caught at :33679.
      - The `fully_applied` branch (:33711-33712) now also re-runs the readiness test (TASK-33005.8, :33731-33735). It then refreshes the widgets and the picker (:33742, :33755) and runs P8's return continuation (:33757-33774).
      - The toast is at :33775-33778. It has no `markup=False` and is pinned in `scripts/markup_interpolation_census.tsv:193`.
      - The failure branch is :33779-33792.
    - `_provider_field_guidance_rows` (:18133), `_provider_field_guidance_rows_base` (:18147), `_with_save_behavior_row` (:18120) and `_guided_field_id` (:26134);
    - `_raw_config_model` (:13239) and `_reload_current_config` (:13222).
  - The constant `MODEL_DISCOVERY_CAPABILITY_WARNING` (:1218).
- **Shared code S1 uses, also checked at e8f475a219:**
  - `Chat/provider_setup_persistence.py`: `persist_provider_settings_atomic(..., locked_snapshot_precondition=)` (:1082; the parameter at :1090 exists, and Settings does not pass it today), `_validate_combined_provider_settings_mutation` (:1421), `resolve_remembered_provider_model` (:783) and `_ownership_for` (:200).
  - `LLM_Provider_Catalog/model_discovery_persistence.py` (no commits since the plan): `persist_discovered_models_to_settings` (:80), `_default_save_callback` (:71) and `append_models_to_provider_list` (:32).
  - `LocalLLMProviderCatalogService`: `persist_discovered_models_to_settings` (:1071) and `refresh_stale_configured_providers` (:682).
    - In production its `providers_config` is `dict(app.providers_models)` (app_service_wiring.py:3052-3055).
    - No save callback is injected, so `_default_save_callback` is the production writer.
  - `config.py`: `get_atomic_config_snapshot` (:8010), `apply_literal_settings_transaction_to_cli_config` (:9108) with `LiteralSettingsMutation` (:7876) and `AtomicLiteralMutationSnapshot` (:7884-7891) (precedent: the hooks writer in config.py and `Chat/console_settings_defaults.py`), `get_cli_providers_and_models` (:10003) and `CONFIG_TOML_CONTENT` (:4081).
  - `UI/Screens/provider_model_resolution.py`: `MODEL_CAPABILITY_UNKNOWN_WARNING` (:21).
  - `Chat/console_session_settings.py`: `provider_left_at_shipped_default` (:2638), `configured_provider_model` (:2495) and `resolve_effective_chat_configuration` (:1565).
- **Phase 8, used by TASK-35012.** Merged (#3144). In `ConsoleModelPopover` (`Widgets/Console/console_model_popover.py`):
  - `_returned_saved_model` (:1234), which resolves through `resolve_effective_chat_configuration(self._app_config, provider=key).model`;
  - `_models_for` (:1143), `_saved` (:681), `_saved_models` (:1119) and `_capped` (:1242);
  - `TOP_MODELS_PER_PROVIDER` (:131);
  - the ready-provider branch ("ADR-020: the active model stays listed, first", :1406).

  P9 reshaped two of these:
  - `_models_for` (TASK-33005.12): once a server's listing has answered, it replaces the saved rows.
  - `_pair_score` and `_capped` (TASK-33641.9).
- **Size facts at e8f475a219** (measured with `splitlines`):
  - These four sit on or near their rows in `Tests/Architecture/test_module_size_ratchet.py`:
    - `settings_screen.py`: 35,770 of 35,770 (:252), no headroom;
    - `providers_models_card.py`: 2,465 of 2,468 (:289), 3 lines of slack;
    - `app.py`: 5,710 of 5,710 (:76), exactly on its row and no longer over it;
    - `app_feature_glue.py`: 742 of 742 (:104).

    A row more than 50 lines above its file also fails (`test_budget_is_not_left_slack`).
  - These have no row: `console_model_popover.py` (2,462), `config.py` (12,151), `provider_setup_persistence.py` (1,896), `model_discovery_persistence.py` (145), `Backup_Recovery/unsaved_editors.py` (791) and `provider_model_resolution.py` (394).
  - Seven ratchet rows are red at P9 and at its base, none of them an S1 file.
  - Re-measure after #3169 and #3140, and at each branch point.
- **Order:**
  - TASK-35010 branches from dev after #3169 merges.
  - TASK-35011 follows TASK-35010, because it uses T1's refresh and write rules.
  - TASK-35012 follows TASK-35010 (the sample-id set) and Phase 8 (TASK-33008, satisfied: #3144 merged).
  - All three tasks depend on TASK-33641 (Phase 9).
  - T2 and T3 touch disjoint files, so TASK-35012's dependencies do not name TASK-35011. Land it after TASK-35011 anyway (the shared order below), because the default it lifts is the pick TASK-35011 saves.
  - With NO-1 (a), T1 and T3 both edit `console_model_popover.py`, in different functions. T1 changes `_offer_probes` and `_not_running`; T3 changes the untyped branch of `_build_rows`. Landing them in sequence avoids the conflict.
- **Landing order across the 2026-10-14 Settings streams.** #3057 first. S1 (TASK-35010 → TASK-35011 → TASK-35012) is independent of the shell streams. TASK-35016 and TASK-35017 land before TASK-35013, TASK-35014, TASK-35015, TASK-35018 and TASK-35019. TASK-35020 can land any time after #3057: it fixes the red guards at today's truth (33 categories) and owns the rail-size guard; TASK-35016 adds no category and leaves that guard alone. Armed #3140 adds `SettingsCategoryId.TOOLS`, which moves TASK-35020's "33 categories". TASK-35021 lands after TASK-35016, and TASK-35022 after TASK-35021. TASK-35023 can land any time after #3057.
- **S1 builds on Phase 9 (TASK-33641).** It relies on:
  - the `s` save's success branch with TASK-33005.8's redraw, which T2 extends;
  - TASK-33005.12's listing rows;
  - TASK-33641.9's Find ranking and `_capped`;
  - TASK-33641.6's write-lock reentry refusal (see Risks, "Config IO on the UI loop");
  - P9's Task 14 lane steps in `.github/workflows/derived-artifacts.yml` (:110-240).

  There is no "whichever PR lands second rebases" step.
- **Stacked base.** `feat/s1-saved-models` starts stacked on P9's head e8f475a219, and this plan revision is committed there. Once #3169 merges, `git rebase --onto origin/dev e8f475a219` moves S1's commits onto dev. T1 branches from that.
- **Drift.** Re-check drift at S1's real branch point. Since the plan base, these files have changed on dev and in P9:

  | File | dev commits | P9 commits |
  |---|---|---|
  | `settings_screen.py` | 129 | 18 |
  | `config.py` | 9 | 4 |
  | `settings.md` | 90 | 26 |
  | `model_discovery_persistence.py` | 0 | 0 |
- **Merge mechanics** follow CLAUDE.md "Merging into dev". The queue is `on` (`gh variable get MERGE_QUEUE`). Arm only when all of these hold:
  - the local code review (superpowers:requesting-code-review) has run on the current head;
  - its findings are fixed and recorded in a PR comment;
  - every thread is resolved.

  This replaces the Qodo review (owner rule 2026-10-14: Qodo is out of credits).

## Scope

- **TASK-35010** (persistence and config, no new UI):
  - Save selected and the startup refresh write to the list as the file holds it, read under the config write lock (C3).
  - `app.providers_models` follows the file after every Settings list write, Advanced Config save and Config reload (C3, corrected cause).
  - The shipped sample ids leave the template. Existing files are cleaned in two ways: their sample ids are ignored when read, and dropped on the next Settings write (M2).
  - The Switch model probe checks every untouched local server with no model (vLLM, llama.cpp, koboldcpp, Oobabooga, llamafile). One that is running is listed; one that refuses is under NOT RUNNING (NO-1 (a), its scope widened on 2026-10-19).
- **TASK-35011** (the owner decision, plus its controls):
  - A pick joins the list on `s`, and it survives leaving Settings.
  - **Add to model list**: label, instant-apply hint and receipts (A2, C4 residual).
  - Tab order (B6), Inspector rows (B19) and the capability line (C-M2).
  - The ADR-002 amendment.
- **TASK-35012:** Switch model, and Chat settings' pick-mode model picker (NO-3), show each provider's saved default first when it is one of the group's rows (C-M1). For a local server whose listing has answered, the rows are what that server listed (TASK-33005.12), so T3 orders them; it never adds a row.

## Out of scope (and why)

- **Closed by P7:**
  - C1/A1/B8 (rows never read "saved"): `_discovered_model_is_saved` reads the saved list.
  - C2 (draft lost on quit): `settings_quit_guard.py`. Its `KeptSettingsDraft` also covers quitting from Console, the verifier's corrected path.
  - C5/A3 (free-text Model): `DefaultModelPicker` with "Saved fallback".
  - C4's contradictory toast: P7's one-line receipt.
  - A5, A6, A-pm-design-M3/M4: P7.
- **B19's Test (t) part, closed by P7.** `#settings-test-provider` has `can_focus = False` (not a Tab stop; `t` runs it), and the Inspector's Key block always shows `PROVIDER_TEST_GUIDANCE`.
- **C6** (Chat settings and the wizard never persist a list). It is BY-DESIGN P3, a copy-only change in surfaces other streams own:
  - `console_settings_modal.py` has zero headroom (6,166 of 6,166; ratchet :335), and P9 Task 6 (TASK-33006.19) reworked its served listings per endpoint.
  - P8 rewrote the wizard model step (`first_run_model_discovery.py`). `first_run_model_step.py` also has zero headroom (997 of 997; :397).
  - The "revisit after P8 and P9 land" point has arrived: P8 is merged and P9 is armed. The owner ruled on 2026-10-19 (NO-4) that C6 stays out of S1. Its copy fix is TASK-35578.
- **A7 and A-pm-design-M1** (search index): the findability stream, overlapping TASK-35020 (and TASK-35020.1). TASK-33003.14 is Done and covered only the Console Behavior fallbacks. S1 keeps every control id.
- **A4/B7/D3** (page order): P7's Advanced disclosures plus the shell-IA stream.
- **The send path.** `configured_provider_model` (Console sends, agent routing, the gateway) and `resolve_remembered_provider_model`'s default behaviour (Dreams stories, briefings) still return a provider table's model even when it is a template sample.
  - Some local servers ignore the model name, so the template value can be what makes a send work today.
  - Dreams raises "provider/model unavailable" on no model.
  - Readiness for template samples is TASK-34100.2 AC#9.
- **Startup auto-refresh rebinding.** `app._init_providers_models` rebinds `app.providers_models` after a refresh write. An open Settings card keeps its pre-refresh Saved fallback until the card rebuilds. This affects cloud providers only, at startup only. Fixing it needs an edit to `app_feature_glue.py:585` (`on_config_saved=self._init_providers_models`) or to `app.py:2094-2103`. Both files are exactly on their rows, and any swap needs at least one import line.
- **TASK-34943 / TASK-34700** (the config cache misses in-place external edits): not fixed here. S1's writes read the file under the write lock, so they no longer depend on that cache. With no in-app write, the in-memory lists stay as stale as the cache. Config reload is not an in-app write, and it does read through that cache; see T1 step 2.
- **Bulk adds stay behind '… N more'.** A model added only through Add to model list, and not made the default, still sits under '… N more' in Switch model. Find still reaches it, and the ruling makes Add the secondary path. On an answered local server, a bulk-added model is a row only if the server lists it.
- **Side observations, not findings:**
  - the legacy `mistral` alias has no `[providers]` key;
  - `s` normalises an untouched `api_url`;
  - live cloud discovery needs keys. Tests stub the transport.

## Decisions (ruled 2026-10-14)

**"Pick = use as default"** (owner choice): see the block at the top. TASK-35011 implements it and records it in ADR-002. It supersedes two P7 criteria:
- TASK-33007.3 AC#4;
- the "[providers] list did not change" clause of TASK-33007.3 AC#10. Its real-service test is `test_settings_default_model_picker.py:517`, asserting at :561-562.

**OD-1. How sample ids already written into existing config files are recognised. Ruled: (a)** (the recommendation; the owner was told and did not object).

- (a) **Ruled.** A frozen set of the 13 legacy ids (vLLM ×4, TabbyAPI ×3, Aphrodite ×2, Custom ×2, Custom_2 ×2). The saved-list loader and Settings' default choice ignore them, and Settings drops them from a list the next time it writes that list. Nothing is written at startup.
- (b) Not chosen. A one-time startup migration that rewrites the user's file.
- (c) Not chosen. Fix new profiles only; existing files keep the samples.

**A deliberate, scoped departure from TASK-34100.2 AC#9.** That AC says template samples are recognised "by an explicit template-sample marker, not by comparing values", and the frozen set compares values; that is why OD-1 went to the owner. The departure is scoped:
- The frozen set covers only ids that older versions already wrote. Those carry no marker, so a marker cannot identify them.
- The set is frozen. These ids leave the template in TASK-35010, so the set can never drift from it.
- It applies only to model lists and Settings' default choice. TASK-34100.2's marker rule still governs the template, and AC#9's readiness half stays with TASK-34100.2. TASK-34100.2 itself is not edited (other branches carry versions of it).

Known limit of (a): a server that really serves a model with exactly one of these ids cannot have it listed.

**NO-1 to NO-4 (ruled 2026-10-19):** see the block at the top.

**Implementer defaults (decided 2026-10-19; the owner raised no objection):**
- **D-a. TASK-35010 AC#8, a Save selected that adds nothing.** If the locked list still holds sample ids, still write the cleaned list. The receipt then says "already saved" and names only ids actually added.
- **D-b. T3 and session endpoints (TASK-35190).** The lifted default is "the configured connection's saved default". It is matched by id only, within the rows already shown. Accept this, and note it in the PR rather than gating on `pick_base_url`.
- **D-c. T3, the highlight with empty Find.** When there is no PREVIOUS row and no CURRENT row, `_highlight_target` falls back to `selectable[0]` (console_model_popover.py:1828). That row is now the lifted default. Accept this and pin it with one node.

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
   - **The builder is pure.**
     - It reads only `snapshot.effective_values` / `raw_values`.
     - It calls no config reader or writer (`save_settings_to_cli_config`, `get_atomic_config_snapshot`, `refresh_runtime_config_from_cli_config`), and it starts no worker or task.
     - Under TASK-33641.6, such a call raises `RecoveryRequired('raw_operation_provenance_invalid')` at once (config.py:7442-7486).
   - **Result mapping.** `failure_phase is not None` maps to `status="error"`. The receipt says "already saved" when no id was added, not when `changed=False`: under D-a a Save selected that adds nothing but drops sample ids writes (`changed=True`) and still says "already saved".
   - **Missing key.** A key held in neither `raw_values` nor the template returns `missing_provider_key`. For a template key deleted from the file, the base is the template's list, because `deep_merge_dicts` replaces lists (config.py:1048-1056).
2. **`refresh_providers_models(app)`.** A new helper in the same module that re-reads `config.get_cli_providers_and_models()` (config imported inside the function) and updates `app.providers_models` **in place**: clear, then update, or assign when it is not a dict.
   - Why in place: `DefaultModelPicker` keeps the dict object it was composed with (`_initial_providers_models`), so rebinding the attribute would leave the picker stale.
   - **The plain read is enough after an in-app write,** because the writer paths already republish the caches.
   - **Config reload needs a `force` mode.** Config reload goes through `SettingsConfigAdapter.load(force_reload=True)` → `load_cli_config_and_ensure_existence` (settings_config_adapter.py:137-139). That rebuilds only the bootstrap cache, so an in-place edit stays invisible to `get_cli_providers_and_models()`.
   - The `force` mode takes one of two routes:
     - call `config.refresh_runtime_config_from_cli_config()` (config.py:9125). It takes the write lock, so never call it inside a transaction.
     - or read `load_settings(force_reload=True)['providers']` through the same filter.
3. **Settings call sites,** all importing the helper inside functions:
   - In `_save_selected_discovered_provider_models`, the `saved` branch (`if status == "saved":`, :16866) calls `refresh_providers_models` instead of `_append_saved_discovered_models` (called at :16868; defined at :16557-16576, 20 lines), which is deleted.
   - The two `applied` callbacks built in `_raw_config_model` (Advanced Config save, :13243-13251) also refresh.
   - Config reload has two sites, and both use the forced re-read (step 2):
     - the `app_config` assignment in `_reload_current_config` (:13235);
     - `_apply_diagnostics_validation_and_reload_result` (:13178-13185). It is the UI-loop apply of the Diagnostics thread worker (:13200-13209), which starts at :34426. Refresh only in the apply, never in the worker thread.
   - Then `_sync_model_picker(provider, reload=True)` and `refresh_advanced` re-read the lists, as the Save selected flow already does.

   The net change in `settings_screen.py` is ≤ 0 lines.

**M2: sample ids.**

4. **`config.py`:**
   - `CONFIG_TOML_CONTENT`'s `[providers]` ships `vLLM`, `Custom`, `Custom_2`, `TabbyAPI` and `Aphrodite` as `[]` (today config.py:4722-4726), with a "fills via Discover models" comment. That matches the discovery-first cloud entries, and every reader treats `[]` and `["None"]` alike.
   - The first `LOCAL_PROVIDERS` literal near the top of `config.py` (:4052-4074) is dead code. It is rebound from `DEFAULT_CONFIG_FROM_TOML` near the module end, and it spells `Custom-2`. Mirror the change there, so a reader is not misled.
   - Add `SHIPPED_SAMPLE_MODEL_IDS: frozenset[str]` (the 13 ids) beside `get_cli_providers_and_models`.
   - `get_cli_providers_and_models()` drops those ids from every list and keeps every key. An emptied list stays an existing key, so discovery and key resolution are unchanged.
   - Filter inside the existing `get_cli_providers_and_models` (config.py:10003-10022). Add no new loader function: `LLM_Calls/recovery_review.py:686` compares loader identity.

   This loader is the single source of `app.providers_models`. It covers Console Alt+M, the picker's Saved fallback and counts, the wizard's list (`first_run_model_step`), `compact_model_bar` and `Plugins/host_references`.
5. **Settings' default choice.**
   - `resolve_remembered_provider_model` gains a keyword, off by default, that treats a sample id as no remembered model.
   - Only Settings' two callers pass it: `_provider_model_default` (the provider switch, :15655) and the empty-Model branch of `_provider_form_values_from_widgets` (:14275).
   - `_provider_catalog_model_default` already reads the filtered `app.providers_models`.
   - Dreams (`story_service.py:251`) and briefings (`briefing_service.py:351`) keep calling it without the keyword, so sends are unchanged.
   - Keep the keyword out of `build_provider_setup_mutation`, which P9 changed (provider_setup_persistence.py:810 onward, :918).
   - Do not filter in `_safe_model`, which `persist_provider_settings_atomic` uses for validation.
6. **Self-heal on write.**
   - `append_models_to_provider_list` drops sample ids from the one list it rewrites. Both Settings writers (Save selected here, and `s` in T2) go through it. Nothing rewrites the file at startup (OD-1 (a)).
   - A Save selected that adds nothing still writes the cleaned list when the locked list holds sample ids (D-a).
   - The startup refresh also uses this function. Its provider set (`AUTO_REFRESH_PROVIDER_LIST_KEYS`) is cloud-only, and those lists never hold these ids, so ADR-020's append-only write-through is unaffected in practice.
7. **Deliberately untouched:** the template's `[api_settings.*].model` values (`aphrodite-engine`, `tabby-model`, `custom-model-alpha`, `custom-model-gamma`). `provider_left_at_shipped_default` compares each provider table with the template. Changing the template would make every existing untouched table look user-configured and move refusing local defaults from NOT RUNNING to NEEDS SETUP (TASK-33005.6 AC#3).
8. **Untouched local servers with no model are probed (NO-1, ruled (a); scope widened 2026-10-19).**
   - **The problem.**
     - vLLM's template table has `model = ""` (config.py:5295). With `vLLM = []`, `_representative_model` returns None (console_model_popover.py:971-975), and readiness returns `model_missing`.
     - That check runs before the endpoint check (console_session_settings.py:2039, :2072).
     - `_offer_probes` (:1025-1040) probes only the blockers in `_PROBED_BLOCKERS` (:215-217).
     - `_not_running` (:1514-1537) needs `_retries_in_place` (:161-173), and a `model_missing` readiness never has `recovery_action == "retry_connection"`.
     - The result: an untouched vLLM would move from NOT RUNNING to NEEDS SETUP, and a running one would never be found (TASK-33005.6 AC#3).
     - The same gate already holds back four more servers on a fresh profile: llama_cpp (config.py:5235), oobabooga (:5250), koboldcpp (:5266) and local_llamafile (:5383). Each has `model = ""` in its template table and no saved model (its list is `["None"]`). T1's emptied vLLM list makes vLLM the fifth.
   - **The fix,** in `console_model_popover.py`:
     - also probe providers that are `model_missing`, URL-based (`URL_BASED_PROVIDER_KEYS`, Chat/provider_endpoint_contract.py:27-44) and still at the shipped default;
     - list a refused probe under NOT RUNNING. Because the resolver puts `model_missing` before the endpoint result, `_not_running` reads the refusal from the probe outcome, not from `_retries_in_place`.
   - **The five providers.** The rule covers vllm, llama_cpp, koboldcpp, oobabooga and local_llamafile, all in `URL_BASED_PROVIDER_KEYS`.
     - **The four that are not vLLM change behaviour.** Today they show under NEEDS SETUP and are never probed. After T1:
       - a refused one is under NOT RUNNING;
       - a running one is probed and listed.
     - **Each Alt+M on a fresh profile probes four more endpoints:** llama.cpp's root :8080, Oobabooga's :5000, koboldcpp's :5001 and llamafile's :8001.
     - **The probes share the per-open cap.** At most `SWITCHER_PROBES_IN_FLIGHT = 3` run at once (UI/Console_Modules/connection_probe.py:58), so a fresh profile's open takes longer to settle.
     - **The other template servers need nothing.** TabbyAPI, Aphrodite, Custom and Custom 2 carry a model in their template tables (config.py:5310, :5325, :5340, :5355), so they are probed already.
   - **The gating is not a P9 change.** It dates from fa6ab8527f (TASK-33005.5), an ancestor of the plan base. The existing pins cannot catch it, because they seed `vllm=['vendor/model:b']` (Tests/UI/test_console_switcher_local_probe.py:68).
   - **The stop rule,** re-checked against the wider scope.
     - It still holds: the five share one predicate (URL-based, `model_missing`, at the shipped default), so the change stays a predicate plus one branch.
     - If it turns out larger, stop and bring option (c) back to the owner. Under the wider scope, (c) means accepting these servers under NEEDS SETUP and amending AC#9.

Files:
- `tldw_chatbook/LLM_Provider_Catalog/model_discovery_persistence.py`
- `tldw_chatbook/config.py`
- `tldw_chatbook/Chat/provider_setup_persistence.py`
- `tldw_chatbook/UI/Screens/settings_screen.py`
- `tldw_chatbook/Widgets/Console/console_model_popover.py` (NO-1 (a); no size row)
- `Tests/Architecture/test_module_size_ratchet.py` (lower the `settings_screen.py` row to its measured size)
- `scripts/ui_pr_gate_census.txt` and `scripts/check_ui_pr_gate_census.py`. `MINIMUM_FILES` becomes the branch-point value + 1 (it is 360 at P9, check_ui_pr_gate_census.py:427).
- `Docs/User_Guide/settings.md`
- `Docs/User_Guide/console.md` (NO-1's line; see Docs)

New tests stay `@private_profile_test` and are never `bootstrap_profile` (ledger R11, 40c63e17b7).

### TASK-35011 — Choosing a discovered model as the default adds it to the saved list when Settings saves

Closes A2 (the owner decision replaces its by-design verdict), B6, B19 (minus Test, which P7 closed), C-M2 and the C4 residual. Implements the 2026-10-14 ruling and supersedes TASK-33007.3 AC#4 and AC#10's "[providers] list did not change" clause. P7's other Saved model list rule stays: AC#9, the TASK-369 fill of an empty Model by Add to model list.

**When the list append happens.** Never at pick time. It happens inside the draft save (`s`), in the same mutation as `chat_defaults` / `api_settings.<p>`.

**How the pick is remembered.** Today `on_model_search_picker_model_selected` only stages `model`. That staging stays, and it now also records a pending addition (provider key and model id). It records one only when the id is in the current listing (`screen._model_discovery_models`) and not in the provider's saved list. Any other pick clears it. Rules:
- **Why record it.** `save_state` does not keep the discovery listing, so a rule derived at save time from the listing would lose the addition after leaving Settings. Leaving and returning is a normal path: `KeptSettingsDraft`'s Keep editing navigates back to a fresh Settings.
- **Record only a real staging.** Record only after `_set_model_field_value` has really staged the pick, so the PM draft holds model == pick. `handle_model_value_changed` stages nothing for the current model or for a deep link's `_navigation_model` (:30743-30752).
- **"Not saved".** Decide it with `_discovered_model_is_saved(model, saved_ids)`.
- **It survives leaving Settings.** `save_state` and `restore_state` carry the record, following the precedent of `provider_return_continuation`.
  - Register the new key in `SETTINGS_SNAPSHOT_KEYS` (Backup_Recovery/unsaved_editors.py:193-207) as view state, with a shape check in `_settings_snapshot_dirty` (:211).
  - Without that entry, recovery raises "unknown retained Settings field" and refuses every kept visit.
  - The pin `Tests/Backup_Recovery/test_unsaved_editors.py::test_retained_settings_check_knows_every_key_settings_save_state_writes` runs in the PR fast lane (`FAST_LANE_TARGETS`, Tests/CI/test_ci_queue_pressure_contract.py:152).
  - Restore the record before the `provider_return_continuation` block (settings_screen.py:4579-4612), which ends `restore_state` with early returns.
- **When it is used.** `s` uses it only while it still matches the staged provider and model.
- **When it is dropped.**
  - On a successful save.
  - Whenever the Providers & Models draft is dropped (Revert, Discard).
  - On a provider, endpoint or credential edit, and in Revert. Clear it at these callers of `_reset_provider_model_discovery_state`:
    - the provider change (`_apply_provider_value_change`, :30596);
    - the endpoint change (`handle_provider_endpoint_changed`, :30861);
    - `handle_provider_credential_env_var_changed` (:30969);
    - `handle_provider_api_key_changed` (:31010);
    - `handle_provider_api_key_clear_pressed` (:31080);
    - `_revert_category` (:34335);
    - a deep link's provider change (`_apply_navigation_provider_context`, :24635).
  - Clear it at those call sites, never inside `_reset_provider_model_discovery_state`. The reset's other callers must keep the record: resume (:5368; clearing there would break AC#3), Discover (:16589, :16595) and Clear (:16953).
- **It is not a draft key,** so the unsaved count stays one per pick, and the quit guard needs no new handling. Revert and Discard clear the record with the draft (above).
- **No new decision.** The rule is the ruling's wording ("choosing a discovered model") plus ADR-002's "write only selected raw endpoint model IDs". Catalog picks and Custom IDs do not join.

**Mechanism.**

1. **New module `tldw_chatbook/UI/Settings_Modules/saved_model_list.py`.**
   - Imports: the card imports it at module level, because `ADVANCED_DISCLOSURES` references its summary and the card is itself imported lazily. `settings_screen.py` imports it only inside functions.
   - What moves in, verbatim first and then changed: the Saved model list compose block, `saved_models_summary` and `saved_models_not_listed` (from the card); `_activate_saved_model_if_field_empty` (no test names it).
   - What it adds:
     - the pending-addition helpers;
     - the `s`-branch preparation;
     - the receipt and retry copy for Add to model list;
     - the `s` suffix and the conflict copy;
     - the Save-row copy and the Inspector rows (item 6).

     `settings_screen.py` has 0 headroom, so new logic goes here.
   - What stays put: `_save_selected_discovered_provider_models` and `_model_to_activate_after_save`, because tests name them.
2. **The `s` save** (the `PROVIDERS_MODELS` branch of `action_settings_save_category`). When a pending addition matches:
   - **Prepare the addition:**
     - Read the provider's list from `config.get_atomic_config_snapshot()` (under the lock, no cache) and resolve the list key with `resolve_provider_list_key`. The result is an `AtomicConfigSnapshot(generation, values)` (config.py:7868-7873), so read `snapshot.values['providers'][key]`.
     - If the model is already there, add nothing.
     - Otherwise build the new list with `append_models_to_provider_list`, which also drops sample ids (T1).
     - Put it in `section_values["providers"] = {list_key: new_list}`.
   - **Guard the write.** Pass `persist_provider_settings_atomic(..., locked_snapshot_precondition=...)`. It compares the locked snapshot's list for that key with the one just read, reading `snapshot.values['providers'][key]` on both sides.
     - Both sides use the same effective view. Never compare with the loader-filtered list.
     - The precondition must be a pure comparison: no config call, no task, no await (TASK-33641.6).
   - **On `fully_applied`:**
     - `_apply_sparse_config_mutation_in_memory` (:33712) already applies the section.
     - Run `refresh_providers_models` (T1), `_sync_model_picker(provider, reload=True)` and `_refresh_model_discovery_widgets()` right after it. That is before :33742 / :33755 and `_update_draft_status_widgets`, so the title drops "· 1 joins on save". The row then reads "saved", and Saved fallback lists the model.
     - Append the suffix to the existing toast, e.g. "· added fake-beta to llama.cpp's model list". Pass `markup=False`, since it interpolates a model id.
     - The toast row is already pinned at 1 in the markup census (`scripts/markup_interpolation_census.tsv:193`), so the checker will not catch a missing `markup=False`. Prove it with a test that uses a model id containing `[/]`.
   - **On `mutation_result.conflict`:** say "The saved model list changed while saving; nothing was written. Press s again." and keep the draft. Put this branch before the generic "the file was not written" else (:33787-33790).
   - **When the key no longer resolves:** save the default and say why the model was not added. Because of the template merge, "key gone" is reachable only for a non-template key or an ambiguous one (e.g. a `llama_cpp` key beside the template's `Llama_cpp`).
3. **Validator.** `_validate_combined_provider_settings_mutation` gains one `providers` branch, admitted only when all of these hold:
   - `model` is non-empty;
   - the section has exactly one key;
   - `provider_config_key(list_key) == ownership.provider_key` (`provider_config_key` is in Chat/provider_readiness.py:504);
   - the value is a list of strings that pass `_safe_model` and contains `model`.

   Do not decide the owner with `_ownership_for(list_key)`: `_ownership_for('Custom_2')` raises. `_ALIASES` holds 'Custom 2' and 'Custom-2' but not 'Custom_2', and the canonical set holds lowercase 'custom_2' (provider_setup_persistence.py:155-210). It is the only one of 61 template keys that fails. A Custom 2 pick plus `s` would lose both the addition and the default.

   Everything else stays refused. This follows the `auth_source` precedent: `test_combined_boundary_writes_a_known_anthropic_sign_in_choice` / `..._refuses_other_sign_in_writes`.
4. **The Saved model list block** (card :2087-2117 composes the buttons before the list):
   - **Tab order (B6).** Discover models gets its own `settings-input-row`, then `SelectionList#settings-discovered-models-list`, then a row with Add to model list and Clear. Tab becomes Discover → list → Add → Clear.
   - **Ids and CSS.** All ids are kept. The buttons' edges are id-based CSS (`#settings-save-discovered-provider-models` …). The only `#settings-advanced-saved-models .settings-input-row` selector is the compact-width stacking rule under `.settings-workbench-compact` (css/features/_settings.tcss). At 211x44 and 235x52 both rows take the shared `.settings-input-row` rule. The relevant `_settings.tcss` lines are :153, :718-724, :1001-1010 and :1042-1046. No CSS change is expected, but measure both rows at both sizes and at the compact width. If one turns out to be needed, edit `css/features/_settings.tcss` (no new bare-type subject) and rebuild with `python tldw_chatbook/css/build_css.py`.
   - **Label and hint (A2).** The label becomes "Add to model list", with a reworded tooltip. Add a `Static(INSTANT_APPLY_BEHAVIOR_COPY, classes="settings-instant-apply-hint")` (`INSTANT_APPLY_BEHAVIOR_COPY` is settings_screen.py:1285), the ADR-033 labelled exception; the class is a standalone rule in `_settings.tcss`. Also rename "Save selected" in the docstrings and comments at settings_screen.py:4016, :15648, :16721, :16906 and :30718.
   - **Title.** `saved_models_summary` appends "· 1 joins on save" while a pending addition matches. It reads only in-memory state, because it runs on every keystroke through `refresh_advanced`. Measure the longest title plus suffix at 211x44, 235x52 and compact width. The one-row pin is test_settings_advanced_disclosures.py:139-143.
5. **Receipts (C4 residual).** Today's receipt is at settings_screen.py:16869-16874, the retry copy at :16832-16847, the fill (`_activate_saved_model_if_field_empty(saved_model_ids)`) at :16888 and the failure report at :16903-16914.
   - Add names the models added, or says they were already listed and that a model becomes the default by choosing it in Default model.
   - When it fills an empty Default model (TASK-369, kept), it says that `s` still saves that default.
   - Retry steps say "press Add to model list".
   - Service messages stay.
6. **Inspector (B19).**
   - **Where the Save row is chosen.** In `_provider_field_guidance_rows` (:18133-18145), not in `_base`.
     - Add the Add id to its instant-apply set.
     - Return the "saves nothing" copy there for Discover, the list and Clear.
   - **The content rows.** They come from a new branch in `_provider_field_guidance_rows_base` (:18147) for `settings-discover-provider-models`, `settings-discovered-models-list`, `settings-save-discovered-provider-models` and `settings-clear-discovered-provider-models`. The rows come from the new module.
   - **Row shape.** Each branch returns the same four content rows as every other branch (Focused setting, Purpose, Saved as, Validation). `_with_save_behavior_row` adds the fifth, Save. That shape is pinned by `test_guidance_row_builders_keep_a_uniform_row_count`.
   - **Save row.** It reads `INSTANT_APPLY_BEHAVIOR_COPY` for Add, and a "saves nothing" line for Discover, the list and Clear.
   - **Focused setting.** `_guided_field_id` then names these controls without further change.
7. **Capability line (C-M2).** Delete Settings' duplicate `MODEL_DISCOVERY_CAPABILITY_WARNING`. The card imports the shared `MODEL_CAPABILITY_UNKNOWN_WARNING` from `provider_model_resolution.py`, which `settings_screen.py` already imports at module scope, so there is no import cost. Reword it once, for example "Capabilities unknown for models no catalog lists; text chat is assumed."
   - It shows on Settings' discovery card (card :2082-2086) and in Console's Chat settings readiness detail (chat_screen.py:8927-8935).
   - The "capabilities unknown" discovery row labels pinned at test_settings_provider_test_draft.py:1972/:2006 are different copy and stay out of scope.
8. **Size ratchets.**
   - `settings_screen.py` has 0 headroom and the card has +3. Re-measure after T1 lands.
   - Lower both rows in the same commit. A row more than 50 lines above its file fails `test_budget_is_not_left_slack`, which runs in no PR lane for these files.
   - Birth row (NO-2, ruled): add a row for `saved_model_list.py`, following the TASK-33007.8 precedent.
     - If no open or armed PR touches `.github/workflows/derived-artifacts.yml` at T2's branch point, also add its `does_not_grow` node to the PR Fast Lane. That means both `derived-artifacts.yml` and `FAST_LANE_TARGETS` in `Tests/CI/test_ci_queue_pressure_contract.py`, as ruling C11 did for the three Settings modules (:130-132, mirrored at derived-artifacts.yml:189-191). Check changed paths, not titles: `gh pr list --state open --limit 200 --json number,files --jq '.[] | select(any(.files[]; .path==".github/workflows/derived-artifacts.yml")) | .number'` must print nothing (on 2026-10-19 it printed 3169 and 3036). An armed PR is an open PR, so this covers both. `--json files` lists at most 100 files per PR (#3169 shows 100 of 545). For any PR whose `changedFiles` is above 100, list its files with `gh api --paginate repos/rmusser01/tldw_chatbook/pulls/<n>/files --jq '.[].filename'`.
     - Otherwise add the row with no PR lane, run `Tests/Architecture` serially, and say so in the PR.
     - An `evict-workflow-file` eviction is recoverable with one manual rebase.
9. **ADR-002 amendment** (`backlog/decisions/002-openai-compatible-model-discovery.md`, "Amendment 2026-10-14, owner decision"; TASK-35011 AC#14):
   - it records the 2026-10-14 "Pick = use as default" ruling: choosing a discovered model as the default persists exactly that id on `s`, and adds it to the provider's saved list;
   - it says that the ruling supersedes TASK-33007.3 AC#4 (a discovered pick stays off the saved list) and the "[providers] list did not change" clause of TASK-33007.3 AC#10;
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
- `tldw_chatbook/Backup_Recovery/unsaved_editors.py`
- `Tests/Architecture/test_module_size_ratchet.py`
- `Tests/CI/test_ci_queue_pressure_contract.py` and `.github/workflows/derived-artifacts.yml` (only under NO-2 (a))
- `scripts/ui_pr_gate_census.txt` and `scripts/check_ui_pr_gate_census.py`
- `scripts/markup_interpolation_census.tsv`, only if a pinned unescaped site moves to a new module or qualname (see Risks)
- `backlog/decisions/002-openai-compatible-model-discovery.md`
- `Docs/User_Guide/settings.md`

### TASK-35012 — Switch model lists each provider's saved default among its first rows

Closes C-M1. Verified cause:
- `append_models_to_provider_list` appends at the end;
- `ConsoleModelPopover` shows `visible[:TOP_MODELS_PER_PROVIDER]` (3) in list order and lifts only the current model.
- For a reachable local server, "list order" is its listing order (TASK-33005.12).
- Cloud catalogs put saved models first, in saved-list order (provider_model_resolution.py:271-297), so an appended default is still last.

The Settings-side half needs nothing more. After T2 the pick is the remembered default (`build_provider_setup_mutation` writes `api_settings.<p>.model`), which `_provider_model_default` reads before the list order, and T1 removed the samples that led the lists.

**Mechanism** (in the popover's untyped ready-provider branch of `_build_rows`):
- **Where.** Reorder `visible` only: right after console_model_popover.py:1426, and before `ready_rows.extend` (:1427-1430). Move the provider's saved default to the front.
  - Never reorder `models` (:1404-1407), `_models_for` or `_representative_model`.
  - `models` feeds typed matches, sorted by (score, rank, index) (:1418, :1466).
  - TASK-33641.9's exact-name score of -1 (:1199-1202) makes the index decide both the order and the highlight.
- **The position.**
  - The default goes second only when the current model's row leads `visible`; otherwise it goes first.
  - If the default is the current model, nothing moves.
  - A pair already shown under PREVIOUS/RECENT is not repeated (:1426).
- **Which model is the default.** It comes from P8's resolver, `resolve_effective_chat_configuration(self._app_config, provider=key).model`, skipping `SHIPPED_SAMPLE_MODEL_IDS` (T1). Cache it per open, beside `_saved`.
  - The skip is required: eight template api_settings tables resolve to sample ids (config.py:5310, :5325, :5340, :5355, :5398, :5413, :5428, :5470).
  - The hard dependency on T1 stays.
  - Optionally share `_returned_saved_model`'s resolver call (:1234-1240), and filter samples only on T3's path.
- **Reorder only.** It never adds a model the group's rows do not hold. That leaves:
  - P8's return-from-Settings row (`_returned_saved_model`'s append), which stays as is;
  - the probe's rows (TASK-33005.12; console_model_popover.py:1143-1186, :204-210).
    - Before a local probe answers, the rows are the saved and cached models, so the default is lifted.
    - After the answer, the rows are the listing. An unlisted default disappears, and a listed one is lifted within the listing.
- **Typed Find, the cap and the '… N more' count are unchanged.** Never send the untyped branch through `_capped` (:1242-1268, :1429-1431).
- **Both surfaces (NO-3, ruled: no gate).** `open_model_picker` (UI/Console_Modules/model_switcher.py:1018-1065, `pick_only=True` at :1053) builds the same `ConsoleModelPopover`, and `_build_rows` serves both surfaces. So the reorder also applies to Chat settings' model picker. Do not gate it on `self._pick_only` (:637). In pick mode the current pair is Chat settings' draft pair (model_switcher.py:1037, :1059), so "the current model" in the position rule means that pair there. Both surfaces then keep one row order, which matches TASK-33006.4's "lists the pairs Alt+M lists" (model_switcher.py:1028).
- **Session endpoints (D-b).** The lifted default is the configured connection's saved default, matched by id only within the rows already shown. Note this in the PR.
- **Empty-Find highlight (D-c).** With no PREVIOUS and no CURRENT row, `_highlight_target` falls back to `selectable[0]` (:1828), which is now the lifted default. Pin it.

Files:
- `tldw_chatbook/Widgets/Console/console_model_popover.py`
- `Docs/User_Guide/console.md` (the READY PROVIDERS bullet at :832-836, "the first three models of each provider…"). Word it as "then that provider's saved default for new chats, when its rows include it", and name Chat settings' model picker too (NO-3). Optionally do the same at :899.

## Pinned tests to update (rewrite on purpose, and name each in the PR)

Line numbers are indicative, at e8f475a219.

**TASK-35010.** After T1, the in-memory lists are re-read from the file after a save, and a fake persist never writes the file:
- `Tests/UI/test_settings_model_discovery_journeys.py`. Each of these asserts `app.providers_models` after a fake persist:
  - `test_discovery_keyboard_selection_survives_rebuild_and_save` (:133; asserts :164, :176);
  - `test_late_save_keeps_receipt_and_activation_out_of_a_changed_form` (def :229; asserts :259-260). The plan named `test_late_discovery_cannot_publish_into_a_changed_form` (:186) here, but that test asserts no `providers_models`.
  - `test_discovery_operations_report_failure_and_allow_retry` (:265; asserts :297).

  The fixture seeds `providers_models` at :34. Seed the file and persist through the real writer, or assert the persist call and the receipt.

  This file is outside the census and runs in no lane. It fails 26/26 at setup at P9 and at its base, with `RecoveryRequired('raw_source_selection_changed')`, so compare failure names against the merge base. Convert the edited tests to `@private_profile_test`, or move their assertions into T1's new census file.
- `Tests/UI/test_settings_configuration_hub.py::test_settings_provider_model_discovery_saves_selected_runtime_models` (def :11601; seed :11603; asserts `app.providers_models["openai"]` at :11690 after a fake persist). It sits in a bootstrap file, is red at baseline and runs in no lane. Say so in the PR, or move the assertion.
- `Tests/UI/test_settings_save_selected_not_cancelled.py::test_save_selected_is_not_cancelled_by_discover_or_clear` (def :160; asserts `_SERVED in app.providers_models["OpenAI"]` at :173 after `_HeldSaveScope`; census line 438).

Run these, and edit only if red:
- `Tests/UI/test_settings_advanced_disclosures.py::test_saved_model_list_counts_and_each_row_says_selected_and_saved` (:164; `saved` is captured after the refresh);
- `Tests/LLM_Provider_Catalog/test_model_discovery_cache_merge_persistence.py`, `test_local_llm_provider_catalog_service.py`, `test_model_auto_refresh.py` and `test_llm_provider_catalog_scope_service.py` (they inject callbacks, whose contract is unchanged);
- the tests that drive `_raw_config_model` or `_reload_current_config`: `test_settings_raw_draft.py`, `test_settings_configuration_hub.py`, `test_settings_capture_journeys.py`, `test_settings_console_storage_journeys.py` and `test_settings_thinking_visibility_journeys.py`;
- `Tests/UI/test_console_switcher_local_probe.py` (`SHIPPED_MODELS` reaches the popover directly; `_with_shipped` copies the template's `api_settings`, which T1 leaves alone);
- `Tests/UI/test_console_session_settings.py::test_console_settings_save_clears_stale_terminal_run_status` (:10776; hand-seeded `custom-model-*` at :10785-10790);
- `Tests/Wizards/test_first_run_setup_wizard.py` and `Tests/UI/test_first_run_wizard_live_contract.py` (they call `get_cli_providers_and_models`);
- `Tests/UI/test_console_provider_persistence_regressions.py` and `Tests/Plugins/test_native_components.py` (they also call the loader);
- `Tests/Chat/test_provider_setup_persistence.py` (the default path of `resolve_remembered_provider_model` is unchanged);
- `Tests/Architecture/test_module_size_ratchet.py`. Seven rows are base reds: app_lifecycle, personas_screen, console_transcript, mcp_workbench, tldw_api/client, llm_screen and change_review_screen.

NO-1's wider probe rule. A search of `Tests/UI/test_console_model_switcher.py` and `Tests/UI/test_console_switcher_local_probe.py` at e8f475a219 looked for NEEDS SETUP expectations on vllm, llama_cpp, koboldcpp, oobabooga and local_llamafile.
- **No pin found.** No pin expects one of them untouched and model-less under NEEDS SETUP. Each place the five appear carries a saved or table model, or an endpoint that is not the template's (test_console_switcher_local_probe.py:57, :61, :66, :68, :69). `_with_shipped` copies only tabbyapi, aphrodite, local_mlx_lm and `_UNTOUCHED`, whose vllm keeps `vendor/model:b`.
- **No rewritten pin is planned.** If the implementer finds one red because of the wider rule, it is **rewritten on purpose** and named in the PR.
- **Run these too.** The full-app tests that open Switch model on the shipped profile:
  - Tests/ProductionApp/test_provider_selection_ownership.py (the palette);
  - in Tests/UI:
    - test_console_inspector_keyboard_route.py;
    - test_console_internals_decomposition.py;
    - test_console_provider_return.py;
    - test_console_settings_geometry.py;
    - test_console_settings_model_change.py;
    - test_console_settings_saved_defaults.py;
    - test_console_switch_model_entry_points.py;
    - test_console_switch_model_fold_hint.py;
    - test_console_switch_model_keys.py.

  They run with the autouse no-op prober (Tests/conftest.py:942-946), whose probes never settle. A newly offered probe can change how their rows read while pending.

**TASK-35011:**
- `Tests/UI/test_settings_default_model_picker.py::test_choosing_a_discovered_model_replaces_a_set_default_and_saves_with_s` (:229; asserts :257, :269-270). It asserts that no `providers` section is in the mutation and that `app.providers_models` is unchanged. Under the ruling, the mutation carries the pick appended to the file's list.
- `Tests/UI/test_settings_default_model_picker.py::test_a_discovered_model_becomes_the_saved_default_without_joining_the_list` (:517; asserts :561-562; TASK-33007.3 AC#10, the real-service test). Rename it `…_and_joins_the_list`. The file's `[providers].OpenAI` must equal the list before plus the served id, and `app.providers_models` must equal the file.
- `Tests/UI/test_settings_advanced_disclosures.py`: `test_save_selected_names_the_provider_in_its_receipt` (:908; the parametrize at :900-905) and `test_save_selected_that_saves_nothing_reports_on_its_line_only` (:957). Move both to the Add vocabulary.
- `Tests/UI/test_settings_save_selected_not_cancelled.py`: the receipt (:176-177), `_RESELECT` (:225-228) and the provider-change step ("press Save selected", :244-246), the `"Try Save again"` param (:238) and the composed failure toast (:289-293).
- `Tests/UI/test_settings_configuration_hub.py`: the receipt (:11666), "Try Save again" (:11792), and `test_settings_provider_model_discovery_controls_render_for_eligible_provider` (:11555; the capability copy, :11595). No lane.
- `Tests/UI/test_console_session_settings.py::test_console_model_resolution_includes_runtime_discovered_models` (:6046; capability copy, :6067). No lane.
- `Tests/Backup_Recovery/test_unsaved_editors.py::test_retained_settings_check_knows_every_key_settings_save_state_writes` (:974-991). It must pass; it is in `FAST_LANE_TARGETS`.
- Two base reds, which T2 edits only by comparing failure names against the merge base. Both fail alone at P9 and at its base (`RecoveryRequired`), neither is `@private_profile_test`, and neither is in any census. The B6 and B19 proof moves to the new file's cases 9 and 10.
  - `Tests/UI/test_settings_save_commit_models.py::test_guidance_row_builders_keep_a_uniform_row_count` (:196): add the four control ids to its `provider_field_ids` (:205).
  - `Tests/UI/test_settings_provider_keyboard_journeys.py`: its `_tab_to` comment (:36-42) says that reaching Save selected "from the discovered list wraps the whole screen", which is no longer true after B6. Update the comment; the loop guard stays at 150.

Run these, and edit only if red:
- `Tests/UI/test_settings_model_picker_sync_teardown.py`;
- `Tests/UI/test_settings_advanced_disclosures.py::test_the_cards_buttons_draw_an_edge` (:844; ids unchanged);
- `Tests/UI/test_settings_custom_endpoint_default.py::test_providers_models_locks_a_registry_default_and_links_to_its_editor` (:472; `_REGISTRY_LOCKED_FIELDS`);
- `Tests/UI/test_settings_search_index.py` (`NON_SETTING_CONTROLS` keeps `settings-discovered-models-list`). It has 2 base reds (TASK-35020, TASK-35020.1).
- `Tests/UI/test_settings_provider_test_draft.py::test_model_to_activate_after_save_prefers_first_saved_when_field_empty` (:1879; the method stays on the screen);
- `Tests/UI/test_settings_discovery_listing_dropped.py` (ids at :33, :54);
- `Tests/Provider/test_provider_model_resolution.py` (asserts the constant, not its text; :142);
- the combined-boundary tests in `Tests/Chat/test_provider_setup_persistence.py`;
- `python scripts/check_markup_interpolation.py`;
- `Tests/Architecture/test_module_size_ratchet.py`.

**TASK-35012:**
- `Tests/UI/test_console_model_switcher.py::test_switcher_lists_four_groups_of_one_line_pairs` (:310) and `::test_typed_rows_are_headed_as_matches_not_top_3_each` (:484). Their `app_config` holds no remembered defaults, so they should stay green; run them.
- P8's `::test_a_ready_returned_provider_lists_the_model_just_saved` (:2095; typed path, unchanged) and `Tests/UI/test_console_provider_return.py::test_a_model_saved_in_the_default_model_field_is_the_reopened_highlight` (:1559; no lane).
- `Tests/UI/test_console_switcher_local_probe.py`: `SHIPPED_MODELS` (:466-468), and its `chat_defaults` with `tabby-model` (:762, a NEEDS SETUP row), a sample that is never lifted.
- `Tests/Chat/test_model_switcher_recents.py`.

## New tests (and their lane)

No workflow file is edited, unless NO-2 (a) applies (see T2 mechanism 8). `derived-artifacts.yml` and the Tests/CI lane contracts change often (#3036, TASK-35570, TASK-35572, TASK-35575), and a PR that edits a workflow file dev also changed is evicted from the merge queue (`evict-workflow-file`). Lane rules:
- **The UI census.**
  - Census entries must be under `Tests/UI/` (check_ui_pr_gate_census.py:910-911).
  - Non-UI targets can run in the PR Fast Lane (`FAST_LANE_TARGETS`, derived-artifacts.yml:110-240), but only through a workflow edit. That is why the new real-writer cases sit in UI files.
- **Profile markers.**
  - A `bootstrap_profile` node may never join the census (ledger R11). Use `@private_profile_test`.
  - P9 moved its bootstrap nodes to the PR Fast Lane's two admission-sensitive jobs (`pr-fast-lane-admission`, and `pr-fast-lane-admission-2` since TASK-35575; `ADMISSION_TARGETS` / `ADMISSION_2_TARGETS`). S1 does not edit them.
  - TASK-35570 decides whether `bootstrap_profile` tests may run in the census.
- **Placement.** Check it with Tests/CI/test_console_p0_regression_gate.py `_lanes()` / `_selected_by` (:208-354).
- **Each new census entry:**
  - passes 3 of 3 serial runs on its own (ruling C16);
  - is timed in its shard (`entries[index::10]`, tail position) with the lane's install set (`-e . pytest pytest-asyncio pytest-timeout packaging`; lessons-testing-evidence :554, :580);
  - goes at the census tail;
  - raises `MINIMUM_FILES` to the branch-point value + 1. The value is 360 at P9, and #3167 raises dev's floor.
- **Every mounted test** carries `@private_profile_test`. A file without the marker needs the collection-time `import tldw_chatbook.app`, or it trips `RecoveryRequired` at setup.

**TASK-35010: `Tests/UI/test_settings_saved_lists_follow_file.py`.** It uses the real `LocalLLMProviderCatalogService` and scope service with only the transport stubbed (`Tests/UI/test_settings_provider_key_check.py::_Provider` / `_route_discovery_to`), at 211x44 with the real stylesheet. Configuration is written to the file through the production writer, never by hand-seeding `app.providers_models`. Cases:
1. A model added after start keeps its place through Save selected, in both variants: another process replaces the file through the app's writer, and an in-place edit keeps `st_ino` equal.
   - The negative control is red on S1's base (dev after #3169).
   - Make it as a temporary source edit, never as a `-p` plugin (lessons-testing-evidence :951).
2. A model deleted through Advanced Config stays deleted after the next Save selected. After that Advanced Config save, three surfaces no longer list it, without a recompose (this proves the in-place update):
   - the picker's Saved fallback;
   - the Saved model list count;
   - a popover built from `app.providers_models`. It must have no listing evidence: use `connection_prober=None`, or a provider whose rows come from its saved list.
3. A Config reload refreshes the lists the same way:
   - after an in-place edit of the file (the forced re-read, T1 step 2);
   - from both reload sites, including the Diagnostics reload apply.
4. The startup refresh write-through, through the real writer, keeps an external addition and appends new ids.
5. On an old-sample file, written explicitly because a fresh profile cannot represent an old one:
   - the loader drops the 13 ids and keeps a real id saved beside them;
   - switching to vLLM or Aphrodite leaves Model "not set", and Save selected then fills it;
   - the next Settings write removes the samples from the file;
   - an untouched template `api_settings` stays under NOT RUNNING;
   - `resolve_remembered_provider_model(..)` without the keyword still returns `aphrodite-engine` (the send path is unchanged).
6. `CONFIG_TOML_CONTENT`'s five lists parse as empty.
7. NO-1, with the five probed:
   - vLLM with an empty list: a refused server stays under NOT RUNNING, and a running server is probed and listed.
   - An untouched llama.cpp and an untouched koboldcpp, one refused and the other running: the refused one is under NOT RUNNING, and the running one is probed and listed. Neither shows under NEEDS SETUP.
   - The autouse guard in Tests/conftest.py (:942-946) replaces the probe seam with a no-op. These cases inject the real prober by name, as Tests/UI/test_console_switcher_local_probe.py does.
8. A `RecoveryRequired` raised inside the builder maps to `status="error"`.

**TASK-35011: `Tests/UI/test_settings_pick_joins_saved_list.py`.** Same harness rules. Cases:
1. Pick plus `s` adds the model, for llama.cpp (local), OpenAI (cloud) and Custom 2, and the receipt names it.
2. Pick then Revert, Discard, or quit-Discard leaves the file byte-identical.
3. Pick, leave to Console, Ctrl+Q, Keep editing, then `s` still adds the model (`KeptSettingsDraft`).
   - Press `s`, not Enter, after Keep editing: first-paint focus is on `#nav-home` (settings_screen.py:5320-5323).
   - Also round-trip `save_state` / `restore_state` through `_settings_snapshot_dirty`.
4. An already-saved id, a catalog id the server did not list, or a Custom ID leaves `[providers]` untouched.
5. A list key that no longer resolves before `s` still saves the default and gives the "not added" receipt. Build it from an ambiguous key or a non-template key.
6. A list changed between the read and the locked write is refused with the draft and the file intact.
   - Patch only the pre-lock read to return a stale list.
   - Assert that the precondition opens no second config read.
7. The "· 1 joins on save" title.
8. The Add to model list label, the instant hint, its Inspector Save row and its receipts, including the already-listed and filled-empty-default receipts. One receipt uses a model id containing `[/]`, which proves `markup=False`.
9. Tab order Discover → list → Add → Clear.
10. Inspector rows for the four controls.
11. The capability line on Settings and Console.
12. The validator's `providers` branch admits only the mutation's own key and a list holding the model. It refuses another provider's key, a list without the model and a non-list (direct function cases). It admits `Custom_2` for Custom 2 (the `provider_config_key` owner check).

**TASK-35012:** new nodes in `Tests/UI/test_console_model_switcher.py`, which is already in the census (line 39):
1. A default saved last of four shows in the top three.
2. With the current model on the same provider, the default is second.
3. A sample-id default is never lifted. Seed the sample directly in `providers_models` / `served_models`.
4. Rows are only reordered, and the "… N more" count holds.
5. Typed ranking is unchanged. Negative controls: exact-name Find ('Anthropic ') and a prefix tie ('claude').
6. A listing that holds the default 4th or later: the default is lifted.
7. A listing without the default: nothing is added.
8. A Settings return: `_capped`'s first-row keep and `_returned_saved_model` still hold.
9. A default already under RECENT is not repeated.
10. Pick mode (NO-3): Chat settings' model picker shows the same order.
11. Empty Find with no PREVIOUS and no CURRENT row highlights the lifted default (D-c).

Commands:
- Run the covering files once at HEAD, always through the scratch-HOME wrapper, with xdist (`-n 6 --dist loadfile`). `Tests/Architecture` runs serially.
- Probe the Tests/CI lanes.
- Run `PYTHON=<venv> ./scripts/preflight.sh` (rc 0, not piped through tail).
- Measure `Tests/Performance/test_ui_ready_module_census.py` and `test_screen_preimport_payload_budget.py` with `PYTHONPATH=<worktree>`.

## Docs

Each task's User Guide edit below is one of its ACs (TASK-35010 AC#11 and AC#12, TASK-35011 AC#15, TASK-35012 AC#5).

`Docs/User_Guide/settings.md`, content only. The targets are :257 ("only **Save selected** does that"), the Advanced row at :259 (its title wording and "Save selected keeps the ones you select") and :284-301. Re-read all of them on the final base.
- **(T1)** The vLLM, TabbyAPI, Aphrodite, Custom and Custom 2 lists start empty and fill through **Discover models**. Old sample ids are ignored, and removed the next time Settings saves that list. **Save selected** / **Add to model list** adds to the list as the file holds it, so a model added or removed elsewhere is respected.
- **(T2) Default model row.** Choosing a discovered model stages it as the default, and **s** saves it and adds it to the provider's saved list. This replaces "it does not add the model to the provider's saved list — only Save selected does that".
- **(T2) Advanced row (:259).** It moves to the new title wording and the Add to model list vocabulary.
- **(T2) Saved model list paragraph:**
  - **Discover models**, then the list, then **Add to model list** (applies immediately; adds several at once) and **Clear**, in that Tab order;
  - the "joins on save" title;
  - the new receipts;
  - the empty-default fill.

`Docs/User_Guide/console.md` (T1, NO-1; TASK-35010 AC#12): beside :851-856, which already says shipped-default servers "are checked like any other local server", add one line. An untouched local server with no model chosen (vLLM, llama.cpp, koboldcpp, Oobabooga, llamafile) is checked too: it lists its models when it is running, and reads under NOT RUNNING when it refuses.

`Docs/User_Guide/console.md` (T3): READY PROVIDERS lists each provider's saved default first, after the open chat's current model, when it is one of that provider's rows (for an answered local server, when the server listed it). The same order holds in Chat settings' model picker (NO-3). Place it next to console.md:878-891, so the two passages agree. Switch model lives in Console, so this is the matching guide page.

ADR-002 amendment (T2, TASK-35011 AC#14): records the "Pick = use as default" ruling and that it supersedes TASK-33007.3 AC#4 and the "[providers] list did not change" clause of TASK-33007.3 AC#10.

No "Verified against" paragraphs. What was verified, on which branch and date, goes in each task's Implementation Notes.

## Live verification script

**Sandbox.**
- Make the scratch dir with `V=$(mktemp -d)` (under /var/folders), with scratch HOME and XDG dirs. A HOME under /tmp is refused by private-path hardening (lessons-live-verification :4288-4304). Leave `TLDW_CONFIG_PATH` unset: the app's config is HOME-based at `$V/home/.config/tldw_cli/config.toml`, as both reviewers found.
- Run under `sandbox-exec` with writes denied on `~/.config/tldw_cli`, `~/.local/share/tldw_cli` and `~/.config/tldw-live`.
- Take a `touch $V/baseline` first.
- **No real or paid provider is ever contacted.**
  - The scratch profile holds no API keys.
  - The launch env unsets the provider key variables (e.g. `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY …`).
  - The `sandbox-exec` profile also denies outbound network except to loopback.
  - Every send in this script goes to stub C (below).

**Stubs and listeners.** One small loopback stub program, run as separate instances. Each instance logs and counts its requests.
- **Stub A**, on 127.0.0.1:0: answers GET `/v1/models` only, listing `fake-alpha`, `fake-beta` and `fake-gamma` (step 6 lists more).
- **Stub B** (step 2): the same program on 127.0.0.1:8000, vLLM's template port, listing `fake-alpha` and `fake-beta`. Also bind ::1:8000 if `localhost` resolves to it first.
- **Stub C** (step 6): the same program with chat on, listing `stub-chat`. It also answers POST `/v1/chat/completions` with a fixed reply, both streaming (SSE) and non-streaming.
- **The probe's timing,** which every frame below has to fit:
  - **The request.** The switcher's probe goes through `probe_settings_endpoint` (UI/Console_Modules/connection_probe.py:121). Each request times out after 2.5 s (`SETTINGS_ENDPOINT_PROBE_TIMEOUT_SECONDS`, UI/Screens/settings_endpoint_probe.py:49).
  - **A stalled probe reads as timed out about 2.5 s after it starts.**
    - A second request only follows Ollama's non-timeout listing failure (settings_endpoint_probe.py:715-721).
    - A timeout returns `unreachable`. `_not_running` needs `connection_refused`, so a stall never reads as refused.
  - **The cap.** At most 3 probes run at once (`SWITCHER_PROBES_IN_FLIGHT`, connection_probe.py:58). With up to ten stalled targets, the last results arrive several seconds after the open.
  - **The 10 s reuse window.** A probe is not sent again within 10 s (`SWITCHER_PROBE_CACHE_SECONDS`, connection_probe.py:56). The window counts from each probe's queue time and from its result's `observed_at` (connection_probe.py:345-349; `_recently_tested`, :296).
  - **Nothing is skipped.** `_UNPROBED_SERVERS` (console_model_popover.py:220) contains none of the five servers that hold sample ids, nor any of NO-1's five, so all of them are probed.
- **`--hold N`:** any stub delays each `/v1/models` answer by N seconds, so a frame "before the probe settles" can be captured. A loopback stub answers, and a closed port refuses, within milliseconds, so without a hold there is no such frame.
  - Keep N under 2.5 s (e.g. 2), or the hold reads as a timeout.
  - If 2 s is too short to capture, hold longer for the before frame only. Then restart the stub without the hold for the after frame.
- **Stall listener:** for a template port nothing listens on, a temporary TCP listener that accepts and never answers, so the probe stays pending.
  - Capture a stall frame within about 2 s of the open; after that the first probes read as timed out.
  - Or stall only the ports the frame needs, so fewer targets queue behind the cap.
  - Close the listener, then reopen Alt+M, to get the refused frame.
- **Wait for the rows to settle, then 10 s more.** Do this before every reopen that must see a new server state: after closing a stall listener, after starting stub B, and after removing a hold.
  - First wait until every probed row has settled.
  - Then wait 10 s more.
  - Otherwise the open reuses a result.
  - Waiting 10 s after the action alone is not enough. A probe queued late, or a result observed late, starts its own 10 s window then.
- **The template ports.** These are every loopback `api_url` in the template whose provider is in `URL_BASED_PROVIDER_KEYS` (config.py:5234-5427):

  | Port | Providers |
  |---|---|
  | 8080 | llama.cpp, TabbyAPI |
  | 5000 | Oobabooga |
  | 5001 | koboldcpp |
  | 11434 | Ollama |
  | 8000 | vLLM |
  | 2242 | Aphrodite |
  | 1234 | Custom |
  | 5678 | Custom 2, local_ollama |
  | 8001 | local_llamafile, local_llamacpp |
  | 8008 | local_vllm |

  - Before the first boot, `lsof -nP -iTCP:<port> -sTCP:LISTEN` prints nothing for each of them.
  - If one prints a listener, stop: a real local server would answer the probes.
- **Record the mechanism.** Every "before the probe settles" frame records what produced it: a held stub, or a stall listener on which port. A frame with neither is not evidence.

**First boot.**
- Boot once to let the app write its config, then quit.
- Make every config edit with a TOML-aware script (`toml` is a dependency). Validate that the file parses with `tomllib` right after each edit and right after the next boot (lessons-live-verification, on config rewrite on boot).

**Driving the app.**
- `tmux -L s1live new-session -d -x 211 -y 44`. Repeat the key frames at **235x52**.
- Send one key per `send-keys`, waiting for the screen to change between keys. A burst reorders bound keys, and plain capture cannot tell a placeholder from a value (lessons-live-verification).
- At each step, take `capture-pane -p` and `-e` captures and a semantic TOML diff of the config.

0. **Baseline, dev after #3169** (the run the critic found missing). Seed the shipped samples and `Llama_cpp = ["None","m-one","m-two","m-three"]`.
   - Capture Alt+M before the probe settles, with stall listeners on the template ports (see Stubs and listeners), within about 2 s of the open. Or stall only the ports this frame needs.
   - Re-derive the expected strings, because the readiness copy changed in TASK-33005.7.
   - Capture:
     - Settings ▸ provider vLLM: Model shows `vllm-model-z`;
     - Alt+M: the samples read as ready rows, and a model saved last sits under "… N more".
1. **T1 / C3:**
   - Point llama.cpp's endpoint at stub A. Advanced ▸ Saved model list ▸ Discover, tick `fake-beta`, Save selected.
   - While the app runs, a second process rewrites the file in place (same inode), adding `hand-added` to `Llama_cpp`. Tick `fake-gamma` and Save selected: the file keeps `hand-added`.
   - Advanced Config: delete `m-two` and save.
     - Verify through the Default model picker's Saved fallback, the Saved model list count, and Alt+M on a provider whose rows come from its saved list. While stub A answers, llama.cpp's Alt+M rows are its listing, so "Alt+M no longer lists `m-two`" would prove nothing there.
     - Save selected `fake-alpha`: `m-two` does not come back in the file.
2. **T1 / M2, on the old-sample profile.** Nothing below edits vLLM's table: it stays at the shipped default throughout this step.
   - Provider vLLM: Model reads "not set".
   - Alt+M, before the probe settles (stall listeners on the template ports; captured within about 2 s of the open), shows no `vllm-model-*`, `tabby-model*` or `aphrodite-engine*`.
     - On base the same frame shows the samples.
     - Once the probes refuse, base hides them too (NOT RUNNING), so only the pre-settle frames compare.
   - **NO-1, refused half.** Before any vLLM edit:
     - With the stall listeners closed, `lsof -nP -iTCP:<port> -sTCP:LISTEN` prints nothing for every port in the table. Those servers read as refused only if their ports are free.
     - Wait until every probed row has settled, then 10 s more, and reopen Alt+M. vLLM (an empty list after the filter, table untouched), TabbyAPI and Aphrodite are quietly under NOT RUNNING.
     - Oobabooga, koboldcpp and llamafile, untouched, are under NOT RUNNING too. On base they are under NEEDS SETUP, unprobed.
     - llama.cpp is not in this check, since step 1 repointed it.
   - **NO-1, running half.** Start stub B on 127.0.0.1:8000. Wait until every probed row has settled, then 10 s more, and reopen Alt+M: stub B logs a GET `/v1/models`, and vLLM is probed and listed with `fake-alpha` and `fake-beta`.
   - **Save selected, against stub B.**
     - vLLM's endpoint is still the template's localhost:8000, so Discover reaches stub B. No repointing is needed.
     - Tick `fake-alpha`, Save selected. The file's `vLLM` becomes `["fake-alpha"]`, and Model is filled as a draft.
     - Do not press `s`. Revert the Providers & Models draft afterwards, so vLLM's table stays at the shipped default.
   - Stop stub B when the step ends.
3. **T2, pick then save:**
   - On llama.cpp after Discover, Default model: type `fake-beta`, Enter.
   - Capture Source "edited \*" and the "· 1 joins on save" title. Config diff: none.
   - Press `s`. Capture the toast naming the addition. The config diff shows `chat_defaults.model`, `api_settings.llama_cpp.model` and `[providers].Llama_cpp` + `fake-beta` in one write.
   - Relaunch: Settings shows `fake-beta` ● CURRENT and its row reads "saved".
4. **T2, leave and return:**
   - Pick `fake-gamma`, go to Console, Ctrl+Q, Keep editing. Back in Settings, press `s` (not Enter: focus lands on `#nav-home`). The list gains `fake-gamma`.
   - Separately, pick, Ctrl+Q, Discard: the file is byte-identical.
5. **T2, keyboard and Inspector:**
   - Tab from Discover: list → Add to model list → Clear.
   - Capture the Inspector Focused setting and Save rows for each, and the instant hint.
6. **T3:**
   - Stub A lists at least 4 ids, with the default 4th or later (e.g. `fake-alpha`, `fake-gamma`, `fake-delta`, `fake-beta`; default `fake-beta`).
   - Set `Llama_cpp = ["None","m-one","m-two","m-three","fake-beta"]`, with `fake-beta` the saved llama.cpp default.
   - **The chat needs a message, from stub C.**
     - After step 3, `chat_defaults` is llama.cpp / `fake-beta`. A fresh chat would therefore make `fake-beta` CURRENT through the ADR-020 lift, which proves nothing.
     - With the app closed, a TOML edit points Custom (keyless) at stub C: `api_settings.custom.api_url` is stub C's `/v1/chat/completions`, and `api_settings.custom.model = "stub-chat"`.
     - Boot, open a new chat, pick Custom ▸ `stub-chat` in Alt+M, and send one message. Stub C answers it, and its log shows the POST.
     - The chat now holds a message on Custom, so it keeps that pair. Its current model is not on llama.cpp.
   - Open Alt+M from that chat. The llama.cpp group leads with `fake-beta`. Capture it twice:
     - before the probe answers (stub A on `--hold 2`, or a longer hold for this frame only; saved and cached rows);
     - after it answers (listing rows). If the before frame used a longer hold, restart stub A without it. Wait until every probed row has settled, then 10 s more, before the reopen.

**Cleanup and checks.**
- Kill every stub and stall listener by its pid, check that `lsof -nP -iTCP:<port> -sTCP:LISTEN` is empty again for every template port in the table, and run `tmux -L s1live kill-server`. Remove only `$V`.
- Every request in the stubs' logs came from the app. No send reached anything but stub C.
- During the run, `lsof -p <app pid>` shows no handle under the real profile. Afterwards, `find ~/.config/tldw_cli ~/.local/share/tldw_cli ~/.config/tldw-live -newer $V/baseline` prints nothing.
- Keep the captures in the session scratchpad and describe the procedure in the PR. Commit no probe scripts.

## Risks & traps

- **Identity of `app.providers_models`.** `DefaultModelPicker` keeps the dict object it was composed with, so refresh in place and never rebind. The startup refresh still rebinds through `app._init_providers_models` (wired at app_feature_glue.py:585); see Out of scope.
- **Exact size ratchets.** The numbers are in Assumptions above.
  - Move code out, then lower the rows in the same commit.
  - `app.py` and `app_feature_glue.py` sit exactly on their rows, so edit neither.
  - Only the `does_not_grow` nodes of the three Settings modules run in a PR lane (derived-artifacts.yml:189-191). The slack test and any new row do not.
- **Moved code** (lessons-testing-evidence, "Tests that read a moved file's source run outside the PR gate").
  - Grep `Tests/` for every moved symbol and for source-reading and AST tests, and run every hit on both trees.
  - Rows in `scripts/markup_interpolation_census.tsv` are keyed by module, enclosing qualname, kind and count (scripts/check_markup_interpolation.py:40-43).
    - A site moved to a new module or function gets a new key, which fails.
    - Make a moved unescaped site `markup=False` or `escape_markup` at the move.
    - Use `--write` only to shrink.
- **Imports.**
  - `saved_model_list.py` is imported by the card, which is itself lazy, and only inside functions in `settings_screen.py`. `model_discovery_persistence.py` keeps importing `config` inside functions.
  - Measure the boot census with `PYTHONPATH=<worktree>` (lessons-testing-evidence, "A boot census run from a worktree can measure the main checkout instead").
- **T2's precondition.** Compare the effective view from `get_atomic_config_snapshot` with the writer's locked snapshot, which comes from the same function. Never compare against T1's loader-filtered list, which would refuse every write to a list still holding samples. The precondition is a pure comparison; see "Config IO on the UI loop" below.
- **Template `api_settings`.** Leave its model values alone: `provider_left_at_shipped_default` compares the tables with the template.
- **The send path stays unchanged on purpose.** That is why the Settings-only keyword exists. Do not filter inside `configured_provider_model` or in the default path of `resolve_remembered_provider_model`.
- **A config default that also ships in the config template cannot be mutation-tested through config** (lessons-testing-evidence :5416, same title). Write the old-sample file in the test.
- **Fakes and synthetic configs.** See lessons-testing-evidence: "A fake written to match your call site validates the mistake", "A synthetic test config lets tests pin states no user can reach", and "Persist test configuration through the same durable seam production reloads".
  - Once T1 lands, a test that hand-seeds `app.providers_models` sees the file's lists after any refresh, so seed the file instead.
  - Persistence claims need the real service and the real writer.
- **Lane and harness lessons** (lessons-testing-evidence):
  - :554, a test's solo time is not its cost inside a lane step;
  - :580, a lane test's child imports with the lane's install set;
  - :951, a negative control passed as a pytest plugin never reaches a `private_profile_test` body;
  - :1022, that plugin's shared profile turns config-writing tests red together;
  - :1071, "A substring wait cannot pin an absence" (TASK-33008);
  - :1126, marking your own nodes `bootstrap_profile` changes how their unmarked neighbours fail;
  - the TASK-35190 pick and probe lessons (:16864, :16892).
- **`RecoveryRequired` at setup** in the journeys file is a profile-selection trip (lessons-testing-evidence, "Tests/UI `RecoveryRequired` at setup…"). Compare failure names against the branch's merge base, not counts and not whatever `origin/dev` is now.
- **Findings may already be fixed on your base** (lessons-testing-evidence, "A finding verified on the review's commit may already be fixed on your base"). Re-check each finding at the branch point, in case more lands on dev in between.
- **Guidance rows** keep the four-plus-one row shape, or `_with_save_behavior_row`'s positional refresh misaligns.
- **Markup.** Receipts that interpolate model ids or provider names use `notify(..., markup=False)`. Tooltips stay static. Run `scripts/check_markup_interpolation.py`.
- **Workflows.** Editing `.github/workflows/*` in parallel with another PR that changes the same file triggers `evict-workflow-file`, so use the UI census instead. NO-2 (a)'s lane node is the one exception, and only when no open or armed PR touches `derived-artifacts.yml`.
- **Config IO on the UI loop.** TASK-33641.6 is in the base (1d660462b5, 930c8101be).
  - From the thread already inside that file's write transaction, these raise `RecoveryRequired('raw_operation_provenance_invalid')` at once (config.py:7442-7486): a config write, `read_cli_config_snapshot`, and `get_atomic_config_snapshot`.
  - T1's builder and T2's precondition use only the snapshot they are handed.
  - Any refresh or save runs after the transaction returns. It never runs from inside a builder, `before_replace`, `after_replace` or a precondition, and never through a task started there.
- **Validator hole.** The `providers` branch must admit only the mutation's own key and a list holding the saved model. Test each refusal.
- **Frozen sample set.** It cannot list a real model whose id is exactly one of the 13 (OD-1). Mark it in code with a `ponytail:` comment naming that ceiling.
- **zsh.** `git show "$ref:path"` applies the `:t` modifier; quote it as `"${ref}:path"`.
- **Live.** Never write the real profile. Verify with `lsof` and the `-newer` check, and validate the TOML after each edit and each boot.

## Rollback

Each task is one PR and reverts on its own; nothing needs a schema or data migration. Revert in reverse order (T3, T2, T1), because T2 uses T1's helper and its sample-dropping append. Expect census and floor-line conflicts on a late revert.
- **Reverting T1:** writes go back to the snapshot base, and the samples return for new profiles. Lists that were already written or cleaned stay valid data.
- **Reverting T2:** a pick stops joining the list, and the button reads Save selected again. Lists already written stay valid.
- **Reverting T3:** Switch model's groups go back to list order.
