---
id: TASK-35010
title: Saved model lists keep what the config file holds, and shipped sample model ids never pose as saved models or defaults
status: Done
assignee:
- '@claude'
created_date: 2026-10-14 17:10
updated_date: 2026-10-20 04:17
labels:
- settings
- providers
- model-discovery
- config
- data-integrity
- ux-review-2026-10-14
dependencies:
- TASK-33007
- TASK-33641
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two verified problems make saved model lists untrustworthy in Settings ▸ Providers & Models.

C3 (data loss): Save selected appends to the startup snapshot of [providers] that the app keeps in memory, then rewrites that provider's whole list. A model added to the list after the app started, through Advanced Config or by another program, is silently removed. A model the user deleted comes back. Switch model and the Settings provider-switch fallback keep listing removed models until restart, because nothing refreshes the in-memory lists after an Advanced Config save. The config cache can also miss an edit made in place (TASK-34943, TASK-34700), so the fix has to read the file itself at write time instead of trusting any cached copy.

A-pm-design-M2: the shipped template seeds the vLLM, TabbyAPI, Aphrodite, Custom and Custom 2 lists with sample ids no server serves (vllm-model-z, tabby-model, aphrodite-engine, custom-model-alpha and others). Switching Settings to one of these providers pre-fills Model with a sample id, so readiness names a model that does not exist and the documented 'Save selected fills an empty Model' rule (TASK-369) never fires. Switch model lists the samples as ready rows. Discovery-first cloud providers already ship empty lists. Existing profiles already hold the samples, so the fix has to cover files written by older versions without rewriting them at startup. It must also leave the shipped-default detection alone (the TASK-33005.6 ruling that keeps untouched local servers under NOT RUNNING) and change nothing about what is sent to a provider. Recognising the old samples by a frozen set of their values is a deliberate, scoped departure from TASK-34100.2 AC#9's marker rule: the set covers only ids that older versions already wrote, which carry no marker, and TASK-34100.2's marker rule still governs the template.

Plan: `backlog/docs/plan-2026-10-14-s1-discovered-model-default-saved-lists.md`. Findings closed: C3, A-pm-design-M2 (Settings UX review, 2026-10-14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Save selected keeps every model the config file holds for that provider at the moment it writes: a model added after the app started, through Advanced Config, by another program replacing the file, or by an in-place edit of the same file, is kept, and a model removed that way does not come back
- [x] #2 Save selected's receipt counts only the models it actually added to the file, and says the selected models are already saved when the file already lists all of them
- [x] #3 The startup catalog refresh with save to config keeps such external changes the same way and still adds newly listed models
- [x] #4 After Save selected, an Advanced Config save or a Config reload, the Switch model (Alt+M) groups whose rows come from the saved list, the Default model picker and the Saved model list rows and counts show the file's saved lists without a restart
- [x] #5 A new profile's saved model lists for vLLM, TabbyAPI, Aphrodite, Custom and Custom 2 are empty
- [x] #6 On a profile whose config file still holds the old sample lists, Switch model, the Default model picker, the Saved model list counts and the first-run wizard's model list show none of the sample ids, and real ids saved beside them still show
- [x] #7 Switching Settings to one of those providers on such a profile leaves Default model not set, also when the provider's settings table still holds the shipped sample model, and Save selected then fills it as an unsaved draft
- [x] #8 The next Settings write to such a provider's list removes the sample ids from the file and keeps the real ids, and the app does not rewrite the config file at startup for this
- [x] #9 A provider whose settings table is still exactly the shipped one is still treated as never set up, so an untouched local default that refuses stays under NOT RUNNING in Switch model and one that is running is still probed and listed. This includes an untouched local server with no model (vLLM, llama.cpp, koboldcpp, Oobabooga, llamafile): it is probed, listed when running and under NOT RUNNING when refused
- [x] #10 Dreams stories, briefings and Console sends use the same model for every provider as before
- [x] #11 The Settings User Guide says the vLLM, TabbyAPI, Aphrodite, Custom and Custom 2 saved lists start empty and fill through Discover models, that old sample ids are ignored and dropped the next time Settings saves that list, and that Save selected adds to the list as the config file holds it
- [x] #12 The Console User Guide says Switch model checks an untouched local server with no model chosen: it is listed when it is running, and under NOT RUNNING when it refuses
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follows the plan's TASK-35010 steps 1-8 (backlog/docs/plan-2026-10-14-s1-discovered-model-default-saved-lists.md).

1. `persist_discovered_models_to_settings`, with no injected save callback, does one read-modify-write under the config write lock (`apply_literal_settings_transaction_to_cli_config`). The builder is pure: it reads the locked snapshot's effective `[providers]` list, decides the added ids and "already saved" against it, and calls no config reader, writer or task. A failed transaction maps to `status="error"`; a key the locked view lacks maps to `missing_provider_key`. Injected callbacks keep today's contract.
2. `refresh_providers_models(app, force=)` in the same module re-reads the loader into `app.providers_models` in place; `force` republishes the caches first (Config reload).
3. Settings: Save selected's saved branch refreshes instead of appending in memory (`_append_saved_discovered_models` deleted); the Advanced Config `applied` callbacks and both Config reload sites adopt the loaded config through one helper that also refreshes. Net settings_screen.py lines <= 0; lower its size row.
4. `config.py`: the template's vLLM, TabbyAPI, Aphrodite, Custom and Custom_2 lists ship empty (and the dead `LOCAL_PROVIDERS` literal mirrors it); `SHIPPED_SAMPLE_MODEL_IDS` (13 ids) beside `get_cli_providers_and_models`, which drops them from every list and keeps every key.
5. `resolve_remembered_provider_model(..., ignore_shipped_samples=)`, off by default; only Settings' provider switch and empty-Model fallback pass it.
6. `append_models_to_provider_list` drops sample ids from the list it rewrites; a Save selected that adds nothing still writes a list that held samples (D-a). Nothing rewrites the file at startup.
7. Template `[api_settings.*].model` values untouched.
8. NO-1 (owner ruling 2026-10-20, "All 5, endpoint too"): the Switch model popover also probes URL-based providers that are `model_missing` at the shipped default. It lists a refused one under NOT RUNNING, and re-requests readiness once a running one's listing supplies a model (a predicate plus two branches). koboldcpp's native `/api/v1/generate` endpoint resolves in `resolve_provider_endpoint` with its models route at `/v1/models`. Its chat URL and persisted endpoint stay the native URL, so a koboldcpp send's URL is unchanged, proved by tests. Stop again if this needs a template change, changes a send, or spreads past the endpoint contract and probe-target layer.

Tests first in Tests/UI/test_settings_saved_lists_follow_file.py (plan cases 1-8, private profile, census tail), each shown red before its fix; rewrite only the pinned tests the plan names; docs in settings.md and console.md; covering run at head vs a base worktree by failure name.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Saved model lists now follow the config file, the shipped sample ids never pose as saved models or defaults, and Switch model probes an untouched local server with no model. The implementer stopped at the account's weekly usage limit after the product, docs and test commits; the coordinator finished the evidence, lanes and records.

**Approach.** Save selected (and the startup refresh's save) appends under the config write lock in one read-modify-write whose builder is pure (TASK-33641.6): the list, the ids added and "already saved" all come from the locked snapshot, never from the startup copy (`_persist_under_config_lock`). `refresh_providers_models` re-reads `app.providers_models` in place after Save selected, an Advanced Config save and both Config reload sites (`SettingsScreen._adopt_config`), so the Default model picker and the Saved model list need no rebuild; the old in-memory `_append_saved_discovered_models` is gone. The template's vLLM, TabbyAPI, Aphrodite, Custom and Custom_2 lists ship empty; `SHIPPED_SAMPLE_MODEL_IDS` (13 ids) is dropped by `get_cli_providers_and_models` and skipped by Settings' default choice (`resolve_remembered_provider_model(..., skip_samples=True)`, passed only at Settings' two default sites). A Settings write drops the samples from the list it rewrites, even when it adds nothing (D-a); nothing rewrites the file at startup. Sends keep the table's model: Dreams (`story_service.py:251`), briefings (`briefing_service.py:351`) and `configured_provider_model` are unchanged.

**NO-1** (owner rulings 2026-10-19 "All five servers" and 2026-10-20 "All 5, endpoint too"). The stop rule fired once, and the owner widened it. The popover change is a predicate plus two branches, as ruled 2026-10-20 (the refused half in `_offer_probes`/`_not_running`, the running half in `_resolve_readiness`): `_untouched_server_without_a_model` (model_missing, URL-based, left at the shipped default) joins the probe offer and the NOT RUNNING rule, and `_resolve_readiness` asks once more, with the listed model, for a server read with no model whose listing has since arrived (the second pass has a model, so it cannot loop). koboldcpp's template endpoint `/api/v1/generate` resolves through the endpoint contract (`_KOBOLD_NATIVE_GENERATE`): its models URL is the root's `/v1/models` and its send URL stays the native route, whole. The template is unchanged.

**Consequences of "endpoint too"** (named by the review, accepted as part of the ruling; no code change). The native endpoint is now valid for every consumer of the endpoint contract, not only Switch model's probe: (a) Settings `s` with the template URL saves instead of raising "Endpoint is invalid."; (b) the first-run wizard accepts it, and Chat settings, Settings `t` and continuation checkpoints can probe or validate it; (c) a configured keyless koboldcpp with a model at a loopback or private native URL is now probed on every Alt+M, and once `/v1/models` answers its rows are the server's listing (TASK-33005.12), not its saved model; (d) Settings ▸ Discover models still refuses the same URL, so Switch model can list a running untouched koboldcpp while Discover reports an unsupported endpoint (TASK-35832). Shapes checked by the review: `/v1/api/v1/generate`, `/v1/chat/completions/api/v1/generate` and `/api/v1/model` stay ambiguous, `/proxy/api/v1/generate` resolves under its prefix, and llama_cpp given the route is still refused.

**Review fixes.** Diagnostics' validate-and-reload re-reads the file in its worker (`refresh_runtime_config_from_cli_config` after `adapter.load(force_reload=True)`), so its apply adopts the config without `force` on the UI thread; only the synchronous Config reload button keeps `force=True`, beside the validate and load it already ran there. `_adopt_config` refreshes the Settings widgets only while the screen is attached (a popped screen stays `is_mounted` in Textual 8.2.8, so the review's second round moved the guard from `is_mounted` to `is_attached`), so an Advanced Config save that finishes after Settings closed updates the app's config and lists and nothing else; `test_a_late_advanced_config_save_after_settings_closed_updates_only_the_lists` pins it and fails with the `is_mounted` guard.

**Evidence** (Tests/UI/test_settings_saved_lists_follow_file.py, 17 cases, private profile; each AC's case named in its docstring). The 16 AC cases all fail against the pre-fix code (beb52531af) on assertions, except the template case, which fails there on an ImportError; with only the loader's sample filter removed at the head, it fails on its assertion too. AC#11 and AC#12 are the docs commit (Docs/User_Guide/settings.md, console.md).

**Pinned tests rewritten on purpose** (the plan's list): in test_settings_model_discovery_journeys.py, `test_discovery_keyboard_selection_survives_rebuild_and_save`, `test_late_save_keeps_receipt_and_activation_out_of_a_changed_form` and `test_discovery_operations_report_failure_and_allow_retry` assert the lists equal the file after a fake persist. Their file fails 26/26 at setup outside a private profile, so they now carry `@private_profile_test`: 11 cases pass, and against the pre-fix code the 7 where a save lands fail. `test_settings_save_selected_not_cancelled.py::test_save_selected_is_not_cancelled_by_discover_or_clear` passes (13/13 in the file). `test_settings_configuration_hub.py::test_settings_provider_model_discovery_saves_selected_runtime_models` was rewritten the same way but is unverified locally: it sits in a bootstrap file, fails at setup with RecoveryRequired at the head and at base, and runs in no lane (the plan allows saying so); the census file covers its behaviour.

**Size rows.** settings_screen.py 35,770 -> 35,766 (the in-memory append goes, -4 net) -> 35,765 after the review fixes. No other touched file has a row; providers_models_card.py and app.py are untouched.

**Lanes.** The new file joins the Tests/UI census as a whole file (private profile, not bootstrap_profile), like its siblings; 86-91 s 3 of 3 serial in the lane's install set (17 cases, after the review's late-save case); the floor rises to 361.

**Covering run.** 72 files (the plan's T1 list plus every test file importing a changed module), xdist, at the head and at beb52531af: no failure at the head that is not also at base, apart from two nodes. Re-run serially at both trees: `test_console_session_settings.py::test_console_settings_modal_save_disabled_during_active_run` fails 3 of 3 at both (pre-existing); `test_console_switcher_local_probe.py::test_the_inspector_setup_row_and_send_hint_read_the_readiness_word[key-rejected]` failed 3 of 10 at the head and 6 of 10 at base, a pre-existing race that Phase 9 fixed in its own branch (b94aece7d6, TASK-33005.7). preflight passes under a scratch HOME; Tests/CI 1015 passed; ruff adds no finding on T1's lines (the two misplaced `noqa: F811` in the new file were moved onto the fixture).

**Files.** tldw_chatbook/config.py, tldw_chatbook/LLM_Provider_Catalog/model_discovery_persistence.py, tldw_chatbook/Chat/provider_setup_persistence.py, tldw_chatbook/Chat/provider_endpoint_contract.py, tldw_chatbook/UI/Screens/settings_screen.py, tldw_chatbook/Widgets/Console/console_model_popover.py; Tests/UI/test_settings_saved_lists_follow_file.py (new), the three rewritten test files, Tests/Architecture/test_module_size_ratchet.py; scripts/ui_pr_gate_census.txt, scripts/check_ui_pr_gate_census.py; Docs/User_Guide/settings.md, Docs/User_Guide/console.md; the S1 plan (the 2026-10-20 ruling).
<!-- SECTION:NOTES:END -->
