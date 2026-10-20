---
id: TASK-35010
title: Saved model lists keep what the config file holds, and shipped sample model ids never pose as saved models or defaults
status: In Progress
assignee:
- '@claude'
created_date: 2026-10-14 17:10
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
- [ ] #1 Save selected keeps every model the config file holds for that provider at the moment it writes: a model added after the app started, through Advanced Config, by another program replacing the file, or by an in-place edit of the same file, is kept, and a model removed that way does not come back
- [ ] #2 Save selected's receipt counts only the models it actually added to the file, and says the selected models are already saved when the file already lists all of them
- [ ] #3 The startup catalog refresh with save to config keeps such external changes the same way and still adds newly listed models
- [ ] #4 After Save selected, an Advanced Config save or a Config reload, the Switch model (Alt+M) groups whose rows come from the saved list, the Default model picker and the Saved model list rows and counts show the file's saved lists without a restart
- [ ] #5 A new profile's saved model lists for vLLM, TabbyAPI, Aphrodite, Custom and Custom 2 are empty
- [ ] #6 On a profile whose config file still holds the old sample lists, Switch model, the Default model picker, the Saved model list counts and the first-run wizard's model list show none of the sample ids, and real ids saved beside them still show
- [ ] #7 Switching Settings to one of those providers on such a profile leaves Default model not set, also when the provider's settings table still holds the shipped sample model, and Save selected then fills it as an unsaved draft
- [ ] #8 The next Settings write to such a provider's list removes the sample ids from the file and keeps the real ids, and the app does not rewrite the config file at startup for this
- [ ] #9 A provider whose settings table is still exactly the shipped one is still treated as never set up, so an untouched local default that refuses stays under NOT RUNNING in Switch model and one that is running is still probed and listed. This includes an untouched local server with no model (vLLM, llama.cpp, koboldcpp, Oobabooga, llamafile): it is probed, listed when running and under NOT RUNNING when refused
- [ ] #10 Dreams stories, briefings and Console sends use the same model for every provider as before
- [ ] #11 The Settings User Guide says the vLLM, TabbyAPI, Aphrodite, Custom and Custom 2 saved lists start empty and fill through Discover models, that old sample ids are ignored and dropped the next time Settings saves that list, and that Save selected adds to the list as the config file holds it
- [ ] #12 The Console User Guide says Switch model checks an untouched local server with no model chosen: it is listed when it is running, and under NOT RUNNING when it refuses
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
