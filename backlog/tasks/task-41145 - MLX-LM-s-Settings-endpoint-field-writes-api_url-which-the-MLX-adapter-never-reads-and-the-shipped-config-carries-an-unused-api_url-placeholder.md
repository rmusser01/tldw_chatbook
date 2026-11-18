---
id: TASK-41145
title: "MLX-LM's Settings endpoint field writes api_url, which the MLX adapter never reads, and the shipped config carries an unused api_url placeholder"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-17 10:36'
created_date: 2026-11-15 06:30
labels:
- follow-up
- settings
- providers
- config
priority: low
references:
- tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/Config_Files/default_config.toml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top d4e8d51ca8 (endpoint stream b3c75c8234, on dev ecad404bd4); line numbers are that tree's.

**Why.** A user who sets the MLX-LM endpoint in Settings ▸ Providers sees it saved, but chats still go to 127.0.0.1:8080. Review r14ep-rev4 (C-1) made Console check and pin the adapter's own URL; the Settings field and the adapter still disagree, and changing the adapter needs a config migration.

**Evidence.**
- `settings_screen.py:1686-1700`: `API_URL_PROVIDER_KEYS` includes `local_mlx_lm`, so its Endpoint row writes `[api_settings.local_mlx_lm] api_url`.
- `chat_with_mlx_lm` (`LLM_API_Calls_Local.py:2522-`) uses only the `api_base_url`/`api_url` arguments; otherwise it builds `http://{host}:{port}/v1` from the section's `host` and `port` (defaults 127.0.0.1 and 8080, `:2557-2564`). It never reads the section's `api_url`.
- `Config_Files/default_config.toml:1429-1430` ships `[api_settings.local_mlx_lm] api_url = "http://localhost:5678/v1/chat/completions"`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings' MLX-LM endpoint and the adapter use one setting: either the adapter reads the saved endpoint (with a config migration for existing host/port values) or Settings edits host and port; the notes record which and why
- [x] #2 The shipped default config matches the setting the adapter reads
- [x] #3 A test saves the endpoint through Settings and checks the URL the adapter calls, red before the change
<!-- AC:END -->

## Implementation Plan

1. Map every writer and reader of the MLX-LM endpoint (Settings, first-run setup, Chat settings Make default, adapter, Console pinning, shipped template).
2. Agreeing needed a choice of which side wins: reported NEEDS_CONTEXT with the options (first pass, below).
3. Owner decisions 2026-11-16: implement "Settings edits host/port" for MLX-LM, and "Adapter reads api_url" for Custom OpenAI API 2. Tests red first, on the merged-config shape.

## Implementation Notes

**Owner decisions, 2026-11-16 (verbatim):**
- MLX-LM: "Settings edits host/port (Recommended)."
- Custom OpenAI API 2: "Adapter reads api_url (Recommended)."

Why host/port wins for MLX-LM (AC#1): it changes no send and needs no migration. Reading `api_url` instead would have moved every MLX user whose profile came from the old template to its localhost:5678 placeholder unless a migration rewrote it.

**MLX-LM: Settings edits host and port.** The adapter is unchanged (no send change, no migration).
- Rules, in `provider_setup_persistence`: `MLX_HOST_PORT_PROVIDER_KEYS`; `mlx_endpoint_refusal` (the contract's reason first, so user info reads "Endpoint URL must not include user information."; then `MLX_HTTPS_REFUSAL`; then `MLX_PATH_REFUSAL` for a path other than `/v1`); `host_port_refusal` (that rule for an MLX endpoint as typed, "" otherwise); `typed_endpoint_refusal` (KoboldCpp's route rule, then MLX-LM's); `mlx_host_port` (an IPv6 host keeps its brackets, as the adapter puts it in its URL verbatim; no port means 80); `mlx_saved_endpoint`. The URL is built once, `console_provider_endpoints.mlx_endpoint` (which `adapter_own_endpoint` now uses), and `configured_provider_endpoint` sits beside `first_configured_endpoint`.
- Writers: `build_provider_setup_mutation` (Settings' **s** and first-run setup) writes `host`/`port`, leaves `api_url` alone and clears both on a blank endpoint; its validator accepts that pair for MLX (and an integer `port`); the write CAS routing keys include `host`/`port`. Make default (`console_settings_defaults._endpoint_values`, its preview, `console_default_endpoint_refusal`) does the same.
- Readers: Settings' Endpoint field and "Endpoint key" row (`category_panels.provider_endpoint_value`/`provider_endpoint_row`; the size ratchet moved both bodies out of `settings_screen.py`, which keeps one-line delegates), first-run setup's prefill, found-server rule and draft resolution show `http://<host>:<port>/v1`, never `api_url`.
- Refusals with copy: Settings' save and the Endpoint row's help as typed; first-run setup's endpoint status line and commit refusal; Make default and Chat settings' pre-submit check (same function).
- AC#2: the template's `[api_settings.local_mlx_lm]` ships `host = "127.0.0.1"`, `port = 8080` (the adapter's own defaults, so nothing sends elsewhere) instead of the unused `api_url`. A profile written from the old template keeps that `api_url`, ignored as before; the User Guide says it shows as the host/port value until re-entered.
- Existing tests moved by the decision: round 14's `test_mlx_sends_to_its_own_host_and_port_never_the_shipped_api_url` set its own saved `api_url` (the template no longer ships one); `test_provider_setup_persistence::test_every_offered_provider_saves_through_setup[local_mlx_lm]` reads host/port.

**Custom OpenAI API 2: the adapter reads `api_url`.** `chat_with_custom_openai_2` reads `[api_settings.custom_2] api_url` (where Settings, first-run setup and Make default write) and falls back to the legacy `[API] custom_openai2_api_ip`, which `load_settings` exposes as `custom_openai_api_2.api_ip` (there is no top-level `[custom_openai_api_2]` table in a profile; review r15ep-rev2 M-2 corrected the earlier wording); `adapter_own_endpoint` follows the same order. Effect on existing files: the template ships `[api_settings.custom_2] api_url = "http://localhost:5678/v1/chat/completions"`, so a profile written from it that relied on the legacy `[API] custom_openai2_api_ip` would have moved to that placeholder; the ruling below keeps such an explicit legacy endpoint. **Ruling 2026-11-17 (engineering): an explicit legacy custom_2 endpoint is not silently redirected.** While `[api_settings.custom_2] api_url` is absent, blank or still the template's `http://localhost:5678/v1/chat/completions`, and the user's `[API] custom_openai2_api_ip` is set to anything but its code default (`http://127.0.0.1:5000/v1/chat/completions`, now `console_provider_endpoints.LEGACY_CUSTOM_2_API_IP_DEFAULT`), the merged config view's `api_url` takes the legacy value; otherwise `api_url` wins as before. One place: `config._merge_user_config_over_shipped_defaults` calls `keep_legacy_custom_2_endpoint`, so `load_settings` (the adapter's read), the runtime snapshot, `adapter_own_endpoint`, Console and Settings all see one value; in memory only, like `_normalize_legacy_provider_api_key`, and the file is never written. Tests: `test_an_explicit_legacy_custom_2_endpoint_is_not_redirected` (legacy explicit + placeholder, and + a blanked slot: legacy used, red before; legacy explicit + user-edited `api_url`: `api_url` used; legacy at its default + placeholder: `api_url` used; each through `chat_api_call` and `adapter_own_endpoint`) and `test_the_loaded_settings_keep_an_explicit_legacy_custom_2_endpoint` (real `load_settings` on a scratch file; red before; the file's bytes unchanged). The edited-slot and default cases match the old behaviour, so they were shown red under a temporary over-eager bridge (no default or edited-slot guard: both failed), then green with it reverted. Also re-pinned `test_custom_openai_execution_keys_constant`'s audited count for `LLM_API_Calls_Local.py` to (4, 3): M-2's two `provider_key` literals; that test's other four offenders are identical with base sources. User Guide: `settings.md`. Not changed: the adapter still requires a key unless the caller resolved one (`api_key_resolved`), while readiness reads custom_2 as ready keyless, so a keyless custom_2 Dreams or ingest send fails with "API Key required" (proposed follow-up).

**Red/green.**
- MLX, lane file `Tests/Chat/test_local_server_discovery.py`: `test_saving_an_mlx_endpoint_writes_the_host_and_port_the_adapter_reads` (5 forms, through `chat_api_call` on the merged config), `test_an_mlx_save_refuses_what_host_and_port_cannot_express`, `test_first_run_setup_reads_and_writes_mlx_host_and_port`, `test_make_default_writes_mlx_host_and_port` (6) and `test_the_shipped_mlx_table_holds_the_host_and_port_its_adapter_reads`: red with the refusal helpers present but unwired (KeyError 'host', DID NOT RAISE, the shown endpoint was the placeholder, '' refusal copy; the template test red on `api_url` present), green after.
- AC#3, `Tests/UI/test_settings_provider_save_scope.py::test_mlx_settings_edits_the_host_and_port_its_send_uses`: drives Settings' Providers card (private profile), shows `http://127.0.0.1:8080/v1` over a saved `api_url`, refuses `https://` with the copy, saves `http://gpu-box.lan:9000/v1`, then `chat_api_call("local_mlx_lm")` posts to `http://gpu-box.lan:9000/v1/chat/completions`. Red with base sources (the field showed `http://localhost:5678/v1/chat/completions`), green after; 70 s serial at load 61-70.
- custom_2: `test_a_dreams_send_on_custom_2_reaches_the_endpoint_settings_saved` (red: "api_ip required" under the merged config; in a real profile it went to the legacy 127.0.0.1:5000 default) and `test_an_ingest_analysis_on_custom_2_reaches_the_endpoint_settings_saved` (red: no POST), green after; `test_custom_2_still_reads_its_legacy_api_ip_when_the_slot_has_none` is the fallback control, since review r15ep-rev2 M-2 on a real scratch file (`[API]` explicit and at its default, slot blanked; a temporary removal of the `api_ip` fallback turned the default case red). The keeps-working case for custom_2 moved from the legacy `api_ip` to the slot, by this decision.

Files: `tldw_chatbook/Chat/provider_setup_persistence.py`, `tldw_chatbook/Chat/console_provider_endpoints.py`, `tldw_chatbook/Chat/console_settings_defaults.py`, `tldw_chatbook/UI/Screens/settings_screen.py`, `tldw_chatbook/UI/Settings_Modules/category_panels.py`, `tldw_chatbook/UI/Settings_Modules/providers_models_card.py`, `tldw_chatbook/UI/Wizards/first_run_setup_state.py`, `first_run_provider_detection.py`, `first_run_provider_step.py`, `first_run_provider_list.py`, `tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py`, `tldw_chatbook/Config_Files/default_config.toml`, `Tests/Chat/test_provider_setup_persistence.py`, the two test files above, and `Docs/User_Guide/settings.md`, `First_Run_Setup.md`, `console.md`.

**First pass (round 15, before the decisions):** NEEDS_CONTEXT, with Option A (the adapter reads `api_url`; needs a migration of the shipped localhost:5678 placeholder) and Option B (Settings edits host and port; no send change). The owner chose B for MLX-LM.

**Review r15ep-rev2 round (2026-11-17).**
- I-1 (coordinator ruling: fix (a)): Settings showed Custom 2's bridged legacy endpoint and a model-only save wrote it into `[api_settings.custom_2] api_url`, so the legacy key stopped being read. The bridge now records, in the merged view, the legacy value and what the slot read before it (`LEGACY_CUSTOM_2_BRIDGE`, a provenance key like `legacy_api_key_providers`, never written); when the Endpoint field was not edited, the Providers card's save writes that earlier value back (`endpoint_to_save_unedited`, called from `category_panels`). So a model-, key- or env-var-only save leaves the template placeholder in the file and the legacy key keeps being read; an edited Endpoint saves as before. The written value is what the slot read before the bridge, which is what this save wrote before the bridge existed: the placeholder for a placeholder or absent slot (an absent slot gains the template's own value, as before), and a blank slot is cleared, as Settings always clears a blank Endpoint. `Tests/UI/test_settings_provider_save_scope.py::test_a_model_only_custom_2_save_keeps_the_legacy_endpoint_unsaved` drives the real Providers card on a private profile: red before (the file's `api_url` became the legacy URL), green after (placeholder kept, the next `chat_api_call` posts to the legacy URL). First-run setup's commit is not changed (it confirms the endpoint it shows).
- I-2: `settings_provider_view_model.configured_provider_keys` counted MLX-LM as Configured for every old-template profile (its ignored `api_url` differs from the new template). MLX-LM now compares `configured_provider_endpoint` (host and port) with the template's. `test_mlx_counts_as_configured_only_for_a_host_and_port_the_user_set`: the old-template and user host/port cases were red, green after.
- Discovery: `LocalLLMProviderCatalogService._resolve_endpoint` (model discovery and the auto-refresh fingerprint) read MLX-LM's `api_url`; with no staged draft it now reads `mlx_endpoint`. `test_mlx_discovery_lists_the_host_and_port_its_send_uses`: 3 red (None, the 5678 placeholder, None), green after; a staged draft still wins.
- M-1: the missing-endpoint copy now reads "Custom OpenAI API 2 API URL is required: set [api_settings.custom_2] api_url." (`test_custom_2_without_an_endpoint_names_the_setting_to_fill`, red before).
- M-2: the adapter comment and these notes now name the real fallback source (above).

**Review r15ep-rev3 round (2026-11-17).**
- I-1, the live view: after an unedited Custom 2 save, Settings patched the written placeholder into `app.app_config`, so it showed (and Test and Discover staged) localhost:5678 while sends used the legacy address until a restart. `category_panels` now re-runs the load-time rule on the live view after the patch (`reapply_legacy_custom_2_bridge`, which calls `keep_legacy_custom_2_endpoint`). `test_a_model_only_custom_2_save_keeps_the_legacy_endpoint_unsaved` now also checks `_provider_endpoint_value("custom_2")` after the save and the Endpoint field after switching to vLLM and back: red before (the placeholder), green after.
- M-1: `load_cli_config_and_ensure_existence`'s DEBUG top-level key list leaves out `legacy_*` provenance keys. A scratch-profile probe with an explicit legacy key: listed before the change, not after.
- First-run note: `First_Run_Setup.md` says committing setup for Custom OpenAI API 2 saves the shown address into its Endpoint, after which the older `[API]` key is no longer read.

**Review r15ep-rev4 round (2026-11-17).**
- Minor, stale provenance: the live view kept the bridge recorded at boot, so after first-run (or an edited Settings save) wrote a real `api_url`, even exactly the legacy URL, a later model-only save in the same session wrote the boot-time placeholder over it. `endpoint_to_save_unedited` now reads the bridge from a fresh `get_atomic_config_snapshot().values`, which has none once the file holds a real `api_url`. It no longer takes the view. A `ValueError` from that read writes the shown value, and the save's own write refuses the same file. `test_a_written_custom_2_endpoint_survives_a_later_model_only_save[first_run|edited_legacy]` were both red before (the file ended on localhost:5678) and are green after. The reviewer's probe, with the new two-argument call, now keeps the legacy URL in the file. The `First_Run_Setup.md` sentence holds.
