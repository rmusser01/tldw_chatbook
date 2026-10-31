---
id: TASK-36381
title: Settings provider and discovery test files that no PR lane runs are red on dev
status: Done
assignee:
- '@claude'
created_date: 2026-10-22 23:58
updated_date: 2026-10-30 04:59
labels:
- follow-up
- tests
- settings
priority: medium
references:
- Tests/UI/test_settings_model_discovery_journeys.py
- Tests/UI/test_settings_endpoint_probe.py
- Tests/Chat/test_local_server_discovery.py
- Tests/UI/test_settings_configuration_hub.py
- Tests/UI/app_factory.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while burning down the Settings UX follow-ups (2026-10-22, dev 165e31fab8). These tests fail on dev, with the same ids on every follow-up branch. No PR lane runs them, so dev's CI never reported them.
- **`Tests/UI/test_settings_model_discovery_journeys.py`** (not in the UI census). Run alone it gives 15 failed and 11 passed. The failures are `RecoveryRequired: raw_source_selection_changed`, raised in the test body: `_app_scope()` → `Tests/UI/app_factory.py:125` `build_test_app_config` → `load_settings()`.
- **`Tests/UI/test_settings_configuration_hub.py`**: run alone, about 242 of its nodes fail; its 11 census nodes pass. The causes seen so far:
  - **RecoveryRequired (about 231).** The same call-phase error as above. Some of these, such as `test_settings_domain_categories_are_grouped_and_have_ownership_records`, also carry a stale assertion that shows only once RecoveryRequired is avoided (Schedules now has `settings_can_mutate=True`).
  - **Unstubbed egress.** Probe tests get "unreachable: blocked by network policy": all four parameters of `test_probe_settings_endpoint_maps_transport_failures`, `test_probe_settings_endpoint_reports_http_status_and_invalid_url`, and probably `test_probe_settings_endpoint_counts_models_and_normalizes_path`, which sends no request.
  - **Stale assertions:**
    - `test_failure_status_text_keeps_type_name_only_copy_unchanged`: the copy says "Logs (F3)" where the test expects "(F8)".
    - `test_settings_category_summaries_cover_every_category_id_exactly_once`: `33 == 31`.
    - `test_inspector_guidance_covers_every_settings_category`: TOOLS has no guidance entry.
    - `test_settings_pane_widths_are_owned_by_stylesheet_not_inline_python`: `ValueError: substring not found`.
- **`Tests/UI/test_settings_endpoint_probe.py`**: run alone, all 101 nodes error at setup through `Tests/UI/conftest.py:193`. The file is not in `Tests/conftest.py`'s list of files kept on the bootstrap profile. Its probe tests also use `https://example.test/...` without stubbing DNS or egress, so the egress check added in f25f707e6e (TASK-32894) reports "unreachable: blocked by network policy".
- **`Tests/Chat/test_local_server_discovery.py::test_probe_handles_recursive_json_as_bounded_failure`** fails with `TypeError`: the test's `json.loads` stub does not accept `object_pairs_hook`.

TASK-35572 already lists two of the hub file's nodes for lane placement. This task does not repeat that.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every test in the named files passes on dev, or is retired with a recorded reason; the task notes list the full failure inventory found at the start
- [x] #2 Each named file runs on a PR lane, or the reason it does not is recorded
- [x] #3 The task notes explain why load_settings() raises RecoveryRequired in these files' plain tests, and the cause is either fixed or recorded as intended
<!-- AC:END -->

## Implementation Plan

1. Record the full failure inventory: run each named file alone (`-n 4`) at base 52b3207ff1 and list every red node with its cause.
2. AC#3 first: trace `load_settings()`'s `RecoveryRequired: raw_source_selection_changed` through `Tests/conftest.py`'s bootstrap profile, `isolate_test_environment`'s per-test redirect and `Backup_Recovery/raw_participants.py`; decide whether the guard is right and the tests are wrong.
3. Fix each file by test-infrastructure changes only, using the seams the repo already has: `@private_profile_test` for tests that build or write app config; the keep-bootstrap-profile list for a file of pure tests that only trip on the app import. No change to the production guard.
4. Fix the other causes against current product truth: stub egress DNS/config the way `Tests/Utils/test_settings_probe_egress.py` does; narrow the `json.loads` stubs to the module under test; update stale assertions after checking the code.
5. AC#2: add the Tests/UI files (or nodes, if a file is too heavy) to the UI census tail with measured runtimes and raise `MINIMUM_FILES`; record why `Tests/Chat/test_local_server_discovery.py` cannot join a lane; do not repeat TASK-35572's two hub nodes.
6. Re-run every named file alone; run preflight and the census checker; write the notes.

## Implementation Notes

Test-infrastructure fixes, plus one production change from review round 1: Tools & MCP's missing inspector guidance entry. The recovery guard is unchanged. Base dev 52b3207ff1.

**Failure inventory at the start** (each file run alone, `-n 4`, through the scratch-profile wrapper):
- `Tests/UI/test_settings_model_discovery_journeys.py`: 15 failed, 11 passed. All 15 were `RecoveryRequired: raw_source_selection_changed` (`raw_participants.py:307`) in the call phase: `_app_scope()` -> `_build_test_app()` -> `build_test_app_config()` -> `load_settings()`. They are the 4 functions without `@private_profile_test`; the 11 passing nodes already had it.
- `Tests/UI/test_settings_endpoint_probe.py`: 101 errors at setup. `Tests/UI/conftest.py`'s `_disable_model_catalog_refresh` monkeypatches `"tldw_chatbook.app.TldwCli._refresh_model_catalogs"` by string, which makes it the first importer of `tldw_chatbook.app`, inside the per-test sandbox; `app.py`'s module-level `load_settings()` is refused. Hidden behind that: 28 nodes probe `https://example.test/...`, which has no DNS, so the probe's egress check (TASK-32894) answered "blocked by network policy". Negative control: with the profile fixed and the DNS stub disabled, 28 failed, 73 passed.
- `Tests/Chat/test_local_server_discovery.py`: 1 failed, 58 passed. `test_probe_handles_recursive_json_as_bounded_failure` patched `local_discovery_module.json.loads`, which is the stdlib `json.loads` for the whole process; `probe_models_endpoint`'s storage admission then read a control record with `json.loads(..., object_pairs_hook=...)` and got `TypeError` (`Backup_Recovery/bootstrap.py:170`).
- `Tests/UI/test_settings_configuration_hub.py`: 242 failed, 216 passed of 458.
  - 231 `RecoveryRequired: raw_source_selection_changed` in the call phase, from `_build_test_app()` -> `load_settings()` (5 of these first pointed `TLDW_CONFIG_PATH` at a tmp file; 4 diagnostics tests did the same before building the app).
  - 6 probe nodes (`test_probe_settings_endpoint_maps_transport_failures` x4, `..._reports_http_status_and_invalid_url`, `..._counts_models_and_normalizes_path`): the same `RecoveryRequired`, raised by the egress check's config read and swallowed by the probe's `except Exception` into "unreachable: blocked by network policy". They probe `127.0.0.1`, so DNS was never the cause.
  - 5 stale assertions: `test_failure_status_text_keeps_type_name_only_copy_unchanged` and `test_failure_status_text_never_carries_raw_exception_text` expected "Logs (F8)" (the Logs key is F3 since 0b7318946b); `test_settings_category_summaries_cover_every_category_id_exactly_once` pinned 31 (Dreams and Guardian, 1b769eb757, made it 33); `test_settings_pane_widths_are_owned_by_stylesheet_not_inline_python` read `components/_agentic_terminal.tcss`, which lost those rules to `features/_settings.tcss` in TASK-24451; `test_inspector_guidance_covers_every_settings_category` found no guidance entry for TOOLS (see AC#1 below).
  - Stale assertions that showed only once RecoveryRequired was gone: Schedules is `settings_can_mutate=True` (`..._grouped_and_have_ownership_records`); Dreams and Guardian have domain contracts (`..._contracts_are_explicit_about_mutation_scope`); "priv" now matches 3 categories, My Profile included since 6c52a2b4b2 (two category-search tests); the Privacy & Security action row sits below the 180x50 fold, under the footer, so a click hit `footer-key-quit` (`..._recovery_actions_navigate_to_existing_categories`).

**AC#3: why `load_settings()` raises.** The guard is behaving as intended, and the tests were wrong. `Tests/conftest.py` sets `TLDW_CONFIG_PATH` to the session's bootstrap profile and then imports `tldw_chatbook.Chunking`, `RAG_Search.config_profiles` and `model_capabilities`. Their guarded config reads register `tldw_chatbook.config` as a raw participant whose `selected` path is the bootstrap config (`_register_raw_participant`). For every node that is not on its keep-bootstrap list, carries no `bootstrap_profile` marker and is not a private-profile child, `isolate_test_environment` monkeypatches `TLDW_CONFIG_PATH` (and HOME/XDG) to `tmp_path`. `load_settings()`'s warm-cache path is keyed by the effective path, so the first read on the new path misses and goes through `_load_settings_guarded` -> the raw config scope -> `_register_raw_participant`, whose `binding` now resolves the per-test path, not the registered one -> `RecoveryRequired("raw_source_selection_changed")`. ADR-126 (item 9: profiles reopen "without hot switching") binds a live config source to one profile for the process lifetime, and refusing a changed selection under a live source is the point; the production guard is unchanged. Which code touches config first decides only where the error lands: at setup when the Tests/UI conftest is the first to import the app (the endpoint probe file; TASK-36384 found the same in `Tests/ProductionApp/conftest.py`), and in the call phase when the module imported the app at collection and the test body builds app config (hub, discovery journeys). The bootstrap-profile file list matters because a listed file keeps the session path, so its reads hit the registered selection. `Tests/UI/app_factory.py`'s docstring said the redirect made `load_settings()` hermetic; it now says the opposite and names the two opt-ins.

**Fixes.**
- Discovery journeys: `@private_profile_test` + `request` on the 4 plain functions.
- Endpoint probe: module `pytestmark` adds `bootstrap_profile` (pure tests; they write no config and build no app); an autouse fixture resolves every host to 93.184.216.34 through `egress._resolve`/`_resolve_async`, the idiom `Tests/Image_Generation/test_http_client.py` uses; the recursive-JSON stub replaces only the module's `json`.
- Local server discovery: the stub replaces `local_discovery_module.json` with `SimpleNamespace(loads=stub)`.
- Hub: `@private_profile_test` on the 211 failing functions that build an app (171 async, 40 sync), `bootstrap_profile` on the 3 probe functions that build none. A first pass put the 40 sync builders on `bootstrap_profile`; a full run then failed the untouched `test_settings_console_behavior_saves_display_name_exactly`, because `build_test_app_config()` returns `load_settings()`'s cached dict by reference, and on the shared bootstrap profile one test's `app.app_config[...] = ...` reaches the next test in that worker. The 5 tests that pointed `TLDW_CONFIG_PATH` at a tmp file now seed the selected profile's file (`_seed_active_config`, the `_seed_config` idiom of `test_settings_dreams.py`, which imports the hub so cannot be imported here); the 4 diagnostics tests build the app before redirecting the path, so only `_config_path()` sees the override. Stale assertions were updated to the product truth above, each checked in the code.

**AC#1.** Every test in the four files passes. Final runs (`-n 4`): discovery journeys 26 passed; endpoint probe 101 passed; local server discovery 59 passed; hub 455 passed, 3 failed under load (average about 27 on 18 CPUs), and those 3 then passed 3 of 3 reruns. `test_inspector_guidance_covers_every_settings_category` passes because review round 1 fixed the product gap it exposed (below). The hub's mounted tests are load-sensitive: across five loaded full runs, 2 to 12 per run failed for reasons other than the stale assertions, and each passed on rerun. They include private-profile tests that passed at base and whose bodies this task did not touch: `test_settings_paste_toggle_keeps_keyboard_focus_after_refresh`, and the save-count tests `test_settings_provider_category_saves_selected_model_profile` (2 of the last 4 loaded runs) and `test_settings_saves_each_mistral_entry_to_its_distinct_owner[mistral]`. Typical errors were teardown "Native app workers remain uncertain" after a `TimeoutError`, a missing `SelectOverlay`, or a save count of 0.

**Review round 1.**
- Tools & MCP (TASK-35150) replaced the MCP Defaults domain category without adding an `_INSPECTOR_GUIDANCE` entry, so its inspector showed the generic fallback and every run logged "No inspector guidance entry". `settings_screen.py` now has a TOOLS entry, and the strict xfail is gone. Each row was checked against the code:
  - "Affected config" (completed in review round 2): the Tools & MCP pane writes `[tools]` gate keys and the Local group's master gate `[console] local_tools_enabled` (`all_tool_gates()` in `builtin_tool_gate.py`; `mcp_servers_mode.py` routes both), `[mcp]` enabled/expose_* (`MCPWorkbench._save_builtin_flag`), the permission-store Tool policy (`TOOLS_HUB_OWNERSHIP_RECORD`), and server definitions. A local server form saves a stdio profile into `local_mcp_store.json` (`save_local_profile` -> `LocalMCPStore.save_profile`); in server source the form creates or updates the external server on the connected server (`server_unified_service.create_external_server`). The pane never writes `mcp_server_targets.json`: it only calls `target_store.list_targets()`. So the row says "saved MCP server definitions", not "saved server targets". A temporary assertion in `test_settings_registers_tools_category_and_mounts_pane` confirmed all three rows render in the mounted Tools inspector, `[tools]` literally, and the fallback copy is gone; the assertion was then removed.
  - "Recovery": re-opening the category builds a fresh workbench that reloads (cached snapshots only when the catalog fingerprint is unchanged, `_collect_snapshots_stale_gated`), and `r` calls `refresh_now()` -> `workbench.reload(force=True)` from inside the pane and from the rail.
  - "Boundary": the category banner's own line, plus the record's Tool Profiles boundary.
  - The base run's failure of that guard is its red; it is green now. The `settings_screen.py` row in `Tests/Architecture/test_module_size_ratchet.py` grows by the entry's 14 lines and is set to the measured size, under the 2026-10-04 owner ruling. `Docs/User_Guide/settings.md` lists no per-category inspector rows, so it needs no edit.
- `test_state_banner_is_pinned_outside_detail_scroll` left `run_test` while Library/RAG's off-thread `_rag_index_status_worker` was still running, so teardown refused config retirement. It now calls `_settle_settings_mount_storm(pilot)` after opening the category. Result: 5 of 5 serial passes, and it passed in the loaded full run.
- `_seed_active_config(request, text)` asserts `is_private_profile_child(request)` before writing. Negative control: with the decorator temporarily removed from one caller, the test failed on "needs @private_profile_test" before any write.
- Review round 2 left `TOOLS_HUB_OWNERSHIP_RECORD.owns_config_sections` alone, so the inspector's "Owns" row still lists only the two MCP sections. That tuple has two consumers: the "Owns" row and the category search's owned-keys tier (`_category_search_rank`, rank 5). Adding "[console] tool gates" would list Tools & MCP for every "console" search and claim a `[console]` section that Console Behavior owns, apart from one key. That widens the record's meaning. No test pins the record.
- `app_factory.py`'s docstring limits `bootstrap_profile` to tests that neither save config nor assign into `app.app_config`, and its Returns line names the selected profile's config file.
- `Tests/Architecture/test_module_size_ratchet.py` has 8 other rows red on dev 52b3207ff1 (`mcp_workbench.py`, `change_review_screen.py`, `llm_screen.py`, `console_transcript.py`, `app_command_providers.py`, `app_ingest_queue.py`, `app_lifecycle.py`, `tldw_api/client.py`). None of those files differ from base, and their base line counts equal the reported counts.

**AC#2.** Census tail (+8 entries, floor raised to the entry count): the endpoint probe file whole (10-20 s alone); one discovery node, `test_late_discovery_cannot_publish_into_a_changed_form[False-endpoint_roundtrip]` (a private-profile child, 13-22 s); six hub nodes, the guards that went stale unseen (the ownership-record one is a private-profile child, about 8 s; the rest 1-2 s). The eight together took 40 s serially here, and round-robin puts each in a different shard. The rest of the discovery file (25 nodes, 2.5 min at `-n 4`) and the whole hub file (458 nodes, 18-26 min at `-n 4` here) stay out for shard time; the hub already had 12 census nodes. TASK-35572's two hub nodes (`test_settings_provider_switch_resets_staged_model_for_each_provider_transition`, `test_settings_provider_switch_updates_inspector_readiness`) now pass under `@private_profile_test` but are not added here, because that task owns their placement. `Tests/Chat/test_local_server_discovery.py` cannot join the UI census (Tests/UI only), and its lane would be a target in `.github/workflows/derived-artifacts.yml`, which this burn-down may not edit (draft PR #3036 changes it), so it runs on no PR lane yet.

**Red/green.** The base inventory above is the red for every rewritten node; each is green in the runs listed. The 28-node DNS negative control is recorded above. The two lessons added to `backlog/docs/lessons-testing-evidence.md` give the swallowed-RecoveryRequired and global `json.loads` incidents.

**Re-sync (2026-10-29).** Rebased onto dev ff815f8559, a clean rebase apart from four conflicts:
- `lessons-testing-evidence.md`: both sides kept, with dev's new entries first.
- Census: dev's census kept exactly; this task's 8 entries appended after dev's last tail entry with no blank line, as dev's tail convention requires. Dev already lists TASK-35648/35649's two nodes, so the copies from the old base were dropped. Floor set to the entry count.
- Checker floor comment: states only the increment.
- Module ratchet row: dev's comments kept, this task's +14 comment added, row set to the measured size.

Dev's #3226 Tools & MCP copy changes leave every TOOLS guidance claim true, and dev added no TOOLS entry of its own. All 211 `@private_profile_test` conversions still apply; no app-building test is left unmarked.

Re-verified:
- endpoint probe 101, local server discovery 59, discovery journeys 26 passed;
- the guidance test plus `test_settings_tools_hub.py`: 21 passed;
- census checker OK, and the collected census (3980 nodes, `--keep-duplicates`) has 0 duplicates;
- Tests/CI: 1060 passed, 1 skipped;
- preflight passed;
- module ratchet: the settings_screen row passes, and the same 8 untouched rows are red on dev too.

The full hub run, at a load average of 145-177 on 18 CPUs, gave 450 passed and 8 failed. 6 of the 8 are this task's own conversions: they are plain tests on dev, so they are red there through RecoveryRequired, the bug AC#3 explains. The other 2 were already private-profile tests on dev and passed there. At head, all 8 passed on a rerun at lower load, so the head failures are load-related, not regressions.

**Re-sync 2026-10-31 onto dev 7536ac3b87.** The only conflict was the census tail and floor, where dev's TASK-33622.6 appended five nodes and TASK-33621.51 two files. Dev's census was kept, this task's block re-appended after dev's last entry, and `MINIMUM_FILES` again equals the entry count. The range-diff shows no other change. Preflight and the census tests in `Tests/CI` pass.

**Re-sync 2026-10-31 onto dev 804dc00231.** The only conflict was the lessons file, where dev's TASK-36389 lesson took the same spot. This task's two lessons are unchanged. They now sit just before TASK-33620.15.8's entry, still in the 2026-10-23 group, so they no longer share a spot with TASK-36385's lesson (PR #3225), which would have conflicted with whichever of the two merged second. Apart from the lessons file, the diff is identical to before. `settings_screen.py`'s module-ratchet row still passes. The same 8 other module rows fail on dev 7536ac3b87 too, and dev changed no `tldw_chatbook/` file between that commit and 804dc00231. Preflight passed.

**Re-sync 2026-10-31 onto dev 57e330e735.** The only conflict was the census tail and floor, after #3224 (TASK-36380, +1) and #3227 (TASK-36382, +9) appended their blocks. Dev's census was kept, this task's 8 entries were re-appended after dev's last entry, and `MINIMUM_FILES` equals the entry count again (dev's 1173 + 8 = 1181). Outside the census, the diff is identical to before. Dev's #3286 re-pinned other rows in `test_module_size_ratchet.py` and left `settings_screen.py` unchanged; its row is still the exact measurement (35616 = dev's 35602 + this task's 14). Preflight passed; the census tests in `Tests/CI` gave 400 passed; this task's test files gave 644 passed (`-n 8`), including the whole hub file.

**Files.** `tldw_chatbook/UI/Screens/settings_screen.py` (TOOLS guidance), `Tests/Architecture/test_module_size_ratchet.py`, `Tests/UI/test_settings_configuration_hub.py`, `Tests/UI/test_settings_model_discovery_journeys.py`, `Tests/UI/test_settings_endpoint_probe.py`, `Tests/Chat/test_local_server_discovery.py`, `Tests/UI/app_factory.py` (docstring), `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `backlog/docs/lessons-testing-evidence.md`.
