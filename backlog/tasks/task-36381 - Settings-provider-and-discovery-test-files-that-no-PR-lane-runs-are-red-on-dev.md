---
id: TASK-36381
title: Settings provider and discovery test files that no PR lane runs are red on dev
status: To Do
created_date: 2026-10-22 23:58
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
- [ ] #1 Every test in the named files passes on dev, or is retired with a recorded reason; the task notes list the full failure inventory found at the start
- [ ] #2 Each named file runs on a PR lane, or the reason it does not is recorded
- [ ] #3 The task notes explain why load_settings() raises RecoveryRequired in these files' plain tests, and the cause is either fixed or recorded as intended
<!-- AC:END -->
