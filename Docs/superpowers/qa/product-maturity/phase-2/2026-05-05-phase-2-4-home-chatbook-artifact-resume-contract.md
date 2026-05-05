# Phase 2.4 Home Chatbook Artifact Resume Contract

Date: 2026-05-05
Task: TASK-9.4
Branch: codex/product-maturity-phase2-4-home-artifact-resume
Workflow: Console-saved Chatbook artifact -> Home active work -> Artifacts details or Console resume

## What Was Verified

Phase 2.4 verifies the final Home-resume seam for the Phase 2 core loop: a Console-saved local Chatbook artifact can appear on Home as resumable work and route back to Artifacts or Console without manual state reconstruction.

Verified contract:

- `LocalChatbookService.list_home_artifact_snapshot(...)` returns a synchronous latest-first snapshot of Console-saved assistant-response Chatbook artifacts.
- Home active-work input includes only the latest Console-saved Chatbook artifact when the local Chatbook service exposes one.
- Home renders the saved artifact as active work with `Open details` and `Open in Console` controls.
- `Open details` routes the artifact target to Artifacts.
- `Open in Console` launches the artifact with Chatbook id, record id, tags, categories, file path, source metadata, conversation id, message id, provider, model, content preview, and truncation status.
- Missing or failing Chatbook snapshot state fails closed without removing existing notification or W+C Home behavior.

## Automated Evidence

Initial red run:

```bash
.venv/bin/python -m pytest Tests/Chatbooks/test_local_chatbook_service.py::test_local_chatbook_service_home_artifact_snapshot_lists_latest_console_saved_artifacts Tests/Home/test_active_work_adapter.py::test_local_notification_adapter_maps_console_saved_chatbook_to_active_work Tests/Home/test_active_work_adapter.py::test_local_notification_adapter_opens_console_saved_chatbook_details_and_console Tests/UI/test_screen_navigation.py::test_app_initializes_watchlists_and_notifications_services -q
```

Result:

- `4 failed, 10 warnings in 10.69s`.
- Failures were expected: the local Chatbook service had no synchronous Home snapshot, the Home adapter did not accept a Chatbook service, and app wiring did not pass the local Chatbook service into Home.

Focused green verification:

```bash
.venv/bin/python -m pytest Tests/Chatbooks/test_local_chatbook_service.py::test_local_chatbook_service_home_artifact_snapshot_lists_latest_console_saved_artifacts Tests/Home/test_active_work_adapter.py::test_local_notification_adapter_maps_console_saved_chatbook_to_active_work Tests/Home/test_active_work_adapter.py::test_local_notification_adapter_opens_console_saved_chatbook_details_and_console Tests/UI/test_screen_navigation.py::test_app_initializes_watchlists_and_notifications_services -q
```

Result:

- `4 passed, 8 warnings in 5.20s`.

Home visible-control verification:

```bash
.venv/bin/python -m pytest Tests/UI/test_home_screen.py::test_home_saved_chatbook_artifact_resume_controls_pass_artifact_target Tests/Home/test_active_work_adapter.py::test_local_notification_adapter_maps_console_saved_chatbook_to_active_work Tests/Home/test_active_work_adapter.py::test_local_notification_adapter_opens_console_saved_chatbook_details_and_console -q
```

Result:

- `3 passed, 8 warnings in 6.13s`.

Latest-only and fail-closed guardrail verification:

```bash
.venv/bin/python -m pytest Tests/Home/test_active_work_adapter.py::test_local_notification_adapter_maps_only_latest_console_saved_chatbook_to_active_work Tests/Home/test_active_work_adapter.py::test_local_notification_adapter_fails_closed_when_chatbook_snapshot_unavailable -q
```

Result:

- `2 passed in 0.23s`.

Broader focused verification:

```bash
.venv/bin/python -m pytest Tests/Chatbooks/test_local_chatbook_service.py Tests/Home/test_active_work_adapter.py Tests/UI/test_home_screen.py Tests/UI/test_screen_navigation.py::test_app_initializes_watchlists_and_notifications_services Tests/UI/test_product_maturity_phase1_harness.py -q
```

Result:

- `49 passed, 8 warnings in 26.93s`.

## Defects Fixed

- `workflow-degradation`: Home had no way to surface a newly saved Console Chatbook artifact as resumable work after Phase 2.3 proved Artifacts could reopen it.
- `recognition`: The user had to remember to go to Artifacts manually; Home did not expose the saved artifact in its active-work model.

## UX Notes

- The saved artifact uses the existing Home active-work model instead of adding a new dashboard section.
- Details route to Artifacts, which remains the authority for generated outputs and Chatbooks.
- Console launch reuses the existing Home control path and preserves the saved-response provenance needed by Console status cards.

## Residual Risk

- Home currently surfaces the latest Console-saved Chatbook artifact, not a full artifact history picker.
- Full `.chatbook` export packaging remains outside this gate.
- Phase 2 should still receive a final closeout replay that walks the full source/question -> grounded answer -> save -> Home resume path in one test or QA artifact.

## Exit Decision

Pass for Phase 2.4. Home can now resume the latest Console-saved Chatbook artifact through Artifacts or Console. Phase 2 should continue with a closeout replay of the complete core loop.
