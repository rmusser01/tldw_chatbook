# Phase 2.5 Core Loop Closeout Replay

Date: 2026-05-06
Task: TASK-9.5
Branch: codex/product-maturity-phase2-5-closeout-replay
Workflow: source/question -> grounded Console -> saved Chatbook -> Artifacts reopen -> Home resume

## What Was Verified

Phase 2.5 closes the Phase 2 product-maturity loop by chaining the verified child contracts into one complete user workflow:

`source/question -> grounded Console -> saved Chatbook -> Artifacts reopen -> Home resume`

Verified contract chain:

- Phase 2.1 proves staged Search/RAG context reaches the model-bound Console request with local source authority and remains staged when send is blocked before generation starts.
- Phase 2.2 proves a completed Console assistant response can be saved as a local Chatbook artifact with bounded Console provenance and recoverable failure notifications.
- Phase 2.3 proves Artifacts recognizes Console-saved Chatbook artifact records and reopens them into Console with saved-response provenance.
- Phase 2.4 proves Home surfaces the latest Console-saved Chatbook artifact as resumable active work and routes it to Artifacts or Console.
- Phase 2.4 review closeout proves a mixed W+C watchlist run plus Console-saved Chatbook artifact state keeps the Chatbook artifact reachable through dedicated Home controls.

This closeout is a focused regression-backed replay of the complete local product loop. It does not claim live external-provider generation coverage.

## Automated Evidence

Pre-closeout seam verification:

```bash
.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_core_loop.py::test_search_rag_result_stages_context_into_console_core_loop Tests/Event_Handlers/Chat_Events/test_chat_events.py::test_handle_chat_send_applies_staged_search_rag_context_to_provider_message Tests/Event_Handlers/Chat_Events/test_chat_events_tabs.py::TestChatEventsTabsHandlers::test_tab_send_preserves_handoff_payload_when_original_handler_does_not_dispatch Tests/Event_Handlers/Chat_Events/test_chat_events.py::test_handle_chat_action_button_pressed_schedules_chatbook_artifact_worker Tests/Widgets/test_chat_message_artifact_actions.py Tests/UI/test_console_live_work_handoffs.py::test_artifacts_destination_reopens_console_saved_chatbook_with_provenance Tests/Home/test_active_work_adapter.py::test_local_notification_adapter_opens_console_saved_chatbook_details_and_console Tests/Home/test_dashboard_state.py::test_dashboard_summary_keeps_chatbook_artifact_reachable_when_mixed_with_watchlist_run Tests/UI/test_home_screen.py::test_home_mixed_active_work_exposes_chatbook_artifact_resume_controls -q
```

Result:

- `14 passed, 10 warnings in 16.13s`.

Closeout tracking verification:

```bash
.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_harness.py::test_product_maturity_phase_two_closeout_evidence_links_parent_task_and_tracker -q
```

Result:

- `1 passed in 0.08s`.

Broader focused closeout verification:

```bash
.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_core_loop.py::test_search_rag_result_stages_context_into_console_core_loop Tests/Event_Handlers/Chat_Events/test_chat_events.py::test_handle_chat_send_applies_staged_search_rag_context_to_provider_message Tests/Event_Handlers/Chat_Events/test_chat_events_tabs.py::TestChatEventsTabsHandlers::test_tab_send_preserves_handoff_payload_when_original_handler_does_not_dispatch Tests/Event_Handlers/Chat_Events/test_chat_events.py::test_handle_chat_action_button_pressed_schedules_chatbook_artifact_worker Tests/Widgets/test_chat_message_artifact_actions.py Tests/UI/test_console_live_work_handoffs.py::test_artifacts_destination_reopens_console_saved_chatbook_with_provenance Tests/Home/test_active_work_adapter.py::test_local_notification_adapter_opens_console_saved_chatbook_details_and_console Tests/Home/test_dashboard_state.py::test_dashboard_summary_keeps_chatbook_artifact_reachable_when_mixed_with_watchlist_run Tests/UI/test_home_screen.py::test_home_mixed_active_work_exposes_chatbook_artifact_resume_controls Tests/UI/test_product_maturity_phase1_harness.py -q
```

Result:

- `23 passed, 8 warnings in 18.03s`.

## P0/P1 Disposition

No open P0/P1 defects remain for the Phase 2 local core loop.

- The prior `workflow-degradation` risk where Home could not resume Console-saved Chatbook artifacts was fixed and verified in Phase 2.4.
- The review-discovered mixed W+C plus Chatbook reachability risk was fixed before closeout: Home now exposes dedicated Chatbook controls instead of allowing watchlist controls to strand the saved artifact.
- Live external-provider generation is not counted as a Phase 2 P0/P1 blocker because the local loop contract is verified through deterministic request construction, artifact persistence, reopen, and resume seams.

## UX Notes

- Beginner orientation remains visible through Home active-work labels, Artifacts provenance copy, and Console source authority status.
- Power-user speed is preserved: Search/RAG can stage directly into Console, assistant responses can be saved inline, Artifacts can reopen directly into Console, and Home can resume the latest saved Chatbook without manual reconstruction.
- Chat remains the primary agentic control surface; Artifacts and Home are recovery and continuation surfaces for the same loop, not competing destinations.

## Residual Risk

- Live provider or local-model generation was not exercised in this closeout because the deterministic test boundary avoids API keys and optional model services.
- Full `.chatbook` export packaging remains outside this Phase 2 closeout; Phase 2 verifies local Chatbook artifact records and reopen/resume behavior.
- Home currently surfaces the latest Console-saved Chatbook artifact, not a full artifact history picker.

## Exit Decision

Pass for Phase 2.5. Phase 2 is verified for the local core agentic loop: source/question context can reach Console, save to a Chatbook artifact, reopen through Artifacts, and resume from Home without manual state reconstruction.
