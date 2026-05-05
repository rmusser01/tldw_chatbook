# Phase 2.2 Console Chatbook Artifact Save Contract

Date: 2026-05-05
Task: TASK-9.2
Branch: codex/product-maturity-phase2-2-artifact-save-contract
Workflow: Console assistant response -> Save artifact action -> local Chatbook artifact record

## What Was Verified

Phase 2.2 verifies the next narrow Phase 2 gate: a completed assistant response in Console can be saved as a local Chatbook artifact record without leaving the chat workflow.

Verified contract:

- Basic `ChatMessage` assistant responses expose a visible save-to-artifact action.
- Enhanced `ChatMessageEnhanced` assistant responses expose the same save-to-artifact action.
- User messages do not expose the assistant artifact save action.
- The artifact action routes through `app.local_chatbook_service.create_chatbook(...)`.
- Created records include bounded Console provenance metadata: source, artifact kind, conversation ID, message ID, role, provider, model, bounded content, and truncation status.
- Missing local Chatbook service and create failures surface recovery notifications without deleting the source message.

## Automated Evidence

Initial red run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Event_Handlers/Chat_Events/test_chat_events.py::test_handle_chat_action_button_pressed_saves_ai_message_as_chatbook_artifact Tests/Event_Handlers/Chat_Events/test_chat_events.py::test_handle_chat_action_button_pressed_reports_missing_chatbook_service Tests/Event_Handlers/Chat_Events/test_chat_events.py::test_handle_chat_action_button_pressed_reports_chatbook_create_failure Tests/Widgets/test_chat_message_artifact_actions.py -q
```

Result:

- `5 failed, 2 passed, 1 warning in 2.64s`.
- Failures were expected: no `#save-artifact` button existed and the action handler never called `local_chatbook_service.create_chatbook`.

Focused green verification:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Event_Handlers/Chat_Events/test_chat_events.py::test_handle_chat_action_button_pressed_saves_ai_message_as_chatbook_artifact Tests/Event_Handlers/Chat_Events/test_chat_events.py::test_handle_chat_action_button_pressed_reports_missing_chatbook_service Tests/Event_Handlers/Chat_Events/test_chat_events.py::test_handle_chat_action_button_pressed_reports_chatbook_create_failure Tests/Widgets/test_chat_message_artifact_actions.py -q
```

Result:

- `7 passed, 1 warning in 1.70s`.

Affected event-handler verification:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Event_Handlers/Chat_Events/test_chat_events.py -q
```

Result:

- `11 passed, 1 warning in 4.41s`.

Affected widget verification:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Widgets/test_chat_message_enhanced.py Tests/Widgets/test_chat_message_artifact_actions.py -q
```

Result:

- `27 passed, 5 warnings in 3.59s`.

Adjacent Artifacts/Console handoff verification:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_live_work_handoffs.py::test_artifacts_destination_keeps_console_launch_disabled_without_chatbooks Tests/UI/test_console_live_work_handoffs.py::test_artifacts_destination_launches_latest_local_chatbook_in_console Tests/UI/test_console_live_work_handoffs.py::test_artifacts_destination_sanitizes_chatbook_metadata_before_console_launch Tests/UI/test_console_live_work_handoffs.py::test_artifacts_destination_uses_numeric_id_tie_break_for_latest_chatbook Tests/UI/test_console_live_work_handoffs.py::test_artifacts_destination_distinguishes_chatbook_service_failure_from_empty_state -q
```

Result:

- `5 passed, 3 warnings in 9.98s`.

Tracker regression:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_harness.py::test_product_maturity_tracker_links_phase_one_harness_and_tasks -q
```

Result:

- `1 passed in 0.12s`.

Diff hygiene:

```bash
git diff --check
```

Result:

- Passed with no whitespace errors.

## Defects Fixed

- `workflow-degradation`: Console assistant responses had no visible path to create a Chatbook artifact, so the Phase 2 loop stopped at generated text.
- `recoverability`: artifact persistence failures had no dedicated recovery message because the action did not exist.

## UX Notes

- The save action is assistant-only so user prompts are not accidentally persisted as result artifacts.
- The action stays inline with existing message actions to preserve power-user speed.
- The source message remains visible on every failure path, which preserves user control and retryability.
- The metadata is intentionally bounded so long assistant responses do not create unbounded registry payloads.

## Residual Risk

- This gate creates a local Chatbook registry record; it does not export a full `.chatbook` package.
- Artifacts reopen and route-back into Console were verified earlier for the latest existing local Chatbook, but this gate does not prove the newly saved record can be selected from Artifacts.
- Home resume/open controls for newly saved artifacts remain unverified.
- Full live-provider generation remains outside this gate; the save contract is tested at the completed-message/widget-handler layer.

## Exit Decision

Pass for Phase 2.2. Console now has a tested assistant-response-to-local-Chatbook-artifact save contract with recovery notifications. Phase 2 should continue with newly saved artifact visibility, reopen, and Home resume gates.
