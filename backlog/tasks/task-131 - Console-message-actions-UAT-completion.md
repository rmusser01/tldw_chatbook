---
id: TASK-131
title: Console message actions UAT completion
status: Done
assignee:
- '@codex'
created_date: 2026-06-21 00:36
labels:
- console
- messages
- uat
dependencies:
- TASK-128
- TASK-129
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and harden message selection and per-message actions so users can operate on transcript messages with both mouse and keyboard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 User can select a transcript message with mouse and keyboard and see the selected-message action row.
- [x] #2 Enter activates the selected message or focused action consistently, and Tab moves between major Console screen areas.
- [x] #3 Copy works and provides visible status feedback.
- [x] #4 Edit opens a dedicated edit modal and preserves the original message when cancelled or invalid.
- [x] #5 Save as opens the conversion modal and clearly labels unavailable destinations as unavailable or WIP.
- [x] #6 Regenerate creates variants and exposes previous/next controls for choosing which variant to continue from.
- [x] #7 Continue extends the selected message/thread context according to its documented meaning.
- [x] #8 Thumbs up/down and delete provide clear feedback and safe recovery or confirmation where destructive.
- [x] #9 Focused regression coverage verifies mouse and keyboard paths for the action row and the implemented actions.
- [x] #10 Rendered CDP/Textual-web evidence is captured before approval and PR completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: TASK-131 verifies and hardens existing Console transcript action behavior. It should not change message storage schema, long-lived variant ownership, provider/runtime contracts, workspace ownership, or destructive-action policy.

1. Audit current `origin/dev` Console message-action tests and implementation to identify which TASK-131 acceptance criteria are already covered.
2. Run the focused message-action regression subset before changing production code.
3. Add failing regression coverage only for uncovered behavior, prioritizing keyboard selection/Enter activation, Save as destination honesty, regenerate variant controls, Continue semantics, and destructive-action feedback.
4. Implement the smallest safe fixes needed to pass any new failing tests, avoiding provider readiness, workspace state, and tab/composer lifecycle changes.
5. Capture actual CDP/Textual-web screenshots for selected-message action row and modal/action feedback states before approval.
6. Update TASK-131 and the parallel acceptance matrix with verified evidence, then rerun the focused regression subset and `git diff --check`.
<!-- SECTION:PLAN:END -->

## Parallel Ownership

Owns transcript message selection and per-message action behavior. Avoid provider readiness, workspace state, and tab/composer lifecycle except where required to create fixture messages for action testing.

ADR required: no, unless implementation changes message storage schema, variant ownership, or destructive-action policy.
ADR path: N/A unless implementation planning identifies a contract change.
Reason: Expected work is behavior hardening over existing message/action UI seams.

## Current QA Evidence

- Added focused regressions for selected-message inspector guidance, safer delete confirmation, Save as message context/excerpts, and friendly workspace conversation labels.
- Verified locally: `python -m pytest -q Tests/UI/test_console_native_chat_flow.py::test_console_selected_message_updates_inspector_action_guidance Tests/UI/test_console_native_chat_flow.py::test_console_selected_message_delete_action_removes_message_from_transcript Tests/UI/test_console_native_chat_flow.py::test_console_save_as_modal_labels_unwired_destinations_as_wip Tests/UI/test_console_native_chat_flow.py::test_console_send_refreshes_workspace_conversation_rail_after_persistence --tb=short` passed, 4 passed.
- Verified locally: `python -m pytest -q Tests/UI/test_console_native_chat_flow.py -k "message_action or selected_message or continue_action or regenerate_action or save_as or workspace_conversation" --tb=short` passed, 18 passed.
- CDP/Textual-web evidence captured on `127.0.0.1:8936`: `Docs/superpowers/qa/console-uat-parallelization/task-131-selected-message-inspector-expanded-cdp-2026-06-21.png` and `Docs/superpowers/qa/console-uat-parallelization/task-131-save-as-context-cdp-2026-06-21.png`.
- Approval status: approved after user review of the actual rendered screenshots.

## Implementation Notes

- Hardened the selected-message action flow by synchronizing transcript selection changes into the Console inspector, exposing selected message role/action/keyboard/variant/excerpt guidance, and preserving keyboard activation paths.
- Added selected-message context to the Save as modal so users can confirm which message is being converted, while keeping unavailable destinations explicitly labeled as WIP.
- Made destructive delete safer by requiring a second activation after visible confirmation feedback.
- Cleaned up workspace conversation labels so the Console rail shows friendly workspace status without leaking raw short IDs.
- Added focused mounted regressions and recorded approved CDP/Textual-web evidence for the selected-message inspector and Save as modal states.
