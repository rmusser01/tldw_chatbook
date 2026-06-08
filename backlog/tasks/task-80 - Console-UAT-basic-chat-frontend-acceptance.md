---
id: TASK-80
title: Console UAT basic chat frontend acceptance
status: Done
labels:
- console
- uat
- ux
- chat
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and harden Console as a basic usable chat front-end: users can create/start chats, change provider/model and model settings, send messages through supported local/provider paths, and operate message actions with visible recovery when unavailable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Actual rendered CDP walkthrough verifies Console can start/create chat sessions and keep tab controls usable.
- [x] #2 Actual rendered CDP walkthrough verifies provider, model, and model-setting changes are visible, persist through the relevant Settings/Console path, and do not contradict Console readiness.
- [x] #3 Actual rendered CDP walkthrough verifies a basic send flow against a local provider path or clear blocked-state recovery when the provider is unavailable.
- [x] #4 Actual rendered CDP walkthrough verifies chat messages render with usable selection and message actions such as Copy, Edit, Save as, regenerate, continue, feedback, and delete or honest unavailable states.
- [x] #5 Keyboard-only operation is verified for tab traversal, composer focus, Enter activation where applicable, and selected-message action access.
- [x] #6 Every confirmed blocker is covered by a failing regression before production code changes.
- [x] #7 Final actual CDP screenshot evidence is captured and approved before PR creation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: this is an actual-use Console UAT and corrective interaction-hardening slice within existing Console, provider, and message-action boundaries; it does not introduce a new storage/schema, runtime boundary, provider contract, sync policy, or long-lived architecture decision.

1. Establish a clean latest-dev Console baseline from the isolated worktree and run focused Console/provider/message-action tests.
2. Launch the actual app through textual-web/CDP with an isolated HOME/XDG profile and capture the baseline Console screenshot.
3. Walk the user acceptance flows: new chat/tab controls, provider/model/settings visibility, local-provider send or blocked recovery, message selection/actions, and keyboard-only traversal.
4. For every confirmed blocker, record reproduction evidence and root cause before editing production code.
5. Add a failing regression for the first blocker, implement the smallest safe fix, rerun focused verification, and repeat for additional P0/P1 blockers in this slice.
6. Capture final actual CDP screenshots and wait for explicit user approval before PR creation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Current UAT slice has CDP evidence for the local llama.cpp path at `127.0.0.1:9099`: Console loads with the `llama_cpp` provider and discovered model visible, the Console Settings modal now renders the selected provider, long model name, inherited Base URL, sampling fields, and context rows readably, keyboard provider selection changes the provider and dependent endpoint/model fields, and the session can send a real prompt that renders `SERVER OK` with `Run: Response complete.`

Regressions added before fixes:
- `test_console_native_tab_strip_isolates_composer_drafts` covers per-tab composer draft isolation.
- `test_continue_from_assistant_message_ends_provider_payload_with_user_instruction` covers provider-compatible Continue payloads.
- Console message action label tests cover `Save as...`, `--->`, `Del`, and variant navigation labels.
- `test_console_settings_modal_shows_inherited_provider_endpoint` covers showing inherited saved endpoints in the modal.
- `test_console_settings_modal_select_current_preserves_visible_value_row` covers non-clipping selected provider/model rows in Console Settings select controls.
- `test_model_options_ignore_none_sentinel_values`, `test_model_options_preserve_current_model_even_when_registry_has_none_sentinel`, and `test_console_settings_modal_provider_round_trip_ignores_none_model_sentinel` cover the confirmed provider round-trip bug where `"None"` placeholder models could become visible/selected after switching away from and back to `llama_cpp`.

Implementation changes so far:
- Native Console session drafts are saved and restored per tab.
- Continue now appends a user continuation instruction so providers such as llama.cpp do not receive an assistant-final payload.
- Message action labels now match the approved Console semantics.
- Console Settings modal no longer stores missing provider/model/base URL as explicit empty drafts, allowing saved provider defaults to display.
- Console Settings select-current rows are styled explicitly in source TCSS and regenerated into `tldw_cli_modular.tcss`.
- Console model option generation now filters placeholder model sentinels such as `"None"`/`"null"` before the settings modal can select or persist them.

Verification run: `python -m pytest -q Tests/UI/test_console_session_settings.py Tests/UI/test_non_obscuring_focus_contract.py::test_console_settings_modal_controls_use_compact_focus_outline Tests/UI/test_non_obscuring_focus_contract.py::test_console_settings_modal_select_current_preserves_visible_value_row Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_workspace_context_rail.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_message_actions.py Tests/Chat/test_console_session_settings.py --tb=short` exited with `215 passed, 8 warnings`.

Additional provider/settings contract verification: `python -m pytest -q Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_console_provider_support.py --tb=short` exited with `59 passed, 1 warning`. `git diff --check` exited 0.

Follow-up CDP evidence on the fixed-code textual-web instance at `127.0.0.1:8878`: Console Settings opened with `Provider llama_cpp`, the full gemma model, Base URL `http://127.0.0.1:9099`, and sampling fields visible. Switching provider to `koboldcpp` showed no model and the kobold endpoint, then switching back to `llama_cpp` restored the gemma model instead of selecting `None`. Saving preserved the left rail `llama_cpp`/gemma summary. A real llama.cpp send with `Reply exactly: FIXED OK` rendered the assistant response `FIXED OK` and top status `Run: Response complete.`

Additional CDP evidence on the patched textual-web instance at `127.0.0.1:8879`: keyboard traversal reached the transcript from the Console shell, selected the assistant message with arrow keys, revealed selected-message actions with Enter, moved focus into `Copy` with Tab, and Enter activation displayed `Copied message to clipboard.` The live run used the local llama.cpp endpoint and rendered `KEYBOARD OK` with `Run: Response complete.` before action verification.

Fresh local LLM server CDP evidence: after confirming `curl http://127.0.0.1:9099/v1/models` returned the active llama.cpp model, a new prompt `Reply exactly: SERVER AVAILABLE` was sent through the rendered Console. The actual browser-captured Console screenshot shows the user prompt, the assistant response `SERVER AVAILABLE`, and top status `Run: Response complete.`

Corrected relaunch evidence: the first relaunched textual-web instance loaded `HOME/.config/tldw_cli/config.toml`, which still contained the older `local-model` test fixture. Root cause was launch-profile ambiguity, not a Console model regression. Relaunching with explicit `TLDW_CONFIG_PATH=/private/tmp/tldw-chatbook-console-uat/config/tldw_cli/config.toml` on `127.0.0.1:8935` loaded the intended gemma model. The actual browser-captured Console screenshots show `Provider: llama_cpp`, `Model: gemma-4-26B-A4B-...`, the full Console Settings modal with provider/model/base URL/sampling fields, the user prompt `Reply exactly: FULL SCREEN OK`, the assistant response `FULL SCREEN OK`, top status `Run: Response complete.`, and selected assistant message actions `Copy`, `Edit`, `Save as...`, `Regen`, `--->`, `Good`, `Bad`, and `Del`. User approved this screenshot set before PR creation.

Additional regression added before the keyboard fix:
- `test_console_selected_message_copy_action_works_from_keyboard` reproduced that focus could reach the selected-message `Copy` action, but Enter did not activate the action because transcript/message action buttons did not own Enter activation. The fix introduces an explicit Console transcript action button that presses on Enter.

Workspace context check: the Console left rail already exposes a `Convos & Workspaces` section, a local registry status, active workspace label, conversation rows, and a `Change workspace` modal when two or more local workspaces exist. In the isolated CDP profile there are no local workspace records, so the rendered `Workspace switching: locked` state is expected and matches existing empty-state tests. Syncing/server-backed workspace switching is not implemented in this UAT slice and is not claimed as verified. The follow-up should remain a server-readiness/contracts task until `tldw_server` exposes the required workspace and ACP handoff APIs.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Console UAT basic chat is verified against the local llama.cpp path and has regressions for the confirmed blockers. The slice hardens per-tab drafts, provider-compatible Continue payloads, approved message action labels, Console Settings visibility, placeholder model filtering, and keyboard activation for selected-message actions. Actual CDP/browser screenshots were captured at a wide viewport and approved before PR creation.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] Acceptance criteria checked.
- [x] Implementation plan documented.
- [x] Regression and focused verification commands run.
- [x] Actual rendered screenshot evidence captured and approved.
- [x] ADR check completed; no ADR required for this corrective UAT slice.
<!-- DOD:END -->
