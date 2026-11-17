# Task62 original parent entry points

Source-only audit for the central integrator. No product/test edits, execution, qualification, production selection or timing claim. Current performance-fix qualification remains the gate. This is a caller map for the already accepted whole-service design, not a replacement design or complete feature audit.

Authority: `task-62-process-integration-plan.md` steps 3–5 and canonical `Docs/superpowers/specs/2026-11-15-task-40608-console-execution-service-design.md` sections 2–4. Select once before the stock singleton/session graph, retain the original lifetime local route for unsupported profiles, and use an explicitly named parent client plus read-only view. Never install the client as a runtime-shaped proxy or pass the read-only view to mutation-bearing store consumers.

## App lifetime and first attachment

| Original entry point | Smallest required integration seam / existing dependency |
| --- | --- |
| `app.py:1215`, `TldwCli.__init__` | Original app-owned holder constructs `ConsoleRuntime(self)` unconditionally. This is the lifetime selection/holder point. The selected branch must retain one actual `ConsoleExecutionClient`; the legacy branch keeps its original runtime. Configuration is loaded at 1172; later profile services must retain their ordering. |
| `app.py:3309`, `on_mount` | Existing loop-owned startup calls `runtime.start_async_lifecycles()` at 3314. The client already has retained `start(ConsoleProcessProfile)` at `console_execution_process.py:253`; preserve startup custody and join readiness for a first Send. Do not start a second graph or invent early prewarming. |
| `app_service_wiring.py:2197`, `_wire_workspace_registry_services` | Direct unconditional `console_runtime.bind_change_review_consent_service(...)` must retain the selected ownership boundary. This occurs before profile Chat DB wiring (`app.py:1561/1574`). Parent Library/Settings repositories remain real parent sources; they are not moved wholesale. `_wire_chat_conversation_services` publishes local/scope services at 2125/2145 and calls local runtime attention refresh at 2131–2138. |
| `chat_screen.py:7420`, `build_console_controllers` | Original UI-controller composition stores late-bound callbacks; `wiring.py:1472–1475` immediately demonstrates callbacks to the authoritative store/controller accessors. Preserve actual UI owners while selecting explicit read-only consumers and named commands. Do not make all these callbacks accept a pretend store. |
| `chat_screen.py:14913/14931`, `_compose` / `compose_content` | Attachment cannot first occur in `on_mount`. Compose calls `prepare_initial_console_compose`; its current selected test is local `_execution_service._local_controls` (`console_parent_settings.py:247–252`). Otherwise the ordinary prefix runs. `_consume_pending_console_launch` at 7725 reads `_pending_console_launch_context` even with no handoff, which invokes `_console_runtime()` through the property at 9218. This is already enough to allocate a parent graph. Control/staged/inspector/workspace/rail derivations follow before mounting. |
| `chat_screen.py:9192`, `_console_runtime`; `console_runtime.py:6814`, `ensure_console_runtime` | Exact fallback hazard: absent/non-`ConsoleRuntime` app owner causes `ConsoleRuntime(app)` at 6843 and `attach_view(view)` at 6851. The selected path must resolve its explicit client before entering these legacy accessors. The properties at 9238/9247/9256 and `_ensure_console_chat_store` / `_ensure_console_chat_controller` at 9271/9435 all depend on this runtime. |
| `chat_screen.py:15548/15585/15614`, mount / current visit / attach reconciliation | Current visit is a `ConsoleAttachVisit` containing the local runtime and attachment generation. Reconciliation calls full sync and native workspace/runtime reconciliation. The client currently initializes `_view=None`, `_visit=0`; it has no public attach/detach operation or production assignment to those fields. Central integration must bind the actual screen and original visit at the natural attachment boundary, with superseded/torn-down publication refusal. |
| `chat_screen.py:16034`, `on_unmount`; `app_lifecycle.py:578`, `_shutdown_console_runtime` | Unmount obtains local runtime and detaches synchronously at 16047 before awaited cleanup. App exit captures/drains original view workers, then disposes at 612. Preserve these distinct visit/lifetime responsibilities using actual client attachment retirement and retained `shutdown(deadline=...)` (`console_execution_process.py:601`). `dispose_console_runtime` intentionally ignores foreign owners; it does not shut down a client. |

## ReadProjection and original UI publication

The session-view publication is now **in live source**, not merely a pending proposal: `ConsoleLocalControls._read_projection` (`console_control_dispatch.py:436`) performs the original due read and captures the frame under the preparation lock; backend `capture_projection_view` (`console_backend_presentation.py:33`) captures original rows/fingerprints/current inputs at that boundary. `ConsoleExecutionClient.read_projection` (`console_execution_process.py:206`) waits for Applied and positive Retired, checks source/channel/view/visit/request sequence, then installs `_projection`, `projection_view` and `pressed_inputs`. No additional local authoritative store is required for these five named view reads.

Remaining concrete publication gaps:

- `ConsoleExecutionClient._project_event` at 563 retains events but has no UI consumer. No client references exist in the inspected App, lifecycle, service-wiring, ChatScreen or Console-wiring files. The reviewed `UI/Console_Modules/execution_projection.py` remains absent live. Its proposed exact `apply_console_process_projection(screen, client, projection, *, visit)` invokes the actual mounted transcript's original methods after currentness checks; central must bind/call it. Its True is synchronous invocation, not paint/retirement.
- `ChatScreen._sync_native_console_chat_ui` at 17888 is the existing whole refresh entry; `_sync_native_console_transcript` at 17247 is the transcript entry. Both obtain local store/controller/domain state. Selected ReadProjection publication must branch before those reads. Do not call the whole legacy sync after rendering a process frame.
- `_sync_console_native_session_tabs` at 18229 reads rows at 18257 but also performs missing-settings preparation, re-ensures the store, inspects controller streaming state and handles workflow/fleet effects. The read-only view can supply its declared rows/fingerprints; unchanged execution of this entire method is not read-only.
- `ConsoleSessionController._ensure_active_console_session_settings` at `session.py:4650` can converge/create settings; `_sync_console_session_draft` at 5769 ensures a session, saves old drafts and loads current drafts. Initial Compose, first session, tab selection and composer refresh need their original named command/checkpoint ownership. A transcript frame alone cannot replace these operations or justify clearing a draft.
- `ConsoleMessageController._native_console_messages` (`message.py:994`) calls the authoritative ensure accessor. The reviewed adapter avoids that call. Title/readiness/control/rail/cost/inspector/activity and decision-card values are not all present in the seven-field frame or five-field read-only view; do not substitute empty/default UI values or claim complete screen rendering.
- `ConsoleBackendPresentation.parent_attachment` remains None. `bind_view_hooks`, `prepare_default_convergence`, `archive_send_inflight` and `present_decision` at 190/205/211/223 explicitly lack actual parent checkpoints. `notice` and `present_pending_projection` return False rather than asserting parent adoption. Keep those distinctions while connecting original hooks and approval/decision surfaces.

## Exact pressed Send route and remaining handoff

`ChatScreen` callers at 4832, 22587 and 24933 converge on `send_acknowledgement.request_visible_send:536`. It first checks unbound composer and retained settings capture, preserves raw-CLI arm/guard behavior, and captures the actual stash before paint. `_schedule_visible_send_capture:587` → `_schedule:784` retains `_ConsolePendingSend` → `_send_console_message_from_visible_action:18608` / `_observed:18627` consumes that exact token/stash/inputs → command classification → `_dispatch_console_draft_send` (`composer_run_controls.py:1039`) → `wiring.receive_console_visible_intent:346` → original local `runtime.accept_received_intent:648`.

The minimal selected handoff must retain that actual press and classification before replacing the local authority call with the existing fixed `UpdateInputs` / `ReceiveTurn` commands. Current obstacles are explicit:

1. `console_parent_settings.retain_parent_visible_send:1343` and `prepare_parent_visible_send_target:1456` select local controls and store/port sources. They do not select the process client.
2. `capture_visible_send_inputs:756` accepts only exact `ConsoleChatStore`; `_record_draft:771` calls `set_session_draft`. Authored edits use `composer_run_controls._publish_console_authored_draft:924`, directly write the store and publish its actual revision. Preserve exact authored token plus ordered UpdateInputs result; use captured `client.pressed_inputs` and row revisions, never inferred backend increments or a later composer recapture.
3. `_dispatch_console_draft_send:1092` enters `receive_console_visible_intent`, which requires local store/controller and the original source/policy/recovery checks. Its worker path reads local `_turn_custody` at 1108; a declined receive can fall into prompt-queue dispatch. The selected branch must consume its real receipt/events and refuse unavailable selected operations rather than fall back into this local graph.
4. `send_acknowledgement._mount_row:1059` reads local active session and `_message._native_console_messages`, applies change-review/ack projections, sets the real transcript and awaits refresh before dispatch. Preserve Sending/user-row paint sequencing using current real process presentation values; the narrow transcript adapter does not itself implement this acknowledgement overlay.
5. `ConsoleExecutionClient.submit:372` already retains fixed receipts and marks a ReceiveTurn potentially committed before transmitting a validated exact `CommitPlanAck` at 423–445. `_project_event` still lacks the original parent plan retention/ACK/custody, pending-decision, terminal and accepted-draft-clearing consumers. Received, Applied, potentially committed and positively Retired remain separate; no optimistic draft clearing or acceptance inferred from paint.

Lifecycle/safety closure is part of this same integration, not permission to enable after transcript-only success: `_confirm_console_runtime_quit` (`app_lifecycle.py:2034`) currently returns True when a local controller is absent; `_console_hooks` (`Backup_Recovery/runtime_maintenance.py:114`) accepts only original local runtime/controller. The source-fence, complete Settings/control-plane ownership and existing loss/maintenance contracts in plan step 5 still apply. `app.py:5322` also uses local custody to postpone background imports. This map does not audit or replace the wider Settings surface.

## Source pins

SHA256 of the files read for this map (repository-relative paths). Product bytes were not written. Reconcile these pins before implementation if central source has advanced.

```text
a5bb093a5476fc257e1bdf11bfe72597efdddd6a34e58f77c5ddac29711acd81 tldw_chatbook/app.py
076490bad73a56c88f5012964a9251b6c84c9e83257c3065e88bf006ab64cac2 tldw_chatbook/app_lifecycle.py
9152b7757cc11265db73d01eda633c8c56cb1c26721e9ffeba943b9e275ed1f6 tldw_chatbook/app_service_wiring.py
372dc2e818e39e7b4e010468b148206b60163aa19f7948bd5e0de7e9438db8ec tldw_chatbook/UI/Screens/chat_screen.py
3ab58656738e30fa681b8fd8552907078ed97c0f27c9bdf0a70ac7780eb8c5b9 tldw_chatbook/UI/Console_Modules/wiring.py
b3c4a7e45f1c0cdddfae0cb5d036e30dc37c407adb645bf7cc4448097879f839 tldw_chatbook/UI/Console_Modules/send_acknowledgement.py
51cd899596274204d613a23a42aa2eed57b5f25f6b1f4476c77ac64d23c59d6a tldw_chatbook/UI/Console_Modules/composer_run_controls.py
31fb07fa49f1018e0bf7b7217cd85d349d88d04db010f650782099d5dbb787f4 tldw_chatbook/UI/Console_Modules/session.py
9ae517387179848d82f69a164683ec788b1eeb43389018f3c6215f8922c79b5b tldw_chatbook/UI/Console_Modules/message.py
3da077c4d01d6e8d2582a10d3a64cffd8a3bcbb25ad6dc6fb1b450f9686d258e tldw_chatbook/Chat/console_runtime.py
4a948794b731e39511e62e0993638e2ce51280d8de17464517d1b5d3a16e536e tldw_chatbook/Chat/console_parent_settings.py
f01912e11a843fd08b77ff773f8b7d419e4b83914c7850341061b94350c26a1a tldw_chatbook/Chat/console_execution_process.py
2262de0bc83e59459cf5982c986d2fa0ea820a008f0d6738fd6d814a20876539 tldw_chatbook/Chat/console_control_dispatch.py
eead85ac83346a0b48a21590cb2724a304546e29f4f1eac9c0f6ec5026a4eced tldw_chatbook/Chat/console_backend_presentation.py
bef35ba1a203cd91bee937eeb3b7de5e0f8aaf09c16f654098e394fadeb08b07 tldw_chatbook/Chat/console_execution_projection_view.py
67206f0a2c3cd9ccd2a6c6c9eda69211ea033ff2036ac9a499811ecdc8dadfae tldw_chatbook/Chat/console_execution_projection_result.py
1016c9c4cef0093529bfcdd54656ca6bed7ef5e4052f2ce63ec39ae0fc91d93e tldw_chatbook/Chat/console_execution_protocol.py
0b24e51b9b83333f1fc608c80c1813290fcfd55fc31fac50a04ecdd590730686 tldw_chatbook/Backup_Recovery/runtime_maintenance.py
```

Audit stopped after this entry-point map. No new implementation/test matrix is proposed here.
