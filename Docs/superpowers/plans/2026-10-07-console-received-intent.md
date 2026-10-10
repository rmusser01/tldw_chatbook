# Early Console received intent implementation plan

Goal: give Send one app-owned lifetime and immediate Preparing feedback before hook/configuration I/O, then promote exact inputs through the existing acceptance path.

ADR required: yes, existing ADR-225 applies.
ADR path: backlog/decisions/225-console-send-preparation-and-io-ownership.md.
Reason: direct implementation of the approved receipt, input revision, promotion and screen-free review contract.

Task: TASK-34563.15. Spec: Docs/superpowers/specs/2026-10-06-console-send-preparation-architecture-design.md.

## Constraints

Candidate worktree only, no primary changes, push or merge. Existing three lanes; root owns integration and every sequential native run. Targeted checks only. Preserve original deadlines, source qualification, actual native retirement, WAL/NORMAL, saved failure keeps draft, temporary chats, all consent/effect/trace/history ordering, origin authority and custom callback signatures. Do not add a scheduler, permission cache, admission map or generic pipeline.

## Ownership

- Shared preparation: new Chat/console_received_intent.py with detached input models and pure checks; session input revision fields/methods in Chat/console_chat_store.py; narrow received-turn, preparation and configuration-default metadata; Tests/Chat/test_console_received_intent_{inputs,transitions}.py.
- Controller/provider integration: existing console_runtime.py and console_chat_controller.py integration; narrow new Chat/console_received_dispatch.py if needed to avoid growing monoliths; UI/Console_Modules/{wiring,prompt_queue,session}.py, UI/Screens/chat_screen.py and Widgets/Console/console_composer_bar.py; Tests/Chat/test_console_received_intent_custody.py. Coordinate the synchronous composer-to-store revision adapter with shared lane before edits.
- Baseline verification: Tests/UI/test_console_received_intent_feedback.py only; source review and exact affected-test selection. No native execution.
- Root: this plan, task/report/evidence, integrated review/runs and any explicitly negotiated remaining finite producer ownership fixes. No overlapping edits.

## Agreed boundaries

1. Extend the existing session with authored draft/input revision, updated synchronously on composer mutation (including clear/retype) and domain draft mutations. Keep view attachment/generation in the UI adapter; no widget, screen callback or stash crosses runtime handoff.
2. Detached ConsoleReceivedTurnIntent carries turn/session identity, exact draft and input revision witness, selected in-memory ConsoleTurnCaptureSelection, attachment IDs/identities, one-shot prefill/revision and staged evidence/revision. Selection uses existing pure selectors; no native reads, credentials or complete configuration before receipt.
3. runtime.accept_received_intent synchronously registers existing custody before a lazy retained driver. Normal input uses the existing claim_received_turn slot; request remains None until full checked configuration succeeds. No release/reacquire or recursive accept_turn. Scheduling failure leaves all staged inputs untouched.
4. Runtime driver yields before slow setup, reads/rechecks initial hook snapshot with the existing physical preparation-read owner, and uses request_initial_hook_review / resident decision host. It retains no screen continuation. Then perform screen-free configuration from captured selected values, check exact input/source revisions, transfer attachments once and construct the complete existing request in the same custody record. Existing submit promotes the same admission claim. Pre-promotion edits refuse the old attempt; after promotion a new draft is independent.
5. Queue intake uses existing queue expected-revision admission while an accepted run occupies the normal slot. Runtime custody owns its capture/review lifetime; queue state remains in the registry. Original queue_prompt decides admission. A REROUTE_NORMAL_SEND result rechecks/acquires the normal claim once. Existing queued execution retains coordinator authorization; no fake manual claim or second queue map.
6. Domain acceptance follows existing origin semantics. Clear the captured draft only by matching authored revision and attached view generation at actual publication; equal text is insufficient. Failed save/Stop/refusal preserve or expose exact recovery and never replace a newer draft. Existing complete-request prefix-transfer behavior remains unchanged.
7. Early receipt covers ready and approval-required paths, Enter/button/spoken sends, queue intake and navigation. Custom callbacks retain explicit existing routes and signatures. Keep slash/raw-console dispatch semantics unchanged.
8. Use the existing finite preparation-read registry for any required original native callback newly placed behind receipt. Do not drain arbitrary provider/custom async waits. Physically held original producers, repeated cancellation and creator disposal are the proof; task cancellation or process-tree emptiness alone is insufficient.

## Execution and verification

- [x] Fresh unchanged d76319 baseline: existing matched whole-Send harness passed; 10.844108, 10.094182 and 8.902770 seconds to adapter. Source unchanged; contained tree and private profile retired normally, zero lookup races/forced retirement. This is action-to-adapter, not physical terminal feedback qualification.
- [x] Review precise interface refinements between shared and integration lanes; record them below without changing approved authority boundaries.
- [x] Root observed meaningful RED before product changes: actual Enter reached held original native configuration without a received claim. Four additional controls failed for the missing authored revision/intake/native-runner APIs. Original SQLite body and lease retired after release; 5 expected failures, pytest 18.27s, bounded harness 23.141s, zero forced retirement/overflow/lookup races. Evidence: .superpowers/sdd/2026-10-07-console-received-intent/baseline-and-red.json.
- [x] Implement both lanes, review exact callback/source/promotion/draft/queue semantics, then freeze relevant source.
- [x] Root runs final new + affected controls after both implementations are ready, sequentially with the existing bounded runner. Preserve initial failures and investigate the actual failing boundary.
- [x] Measure actual Enter/button event-to-natural Preparing frame and input response with original native operation held; no forced paint or offscreen observer. Target <=100ms but report real limits honestly. Repeat unchanged whole-Send harness after adapting only its accepted-lifetime observation for the new intake API; include all app setup and report raw samples.
- [ ] Review source/static differences without raising caps; update task/report and commit bounded qualified work. Coordinate CLOSED native window with combined UAT; do not call this the overall latency fix unless whole-Send and feedback targets are demonstrated.

Focused original controls are the received admission/custody/saved-turn, exact attachment/prefill/evidence custody, identical-retype draft and navigation, resident hook-review and configuration-worker controls selected by baseline lane. Full test suite remains excluded.

## Reviewed interface refinements

Shared input API: ConsoleSessionInputSnapshot and ConsoleReceivedTurnIntent; store.session_input_snapshot, session_inputs_are_current, commit_session_input_draft. Existing set_session_draft gains optional authored_token=(composer generation, edit serial), published synchronously on mutations. Session settings/identity/workspace revisions and exact source witnesses are captured at receipt. CAS clear validates draft/source only after promotion, so later independent settings edits do not suppress rightful acceptance. Pending agent-handoff persistence must not be lost when the queued mirror sees already-mirrored text.

Integration owns Chat/console_received_dispatch.py as the narrow driver helper. Controller capture accepts selection=None for the detached DTO; existing mounted/custom entry points retain original behavior. Early UI adapter cannot use the existing selector blindly: its stock provider/config helpers can perform I/O. Read already resident settings, app config, RAG and workspace values with pure builders; qualify supported stock callbacks before using that route. Cold/custom routes remain explicitly compatible rather than silently replacing callbacks.

Root owns the two original finite-native leaf seams in console_provider_gateway.py and console_library_policy_coordinator.py plus Tests/Chat/test_console_received_native_preparation.py. Optional private _run_native takes one zero-argument original callback. Controller supplies its existing retained reader only for defining-module-qualified stock methods; custom callbacks keep the original call shape. Reference expansion uses the same physical-read owner in integration lane. These adaptations change physical lifetime, not authority or effect ordering.

Native coordination: fresh baseline and meaningful RED completed sequentially and the window was returned CLOSED. Combined UAT currently owns the next native window. Source/test authoring continues; integrated green checks await both implementation lanes and the returned window. No overlapping native run is authorized.

Root also owns the narrow whole-Send harness observer adaptation: append custody tasks from accept_received_intent as well as legacy accept_turn. Preserve action timestamp, three real storage/trace turns, two non-streaming plus one streaming adapter mode, heartbeat, original deadlines and cleanup. Its pre-existing gateway wrapper means private stock-only provider injection is not selected in this diagnostic; disclose this limit. No receipt-driver recursive accept_turn is introduced.

## Observed disposal boundary

The isolated actual Enter control positively retired its held original configuration reader, but the strict factory still observed an active WorkspaceDB operation after runtime.dispose. Passive worker-frame identity inspection identified the exact original ConsoleWorkspaceController._read_workspace_files_availability callback and matched its database to the retained participant. Its screen-owned worker drains an inner read on cancellation, but runtime disposal does not join that owner. Task AC5 records the necessary finite-owner correction before implementation. Integration lane owns the narrow workspace/runtime production wiring; baseline lane owns the held-original disposal regression and existing cancellation/borrowed controls. No global run_owned_db_call behavior change, deadline increase, teardown weakening or source-policy relaxation is authorized.

Five transition controls now pass (received-intent-transitions-fourth, 23.82s pytest, 27.750s contained driver), including distinct and identical new drafts after the complete request. The promotion assertion observes the original successful return before terminal retirement removes its registry entry.

The held-original availability shutdown control now reproduces the failure at its ownership assertion (received-workspace-owner-red): runtime disposal returns cancelled while the exact original SQLite operation, connection and lease are still live; releasing the original reader retires it. The fix reuses ConsolePreparationRead for the same finite database scopes and observes existing creator reads on runtime attachment. The low-level run_owned_db_call API remains unchanged; its former caller-specific test observer will follow the new original finite callback while retaining SQL/lease/borrowed/cancellation assertions.

Final source review identified a custom async queue callback that could bypass the private pre-admission input guard. Shared lane owns the narrow wiring/received-dispatch refinement: unsupported queue callbacks retain the legacy route; replacing a stock queue callback after early receipt refuses before invocation. This preserves the existing queue authority and supported callback signature without adding another admission mechanism.


## Integrated result

Both implementations and root ownership fixes are integrated. See Docs/Development/2026-10-07-console-received-intent-verification.md for exact final selections, preserved failures, native cleanup and limits. 68 domain controls, 9 availability/cleanup controls and all 11 selected UI behaviors pass across the documented final and corrected runs. The custom launch pair now respects a replaced sync callback; saved acceptance publishes the draft only for the current uncancelled original owner. Actual Preparing frames are 19–23 ms, but full Send remains 10.647/6.821/6.049 seconds. Next work must attribute and simplify the remaining postcommit interval; this checkpoint does not close the overall performance goal. Static baseline-relative findings are unchanged; inherited module-size/static debt keeps the task In Progress.
