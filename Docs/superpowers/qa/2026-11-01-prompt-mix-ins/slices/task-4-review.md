### Spec Compliance

- ❌ Issues found: uncustodied direct Queue and manual Submit do not retain the PromptScope/backend/DB source fence across the hook await after mix-in capture. This misses TASK-38539 AC #3 and the binding requirement that source changes invalidate pending preparation/handoffs (`tldw_chatbook/Chat/console_chat_controller.py:6257`, `:6280`, `:9372`).
- ✅ The complete-supplied-configuration ruling is respected: supplied configuration bypasses new capture (`tldw_chatbook/Chat/console_chat_controller.py:6249`, `:9364`); the finding below concerns only the ordinary uncustodied paths.
- ⚠️ Full unchanged runtime/turn-preparation copy behavior and eventual durable dispatch integration cannot be established from this task diff alone. Detached configuration copies carry the field (`tldw_chatbook/Chat/console_turn_context.py:473`), and Task 5 owns composition/attempt recording. No extra implementation is requested for deferred features.

### Strengths

- Received admission freezes preferences before deferred work, carries exact source witnesses, and rechecks them through preparation (`tldw_chatbook/Chat/console_received_dispatch.py:132`, `:407`; `tldw_chatbook/Chat/console_received_intent.py:235`). Held-source and held-native-worker tests exercise the actual received owner (`Tests/Chat/test_console_received_intent_transitions.py:439`).
- Queue text edits explicitly preserve the accepted snapshot, and machine follow-ups get empty snapshots (`tldw_chatbook/Chat/console_prompt_queue.py:843`; `tldw_chatbook/Chat/console_prompt_queue_coordinator.py:452`, `:848`). The edit/toggle/delete/drain regression checks no second library read (`Tests/Chat/test_console_prompt_mixin_admission.py:511`).
- Preference saves join the existing settings drain with exact identity fences and coalescing (`tldw_chatbook/Chat/console_chat_store.py:8860`, `:9448`). The recovery row stays hidden when healthy and retains exact failure identity (`tldw_chatbook/UI/Console_Modules/left_rail.py:791`; `tldw_chatbook/UI/Screens/chat_screen.py:5898`).
- Durable retry inspects raw strict marker JSON and verifies the assistant anchor before repository replay, distinguishing recorded empty, legacy, and unavailable (`tldw_chatbook/Chat/chat_persistence_service.py:1248`). Body-bearing snapshot fields remain private/repr-hidden (`tldw_chatbook/Chat/console_turn_context.py:243`; `tldw_chatbook/Chat/console_chat_store.py:1452`).

### Issues

#### Critical (Must Fix)

None found in this task-scoped review.

#### Important (Should Fix)

1. **Retain and enforce the source witness until uncustodied direct admission.** `tldw_chatbook/Chat/console_chat_controller.py:6257` awaits hook admission after `capture()` has completed; the subsequent guard at `:6280` checks only the preference revision. Likewise, direct manual submission awaits its hook at `:9372`, checks only preference revision at `:9373`, and later transfers a tuple containing only the snapshot and session witness (`:10020`). `ConsolePromptMixinService.capture()` checks the source only while that coroutine runs, so replacing the PromptScope/local service/DB or retargeting its path during the subsequent await admits a snapshot from the stale source. This violates the explicit pre-acceptance source invalidation contract; the frozen-custody exception begins only after acceptance. Preserve a private exact source witness/currentness callback along with the captured value and validate it after subsequent awaited preparation and immediately before admission. Keep already-complete supplied configuration authoritative. Add barrier regressions for direct Queue and manual Submit; the received-path guard illustrates the required behavior.

   **Focused reproduction:** instantiated the real `ConsoleChatController` and store with the existing test `Scope`; enabled A; left `configuration=None`; let capture finish; replaced `scope.database` in `hook_admission_reason`; called real `queue_prompt`. It returned `QueueMutationStatus.APPLIED`, `applied=True`. The probe asserted that erroneous acceptance and exited 0. It used a minimal resolved provider configuration and the existing `SequencedGateway`; no library read or stale-source check was mocked away. This is source replacement before queue custody, not a later library edit of an accepted entry.

#### Minor (Nice to Have)

- The implementer's report records aggregate FD growth of 268 in a mixed storage run (`task-4-report.md`, “Concerns and excluded checks”). Prior Task 3 also reported this class of warning; no evidence attributes it to Task 4. Keep it qualified as inherited/unattributed test noise, not a request for blanket descriptor cleanup or weakened thresholds. The retained final focused output has no warning summary (`task-4-evidence/final-proof.log:5`).

### Checks and scope

- Read the supplied diff once in sequential chunks. The initial tool output truncated a small middle section of the new admission tests; read only the omitted section afterward. No git diff regeneration or HEAD/index/product mutation.
- Completed cut-off controller context for the named risk of source changes between capture and direct admission: `queue_prompt` around lines 6227–6324 and `_submit_draft_lifecycle` around lines 9323–9428. Read the constructor signature/app wiring only to assemble the focused reproduction.
- Named risk: partial in-memory replay markers being mistaken for legacy. A focused check of unchanged `Chat/message_metadata.py:464` confirmed that typed metadata validates paired attempt/schema fields; no finding on that typed path.
- The first reproduction invocation stopped at `recovery_scope_uncertain` before exercising product behavior; using the existing test bootstrap preserved the normal profile guard. A harness construction typo (`app` is assigned after construction, not a constructor argument) was corrected. The final same-risk probe passed and demonstrated stale acceptance. No recovery guard was disabled.
- Read retained `task-4-evidence/final-proof.log`: 34 passed in 9.76s, no warning summary. Read the differential lint evidence; visible touched-file entries contain no introduced diagnostics. Did not rerun the implementer's suites or broaden validation.

### Assessment

**Task quality: Needs fixes.**

Received ownership, private snapshot custody, and settings integration are coherent, but the unguarded direct-path interval permits stale-source admission under an ordinary async race. Fix that fence and qualify both affected entry points before approving Task 4.

Skipped: full suite, live-provider/UI verification, deferred Tasks 5/7/8 behavior, and broad unchanged-code review. Risk: direct Queue/manual Submit can currently accept a source that changed during preparation.
