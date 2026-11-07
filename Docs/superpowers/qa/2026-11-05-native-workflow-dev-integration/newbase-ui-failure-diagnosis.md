# New-base UI timeout diagnosis

## Proven cause of this timeout

The test never reaches the cadence scenario. A real visible Send is refused before provider validation; `_held_run` then waits for an event which that refused send cannot set. **The individual refusal reason is not present in the saved log, so its underlying fixture/product cause is not yet established. No production or fixture behavioral correction is justified from this log alone.** In particular, this is not evidence of an eager-coroutine fake-worker mismatch.

Evidence:

- `newbase-ui-intersections.log:4`–`:11`: failure is at `test_console_projection_reads_off_loop.py:262` → `test_console_config_sync_cadence.py:306` → `until(gateway.validation_started.is_set, timeout=30)`. Every coalescing/freshness assertion comes later and was unexecuted.
- Log `:396`–`:399`: one actual `ui_action` enters, one `ui_dispatch` enters, and the dispatch reports **refused in133ms**, followed by **not_dispatched**. The remaining wait is not slow provider validation or a measured UI projection stall. The receipt is a stable9109-file source selection, before/after digest `69dbbe0cf762c493cc35449ce2095c73415e41c24885ad9ac2b945b51d78ad72`, with `changed_during_command=[]`. It reports1F/3P and stops at this first failure.
- `_held_run` (`Tests/UI/test_console_config_sync_cadence.py:302`–`:308`) calls the ordinary mounted harness, sends Enter through the real terminal driver, and waits on the gateway. It does not install a `run_worker` fake. `ConsoleHarness` uses real Textual worker machinery. `eager_tasks()` changes the loop task factory to match the shipping App; it does not unwrap or filter worker payloads.
- `HeldGateway.resolve_for_send` (`test_console_send_acknowledgement.py:174`–`:179`) marks validation and sets the event **before** waiting and before invoking `_ReadyResolutionGateway.resolve_for_send`. Therefore the latter’s older SimpleNamespace response could be a later contract problem, but cannot explain this event never being set. Similarly, current cadence paint/readiness changes are not established as the cause of a pre-provider refusal.

## Exact dispatch chain and missing observation

`ChatScreen._send_console_message_from_visible_action_observed` reaches resident `composer_run_controls._dispatch_console_draft_send`. That function stops the draft-spend refresh, tries `wiring.receive_console_visible_intent`, and otherwise awaits the existing hook/queue dispatcher. The observed final string `refused` is assigned from `result.status.value` at `composer_run_controls.py:1120`, after `_hooks.dispatch` returns. An acquired received-intent path instead marks `awaiting_review`; an uncaught error would produce a failed diagnostic. This narrows the observed route to the ordinary fallback dispatcher returning a refusal, not a silently dropped Textual refresh callable.

The exact result detail is discarded by this thin UI delegate after `return result.accepted`, and `_held_run` neither retains that result nor checks the warning notification. The log records the body-free diagnostic status, not the detail. Relevant existing refusal seams are:

- `hooks.py::_refused:455` returns `ConsolePromptDispatchResult(REFUSED, session_id, detail)` for busy/review/source changes.
- `prompt_queue.py::dispatch` checks existing Guardian/setup/display gates, activity, configuration/source inputs and the controller send-refusal copy before `_stage_normal_chain`.
- `prompt_queue.py::_stage_normal_chain` catches `RuntimeError`/`ValueError` from original async preparation/launch at `:1588` and returns a refusal with `_preparation_refusal_detail`’s text.
- `wiring._prepare_console_turn_to_runtime:3642` → `_capture_console_configuration_async:3591` → controller capture protects original runtime/store/session/config/callback identities and immutable input/source witnesses. These checks must not be bypassed to make the validation event fire.

The saved DB `OperationalError`/trace-maintenance warnings and `wake recovery failed` are separate observations. Without the returned refusal or exception receiving frame, they cannot be promoted to this send’s cause. The three passing preceding tests exercise presentation without sending a held provider turn; they do not qualify this fixture’s current send setup.

## Minimal next diagnostic, preserving the real route

Root should run the single failing node once with a **read-only profiler observer**, not replacement dispatch/worker callbacks. Observe `return` events for the original functions `ConsolePromptQueueUIController.dispatch`, `_stage_normal_chain`, and the hook controller’s `_refused`; retain only a returned `ConsolePromptDispatchResult` status/detail, selected session ID and originating function/line. Observe `_preparation_refusal_detail` input exception class and its bounded internal reason plus actual returned text. A profiler sees async-generator/coroutine suspension returns too: ignore None/non-result return events and keep the final typed result.

If the result selects preparation refusal, a scoped trace observer on the same original `_prepare_console_turn_to_runtime` and `_capture_console_configuration_async` frames can retain the actual exception type/internal code and its original line before the queue catches it. Do not wrap methods registered in `_console_received_live_adapters`, change native source methods or replace `run_worker`: those identity changes can deliberately force a different route and invalidate the diagnosis. Keep the existing30-second event bound and warnings-as-errors policy; no resets, permission exception, invented provider result or force-true readiness.

This one observation distinguishes a stale harness contract from a real new product refusal. Existing source equality/permission/ownership guards, actual provider-validation event and cadence assertions all remain authoritative. The root was sent this observer proposal promptly before report completion.

## Minimal correction and regression after exact reason is captured

There is **no unconditional fix proposal** yet. Apply one correction at the concrete existing seam identified by the result:

- If setup/configuration fixture drift is confirmed, make this mounted harness satisfy the same real provider/configuration/persistence contract using its existing app-factory/provider setup helper, rather than replacing native capture or weakening the gate. Keep the in-memory harness’s intentional non-agent route unless the actual test requires a file-backed native agent runtime.
- If original-source capture or preparation returns a real product refusal on valid unchanged inputs, fix only that original receiving/owning boundary and add its corresponding negative preservation control. Do not substitute an old monolithic implementation or introduce a new scheduling owner.
- First rerun this existing failing node with its original cadence assertions. Retain a positive observation of the real HeldGateway validation entry and one legitimate held turn; a synthetic `validation_started.set()` is not a regression check. Its current Event timeout is the existing red result, not a performance budget to enlarge.
- A small fixture-contract assertion should expose a refused send’s bounded detail before waiting for provider entry, while keeping exactly-one-send and draft/custody preservation checks. This makes later fixture drift fail at the real boundary. Keep separate before/after source receipts and the previously failing log.

## Limits and identities

Read-only code/log/receipt inspection only. No tests, runtime/profile actions, network, source changes, Git/index/ref mutations or helper agents. Only this requested report was written. Underlying refusal classification remains **unresolved pending the one exact observer**; no claim of an upstream performance bug, coroutine regression or harmlessness is made.

Observed current files (source writer may subsequently change them):

- `Tests/UI/test_console_projection_reads_off_loop.py`: `38f5e8e624792108d4fe52d4c7998e6f526c93e99f3ef8fbc4567d8bc8e042a8`
- `Tests/UI/test_console_config_sync_cadence.py`: `c5c665c8c1e71e34a5ad31afefff08312af2c4a0cffffd5828abb38d857eae1d`
- `Tests/UI/test_console_send_acknowledgement.py`: `dd823effba6096407dacce9cdc7cb39bad257cb86e7e385ddc0d506577ed2124`
- `Tests/UI/test_console_native_chat_flow.py`: `3f8cdda5d3812de7bc5759c60aca1025c6b92bc78d79432a2b9f598a495f81ca`
- `tldw_chatbook/UI/Console_Modules/composer_run_controls.py`: `3f4788de616374946e1ac3ef170a696416af7bb5fcdc68c3a00aec4361956208`
- `tldw_chatbook/UI/Console_Modules/prompt_queue.py`: `bb027d85f6a52c96eb6863ba869c4fa3aa268d97301d81925bb7d8b738752d5e`
- `tldw_chatbook/UI/Console_Modules/hooks.py`: `dbfd988f20e261a20c74e61a083239a0ea219f10e4c146a6d7a784d4f94fd0e3`


## Follow-up — exact refusal text and managed-memory eligibility

Root's first original observer established an actual `RuntimeError` with exact detail `Console preparation source unavailable; Send again.` received by `_preparation_refusal_detail`. This is new evidence after the initial report. Its ordinary mapping returns this text unchanged. The individual throwing line still needs the original exception traceback; this report does not promote the expected line below to an executed fact.

### Strongest source-level chain

1. `build()` uses `_build_console_send_test_app` → `attach_chachanotes_db`, which deliberately installs `CharactersRAGDB(":memory:")` before mounting. The app factory’s contract explicitly selects that shape to prevent automatic native agent-bridge construction while retaining a real persistence adapter. Textual itself is real.
2. The mounted screen has a real managed ConsoleRuntime. `_standard_configuration_references` rejects any participating memory DB at `console_configuration_preparation.py:403`–`:414`. Its refs include `store.persistence.db` and `app.chachanotes_db` (`_source_references:227`–`:298`). Thus this original memory DB cannot satisfy the stock off-loop native source contract. Cold skill/trust preparation does not remove that restriction: `capture_console_received_configuration_preparation:1350` repeats the same complete `_standard_configuration_references` test before returning an eligible preparation.
3. Queue precapture is normally **not selected** for stock async preparation: `prompt_queue.py:1306`–`:1313` requires the async capture to be absent or the synchronous callback to differ from its original async-source association. The stock wiring has both callbacks associated. So wiring’s `_precapture_turn_authority:801` is not the expected initial source of this refusal on the unchanged fixture, although the original traceback must verify the actual route.
4. The ordinary async launch calls `_prepare_console_turn_to_runtime` → `_capture_console_configuration_async` → `_controller_capture_turn_configuration_snapshot`. With its stock mounted provider, the controller attempts `capture_mounted_configuration_owned`. The ineligible memory sources lead to no owned capture; the exact first None branch may also expose earlier cold/trust readiness, so capture its return line instead of inferring which one ran.
5. On None, `_controller_capture_turn_configuration_snapshot:2949`–`:2976` still invokes the old finite-worker `capture_authority` fallback for a standard screen. `turn_admission.capture_authority:183`–`:189` explicitly refuses managed runtime/screen callers. **Expected original throwing site: turn_admission.py:189.** This is a source-level compatibility gap for the intentional managed-memory fixture, not evidence that native source checks should admit a memory DB on another thread.

### Smallest decisive second observer

At the existing `_preparation_refusal_detail` **call**, read the passed original exception’s existing `__traceback__` chain and retain only `(source filename, function name, line number)` plus the already-known fixed message/class. This directly selects one of the five literals without changing callbacks. Do not replace the error, clear it, replay a call or dump arbitrary frame locals.

Additionally observe genuine **terminal RETURN_VALUE** events (ignore coroutine yield/suspension returns) for these three exact original functions:

- `capture_mounted_configuration_owned`: record final None/non-None and the original return line. Its None points are `2662` (no cold preparation/trust source), `2666` (stock capture unavailable), `2670` (scratch adapter shape), `2677` (installed adapters), and `2680` (no selection).
- `capture_console_received_configuration_preparation`: record its final None/non-None and original return line. This distinguishes missing resident skill wiring/trust publication from the repeated structural-eligibility rejection.
- `_standard_configuration_references`: record True/False, the original return line, and only the already-held branch-local database’s concrete type/object identity and memory boolean when that branch is actually reached. No lazy app/service getters, database calls or path/body dumps. A False at the DB loop with the original chat DB and memory=true proves the memory qualification limit; other earlier False lines remain separate facts.

A traceback ending at turn_admission189, plus the original mounted-capture None return and ineligible original memory reference, is sufficient. No broad source tracing, helper replacement, timing alteration or further full selection is needed.

### Existing supported provider-direct fixture route

If the intended coverage is provider-direct execution and cadence rather than the narrower “construct no native bridge” implementation detail, there is an existing supported alternative: file-backed source custody **with `[console] agent_runtime=false` explicitly persisted before mount**. `console_live_runtime_enabled:292`–`:305` owns that execution switch; it is not equivalent to inferring plain mode from `:memory:`. `Tests/UI/test_console_cited_sources_persisted.py::send_path` demonstrates the actual config writer, and `_build_app:206`–`:225` uses the installed file-backed DB/service wiring while retaining direct mode. `_SendPath.assert_taken:179`–`:183` requires zero real agent turns on that direct path.

For a bounded fixture-only migration, reuse those existing config/storage construction patterns in the affected cadence/send harness, retain the same real Textual ConsoleHarness and HeldGateway, and assert the accepted configuration’s `agent_runtime_enabled` is false and no native agent loop executes. Do not just replace memory with a file under default true: that changes the tested dispatcher. A bridge object may legitimately be composed even though execution is direct; this alternative does **not** preserve the old stronger no-bridge-allocation assertion. It also needs the original owner’s file/thread retirement and private profile protection. The correct fixture scope/cost must be chosen by root after the observed branch is established; do not silently broaden every shared build() caller.

If the required contract is specifically an in-memory managed runtime that constructs no bridge, no existing supported native capture path was found in this read-only pass. Supporting it would require a narrow, explicit memory-on-owning-loop compatibility design that retains native source custody; allowing memory through the off-loop standard source gate or reviving unowned `capture_authority` is not an acceptable shortcut. That is product behavior beyond a fixture spelling fix and should not be smuggled into the observer.

Regression remains the exact original failing cadence node: real provider validation entry, held run, unchanged coalescing/freshness assertions and original bounds, followed by the alias-cancellation/acknowledgement controls already in root's selection. Preserve wrong-identity/source-loss negatives and add an explicit direct-path/no-agent observation if the supported file-backed fixture route is adopted. No new passing claim is made here.
