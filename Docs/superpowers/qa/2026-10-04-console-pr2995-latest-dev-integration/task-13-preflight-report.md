# Task13 newest-dev read-only preflight

Status: coherent bounded integration candidate. Actual reviewed source `a6887fdb8931adb9d1b99a376d63d9b9b66a5bb5`; exact incoming delta `7d155170dc95557736a239a1ce7427981f4d50ec` → `8f83422dde2a5b95da648882b8f7e09da5b41f08`: **6 commits / 17 paths**, PR3005. Ten production paths, three new test files, one existing test helper change, two guide appends and the existing Backlog task. No tracked/index/HEAD/ref/object edit, fetch, worktree creation, suite/collection/production execution or child agent. Task12 and original35 final-loading receipts are untouched.

Exact API blobs, immutable actual/base source, private patch sketches and formatter measurements are under `/private/tmp/task13-preflight`. The two own SDD artifacts are this report and `task-13-preflight-selection.json`; that JSON records scoped source/hash/span/AST/formatter data and exact proposed selectors. No whole configuration/source bodies or broad historical/tree census were saved into these new artifacts.

## Source intersections and coherent resolution

All ten production patches apply privately to the actual source without fuzz. Nine are existing files and one is the exact new129-line pure trace-row owner. Nonoverlapping incoming changed definitions match incoming ASTs; no additional declaration changed beyond the upstream set. Four genuine shared-method intersections require preserving the reviewed feature body while applying the narrow incoming hunk:

| Actual shared method | Incoming change | Reviewed behavior that remains |
| --- | --- | --- |
| `ConsoleChatController._submit_draft_body` | A second send behind a PAUSED preparation reports `Last send is blocked; resolve it first.` | Machine-start authorization and literal input; manual withdrawal/draining before busy gate; fresh staged input exclusions; capture/durable preparation; receipt and coordinator fields; source/acceptance fences. Incoming changes only the existing-preparation refusal copy at this site. |
| `ConsolePromptQueueUIController.presentation_for` | Pass oldest recovery's `turn_recovery_reason` to pure derivation. | Existing final prepared-start Send override and accepted-start Running override remain after ordinary derivation. |
| `build_console_controllers` | Add named live `turn_recovery_reason` callback to prompt queue, reading the runtime's oldest recovery entry. | ADR220 required recovery dependency arguments and Session-bound trace-recovery callbacks, current settings accessors and hook routes remain. Do not restore old screen recovery receivers or alter callback binding. |
| `ChatScreen._build_console_workbench_state` | Pass `blocked_turn_reason(controller)` into workbench state. | Prepared-start `run_allows_send` exception and image-edit gate remain exact. |

Other changed controller owners: `_build_speculative_voice_capture_request` retargets the old role helper into local `console_trace_row_sources.unsaved_trace_artifact_source`; `_build_durable_trace_request` uses local `saved_message_id`/`unsaved_row_sources`; `_handle_durable_trace_provenance_failure` records the keyed content-free stage. Existing `_unsaved_trace_artifact_source` is removed after all known repository consumers retarget. Bounded immutable grep finds only its original controller definition/two internal consumers, no test/private patch route; arbitrary external private consumers remain unproven, as in the upstream relocation.

`unsaved_row_sources` respects aggregate partition: leading system context or tagged memory, later system history, original TOOL_CALL/TOOL_RESULT mapping and final active request. `saved_message_id` rejects saved-revision provenance when provider-visible rows omit saved images, preserving exact durable dispatch-surface comparison rather than weakening it. The voice builder retains its existing saved-owner mapping and only relocates the unsaved helper; upstream explicitly records its omitted-media limitation as a residual.

Runtime refusal propagation stays outside `_ConsoleTurnCustodyRecord`. New `_ConsoleTurnRefusedError` carries reason separately from fixed exception `args`; `_run_custodied_turn` preserves archive/controller refusal copy; `_finish_custodied_turn` forwards it to `_record_turn_recovery`; `ConsoleTurnRecoveryEntry.reason` defaults empty and is excluded from repr. Actual machine-start runtime amendments are in other declarations and retain exact AST. Do not move prompt/attachment/refusal content into lifetime handles or logs.

Presentation state adds `ConsoleInspectorState.run_blocked_reason`, `TraceCallRecoveryState.provenance`, a pure `blocked_turn_reason` based on the actionable paused preparation while no active run exists, and bounded shelf copy. Header, run chip, inspector authority and trace callout consume that state. The helper does not introduce storage, scheduling or acceptance authority. The shelf constructor adds a default empty named live callback; new wiring reads the oldest runtime recovery, so neither a screen snapshot nor a new durable field is introduced.

No store/DB/AgentRuns/schema/source-Close/confirmation-host/compaction owner changes. Existing ADR219/220 ownership, ADR097 boot/ratchet policy and upstream TASK33621.2 repair contract apply; this integration creates no new architectural decision. Follow the existing design language for the unchanged shipping UI tokens and geometry; no CSS/token edits are incoming.

## Exact cap and formatter result

Actual installed Ruff0.16.6 through stdin with actual frozen `pyproject.toml`; no install or tracked formatting:

| Owner | Reviewed raw | Private integrated raw | Whole-file Ruff measurement | Final incoming-only candidate |
| --- | ---: | ---: | ---: | ---: |
| Controller |29327|29299|29294|**29301**|
| ChatScreen |25192 /759methods|25192 /759|25192|**25192 /759**|
| Runtime |5460|5497|5510| preserve inherited formatting |
| Prompt queue |1161|1199|1199|1199|
| Wiring |2430|2442|2442|2442|
| New pure owner |absent|129|129|129|

The incoming-only controller candidate corrects exactly `rows =tuple` spacing and wraps the incoming `saved_message_id` call, adding2lines to raw29299. All original compaction/trailing whitespace formatter inheritance remains; whole-file29294 is measurement only. Existing controller row29367 would leave66lines of slack against29301, exceeding the unchanged50-line rule. **Lower that one literal to actual final measured count** (private candidate29301); do not raise caps, tolerance or format unrelated controller bodies. Screen row25218/759 remains with26line slack and unchanged methods. Store22344, interrupt6479 and compaction rows are untouched.

Other inherited whole-file formatter drift is recorded in selection JSON for actual/dev/proposal. Do not convert that into broad formatting scope. New helper and changed already-formatted queue/wiring/screen/widget sources fit their existing formatting. This preflight does not claim final source bytes or a ratchet pass.

## Loading intersection and prior35 receipts

**No automatic replay of the35-case final loading cohort is proposed.** The new helper imports only inside the two actual trace-request builders. The screen imports one new function from already resident `provider_continuation_recovery`; that module's FEEDBACK_ACTIVE_RUN_STATUSES addition comes from its already imported `console_chat_models`. New helper dependencies are models/prepared-request/provenance modules already in the controller closure. The stage recorder import stays request-failure-local and uses the existing diagnostic owner. Default fields, label helpers and reason callbacks add no eager module identity, worker, service construction or CSS.

The ready harness snapshots modules synchronously at `_ui_ready` without sending; app-import identity/deferred dependency guards have no changed eager target. Preimport payload measures modules added after prewarming ChatScreen; these changed Chat dependencies are outside that marginal payload, and the new first-send-only helper does not execute in the import pass. App/route/worker/preimport/boot-CSS owner sources are unchanged. Changed header/recovery method bytes have real mounted coverage in the incoming tests, so presentation behavior must qualify even though module identities do not change.

Keep35 passed at a688 as historical. Implementer must provide a truthful scoped import/constructor/route/callback carry map for the integrated source, preserving old receipts and unchanged public continuity/Close guards; do not state a fresh loading pass. If actual implementation changes import placement or observed qualification loads a new module before ready, root selects that specific loading node. No such executed observation exists here.

## Smallest targeted proposal

`task-13-preflight-selection.json` provides exact nodes. Proposed behavioral selection is51cases by source inspection, not collection/pass evidence:

- All three new files:7real-controller triggers/log category,13pure aggregate/media cases,12installed UI triggers/recovery/archive cases.
- Two existing changed shelf-walk nodes:12widths plus input-coverage guard.
- One actual voice-builder provenance consumer and lifetime-only runtime custody control.
- Four directly combined feature nodes: native literal/dual receipts, manual prepared-start withdrawal, mounted prepared Send, and mounted accepted target/Stop.

Then only controller two-row ratchet and ChatScreen two-row ratchet. Preserve all original assertions, profile boundaries and upstream test-only maintenance deferral. No evidence currently requires an incoming fixture repair. A future actual failure must be attributed before amending its fixture; do not presume the older video resolve seam incident applies to these new manual-send tests.

The new `record_send_stage` call must retain content-free category/exception type/attempt ID; incoming log test qualifies that route. Refresh the scoped existing diagnostic inventory only if the actual scanner projection changes; no broad scanner/schema/provider/fork/Console sweeps. All other current feature/runtime authority and loading outcomes carry by exact unaffected source or enumerated shared-method hunk preservation, not by inventing new-tree pass results.

Upstream records unresolved post-mount trace GC timing, voice media admission and Temporary shelf duplicate recovery limitations. They are explicit residuals outside this17-path integration and do not justify expanding work. Root can now scope Task13 for one source writer, including the four combined-method preservation obligations, exact incoming-only formatting/downward controller row, targeted qualification and independent review.
