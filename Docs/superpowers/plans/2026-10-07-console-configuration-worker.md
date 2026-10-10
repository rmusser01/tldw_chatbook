# Console native configuration capture implementation plan

> Use the three authorized parallel lanes with root as sole integration and native-run owner.

**Goal:** Keep input responsive while stock Console configuration is captured. Select view values once, run eligible native capture through one Chat-owned boundary, and return the existing complete snapshot.

**Base:** a7f93c6ff41dfd44fb45ee4918a99225b87b8ce9.
**Task:** TASK-34563.14.
**Spec:** Docs/superpowers/specs/2026-10-06-console-send-preparation-architecture-design.md.

ADR required: yes, existing ADR-225 applies.
ADR path: backlog/decisions/225-console-send-preparation-and-io-ownership.md.
Reason: directly implements selected-input and finite domain capture ownership without changing authority or persistence.

## Current evidence and scope

The real mounted path awaits full configuration before runtime custody. The existing async controller method moves MCP source reads to a worker, then calls the shared configuration producer synchronously on the input loop. Workspace/project admission, character/emote, dictionary/world and skill/trust/plugin capture can perform native work there. The existing shared producer and complete snapshot model remain the result boundary.

This slice immediately serves the current Send/queue capture entry points. It does not activate early received UI intent, change draft/staging ownership, rewrite queue intake, or claim that all pre-dispatch work is off-loop. It does not broaden into reference expansion, Library search, provider/history or durability redesign. Those latter operations already follow complete-request custody.

Preserve saved-turn refusal with draft retained, existing temporary chats, WAL/NORMAL, required consent/checkpoint/effect ordering, and awaited history. Preserve domain defaults and optional/error behavior. Keep initial MCP ceilings distinct from later live composition and execution authority.

No primary checkout edits, push, merge, reset or stash. Targeted checks only. Native runs and timing are sequential and coordinated with the UAT chat.

## Interface and ownership

Shared boundary in Chat/console_configuration_preparation.py:

- Frozen ConsoleTurnCaptureSelection contains detached existing adapter choices: provider selection, presentation, selected RAG defaults, normalized tool configuration, and explicit skill workspace (including None).
- capture_console_turn_configuration_owned(app, store, session_id, *, selection, creator, reads, observers=(), require_current) returns the existing ConsoleTurnConfigurationSnapshot.
- Native sources come from exact captured app/domain owners, not screen callbacks. The stock worker resolves eligible scratch/project, workspace/profile/persona, character/prompt and skill inputs through their existing owners. Pure selected values and live execution authority stay distinct.
- Final adapter selection fields and eligibility rules are reviewed before implementation. Custom synchronous context providers and supported injected callbacks keep their existing arguments, one-call behavior, loop affinity, errors and live selection. Do not move arbitrary custom callbacks into a worker.
- Retain the shared synchronous producer/API and one snapshot construction body. Thin stock adapters use the owned async entry; do not duplicate two snapshot assemblers.
- Separate loop-only adapter/currentness checks from pure worker source validation. Refuse changed controller/store/session/incarnation/binding/settings/config/service sources before publication. Do not reread replacement owners to assemble an old result.
- Actual native operations use operation_owned_connection for their eligible database owners; no connection or native lease spans an await. Existing memory-database and custom-owner affinity remains explicit.

Extract the already qualified physical-read mechanism into Chat/console_preparation_reads.py with generic names. Keep hook session attribution and compatibility reexports in console_hook_preparation.py. Rename the existing controller/runtime observation sets to _preparation_reads, without another index or scheduler. Both domains observe exact records and original retirement notices through the existing Stop/Close/dispose drains.

MCP remains a segmented domain operation: native source read, loop-owned inventory/governance, and optional native audit. A private optional _run_native parameter on capture_console_definition_maximum allows this owned stock capture to register exact original workers with the same physical lifetime mechanism. Preserve its existing two positional arguments and default/custom behavior; do not collapse loop-owned callbacks into a giant worker or reuse a ceiling as execution permission.

| Lane | Exclusive file ownership |
| --- | --- |
| Shared preparation | Chat/console_configuration_capture.py; new Chat/console_configuration_preparation.py; new Chat/console_preparation_reads.py; Chat/console_hook_preparation.py; narrow MCP/console_snapshot.py change; new Tests/Chat/test_console_configuration_preparation.py |
| Controller/provider integration | Chat/console_chat_controller.py; Chat/console_runtime.py; UI/Console_Modules/session.py; narrow stock-provenance wiring in UI/Console_Modules/wiring.py; mechanical observation-set references only in Tests/Chat/test_console_hook_preparation_lifetime.py |
| Baseline verification | New Tests/Chat/test_console_configuration_worker_lifetime.py and, if necessary, Tests/UI/test_console_configuration_worker_feedback.py; original control selection and independent source review |
| Root integration | Plan/task/report, ownership/API rulings, every native run, final review/static checks and exact commit |

All product paths above are under tldw_chatbook/. Do not edit another lane's files. Root resolves interface changes before concurrent implementation.

## Verification and execution

- [x] Review the selected-value/API and stock/custom boundaries before code.
- [x] Add a meaningful original-entry RED: observe a real native configuration producer on the original loop path, with bounded independent release and actual resource retirement. Do not count a missing new API as the behavioral regression.
- [x] Implement the shared producer and thin adapters in parallel after the original RED. Preserve .9 mounted/runtime RAG, skill workspace, boolean and explicit-None semantics.
- [x] Verify original native resource scopes and input scheduling under a held producer, source replacement before/after issuance, repeated cancellation, teardown and exact record transfer. Do not wait indefinitely for unrelated custom/provider tails.
- [x] Run final integrated controls after both implementation lanes are ready: new capture tests, existing asynchronous MCP/capture parity, hook physical ownership, received admission and real mounted saved/refused Send controls.
- [x] Observe natural compositor updates and actual input mutation where the fixture supports it. Do not force rendering or synchronization inside the measured interval. Headless frame readiness is not physical terminal write/flush. Keep the 100 ms target explicit when not established.
- [x] Inspect source hashes, physical retirement receipts, bounded diagnostics and warnings. Run scoped AST/Ruff/formatter/diff and existing size ratchets without increasing caps. Record inherited failures separately; no full-suite claim.
- [x] Review integrated source, record exact evidence/remaining limitations, and commit only qualified scope.

Performance attribution is bounded and separate from clean timing. The earlier .7 observer-free Send-to-adapter samples (10.304/9.710/8.964 seconds) are historical; they are not this slice's baseline or gain. A worker move alone does not establish lower total latency, 100 ms feedback, or subsecond dispatch.

## Review record

The concrete selection/eligibility review and original-producer RED completed before implementation was released. Existing user approval authorized the three implementation lanes and sequential integration checks.

## Concrete adapter review refinements

The stock mounted adapter is _build_console_turn_capture_selection(session_id). It returns the detached selection or declines eligibility before invoking the existing synchronous builder exactly once. Known custom context providers stay on their existing route. Selected provider/RAG/presentation callbacks execute on the loop with their existing errors; no screen callback reaches the worker.

Custom profile/persona/scratch adapters use synchronous fallback; this slice adds no override bag. Production scratch wiring receives explicit named stock provenance with lazy runtime lookup. Do not identify it by filename/name heuristics, treat custom None as a stock signal, or eagerly construct runtime/scratch during wiring. Replaced stock callback bindings must decline/refuse rather than disappear behind the optimization.

The baseline RED calls the existing controller.capture_turn_configuration_snapshot with real WorkspaceDB/registry sources. A passive observer holds the original get_workspace only after its actual connection opens; an independent bounded helper releases it so the old loop path cannot deadlock. Assert original worker affinity and heartbeat-before-release. Cleanup observations distinguish pre-existing borrowed handles from handles this operation created. Optional mounted coverage uses real input events and natural supplied compositor updates, without forced rendering.

### Final interface rulings and original regression

- Include explicit project_bindings_eligible in selection: mounted and runtime route eligibility intentionally differ.
- Runtime-native CLI tool and RAG defaults resolve on the worker. The new internal selection may use explicit absent fields for those defaults, with a selected runtime agent-enabled flag; mounted values remain detached and console_run_budget resolves on the worker. This does not reinterpret explicit None in the existing public synchronous producer.
- Provider/presentation and required selected-value callbacks remain loop-affine. This slice does not claim every possible selection-time configuration read disappears.
- The mounted stock scratch provider is a named lazy functools.partial adapter in wiring. Check exact provenance and the current resident scratch owner at selection; do not invoke scratch.snapshot on the loop. Replaced scratch/profile/persona callbacks decline to the original synchronous route.
- Original a7f93c6 regression observed: the actual Workspace get_workspace SQL body ran on MainThread. The observer reached a real admitted native connection; the assertion failed at the original worker-affinity requirement. The independent releaser avoided loop deadlock. Containment retired normally with force, overflow and lookup races zero; private profile removed. Evidence: configuration-worker-native-red and red-audit.json.
- Product implementation is now released to the two assigned lanes. Final integration tests wait until both implementations are ready. No broader latency/feedback result follows from the original RED.

### Ready-source eligibility and scope limit

Stock eligibility reads only existing slots and static descriptors; it must not trigger lazy skill/trust/plugin construction on the input loop. This slice offloads already initialized stock sources (and absent optional sources). Cold lazy or custom source graphs retain their original synchronous route and affinity. No new service construction policy, profile-wide lock or speculative dependency-graph reflection is added. This limitation must appear in final results; it does not establish cold-start responsiveness.

The mounted tests use normal app/mount behavior, with no test-only prewarming to force eligibility. If the normal source graph stays cold and the test observes loop execution, that failure must be reconciled honestly before a mounted responsiveness claim.

Do not relocate the controller project-authority helper or its dependency closure in this slice. The shared worker can lazily invoke its existing single Chat-owned body after the calling controller is loaded. This avoids duplicate logic and additional migration scope while remaining screen-free.

### Integrated verification findings

The first combined lifecycle run passed 33 controls and failed both mounted input cases. The capture batch passed 40 controls and found an empty optional-skill test fixture plus five preserved refusals whose reason code had changed. Restore the existing reason code and seed real skill data through the original service API; retain the assertions.

Passive mounted diagnostics proved that the existing app factory placed its real LocalMCPStore outside the configured bootstrap profile: exact source types and every original catalog method matched, but the source had no native binding. The existing MCP guard correctly declined it before new configuration eligibility. Do not loosen production guards or prewarm services to mask this. Baseline verification additionally owns a narrow optional caller-owned data-directory argument in Tests/UI/app_factory.py; mounted controls use their genuine configured private profile. Default factory behavior remains unchanged. Remove temporary diagnostics before final qualification. This fixture correction still does not authorize a cold-construction policy change.

Root additionally owns a narrow app_service_wiring.py change: replace the stock disabled-builtins closure with a named lazy partial and an original function/code witness. Shared eligibility consumes exact provenance instead of a module-name heuristic; config remains read live and cold construction policy is unchanged.

### Mounted cold-plugin finding and narrow domain correction

The corrected canonical-profile Enter test still observed original Workspace SQL on MainThread (`configuration-worker-mounted-canonical`, 2026-10-07). Source review identifies the cold local plugin factory as an actual readiness blocker: ordinary skill capture always asks for the lazy facade, even though PluginService.capture_maximum only reads its already published catalog and a never-created facade has no rows.

The shared capture adapter reads the resident plugin facade only for the exact stock app-bound lazy factory. A ready local facade retains precedence. Root owns the pure resident-app getter and defining-module factory identity; no LocalSkillsService API changes are required. Shared preparation owns only a defining-module identity for its directly bypassed plugin_service property, so replaced getters decline to the original synchronous route. The sole shared snapshot producer accepts a private sentinel-default plugin owner input, analogous to its supplied MCP maximum: the owned worker supplies its qualified facade (including explicit None), while default synchronous/custom callers keep the original lazy property. Eligibility/currentness checks track the exact consumed factory and resident facade. No plugin construction, native worker startup, new lock, authority or activation change is introduced. The previous ready-only rule is refined only for this absent published metadata source; cold trust/scope construction remains unchanged. This demand-side correction avoids prewarming and the construction races an off-thread lazy initializer would introduce.

### Natural-frame fixture correction

After the published-owner correction, the original mounted Workspace producer ran on a worker and driver input mutated the composer in 4.13 ms, but no text frame was observed. A single bounded diagnostic showed normal refresh callbacks, batch count zero and the observed text below the viewport. The navigation harness omitted the production bundled stylesheet required by ChatScreen. The final control uses a local harness subclass loading that existing bundle, keeps 120x40, eager navigation and original deadlines, and asserts visible geometry before observing Send. No render is forced, and temporary diagnostics are removed before final qualification.

Final verification: [configuration worker report](../../Development/2026-10-07-console-configuration-worker-verification.md). Core 31 + adapter 24 + lifecycle 35 passed; one built-in skill control passed; two native plugin checks stop at the unchanged macOS-only gate. Full DoD and larger latency qualification remain open as recorded in the report.
