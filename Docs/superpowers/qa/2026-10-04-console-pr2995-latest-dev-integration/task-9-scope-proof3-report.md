# Task 9 proof3: latest-dev structural scope

## Result

**FIT as a measured structural proposal, conditional on root selecting both additional boundaries. This is not source qualification or implementation approval.**

| Source/proposal | Lines | Methods | Gate/result |
|---|---:|---:|---|
| Approved I1 controller at d344 | 33,027 | — | Original proof2 projected 29,315 |
| I1 + actual dev controller union | 33,695 | 494 | Clean three-way merge; +668 lines |
| Same union with prior InterruptRoundHost boundary alone | 29,983 | — | **NON-FIT**, 616 over 29,367 |
| Prior boundary + five-method compaction proposal | **29,336** | — | **FIT**, 31 lines below 29,367 |
| Reviewed Task7 screen + actual dev union | 25,239 | 761 | **NON-FIT**, 21 lines and two methods over |
| Screen + direct relocation of both new adapters to Session | **25,194** | **759** | **FIT**, 24 lines and zero methods spare |

The concrete controller formula is `33,695 − 4,644 + 657 + 280 − 5 − 856 + 93 + 115 + 1 = 29,336`. The last line is a real blank separator between constructor statements. The concrete screen formula is `25,239 − 47 + 2 = 25,194`; removing both private definitions changes 761 to 759. A proposal retaining screen forwarders remains NON-FIT on methods.

Counts use actual source text and Ruff 0.16.6 formatted fragments. A final cache-disabled format check confirms all five new snippets are already formatted. Full disposable projected files were assembled, parsed and compiled, without executing them. Existing surrounding whitespace, comments and documentation were not removed for savings. No cap changes, tests, production imports, fetches, tracked source changes, index/ref changes or subagents occurred.

## Inputs and union disposition

- Base: `5e0341d1ec701865e019eb2fd8a5e2028ab2d474`.
- Approved I1: `d3443b9e4297fa20897cf99e77a3ae2c8c562b10`.
- Actual fetched dev: `f800952214844c3d41ddbe4b6522c93d4be25b7c`.
- Metadata checkpoint `7c5528654742512475c60757036149e2804a89e4` was previously source-identical to I1; it is not used to invent new source qualification.

`git merge-file -p` on disposable snapshots in this SDD returned zero for both controller and screen. Both unions parse. The incoming controller diff is 729 insertions / 61 deletions, net +668; screen is 51 insertions / 11 deletions, net +40. Root's separate Task10 work and the future real Task11 rebase are not represented as already qualified here.

Both branches modify four controller methods: `__init__`, `_apply_conversation_memory_preflight`, `_submit_draft_body`, and `resume_durable_postcommit`. The three-way union retains their merged text; it has no conflict markers or manual resolution. The map enumerates all changed AST definitions. The `ConsoleChatController` class aggregate also appears in the overlap inventory because its body includes those methods.

**All 105 selected InterruptRoundHost boundary definitions remain AST-identical to I1.** Its prior seven private-wrapper deadness proof, 98 retained forwarding contracts, exact patch map and explicit late-bound dependency inventory are reused. The 427-file census was not rerun. Proof1 and proof2 artifacts remain immutable; their hashes are recorded in the proof3 map. The legacy host's 27 `_seams` uses still require the previously enumerated named production dependencies; none is silently removed to obtain the new fit.

## Additional compaction responsibility

Proposed implementation location: `tldw_chatbook/Chat/console_context_compaction.py`, in a **new explicit `ConsoleCompactionPreflight` collaborator inside the existing resident compaction module**. This is a new cross-module contract requiring a canonical decision; it is not an assertion that the current `ConsoleCompactionService` already owns controller policy orchestration. The transaction service remains unchanged.

The coherent responsibility is request assessment and execution of the existing compaction preflight, including its explicit Compact now entry and its shared capacity/overflow verdicts. The admission mode gate and all hold transitions stay with their existing controller authority.

| Method | Union source span | Removed lines |
|---|---:|---:|
| `compact_context_now` (including controller decorator) | 27823–27904 | 82 |
| `_apply_conversation_memory_preflight` | 28268–28858 | 591 |
| `_assess_context_compaction` | 28155–28214 | 60 |
| `_assess_request_capacity_only` | 28021–28089 | 69 |
| `_context_overflow_alert` | 27966–28019 | 54 |
| Total | | **856** |

The formatted controller fragment has **93 wrapper lines + 115 constructor lines**, plus one real separator in the complete projection. There are **26 named live controller getters + 41 named live module-global getters**, each explicitly declared and assigned in the owner. There is no generic receiver, proxy, resolver, mapping of dependencies, `_seams` object, new lock or snapshotted bound method.

The owner sketch contains actual moved bodies, all original full docstrings and comments, and narrow binding substitutions. It measures **1,139 class lines**, including its explicit dependency signature and assignments; with the existing 3,034-line compaction module and two separating blank lines, the proposed module measures **4,175 lines**. Its standalone snippet is 1,143 lines because it also has snippet imports. The snippet's imports already exist in the resident module. The existing interrupt module is 930 lines before its separately proven extraction; proof2 did not produce a complete formatted implementation of that host, so this report does not invent a final interrupt-owner file count.

All five body ASTs compare equal to the union after reversing only the named binding substitutions. All five argument/default ASTs and raw docstrings compare equal. `@_maintenance_boundary("compact")` stays **only on the controller wrapper**, preventing a duplicate maintenance fence or an owner masquerading as the controller. All wrappers preserve signatures, defaults and sync/async shape. Full owner documentation is not duplicated into wrappers; the accurate forwarding docstrings are counted.

### State and order

- The owner holds only named accessors. No selected method directly assigns a controller attribute. The live `_context_accounting_by_session` mapping still receives its existing item mutation; replacing that mapping remains visible through its getter. `store`, `_context_repository`, `_compaction_service`, `provider_gateway`, `_agent_bridge` and runtime-enabled facts remain live dependencies.
- Optional gateway `prepare_chat_request` and bridge `preview_tool_schemas` lookup, their `None`/callability checks and exception paths remain in the exact bodies. No optional dependency becomes a no-op stand-in.
- Controller `_compaction_admission`, `_automatic_memory_admission`, `_block_context_preflight`, `_hooks_for_compaction`, durable snapshot and memory selection callbacks retain ownership and call order. ADR-052 admission recheck, failed-attempt latch, bounded summary, commit fence and retry behavior remain on the same objects.
- `_compaction_mode_for`, `_compaction_admission_check`, `_hold_send_for_compaction`, `_cancel_context_compaction_hold`, `context_compaction_hold`, `_held_for_compaction`, `_held_send_echo_id`, `_consume_compaction_hold_answer`, `compact_and_send`, `send_without_compacting`, `_resume_compaction_hold` and their maps/sets remain controller-owned. The early and late admission checks and after-effects resume behavior remain in the incoming union.
- `_snapshot_staged_evidence`, `_capture_frozen_rag_context`, `_PreparedEvidenceLease`, freeze/capture/release wiring, no-evidence admission freeze and `_drop_preparation` behavior remain in place. No new source, launch, acceptance or evidence authority is transferred.
- Nested calls to moved methods still live-read the original controller wrapper, so instance/class patches continue to work. Module-global helpers still resolve the original controller-module binding at each use. All existing method names in this new boundary remain; no additional private wrapper is removed.

### Patch and reflection routes

`task-9-scope-proof3-routes.txt` records every literal/identifier hit for the five new compaction members and two new screen members in tracked source/tests at f800. The existing private preflight patches in automatic library preparation and local citation tests continue to target the same controller wrapper; the `compact_context_now` settings patch also remains. Direct call sites in live compaction, context budget, memory, maintenance, handoff and rewind tests retain their entrypoints.

The new computed `getattr(controller, action)` in `test_console_compaction_ask_hold.py:278` is bounded by its parameter list at :262 to `compact_and_send` / `send_without_compacting`, both retained. Added actual-dev computed/reflection candidate lines are recorded separately; worker-scanner snippets and fixture prose do not imply a production dynamic method route. The changed production bridge/gateway getters are the exact optional protocol lookups described above. No new route to the prior seven removed interrupt helpers is introduced by the actual delta.

The previous callback assertion remains exact: in `Tests/Chat/test_console_interrupt_host_wiring.py:65`, `hook.__func__ is ConsoleChatController._maybe_fire_permission_summary`; :66 asserts `set(controller._interrupt_host.after_remount) == {"approval"}`. The projection retains the bound controller callback.

## Screen: only the two incoming private adapters

Root's continuation permits direct ownership, because forwarding definitions cannot meet 759 methods. Move `_dispatch_console_trace_recovery` and `_console_trace_recovery_state` into existing `ConsoleSessionController`. They form one session draft/recovery adapter responsibility, using existing live controller and composer dependencies.

Their bodies contribute **47 formatted lines** to Session, with four new named constructor reader parameters and four explicit assignments. Wire the two original screen-module global readers through two new named `build_console_controllers` parameters. Two additional Session readers return the **current bound screen** start-timer and UI-sync methods. No new runtime import is required; no widget is retained. The screen constructor gains two measured lines.

`_build_console_center` passes `self._session._console_trace_recovery_state` and `self._session._dispatch_console_trace_recovery` directly. These are the only actual named source consumers, at f800 screen :16171/:16172. The definitions are :16096/:16131; union spans are 15875–15908 and 15910–15922. Existing `_build_console_center` callers at f800 :2348 and :16489 continue to build the same region through the same screen method. Public navigation and action names are untouched.

**Intentional identity change:** the region's two private callbacks become Session-bound, not screen-bound. Any external private-name lookup or external `__func__`/`__self__` assertion could break. No tracked source/test named consumer, borrowed unbound adapter, or identity assertion for these two methods was found. This is a normal private-interface relocation only if root accepts that compatibility boundary. It is not identity-preserving for those two adapters.

Within dispatch, `on_started` and `on_finished` are still the actual screen methods read at action time, preserving their receiver and late patch behavior. The original order remains: read held text → dispatch controller action with start/finish callbacks → resolve current composer → refill only after cancel, cleared hold, nonempty held text and empty composer → return the exact result. State projection still passes the same preparation and hold. Original owner docstrings and comments are retained.

Downstream `ProviderContinuationTranscriptRegion` stores its supplied callbacks (:686/:687), calls the state builder while composing/syncing (:696/:720), invokes the action callback and awaits an awaitable (:729 onward), and rechecks the builder (:734). It performs no adapter-name or bound-receiver reflection. The compatibility `trace_call_recovery` module simply exports those existing helpers. The prior `dir(controller)` headless-wake probe only selects attributes bound to a screen; it does not discover either new adapter on the chat controller, and the new Session getters themselves are ordinary lambda dependencies.

### Exact fixture adaptation

Two tracked tests call `build_console_controllers` directly and must receive the two new named global readers:

- `Tests/Chat/test_console_video_actions.py:96` — bare real screen construction.
- `Tests/UI/test_console_auto_speak_wiring.py:160` — MagicMock screen construction.

Add `read_trace_recovery_dispatch` and `read_trace_recovery_state` lambdas resolving the existing `chat_screen` module's exported helpers; import that module alias in the fixture if absent. Do not replace any assertion, synthesize a new screen callback or borrow a method onto an incomplete owner. The same exact caller/name search at d344 finds the same two wiring fixtures and no additional adapter or borrowed-center route. No direct `ConsoleSessionController(...)` constructor in tracked tests was found at either input. The real mounted `test_console_compaction_hold_flow.py` continues to assert exact held-draft refill, one accepted user message, expected provider counts and no recovery panel; `test_console_trace_call_recovery.py` retains button, refusal, keyboard and card behavior assertions. Neither test is claimed passing here.

## Canonical selection needed and remaining limits

Root must separately select/record:

1. The unchanged interrupt-host expansion and canonical amendment of the legacy design §3.2 restriction, including the enumerated replacement for production `_seams`, existing locks/aliases and callback identity.
2. The five-method preflight contract in the resident compaction module, with controller authority/state retained, original full owner docs, controller-only maintenance decoration, and named late dependencies. Reference canonical ADR-052 for admission/commit/latch invariants. This is a real cross-module contract, so a mechanical-placement-only rationale is insufficient.
3. The two private screen adapters' direct Session ownership and changed region callback receiver, plus the two test fixture wiring adaptations. No cap increase or unrelated private API cleanup is proposed.

The final implementation and actual Task11 rebased source must be remeasured. Headroom is only 31 controller lines and 24 screen lines, with no screen method headroom. The proposal has no uncounted controller loader, dependency stubs or body placeholders; further required source lines could still consume that headroom. Task10 changes to Session are independent concurrent source and must be combined before reporting a final Session module size; this report gives the exact proposal additions instead.

Existing ready/preimport caps remain **1,033 / 557**. Compaction is already imported by the controller at union :168; Session and wiring are resident. Imports inside moved bodies remain local. This supports a no-new-module expectation, not a measured cold-boot result. No import/prewarm, worker/census, diagnostic-inventory, behavioral, targeted-test or live qualification was run. The final diagnostic/worker owner paths must be updated and checked under the later selected plan without excluding moved bodies from inventories.

## Artifacts

All paths are in this plan's SDD and begin `task-9-scope-proof3-`:

- `report.md`, `map.json`: outcome, counts, binding/state/route map and prior immutable evidence.
- `controller-{base,ours,dev,union,projected}.py`, `screen-{base,ours,dev,union,projected}.py`: disposable immutable-input snapshots, clean unions and measured proposals.
- `compaction-controller.py`, `compaction-owner.py`: actual formatted dependency/wrapper fragment and complete moved bodies.
- `screen.py`, `screen-session.py`, `screen-wiring.py`: actual formatted caller, Session additions and explicit wiring additions; class/constructor headers in fragments are context scaffolding, not extra production replacements.
- `routes.txt`, `new-reflection-map.json`, `equivalence-map.json`, `projection-map.json`, `compaction-measure.json`, `screen-measure.json`, `union-map.json`: exact supporting measurements and routes.
- `measure.py`, `screen-measure.py`, `project.py`: proposal-only generation/measurement scripts; these use parsing, formatting and text assembly, never production imports/execution.
