# Task 9 scope proof: existing InterruptRoundHost candidate

## Result: reject the current fit

Base: approved I1 `d3443b9e4297fa20897cf99e77a3ae2c8c562b10`. The metadata-only root checkpoint is not used as a different source baseline. Controller: **33,027** lines; immutable maximum: **29,367**.

Both actual Ruff-formatted controller sketches exceed the cap:

| Documentation contract | Original AST lines removed | Formatted public wrappers | New construction lines | Existing construction lines replaced | Projected controller | Over cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Preserve existing public docstrings | 4,644 | 2,004 | 301 | 5 | **30,683** | **1,316** |
| Relocate full docs to owner; one-line public pointers | 4,644 | 702 | 301 | 5 | **29381** | **14** |

Formula: base − removed AST spans + formatted wrapper AST spans + new construction body − replaced construction statements. Existing surrounding source whitespace is retained; no removal of unrelated comments/imports/initialization is credited. Only the fragment's artificial class/constructor header, explanatory constructor docstring and already-existing future import are excluded. Construction includes the actual import, nested write-through function, all keyword bindings, explanatory comment, and exact bound after-remount assignment.

The second sketch is a measured alternative for this same candidate, not authorization to change public docstrings. Its **1,403** lines of full public documentation remain verbatim in the owner-interface stubs; **101** forwarding wrappers carry one-line pointers. Even this relocation fails by **14 lines**, with no headroom for later qualification fixes. The earlier 250–350 estimate is superseded by these measured snippets.

## Written artifacts and static checks

- `task-9-scope-proof-controller.py`: full public signatures/decorators/docstrings, module and static deferred imports/forwarders, instance/property/async forwarders, and complete production construction fragment.
- `task-9-scope-proof-controller-relocated-docs.py`: same exact dependencies and forwarding with explicit documentation relocation.
- `task-9-scope-proof-host-contract.py`: keyword-only live dependency signature/assignments and original full method documentation in non-implemented owner-interface stubs.
- `task-9-scope-proof-function-contracts.py`: explicit module/static helper keyword-only global getter contracts and original documentation. Owner defaults are removed because public wrappers evaluate and supply the original defaults.
- `task-9-scope-proof-test-adapter.py`: clearly labeled minimal legacy fixture binding illustration, not production construction or a complete executable adapter.
- `task-9-scope-proof-map.json`: exact spans, source hashes, every selected self/global dependency, documentation destination, dynamic receiver sites, prior patch/census references, the exact bound-callback assertion, retarget rules and line accounting.

Formatted with existing **Ruff 0.16.6**. All five snippets were `ast.parse`d and passed Python `compile(..., 'exec')` without execution/import of production classes. These are syntax checks, not tests. Full-doc wrapper arguments/defaults, decorators and raw docstring values were compared to frozen I1: **zero mismatches**. No tracked source/tests/plan/index/HEAD/branch changes, test runs, installations or agents.

## Coverage proof and precise boundary

The frozen selected set has one necessary retention: `capture_worktree_recovery_intent` remains unchanged in the controller. It calls `capture_intent(self, session_id)`, which requires the actual controller receiver. Injecting the controller into the new owner would violate the brief; deleting five lines of purported savings is the honest disposition. The remaining **88 methods + 17 functions** remove 4,644 AST lines.

Production host construction has **93 separately named live controller getters**, **one explicit write-through callback**, and **35 separately named global getters**. There is no controller argument, arbitrary-name resolver, dependency bag, proxy/mixin, new lock or snapshot of a mutable callback. Getter lambdas read the exact original attribute/global at invocation time. Nullable setters/timeouts remain nullable values; the owning implementation checks the returned value using the original branch logic.

`dependency_dispositions` lists each read/call/write with users and a named target. Existing generic host registry/payload/lock aliases retarget only to their real existing host objects. Other state stays controller-owned and is read through explicit accessors. The only direct attribute assignment in moved methods is the lock-protected `_pending_decision_order += 1`; its separate setter writes through to the original field at the same point. In-place dict/set mutations continue through the live retrieved object, preserving the original lock rather than copying state.

The frozen I1 method adds `_observe_chat_creation_record` to the initial preflight dependency set; it is now a named getter returning the current controller method. The previously omitted dynamic `_raw_shell_providers` read is explicit with its original empty-tuple fallback. `_agent_bridge` retains its `None` fallback. The `getattr(self, '_interrupt_host', None)` site is enumerated as existing native-host ownership; the normal constructed-controller route has that owner. Behavior for bypassed construction remains a targeted compatibility check, not a claim established by compilation.

Every remaining bare `self` use was inspected: the three `getattr` sites above are mapped, and the recovery receiver call is retained unchanged. All runtime global reads in moved methods/functions are enumerated from Python symbol tables and converted to individually named getters, including helper-to-helper calls and nested callback closures. Public function defaults remain evaluated at the original public definitions and are forwarded explicitly; owner implementation defaults are omitted so no duplicate evaluation or eager back-import is needed. All original local imports retain their execution boundary.

The map carries each existing patch/assignment site and source hash. Moved-to-moved calls are routed via `read_controller_<method>()(...)`, so controller instance/class patches continue to intercept them after host construction. Moved free-helper reads use `read_global_<name>()`, whose closure resolves the original controller-module symbol each time; simple re-exports are not substituted for that routing. Constructor assignments store getter callables, not their returned mutable values/functions.

## Legacy contract amendment required

The historical interrupt-host design **section 3.2 expressly keeps bridge timeout resolution, payload building, return mapping, detached announcements and cancelled-decision audit behavior in the controller**. This proposal moves those bodies into the host, so it is not direct execution of the historical narrow design.

Before any source task, root must record a canonical amendment with these exact changes:

1. Expand the host from generic round lifecycle to the enumerated decision bridges, projection/summary behavior and verdict-construction helpers; retain public controller/module names as routing contracts.
2. Replace production `_seams` receiver storage with the explicit keyword-only callable contract. The old **27 receiver sites** and five dynamic kind setters are listed. Each getter must preserve missing/nullable behavior and call-time rebinding. No generic resolver may replace the five named setters.
3. Preserve native generic registry/payload/lock object identity and the separate chat-create lock. The chat creation records/grants/revoked set, observation/locked-source routines, launch and acceptance authority remain in the controller; no new host-owned copy is authorized.
4. Keep the current `after_remount['approval']` assignment to the exact bound controller wrapper. The assertion is `hook.__func__ is ConsoleChatController._maybe_fire_permission_summary`; the construction fragment preserves it verbatim in meaning. New production dependency getters remain late-bound separately.
5. Decide and document public documentation ownership explicitly if choosing the second sketch. Full docstring relocation is an API/introspection change even though the complete prose remains present.
6. Specify fixture compatibility separately. Existing `FakeSeams`/`FakeSeamsFull` callers mutate/delete setters and inspect `host._seams`; a completed explicit fixture adapter must preserve those observations without restoring broad production receiver access. The included fixture illustration intentionally does not pretend to supply all newly required dependencies.

Keep ADR-067's indefinite human waits, paused execution deadline and revocation ordering. Preserve shutdown/event-arm semantics from the existing design section 3.1 and non-reentrant locking from section 3.3. No SQLite read, callback, UI dispatch or await is moved into a registry lock. Full body retargeting is not implemented by this scope proof and remains review work for a later authorized task.

## Loading, carry, and targeted implications

All new module/static forwarder imports load the existing `console_interrupt_rounds` module at the function's actual execution boundary. Construction retains its existing local import. Original local service imports stay local; no new eager module, cycle, exclusion, or increase is proposed. Ready cap **1,033** and preimport allowance **557** remain unchanged; static proof does not establish measured boot residency.

Carry the exact approved I1 body of `request_chat_create_confirm`, including its `_observe_chat_creation_record` route. All capture/read/match/record authority helpers stay in place. The map hashes and symbolic spans are tied to approved I1; dependency retargets are narrowly enumerated, while untouched source must remain blob/AST-equivalent apart from line movement. Neither earlier correctness approval nor snippet compilation substitutes for review of that binding change.

Later targeted qualification remains the existing interrupt/host-wiring/attention, confirmation, summary/payload and selected review-builder tests named in the preflight map, plus the established I1 regressions once through the moved route. Preserve exact original assertions, especially the callback identity and registry/lock alias assertions. Check post-construction module/instance patch routing and the retained recovery receiver contract. Reattribute moved diagnostic inventories, retain external-fork-writer source coverage for the new owner, and measure module/ready/preimport guards with this worktree on `PYTHONPATH`. No tests were run here; no broader Console/provider/schema/encryption sweep is proposed.

## Specific additional disposition needed

Reject the present candidate as scoped. With current public documentation it needs **1,316 further net lines**. Even after the explicit documentation relocation it needs **at least 14 further net controller lines plus honest headroom**.

The first coherent additional disposition to propose separately is **pruning only existing-host shim wrappers proven externally unreachable**, while retaining every public, patched, unbound-called, callback-identity or source-census-dependent name. Such pruning needs exact receiver/deadness evidence and actual formatted counts; the current map does not authorize deleting any named wrapper. No broader candidate search or pruning was performed. Root should define that narrow follow-up or reject the architecture; do not squeeze bindings/documentation or raise the cap.
