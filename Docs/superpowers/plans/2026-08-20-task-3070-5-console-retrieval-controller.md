# TASK-3070.5 Console Retrieval Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ConsoleRetrievalController` the sole non-DOM owner of Console retrieval scope, Library RAG, auto-retrieve, and cached dictionary/world-book inspector policy without changing behavior.

**Architecture:** Move the 32 reviewed non-framework methods and six state defaults from `ChatScreen` into one DOM-free controller. Keep the two Textual `@work` names on `ChatScreen` as bounded delegates, keep picker/modal and widget synchronization on the screen, and pass every remaining screen or sibling dependency through an explicit late-bound callable in `wiring.py`.

**Tech Stack:** Python 3.11+, Textual 8, pytest/pytest-asyncio, Ruff, stdlib AST checks.

---

**ADR required:** no

**ADR path:** N/A

**Reason:** This implements the ownership boundary already approved in `Docs/superpowers/specs/2026-08-13-console-decomposition-wave6-design.md` and `DESIGN.md` section 7; it changes no storage, security, provider, dependency, or service contract.

## Constraints and evidence rules

- Preserve the exact method inventory in `Tests/Architecture/test_console_wave6_inventory.py`: 32 moved methods, two bounded delegates, and six assignable compatibility fields.
- The controller must never call `query_one`, `query`, `push_screen`, or another controller through the screen. DOM, modal presentation, and decorated worker registration stay screen-owned.
- Preserve worker decorators and groups exactly: `@work(thread=True)` for preference persistence and `@work(exclusive=True, group="console-library-rag-search")` for explicit retrieval.
- Preserve scope persistence and cache keys, in-memory SQLite behavior, effective conversation/workspace intersection, auto-retrieve timeout/cancellation/placeholder identity, and cached inspector zero-DB-on-build behavior.
- Existing mounted tests may be retargeted mechanically from `screen.<moved method>` to `screen._retrieval.<moved method>`; their assertions and user-visible expectations do not change.
- Do not run the full repository suite. The user explicitly limited verification to touched files and functionality.
- The focused pre-change baseline is 132 passed and one inherited architecture failure: current `chat_screen.py` is 21,357 lines versus the committed 20,943 incremental ceiling. The reviewed retrieval extraction removes at least 982 net lines and must make that gate green without raising the ceiling.
- After every mutation probe, restore the exact candidate bytes before continuing.

## File map

- Create `tldw_chatbook/UI/Console_Modules/retrieval.py`: controller state, scope persistence/effective-state policy, inspector projections, Library RAG outcome policy, and auto-retrieve orchestration.
- Modify `tldw_chatbook/UI/Console_Modules/wiring.py`: construct `_retrieval`, expose named late-bound dependencies, and repoint Workspace/Session retrieval callables directly to the controller.
- Modify `tldw_chatbook/UI/Screens/chat_screen.py`: add six `_ControllerState` descriptors, remove moved defaults/methods/imports, retain two bounded worker delegates, and route production callers to `_retrieval`.
- Create `Tests/UI/test_console_retrieval_controller.py`: isolated no-mount controller tests.
- Modify `Tests/Architecture/test_console_wave6_inventory.py`: require complete retrieval ownership, exact delegate spans/decorators/groups, descriptor targets, DOM prohibition, and non-vacuity.
- Modify `Tests/UI/test_console_controller_wiring.py` and `Tests/UI/test_console_moved_seam_guard.py`: register `_retrieval`, prove late binding, and reject stale screen method calls.
- Mechanically retarget only directly affected tests, expected primarily in `Tests/UI/test_console_auto_rag_on_send.py`, `Tests/UI/test_console_dictionaries_screen.py`, `Tests/UI/test_console_worldbook_inspector.py`, `Tests/UI/test_console_rag_settings_modal.py`, `Tests/UI/test_console_scope_row.py`, `Tests/UI/test_console_live_work_handoffs.py`, `Tests/UI/test_console_staged_evidence_strip.py`, `Tests/Library/test_library_rag_scope.py`, and the small citation/cost/resume/character seam tests found by the moved-seam guard.
- Modify `backlog/tasks/task-3070.5 - Extract-Console-retrieval-controller.md`: link this plan now; complete ACs and notes only after all focused gates and review pass.
- Modify `Docs/security/production-diagnostic-inventory.json` only if the canonical non-write checker proves an expected owner/digest move; regenerate once and review the exact generated diff.

### Task 1: Lock retrieval ownership and no-mount behavior with RED tests

- [x] Add `Tests/UI/test_console_retrieval_controller.py` with a small constructor fixture using plain call recorders and no Textual mount.
- [x] Assert controller defaults for `_console_retrieval_scope_cache`, `_console_effective_scope_cache`, `_active_dictionaries_summary`, `_last_console_dictionary_scope_ids`, `_active_world_books_summary`, and `_last_console_world_book_scope_ids`.
- [x] Characterize the four high-risk policy families before moving code:
  - effective-scope cache build/read and persisted-vs-unpersisted save;
  - dictionary/world-book summary refresh guards and cached row/action projection;
  - Library RAG scoped/empty/results/blocked outcomes and stage ordering;
  - auto-retrieve gates, placeholder identity, timeout/failure containment, cancellation, and same-send capture.
- [x] Extend the architecture test so the retrieval family must be complete: all 32 M names only on `ConsoleRetrievalController`; the two D names remain on `ChatScreen` with their exact decorators/groups and at-most-five-line definition spans; all six descriptors target `_retrieval`; moved methods contain no DOM calls or sibling-controller reach-through.
- [x] Add a structural assertion that ordinary production callers use `_retrieval` directly and that no moved default is assigned in `ChatScreen.__init__`.
- [x] Run the new controller and architecture nodes. Confirm RED is caused by the absent controller/current screen ownership, not the known line-ceiling failure alone.

### Task 2: Create the controller and move scope/cache ownership

- [x] Create `ConsoleRetrievalController` with `app_instance` plus explicit keyword-only late-bound callables for active session/conversation, pending-launch state transitions, source/query setters, screen sync/refresh, scope-row/control-bar sync, visible-run dispatch, launch payload inspection, and evidence consume/release.
- [x] Initialize all six compatibility defaults in the controller constructor.
- [x] Move the scope/cache methods verbatim in behavior: staged capture, pure display-state build, recipe count, effective resolution/warm, DB read/write, lister construction, and scope save.
- [x] Add six `_ControllerState("_retrieval", ...)` assignments to `ChatScreen` and remove the corresponding `__init__` assignments.
- [x] Repoint Workspace and Session late-bound scope dependencies in `wiring.py` to `screen._retrieval` at call time.
- [x] Run the focused controller/scope/architecture tests. Expect defaults, descriptors, cache, and save cases green while later inspector/RAG ownership remains RED.

### Task 3: Move inspector and Library RAG policy

- [x] Move cached dictionary/world-book scope IDs, refresh guards, summaries, rows, and action projections. Keep attach/detach pickers and workers on `ChatScreen`; route their refresh calls to `_retrieval`.
- [x] Move source-status and source-scope-label builders. Keep screen-owned inspector composition and widget label updates, but source their plain values from `_retrieval`.
- [x] Move stage, settings-choice, scope resolution, outcome application, service-initialization detection, notification, placeholder clearing, and auto-retrieve orchestration.
- [x] Preserve the pending-launch screen state through explicit getter/setter callbacks and keep all DOM refresh/recompose operations behind named screen callbacks.
- [x] Move the body of `_capture_console_staged_rag` and rewire both provider registration sites directly to `_retrieval`.
- [x] Run the no-mount controller tests plus existing auto-RAG, dictionary, world-book, RAG-settings, and Library-scope tests to GREEN.

### Task 4: Finish worker delegates, wiring, and call-site migration

- [x] Add `_retrieval = ConsoleRetrievalController(...)` to `build_console_controllers`; update its documented build order/count and wiring tests.
- [x] Keep `_persist_console_rag_auto_retrieve_on_send` on `ChatScreen` with `@work(thread=True)` and a one-statement controller delegation.
- [x] Keep `_execute_console_library_rag_search` on `ChatScreen` with `@work(exclusive=True, group="console-library-rag-search")` and an awaited controller delegation.
- [x] Repoint all ordinary production callers to `_retrieval`, including compose/inspector/sync, explicit RAG run, scope flush/resume, staged handoff, attach/detach refresh, and the Session/Workspace seams.
- [x] Mechanically retarget direct private-method tests to the controller owner. Do not alter assertions or mounted user flows.
- [x] Register `ConsoleRetrievalController` in the moved-seam AST guard and prove the guard reports a synthetic stale `screen.<moved method>()` call.
- [x] Remove imports from `chat_screen.py` only when `rg` and Ruff prove they are no longer screen-owned; import them in `retrieval.py` from their defining modules.
- [x] Run the controller, architecture, wiring, moved-seam, and exact mounted retrieval files to GREEN.

### Task 5: Mutation and focused regression evidence

- [x] Remove one `_retrieval` descriptor setter/retarget it to screen shadow state; confirm the descriptor test fails, then restore.
- [x] Put one moved method back on `ChatScreen` or add a DOM call inside the controller; confirm the ownership/DOM AST test fails, then restore.
- [x] Bypass the effective-scope empty short-circuit; confirm the scoped retrieval test fails, then restore.
- [x] Remove placeholder identity guarding or failure cleanup; confirm the auto-RAG test fails, then restore.
- [x] Remove the dictionary/world-book unchanged-scope guard; confirm the zero-repeat summary test fails, then restore.
- [x] Remove each worker delegation in turn; confirm the exact delegate/behavior test fails for the intended reason, then restore.
- [x] Run only touched-functionality groups: retrieval controller/architecture/wiring, scope/modal/Library RAG, auto-send/capture/staging, dictionary/world-book inspector, and any mechanically retargeted seam tests.

### Task 6: Static checks, diagnostics, review, and closeout

- [x] Run Ruff format/check on every changed Python file only. Do not bulk-format unrelated files.
- [x] Run `py_compile` for `retrieval.py`, `wiring.py`, and `chat_screen.py` under one validated temporary pycache root; remove only that exact root and prove it absent.
- [x] Run `git diff --check`, inspect the cumulative diff for the one-controller/YAGNI boundary, and verify no CSS, DOM IDs, storage schema, dependency, or user-visible copy changed.
- [x] Run the persistent-diagnostic checker and its two focused architecture tests. Regenerate the manifest only for proven moved-owner metadata with unchanged sink topology.
- [x] Obtain an independent correctness/spec review if the active collaboration policy permits it; otherwise perform and record a separate self-review pass without changing scope.
- [x] Update the task ACs and concise Implementation Notes with exact RED/GREEN/mutation/static evidence, ADR decision, modified files, inherited baseline classification, and the explicit no-full-suite user constraint.
- [ ] Commit the implementation, rebase onto latest `origin/dev`, rerun the same focused gates on the rebased head, then push and open one atomic PR against `dev`.
