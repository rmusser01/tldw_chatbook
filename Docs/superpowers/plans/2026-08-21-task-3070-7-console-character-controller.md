# TASK-3070.7 Console Character Controller Implementation Plan

> **For implementers:** Follow strict RED/GREEN TDD and the repository's affected-functionality-only test constraint. Do not run the full repository suite.

**Goal:** Make the existing DOM-free `ConsoleCharacterController` own character picker projection, session handoff policy, active character/conversation identity, card retrieval, and avatar refresh state while leaving the picker modal and rail pixels screen-owned.

**Architecture:** Move the six remaining approved non-DOM `ChatScreen` methods into the existing character controller and keep the already-moved avatar refresher there. Wire explicit late-bound callables for session/store/config/notification/UI-sync edges; retain no screen handle and no DOM access in the controller. Keep `_open_console_character_picker`, the worker-launching picker callback, and avatar rendering on `ChatScreen`, but route them directly to `_character`.

**Tech Stack:** Python 3.11+, Textual 8, pytest/pytest-asyncio, Ruff, stdlib AST checks.

---

**ADR required:** no

**ADR path:** N/A

**Reason:** This is a behavior-preserving implementation of the character ownership boundary already approved in `Docs/superpowers/specs/2026-08-13-console-decomposition-wave6-design.md`; it changes no storage, runtime/service contract, dependency, security boundary, or user-visible application structure.

## Constraints and evidence rules

- Treat latest merged `origin/dev` as authoritative: `_refresh_active_character_avatar_if_scope_changed` already lives in `character.py`, and `_fetch_expression_image_bytes` is already retired. Move only the six remaining screen methods and preserve the seven-method controller inventory.
- `ConsoleCharacterController` must not hold `ChatScreen`, query the DOM, push a modal, or launch a Textual worker. Every non-owned edge is an explicit late-bound callable.
- `ChatScreen` retains `_open_console_character_picker`, `_apply_console_character_choice`, and `_render_character_avatar_into_section` as bounded presentation/framework seams.
- Preserve prompt templates, raw character identity, sanitized notification copy, new-vs-current placement, durable session behavior, avatar request fencing, picker ordering, and rail rendering.
- Preserve the four existing `_ControllerState` compatibility descriptors and defaults; do not add aliases for moved methods.
- Preserve `test_console_controller_wiring.py`'s original six-controller `_EXPECTED_SLOTS`, order, and shared-accessor characterization. Add only focused `_character` construction and late-binding checks.
- Repair the inherited bare-screen avatar fixture by constructing/using the real character ownership seam; do not add a production fallback for missing `_skill` or partially initialized `ChatScreen` instances.
- Run only tests related to changed files or character behavior, per the user's standing instruction. After each mutation, restore candidate bytes exactly.

## File map

- Modify `tldw_chatbook/UI/Console_Modules/character.py`: own picker projection, active rail identity, card retrieval, character choice/session handoff, and existing avatar refresh orchestration.
- Modify `tldw_chatbook/UI/Console_Modules/wiring.py`: construct `_character` with named late-bound dependencies and route retrieval/agent conversation access through `_character`.
- Modify `tldw_chatbook/UI/Screens/chat_screen.py`: route picker presentation, picker callback, avatar rendering inputs, and all remaining production callers directly to `_character`; remove the six moved method bodies and obsolete imports.
- Modify `tldw_chatbook/UI/Console_Modules/session.py`: update the historical ownership documentation that still says the picker policy remains screen-owned.
- Create `Tests/UI/test_console_character_controller.py`: isolated no-mount controller behavior and mutation-sensitive tests.
- Modify `Tests/Architecture/test_console_wave6_inventory.py`: exact moved/deleted/state/default/no-DOM dependency contracts and non-vacuity.
- Modify `Tests/UI/test_console_controller_wiring.py`: focused `_character` construction/late-bound edge checks only.
- Modify focused character behavior tests (`test_character_session_prompt_seed.py`, `test_console_character_avatar.py`, `test_console_composer_menu.py`, and directly affected controller-consumer tests) only where ownership changed.
- Modify `Docs/security/production-diagnostic-inventory.json` only if fresh generation proves an owner-file move with identical diagnostic content and unchanged sink topology.
- Modify the TASK-3070.7 task and this plan for truthful closeout evidence.

### Task 0: Freeze the latest-dev baseline and plan

- [x] Record the exact worktree branch/base, source line/direct-method counts, task status, and current seven-method/latest-dev character inventory.
- [x] Reproduce and classify the focused baseline. Record the inherited avatar-fixture failure caused by bypassing `ChatScreen.__init__` after TASK-3070.6 added `_skill` to runtime hooks; prove no character production path is involved.
- [ ] Commit only this reviewed plan/task metadata before production implementation.

### Task 1: Lock controller ownership and behavior with RED tests

- [ ] Add plain, no-mount controller tests for picker option projection/error containment, active rail conversation/character identity, card fetch containment, and new/swap character choice behavior.
- [ ] Extend the Wave 6 inventory so all seven current M methods exist only on `ConsoleCharacterController`, the already-deleted expression helper stays absent, four compatibility descriptors remain controller-backed, and the controller has only named non-DOM dependencies.
- [ ] Add a non-vacuity mutation/oracle proving a synthetic screen-owned moved method or DOM access fails the architecture contract.
- [ ] Run only the new controller/architecture nodes and confirm RED comes from missing controller ownership, not an unrelated fixture failure.

### Task 2: Move character policy into the controller

- [ ] Move `_console_character_picker_options`, `_current_console_rail_conversation_id`, `_current_console_rail_character_id`, `_current_console_rail_character_name`, `_fetch_character_card_for_avatar`, and `_apply_console_character_choice_async` into `ConsoleCharacterController`.
- [ ] Add the minimum explicit dependencies for DB access, active/current session identity, store/config access, default settings/swap, notifications, and final UI synchronization; keep all callbacks late-bound.
- [ ] Repoint `character.py`'s existing avatar request name lookup to its own character-identity method and retain existing avatar request/stale-result behavior.
- [ ] Route screen picker presentation and worker callback to `_character`; route screen/retrieval/agent and other production conversation/character consumers directly to `_character`; retain no moved-method screen shim.
- [ ] Update the stale `session.py` ownership documentation and remove imports that lost their final screen caller.
- [ ] Run isolated controller, architecture, wiring, picker, prompt-seed/handoff, avatar, composer-menu, and rail-focused nodes to GREEN.

### Task 3: Repair affected fixtures without weakening product wiring

- [ ] Replace the stale avatar bare-screen setup with the smallest real/controller-level fixture that does not attach runtime hooks through an incompletely initialized `ChatScreen`.
- [ ] Update tests that monkeypatch or call moved screen methods to patch/call `_character` instead; preserve assertions, cardinality, copy, and durable-state oracles.
- [ ] Keep presentation tests mounted where pixels/layout matter and controller tests unmounted where only policy/data matters.
- [ ] Rerun the exact formerly failing avatar nodes and the focused affected matrix.

### Task 4: Prove behavior and mutation sensitivity

- [ ] Mutate picker validation/error containment and confirm the no-mount picker test fails, then restore.
- [ ] Mutate new/swap placement or prompt seed/config propagation and confirm the focused handoff test fails, then restore.
- [ ] Break one late-bound wiring edge and confirm the wiring test fails, then restore.
- [ ] Reintroduce one moved method on a synthetic `ChatScreen` AST fixture and confirm the ownership oracle fails, then restore.
- [ ] Re-run the affected functionality matrix and record exact passes, skips, warnings, and any inherited exceptions.

### Task 5: Static checks, review, and closeout

- [ ] Run Ruff lint/format checks on changed Python files only, `py_compile` for changed production modules under a validated temporary cache root, `git diff --check`, and the focused screen-size/Wave 6 architecture nodes.
- [ ] Run the persistent-diagnostic checker; update its inventory only for a proven content-identical owner move and verify sink topology is unchanged.
- [ ] Review the cumulative diff for Ponytail/YAGNI scope: one existing controller, no new abstraction/dependency/config/schema/copy/layout/DOM ID, no compatibility method shim, and no unrelated formatter churn.
- [ ] Update plan checkboxes and TASK-3070.7 ACs/Implementation Notes with exact RED/GREEN/mutation/static evidence, ADR decision, modified files, and the affected-only test constraint.
- [ ] Commit the candidate, rebase on latest `origin/dev`, rerun touched-functionality gates, push, open one atomic PR against `dev`, address validated Qodo/CI comments, and merge when required checks are green.
