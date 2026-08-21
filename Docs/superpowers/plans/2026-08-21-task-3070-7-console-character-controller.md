# TASK-3070.7 Console Character Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

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

## Recorded starting point

- Branch: `codex/task-3070-7-console-character`
- Immutable implementation base: `f4c45fc14a47d79ae86c0c58bd97af0a759a6f87` (`origin/dev` when planning began)
- `ChatScreen`: 20,202 physical lines and 655 direct methods
- Remaining screen-owned character inventory: six methods / 167 physical definition lines
- Already controller-owned: `_refresh_active_character_avatar_if_scope_changed`
- Already deleted: `_fetch_expression_image_bytes`
- Focused baseline: 2 failed, 21 passed, 2 warnings. Both failures are the inherited `ChatScreen.__new__` avatar fixture reaching `console_view_hooks()` without `_skill`; all character production and architecture nodes in that command passed.
- Current RED slice: 1 failed, 6 passed, 1 warning. The failure is the exact ownership assertion listing the six methods still on `ChatScreen`; controller-policy and oracle tests collect and pass.

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

The initial plan/task commit is `d6cd2640a`. Before resuming implementation, run:

```bash
git status --short
git diff --check
```

Expected: only reviewed paths from the file map and no whitespace errors. Preserve and
inspect existing dirty bytes; never overwrite them merely to obtain a clean tree.

### Task 1: Lock controller ownership and behavior with RED tests

- [ ] Add plain, no-mount controller tests for picker option projection/error containment, active rail conversation/character identity, card fetch containment, and new/swap character choice behavior.
- [ ] Extend the Wave 6 inventory so all seven current M methods exist only on `ConsoleCharacterController`, the already-deleted expression helper stays absent, four compatibility descriptors remain controller-backed, and the controller has only named non-DOM dependencies.
- [ ] Add a non-vacuity mutation/oracle proving a synthetic screen-owned moved method or DOM access fails the architecture contract.
- [ ] Run only the new controller/architecture nodes and confirm RED comes from missing controller ownership, not an unrelated fixture failure.

**Files:**

- Create: `Tests/UI/test_console_character_controller.py`
- Modify: `Tests/Architecture/test_console_wave6_inventory.py`

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_character_controller.py \
  Tests/Architecture/test_console_wave6_inventory.py::test_character_family_has_completed_controller_ownership \
  Tests/Architecture/test_console_wave6_inventory.py::test_character_move_ownership_oracle_is_non_vacuous \
  Tests/Architecture/test_console_wave6_inventory.py::test_character_controller_has_only_named_non_dom_dependencies
```

Expected before screen extraction: the ownership node fails and names exactly the six
remaining screen methods; the no-mount policy and structural-oracle nodes pass. A
collection/fixture error is not acceptable RED evidence.

Commit the reviewed RED contract before completing production movement:

```bash
git add Tests/UI/test_console_character_controller.py \
  Tests/Architecture/test_console_wave6_inventory.py
git commit -m "test(console): lock character controller extraction"
```

### Task 2: Move character policy into the controller

- [ ] Move `_console_character_picker_options`, `_current_console_rail_conversation_id`, `_current_console_rail_character_id`, `_current_console_rail_character_name`, `_fetch_character_card_for_avatar`, and `_apply_console_character_choice_async` into `ConsoleCharacterController`.
- [ ] Add the minimum explicit dependencies for DB access, active/current session identity, store/config access, default settings/swap, notifications, and final UI synchronization; keep all callbacks late-bound.
- [ ] Repoint `character.py`'s existing avatar request name lookup to its own character-identity method and retain existing avatar request/stale-result behavior.
- [ ] Route screen picker presentation and worker callback to `_character`; route screen/retrieval/agent and other production conversation/character consumers directly to `_character`; retain no moved-method screen shim.
- [ ] Update the stale `session.py` ownership documentation and remove imports that lost their final screen caller.
- [ ] Run isolated controller, architecture, wiring, picker, prompt-seed/handoff, avatar, composer-menu, and rail-focused nodes to GREEN.

**Files:**

- Modify: `tldw_chatbook/UI/Console_Modules/character.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/Console_Modules/session.py`
- Modify: `Tests/UI/test_console_controller_wiring.py`

Keep the constructor keyword-only. Add exactly these policy edges beside the existing
avatar edges: `active_native_session_accessor`, `current_conversation_id_accessor`,
`character_db_accessor`, `ensure_chat_store`, `provider_readiness_config_accessor`,
`default_session_settings`, `swap_session_character`, `sync_temporary_chip`,
`sync_native_chat_ui`, and `notify`. Use the controller's own
`_current_console_rail_character_name()` in the avatar request; delete the redundant
`character_name_accessor` edge.

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_character_controller.py \
  Tests/UI/test_console_controller_wiring.py \
  Tests/Architecture/test_console_wave6_inventory.py::test_character_family_has_completed_controller_ownership \
  Tests/Architecture/test_console_wave6_inventory.py::test_character_move_ownership_oracle_is_non_vacuous \
  Tests/Architecture/test_console_wave6_inventory.py::test_character_controller_has_only_named_non_dom_dependencies \
  Tests/Architecture/test_console_wave6_inventory.py::test_wave6_compatibility_inventory_is_complete_and_phase_safe
```

Expected: all selected nodes pass. Then commit the controller move:

```bash
git add tldw_chatbook/UI/Console_Modules/character.py \
  tldw_chatbook/UI/Console_Modules/wiring.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/UI/Console_Modules/session.py \
  Tests/UI/test_console_controller_wiring.py
git commit -m "refactor(console): extract character controller"
```

### Task 3: Repair affected fixtures without weakening product wiring

- [ ] Replace the stale avatar bare-screen setup with the smallest real/controller-level fixture that does not attach runtime hooks through an incompletely initialized `ChatScreen`.
- [ ] Update tests that monkeypatch or call moved screen methods to patch/call `_character` instead; preserve assertions, cardinality, copy, and durable-state oracles.
- [ ] Keep presentation tests mounted where pixels/layout matter and controller tests unmounted where only policy/data matters.
- [ ] Rerun the exact formerly failing avatar nodes and the focused affected matrix.

**Files:**

- Modify: `Tests/UI/test_console_character_avatar.py`
- Modify: `Tests/UI/test_character_session_prompt_seed.py`
- Modify: `Tests/UI/test_console_composer_menu.py`
- Modify only additional tests returned by an exact moved-name caller scan.

First prove the inherited fixture repair:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_character_avatar.py::test_current_console_rail_character_id_reads_active_session \
  Tests/UI/test_console_character_avatar.py::test_current_console_rail_character_id_none_for_generic_session
```

Expected: 2 passed. Then run the focused behavior matrix:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_character_controller.py \
  Tests/UI/test_character_session_prompt_seed.py \
  Tests/UI/test_console_character_avatar.py \
  Tests/UI/test_console_composer_menu.py::test_character_picker_new_chat_clears_a_stale_temporary_chip \
  Tests/UI/test_console_controller_wiring.py
```

Expected: all selected nodes pass. If the avatar file reaches the repository's
20-minute checkpoint, stop, record completed nodes, and resume in named node groups;
do not silently narrow the claimed matrix.

Commit only ownership-driven fixture changes:

```bash
git add Tests/UI/test_console_character_avatar.py \
  Tests/UI/test_character_session_prompt_seed.py \
  Tests/UI/test_console_composer_menu.py
git commit -m "test(console): cover character controller behavior"
```

### Task 4: Prove behavior and mutation sensitivity

- [ ] Mutate picker validation/error containment and confirm the no-mount picker test fails, then restore.
- [ ] Mutate new/swap placement or prompt seed/config propagation and confirm the focused handoff test fails, then restore.
- [ ] Break one late-bound wiring edge and confirm the wiring test fails, then restore.
- [ ] Reintroduce one moved method on a synthetic `ChatScreen` AST fixture and confirm the ownership oracle fails, then restore.
- [ ] Re-run the affected functionality matrix and record exact passes, skips, warnings, and any inherited exceptions.

Use `apply_patch` for each mutation and its inverse. Before each mutation, record
`git diff --binary -- <mutated-paths> | shasum -a 256`; after the inverse patch, require
the same checksum so restoration proves exact candidate bytes rather than mere syntax.
Required discriminators:

1. Accept one invalid picker card or leak one DB exception; the picker projection node fails.
2. Change `new`/`current` placement or drop prompt-derived settings; the handoff node fails.
3. Eagerly bind one wiring edge; the `_character` late-binding node fails.
4. Add one moved method to the synthetic screen map; the ownership oracle fails.

After every failure, restore immediately, compare the exact diff checksum, rerun the
node to GREEN, and confirm `git diff --check` plus a focused residue scan are clean.

### Task 5: Static checks, review, and closeout

- [ ] Run Ruff lint/format checks on changed Python files only, `py_compile` for changed production modules under a validated temporary cache root, `git diff --check`, and the focused screen-size/Wave 6 architecture nodes.
- [ ] Run the persistent-diagnostic checker; update its inventory only for a proven content-identical owner move and verify sink topology is unchanged.
- [ ] Review the cumulative diff for Ponytail/YAGNI scope: one existing controller, no new abstraction/dependency/config/schema/copy/layout/DOM ID, no compatibility method shim, and no unrelated formatter churn.
- [ ] Update plan checkboxes and TASK-3070.7 ACs/Implementation Notes with exact RED/GREEN/mutation/static evidence, ADR decision, modified files, and the affected-only test constraint.
- [ ] Commit the candidate, rebase on latest `origin/dev`, rerun touched-functionality gates, push, open one atomic PR against `dev`, address validated Qodo/CI comments, and merge when required checks are green.

Run Ruff on every changed Python path; the initial expected set is:

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/UI/Console_Modules/character.py \
  tldw_chatbook/UI/Console_Modules/wiring.py \
  tldw_chatbook/UI/Console_Modules/session.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/UI/test_console_character_controller.py \
  Tests/UI/test_console_character_avatar.py \
  Tests/UI/test_character_session_prompt_seed.py \
  Tests/UI/test_console_composer_menu.py \
  Tests/UI/test_console_controller_wiring.py \
  Tests/Architecture/test_console_wave6_inventory.py

../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/UI/Console_Modules/character.py \
  tldw_chatbook/UI/Console_Modules/wiring.py \
  tldw_chatbook/UI/Console_Modules/session.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/UI/test_console_character_controller.py \
  Tests/UI/test_console_character_avatar.py \
  Tests/UI/test_character_session_prompt_seed.py \
  Tests/UI/test_console_composer_menu.py \
  Tests/UI/test_console_controller_wiring.py \
  Tests/Architecture/test_console_wave6_inventory.py
```

Append any additional changed Python test before running both commands. Expected: both
exit 0. Compile changed production modules with `PYTHONPYCACHEPREFIX` under one fresh,
owner-validated `/private/tmp/task-3070-7-pycache.*` directory; reject links/non-regular
entries, remove exactly that validated root, and prove it absent.

Run the diagnostic gate before writing:

```bash
../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
```

If red, prove it is only a content-identical diagnostic transfer from `chat_screen.py`
to `character.py`, with aggregate counts and sink topology unchanged. Only then run:

```bash
../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py --write
git diff -- Docs/security/production-diagnostic-inventory.json
../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
../../.venv/bin/python -m pytest -q \
  Tests/Architecture/test_persistent_diagnostic_inventory.py::test_production_diagnostic_inventory_and_sink_topology_are_unchanged \
  Tests/Architecture/test_persistent_diagnostic_inventory.py::test_reviewed_diagnostic_changes_are_metadata_only
```

Expected: non-write checker and both nodes pass; JSON changes only redistribute
identical moved calls and do not alter sink topology.

Final focused architecture gate:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Architecture/test_console_wave6_inventory.py::test_character_family_has_completed_controller_ownership \
  Tests/Architecture/test_console_wave6_inventory.py::test_character_move_ownership_oracle_is_non_vacuous \
  Tests/Architecture/test_console_wave6_inventory.py::test_character_controller_has_only_named_non_dom_dependencies \
  Tests/Architecture/test_console_wave6_inventory.py::test_wave6_projection_clears_both_ratchet_overages \
  Tests/Architecture/test_console_wave6_inventory.py::test_wave6_compatibility_inventory_is_complete_and_phase_safe \
  Tests/Architecture/test_console_wave6_inventory.py::test_wave6_structural_oracles_are_non_vacuous
git diff --check
```

Expected: all selected nodes pass and `git diff --check` is silent. Do not run the full
repository suite. Record the user-authorized affected-only deviation in task notes,
check every AC, and set TASK-3070.7 Done only after all focused/static/diagnostic gates.

After rebasing onto the then-current `origin/dev`, do not reuse any pre-rebase generated
or architecture evidence. Rerun the affected behavior matrix, both Ruff commands, the
non-write diagnostic checker, the two diagnostic nodes, and the exact final focused
architecture command above on the rebased SHA. If the diagnostic checker is red, repeat
the same owner-transfer/count/topology review and conditional single writer sequence
before proceeding. Only these fresh post-rebase results may support task Done, push, or
merge.
