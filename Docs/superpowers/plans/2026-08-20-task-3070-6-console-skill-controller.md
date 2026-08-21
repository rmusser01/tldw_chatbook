# TASK-3070.6 Console Skill Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make one DOM-free `ConsoleSkillController` own the live Console skill policy while deleting the unreachable slash-fallback picker chain and preserving `/skills`, `$name`, trust, refusal, install, and script behavior.

**Architecture:** Move the nine approved non-framework methods and the one assignable candidate cache into a focused controller with explicit late-bound callbacks. Keep the registered `/skills` handler and two Textual decision handlers on `ChatScreen` as bounded delegates. Delete the four unreachable fallback/picker methods, picker widget, CSS, obsolete tests, and stale runtime documentation instead of extracting them.

**Tech Stack:** Python 3.11+, Textual 8, pytest/pytest-asyncio, Ruff, stdlib AST checks.

---

**ADR required:** no

**ADR path:** N/A

**Reason:** This is the behavior-preserving implementation of the skill ownership and dead-surface removal already approved in `Docs/superpowers/specs/2026-08-13-console-decomposition-wave6-design.md`; it changes no storage, trust policy, security boundary, dependency, or public service contract.

## Constraints and evidence rules

- Preserve the approved live inventory exactly: nine moved methods, three at-most-five-line screen delegates, one read/write compatibility descriptor, and four deleted unreachable methods.
- `ConsoleSkillController` must not query the DOM, push a screen, call `run_worker`, or reach a sibling controller through `ChatScreen`.
- Context fetches remain fresh and fail closed to an empty mapping; candidate filtering stays user-invocable, non-blocked, stable, and case-fold sorted.
- Pending install/script updates replace only their named `TaskResumeState` field; request IDs and event decisions pass through unchanged.
- Delete the generic-dispatch `KIND_FALLBACK` branch only from `ChatScreen`; the grammar extension point remains available to other callers.
- Delete the unused skill picker rather than retaining an unregistered product path. Do not replace it with a new picker, alias, compatibility shim, or controller abstraction.
- Regenerate the modular CSS bundle from source after removing the picker block; never hand-edit generated output.
- Run only tests related to touched files and functionality, per the user's standing instruction; do not run the full repository suite.
- After every mutation, restore the candidate bytes exactly before continuing.

## File map

- Create `tldw_chatbook/UI/Console_Modules/skill.py`: candidate state, fresh context retrieval, trust/block projection, `/skills` policy, refusal rows, pending state updates, and decision forwarding.
- Modify `tldw_chatbook/UI/Console_Modules/wiring.py`: construct `_skill` with named late-bound screen/app dependencies and update the controller count/order contract.
- Modify `tldw_chatbook/UI/Screens/chat_screen.py`: add the `_console_skill_candidates` descriptor, retain three bounded delegates, route live callers to `_skill`, and delete the fallback/picker chain and imports.
- Create `Tests/UI/test_console_skill_controller.py`: isolated no-mount controller tests.
- Modify `Tests/Architecture/test_console_wave6_inventory.py`: exact skill ownership/delegate/deletion/descriptor/DOM contracts and non-vacuity.
- Modify `Tests/UI/test_console_controller_wiring.py`: add only a focused `_skill` construction/class and late-bound dependency test; preserve the existing six-controller `_EXPECTED_SLOTS`, order, and shared-accessor characterization unchanged.
- Modify the focused live behavior tests under `Tests/UI/test_console_skill_commands.py`, `Tests/UI/test_console_skill_install_confirm.py`, `Tests/UI/test_skill_script_confirm_card.py`, `Tests/UI/test_console_visit_dispatch_dedupe.py`, `Tests/UI/test_console_command_popup.py`, and `Tests/ProductionApp/test_chat_root_state_removal.py` only where ownership changed.
- Delete `tldw_chatbook/Widgets/Console/console_skill_picker_modal.py` and `Tests/UI/test_console_skill_picker.py`.
- Modify `Tests/UI/test_console_modal_dismissal.py`, `tldw_chatbook/css/components/_agentic_terminal.tcss`, and regenerated `tldw_chatbook/css/tldw_cli_modular.tcss` to remove the picker surface.
- Rewrite stale live-code references in `tldw_chatbook/Chat/console_skill_resolver.py`, `tldw_chatbook/Widgets/Console/console_style_picker_modal.py`, and `Tests/Chat/test_console_style_picker.py`; leave historical plans/specs that document the migration intact.
- Rewrite moved-owner precedent references in `tldw_chatbook/Chat/console_agent_bridge.py`, `tldw_chatbook/Chat/console_chat_controller.py`, and `tldw_chatbook/UI/Screens/chat_screen_state.py` so live documentation points to `ConsoleSkillController`, not removed `ChatScreen` methods.
- Modify the TASK-3070.6 backlog file for plan linkage and final evidence.

### Task 0: Rebase and record the immutable focused baseline

- [ ] Commit only the reviewed task/plan records, fetch `origin/dev`, rebase this isolated branch, and prove the worktree contains no implementation change before baseline execution.
- [ ] Record exact results for the existing skill command, install/script decision, command-popup, visit-refresh, root-state, resolver, substitution, modal-dismissal, CSS-sync, and Wave 6 inventory tests before any production edit:

  ```bash
  ../../.venv/bin/python -m pytest -q \
    Tests/UI/test_console_skill_commands.py \
    Tests/UI/test_console_skill_install_confirm.py \
    Tests/UI/test_skill_script_confirm_card.py \
    Tests/UI/test_console_visit_dispatch_dedupe.py \
    Tests/UI/test_console_command_popup.py \
    Tests/ProductionApp/test_chat_root_state_removal.py \
    Tests/Chat/test_console_skill_resolver.py \
    Tests/Chat/test_console_skill_substitution.py \
    Tests/UI/test_console_modal_dismissal.py \
    Tests/Chat/test_console_style_picker.py \
    Tests/UI/test_css_bundle_sync_guard.py
  ../../.venv/bin/python -m pytest -q Tests/Architecture/test_console_wave6_inventory.py
  ```
- [ ] Record `chat_screen.py` line/direct-method counts and the current diagnostic non-write result so inherited failures remain distinguishable from TASK-3070.6 changes.
- [ ] Stop before implementation if the rebased source no longer matches the approved nine-M/three-D/four-X inventory.

### Task 1: Lock ownership and deletion requirements with RED tests

- [ ] Add `Tests/UI/test_console_skill_controller.py` with plain async fakes/call recorders and no Textual mount.
- [ ] Assert default candidate state, fresh context success/failure, trusted candidate filtering/order, blocked exact/prefix behavior, `/skills` list/hint copy, and pending install/script field isolation.
- [ ] Extend the Wave 6 architecture inventory so all nine M methods exist only on `ConsoleSkillController`; all three D methods remain on `ChatScreen` with their framework binding and at-most-five-line spans; the candidate descriptor targets `_skill`; the controller has no DOM/sibling reach-through.
- [ ] Add deletion assertions for the four X methods, `KIND_FALLBACK` screen dispatch, picker module/import/test/CSS selectors, and stale fallback/picker runtime references across the new `skill.py`, `chat_screen.py`, resolver, and style-picker files; scan all production Python sources for removed `ChatScreen` skill-policy ownership tokens.
- [ ] Add a non-vacuity mutation fixture proving the deletion oracle catches a synthetic fallback/picker reference.
- [ ] Run the new controller and architecture nodes and confirm RED is caused by absent ownership and still-present dead surface.

### Task 2: Move the live skill policy into the controller

- [ ] Create `ConsoleSkillController` with `app_instance` plus named callbacks for transcript append, command-popup sync, task-resume-state read/write, and current chat-controller lookup.
- [ ] Initialize `_console_skill_candidates` in the controller.
- [ ] Move the nine M methods without changing user-visible copy, filtering, ordering, exception containment, or field-replacement semantics; rewrite their internal documentation so it no longer names the deleted fallback/picker chain.
- [ ] Add controller bodies for the three D entry points: registered `/skills`, install decision, and script decision.
- [ ] Construct `screen._skill` in `build_console_controllers`; add a focused wiring test for the concrete controller type and late-bound callbacks, update only `wiring.py`'s full-graph count/order documentation, preserve `test_console_controller_wiring.py`'s original six-controller `_EXPECTED_SLOTS`/order/shared-accessor contract unchanged, and expose no screen-module class re-export.
- [ ] Add `_ControllerState("_skill", "_console_skill_candidates")` and remove the screen's eager default assignment.
- [ ] Repoint mount/resume refreshes, unknown-command blocked checks, popup candidate reads, and chat-controller pending callbacks directly to `_skill`.
- [ ] Reduce each D screen entry point to plain-value extraction/event stop plus one controller call.
- [ ] Run the isolated controller, architecture, wiring, command, popup, visit, pending-card, and root-state nodes to GREEN.

### Task 3: Delete the unreachable fallback/picker surface

- [ ] Remove `KIND_FALLBACK` from the screen import and command-send/dispatch branches.
- [ ] Delete `_console_skill_search`, `_console_command_run_skill`, `_run_resolved_console_skill`, and `_open_console_skill_picker`.
- [ ] Delete `_CONSOLE_SKILL_SEARCH_LIMIT` and imports/constants used only by that picker/fallback chain; prove remaining imports still have live callers and that neither the moved controller nor the screen retains picker/fallback documentation.
- [ ] Delete the picker module and its dedicated test file; remove its modal-dismissal contract/import and update the exact contract cardinality.
- [ ] Remove the picker CSS source block and regenerate the modular bundle using `python tldw_chatbook/css/build_css.py`; verify with `python tldw_chatbook/css/check_bundle_sync.py`.
- [ ] Delete obsolete dead-path tests (including the direct resolved-picker send test) while preserving all live `/skills`, `$name`, blocked/refusal, command-popup, and modal contracts.
- [ ] Rewrite resolver/style-picker source/test documentation and moved-owner precedent references so no live code names the deleted picker, dead slash fallback, or removed `ChatScreen` skill-policy methods.
- [ ] Run the deletion architecture nodes, CSS sync/build integrity nodes, modal-dismissal contract, resolver tests, style-picker tests, and live skill tests to GREEN.

### Task 4: Prove behavior and mutation sensitivity

- [ ] Run the focused behavior matrix recorded in the baseline: skill commands, install/script confirmation, command popup, visit refresh, root-state pending bridge, resolver, substitution, modal dismissal, plus the new controller/architecture tests.
- [ ] Mutate fresh-context retrieval to reuse cached candidates; confirm the freshness/refusal test fails, then restore.
- [ ] Remove the trust-block filter or stable sort; confirm the isolated projection test fails, then restore.
- [ ] Replace the whole resume state or update the wrong pending field; confirm the field-isolation test fails, then restore.
- [ ] Remove each of the three screen delegations in turn; confirm the delegate/behavior test fails, then restore.
- [ ] Reintroduce one dead picker/fallback token in a synthetic fixture; confirm the deletion oracle fails, then restore.
- [ ] Measure `chat_screen.py` line/method change without changing the ratchet ceiling; stop if the approved inventory no longer produces the expected reduction.

### Task 5: Static checks, review, and closeout

- [ ] Run Ruff lint and format checks on changed Python files only; do not bulk-format unrelated inherited drift.
- [ ] Run `py_compile` for `skill.py`, `wiring.py`, and `chat_screen.py` under one validated temporary pycache root, then remove only that root.
- [ ] Run CSS bundle sync, `git diff --check`, the metadata-only diagnostic architecture node, and the persistent-diagnostic checker; update the inventory only for a proven owner move with unchanged sink topology.
- [ ] Review the cumulative diff for one-controller/YAGNI scope: no new dependency, picker replacement, storage/config change, trust-policy change, DOM ID addition, or user-visible copy change.
- [ ] Obtain independent spec/correctness and minimality review if collaboration policy permits; address validated findings one at a time with focused tests.
- [ ] Update plan checkboxes and TASK-3070.6 ACs/Implementation Notes with exact RED/GREEN/mutation/static evidence, ADR decision, modified/deleted files, inherited exceptions, and no-full-suite constraint.
- [ ] Commit the final candidate, rebase onto latest `origin/dev`, rerun the same touched-functionality gates, push, open one atomic PR against `dev`, address review findings, and merge only after final verification.
