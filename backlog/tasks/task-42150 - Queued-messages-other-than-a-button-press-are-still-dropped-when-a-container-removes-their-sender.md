---
id: TASK-42150
title: "Queued messages other than a button press are still dropped when a container removes their sender"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 20:06
updated_date: 2026-11-21 20:04
labels:
- follow-up
- ui
- textual
priority: low
references:
- tldw_chatbook/__init__.py
- Tests/Library/test_library_notes_canvas_pending_lookup.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by review of the library stream's TASK-42014 (round 18). TASK-42014 (round-18 library branch, not yet on dev) makes every recompose hand a queued `Button.Pressed`, whose sender is being removed, to the nearest ancestor that stays (`keep_queued_presses` in `tldw_chatbook/__init__.py`). Two cases remain:
- other user messages queued the same way are still dropped: `Input.Submitted`, `Checkbox.Changed` and `Switch.Changed`, `Select.Changed`, `ListView.Selected`;
- a press is still dropped when a container swaps its children without recomposing (`remove_children()` then `mount()`); the keeper runs on the recompose route and in the two overrides TASK-42014 patched.

Each loses a user action that arrived just before the rebuild. This builds on TASK-42014's keeper.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The notes list which message types and which child-swap routes (recompose, `remove_children()` then `mount()`) can drop a queued user message, and where each occurs in the app
- [x] #2 Each listed case either keeps its queued message (delivered to the nearest staying ancestor, as TASK-42014 does for presses) or is recorded as safe with the reason
- [x] #3 For each case that keeps its message, a pilot fails without the change
<!-- AC:END -->

## Implementation Plan

1. Pin the mechanism for each candidate message: where Textual posts it (the sender's own queue) and whether a removal whose closing parent sits above the sender drops it, by a held pilot per message type and per swap route, red on dev.
2. Inventory, with evidence: which user-action message types the app handles above their sender (static scan of `@on(...)`/`on_*` handlers), and which removal routes (recompose, `remove_children()`/`remove()` then `mount()`) hold such a sender below a removed child (a census of real removals from the app's own UI tests).
3. Move the keeper from `Widget.recompose` to the one function every removal goes through (`App._prune`), and widen it from `Button.Pressed` to the message types the inventory keeps, with the TASK-42014 rules (nearest live ancestor, never the App or past the exit fence, never twice, own-selector handlers only).
4. Drop what the move makes redundant (the recompose wrapper and its lock, the two explicit calls, the override scan), keeping the lock-order pilot green.
5. Measure the hook's cost on hot removal paths and the boot budget; run the covering tests, ratchets, Tests/Architecture serially, preflight and the ruff gates.

## Implementation Notes

One hook now covers every removal. The keeper moved from `Widget.recompose` to `App._prune`, which every removal goes through (a recompose's `query(...).remove()`, `remove_children()`, `remove()`). It keeps five more user actions besides `Button.Pressed`.

**Mechanism.** The wrapped `App._prune` runs native `_prune`, then `tldw_chatbook.keep_queued_messages(nodes)`. The call is synchronous: the nodes are already marked `_pruning` and their `Prune` posted, but no removed node has handled a message yet, so no lock is needed.
- For each removed node whose parent stays, it walks the subtree for the six senders (`Button`, `Input`, `Checkbox`, `Switch`, `Select`, `ListView`). It skips a node whose parent is the App (a removed screen) or is itself pruning (walked with that parent).
- It wraps `_on_message` on every node from the sender up to the removed node, that is, every node whose parent is going.
- After a wrapped node's own dispatch, an unstopped kept message (`Button.Pressed`, `Input.Submitted`, `Checkbox.Changed`, `Switch.Changed`, `Select.Changed`, `ListView.Selected`) whose parent is closing is bubbled to the nearest ancestor that is not pruning.
- TASK-42014's rules are unchanged: never to the App and never past the exit fence (`_closing`, `is_attached`); never twice (it forwards only when native did not bubble); and once the sender detaches, only handlers whose selector the sender matches on its own run.
- Removed as redundant: the `Widget.recompose` wrapper and its lock, `keep_queued_presses`, its calls in `SettingsRegion.recompose` and the Console tray's kept-row rebuild, and the AST override scan. A recompose override can no longer remove a child without going through `_prune`.

**AC#1: which messages and routes drop a queued user action, and where.** Each of these messages is posted into its sender's own queue (`press()`, `action_submit()`, a `value` watcher, `action_select_cursor()`) and then bubbles one node at a time. Textual removes a subtree top down, so a node below a removed child that handles the message after its parent started closing does not bubble it (`MessagePump._on_message`: `is_parent_active`). A sender that is a direct child of the remover is safe, because its parent stays.
- *Handlers above the sender* (static scan of `@on(...)` and `on_*` methods in `tldw_chatbook`; a handler on the sender's own class would run anyway): `Button.Pressed` 1916, `Input.Submitted` 101, `Checkbox.Changed` 65, `Switch.Changed` 13, `Select.Changed` 170, `ListView.Selected` 12.
- *Recompose route.* TASK-42014's table lists the 65 or so recomposing containers; the message pilots below run on its `refresh` route. A run-time census confirms it: a temporary pytest plugin (never committed) wrapped `App._prune` and recorded each removal that held a sender below a removed child. Over 102 tests in 17 widget-level files (`Tests/Widgets` and a sample of `Tests/UI`, stopped early for machine load) it recorded 1126 such removals. Inputs sat below a removed child in 125 recomposes of five Library containers: `LibraryNotesCanvas`, `LibraryScreen`, `LibraryRail`, `LibraryMediaCanvas` and `LibraryNoteWorkPane`.
- *Swap routes* (`remove_children()` or `remove()`, then `mount()`, outside a recompose). A static scan finds 143 methods that remove and mount. The ones that build a sender below the removed child (directly nested or inside a package widget's own compose, one hop into same-class helpers): `Button` 38, `Input` 23, `Select` 14, `Checkbox` 10, `Switch` 1, `ListView` 1. Examples with a handler above the sender:
  - `Input.Submitted`: `ChatQuestionCard._paint` (`#question-sections` swap; `on_input_submitted` on the card submits it).
  - `Checkbox.Changed`: `MCPServersMode.show_server_mutations` (`on_checkbox_changed` on the mode).
  - `Select.Changed`: `StudyWindow.watch_current_view` and `STTSWindow._mount_view_unlocked`.
  - `ListView.Selected`: `StudyWindow.watch_current_view` (`content_container.remove_children()`; `#card-list` handled on the window).
  - `Button.Pressed`: `BenchEditor._refresh_targets_section` and `ConsoleTranscript._reconcile_rows`.
  - `Switch.Changed`: the one swap site (`STTSWindow`) handles its switches inside the removed view, so it is safe there. Switches still sit below removed children on the recompose route, for example `SpeechTTSSettingsPanel` (`.settings-speech-draft-field`, handled on the panel) and the Watchlists `RulesPane` and `SourcesPane`.
- The census also saw the swap routes at run time. `RemoteView._refresh_details_with_status` removed an `Input`, a `Select` and Buttons below the removed child, and `_refresh_variant_list` and `_refresh_with_status` removed Buttons. `MCPInspector._hide_advanced` (`remove()`) removed a `Select`, and `LibraryScreen._project_library_canvas_child` removed Buttons and an `Input`. Every `Checkbox`, `Switch` and `ListView` row it saw came from a removal made by test code, so for those three the static scan is the evidence.

**AC#2: verdicts.** All five types, on every route, keep the message: delivered to the nearest staying ancestor, as TASK-42014 does for presses. Recorded as safe, with the reason:
- A sender that is a direct child of the remover: its parent stays open, so it bubbles natively.
- A node inside a removed screen: the only ancestor left is the App, which the keeper never targets. `test_a_press_queued_in_a_popped_screen_never_reaches_the_app` pins this.
- A handler inside the removed subtree: it is being removed with its widget.

Handler audit for the new types, since a forwarded message reaches its handler after the rebuild with the sender detached:
- A message that had already bubbled past the removed child before the removal reaches these same handlers on dev, in the same post-rebuild state, so the keeper adds no new kind of delivery.
- One new-type handler reaches through its sender's tree: `PersonasPolicyRulesEditor._rule_selected` (`event.list_view.children.index`). A detached list raises `ValueError`, which it catches.
- Six new-type handlers resolve their target by position. `ConsoleHooksReviewModal._selection_changed` returns for an unmounted checkbox. `ChatApprovalCard._on_batch_row_select_changed` returns for a select no longer in its list. `ResponseRulesModal.change_scope` reads `event.value` against a fixed list. `PersonasPolicyRulesEditor._rule_selected` is covered above. `ConsolePromptMixinsView._toggle` is positional, but it sits on the view itself: the view rebuilds only its editor area, and `ConsolePromptsModal._refresh_mixins` removes the whole view, so a forwarded toggle skips the handler (the modal's own checkbox handler selects `#console-prompts-include-system` only). `AgentsSettingsPanel.on_list_view_selected` is positional, but the panel refills its lists with `ListView.clear()`/`extend()`, which keep the list widget.
- Press handlers that decode an index from the button id (for example `BenchEditor._on_remove_target_pressed`, whose rows are rebuilt by `remove_children()`) get a queued press resolved against the rebuilt list. A press that had reached the editor before the rebuild already does that on dev. Proposed follow-up below.

**AC#3: pilots** (`Tests/Library/test_library_notes_canvas_pending_lookup.py`, a whole-file target of the fast PR contract in `derived-artifacts.yml`, so no census edit).
- `test_every_removal_route_keeps_a_press_queued_below_a_removed_child` (renamed from `..._recompose_route_...`) gains `remove-children-then-mount` and `remove-then-mount`. Each swap runs on the container's own pump, as the container's own method would.
- `test_a_user_message_queued_below_a_removed_child_is_kept` covers the five message types across a `refresh(recompose=True)`. Each holds the message in its sender until the toolbar above it is closing, and asserts that the screen's `#opener` handler ran exactly once, for the old sender.
- `test_a_press_queued_in_a_popped_screen_never_reaches_the_app` pins the App guard on the new hook.
- `test_every_removal_reaches_the_message_keeper` replaces the override scan: it asserts that `App._prune` carries the marker.
- Red on dev `dc371f0842` (tests only): the 7 new pilots failed on their assertions (`the remove-children-then-mount route dropped the press`, `assert 0 == 1`, and `assert [] == ['Input.Submitted']` and the like). The popped-screen pilot is an invariant dev keeps. With both App guards removed (temporary edit, reverted) it failed with `assert 1 == 0`.
- Keeper off (install returned early, temporary): 19 of the file's 26 failed, all the press and message pilots plus the marker pin. Green: 26 of 26, then 3 more runs at `-n 4` (load 46).
- Serial, the fast lane's way (`-p no:xdist`), the file takes 47.8 s at load 60 (TASK-42014 measured 10.1 s at load 21-23). No new test is in its top five.

**Cost.** The keeper's own time per removal is measured with a probe that recomposes a container of N rows, each `Horizontal(Static, Button, Button, Checkbox, Input)`, 20 times. At 120 nodes it takes 0.10 ms. At 1200 nodes it takes 2.60 ms, and 0.86 ms when the rows hold Statics only; TASK-42014's Button-only walk took 0.44-0.62 ms on the same tree. The whole recompose took 0.7-3.1 s under load 34-60, so the keeper is about 0.1-0.4% of a removal. A removed screen skips the walk. Boot budget (`bootbudget.sh`, the CI step's seven files, serial): 34 passed; module counts, headroom and the ui-ready snapshot drift (+1/-0) match round 18's logs, and preimport LOC is 2 lower.

**Runs** (all through the wrapper; the machine was at load 34-65 throughout).
- Covering set at `-n 4`: the lane file, the Settings region, Console tray, rail search, workspace-context rail, tray-guard, Library canvas-sync, notes-navigator, phase-C residency, focus-on-removed, roleplay item-list, external-view, Speech panel, plugin review, sidebar debounce, question card, bench editor, MCP servers mode and Watchlists rebuild files, the four Textual-shim files, and `Tests/Widgets/Library`. Head: 250 failed, 983 passed.
- The same 250 nodes with the four source files checked out from dev `dc371f0842`: 248 failed. At least 174 of the head failures are the local `RecoveryRequired` trap.
- The two nodes that passed at dev are in `test_evals_bench_editor.py`, which is flaky under this load on both sides. More runs of that file, three of each pair interleaved: at `-n 4`, dev failed 0 to 12 tests in six runs and head 0 to 29 in thirteen; serially, dev failed 2, 2 and 1 and head 0, 6 and 0. Every failure is a different test, mostly `NoMatches` on an inspector node, and the slower side alternates (serial 108 s dev against 131-136 s head in one pair, 121 s dev against 88 s head in the last).
- The keeper is not the cause. A temporary probe logged one forwarded message per run of that file: the mount echo of `#evals-bench-add-target` to `#lab-body`, which nothing handles, in a test that passed. A timing probe put the keeper at 2.6-5.8 ms per worker for the whole file.
- `Tests/Architecture`, serial: 3 failed and 1489 passed. The three (`test_reviewed_diagnostic_changes_are_metadata_only` on `agent_service.py`, and two `test_timer_path_static_update_inventory.py` nodes on `project_index_binding_card.py`) also fail at dev with the same command. That run includes the module-size and code-identity ratchets.
- Preflight passes. `ruff check --select F` on the touched files and `Tests/lint/ruff_gate.py` pass.

**Size.** `tldw_chatbook/__init__.py` 590 -> 609 (no row); `settings_screen.py` 34980 -> 34978 (budget 34993); `console_workspace_context.py` -2 (no row); `library_canvas_sync.py` unchanged in lines.

Files: `tldw_chatbook/__init__.py`, `tldw_chatbook/UI/Screens/settings_screen.py`, `tldw_chatbook/Widgets/Console/console_workspace_context.py`, `tldw_chatbook/Widgets/Library/library_canvas_sync.py` (docstring), `Tests/Library/test_library_notes_canvas_pending_lookup.py`, `backlog/docs/lessons-textual.md` (the TASK-41703 entry: where the keeper lives, the swap-route incident, the hold for non-press messages, the falsy unmounted `ListView`).

**Not this task's list.** The same scan finds other actions handled above their sender: `RadioSet.Changed` 20, `OptionList.OptionSelected` 30, `DataTable.RowSelected` 23 and `CellSelected` 7, `Tree.NodeSelected` 6, `SelectionList.SelectedChanged` 5, `Collapsible.Toggled` 12, `TabbedContent.TabActivated` 1, `RadioButton.Changed` 1. The mechanism drops them the same way; keeping one is a tuple entry and a pilot. A custom message that a widget posts from its own press handler is dropped too, when that widget sits below a removed child. `Input.Changed` and `TextArea.Changed` echo a removed widget's own text, which goes with it. These are proposed as follow-ups, not kept here.
