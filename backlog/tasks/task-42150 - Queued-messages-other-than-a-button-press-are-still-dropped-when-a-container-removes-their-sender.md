---
id: TASK-42150
title: "Queued messages other than a button press are still dropped when a container removes their sender"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 20:06
updated_date: 2026-11-21 21:30
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
- [x] #4 Handlers that read a position from the sender's id, and whose sender sits below a child that a rebuild removes (so the keeper can hand them a press from an earlier rebuild), resolve their target from the sender instance of the current rebuild; each destructive or state-changing one has a pilot that fails without the change, and each non-destructive one (navigation, rerun) is recorded with the reason
<!-- AC:END -->

AC#4 added 2026-11-21 (coordinator decision, review r19rc-rev1): round 18's TASK-42014 exposed these handlers on the recompose route, and this task's swap routes widen the exposure.

Coordinator note 2026-11-21 (review r19rc-rev1): `Select.Changed`, `Checkbox.Changed` and `Switch.Changed` are not kept. A mount echo or a programmatic `value =` posts the same message, so forwarding one from a removed generation could deliver a stale echo as if the user had acted. Keeping them needs a user-origin signal; they stay listed under follow-ups. The keeper keeps `Button.Pressed`, `Input.Submitted` and `ListView.Selected`.

## Implementation Plan

1. Pin the mechanism for each candidate message: where Textual posts it (the sender's own queue) and whether a removal whose closing parent sits above the sender drops it, by a held pilot per message type and per swap route, red on dev.
2. Inventory, with evidence: which user-action message types the app handles above their sender (static scan of `@on(...)`/`on_*` handlers), and which removal routes (recompose, `remove_children()`/`remove()` then `mount()`) hold such a sender below a removed child (a census of real removals from the app's own UI tests).
3. Move the keeper from `Widget.recompose` to the one function every removal goes through (`App._prune`), and widen it from `Button.Pressed` to the message types the inventory keeps, with the TASK-42014 rules (nearest live ancestor, never the App or past the exit fence, never twice, own-selector handlers only).
4. Drop what the move makes redundant (the recompose wrapper and its lock, the two explicit calls, the override scan), keeping the lock-order pilot green.
5. Measure the hook's cost on hot removal paths and the boot budget; run the covering tests, ratchets, Tests/Architecture serially, preflight and the ruff gates.

Review r19rc-rev1 (2026-11-21):
6. Narrow the keeper to `Button.Pressed`, `Input.Submitted` and `ListView.Selected`; drive the message pilots with real input (a click, Enter in an Input, Enter on a ListView); add a negative pilot: a mount echo below a removed child is not forwarded.
7. Resolve the target of an index-from-id handler from the sender instance of the current generation: a shared `rendered_target(targets, sender)` helper next to the keeper, and per-widget button maps. First `BenchEditor` (remove and add), with a pilot from the reviewer's probe, red first.
8. Audit the remaining index-from-id handlers below a rebuilt child: fix each destructive or state-changing one by identity with a pilot, and record each navigation-only one with the reason.
9. Reword the lock-order pilot's docstring; add the identity rule to the TASK-41703 lesson.

## Implementation Notes

One hook now covers every removal. The keeper moved from `Widget.recompose` to `App._prune`, which every removal goes through (a recompose's `query(...).remove()`, `remove_children()`, `remove()`). Besides `Button.Pressed` it keeps the two other messages that only the user's own input posts: `Input.Submitted` and `ListView.Selected` (narrowed from five in review r19rc-rev1). Handlers that read a row's position from a button id now resolve the row from the button instead (AC#4).

**Mechanism.** The wrapped `App._prune` runs native `_prune`, then `tldw_chatbook.keep_queued_messages(nodes)`. The call is synchronous: the nodes are already marked `_pruning` and their `Prune` posted, but no removed node has handled a message yet, so no lock is needed.
- For each removed node whose parent stays, it walks the subtree for the three senders (`Button`, `Input`, `ListView`). It skips a node whose parent is the App (a removed screen) or is itself pruning (walked with that parent).
- It wraps `_on_message` on every node from the sender up to the removed node, that is, every node whose parent is going.
- After a wrapped node's own dispatch, an unstopped kept message (`Button.Pressed`, `Input.Submitted`, `ListView.Selected`) whose parent is closing is bubbled to the nearest ancestor that is not pruning.
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

**AC#2: verdicts.** `Input.Submitted` and `ListView.Selected`, on every route, keep the message: it goes to the nearest staying ancestor, as TASK-42014 does for presses. `Select.Changed`, `Checkbox.Changed` and `Switch.Changed` are not kept (coordinator decision, review r19rc-rev1). A mount echo or a programmatic `value =` posts the same message. The reviewer's probe, now `test_a_mount_echo_below_a_removed_child_is_not_forwarded`, showed a removed Select's echo reaching the screen as if the user had picked it: `[(1, False), (2, True)]`. Keeping them needs a user-origin signal. Recorded as safe, with the reason:
- A sender that is a direct child of the remover: its parent stays open, so it bubbles natively.
- A node inside a removed screen: the only ancestor left is the App, which the keeper never targets. `test_a_press_queued_in_a_popped_screen_never_reaches_the_app` pins this.
- A handler inside the removed subtree: it is being removed with its widget.

Handler audit for the new types, since a forwarded message reaches its handler after the rebuild with the sender detached:
- A message that had already bubbled past the removed child before the removal reaches these same handlers on dev, in the same post-rebuild state, so the keeper adds no new kind of delivery.
- One new-type handler reaches through its sender's tree: `PersonasPolicyRulesEditor._rule_selected` (`event.list_view.children.index`). A detached list raises `ValueError`, which it catches.
- Two kept-type handlers resolve their target by position. `PersonasPolicyRulesEditor._rule_selected` is covered above. `AgentsSettingsPanel.on_list_view_selected` is positional, but the panel refills its lists with `ListView.clear()`/`extend()`, which keep the list widget. No `Input.Submitted` handler reads a position.
- Press handlers that decode a position from the button id are AC#4 below.

**AC#3: pilots** (`Tests/Library/test_library_notes_canvas_pending_lookup.py`, a whole-file target of the fast PR contract in `derived-artifacts.yml`, so no census edit).
- `test_every_removal_route_keeps_a_press_queued_below_a_removed_child` (renamed from `..._recompose_route_...`) gains `remove-children-then-mount` and `remove-then-mount`. Each swap runs on the container's own pump, as the container's own method would.
- `test_a_user_message_queued_below_a_removed_child_is_kept` covers `Input.Submitted` and `ListView.Selected` across a `refresh(recompose=True)`. Each holds the message in its sender until the toolbar above it is closing, and asserts that the screen's `#opener` handler ran exactly once, for the old sender.
- Since review r19rc-rev1 these pilots are driven by real input. The route pilots and the external-model pilot use `pilot.click`; the message pilots and the Speech panel pilot use Enter on the focused sender. Without the app stylesheet, the Speech panel's row covers its button in a mouse hit test. The helper queues the hold from inside the action the input runs (`press`, `action_submit`, `action_select_cursor`), just ahead of the message.
- `test_a_mount_echo_below_a_removed_child_is_not_forwarded` (rev1) holds a Select's mount echo below a removed child. With the keeper still forwarding `Select.Changed` it was red (`[(1, False), (2, True)] == [(2, True)]`); it passes once the three `Changed` types are left out.
- `test_a_press_queued_in_a_popped_screen_never_reaches_the_app` pins the App guard on the new hook.
- `test_every_removal_reaches_the_message_keeper` replaces the override scan: it asserts that `App._prune` carries the marker.
- Red on dev `dc371f0842` (tests only): the 7 new pilots failed on their assertions (`the remove-children-then-mount route dropped the press`, `assert 0 == 1`, and `assert [] == ['Input.Submitted']` and the like). The popped-screen pilot is an invariant dev keeps. With both App guards removed (temporary edit, reverted) it failed with `assert 1 == 0`.
- Keeper off (install returned early, temporary): 19 of the file's 26 failed, all the press and message pilots plus the marker pin. Green: 26 of 26, then 3 more runs at `-n 4` (load 46). After rev1's narrowing and real-input switch: keeper off, 16 of the then 24 failed (every press and message pilot, the external-model and Speech pilots, the pin).
- Serial, the fast lane's way (`-p no:xdist`), the file takes 47.8 s at load 60 (TASK-42014 measured 10.1 s at load 21-23). No new test is in its top five.

**AC#4 (review r19rc-rev1): handlers that read a position from a button id.** A scan of every `Button.Pressed` handler that reads a number from the button id, plus the kept-type handlers that read a list index, gave the list below. For each, I checked whether the button sits below a child that a rebuild removes, which is what lets the keeper hand on a press from an earlier rebuild. Such a press also reaches the handler late on dev when it had already left the row.

Fixed by identity. `tldw_chatbook.rendered_target(targets, sender)`, next to the keeper, returns what the sender was rendered for, or None for a sender from an earlier rebuild or one being removed. Each builder keeps a new map per rebuild. Each fix has a pilot in the lane file, red before the fix:
- `BenchEditor` Remove (`_on_remove_target_pressed`) and Add (`_on_add_target_pressed`, which read whichever picker was current), across the targets section's `remove_children()` swap.
  - `test_a_stale_bench_target_remove_never_unstages_another_target` was red: `['warned'] == ['warned', 'blocked']`. The pilot is the reviewer's probe.
  - `test_a_stale_bench_target_add_never_stages_the_rebuilt_pickers_default` was red: it staged "spare-b", the rebuilt picker's default, where the user had picked "spare-a".
- `BriefingPresetModal` speaker Remove (`bpm-speaker-remove-{index}`), across the modal's recompose: `test_a_stale_briefing_speaker_remove_never_removes_another_speaker` was red (`['B'] == ['B', 'C']`).
- `ConsoleHooksReviewModal` Revoke and Disable (`hook-review-{action}-{index}`), across `#console-hooks-list`'s recompose. `_mutate` now takes the key rather than indexing. `test_a_stale_hook_revoke_never_revokes_another_hook` was red: it revoked "c".
- `PersonalContextReviewModal` Apply edit (`personal-context-review-apply-{index}`), across the modal's recompose: `test_a_stale_personal_context_apply_never_rewrites_another_change` was red: it rewrote "change-1".

Recorded, not changed:
- Navigation, view or copy only:
  - `MCPServersMode` callout (`mcp-callout-{index}` selects a server row);
  - `WatchlistTree` tag (`wl-tree-tag-{index}` toggles the tree's tag filter);
  - `InspectorPane` breadcrumb;
  - `ConsoleTerminalWorkspace` session select;
  - `ConsoleWorkspaceFilesModal` binding and file (they load a preview);
  - `ConsoleCitationSourcesModal._open_source`;
  - the hook review's Details toggle;
  - `PersonalContextSettingsPanel` proposal, record and deleted rows (each opens or selects; any write takes a further explicit step);
  - `LibraryIngestCanvas._copy_preflight_install_command` (copies to the clipboard).
- Stable ids, not positions: `ArtifactsScreen` briefing ids, `KeptBriefingsModal` (`kbm-kept-btn-{kept_id}`), `BriefingPresetModal` presets (`bpm-preset-btn-{preset_id}`), `WatchlistTree` watchlist and source nodes, and `StudyWindow` ratings (the id carries the rating value).
- Never rebuilt, so no press outlives its row: the OPML dialogs, and `ProjectInstructionSetupModal` (its options are fixed at construction).
- The target travels on the button: `LibraryNotesAddFromFilesCanvas` (review root, observation token and binding id are button attributes).
- Never runs for a forwarded press: `WorkspaceCreateModal._remove_folder` selects through an ancestor (TASK-42014).
- Never forwarded, because its rows are direct children of the list it swaps: `ConsoleReactionPickerModal._row_pressed`. A press that reaches it after a re-filter still resolves by position; proposed follow-up.

**Cost.** The keeper's own time per removal is measured with a probe that recomposes a container of N rows, each `Horizontal(Static, Button, Button, Checkbox, Input)`, 20 times. At 120 nodes it takes 0.10 ms. At 1200 nodes it takes 2.60 ms, and 0.86 ms when the rows hold Statics only; TASK-42014's Button-only walk took 0.44-0.62 ms on the same tree. The whole recompose took 0.7-3.1 s under load 34-60, so the keeper is about 0.1-0.4% of a removal. A removed screen skips the walk. Boot budget (`bootbudget.sh`, the CI step's seven files, serial): 34 passed; module counts, headroom and the ui-ready snapshot drift (+1/-0) match round 18's logs, and preimport LOC is 2 lower.

**Runs** (all through the wrapper; the machine was at load 34-65 throughout).
- Covering set at `-n 4`: the lane file, the Settings region, Console tray, rail search, workspace-context rail, tray-guard, Library canvas-sync, notes-navigator, phase-C residency, focus-on-removed, roleplay item-list, external-view, Speech panel, plugin review, sidebar debounce, question card, bench editor, MCP servers mode and Watchlists rebuild files, the four Textual-shim files, and `Tests/Widgets/Library`. Head: 250 failed, 983 passed.
- The same 250 nodes with the four source files checked out from dev `dc371f0842`: 248 failed. At least 174 of the head failures are the local `RecoveryRequired` trap.
- The two nodes that passed at dev are in `test_evals_bench_editor.py`, which is flaky under this load on both sides. More runs of that file, three of each pair interleaved: at `-n 4`, dev failed 0 to 12 tests in six runs and head 0 to 29 in thirteen; serially, dev failed 2, 2 and 1 and head 0, 6 and 0. Every failure is a different test, mostly `NoMatches` on an inspector node, and the slower side alternates (serial 108 s dev against 131-136 s head in one pair, 121 s dev against 88 s head in the last).
- The keeper is not the cause. A temporary probe logged one forwarded message per run of that file: the mount echo of `#evals-bench-add-target` to `#lab-body`, which nothing handles, in a test that passed. A timing probe put the keeper at 2.6-5.8 ms per worker for the whole file.
- `Tests/Architecture`, serial: 3 failed and 1489 passed. The three (`test_reviewed_diagnostic_changes_are_metadata_only` on `agent_service.py`, and two `test_timer_path_static_update_inventory.py` nodes on `project_index_binding_card.py`) also fail at dev with the same command. That run includes the module-size and code-identity ratchets.
- Preflight passes. `ruff check --select F` on the touched files and `Tests/lint/ruff_gate.py` pass.

**Runs after review r19rc-rev1** (load 18-25).
- The lane file: 29 passed in each of 5 runs at `-n 4`.
- The touched widget files at `-n 4`: the bench editor, authoring e2e, briefing presets, personal-context review, the five hook-review files and the external-model view. 290 passed, 11 failed:
  - the 4 `open_briefing_preset_manager` tests and the authoring e2e test are the local `RecoveryRequired` trap;
  - the 5 hook-review tests fail with dev's source files too, the same 5 (checked by an A/B run);
  - `test_external_view_keeps_actions_and_status_reachable_at_80_columns` failed at dev in the first round's comparison.
  - The whole bench editor file passed in that run.
- Boot budget: 33 passed, 1 failed. `test_boot_worker_and_thread_starts_stay_within_the_allowlist` (the `console-skill-trust-setup` worker) failed 2 of 2 with dev's source files as well; it passed in round 1's run, so it is timing-dependent. Module counts and headroom are unchanged. Preimport LOC is 423431, 13 over round 1, from `bench_editor.py`'s map and the `rendered_target` helper.
- `ruff check --select F` on every touched Python file and `Tests/lint/ruff_gate.py` pass.
- `Tests/Architecture`, serial: 3 failed and 1489 passed, the same three dev-red nodes as round 1 (module-size and code-identity ratchets included).
- Preflight passes.

**Size.** `tldw_chatbook/__init__.py` 590 -> 626 (no row; 609 before rev1); `bench_editor.py` 1557 -> 1560, `briefing_preset_modal.py` 685 -> 695, `console_hooks_review_modal.py` 698 -> 714, `personal_context_review_modal.py` 938 -> 944 (no rows); `settings_screen.py` 34980 -> 34978 (budget 34993); `console_workspace_context.py` -2 (no row); `library_canvas_sync.py` unchanged in lines.

Files: `tldw_chatbook/__init__.py`, `tldw_chatbook/UI/Screens/settings_screen.py`, `tldw_chatbook/Widgets/Console/console_workspace_context.py`, `tldw_chatbook/Widgets/Library/library_canvas_sync.py` (docstring), and since rev1 `tldw_chatbook/UI/Evals/bench_editor.py`, `tldw_chatbook/UI/Watchlists_Modules/briefing_preset_modal.py`, `tldw_chatbook/Widgets/Console/console_hooks_review_modal.py`, `tldw_chatbook/Widgets/Settings_Widgets/personal_context_review_modal.py`; `Tests/Library/test_library_notes_canvas_pending_lookup.py`; `backlog/docs/lessons-textual.md` (the TASK-41703 entry: where the keeper lives, why the three `Changed` types are not kept, the swap-route incident, the identity rule, the real-input hold, the falsy unmounted `ListView`).

**Not this task's list.** `Select.Changed` 170, `Checkbox.Changed` 65 and `Switch.Changed` 13 handlers above the sender are left out by the r19rc-rev1 decision; they need a user-origin signal. The same scan finds other actions handled above their sender: `RadioSet.Changed` 20, `OptionList.OptionSelected` 30, `DataTable.RowSelected` 23 and `CellSelected` 7, `Tree.NodeSelected` 6, `SelectionList.SelectedChanged` 5, `Collapsible.Toggled` 12, `TabbedContent.TabActivated` 1, `RadioButton.Changed` 1. The mechanism drops them the same way; keeping one is a tuple entry and a pilot. A custom message that a widget posts from its own press handler is dropped too, when that widget sits below a removed child. `Input.Changed` and `TextArea.Changed` echo a removed widget's own text, which goes with it. These are proposed as follow-ups, not kept here. Also proposed: `ConsoleReactionPickerModal._row_pressed` resolves a late press by position (AC#4); handlers on index-id Checkboxes (`PersonalContextReviewModal.handle_checkbox_changed`, `ConsolePromptMixinsView._toggle`) still resolve a late `Changed` by position, though the keeper no longer forwards one; and a static-id action (Delete, Save) acts on whatever is current when a press from an earlier rebuild lands.
