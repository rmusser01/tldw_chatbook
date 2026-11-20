---
id: TASK-42014
title: "Containers other than Library canvases drop a queued button press when they recompose"
status: Done
assignee: ['@claude']
created_date: 2026-11-19 19:39
updated_date: 2026-11-20 18:23
labels:
- follow-up
- ui
priority: low
references:
- tldw_chatbook/Widgets/Library/library_canvas_sync.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41703 (round 17). Textual removes a recompose's old children top down, so a queued `Button.Pressed` below a removed child is dropped. TASK-41703 forwards such presses only for `PostRecomposeCallback` canvases; any other container that recomposes while a child holds a queued press still drops it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The notes list which other recomposing containers can hold a queued press, and each one keeps it or is shown unable to
- [x] #2 Each recompose route a container can take has a pilot test that holds a press across it, red before the fix, and a structural test fails for any override in the package that removes children before calling the keeper
<!-- AC:END -->

AC#2 reworded on 2026-11-20 (coordinator decision). As filed it asked for a pilot per container; the eight route pilots plus the structural test meet its intent, so no per-container pilots are added.

## Implementation Plan

1. Pin the mechanism: which presses a recompose can drop (a press queued in a node whose parent the removal closes), and which it cannot (a Button that is a direct child of the recomposing container bubbles to a parent that stays open).
2. Inventory every recomposing container in `tldw_chatbook` (an AST scan for `refresh(recompose=True)`, `.recompose()` and `reactive(..., recompose=True)`), and classify each by whether it has Button descendants below a removed child.
3. Generalise TASK-41703's keeper so one shared mechanism covers every recompose rather than per-container copies.
4. Pilot tests that hold a press across each such container's recompose, red before the fix.

## Implementation Notes

One shared mechanism: TASK-41703's keeper now lives in `Widget.recompose` itself, so every container's recompose keeps a press queued below a child it removes, not only `PostRecomposeCallback` canvases.

**Mechanism.** `tldw_chatbook.keep_queued_presses(container)` (package `__init__.py`, next to the zero-width fold guard) is TASK-41703's `_keep_presses_through_the_removal`, generalised. Before the removal it wraps `_on_message` on each node between a Button and the container's own child. After the node's own dispatch, an unstopped `Button.Pressed` whose parent is closing is bubbled to the nearest ancestor that is not `_pruning`, which is the recomposing container. Two changes from the canvas version. (1) The target is that nearest live ancestor rather than a fixed canvas, so a container removed by its parent's recompose hands the press to the parent; it was dropped before, as the canvas version also dropped it. (2) Climbing continues past nodes that are already wrapped, so a row a rebuild keeps (the Console tray's filter row) is covered by the next removal too. A press is never forwarded past a removed screen to the App (`is_dom_root`) or past the exit fence (`is_attached`). `_install_recompose_press_keeper` wraps `Widget.recompose` once (marker `RECOMPOSE_PRESS_KEEPER`) from `_install_textual_compatibility_shims`, which every `tldw_chatbook.UI`/`Widgets` import runs. `PostRecomposeCallback.recompose` no longer calls its own copy (`library_canvas_sync.py` -43 lines). A Button that is a direct child of the container needed nothing: its parent stays open, so it bubbles normally, and the keeper wraps nothing for it.

Two overrides remove their children before (or without) `Widget.recompose`, so the hook ran too late for them: `SettingsRegion.recompose` (removes, records the focus reset, then `super()`) and the Console tray's kept-filter-row rebuild (`_KeepsConversationFilterRow.recompose`, `remove_children`, never `super()`). Each now calls `keep_queued_presses` before it removes. Every other `recompose` override in the package awaits `super().recompose()` before any removal of its own, directly or through a method of its own class (AST check, pinned below). `ConsoleBoundedSection.recompose` removes nothing.

**AC#1: the inventory.** An AST scan of `tldw_chatbook` for `refresh(recompose=True)`, `.recompose()` and `reactive/var(..., recompose=True)` found 290 sites, which resolve to the containers below. The Button column comes from a static scan of what each class constructs, transitively; injected builders were read by hand (`SettingsRegion`, `_SpeechSettingsCard`, the artifacts detail regions, `HomeCanvas`, the hooks list). Routes: **W** inherits `Widget.recompose`; **G** goes through `RecomposeCaptureGuard.recompose` then `super()`; **B** through `BaseAppScreen.recompose` then `super()`; **O** has its own override that awaits `super()` first; **S** `SettingsRegion`; **K** the Console tray's kept-row rebuild. Every container with Buttons keeps the press on its route, and the route's pilot shows it. A forwarded press reaches only the handlers whose selector the button matches on its own (an id or a class on the button); once the button is detached, `@on(Button.Pressed, "<ancestor> Button")` cannot match. Review r18lib-rev1 found three such handlers on containers in this table, and two were switched (marked below).

| Container | Route | Buttons | Verdict |
|---|---|---|---|
| `LibraryScreen` (its own refreshes, its controllers', `canvas_sync`'s screen fallbacks, unavailable-navigation, inspection admission, Parakeet install), `ChatScreen` (`wiring.py`), `ACPScreen`, `ArtifactsScreen`, `WorkflowsScreen` | B | yes | keeps it |
| `LabScreen`, `STTSScreen`, `SettingsScreen` | O, then B | yes | keeps it |
| `BriefingPresetModal`, `KeptBriefingsModal`, `DreamsStoryModal`, `PersonalContextReviewModal`, `ProjectSkillsImportModal`, `PromptCollectionManagerModal` | W | yes | keeps it |
| `WorkspaceCreateModal` | W | yes | keeps it, except `_remove_folder` (`#workspace-create-folder-list Button`), which drops it on purpose: the button's id is an index into the folder list, so a late press after the recompose could remove a different folder |
| `ConsoleHooksReviewModal`'s `#console-hooks-list` (`VerticalScroll` with the modal's `_rows` as compose) | W | yes, 3 deep | keeps it |
| Watchlists: `ArtifactsPane`, `BriefingDetailRegion`, `ScriptDetailRegion`, `ContentPane`, `InspectorPane`, `NotificationsPane`, `RulesPane`, `RunsPane` | G | yes | keeps it |
| Watchlists: `ItemsPane`, `SourcesPane` | O, then G | yes | keeps it |
| Watchlists: `WatchlistTree` | W | yes | keeps it |
| Watchlists: `OverviewPane` | G | no (Statics only) | cannot hold one |
| Console: `ConsoleInspectorSection`, `ConsoleRetrievalScopeRow`, `ConsoleSettingsSummary`, `ConsoleStagedContextTray`, `ConsoleStagedEvidenceStrip`, `ConsoleTranscriptEmptyPanel`, `ConsoleWorkspaceDetailsTray` | G | yes | keeps it |
| Console: `ConsoleRunInspector` | O, then G | yes | keeps it |
| Console: `ConsoleTranscript` | O | yes | keeps it |
| Console: `ConsoleWorkspaceContextTray` | O, G, then K (kept row) or W | yes | keeps it (fixed in K) |
| Console: `ConsoleCharacterContext`, `ConsoleConversationActionMenu`, `ConsoleDispatchRecoveryRegion`, `ConsoleMessageHeader`, `ConsoleProjectInstructionContextPanel`, `ConsoleSelectedTurnActivity`, `ConsoleWorkspaceActionMenu` | W | yes | keeps it |
| Console: `FixedImage` (avatar image) | W | no (an image, no children) | cannot hold one |
| Console: `ConsoleBoundedSection` | own override that removes nothing | n/a | cannot hold one |
| Settings: `SettingsRegion` (category panes, overview sync, `#settings-custom-endpoints`) | S | yes (screen builders) | keeps it (fixed in S) |
| Settings: `HooksSettingsPanel`, `ImageGenSettingsPanel`, `VideoGenSettingsPanel`, `PersonalContextSettingsPanel`, `_SpeechSettingsCard`, `ToolProfilesPanel` | W | yes | keeps it |
| Settings: `SpeechTTSSettingsPanel` | W | yes | keeps it (the credential handler switched from `.settings-speech-credential Button` to the buttons' own `.settings-speech-credential-action` class) |
| Library (not `PostRecomposeCallback`): `LibraryCollectionsScopeRows`, `LibraryCollectionsItemsPane`, `LibraryCollectionsWorkPane`, `LibraryIngestPreflightSummary`, `LibraryNotesAddFromFilesCanvas`, `LibraryNotesSyncRootsCanvas`, `LibrarySkillsTrustHeader` | W | yes | keeps it |
| Library: `LibraryPromptHistoryRegion` | O | yes | keeps it |
| Home: `HomeCanvas` (`action_button_factory`), `HomeRail`; MCP: `MCPRail` | G | yes | keeps it |
| Evals: `LibraryRail` (`UI/Evals`), `SkillEvalDetail`; Models: `CuratedView`, `InstalledView`, `RemoteView`; Wizards: `SetupStep`, `SpeechSetupStep`; `WorkspacePersonaPicker` | W | yes | keeps it |
| Models: `ExternalModelView` | W | yes | keeps it (the action handler switched from `.external-model-actions Button` to the buttons' own `.external-model-action` class) |
| Evals: `_SnippetResults`; Wizards: `SetupWizardProgress` | W | no (Statics only) | cannot hold one |

The TASK-41703 `PostRecomposeCallback` canvases (`LibraryConversationsCanvas`, `LibraryExportCanvas`, `LibraryIngestCanvas`, `LibraryIngestQueuePanel`, `LibraryLandingCanvas`, `LibraryMediaCanvas`, `LibraryMediaTrashCanvas`, `LibraryMediaViewer`, `LibraryNoteImportCanvas`, `LibraryNotesCanvas`, `LibraryPromptWorkPane`, `LibraryPromptsListCanvas`, the Library `LibraryRail`, `LibrarySearchRagPanel`, `LibrarySkillsListCanvas`) keep it through the same `Widget.recompose` hook now, via their override's `super()`.

**AC#2: the route pilots and the structural test.** `Tests/Library/test_library_notes_canvas_pending_lookup.py` (the TASK-41703 pilot's file, a whole-file target in the `derived-artifacts.yml` fast lane, so no census edit) gains `test_every_recompose_route_keeps_a_press_queued_below_a_removed_child`, parametrized over the eight routes the inventory uses: `refresh` (W), `reactive` (W), `awaited` (W), `chained-override` (G, standing in for O and B), `screen`, `removed-by-its-parent` (a container removed by its parent's recompose), `settings-region` (the real `SettingsRegion`) and `console-kept-filter-row` (the real `_KeepsConversationFilterRow`). Each holds the press in the button until the child above it is closing, the same driven hold as the canvas pilot, and asserts the screen's handler ran once. `test_every_recompose_reaches_the_press_keeper` asserts that `Widget.recompose` carries the keeper, and fails for any `recompose` override in the package that calls `.remove`/`.remove_children` before `keep_queued_presses`, directly or one hop away through a method of the same class (deeper chains are not followed).
- Red: with the keeper off (install returned early, both explicit calls removed), 10 of the file's 15 tests failed on their assertions: the 8 routes ("the <route> recompose dropped the press", `assert 0 == 1`), the TASK-41703 canvas pilot, and the structural test. With only the `Widget.recompose` hook (the two explicit calls removed), 3 failed: `settings-region`, `console-kept-filter-row` and the structural test, which named `settings_screen.py` and `console_workspace_context.py`. Green: 15 of 15, then 75 of 75 over 5 runs at `-n 4` (load 20-23). Serial with the lane's flags the file takes 10.1 s (load 21-23); no new test is in its top three.
- **Per container.** About 65 containers above can hold a press, and they share these routes; a pilot per container would run the same `Widget.recompose` code once per container. The structural test covers each container's own override instead. AC#2 was reworded to say so (coordinator decision, 2026-11-20).

**Covering runs** (`-n 4`): this file; the console tray, rail search, workspace-context rail and tray-guard files; `test_settings_panel_scoped_updates.py` (it patches `Widget.recompose`); the Library canvas, notes-navigator, phase-C residency and roleplay item-list files (two of them patch `Widget.recompose`); and `Tests/Widgets/Library`. Head: 75 failed, 626 passed (load 19-35). Base `d85e90bc9e`, all of this branch's source and test changes reverted: 80 failed, 615 passed (load 21-30). 70 of the head failures are the local `RecoveryRequired` trap, and every head failure also fails at base except two nodes in `test_console_workspace_context_rail.py`. That file is flaky at base under load: four of its nodes, three rounds each, failed 4 of 12 at head and 2 of 12 at base, each time on a different node, with the same state-timing child failures as base's four. Under a temporary `@private_profile_test` marker (removed afterwards), `test_console_rail_search_debounce.py` (the kept-filter-row path) and `test_library_canvas_sync_defects.py` passed 23 of 24. The one failure, `test_a_burst_of_keystrokes_runs_the_debounced_pass_once` (`assert 5 <= 4` syncs), failed 3 of 4 at both head and base. `Tests/Architecture` run serially: 3 failed, 1489 passed. All three (`test_reviewed_diagnostic_changes_are_metadata_only` on `agent_service.py`, and two `test_timer_path_static_update_inventory.py` nodes on `project_index_binding_card.py`) fail at base with the same command; they are not this branch's files. Preflight passes; `ruff --select F` and `Tests/lint/ruff_gate.py` pass. Boot budget ratchets (the CI step's seven files, serial): 33 passed, 1 failed (`test_boot_worker_and_thread_starts_stay_within_the_allowlist`: the `console-skill-trust-setup` worker), which fails at base with the same command; preimport headroom 4414 LOC.

**Size.** `settings_screen.py` 34978 -> 34980 (budget 34993, within slack); `console_workspace_context.py` has no size row; `library_canvas_sync.py` 291 -> 251; `tldw_chatbook/__init__.py` 509 -> 579 (no row).

Files: `tldw_chatbook/__init__.py`, `tldw_chatbook/Widgets/Library/library_canvas_sync.py`, `tldw_chatbook/UI/Screens/settings_screen.py`, `tldw_chatbook/Widgets/Console/console_workspace_context.py`, `Tests/Library/test_library_notes_canvas_pending_lookup.py`, `backlog/docs/lessons-textual.md` (the TASK-41703 entry updated, plus the too-late-hook trap).

**Review r18lib-rev1 fixes (2026-11-20).**
- **Ancestor selectors.** A forwarded press only runs handlers whose selector the button matches on its own. The selector audit (an AST scan of every `@on(Button.Pressed, ...)` in the package) found six that name an ancestor. Three are on the recomposing containers above. The other three are on the Research source list and receipt widgets, which never recompose themselves, so a press forwarded past them skips their handlers anyway (the removed-subtree limit).
  - `ExternalModelView._action_pressed` now selects `.external-model-action`, a new class on the change, stop and copy buttons only.
  - `SpeechTTSSettingsPanel.handle_credential_action` now selects `.settings-speech-credential-action`, a new class on the credential edit and clear buttons only.
  - No other button in either widget carries the new class, and the handler bodies are unchanged.
  - `WorkspaceCreateModal._remove_folder` is left to drop the press on purpose; the table gives the reason.
- **Tests.**
  - `test_an_external_model_action_pressed_across_a_reload_still_runs` and `test_a_speech_credential_action_pressed_across_a_recompose_still_runs` hold a press across `reload()` and across the panel's `refresh(recompose=True)`. With the old selectors both were red 3 of 3 runs: `assert [] == [ChangeRequested]`, and `assert 'Screen' == '_CredentialEditorModal'`. Both pass after the switch. The Speech test passes `config_snapshot={}`, so the panel reads no profile in this lane.
  - The route pilot's screen has a `#toolbar Button` counter. **Deviation:** it is pinned at 0 for seven routes, not all eight. When another task awaits `container.recompose()` (the `awaited` route), the container handles the press before the button detaches, and the counter read 1 in 6 of 6 runs. In the seven routes where the container's own pump runs the recompose, it read 0 every time. So an ancestor selector never runs there, and may run when the recompose is awaited from another task. The docstring and the lesson say so.
- **Lock.** The `Widget.recompose` wrapper now runs the keeper and the native recompose inside `async with self.lock:`, which is re-entrant per task and is the same lock `batch()` takes. It re-checks `is_attached` and `_pruning` inside the lock, so a concurrent batch or recompose cannot mount buttons between the keeper and the removal.
- **Structural test, one hop.** `_unkept_removals` also follows a call to a method of the same class that removes. A built-in `_ONE_HOP` sample pins this: with direct-only scanning it was red (`assert [] == ['Pane.recompose:4 self._clear']`); with one hop it passes, and the package has no new offenders.
- **Old quotes.** `library_skills_controller.py` (line count kept, 3170) and `backlog/docs/library-decomposition-recipe.md` now quote the helper's TASK-42012 body.
- **Hints.** `keep_queued_presses(container: "Widget")`, with a `TYPE_CHECKING` import in the package `__init__`.
- **Size.** `speech_tts_settings_panel.py` went 6108 -> 6110; its module-size row is raised by 2, with a dated comment.
- **Runs** (all through the wrapper). The lane file: 17 of 17 at `-n 4`. Size, code-identity, fold-guard and support-surface ratchets, serial: 321 passed. The covering set again (the first round's files, plus `test_model_external_view.py`) at `-n 4`: head 77 failed and 633 passed, base `d85e90bc9e` 77 failed and 622 passed (load 10-14). 71 of the head failures are the local `RecoveryRequired` trap. The two sides differ only by one node each in the flaky `test_console_workspace_context_rail.py`. The head-only node, `test_console_f1_tolerates_missing_tree_context_snapshot`, failed 2 of 3 at head and 2 of 4 at base, each time with the same `KeyError: 'workspace-1'`. `test_model_external_view.py`, `test_settings_speech_tts_panel.py` and `test_settings_speculative_voice_panel.py` against base fail the same nodes (125 of them `RecoveryRequired`), apart from `test_speech_save_and_revert_clicks_work_while_a_field_is_focused`. That node hits a timing `OutOfBounds` click on both sides: 10 of 11 passed at head and 10 of 11 at base. Boot budget: 33 passed and the same base-red `test_boot_worker_and_thread_starts_stay_within_the_allowlist`; the module counts are unchanged.
