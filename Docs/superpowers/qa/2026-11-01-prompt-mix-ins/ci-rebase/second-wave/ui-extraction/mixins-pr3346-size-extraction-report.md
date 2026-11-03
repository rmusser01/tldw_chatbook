# Roleplay / Console size extraction result

Base ba1cafc55dd5ac9b6b1f066c936206e652ed6b65. Product edits complete; no commits, pushes, manifests, or size-row edits. Root owns budget lowering and helper enrollment. Owned inventory: /private/tmp/mixins-size-owned-files.txt (15 product files, 4 test files).

Original governed owners now:

| Owner | Before | After | Existing ceiling |
|---|---:|---:|---:|
| UI/Screens/personas_screen.py |17507|16829|16859|
| UI/Persona_Modules/roleplay_frame_state.py |1322|818|855|
| Widgets/Persona_Widgets/personas_library_pane.py |1582|992|1297|
| UI/Console_Modules/session.py |7059|6990|7005|
| UI/Console_Modules/workspace.py |8948|8935|8938|
| Widgets/Console/console_transcript.py |8594|8556|8584|

New helpers: UI/Console_Modules/session_prompt_seed.py109; Widgets/Console/console_selection_geometry.py52; Widgets/Persona_Widgets/roleplay_item_list.py620. First two load only when their actual calculation is called. ItemList necessarily loads with Roleplay's event definition (@on(RoleplayItemList.WindowWanted)); no fake lazy class replacement.

Existing helper measurements: character_avatar_helpers365, personas_preview_coordinator264, roleplay_list_window_state1048, personas_character_tts_widget650, personas_state360, Workspaces/display_state1315.

Behavior/ownership:
- Exact authority dataclasses/exceptions moved to existing personas_state with annotation-only UI/runtime dependencies behind TYPE_CHECKING. Original screen names refer to the same class objects.
- Existing native completion/drain/decorator bodies moved to already imported preview_coordinator; counters, acquisitions, task cancellation and reconciliation order unchanged.
- Pure TTS and visual projections moved to their existing presentation owners. Original screen method names remain. Portrait validation explicitly receives the current screen validator at call time; its test monkeypatch seam remains live.
- Frame item values/projections moved into already imported pure list-window state. Original aliases/constants retain identity. Frame's explicit __all__ preserves its previous public star-import inventory. Existing pure-source tests now check both source owners.
- Entire item-list widget moves with row value and WindowWanted identity. Pane still owns composition/toolbar; list shares the exact ListWindowState and calls frame_state.build_item_prompt dynamically. Its parent-type lookup is locally imported at action time to avoid a module cycle. Source guards retain old-owner coverage and add the new owner.
- Session keeps both original seed dataclasses. Two first-call facades pass current constructors, expander and sanitizer to the pure computation helper. No composer import is made eager.
- Workspace normalization takes the global sentinel explicitly in existing display_state, avoiding a new Workspaces-to-Chat import.
- Transcript keeps its original @lru_cache(maxsize=4) facade and cell mapper. The uncached Textual projection moves behind first use; the current Content class is passed on cache miss, preserving the existing monkeypatch/counter seam and cache_info/cache_clear API. No Textual import added to pure console_selection.

Mechanical evidence:
- /private/tmp/mixins-pr3346-size-body-proof.txt and /private/tmp/mixins-size-body-proof.json: 68 definitions/body comparisons plus18 constants matched. Normalization is limited to explicitly injected constructor/Content/sentinel identifiers, the staticmethod decorator at extraction, and the one late pane import. Script /private/tmp/mixins_size_proof.py. Before copies /private/tmp/mixins-size-before-ba1.
- Scoped19-file Ruff:406 before,403 after, zero additional code/message diagnostics. Three import-order findings resolved; six diagnostics merely moved with unchanged bodies and remain inherited debt. /private/tmp/mixins-pr3346-size-lint-global.txt.
- git diff --check passed.

Focused tests (not a full sweep):
- Initial state/items/seeds/selection/workspace batch:327 passed,4failed in78.88s. The four old drag tests stopped at raw_source_selection_changed before reaching geometry. /private/tmp/mixins-pr3346-size-pure-tests.log.
- Follow-up mounted/lifetime/focus/TTS batch:24 passed,7failed in55.16s. Positives include stale window rejection, first-match Enter, focus/typing stability, resize without mounts, marks, lore toggle/undo, no-work cursor motion, native cancellation/draining, TTS mixed/unavailable handling and persona seed projections. /private/tmp/mixins-pr3346-size-native-tests.log.
- Exact ba1 product archive, with the same admitted drag fixture/stronger wrap observation, reproduces all7 failed nodes in12.87s. Four drag fixtures never arm selection before geometry; two older Actor Pack fixtures hit lazy editor/profile assumptions; mode-transition fixture fails before expected callback (baseline also logs missing ResolvedReferenceImage). No product fix or weakened assertion applied to these unrelated failures. /private/tmp/mixins-pr3346-size-ba1-failures.log; archive /private/tmp/mixins-size-ba1-baseline.
- Architecture/source/native keyboard batch:29 passed,4failed in18.40s. All new identity/cache/source and two real keyboard-selection controls pass. Failures are outside owned changes: Chat first-send warmup domain->UI lazy import, Chat/UI lookup counts178/1282 vs174/1278, and two new navigation private names. Root informed and owns those upstream/concurrent repairs. /private/tmp/mixins-pr3346-size-contracts.log.
- Final focused identity/live constructor/current validator/real Content cache/source-owner controls:8passed in1.92s. /private/tmp/mixins-pr3346-size-final-seams.log.

No claim that the broad suite is green. The old drag test now has the documented bootstrap profile marker and a stronger wrap-count lower bound, so zero instrumented work cannot falsely pass. A separate passing positive directly exercises the real Content projection, old cache facade and current monkeypatch seam. Old failing controls remain explicit evidence limits.

No further product edits or tests are running. Root will run the coordinated final performance/source-catalogue cohort after all owners settle.
