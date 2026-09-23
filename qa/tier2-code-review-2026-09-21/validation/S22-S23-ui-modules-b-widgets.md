# S22+S23 validation — `UI/{Evals,LLM_Management,CCP_Modules,…}/`; `Widgets/{Media,Prompts,TTS,NewIngest,…}/`

Validated against worktree HEAD `d0face3ebe` (review baseline `3722a85748`, +25 commits).

## 1. P1 — Flashcard queue-state badge deleted by markup parsing
- Verdict: CONFIRMED (verified by execution)
- Site now: `UI/Study_Modules/flashcards_handler.py:664-666` — exact match (`queue_state = str(card.get("queue_state") or "unknown")`, `label = f"{card.get('front', '')} [{queue_state}]"`, `ListItem(Label(label))`).
- Proof: `.venv/bin/python -c "from textual.content import Content; print(Content.from_markup('What is DNA? [new]').plain)"` → `'What is DNA? '` — the badge is silently deleted for `[new]`/`[learning]`/`[review]`/`[suspended]`/`[unknown]`, every value `study_normalizers.py` can produce. `UI/Evals/library_rail.py:76` does `from tldw_chatbook.Utils.input_validation import escape_markup` — canonical helper exists two files over, unused here.

## 2. P1 — 11-file table of user/wire text reaching markup sinks
- Verdict: CONFIRMED (cluster finding — spot-checked, not per-row re-derived)
- Site now (spot-checked rows): `vllm_setup_view.py:1010-1011` `external_model.set_options([(model_id, model_id) for model_id in self._discovered_model_ids])`; `:1070-1073` `profile_select.set_options([(profile.name, profile.profile_id) ...])`; `UI/Evals/skill_eval_panel.py` — lines shifted from the claimed 91/110/115 to `set_targets`/`set_subjects` around 250-284 (`options.append((f"{name} ({trust})", name))` → `picker.set_options(options)`, `grep -n escape_markup skill_eval_panel.py` → 0 hits); `Widgets/Media/media_navigation_panel.py:127` `yield Button(media_type, id=...)` (raw DB value into a Button label) and `:189-192` `display_name = str(event.button.label)` read back into `MediaTypeSelectedEvent`.
- Proof: sibling `UI/Evals/bench_editor.py:907-913` in the same package does `escape_markup(f"{row['name']} ({row['model_id']})")` for the identical `Select.set_options` shape, with a comment naming exactly this hazard — confirms `skill_eval_panel.py` is a non-adopter, not a false positive. Mechanism (`MarkupError` inside `_compositor.reflow`, not `compose()`) and `_is_admissible_model_id` allowing brackets both independently re-verified as CONFIRMED in `phase4-verification.md` "S22/S23".
- Note: did not individually re-verify all 11 rows (Study_Modules quiz rows, Research_Workspace_Modules rows) — 4 of 11 rows read directly, all consistent with the claim; treating the remainder as covered by the same AST-sweep methodology that the spot-checks corroborate.

## 3. P1 — `home_rail.py` has a non-escaping same-named twin of `library_rail.py`'s escaper; TASK-32802.1's text is wrong
- Verdict: CONFIRMED
- Site now: `Widgets/Home/home_rail.py:21` `_visible_row_title` (truncate only, no escape) → used at `:127`, `:170`; `Widgets/Library/library_rail.py:377` uses `escape_markup(_truncate_row_title(...))`.
- Proof: `grep -c escape_markup tldw_chatbook/Widgets/Home/home_rail.py` → `0`; same on `library_rail.py` → `4`. `home_rail.py` never imports `input_validation`. Independently re-verified in `phase4-verification.md` "S22/S23 — TASK-32802.1's finding text names a consumer that does not consume" — I confirmed the same grep results directly rather than deferring to that entry.

## 4. P1 — `Widgets/NewIngest/` (5 modules) has zero production importers, unreachable
- Verdict: CONFIRMED (verified by execution)
- Site now: `Widgets/NewIngest/__init__.py`, `BackendIntegration.py`, `ProcessingDashboard.py`, `SmartFileDropZone.py`, `UnifiedProcessor.py`.
- Proof: `grep -rln "NewIngest" tldw_chatbook/` outside the package itself → 0 hits; `grep -rln "NewIngest" Tests/` → 6 test files only. `__init__.py:1` = `"""Legacy NewIngest compatibility exports."""`, `SmartFileDropZone.py:1` = `"""Compatibility smart file drop zone for legacy NewIngest tests."""`. Ran the independent corroborating check: `_browse_files` (`:242`) is a plain `def` (not `async def`) that calls `self.app.push_screen_wait(...)` with no `await`; `.venv/bin/python -c "import inspect; from textual.app import App; print(inspect.iscoroutinefunction(App.push_screen_wait))"` → `True` — so `selected` is an un-awaited coroutine object (always truthy), and `list(selected)` would raise `TypeError` if this path ever ran. Confirms the module has never actually executed.

## 5. P2 — Evals snippet import reads a user file with no size ceiling on the event loop
- Verdict: CONFIRMED
- Site now: `UI/Evals/snippet_editor.py:634` `content = file_path.read_text(encoding="utf-8")` — exact match. Sibling `UI/Chunking_Lab_Modules/sample_region.py:15,36-57` `SAMPLE_BYTES = 2 * 1024 * 1024`, reads `SAMPLE_BYTES + 1` and refuses over-cap.
- Proof: read both files directly; `snippet_editor.py` validates the path and handles `OSError`/`UnicodeDecodeError` but has no byte-size guard anywhere in the function.

## 6. P2 — `CCPMessageManager` never constructed in production
- Verdict: CONFIRMED
- Site now: `UI/CCP_Modules/ccp_message_manager.py`, `@work(thread=True)` at `:67` `load_conversation_messages` (no `group=`, no `exclusive=True`).
- Proof: `grep -rn "CCPMessageManager(" tldw_chatbook/ Tests/` → exactly one hit, `Tests/UI/test_ccp_handlers.py:415`. `personas_screen.py:344-346` imports `ccp_character_handler` and `ccp_enhanced_handlers.setup_ccp_enhancements`, never `ccp_message_manager`.

## 7. P2 — Three more unreferenced modules, one short of the existing deletion tasks
- Verdict: CONFIRMED
- Site now: `UI/Workbench/route_inventory.py`, `UI/Widgets/config_search_widget.py`, `Widgets/Coding_Widgets/repo_tree_widgets.py`, `Widgets/Note_Widgets/note_creation_modal.py`.
- Proof: `route_inventory.py` — `screen_registry.py:198,263` reference it only inside a `#` comment (`grep -n route_inventory screen_registry.py` → two comment lines, no import), and `UI/Workbench/__init__.py:60-62` reaches it only via a `_LAZY_EXPORTS`-shaped string-keyed dict — `grep -rn "WorkbenchRouteCoverage|WORKBENCH_ROUTE_OWNERS|build_workbench_route_coverage" tldw_chatbook/` outside `UI/Workbench/` → 0 hits, so the lazy export itself is never pulled. `config_search_widget.py` — its only consumer is `UI/Tools_Settings_Window.py:98` (`from .Widgets import ConfigSearchResult, UIElementSearchEngine`, lazily resolved through `UI/Widgets/__init__.py`'s own `_LAZY_EXPORTS`-style table), and `grep -in config_search_widget "task-32807.4 - Delete-the-deprecated-Tools-and-Settings-window-and-its-wrapper.md"` → 0 hits, confirming it is unnamed in that deletion task's ACs. `repo_tree_widgets.py`'s only importer `UI/CodeRepoCopyPasteWindow.py` has 0 hits in `app.py`/`screen_registry.py`. `note_creation_modal.py`'s only importer `Widgets/document_generation_modal.py` is confirmed on TASK-32807.1's own **Deferred** list (`task-32807.1...md:34`: "Deferred (share multi-purpose tests or a live sibling...): ... document_generation_modal ...").

## 8. P2 — `MediaViewerPanel` defines a `ModalScreen` subclass inside a worker body, new type per Delete press
- Verdict: CONFIRMED
- Site now: `Widgets/Media/media_viewer_panel.py:2018` `@work(exclusive=True, group="media-viewer-delete-confirmation")` → `:2019 async def _run_delete_confirmation`, `class DeleteConfirmDialog(ModalScreen)` defined inside the method body at `:2036`, with function-body imports of `textual.widgets`/`textual.containers`/`textual.screen` at `:2032-2034` immediately above it.
- Proof: read `:2018-2060` directly — confirms the nested class definition (with its own `DEFAULT_CSS`) executes fresh on every call to `_run_delete_confirmation`, i.e. every Delete press.

## 9. P2 — Study controllers share 7 byte-identical + 5 constant-drifted methods
- Verdict: CONFIRMED
- Site now: `UI/Study_Modules/flashcards_handler.py` ↔ `quizzes_handler.py` — `_current_mode`, `_is_blank_select_value`, `_notify`, `_policy_action_allowed`, `_scope_state`, `_scope_type`, `_scope_type_value` all present in both files.
- Proof: grepped all 7 method names in both files, all present. Confirmed the drift claim precisely: `flashcards_handler.py:113,122` compare `!= "workspace"` (string literal) while `quizzes_handler.py:78,87,123` compare `!= StudyScopeType.WORKSPACE.value` (enum) — same runtime value today, no shared enforcement that it stays that way.

## 10. P3 — Three `re.compile` of constant patterns inside pydantic validator bodies
- Verdict: CONFIRMED
- Site now: `UI/CCP_Modules/ccp_validators.py:83` (`url_pattern`, in `validate_avatar_url`), `:101` (`version_pattern`, in `validate_version`), `:142` (`template_pattern`) — exact line matches.
- Proof: read the file; all three compile constant, non-user-supplied regex literals inside `@field_validator` bodies, so each field validation recompiles them. `:188`'s compile (not part of this finding) is of user input and is correctly excluded per the review's own note.

## 11. P3 — `personas_screen.py` reaches into `ccp_character_handler`'s private helper
- Verdict: CONFIRMED
- Site now: `UI/Screens/personas_screen.py:6961, 14600, 14632` — exact line matches, all `ccp_character_handler._default_character_db()`.
- Proof: `grep -n _default_character_db tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py` — three call sites in `personas_screen.py` at exactly those lines, reaching a function named with a leading underscore in `ccp_character_handler.py:41`.

TOTALS: confirmed=11 fixed=0 wrong=0 demoted=0 promoted=0
