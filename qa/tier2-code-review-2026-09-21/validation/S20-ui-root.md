# S20 — `UI/*.py` (top level only) — validation

Validated against `d0face3ebe` (origin/dev). Review was done at `3722a85748`.

## 1. P1 [D1] — `LogsWindow.append_record` mutates the Textual DOM from whatever thread called `logger.*`, no marshalling
- Verdict: CONFIRMED
- Site now: `UI/Logs_Window.py:402-431` (`append_record`; `RichLog.write` at `:429`, `_passes`'s `query_one` at
  `:437`), caller `app.py:12176` (`PersistentLogHandler.emit`), handler registered on the root logger at
  `app.py:12212` (`logging.getLogger().addHandler(...)`) — same shapes/near-identical lines as filed
  (review cited `:12149-12167`/`:12186`, now `:12160-12204`/`:12212` — a few dozen lines of drift, same code).
- Proof: `rg -n "enqueue" tldw_chatbook/Logging_Config.py` → zero matches, confirming the loguru→stdlib bridge is
  synchronous on the emitting thread. `rg -n "call_from_thread|call_later" UI/Logs_Window.py` → zero hits inside
  `append_record` (the file's other thread hop, `_persist_filter_state_off_loop:367-...`, is a separate,
  documented seam). `PersistentLogHandler.emit` wraps the call to `logs_window.append_record(...)` in
  `except Exception: pass`, confirming failures are silent.

## 2. P1 [D1] — Voice Cloning ▸ Delete profile raises `AttributeError` before the confirm dialog can render
- Verdict: CONFIRMED
- Site now: `UI/Voice_Cloning_Window.py:550-580` (`_delete_profile`), method-local `class ConfirmDialog(ModalScreen)`
  at `:560-566` — identical lines to the review's citation.
- Proof: read the method directly —
  `yield Label(f"Delete profile '{self.selected_profile}'?")` inside `ConfirmDialog.compose`, where `self` is the
  `ConfirmDialog` instance (its own `compose(self)` parameter), which has no `selected_profile` attribute; the
  outer `VoiceCloningWindow` instance is shadowed. Reachability confirmed:
  `UI/Screens/stts_screen.py:61` registers the `voice-cloning` rail row, `STTS_Window.py:2058` mounts
  `VoiceCloningWindow()` for it, and `Voice_Cloning_Window.py:723-726`/`:473` dispatch `_delete_profile`.
  `rg "ConfirmDialog|confirm-dialog" Tests/` → only a CSS-tier registry row, no behavioural test.

## 3. P2 [D4] — Chatbooks card always appends `"..."`, so a chatbook with no description renders as literally `"..."`
- Verdict: CONFIRMED
- Site now: `UI/Chatbooks_Window_Improved.py:93` (grid card,
  `self.chatbook_data.get("description", "No description")[:100] + "..."`, unconditional) and `:562-564` (list
  row, `f"... {cb_data.get('description', 'No description')[:50]}..."`, also unconditional) — matches the
  review's `:92-94`/`:560-565` citation closely.
- Proof: `_scan_chatbooks` (`:609-660`) sets the `"description"` key **only** inside the `manifest.json` branch
  (`:635-637`); a zip without a manifest never gets the key at all (→ `.get(...,"No description")` →
  `"No description..."`), and a zip with a manifest but an empty description gets `""` (→ `"..."`). Verified the
  4 correct sibling sites in `STTS_Window.py:567-568,678-682,841-845,1428-1432` all use the conditional
  `x[:N] + "..." if len(x) > N else x` idiom. Route is live: `chatbooks_screen.py:13,32` composes
  `ChatbooksWindowImproved` and is the routed screen for `"chatbooks"`.

## 4. P2 [D1] — `MediaWindow_v2.py` (2,562 lines) and `media_screen.py` are unreachable by construction
- Verdict: CONFIRMED
- Site now: whole file, 2,562 lines (exact match). `app.py:17840-17851` (`shadowed_route_ids` pre-importer
  skip logic, comment "a route no click can ever reach should not be attempted" present verbatim) — was cited
  `:17814/17822`, drifted ~20 lines, same code.
- Proof: ran the review's own repro —
  `python -c "from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_route; print(resolve_screen_route('media'))"`
  → `ScreenRoute(screen_name='library', ..., module_path='...library_screen', class_name='LibraryScreen', ...)` —
  the alias wins, `_SCREEN_ROUTES["media"]` (→ `media_screen.MediaScreen`) can never be selected.
  `rg -n "media_screen|MediaScreen" tldw_chatbook/` → only the registry row, the lazy `Screens/__init__.py`
  map, and docstring mentions in `MediaWindow_v2.py` itself — no constructor call anywhere in production code.

## 5. P2 [D1] — Untrusted names interpolated raw into `markup=True` `RichLog`s
- Verdict: CONFIRMED
- Site now: `UI/STTS_Window.py:1435` (`log.write(f"[yellow]Generating preview for: {chapter.title}[/yellow]")`,
  `markup=True` declared at `:434`) — exact match to the review's citation. `UI/Voice_Cloning_Window.py:646`
  (`test_log.write(f"[yellow]Generating with profile '{test_profile}'...[/yellow]")`, `markup=True` at `:263`) —
  review cited `:647`, off by one line, same statement.
- Proof: `Text.from_markup("[yellow]t: a[/b]b[/yellow]")` raises `MarkupError: closing tag '[/b]' does not match
  any open tag` (repro'd against the shared venv). At the STTS site, the enclosing `try/except Exception as e`
  (`_preview_chapter_audio:1414-1461`) does catch the raise, matching the review's "preview never generates,
  cryptic toast" — confirmed by reading the `except` body at `:1459-1461`
  (`self.app.notify(f"Failed to generate preview: {e}", severity="error")`). At the Voice Cloning site,
  `_test_generate_voice` (`:632-...`) has **no** try/except around the `test_log.write` call, and its only caller
  path via `action_test_voice` → `_spawn_action` → bare `asyncio.create_task` (see finding 6) has no exception
  observer beyond discarding the task from a set — confirming "silently does nothing."

## 6. P2 [D1] — `Voice_Cloning_Window` drives user actions with bare `asyncio.create_task`, never cancelled at unmount
- Verdict: CONFIRMED
- Site now: `UI/Voice_Cloning_Window.py:706-716` (`_spawn_action`), consumers at `:721,726,731,735,739`
  (`action_new_profile`/`action_delete_profile`/`action_export_profile`/`action_import_profile`/
  `action_test_voice`); correct sibling `@work(exclusive=True, group="voice-cloning-load-profiles")` at `:325`.
  `grep -c "on_unmount" Voice_Cloning_Window.py` → 0.
- Proof: `_action_tasks: Set[asyncio.Task[Any]]` (`:184`), populated/discarded at `:715-716`; no `on_unmount`
  anywhere in the file, so a live task is never cancelled on screen teardown. `BINDINGS` (`:68-75`) map exactly
  these 5 actions (`ctrl+n/d/e/i/t`) to `action_*` → `_spawn_action`.
- Note: the review's headline says "eight user actions"; the file has exactly **5** `_spawn_action` call sites
  (`action_refresh`/`ctrl+r` calls `_load_profiles()` directly, not through `_spawn_action`), and the review's
  own Evidence section cites only these same 5 lines (`:719-739`). The "eight" count in the finding's title
  overstates against its own evidence — the underlying defect (uncancelled fire-and-forget tasks touching the
  DOM after a pushed screen) is real and reachable via all 5, so the verdict stands as CONFIRMED with the count
  corrected, not WRONG.

## 7. P3 [D3] — `Study_Window._configure_flashcards_lifecycle_controls` discards a `query_one`, a computed flag, and double-writes a property
- Verdict: CONFIRMED
- Site now: `UI/Study_Window.py:898-912` — exact match to the review's citation.
- Proof: read the body directly — `self.query_one("#delete-deck-button", Button)` (no assignment),
  `bool(scope_checker())` (no assignment/use), and `delete_deck_note.display = server_mode` written twice in a
  row. `git log -1 8a9ce5cf9c` → `feat(scheduling): Phase 5 watchlist migration + TASK-299 close-out (#707)`,
  matching the review's cited commit. Sibling `_configure_quizzes_lifecycle_controls` (`:932-...`) does gate
  multiple widgets from the same inputs, and `Study_Modules/flashcards_handler.py:374,383-388` confirms the
  button gating isn't actually lost elsewhere.

## 8. P3 [D3] — Dead dispatch branch: `STTS_Window` pushes a `Widget` as a `Screen` for a never-composed button id
- Verdict: CONFIRMED
- Site now: `UI/STTS_Window.py:2525-2529` — exact match to the review's citation
  (`elif event.button.id == "view-voice-cloning-btn": ... self.app.push_screen(VoiceCloningWindow())`).
- Proof: `rg -n "view-voice-cloning-btn" tldw_chatbook/` → exactly one hit, this branch; no `Button` with that id
  exists anywhere in the tree. `VoiceCloningWindow(DataTableClickSelectMixin, Vertical)` — confirmed a `Vertical`,
  not a `Screen` (`Voice_Cloning_Window.py:51`), so `push_screen()` would raise `TypeError` if ever reached.

## 9. P3 [D4] — Non-adopters of `Widgets/status_line.py::set_status_line`, with a drifted missing-widget contract
- Verdict: CONFIRMED
- Site now: `UI/stts_profile_library.py:2759-2770` (full non-adopter — `_set_status` guards only
  `if not self.is_mounted: return` then does 3+ unguarded `query_one` calls), `UI/Writing_Window.py:258-268`
  (half-adopter — `set_status_line(self, "#writing-status", message)` at `:262`, then a hand-rolled
  `try: ... query_one("#writing-source-status", Static).update(message) except Exception: pass` at `:263-268`
  in the same method) — matches the review's citation almost exactly.
- Proof: `Widgets/status_line.py:28-34` — `set_status_line(..., missing_ok: bool = True)` swallows a missing-line
  lookup and returns `False`; `stts_profile_library._set_status` has no equivalent per-lookup guard, so a
  `NoMatches` on any of its `query_one` calls after the `is_mounted` check propagates into the caller (confirmed
  its callers are async page-load paths at `:2369,2396,2424,2443,2447,2451`). `Research_Window.py:31,804` and
  `Writing_Window.py:18,262` both import and use `set_status_line` — confirming they are otherwise adopters, as
  the review's "RETIRED" triage note states.

TOTALS: confirmed=9 fixed=0 wrong=0 demoted=0 promoted=0
