---
id: TASK-32533
title: >-
  Library Notes: the app exits with an unhandled InvalidSelectValueError during
  the first-timer edit journey, and the crash handler records no site
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-13 06:45'
updated_date: '2026-09-14 08:40'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac, 2026-09-13), assessor A, persona Jordan (first-timer), Edit workflow. P0.

**What happened.** Fresh profile, no provider. Preview → Tab×6 → Shift+Tab → F6 → Escape → Enter → Shift+Tab×3 → Enter, and the app was gone: no UI message, the tmux server died with it (A capture 14 is empty). Fresh-profile `tldw_cli_app.log:300`: `event=unhandled_exception component=app exception_type=InvalidSelectValueError` at 22:50:33, then "Screen library/chat/home unmounted", `app_stopping`. `faulthandler.log` empty. The same keys replayed after relaunch did not reproduce (A 16: they landed on Save). Captures: A 13, 14, 15, 16.

**Cause — crash PROVEN, site INFERRED (≤15 min read-only trace).**
1. No traceback survives: `TldwCli._handle_exception` (`tldw_chatbook/app.py:18225-18257`, TASK-1240) persists only `type(error).__name__` and lets Textual print the traceback to stderr — the dead pane. That is itself a defect: the site cannot be recovered from the profile.
2. Textual 8.2.8 raises `InvalidSelectValueError` only in `Select._validate_value` (`widgets/_select.py:577-598`): any `select.value = X` with X outside the options, including the mount of a `Select(..., value=X)` composed with a value not in its options.
3. Every `Select` reachable inside Library ▸ Notes is guarded: the only one, `LibraryNoteFolderTargetDialog` (`Widgets/Library/library_note_folder_dialog.py:105-121`), inserts `("Top level", "")` whenever it passes `value=""` and uses `allow_blank=True` otherwise; the Search/RAG panel composes none.
4. The keys did not reopen the note. F6 in Notes is `action_focus_next_workbench_pane` (`UI/Screens/library_screen.py:8597`) whose `_NOTES_WORKBENCH_FOCUS_TARGETS` (`:1433`) go preview region → rail `#library-search-input`; Escape then Enter there is a blank rail-search submit, which "still lands on the Search canvas" (`UI/Library_Modules/library_rag_search_controller.py:773-800`) — and the log shows `Media search completed` at 22:50:30, three seconds before the crash. Shift+Tab×3 → Enter from the Search canvas is the unknown step.
5. The one unguarded compose-time `Select` on the Console hand-off path from there: `Widgets/Console/console_model_popover.py:436-440` — `Select(provider_options, value=settings.provider, id="console-popover-provider")` with no membership check of `settings.provider` against `_provider_select_options()` (`:462-480`) and no blank guard for the empty provider a no-provider draft can carry. The model select four lines down IS guarded (`value=(settings.model if settings.model else Select.NULL)`, TASK-16502 fixed exactly this class of crash there). Opened only from `UI/Screens/chat_screen.py:5492`. Second candidates on the same profile: `UI/Screens/settings_screen.py:12866` and `:21423` (`provider_select.value = …`, Providers / Speech), and the still-armed first-run wizard's Provider step (A 15 shows "Continue setup?" on relaunch). No `Select`-touching commit landed in wave 3 (`git log -S'Select(' --since=2026-09-10`), so this is pre-existing, newly hit.

Do not fix by catching the exception at the site alone: the app-level exit and the missing site are what made this a P0.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No provider Select reachable on a no-provider profile can be handed a value outside its options -- the Console model popover (a no-provider or unknown-provider draft mounts with a blank selection, the same guard its model select already has), the Settings Providers pane (#settings-provider-value) and the Settings Speech pane (#settings-speech-configure-provider), the last two keeping their current selection instead of raising when the catalog they read disagrees with the options they were built from
- [x] #2 An unhandled exception raised from a widget handler no longer exits the app: the screen stays alive and a notification names what failed and where to look
- [x] #3 The persisted unhandled_exception diagnostic carries the raising frame (module:function:line, never the message) so the site of the next crash is recoverable from the profile log; the diagnostic inventory is updated
- [x] #4 A regression test mounts the popover with a provider value absent from its options and a second one with an empty provider, and asserts no InvalidSelectValueError
- [x] #5 The Library Notes folder-target dialog (Add to folder / Move note, the site proven live on the fixed build: widget_id=library-note-folder-target) mounts with a blank selection instead of raising InvalidSelectValueError, and Choose with nothing selected does not dismiss with a bogus folder id
- [x] #6 The persisted unhandled_exception diagnostic also names the raising message pump (widget_type, widget_id) so a Textual widget that fails while mounting -- which leaves no Chatbook frame on the stack -- is still locatable from the profile log
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce both ways on a fresh no-provider profile: the assessor's recorded Notes key sequence (3 tries) and the inferred site driven directly (Console alt+m; headless popover mount with provider='' and provider outside its options).
2. RED tests: Tests/UI/test_console_model_popover_no_provider.py (two drafts mount without InvalidSelectValueError) and Tests/ProductionApp/test_app_unhandled_exception_keepalive.py (handler exception keeps the screen alive + notifies + persists the raising frame; WorkerFailed still exits; headless default still raises).
3. Guard the popover provider Select exactly like its model select (membership check, Select.NULL, allow_blank=True); confirm the read side tolerates NULL; record the settings_screen.py Select sites as guarded/unguarded.
4. TldwCli._handle_exception: keep the WorkerFailed unwrap; walk the traceback for raise_*/site_* fields (module:function:line, never the message); keep-alive only for a message-pump _dispatch_message frame and only when not headless; bell + notify naming the site; else super() as before.
5. Add the six frame fields to the persistent-diagnostics schema; re-pin the diagnostic inventory (--write); keep Tests/Architecture/test_persistent_diagnostic_inventory.py green.
6. GREEN: the new tests, the three existing popover test files, Tests/App/test_unhandled_exception_event.py, sibling pins; FAILED-name comparison vs a detached origin/dev baseline. Live: alt+m opens with a blank provider select; the recorded Notes sequence at 235x52 and 100x30 survives.
7. Guide stamp (console.md), Implementation Notes, Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Four fixes: the proven crash site, the inferred one, the same class in two more
places, and the app-level handler that made either of them fatal.

**The site, proven.** `LibraryNoteFolderTargetDialog` -- the dialog behind
Notes' **Add to folder** and **Move note** -- spelled its blank value
`Select.BLANK`. That is not a `Select` attribute on Textual 8.x: it silently
resolves to the inherited `Widget.BLANK` (`False`), which is not one of the
options, so `Select._validate_value` raised `InvalidSelectValueError` at mount
and the app exited. Live on the fresh profile, with the frame walk already in
place: `event=unhandled_exception exception_type=InvalidSelectValueError
widget_id=library-note-folder-target widget_type=Select` (profile log
12:42:49). `_submit` carried the same typo, so **Choose** with nothing picked
compared against `False`, passed, and dismissed with the literal folder id
`"Select.NULL"` -- which the caller forwards to `add_placement` /
`move_placement` as a real destination. Both now use `Select.NULL`.

**The site, inferred (Description §5).** The Console model popover's provider
`Select` took `value=settings.provider` with no membership check; a
no-provider draft carries `""`. Guarded exactly like the model select four
lines down (membership + `Select.NULL` + `allow_blank=True`, TASK-16502).
`ModelSearchPicker._current_provider` reads that select and had to learn that
`Select.NULL` is a truthy sentinel, not a provider key. It is not reachable
live on a no-provider profile -- `alt+m` in Console is refused by the setup
gate ("Typing is locked until setup finishes"), capture
`crash-01-altm-blocked-by-setup-gate` -- so it stays inferred and is pinned
headless.

**The rest of the class (AC#1 amended, controller ruling).** The two sites the
Description names as second candidates are the same defect: both write a value
read from a catalog into a Select built from a *different* read of it.
`#settings-provider-value` is composed once and `_sync_provider_manual_widget`
re-reads the catalog afterwards; `#settings-speech-configure-provider` is built
from `BUILT_IN_TTS_PROVIDER_ORDER` while its deep-link target is validated
against the separate, hand-maintained `BUILT_IN_TTS_PROVIDER_IDS`. Identical
tuples today, nothing links them. One helper,
`_assign_select_value_if_offered`, guards both: a stale selection beats a dead
pane. Not fixed by unifying the two TTS tuples -- that leaves the mutation site
unguarded for every other source of a provider id.

**The app-level exit (AC#2 as amended, AC#3, AC#6).** `TldwCli._handle_exception`
persisted only `exception_type` and then called `super()`, which exits. It now
also records the raising frame and the deepest Chatbook frame as identifiers
(module, function, line) plus the class and DOM id of the message pump that was
dispatching -- never the message, never a path. The pump fields earn their keep
here: a Textual widget that fails while *mounting* leaves no Chatbook frame on
the stack at all, which is why the first live capture of this crash read
`raise_module=textual.widgets._select` and nothing else useful. And an
exception raised inside a widget's own message handler (a
`MessagePump._dispatch_message` frame, no `WorkerFailed`) no longer calls
`super()`: the screen stays and a notification names the site. Textual has
already broken the raising widget's own message loop by then, so the copy says
that panel may stop responding until it is closed and reopened -- the AC#2
caveat. Gated off under headless `run_test` so the suite keeps its exception
signal; workers, the run loop, the compositor and the driver are unchanged.

**Trade-off worth knowing.** The keep-alive turns the two Settings sites above
from a hard exit into a pane that survives but is visibly wrong: both routes
run inside a screen message handler, so Textual stops the *Settings screen's*
own pump and the pane stays on screen without responding. That is a better
failure than losing the app, but it is a different bug from the user's point of
view; the guards are what actually prevent it, and any future unguarded Select
will land in that state rather than exiting.

**Evidence.** RED proven for every fix by patching it out against a scratch
backup (never `git stash`, never reverted on the branch) and re-running the
pin: folder dialog 2 FAILED, popover 2 FAILED, `_handle_exception` 2 FAILED,
settings guards 2 FAILED -- all green with the fix. Live at 235x52 and 100x30
on a fresh no-provider profile: **Add to folder** opens with a blank picker and
the app survives, **Choose** with nothing picked keeps the dialog open, and the
assessor's recorded key sequence (Preview → Tab×6 → Shift+Tab → F6 → Escape →
Enter → Shift+Tab×3 → Enter) leaves the app running with no new
`unhandled_exception` or `app_stopping` line. The same profile log holds the
before/after pair: 08:18:38 crash → screens unmounted → `app_stopping`;
12:42:49 the same crash, now with its site, and the app still logging five
minutes later. The diagnostic inventory needed no re-pin (`--write` reproduces
it byte-identically); `Tests/Architecture/test_persistent_diagnostic_inventory.py`
has two failures that are identical on a detached `origin/dev` baseline
(`console_agent_bridge.py`, `console_fleet_wake.py` -- unrelated files).

**Files.** `tldw_chatbook/Widgets/Library/library_note_folder_dialog.py`,
`tldw_chatbook/Widgets/Console/console_model_popover.py`,
`tldw_chatbook/Widgets/model_search_picker.py`,
`tldw_chatbook/UI/Screens/settings_screen.py`, `tldw_chatbook/app.py`,
`tldw_chatbook/Utils/persistent_diagnostics.py`,
`Docs/User_Guide/library/notes.md`; new pins
`Tests/UI/test_console_model_popover_no_provider.py`,
`Tests/ProductionApp/test_app_unhandled_exception_keepalive.py`,
`Tests/UI/test_settings_provider_select_out_of_options.py`, plus two cases in
`Tests/Widgets/Library/test_library_note_folder_dialog.py`.
<!-- SECTION:NOTES:END -->
