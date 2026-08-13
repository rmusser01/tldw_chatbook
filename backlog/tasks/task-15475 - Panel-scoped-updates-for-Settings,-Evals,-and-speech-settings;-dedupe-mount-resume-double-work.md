---
id: TASK-15475
title: Panel-scoped updates for Settings, Evals, and speech settings; dedupe mount/resume double-work
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 12:05'
labels:
  - perf
  - settings
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: Settings' `active_category` is a screen-level recompose reactive (`settings_screen.py:1930`) — each rail click rebuilds the category buttons plus a 60-150-widget detail pane, and entering Overview triggers a SECOND full-screen recompose via the sync-rows refresh (`:6659-6660`); "Sync preview"/"Run" full-screen-recompose twice per click to change three Statics (`:14970/:15002`). Evals rail selection recomposes 150-300 widgets per click (`UI/Evals/evals_screen.py:409`). The speech/TTS settings panel rebuilds ~200 widgets on every provider/policy dropdown change (`Widgets/Settings_Widgets/speech_tts_settings_panel.py:3741-3787`; `speech_playground_pane.py:700` shows the correct region-swap pattern in the same feature). Also duplicated per-visit work: Console dispatches `_refresh_console_skill_candidates` twice per visit (`chat_screen.py:13956` and `:19165`, exclusive=False) and syncs task-resume state twice; Settings runs `_queue_sync_rows_refresh` on both on_mount and on_screen_resume (`:2398/:2444`).

Fix direction: detail-pane-scoped swaps and targeted Static patches; per-instance flags to dedupe the mount/resume double-dispatch. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings rail clicks and Evals selections rebuild only the detail region (evidence per surface); sync preview updates statics in place
- [x] #2 Speech panel dropdown changes rebuild only the provider-form subsection
- [x] #3 The duplicated mount/resume workers run once per visit (evidence); all touched surfaces behaviorally unchanged (tests)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Four surfaces, one discipline: pin current behavior (render + focus + screen-path
side effects) with a born-red evidence test, convert to a scoped update, re-run the
surface's suites.

1. **Console dedupe** (`chat_screen.py`). `on_mount` and the mount's OWN
   `on_screen_resume` both dispatch `_refresh_console_skill_candidates` and both
   sync task-resume state. Add a one-shot per-instance token set at mount and
   consumed by exactly the next resume (class-level default, so `__new__`-built
   fixtures read it safely). Later resumes (modal pop, tab return) still refresh.
2. **Speech panel** (`speech_tts_settings_panel.py`). The four global-defaults
   `Select.Changed` handlers and `_apply_configure_provider` call
   `await self.recompose()` (whole panel, ~200 widgets). Add an awaited
   region-swap helper (`remove_children()` then `mount_all()`, both awaited --
   Textual's `remove` is deferred, see `speech_playground_pane`'s
   `_replace_provider_regions`) and route each handler to the card it actually
   changes: `#settings-speech-global-defaults` for the four, and
   `#settings-speech-provider-setup` for Configure Provider; the inspector card
   rebuilds alongside since it reads both.
3. **Evals** (`evals_screen.py`). `select()` recomposes the whole screen. Swap the
   Lab frame regions instead: rebuild `#evals-detail-pane` in `#lab-body` and
   `#evals-inspector-pane` in `#lab-inspector`. The rail's row SET only changes on
   a mutation, so `select()` grows `rail_dirty: bool = True` (safe default) and the
   rail-click path -- the audit's hot path -- passes `False` and just re-marks
   `is-active` in place via `LibraryRail`'s `_row_targets`.
4. **Settings** (`settings_screen.py`). Drop `recompose=True` from
   `active_category` and give the detail and impact panes their own recomposing
   container widgets; `_select_category` patches the mode line, the rail's
   active button, and the focus-help line in place and recomposes only the two
   panes. Drop `recompose=True` from the two sync-row reactives too: their
   Overview rows live in dedicated containers that rebuild alone, and the summary
   row is a `Static.update`. Dedupe `_queue_sync_rows_refresh` between `on_mount`
   and the mount's own `on_screen_resume` with the same one-shot token as (1).
   Container-scoped recompose does NOT order against `screen.call_after_refresh`
   (two pumps), so every post-swap follow-up (focus restore, overflow hint,
   reveal-active-button) hangs off the container's own `recompose()`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Four surfaces, four commits, one discipline: pin the current behavior, convert
to a scoped update, prove the scope with widget IDENTITY (a widget that
survives is the same Python object), then re-run the surface's suites.

**Console** (`chat_screen.py`). Textual posts `ScreenResume` when a screen is
PUSHED, so `on_mount` and the mount's own resume both fired and both dispatched
the (non-exclusive, uncancelled) skill-candidate worker and the task-resume
sync. A one-shot per-instance token set at mount and CONSUMED by the next
resume collapses the pair; every later resume still refreshes, because a skill
may have been installed while Console was suspended. 2 dispatches -> 1.

**Speech panel** (`speech_tts_settings_panel.py`). The three cards a dropdown
can invalidate became `_SpeechSettingsCard`s (Vertical subclasses with their
own `compose()`), so `await card.recompose()` rebuilds one. Global-defaults
handlers rebuild defaults+inspector (83 of 188 widgets, median 47.6 ms vs
84.7 ms for the whole panel); Configure Provider rebuilds setup+inspector (40
of 188). Focus is now retained on the control the user just operated -- the
whole-panel recompose left `app.focused is None` every time.

**Evals** (`evals_screen.py`, `library_rail.py`). `select()` swaps the Lab
frame's regions instead of recomposing the screen. The rail is only rebuilt
when its ROWS changed: `EvalsSelectionChanged` carries `rail_dirty` (default
True) and only the row-press path opts out, because the rail also posts that
message for its own mutations. End to end, 325 ms -> 146 ms median.

**Settings** (`settings_screen.py`). All three screen-level `recompose=True`
reactives are gone. The detail and inspector panes are `SettingsRegion`s; the
mode line, rail active marker/label and focus-help are patched in place; the
Overview sync rows own two small regions and the front-door summary is a
`Static.update`. 69 of 138 widgets per category switch, 13 per sync-rows
refresh; 270 ms -> 98 ms median.

Trade-offs and things that were measured, not assumed:

- Region swaps must be BATCHED. Two separately-awaited swaps each drove their
  own layout pass (Evals: median 105 ms -> 88 ms inside `self.batch()`).
- A naive "recompose vs swap" timing flatters the recompose, because the Lab
  frame defers its body mount out of the recompose entirely. Every number
  above is trigger-to-content-on-screen.
- Four things the whole-screen recompose provided for free had to be ported,
  each caught by an existing test: mouse-capture release around the teardown
  (task-627; extracted to `BaseAppScreen.release_mouse_capture_for_teardown` /
  `sweep_stale_mouse_capture` and used by both Settings and Evals), callback
  ORDERING (`_after_category_panes`), post-layout timing for the inspector's
  fold indicator, and `repaint=False` on the de-recomposed reactives.
- No focus restore in the Evals swap, deliberately: the clicked row survives
  now, and a fresh `ResultsGrid` focuses its own DataTable on purpose -- a
  restore queued after it wins the FIFO race and kills the shortcuts the
  footer advertises.

Two existing tests were updated to keep MECHANISM assertions honest about the
new mechanism (the capture-drain victim now comes from the pane that is
actually torn down; the Overview disclosures test asserts the stronger fact
that the Collapsibles are not rebuilt at all). task-15510's strict xfail
flipped to XPASS: the reordered `_apply_navigation_provider_context` lands the
deep-linked model before the preselect's own dirty mark, while a genuine
pre-existing draft still trips the guard. Marker removed, reasoning recorded
in the test; task-15510 itself is left for its owner to close.

**Review round 1** (1 Critical, 3 Important, 4 minors — all addressed):

- *Critical.* `_replace_region` called `remove_children()` on `#lab-rail` /
  `#lab-inspector`, whose FIRST child is the frame's collapse header (composed by
  `LabWorkbench`, which is exactly why `_populate_regions` APPENDS with
  `mount_all`). One rail click destroyed the Catalog and Inspector collapse
  buttons permanently — no keyboard binding, and no screen recompose left to
  rebuild them. The swap now removes only the mode-owned children. Test drives a
  real `EvalsScreen` through three rail clicks and a rail-rebuilding swap.
- *Focus escape (both surfaces).* Textual's `_reset_focus` dropped the user on the
  rail's Domain Defaults GROUP TOGGLE after a same-category Settings rebuild
  (measured: `settings-category-group-domain-defaults`) and on `lab-rail-collapse`
  after an Evals rail rebuild — one Space from collapsing something they were not
  looking at. Both swaps now capture the focus token when focus is inside a region
  they are about to rebuild and restore it by id, mirroring
  `_replace_card_bodies`. The Evals restore is deferred and yields to `#lab-body`,
  so `ResultsGrid`'s deliberate autofocus still wins.
- *task-15510 comment corrected.* Spy-probed instead of reasoned: the guard
  returns early on this tree too (`guard_dirty=True`), so
  `_apply_navigation_provider_context` writes nothing either way. What lands the
  model is the detail pane composing once through
  `_provider_display_setting_values()` while no draft exists. The guard-vs-draft
  ordering task-15510 describes is untouched and remains its owner's.
- *Overview statics under-protected.* Neutering the region rebuild left every
  test green, because the added assertion read a summary Static a different path
  keeps current. The test now reads the ROW statics inside the two regions, plus a
  second test for a changed row SET; mutation-checked (`refresh(recompose=True)`
  → `refresh()` fails 2 tests).
- *Minors.* Exclusive worker groups replaced on both surfaces by a lock + revision
  check: cancellation could land inside `remove_children` and strand a region
  emptied but never refilled (and skip the capture sweep), whereas a superseded
  swap now returns before touching a widget; `rail_dirty` accumulates across
  superseded calls so a rebuild is never lost. `_category_pane_swap_pending` is
  cleared only by the swap that still owns the revision. Stale `evals_screen`
  module docstring rewritten. The duplicate `_update_inspector_overflow_hint`
  queued by `_select_category` (which fired first, against the pane the swap had
  not rebuilt yet) is now only queued on the no-switch path.

Files: `UI/Screens/settings_screen.py`, `UI/Screens/evals_screen.py`,
`UI/Screens/chat_screen.py`, `UI/Evals/library_rail.py`,
`UI/Navigation/base_app_screen.py`,
`Widgets/Settings_Widgets/speech_tts_settings_panel.py`; new tests
`Tests/UI/test_{console_visit_dispatch_dedupe,speech_settings_panel_scoped_updates,evals_selection_scoped_regions,settings_panel_scoped_updates}.py`.
<!-- SECTION:NOTES:END -->
