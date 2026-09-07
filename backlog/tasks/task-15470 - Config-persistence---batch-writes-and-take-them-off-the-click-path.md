---
id: TASK-15470
title: 'Config persistence: batch writes and take them off the click path'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 12:05'
labels:
  - perf
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: `UI/Dictation_Window_Improved.py:765-781` `_save_settings` performs 8-10 sequential `save_setting_to_cli_config` calls — each a full config.toml read + atomic rewrite + settings-cache reload, synchronous on the event loop — and the buffer-duration input triggers it PER KEYSTROKE whenever the typed value parses (`:502-513`). The batch API (`save_settings_to_cli_config`) already exists and is simply not used. Separately, every Chat sidebar collapsible toggle rewrites `ui_state.toml` synchronously (read+parse+serialize+write) via `watch_sidebar_state` (`chat_screen.py:19744-19807`). About ten more verified one-rewrite-per-click sites: Library notes-sync controls (`library_screen.py:19534-19657`), ingest backend/browse/submit/reset (`:20040/:20316/:21070/:21681`), `Widgets/enhanced_file_picker.py:1137`, `Widgets/settings_splash_screen_viewer.py:237`, `settings_screen.py:3581`, `UI/Console_Modules/console_settings_modal.py:1201`.

Fix direction: batch the dictation save into one call; debounce + `to_thread` the writes; keep parsed ui-state in memory instead of re-loading the file per save. Stability constraints: the config writer's lock + atomic-replace semantics must be preserved, and any debounced write must flush on unmount/quit so no setting is lost. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The dictation save performs one batched config write; no config rewrite fires on a parsing keystroke
- [x] #2 Sidebar toggles perform no synchronous file I/O on the event loop; state survives restart including quit-immediately-after-toggle (flush test)
- [x] #3 Each listed per-click rewrite site is converted or explicitly justified in the notes
<!-- AC:END -->

## Implementation Plan

1. Verify each audit-cited site at HEAD (line numbers have shifted / some already partially fixed by other sessions since the audit).
2. Dictation window: batch `_save_settings` into one `save_settings_to_cli_config` call (`dictation` + `dictation.privacy` sections); make `_persist_settings` (the single mount-noise gate) debounce (0.6s) instead of writing synchronously; dispatch the write to a worker off the loop; snapshot `self.settings` on the main thread before handing to the worker; add `on_unmount` to force-flush a pending write.
3. Chat sidebar: convert `watch_sidebar_state` from a synchronous `_save_sidebar_state()` call to a debounce (0.5s) + worker dispatch (`asyncio.to_thread`), snapshotting `ui_state` on the main thread first; add `on_unmount` flush (waiting on an in-flight worker rather than double-writing); route `handle_reset_settings` through the same scheduling helper since reassigning an already-`{}` `sidebar_state` is a reactive no-op that never calls the watcher.
4. Library screen notes-sync cluster (6 call sites) and ingest backend/browse/submit/reset (4 call sites): wrap each write in a `@work(thread=True)` helper, matching the file's existing `_save_library_search_history`/`_save_library_rail_preferences` convention. Keep `_remember_library_ingest_location` itself synchronous (existing tests call it directly without a mounted app) with a thin `@work` wrapper (`_persist_library_ingest_location`) at its one production call site instead.
5. Remaining four listed sites: `enhanced_file_picker.py`'s `_save_last_directory` (`@work`, it's a `ModalScreen` so `self.app` is always available), `settings_splash_screen_viewer.py`'s `_save_config_value` (split the in-memory update from the deferred write), `settings_screen.py`'s remote-images toggle AND the model-catalog stale-hours `Input.Changed` handler (found in the same ADR-020 immediate-write cluster while fixing the toggle -- same per-keystroke defect class as dictation, not in the audit's literal citation but the same bug), `console_settings_modal.py`'s Save-as-default button (`asyncio.to_thread`, not fire-and-forget, since its success/failure UX contract needs the awaited result before deciding whether to dismiss or show an inline error).
6. Update every existing test that called a handler on a bare/unmounted screen and hit `run_worker`'s `NoActiveAppError` -- patch the new per-site worker method instead of the module-level config function it wraps, preserving each test's actual subject.
7. Write flush-on-quit tests for dictation and sidebar; mutation-verify by disabling the flush and confirming the test goes red, then restore via Edit.
8. Run the full affected test surface; write the report.

## Implementation Notes

All thirteen sites from the audit (dictation, sidebar, plus the ~10 per-click
sites) are converted to debounce+worker or `to_thread` dispatch; one more
(`settings_screen.py`'s model-catalog stale-hours `Input.Changed` handler) was
found and fixed alongside the remote-images toggle -- same per-keystroke
defect class as dictation, not in the audit's literal citation.

**Per-site classification:**

| Site | Treatment |
|---|---|
| Dictation `_save_settings`/`_persist_settings` (buffer-duration keystroke + 4 switches) | Debounced (0.6s) + batched into ONE `save_settings_to_cli_config` call (`dictation`+`dictation.privacy`), dispatched via `run_worker`+`asyncio.to_thread`; `on_unmount` flush |
| Chat sidebar `watch_sidebar_state` (Collapsible toggles, expand/collapse/reset) | Debounced (0.5s), `asyncio.to_thread` dispatch; `on_unmount` flush (waits on in-flight worker instead of double-writing) |
| Library notes-sync (folder submit/browse, direction/conflict choice, auto-sync toggle, validated sync run -- 6 sites) | `@work(thread=True)` via shared `_save_library_notes_sync_setting` helper (one-shot user actions, no burst risk -- no debounce needed) |
| Library ingest backend switch | `@work(thread=True)` |
| Library ingest browse (`_remember_library_ingest_location`) | `@work(thread=True)` thin wrapper (`_persist_library_ingest_location`); the method itself stays synchronous since existing tests call it directly without a mounted app |
| Library ingest submit (option batch save) | `@work(thread=True)`; the batching itself was already done by earlier work (task-3313-adjacent), this task only moved it off the loop |
| Library ingest option reset | `@work(thread=True)`, reusing the submit path's helper |
| `enhanced_file_picker.py` `_save_last_directory` | `@work(thread=True)` (it's a `ModalScreen`, `self.app` always available) |
| `settings_splash_screen_viewer.py` (6 Checkbox/Select/Input handlers via `_save_config_value`) | In-memory update + message stay synchronous; only the disk write deferred to a new `@work(thread=True)` worker |
| `settings_screen.py` remote-images toggle | `@work(thread=True)`; in-memory `app_config` poke (read live by the transcript gate) stays synchronous |
| `settings_screen.py` model-catalog stale-hours `Input.Changed` (bonus find) | Same treatment as the toggle; the existing no-op guard (cheap, cache-only) stays synchronous |
| `console_settings_modal.py` "Save as default" button | `await asyncio.to_thread(...)`, NOT fire-and-forget -- its success/failure UX contract (inline error vs. dismiss) needs the awaited result |
| `enhanced_file_picker.py`'s `RecentLocations`/`BookmarksManager` | NOT converted -- plain non-Widget classes with no `self.app`; not the line the audit cited (`:1137` = `_save_last_directory` only); out of the enumerated scope |

**Stability guards added on review:** every new `@work(thread=True)`/`run_worker` callback wraps its `save_setting(s)_to_cli_config` call in `try/except` -- an uncaught worker exception is fatal to the whole app by default (`exit_on_error=True`), which would have been a worse regression than the synchronous-write bug this task fixes. Two workers (`_persist_remote_images_toggle`, `_persist_model_catalog_section_values`) were missing this guard in an earlier pass and were fixed in a follow-up commit. Every debounced write that snapshots mutable state (`self.ui_state`, `self.settings`) does so on the calling (main) thread before handing off to `to_thread`, so a further edit arriving mid-write cannot race the worker's read of the same dict.

**Tests:** two new files (`test_chat_screen_sidebar_state_debounce.py`, `test_dictation_settings_debounce.py`) cover the debounce-armed / no-sync-write / natural-fire / flush-on-quit behavior; the flush-on-quit tests were mutation-verified (temporarily disabled the `on_unmount` flush call, confirmed the test goes red, restored via Edit). Eight existing tests across `test_library_ingest_canvas.py`, `test_library_screen.py`, `test_library_ingest_retry_last.py`, `test_library_ingest_flow.py`, and `test_settings_configuration_hub.py` called a handler on a bare/unmounted screen and broke on `run_worker`'s `NoActiveAppError`; each now patches the new per-site worker method instead of the module-level config function it wraps.

Consolidated run across every directly touched test file: 612-619 passed
(counted across two overlapping runs), 7 failures -- all confirmed
pre-existing and unrelated to this task: a `QwenCloud`-provider test-data
drift (`test_model_catalog_toggles_initialize_from_saved_config`, caused by
an unrelated earlier commit, `assert set` doesn't touch anywhere this task
edited), a generic-ingest-options content mismatch exposed only once the
`run_worker` crash this task fixed stopped masking it
(`test_options_persist_to_config` -- schema drift, not a threading bug),
two dependency-presence governance fixtures whose premise requires PDF/audio
tooling to be ABSENT (`test_forecast_counts_equal_the_real_receipt_for_a_*`),
one OCR-backend-absence-dependent consent-routing test
(`test_the_same_folder_still_imports_on_this_machine`), and two rendering/
geometry tests (`test_schema_disabled_fields_paint_legibly_inert`,
`test_the_fold_pays_for_itself_in_the_shipped_screen`) -- the latter two
directly A/B-probed by temporarily reverting the one changed line back to
the synchronous call and confirming the SAME failure. A separate full run
of `Tests/UI/test_library_shell.py` + 3 sibling files (598 passed, 15 failed)
surfaced the same failure classes (missing-dependency governance fixtures,
label-text drift, geometry, `Local prompt/quiz/study backend is
unavailable` fixture gaps) plus two directly probed the same way (the
"reset to defaults" title-receipt test and a sync-navigator-focus test, the
latter never even calling any handler this task touched); none intersect
this task's diff.

**Modified files:** `tldw_chatbook/UI/Dictation_Window_Improved.py`,
`tldw_chatbook/UI/Screens/chat_screen.py`,
`tldw_chatbook/UI/Screens/library_screen.py`,
`tldw_chatbook/UI/Screens/settings_screen.py`,
`tldw_chatbook/Widgets/Console/console_settings_modal.py`,
`tldw_chatbook/Widgets/enhanced_file_picker.py`,
`tldw_chatbook/Widgets/settings_splash_screen_viewer.py`, plus test files
(`Tests/UI/test_chat_screen_sidebar_state_debounce.py` and
`Tests/UI/test_dictation_settings_debounce.py` new;
`Tests/Local_Ingestion/test_dictation_window_provider_ids.py`,
`Tests/UI/test_library_ingest_canvas.py`, `Tests/UI/test_library_screen.py`,
`Tests/UI/test_library_ingest_retry_last.py`,
`Tests/integration/test_library_ingest_flow.py`,
`Tests/UI/test_settings_configuration_hub.py` updated).

### Review round 1 (fix commit a73eda6cc)

The independent review found 1 Critical + 2 Important defects in the original round, all
probe-proven: the splash viewer's worker error branch called `self.call_from_thread` (App-only
API) and crashed the app via exit_on_error; both debounced sites could silently lose an edit
landing during an in-flight write on quit (dirty cleared after the write; flush returned without
re-checking); and the file-picker conversion was neutered by `recent_locations.add()`'s sync
write on the same dismiss path. All fixed: `self.app.call_from_thread` + failure reverts the
in-memory value; dirty clears at snapshot time and the flush re-checks after awaiting in-flight
(race tests born red at both sites); recents + last-dir coalesced into one deferred write with
an end-to-end persistence test. The three library swallows now log; one test pins production's
real dotted config strings.

Provenance: the fix was authored by the task's implementer agent, which was interrupted
mid-verification (session limit, then stopped); the controlling session completed the remaining
test batches (50 + 31 green, ruff attributed) and committed the staged work as a73eda6cc. The
scoped re-review independently mutation-verified the Critical and one flush-race site.
