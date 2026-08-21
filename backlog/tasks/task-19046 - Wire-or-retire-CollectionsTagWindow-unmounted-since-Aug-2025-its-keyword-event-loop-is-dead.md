---
id: TASK-19046
title: >-
  Wire-or-retire CollectionsTagWindow — unmounted since Aug 2025, its whole
  keyword-event loop is dead
status: Done
assignee:
  - '@claude'
created_date: '2026-08-20 08:40'
labels:
  - ui
  - dead-code
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Review-confirmed during the wave-2 close-out and re-verified at dev
`1bf7f234e`: `Widgets/collections_tag_window.py::CollectionsTagWindow` (:128,
the media-DB keyword/tag manager) is constructed nowhere in production — a
definitive whole-tree grep finds only the class definition, lazy imports
inside `Event_Handlers/collections_tag_events.py` handlers, and tests. Its
mount was lost at commit `de367762a` (2025-08-02).

The entire loop behind it is therefore unreachable: the
KeywordRename/Merge/Delete events are posted only from inside the widget
itself (:523/:529/:542), so `app.py`'s dispatch (:11953-11960) and every
`collections_tag_events` handler can never fire — and those handlers
`query_one(CollectionsTagWindow)`, which would raise on the unmounted widget
even if an event somehow arrived. TASK-15471's collections keyword-delete
threading repair (Done) landed on this dead path — corpse-groundwork, exactly
the shape this queue item predicted.

Tests keeping the corpse green: `Tests/UI/test_bulk_selection_tooltips.py`,
`Tests/UI/test_tag_action_recovery_tooltips.py`, the CollectionsTagWindow
slice of `Tests/Widgets/test_reactive_default_aliasing.py`, and
`Tests/Event_Handlers/test_collections_tag_events.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CollectionsTagWindow is either mounted and reachable from a live screen (with the event loop verified end-to-end) or retired — widget, events, handlers, the app.py dispatch block, and its tests handled together with provenance; per the owner ruling, prefer the durable option over speculative resurrection
- [x] #2 No unreachable keyword-event dispatch remains in app.py
- [x] #3 Targeted suites green; if retired, whole-tree grep for the removed names returns nothing
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify reachability at branch base `25500ad87` (origin/dev): whole-tree grep for
   `CollectionsTagWindow`, the three Keyword events, `collections_tag` (all file types),
   dynamic shapes (screen registry, Constants, string-built ids, modal pushes), and the
   `media-nav-collections-tags` nav path through `MediaWindow_v2`.
2. Decision per owner ruling: RETIRE (unless step 1 finds a live mount — it did not).
3. Delete `tldw_chatbook/Widgets/collections_tag_window.py` (CollectionsTagWindow,
   KeywordRenameDialog, KeywordMergeDialog, nested DeleteConfirmationModal + its
   BUNDLED_SCREEN_CSS) and `tldw_chatbook/Event_Handlers/collections_tag_events.py`
   (3 events, 3 handlers, `load_keyword_statistics`, `_media_db_off_loop`).
4. Remove `app.py`'s `on_collections_tag_message` dispatch block and the now-unused
   `from textual.message import Message` import (its only consumer).
5. Tests: delete `Tests/UI/test_tag_action_recovery_tooltips.py` and
   `Tests/Event_Handlers/test_collections_tag_events.py`; remove only the
   CollectionsTagWindow test from `Tests/UI/test_bulk_selection_tooltips.py` (its
   NoteSelectionDialog test covers a LIVE widget — pushed by `UI/STTS_Window.py:637`,
   embedded by `UI/Screens/stts_screen.py`); remove only the CollectionsTagWindow slice
   from `Tests/Widgets/test_reactive_default_aliasing.py` and verify the rest still
   collects and passes.
6. Regenerate the lifted CSS bundles via `tldw_chatbook/css/build_css.py` (drops the
   DeleteConfirmationModal block from `screen_css_scoped.tcss`); run the bundle sync
   guard. Leave `css/features/_media.tcss` / `tldw_cli_modular.tcss` untouched — the
   Collections/Tags section shares selectors with live widgets (`selection-info` is
   note_selection_dialog's; `usage-stats` is scope_picker_listers') and TASK-16835 left
   the sibling Multi-Item Review section the same way (retired-media-chrome residue).
7. Hand-edit `Docs/security/production-diagnostic-inventory.json` per the
   TASK-16196/16835 playbook: remove the two owner rows (call_count 10 + 3, both
   TASK-494), decrement `owner_files` 503→501 and `task_494_calls` 6991→6978; verify
   with `scripts/check_persistent_diagnostic_inventory.py`.
8. Gates: targeted suites (reactive aliasing, bulk tooltips, Event_Handlers,
   diagnostic-inventory architecture suite, css bundle sync guard, screen navigation
   pin) + repo-wide `pytest --collect-only -q` before/after comparison; ruff
   check/format on touched files; final whole-tree greps for removed names.
9. Record provenance + the live-surface judgment in Implementation Notes; tick ACs;
   status Done; commit on `task/19046-burn`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Decision: RETIRE** (per the owner ruling — re-verification at branch base `25500ad87`
found no live mount path).

**Re-verification (fresh, at `25500ad87`).** Whole-tree greps (all file types, plus
dynamic shapes — screen registry, `Constants.py`, string-built ids, modal pushes)
confirmed the filing exactly: the widget was constructed nowhere in production (hits:
definition, three lazy imports inside its own handlers, tests); the three Keyword
events were posted only from inside the widget; `app.py`'s `on_collections_tag_message`
(an `@on(Message)` catch-all!) could therefore never take a keyword branch; and the
handlers' `query_one(CollectionsTagWindow)` refresh would have raised on the unmounted
widget even if an event had arrived. The one dynamic path — the Media nav panel's
`media-nav-collections-tags` button — posts `MediaTypeSelectedEvent("collections-tags")`
into `MediaWindow_v2._perform_search`, which special-cases the slug and returns without
mounting anything; and that whole surface is itself route-unreachable (task-2851,
pinned by `test_no_route_reaches_the_retired_media_screen`, which passes on this
branch).

**Provenance.** The mount was lost at `de367762a` (2025-08-02, "i fucking hate css"):
the parent's `UI/MediaWindow.py` composed `CollectionsTagWindow` for the
`collections-tags` slug (old `:284-285`) and that commit deleted `MediaWindow.py`
wholesale; `MediaWindow_v2.py` never mounted the widget at any revision. Everything
that landed on the path since is corpse-groundwork: TASK-15471's keyword-delete
threading/dedupe repair (its reference `_media_db_off_loop` pattern lives on in
`multi_item_review_events`' 16194 fix — itself since retired by 16835), TASK-15476's
search debounce, TASK-15771's reactive-default fix + test slice, and the tooltip
tests. All deleted with the module.

**Live-surface judgment.** This retirement removes the only conceivable media-DB
keyword MANAGEMENT UI: after it, `Client_Media_DB_v2.rename_keyword` and
`.merge_keywords` have zero production callers, and `.soft_delete_keyword`'s only
production caller is gone (the DB API itself stays; `get_keyword_usage_stats` keeps a
live read-only consumer in `Chat/scope_picker_listers.py`, and keyword *filtering*/
assignment during browse+ingest is unaffected). Nothing is lost — the loop has been
unreachable since 2025-08-02 — but if media keyword rename/merge/delete should exist
as a product feature, it needs a deliberate owner-commissioned rebuild on a live
surface (Library's media canvas), not this corpse.

**What was removed.** `Widgets/collections_tag_window.py` (CollectionsTagWindow,
KeywordRenameDialog, KeywordMergeDialog, nested DeleteConfirmationModal +
BUNDLED_SCREEN_CSS); `Event_Handlers/collections_tag_events.py` (3 events, 3 handlers,
`load_keyword_statistics` — zero callers, `_media_db_off_loop`); `app.py`'s
`on_collections_tag_message` block + the now-orphaned `from textual.message import
Message` (its only use); `Tests/UI/test_tag_action_recovery_tooltips.py`;
`Tests/Event_Handlers/test_collections_tag_events.py`; the CollectionsTagWindow test +
exclusive imports from `Tests/UI/test_bulk_selection_tooltips.py` (kept the
NoteSelectionDialog test — that widget is LIVE via `UI/STTS_Window.py:637` /
`stts_screen.py`, a deliberate deviation from the filing's wholesale-delete list) and
from `Tests/Widgets/test_reactive_default_aliasing.py` (rest of the guard intact:
5 tests collect and pass). Regenerated the lifted CSS bundles (`build_css.py`): only
the DeleteConfirmationModal block left `screen_css_scoped.tcss`; sync guard green.
Hand-edited `Docs/security/production-diagnostic-inventory.json` per the 16196/16835
playbook: removed the two TASK-494 owner rows (call_count 10 + 3), `owner_files`
503→501, `task_494_calls` 6991→6978.

**Deliberately left** (16835 precedent — chrome/CSS owned by task-2851's retired media
surface, shared selectors, and historical records): the `media-nav-collections-tags`
button + the `["collections-tags", "multi-item-review"]` skip-list in
`MediaWindow_v2.py:2391`; the `_media.tcss`/`tldw_cli_modular.tcss` Collections/Tags
CSS section (`selection-info` is also NoteSelectionDialog's, `usage-stats` also
scope_picker_listers'); mentions in merged task files, `lessons-textual.md`, and two
historical plan docs. No `Docs/User_Guide` page documents the affordance — nothing to
update.

**Gates** (venv, PYTHONPATH+cwd pinned to the worktree, outputs read from files).
Targeted: `test_reactive_default_aliasing.py` + `test_bulk_selection_tooltips.py` +
`test_css_bundle_sync_guard.py` + `Tests/Event_Handlers/` → **61 passed, 1 skipped
(pre-existing env skip), 0 failed**; `Tests/Architecture/
test_persistent_diagnostic_inventory.py` → **64 passed, 1 failed (pre-existing — see
below)**; `Tests/UI/test_screen_navigation.py` → **128 passed, 1 failed (pre-existing —
see below)**. Repo-wide `--collect-only -q`: **51470 before → 51463 after** (exactly
the 7 removed tests), zero collection errors. `ruff check` clean on touched files
(`ruff format --check` already failed on all three at base — pre-existing, whole-file
reformats out of scope). Final grep for every removed name over code/config/css:
zero hits.

**Two pre-existing dev reds documented, NOT absorbed** (both reproduced bit-identically
at pristine `25500ad87` in a throwaway baseline worktree): (1)
`test_production_diagnostic_inventory_and_sink_topology_are_unchanged` — dev's
committed inventory drifts from a rebuild by three rows this branch never touched
(`DB/Client_Media_DB_v2.py` 354→338 calls, `UI/Screens/library_screen.py` 110→109,
missing row for `UI/Library_Modules/library_media_browse_controller.py` +2; recent
library PRs, e.g. `1ba3d4755`/`b4ebe85e8`-series, landed diagnostic changes after the
`d64608b84` inventory regen without updating it). This branch's residual drift is
byte-identical to base's — the hand-edit here contributes zero new drift (base rebuild
504/6976 vs committed 503/6991; branch rebuild 502/6963 = base rebuild minus exactly
this task's two rows). (2) `test_screen_navigation.py::
test_action_library_media_viewer_back_returns_to_list_and_refocuses_it` — same
`LookupError: ContextVar active_app` at base and on branch.
<!-- SECTION:NOTES:END -->
