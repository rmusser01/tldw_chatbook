---
id: TASK-1494
title: The reader's full page and previous snapshot affordances were never built
status: Done
assignee: []
created_date: '2026-07-30 13:30'
labels:
  - watchlists
  - spec-divergence
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Watchlists design spec's Content-pane mockup
(`Docs/superpowers/specs/2026-07-25-watchlists-console-rebuild-design.md`, "Content pane") promises
`[full page]` and `[previous snapshot]` affordances on a change item, reading from `url_snapshots`.
Phase D shipped both renderers without those buttons, and no reference to either exists anywhere in
the UI (verified by grep during TASK-1393). Since TASK-1343 made `content` the diff, the reader
shows *what changed* but offers no way to see the page it changed on — the data is stored and
unreachable.

The storage side is ready and deliberately protected: snapshots are per-(subscription, url) with
the newest 3 kept (TASK-1393, `_SNAPSHOTS_KEPT_PER_URL` — slot 2 exists specifically for this
affordance), and the `[previous snapshot]` needs the second-newest row for the item's URL.

Do not render `raw_html` as markup or hyperlinks — remote content; the reader's escaping decisions
are documented in `content_pane.py` and TASK-1348.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A change item in the reader offers [full page] and [previous snapshot], reading the newest and second-newest url_snapshots rows for the item's URL
- [x] #2 When no previous snapshot exists (first check, or pruned by cap), the affordance degrades honestly rather than erroring or hiding silently
- [x] #3 Remote snapshot content renders as text, never as markup or live hyperlinks
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. DB: `Subscriptions_DB.get_url_snapshots(subscription_id, url, limit=2)` → newest-first rows
   (id, extracted_content, created_at) `ORDER BY created_at DESC, id DESC` (SAME order as
   `_store_snapshot`'s prune, so "newest"/"second-newest" agree with what is kept). Service async
   wrapper `LocalWatchlistsService.get_url_snapshots(source_id, url, limit=2)`.
2. ContentPane: on a `change`-kind item, render compact `[full page]` / `[previous snapshot]`
   affordances (mark-unread button precedent); press posts `ViewSnapshotRequested(item, which)`.
   Article items show neither. Pane has no DB — honest degradation happens at the screen.
3. Screen `@on(ViewSnapshotRequested)`: fetch via service; newest = full page, second-newest =
   previous. Absent (first check / pruned) → honest toast (markup=False), no empty modal (AC#2).
4. `SnapshotViewModal(ModalScreen)`: url + captured-at header, snapshot `extracted_content` as
   `Static(Text(raw))` — Text never parses markup, NO Markdown, NO hyperlinks (AC#3).
5. Tests: query scoping/order + fewer-than-limit; pane shows affordances only on change items +
   posts the right message; screen full-page→modal, previous-with-one-snapshot→honest toast; AC#3
   remote markup renders literally.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Built exactly as planned, no deviations.

- `DB/Subscriptions_DB.py`: `get_url_snapshots(subscription_id, url, *, limit=2)` — parameterized
  `SELECT id, extracted_content, created_at ... ORDER BY created_at DESC, id DESC LIMIT ?`, the same
  ordering `_store_snapshot`'s prune and `URLMonitor.check_url`'s baseline SELECT both use (the
  "TASK-1393 ordering pact"), so "newest"/"second-newest" here can never disagree with what the prune
  actually kept.
- `Subscriptions/local_watchlists_service.py`: `get_url_snapshots` is a thin passthrough — checked
  first whether this service's reads go through `asyncio.to_thread` (they do not, anywhere in the
  file) and matched that convention rather than introducing a one-off thread hop.
- `UI/Watchlists_Modules/content_pane.py`: `ContentPane.compose` yields `#content-full-page-button`/
  `#content-previous-snapshot-button` (both `compact=True`) only when `content_kind ==
  CONTENT_KIND_CHANGE` (imported from `item_persist`, the module that owns the vocabulary). New
  `ViewSnapshotRequested(item, which)` message, posted on press; the pane still holds no DB handle.
- `UI/Screens/watchlists_collections_screen.py`: new `_local_watchlists_service()` accessor
  (`getattr(app_instance, "local_watchlists_service", None)`) — `url_snapshots` is local-only storage
  with no server-backend equivalent, so this reaches the service directly rather than through
  `self._controller` (`WatchlistsBackendController`), which exists to route local/server and has no
  reason to grow a local-only method for one read. `@on(ViewSnapshotRequested)` dispatches a worker
  (`group="wl-view-snapshot"`, no `exclusive=True` — mirrors `handle_kept_briefings_requested`'s own
  reasoning: cancelling a modal-owning worker mid-view would strand the modal). `_open_snapshot_view`
  resolves `which` to index 0 (full page) or 1 (previous) into the fetched rows; an absent index
  degrades to an honest `markup=False` toast via `_notify_watchlists` and returns — never an empty
  modal (AC#2).
- `UI/Watchlists_Modules/snapshot_view_modal.py` (new): `SnapshotViewModal(ModalScreen[None])`, styled
  in `css/features/_watchlists.tcss` (`#svm-*`, bundle regenerated via `build_css.py`). Body renders
  through `rich.text.Text` (`_snapshot_body`), never `Markdown`/`markup=True`/a bare `str` handed to
  `Static` — `Text.append`/the constructor never parses Rich markup, so a scraped page's
  `[bold red]x[/]`-shaped fragment paints as literal characters (AC#3), matching the doctrine
  `content_pane.render_article`/TASK-1348/`KeptBriefingsModal` already state for the same class of
  remote content.

Tests (63 new, all passing): 5 in `Tests/DB/test_subscriptions_db.py` (order, tie-break, scoping,
fewer-than-limit, empty); 8 in `Tests/UI/test_watchlists_content_pane.py` covering compose-gating,
message-posting, and full-shell (`DestinationHarness`) screen behaviour — full-page opens the newest
row, previous opens the second-newest, a lone snapshot degrades to the AC#2 toast with no modal, and
AC#3's markup-literalness assertion runs through a real `rich.console.Console` (plain text present,
no ANSI style/hyperlink codes) rather than a `str(Text)` proxy. Three mutations applied and reverted
by hand (query `ORDER BY` → ASC, affordance gate → always-true, modal body → `markup=True`) each
turned exactly the test(s) that should catch it red, then were reverted; `git status --short` clean
of mutation residue afterward.

Modified/added files: `tldw_chatbook/DB/Subscriptions_DB.py`,
`tldw_chatbook/Subscriptions/local_watchlists_service.py`,
`tldw_chatbook/UI/Watchlists_Modules/content_pane.py`,
`tldw_chatbook/UI/Watchlists_Modules/snapshot_view_modal.py` (new),
`tldw_chatbook/UI/Screens/watchlists_collections_screen.py`,
`tldw_chatbook/css/features/_watchlists.tcss`, `tldw_chatbook/css/tldw_cli_modular.tcss` (regenerated),
`Tests/DB/test_subscriptions_db.py`, `Tests/UI/test_watchlists_content_pane.py`.

Left In Progress per dispatch instruction (not marking Done from this session).
