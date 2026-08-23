# Watchlists Collapsible Reader Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Watchlists Reader the permanent centre of a NetNewsWire-style Read screen while Navigation, Feed Items, and Inspector collapse independently through persistent five-column ASCII grips.

**Architecture:** Keep `WatchlistsCollectionsScreen` as the sole controller. Store only the user's preferred side-pane layout; derive responsive and Article Focus layouts with a pure resolver; pass only the effective layout and explicit Read/management mode into a Watchlists-local workbench. Preserve the already-shipped article list, Smart Feeds, search, refresh, pagination, mutation, selection, and scroll contracts. Do not introduce a shared Media/Watchlists split-pane abstraction in this change.

**Tech Stack:** Python 3.11+, Textual 8.x, pytest/pytest-asyncio, Rich/Textual production CSS bundle, Backlog.md.

**Delivery base:** Work only in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-21281-watchlists-collapsible-reader-layout` on `codex/task-21281-watchlists-collapsible-reader-layout`. It is based on `origin/dev` commit `527152ad3`; approved documentation was transplanted as `ade0a5ab7`, `6af8780f5`, `4561c4199`, and `730d908d2`. Do not edit, clean, reset, or rebase the unrelated dirty `feat/task-3401-video-generation-foundation` worktree.

**ADR required:** no new ADR

**ADR path:** `backlog/decisions/042-watchlists-reader-first-ia.md`

**Reason:** ADR-042 already owns this long-lived Watchlists information architecture and contains the approved 2026-08-23 permanent-Reader/collapsible-side-pane amendment. This plan directly implements that accepted amendment without creating another architectural boundary.

---

## Task 1: Replace collapse/solo state with preferred side-pane state and a pure responsive resolver

**Files:**

- Modify: `tldw_chatbook/UI/Watchlists_Modules/region_layout.py`
- Modify: `Tests/Watchlists/test_region_layout.py`
- Create: `Tests/Watchlists/test_watchlists_responsive_layout.py`

- [ ] **Step 1: Rewrite the pure-state tests around the approved pane model.**

  Replace the old Reader/Content toggle and centre-solo assertions in `test_region_layout.py` with tests that pin:

  - `COLLAPSIBLE_REGIONS == (LEFT_RAIL, ITEMS, RIGHT_RAIL)`;
  - the new strict preferred-layout toggle rejects `CONTENT`;
  - Navigation, Feed Items, and Inspector toggle independently;
  - the responsive resolver never creates or persists solo/transient state;
  - the legacy `toggle`/`solo` accessors remain only as transitional compatibility until Task 7.

- [ ] **Step 2: Add failing boundary tests for the effective-layout resolver.**

  In `test_watchlists_responsive_layout.py`, cover these exact contracts using the declared widths in the design:

  - Read with every side pane preferred open fits at 145 columns; at 144 Inspector is effectively collapsed.
  - With Inspector collapsed, Navigation and Feed Items fit at 115; at 114 Navigation also collapses.
  - Feed Items alone fits at 91; at 90 every side pane is collapsed.
  - Management with Navigation and Inspector open fits at 108; 107 collapses Inspector; 77 collapses Navigation too.
  - Article Focus collapses every mounted side pane but leaves the preferred object unchanged.
  - A responsive priority target is protected until every other eligible pane has collapsed.
  - A preferred-closed pane stays closed, repeated resolution is idempotent, and sub-60 widths return all grips collapsed without raising.

- [ ] **Step 3: Run the new pure tests and confirm they fail for the old centre-solo model.**

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Watchlists/test_region_layout.py Tests/Watchlists/test_watchlists_responsive_layout.py -q
  ```

  Expected: failures for missing `COLLAPSIBLE_REGIONS`/resolver and old `CONTENT`/solo behavior.

- [ ] **Step 4: Implement the smallest pure model and resolver.**

  In `region_layout.py`:

  - keep `Region` values stable for existing factories and persisted strings;
  - define `COLLAPSIBLE_REGIONS`, Read/management mounted-side-pane order, fixed grip width 5, pane minimums 24/32/30, centre comfort 44, and default collapse priorities;
  - add a strict `toggle_preferred()` operation that rejects non-collapsible regions;
  - add `resolve_effective_layout(preferred, *, width, read_mode, article_focus, priority_target) -> RegionLayout`;
  - start from preferred collapses, apply Article Focus first, otherwise collapse expanded mounted panes until the declared sum fits;
  - reorder the collapse candidates so `priority_target` is last, and never collapse `CONTENT` or a management canvas;
  - once every side pane is collapsed, return that layout even below the comfort floor so CSS may give the permanent centre `min-width: 0`.

  Keep the old `solo_region`, `_pre_solo`, `solo()`, `toggle()`, and
  `collapsed_for_persistence()` members temporarily so the live screen remains importable and
  composable through Tasks 1–6. Mark them as migration-only in their docstrings; the resolver and
  store must not create new solo state. Task 7 removes them immediately after migrating the final
  screen call sites and tests to preferred/effective state.

- [ ] **Step 5: Run the pure tests.**

  Run the Step 3 command. Expected: pass.

- [ ] **Step 6: Commit the pure layout model.**

  ```bash
  git add tldw_chatbook/UI/Watchlists_Modules/region_layout.py Tests/Watchlists/test_region_layout.py Tests/Watchlists/test_watchlists_responsive_layout.py
  git commit -m "feat(watchlists): model preferred and responsive pane layouts"
  ```

## Task 2: Version and atomically normalize the persisted preferred layout

**Files:**

- Modify: `tldw_chatbook/UI/Watchlists_Modules/region_layout_store.py`
- Modify: `Tests/Watchlists/test_region_layout_store.py`

- [ ] **Step 1: Replace Phase-D marker tests with versioned-normalization tests.**

  Cover:

  - no saved keys returns Navigation/Feed Items open and Inspector collapsed;
  - an explicit saved empty list means every side pane expanded;
  - valid `left_rail`, `items`, and `right_rail` values round-trip;
  - old `content`, retired `feeds`, unknown strings, strings-as-singletons, and non-sequences normalize safely;
  - a stale/missing layout version triggers exactly one `save_settings_to_cli_config` call containing both normalized `collapsed_regions` and `layout_version`;
  - the same mutation deletes the retired `content_reader_migrated` key;
  - a `False` return or exception leaves the version logically stale so a second load retries;
  - `save_region_layout` writes only valid side panes plus the current version and returns the writer's success.

- [ ] **Step 2: Run the store tests and confirm the old single-key marker implementation fails.**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Watchlists/test_region_layout_store.py -q
  ```

- [ ] **Step 3: Implement version 2 normalization with one configuration mutation.**

  In `region_layout_store.py`:

  - import `save_settings_to_cli_config` instead of the single-key writer;
  - define `LAYOUT_VERSION = 2`, `layout_version`, `collapsed_regions`, and the retired marker name as constants;
  - normalize every raw value to a `RegionLayout` containing only `COLLAPSIBLE_REGIONS`;
  - when the version is stale or the stored value differs from normalized output, call:

    ```python
    save_settings_to_cli_config(
        {"watchlists": {
            "collapsed_regions": normalized_values,
            "layout_version": LAYOUT_VERSION,
        }},
        delete_keys={"watchlists": ("content_reader_migrated",)},
    )
    ```

  - apply the safe normalized value in memory even when the write fails; do not write a separate marker;
  - make `save_region_layout()` use the same two-key atomic mutation and preserve its Boolean result.

- [ ] **Step 4: Run the store and config mutation contract tests.**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Watchlists/test_region_layout_store.py Tests/test_config_save_settings_semantics.py Tests/test_config_delete_settings.py -q
  ```

- [ ] **Step 5: Commit persistence normalization.**

  ```bash
  git add tldw_chatbook/UI/Watchlists_Modules/region_layout_store.py Tests/Watchlists/test_region_layout_store.py
  git commit -m "feat(watchlists): version preferred pane layout"
  ```

## Task 3: Add the focused, clickable five-column ASCII grip

**Files:**

- Create: `tldw_chatbook/UI/Watchlists_Modules/pane_grip.py`
- Create: `Tests/Watchlists/test_watchlists_pane_grip.py`

- [ ] **Step 1: Write grip direction, semantics, and rendering tests.**

  Use a minimal Textual app and assert:

  - Navigation/Feed Items render `--->` while collapsed and `<---` while expanded;
  - Inspector renders `<---` while collapsed and `--->` while expanded;
  - each grip is focusable and clicking or pressing Enter posts one `RegionToggled(region)` message;
  - tooltip/name copy says `Expand/Collapse <pane>` rather than exposing only arrows;
  - explicit inline `line_pad == 0` defeats Textual `Button` padding;
  - the direction/accessible-copy update changes the existing widget in place.

- [ ] **Step 2: Run the new grip test and confirm it fails because the widget does not exist.**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Watchlists/test_watchlists_pane_grip.py -q
  ```

- [ ] **Step 3: Implement `WatchlistsPaneGrip`.**

  Subclass `Button`, define the destination-local `RegionToggled` message beside it, accept `region` and `expanded`, set `compact=True`, set `styles.line_pad = 0`, compute visible label/tooltip/name from the pane and action, and post the message exactly once. Expose an `expanded` update method/reactive that relabels in place without changing widget identity. Import that message from `watchlists_workbench.py` and the screen so no circular import is introduced. Keep all code Watchlists-local.

- [ ] **Step 4: Run the grip tests.**

- [ ] **Step 5: Commit the grip widget and its non-CSS tests.**

  ```bash
  git add tldw_chatbook/UI/Watchlists_Modules/pane_grip.py Tests/Watchlists/test_watchlists_pane_grip.py
  git commit -m "feat(watchlists): add ASCII pane grips"
  ```

## Task 4: Rebuild the workbench around a permanent horizontal centre anchor

**Files:**

- Modify: `tldw_chatbook/UI/Watchlists_Modules/watchlists_workbench.py`
- Modify: `tldw_chatbook/css/features/_watchlists.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Tests/Watchlists/test_watchlists_workbench.py`
- Modify: `Tests/Watchlists/test_watchlists_scoped_rebuilds.py`

- [ ] **Step 1: Replace vertical-stack expectations with explicit Read/management DOM contracts.**

  Add or update tests to assert the exact body order:

  - Read expanded: Navigation body, Navigation grip, Feed Items body, Feed Items grip, permanent Reader body, Inspector grip, Inspector body;
  - Read collapsed panes leave their grip mounted and remove only that pane body;
  - management: Navigation body/grip, permanent active `Region.ITEMS` canvas, Inspector grip/body; no Feed Items grip and no Reader;
  - header remains above the horizontal body on every tab;
  - `Region.CONTENT` is always mounted in Read even if an old effective layout contains it;
  - toggling one pane preserves every unaffected pane/grip instance and the permanent Reader instance;
  - `refresh_region_content` and `refresh_header_content` still replace only their named factory output and remain failure-safe;
  - changing Read/management mode through `apply_section_view` changes only the required centre factories and does not recompose the full workbench.

  Remove obsolete tests for stacked 10–50/20–50 row caps, centre solo classes, collapsed text headers, and outer-centre scrolling. Keep the Reader's own body-scroll/action/footer containment tests that remain meaningful.

- [ ] **Step 2: Run the workbench/scoped-rebuild tests and confirm the old `VerticalScroll` composition fails.**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Watchlists/test_watchlists_workbench.py Tests/Watchlists/test_watchlists_scoped_rebuilds.py -q
  ```

- [ ] **Step 3: Introduce explicit workbench mode and permanent-centre composition.**

  In `watchlists_workbench.py`:

  - change the root to `Vertical`: optional `#wl-centre-status` first, then a `Horizontal#wl-workbench-body`;
  - add explicit `read_mode: bool` while temporarily accepting the current `hidden=` constructor/
    `apply_section_view` keyword as a compatibility adapter (`CONTENT` hidden means management;
    otherwise Read), because the live screen migrates in Task 7;
  - in Read, always mount the `Region.CONTENT` factory as the flexing centre and conditionally mount Navigation/Feed Items/Inspector bodies around their always-mounted grips;
  - in management, always mount the `Region.ITEMS` factory as the centre canvas with only Navigation and Inspector grips;
  - keep body ids (`#wl-region-<value>`) and factory-based rebuild safety where existing screen/test contracts rely on them;
  - update grip labels in place and mount/remove only a pane body when effective collapse changes;
  - remove generic collapsed-header/sole-centre rendering machinery, but retain the
    `collapsed_suffixes=` constructor keyword and `set_collapsed_suffixes(...)` as documented
    no-op compatibility because the live screen still calls them through Task 6;
  - keep `refresh_region_content`, `refresh_header_content`, and `apply_section_view` incremental—no `recompose=True` regression.

  The `hidden=` and collapsed-suffix compatibility adapters must be covered by a real-screen
  scoped-rebuild test and removed in Task 7 when their last screen call sites are replaced. This
  makes the Task 4 checkpoint runnable rather than leaving the screen and workbench APIs out of
  sync.

  In the same step, make the minimum structural CSS change required for this DOM to be usable:
  vertical workbench root, horizontal `#wl-workbench-body`, flexing permanent centre, fixed
  five-column grips, and the declared pane target/min widths. Remove the old vertical-stack/collapsed-
  header selectors, regenerate `tldw_cli_modular.tcss`, and run the bundle-sync guard. Task 6 adds
  exhaustive boundary/compositor coverage and any evidence-driven refinements; it must not be the
  first commit in which the new DOM can render.

- [ ] **Step 4: Run the focused workbench tests.**

  Run the Step 2 command plus:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
  ```

  Expected: pass. Do not commit a workbench DOM that still depends on Task 6 to become renderable.

- [ ] **Step 5: Commit the workbench composition.**

  ```bash
  git add tldw_chatbook/UI/Watchlists_Modules/watchlists_workbench.py tldw_chatbook/css/features/_watchlists.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/Watchlists/test_watchlists_workbench.py Tests/Watchlists/test_watchlists_scoped_rebuilds.py
  git commit -m "feat(watchlists): anchor Reader in horizontal workbench"
  ```

## Task 5: Reduce Reader chrome to the approved actions and move advanced change actions to Inspector

**Files:**

- Modify: `tldw_chatbook/UI/Watchlists_Modules/content_pane.py`
- Modify: `tldw_chatbook/UI/Watchlists_Modules/inspector_pane.py`
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Modify: `Tests/Watchlists/test_watchlists_workbench.py`
- Modify: `Tests/Watchlists/test_watchlists_collections_screen.py`
- Modify: `Tests/Watchlists/test_watchlists_scoped_rebuilds.py`

- [ ] **Step 1: Write the approved empty-state and action-surface tests.**

  Assert:

  - no selection renders exactly `Select a feed item to display it here.`;
  - the Reader action row contains only `content-star-button`, `content-mark-unread-button`, and `content-open-button` in that order;
  - Ingest and Queue/Unqueue remain present in Inspector for item entities;
  - change items expose Full page and Previous snapshot through Inspector and still post the existing snapshot-view request;
  - Reader has no Expand/Restore button because `Z` owns Article Focus;
  - the footer's position and Next unread affordance remain unchanged.
  - both the keyboard and button Open paths validate first, schedule the actual
    `webbrowser.open` call on a thread worker, and report a worker failure without disturbing Reader.

- [ ] **Step 2: Run the focused Reader/Inspector tests and confirm they fail.**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Watchlists/test_watchlists_workbench.py Tests/Watchlists/test_watchlists_collections_screen.py Tests/Watchlists/test_watchlists_scoped_rebuilds.py -q
  ```

- [ ] **Step 3: Simplify `ContentPane` and preserve advanced capability in Inspector.**

  - change the empty copy;
  - remove Reader Ingest, Queue, Expand, Full page, and Previous snapshot buttons and their reader-only dispatch branches;
  - remove `expanded`, `watch_expanded`, and `ExpandReaderRequested`;
  - move `ViewSnapshotRequested` to `inspector_pane.py` beside the other Inspector action messages;
  - render Full page/Previous snapshot Inspector buttons only for a selected `content_kind == "change"` item;
  - keep the screen's existing snapshot modal handler but import the message from Inspector;
  - remove the obsolete expand-reader handler and `_sync_reader_expanded_state` calls;
  - retain the existing shared screen handlers for status, star, open, ingest, and queue so semantics do not fork;
  - split `_open_item_in_browser` into UI-thread validation/dispatch and a thread worker that calls
    `webbrowser.open`, then acknowledges failure through `call_from_thread`.

- [ ] **Step 4: Run the focused Reader/Inspector tests.**

- [ ] **Step 5: Commit the Reader surface change.**

  ```bash
  git add tldw_chatbook/UI/Watchlists_Modules/content_pane.py tldw_chatbook/UI/Watchlists_Modules/inspector_pane.py tldw_chatbook/UI/Screens/watchlists_collections_screen.py Tests/Watchlists/test_watchlists_workbench.py Tests/Watchlists/test_watchlists_collections_screen.py Tests/Watchlists/test_watchlists_scoped_rebuilds.py
  git commit -m "refactor(watchlists): keep core actions in Reader"
  ```

## Task 6: Apply production geometry for fixed grips and flexing Reader

**Files:**

- Modify: `tldw_chatbook/css/features/_watchlists.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Tests/Watchlists/test_watchlists_workbench.py`
- Modify: `Tests/Watchlists/test_watchlists_pane_grip.py`

- [ ] **Step 1: Finish production-CSS geometry tests before editing CSS.**

  Use harnesses whose `CSS_PATH` is the real generated bundle. At 145, 144, 115, 114, 91, 90, 60, and a sub-floor width, assert:

  - every mounted grip has outer width/min/max 5 and all four arrow characters occupy its four label columns beside the divider;
  - Navigation target/min widths are 28/24, Feed Items 40/32, Inspector 34/30;
  - Reader/management canvas gets the remaining width, has `min-width: 0`, and is never absent;
  - pane/grip/centre sibling regions are contained, non-overlapping, in the approved order;
  - `#wl-workbench-body.max_scroll_x == 0` and no child extends beyond its content region;
  - Reader body scroll moves without shifting actions, footer, grips, or adjacent panes;
  - focus styling does not widen or clip a grip.

- [ ] **Step 2: Run geometry tests and confirm old CSS fails.**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Watchlists/test_watchlists_pane_grip.py Tests/Watchlists/test_watchlists_workbench.py -q
  ```

- [ ] **Step 3: Replace stacked-centre CSS with the approved horizontal geometry.**

  In `_watchlists.tcss`:

  - make the workbench vertical and `#wl-workbench-body` horizontal, `width: 100%`, `height: 1fr`, `min-height/min-width: 0`, without horizontal scrolling;
  - set expanded pane target/min values to 28/24, 40/32, and 34/30;
  - set every `.watchlists-pane-grip` width/min/max to 5, height 100%, padding/margin 0, one divider border, and `content-align: center middle`;
  - make Reader/management centre `width: 1fr; min-width: 0; height: 100%; min-height: 0`;
  - remove obsolete stacked height caps, collapsed 16-column header rules, and sole-centre rules;
  - preserve the Reader's internal `VerticalScroll`, fixed action row, and fixed footer rules.

- [ ] **Step 4: Regenerate and verify the bundle.**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
  ```

- [ ] **Step 5: Run geometry tests and inspect at least one compositor capture for a full arrow label and non-overlap.**

- [ ] **Step 6: Commit source CSS, generated CSS, and geometry tests together.**

  ```bash
  git add tldw_chatbook/css/features/_watchlists.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/Watchlists/test_watchlists_pane_grip.py Tests/Watchlists/test_watchlists_workbench.py
  git commit -m "style(watchlists): size collapsible reader panes"
  ```

## Task 7: Wire preferred/effective/Article Focus state into the screen

**Files:**

- Modify: `tldw_chatbook/UI/Watchlists_Modules/region_layout.py`
- Modify: `tldw_chatbook/UI/Watchlists_Modules/region_layout_store.py`
- Modify: `tldw_chatbook/UI/Watchlists_Modules/watchlists_workbench.py`
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Modify: `Tests/Watchlists/test_region_layout.py`
- Modify: `Tests/Watchlists/test_region_layout_store.py`
- Modify: `Tests/Watchlists/test_watchlists_workbench.py`
- Modify: `Tests/Watchlists/test_watchlists_collections_screen.py`
- Modify: `Tests/Watchlists/test_watchlists_scoped_rebuilds.py`
- Modify: `Tests/Watchlists/test_watchlists_cold_open_layout.py`
- Modify: `Tests/Watchlists/test_watchlists_pagination.py`

- [ ] **Step 1: Add failing controller tests for all three layout layers.**

  Cover:

  - `region_layout` is preferred state and the workbench receives a separately resolved effective state;
  - resize changes effective state only and calls no config writer;
  - `Z` on Read toggles Article Focus, every side pane collapses effectively, and a second `Z` restores the exact preferred state;
  - `Z` off Read is refused without changing either state;
  - a grip/manual action during Article Focus exits focus first, then applies its action;
  - clicking a responsively collapsed but preferred-open grip protects that pane instead of closing its preference;
  - expanding a truly preferred-closed grip opens and persists it; collapsing an effective-open grip closes and persists it;
  - `z` toggles only a focused collapsible pane/grip, does nothing to Reader or a management canvas, and `[`/`]` continue to target Navigation/Inspector;
  - switching tabs preserves the one Inspector preference and parks Feed Items preference off Read;
  - a pane collapsed by resize hands focus to its grip; reopening returns focus inside the pane when possible;
  - repeated shrink/expand cycles are idempotent.

- [ ] **Step 2: Add persistence-worker failure/retry tests.**

  Pin that `_last_persisted_collapsed` advances only after `save_region_layout` returns `True`; `False`/exception retains the pending newest preferred value, and a later manual gesture schedules another attempt. Also pin last-request-wins behavior under rapid toggles.

- [ ] **Step 3: Add state-survival tests around effective layout changes.**

  Starting from a loaded ArticleListPane/Reader, record selected item id, committed scope, filter, search, loaded page count, visible-row anchor/focus, and Reader body scroll. Manually collapse/expand each side pane, trigger responsive collapses, and toggle Article Focus; assert those semantic values and live unaffected widget instances survive. Keep the existing rule that a genuine scope change clears Reader and never auto-selects/auto-reads a first item.

- [ ] **Step 4: Run the controller tests and confirm the old single-layout/solo behavior fails.**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Watchlists/test_watchlists_collections_screen.py Tests/Watchlists/test_watchlists_scoped_rebuilds.py Tests/Watchlists/test_watchlists_cold_open_layout.py Tests/Watchlists/test_watchlists_pagination.py -q
  ```

- [ ] **Step 5: Implement screen-owned preferred/effective state.**

  - keep `region_layout` as the persisted preferred `RegionLayout` to minimize call-site churn;
  - add non-persisted instance state for the effective layout,
    `_article_focus_active`, and `_responsive_priority_target` (do not add another class-level
    reactive default merely to hold derived state);
  - centralize `resolve_effective_layout(...)` in `_recompute_effective_layout`, called from mount, `Resize`, section switches, preferred toggles, and Article Focus changes;
  - pass `read_mode=self.active_section == "items"` and only the effective layout to `WatchlistsWorkbench`/`apply_section_view`;
  - clear the priority target when the full preferred layout fits again or the target is manually collapsed;
  - implement manual grip action from effective state: capture the grip's requested action before
    changing Article Focus; if it requested Open, exit focus and keep/open the preferred pane plus
    its priority target rather than accidentally closing the now-restored pane; if it requested
    Close, close the preferred pane and clear its priority target;
  - replace `action_solo_region` with `action_article_focus`; bind `Z` to Article Focus and update the binding label;
  - ensure only preferred manual gestures call `_schedule_layout_persist`;
  - hand focus to the relevant grip before an effectively hidden body is removed.

  At the end of this step, remove the Task 1 legacy `solo_region`, `_pre_solo`, `solo()`, permissive
  `toggle()`, and `collapsed_for_persistence()` compatibility plus Task 4's `hidden=` adapter. Change
  every remaining screen/workbench/store call site to strict preferred/effective APIs in the same
  commit. Remove the screen's collapsed-suffix construction/update calls and then remove Task 4's
  no-op `collapsed_suffixes=`/`set_collapsed_suffixes(...)` adapter. Update the Task 1 transitional
  pure tests to assert the final compatibility members are absent, update the workbench/store tests,
  and run an import/compose smoke test before the broader controller suite.

- [ ] **Step 6: Make ordinary preference persistence acknowledge success.**

  Give each requested write a monotonically increasing generation. The thread worker snapshots the latest generation/layout under the existing lock, calls `save_region_layout`, and uses `call_from_thread` to acknowledge the exact generation. Advance `_last_persisted_collapsed` and clear pending state only for a successful current generation; leave a failed current value pending/retryable; if a newer generation arrived while an older write ran, drain the newest value next. Never mutate Textual reactive state directly from the worker thread.

- [ ] **Step 7: Run the controller tests.**

  Run the Step 4 command plus the compatibility-removal suites:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Watchlists/test_region_layout.py Tests/Watchlists/test_region_layout_store.py Tests/Watchlists/test_watchlists_responsive_layout.py Tests/Watchlists/test_watchlists_workbench.py -q
  ```

- [ ] **Step 8: Commit the screen controller integration.**

  ```bash
  git add tldw_chatbook/UI/Watchlists_Modules/region_layout.py tldw_chatbook/UI/Watchlists_Modules/region_layout_store.py tldw_chatbook/UI/Watchlists_Modules/watchlists_workbench.py tldw_chatbook/UI/Screens/watchlists_collections_screen.py Tests/Watchlists/test_region_layout.py Tests/Watchlists/test_region_layout_store.py Tests/Watchlists/test_watchlists_responsive_layout.py Tests/Watchlists/test_watchlists_workbench.py Tests/Watchlists/test_watchlists_collections_screen.py Tests/Watchlists/test_watchlists_scoped_rebuilds.py Tests/Watchlists/test_watchlists_cold_open_layout.py Tests/Watchlists/test_watchlists_pagination.py
  git commit -m "feat(watchlists): derive responsive and focus layouts"
  ```

## Task 8: Close cross-tab, help, restart, and destination-shell regressions

**Files:**

- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Modify: `Tests/UI/test_destination_shells.py`
- Modify: `Tests/Watchlists/test_watchlists_collections_screen.py`
- Modify: `Tests/Watchlists/test_watchlists_cold_open_layout.py`
- Modify: `Tests/Watchlists/test_watchlists_scoped_rebuilds.py`
- Modify as required by failures: focused `Tests/Watchlists/test_watchlists_*` files that assert retired collapsed headers/solo behavior

- [ ] **Step 1: Add one seven-tab acceptance test.**

  Open Inspector on Read, visit Sources/Runs/Rules/Notifications/Artifacts/Overview, assert the same preferred Inspector state and correct `<---`/`--->` grip action in each tab, then collapse Inspector on a management tab and assert Read sees it collapsed. Also assert each management centre canvas remains mounted and Feed Items/grip are absent.

- [ ] **Step 2: Pin and implement honest Server-backed Read recovery.**

  Add a test that enters Read with `runtime_backend == "server"`, asserts the permanent centre shows
  the existing local-only explanation plus **Switch to Local**, and spies that no local item, Smart
  Feed count, search, or refresh query is issued under the Server label. Add Read to the screen's
  local-only section policy, keep the backend selector truthful/disabled for that state, and route
  Switch to Local through the normal backend change/load path before mounting local rows.

- [ ] **Step 3: Add an isolated-config restart test.**

  Under a temporary `HOME`, `XDG_CONFIG_HOME`, and `TLDW_CONFIG_PATH`, save a non-default preferred combination, construct a fresh screen/app, and assert the same three preferences return while responsive/Article Focus state does not. Assert a legacy `content` collapse is removed and does not return after the versioned write.

- [ ] **Step 4: Update help/footer copy and test it.**

  In `BINDINGS` and `action_show_help`, advertise only implemented actions:

  - `z`: toggle focused side pane;
  - `Z`: Article Focus on Read;
  - `[`/`]`: Navigation/Inspector;
  - Reader is permanent and has no collapse/Expand action.

  Do not bind any terminal-convention or global reserved key prohibited by ADR-031.

- [ ] **Step 5: Run the full focused Watchlists/destination shell set.**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Watchlists Tests/UI/test_destination_shells.py -q
  ```

  When an old test fails only because it asserts a retired collapsed header, vertical centre cap, Reader collapse, or solo contract, update it to the approved permanent-centre contract. Do not weaken unrelated Smart Feed, list, mutation, pagination, or snapshot assertions.

- [ ] **Step 6: Commit the cross-tab/help closeout.**

  ```bash
  git add tldw_chatbook/UI/Screens/watchlists_collections_screen.py Tests/Watchlists Tests/UI/test_destination_shells.py
  git commit -m "test(watchlists): cover collapsible reader layout end to end"
  ```

## Task 9: Verify live behavior, document evidence, and complete Backlog hygiene

**Files:**

- Modify: `Docs/superpowers/specs/2026-08-23-watchlists-netnewswire-reader-collapsible-rails-design.md` only if implementation facts differ
- Modify: `backlog/decisions/042-watchlists-reader-first-ia.md` only if an accepted consequence needs factual clarification
- Modify: `backlog/tasks/task-21281 - Make-Watchlists-Reader-permanent-with-independently-collapsible-panes.md`
- Modify only if a genuine reusable incident occurred: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md`

- [ ] **Step 1: Run focused tests, CSS integrity, lint, and whitespace checks from the dedicated worktree.**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Watchlists Tests/UI/test_destination_shells.py -q
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/UI/Watchlists_Modules tldw_chatbook/UI/Screens/watchlists_collections_screen.py Tests/Watchlists Tests/UI/test_destination_shells.py
  git diff --check
  ```

- [ ] **Step 2: Run the broader regression suite or establish an identical-base failure comparison.**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q
  ```

  If unrelated repository/environment failures occur, rerun the exact failing node ids against `origin/dev` in an isolated comparison worktree and record only identical failure sets as pre-existing. Do not claim success from a different command or a bare-widget harness.

- [ ] **Step 3: Perform isolated-profile live Textual verification.**

  Launch with absolute interpreter and isolated config paths. Verify Read at 145, 144, 115, 114, 91, 90, and 60 columns plus one sub-floor width; click each grip using terminal character positions (not UTF-8 byte offsets); verify arrow direction, focus, `z`, `Z`, `[`/`]`, Article Focus restoration, cross-tab Inspector state, restart persistence, Reader empty copy, selected article reading, three-action Reader row, and management centre survival. Confirm no compositor exception, overlap, or horizontal overflow.

- [ ] **Step 4: Self-review the final diff against every acceptance criterion.**

  Check especially: no Media/shared framework edits, no database/schema change, no Reader collapse path, no responsive/focus config writes, failed write retryability, production CSS rather than test-only CSS, and preservation of list/Reader semantic state.

- [ ] **Step 5: Update Backlog task 21281.**

  Add concise Implementation Notes with approach, decisions, modified files, commits, automated/live evidence, ADR-042 link, deviations, and any pre-existing failures. Check each acceptance criterion only after its evidence exists, then set status Done only if the repository Definition of Done is satisfied.

- [ ] **Step 6: Commit documentation/task closeout.**

  ```bash
  git add Docs/superpowers/specs/2026-08-23-watchlists-netnewswire-reader-collapsible-rails-design.md backlog/decisions/042-watchlists-reader-first-ia.md backlog/docs/lessons-testing-evidence.md backlog/docs/lessons-live-verification.md 'backlog/tasks/task-21281 - Make-Watchlists-Reader-permanent-with-independently-collapsible-panes.md'
  git commit -m "docs(watchlists): close collapsible reader layout task"
  ```
