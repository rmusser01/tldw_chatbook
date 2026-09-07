# Watchlists Reader-First Re-IA — Phase 1: The Reading Loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Watchlists screen's Read tab a working feed-reader loop: pick a scope in the rail → its items load → read with `j`/`k`/`space` → catch up with `a` (undo with `u`) — and make Read the screen's landing tab.

**Architecture:** All work happens in the worktree `.worktrees/watchlists-reader-first` (branch `feat/watchlists-reader-first`, off `origin/dev @ 06bf63a62`). New scope filters and bulk operations are added to `Subscriptions_DB` / `LocalWatchlistsService` / `WatchlistScopeService` as pass-through parameters (the backend controller already forwards `**kwargs`). The FEEDS region is removed from the five-region workbench; the tab strip moves to the centre header on every tab. No schema changes.

**Spec:** `Docs/superpowers/specs/2026-08-05-watchlists-reader-first-design.md` (approved 2026-08-05)

**ADR required:** yes
**ADR path:** `backlog/decisions/042-watchlists-reader-first-ia.md`
**Reason:** IA change to a shipped destination (Read-first landing, FEEDS region removal, ops-tab recession, new keymap, bulk status writes) — amends ADR-018's pane set for the second time. Created in Task 1, before any code change.

**Tech Stack:** Python 3.11+, Textual, SQLite (stdlib `sqlite3`, `RETURNING` supported — precedent `tldw_chatbook/Subscriptions/item_persist.py:138`), pytest + `textual.pilot` for widget tests.

**Conventions:**
- Run everything from the worktree root: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/watchlists-reader-first`
- Tests: `python3 -m pytest Tests/Watchlists/ -x -q` (the repo venv at the MAIN checkout `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv` provides deps if the worktree has none: `source /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/activate`)
- The `timeout` command is NOT available; use pytest's own timeouts.
- DB tests use real SQLite in-memory (repo convention). Find the existing home with `grep -rl "get_watchlist_item_counts" Tests/` and co-locate new DB tests there.
- Screen under change: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (9,526 lines — navigate by the line refs below; they are exact as of `06bf63a62`).
- Commits: `type(scope): summary` conventional style, e.g. `feat(watchlists): ...`. Reference the backlog task id once it exists.

---

### Task 1: ADR-042, backlog task, docs commit

**Files:**
- Create: `backlog/decisions/042-watchlists-reader-first-ia.md`
- Create: backlog task via CLI
- Already present (uncommitted): `Docs/superpowers/specs/2026-08-05-watchlists-reader-first-design.md`, this plan

- [ ] **Step 1: Write ADR-042**

Read `backlog/decisions/018-watchlists-tui-screen.md` for the house format, then write `042-watchlists-reader-first-ia.md`: Status: Accepted; Context: the 2026-07-25 rebuild shipped a management console with a reader bolted on — the reading loop (scope→items→catch-up) is missing; Decision: Read-first landing (tab `items` relabeled "Read", reordered first, default), FEEDS region removed from the workbench, inspector collapsed by default for new users, scope-plumbed item queries, bulk mark-all-read with session undo batch, new Read keymap (`m`/`a`/`u`/`space`); Consequences: amends ADR-018's pane set/IA; persisted `feeds` collapse state is silently dropped by the existing unknown-region guard in `region_layout_store.py:132-137`; section ids are unchanged so deep links keep working; spec at `Docs/superpowers/specs/2026-08-05-watchlists-reader-first-design.md`.

- [ ] **Step 2: Create the backlog task**

```bash
backlog task create "Watchlists reader-first re-IA, phase 1: reading loop" \
  -d "Make the Watchlists Read tab a daily-driver feed-reader loop per Docs/superpowers/specs/2026-08-05-watchlists-reader-first-design.md (ADR-042). Scope-plumbed items, per-feed unread badges, mark-all-read + undo, next-unread, Read-first landing." \
  --ac "Picking any rail node scopes the items list,Per-feed unread badges render in the tree,Mark-all-read is one key and undoable,Next-unread and read/unread toggle keys work,Read is the landing tab,Tests/Watchlists green" \
  -l watchlists,ux
```

Note the task id it prints; use it in commit messages (`task-NNNN`).

- [ ] **Step 3: Move the task to In Progress and add the plan reference**

```bash
backlog task edit <id> -s "In Progress" --plan "Execute Docs/superpowers/plans/2026-08-05-watchlists-reader-first-phase-1-reading-loop.md"
```

- [ ] **Step 4: Commit the docs**

```bash
git add backlog/decisions/042-watchlists-reader-first-ia.md \
        Docs/superpowers/specs/2026-08-05-watchlists-reader-first-design.md \
        Docs/superpowers/plans/2026-08-05-watchlists-reader-first-phase-1-reading-loop.md
git commit -m "docs(watchlists): reader-first re-IA spec, ADR-042, phase 1 plan"
```

---

### Task 2: DB scope filters + per-source counts

**Files:**
- Modify: `tldw_chatbook/DB/Subscriptions_DB.py` (`get_new_items` at :1763, add `get_source_item_counts` after `get_watchlist_item_counts` at :1041-1098)
- Test: co-locate with the existing Subscriptions_DB tests (find: `grep -rl "get_watchlist_item_counts" Tests/`)

- [ ] **Step 1: Write the failing tests**

```python
def test_get_new_items_filters_by_watchlist(db_with_memberships):
    # two sources, only one in the watchlist; items on both
    rows = db_with_memberships.get_new_items(status=None, watchlist_id=WATCHLIST_ID)
    assert rows and all(r["subscription_id"] == IN_WATCHLIST_SOURCE for r in rows)

def test_get_new_items_unassigned_only(db_with_memberships):
    rows = db_with_memberships.get_new_items(status=None, unassigned_only=True)
    assert rows and all(r["subscription_id"] == UNASSIGNED_SOURCE for r in rows)

def test_get_new_items_statuses_multi(db_with_memberships):
    rows = db_with_memberships.get_new_items(status=None, statuses=["new", "ingested"])
    assert {r["status"] for r in rows} <= {"new", "ingested"}

def test_get_new_items_rejects_status_and_statuses(db_with_memberships):
    with pytest.raises(ValueError):
        db_with_memberships.get_new_items(status="new", statuses=["new"])

def test_get_source_item_counts(db_with_memberships):
    counts = db_with_memberships.get_source_item_counts()
    assert counts[SOURCE_WITH_ITEMS]["total"] == 3
    assert counts[SOURCE_WITH_ITEMS]["unread"] == 2  # statuses: new, new, reviewed
```

Build `db_with_memberships` with the same fixture style the found test file uses (real in-memory SQLite; `add_subscription`, `persist_subscription_item` or the fixture's own insert helper, and `INSERT INTO watchlists / watchlist_sources` for membership — `watchlists`/`watchlist_sources` are created by `_initialize_schema`, `Subscriptions_DB.py:719`/`:755`).

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest <testfile> -x -q`
Expected: FAIL (`TypeError: get_new_items() got an unexpected keyword argument 'watchlist_id'`, `AttributeError: get_source_item_counts`)

- [ ] **Step 3: Implement**

In `get_new_items` (:1763) extend the signature and predicate builder (values stay bound parameters; only fixed predicate text is assembled — the established pattern in that method):

```python
def get_new_items(
    self,
    subscription_id: Optional[int] = None,
    status: Optional[str] = "new",
    limit: int = 100,
    run_id: Optional[int] = None,
    watchlist_id: Optional[int] = None,
    unassigned_only: bool = False,
    statuses: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
```

```python
        if status is not None and statuses is not None:
            raise ValueError("Pass either status or statuses, not both.")
        ...
        if watchlist_id is not None:
            predicates.append(
                "i.subscription_id IN (SELECT subscription_id FROM watchlist_sources WHERE watchlist_id = ?)"
            )
            params.append(watchlist_id)
        if unassigned_only:
            predicates.append(
                "NOT EXISTS (SELECT 1 FROM watchlist_sources ws WHERE ws.subscription_id = i.subscription_id)"
            )
        if statuses is not None:
            placeholders = ", ".join("?" for _ in statuses)
            predicates.append(f"i.status IN ({placeholders})")
            params.extend(statuses)
```

Keep the existing docstring style; document that callers wanting the unread bucket by name pass `status="new"`, and that `statuses` requires `status=None`.

Add after `get_watchlist_item_counts` (:1098):

```python
    def get_source_item_counts(self) -> Dict[int, Dict[str, int]]:
        """Per-source item totals and unread counts, for rail badges.

        One grouped query, mirroring `get_watchlist_item_counts`: adding
        sources never adds round-trips. Sources with no items are absent
        (a missing key renders as no badge, which is the honest state).
        """
        rows = self.conn.execute(
            """
            SELECT subscription_id,
                   COUNT(id) AS total,
                   SUM(CASE WHEN status = 'new' THEN 1 ELSE 0 END) AS unread
            FROM subscription_items
            GROUP BY subscription_id
            """
        ).fetchall()
        return {
            row[0]: {"total": row[1] or 0, "unread": row[2] or 0}
            for row in rows
        }
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest <testfile> -x -q`
Expected: PASS. Then `python3 -m pytest Tests/ -k subscriptions -q` — no regressions in existing DB tests.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/DB/Subscriptions_DB.py <testfile>
git commit -m "feat(watchlists): scope filters and per-source counts in Subscriptions_DB (task-NNNN)"
```

---

### Task 3: Service pass-through for scope filters and counts

**Files:**
- Modify: `tldw_chatbook/Subscriptions/local_watchlists_service.py` (`list_items` at :354-410)
- Modify: `tldw_chatbook/Subscriptions/watchlist_scope_service.py` (`list_items` at :214-257)
- Modify: `tldw_chatbook/Subscriptions/watchlist_bundle_service.py` (add delegate after `get_watchlist_item_counts` at :353)
- Test: `Tests/Watchlists/test_watchlist_scope_service.py`

- [ ] **Step 1: Write the failing tests**

In `test_watchlist_scope_service.py` (follow its existing fixture style):

```python
async def test_list_items_forwards_watchlist_scope(scope_service):
    await scope_service.list_items(runtime_backend="local", watchlist_id=3, statuses=["new", "reviewed"])
    # assert the underlying local service received watchlist_id=3 / statuses=[...]
    # (the existing tests already mock/spy the local service — mirror them)

async def test_list_items_watchlist_scope_server_backend_rejected(scope_service):
    with pytest.raises(ValueError):
        await scope_service.list_items(runtime_backend="server", watchlist_id=3)

def test_bundle_service_delegates_source_counts(bundle_service):
    assert bundle_service.get_source_item_counts() == bundle_service._db.get_source_item_counts()
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest Tests/Watchlists/test_watchlist_scope_service.py -x -q`
Expected: FAIL (unexpected kwarg / missing method).

- [ ] **Step 3: Implement**

`local_watchlists_service.py` `list_items` — add `watchlist_id=None, unassigned_only=False, statuses=None` kwargs, pass through to `get_new_items`:

```python
        rows = db.get_new_items(
            subscription_id=subscription_id,
            status=status_filter,
            limit=fetch_limit,
            run_id=int(run_id) if run_id is not None else None,
            watchlist_id=int(watchlist_id) if watchlist_id is not None else None,
            unassigned_only=bool(unassigned_only),
            statuses=list(statuses) if statuses is not None else None,
        )
```

`watchlist_scope_service.py` `list_items` — add the same three kwargs to the signature, document them in the docstring Args, and forward them in the `service.list_items(...)` call. Server-backend rejection stays as-is (it raises before any param matters). The backend controller (`watchlists_backend_controller.py:41-47`) forwards `**kwargs` already — no change needed there.

`watchlist_bundle_service.py` — add:

```python
    def get_source_item_counts(self) -> dict[int, dict[str, int]]:
        """Per-source {total, unread} for tree source badges.

        Thin delegation, same contract as `get_watchlist_item_counts` below.
        """
        return self._db.get_source_item_counts()
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest Tests/Watchlists/ -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Subscriptions/ Tests/Watchlists/test_watchlist_scope_service.py
git commit -m "feat(watchlists): pass scope filters and source counts through services (task-NNNN)"
```

---

### Task 4: Bulk mark-all-read + undo restore

**Files:**
- Modify: `tldw_chatbook/DB/Subscriptions_DB.py` (after `mark_item_status` at :1917-1950)
- Modify: `tldw_chatbook/Subscriptions/local_watchlists_service.py`, `watchlist_scope_service.py` (near their `update_item` methods)
- Test: same DB test file as Task 2 + `test_watchlist_scope_service.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_mark_all_read_returns_ids_and_only_touches_new(db_with_memberships):
    ids = db_with_memberships.mark_all_read()  # scope: all
    assert set(ids) == {ID_NEW_1, ID_NEW_2}
    assert db_with_memberships.get_item_status(ID_REVIEWED) == "reviewed"   # untouched
    assert db_with_memberships.get_item_status(ID_INGESTED) == "ingested"   # untouched

def test_mark_all_read_scoped_to_watchlist(db_with_memberships):
    ids = db_with_memberships.mark_all_read(watchlist_id=WATCHLIST_ID)
    assert set(ids) == {ID_NEW_IN_WATCHLIST}

def test_restore_items_new_only_restores_reviewed(db_with_memberships):
    db_with_memberships.mark_item_status(ID_INGESTED, "ingested")
    n = db_with_memberships.restore_items_new([ID_MARKED_READ, ID_INGESTED])
    assert n == 1
    assert db_with_memberships.get_item_status(ID_MARKED_READ) == "new"
    assert db_with_memberships.get_item_status(ID_INGESTED) == "ingested"
```

- [ ] **Step 2: Run to verify they fail** — `AttributeError: mark_all_read`.

- [ ] **Step 3: Implement**

`Subscriptions_DB.py`, after `mark_item_status`:

```python
    def mark_all_read(
        self,
        subscription_id: Optional[int] = None,
        watchlist_id: Optional[int] = None,
        unassigned_only: bool = False,
    ) -> List[int]:
        """Mark every ``new`` item in scope ``reviewed``; return the affected ids.

        One transactional UPDATE. Only ``new`` rows are touched —
        ``reviewed``/``ingested``/``ignored``/``error`` all record deliberate
        user actions and are never rewritten here (same rule
        `persist_subscription_item`'s upsert follows, `item_persist.py:132-136`).
        The returned ids are the undo batch for
        `WatchlistsCollectionsScreen.action_undo_mark_all_read`.
        """
        predicates = ["status = 'new'"]
        params: List[Any] = []
        if subscription_id is not None:
            predicates.append("subscription_id = ?")
            params.append(subscription_id)
        if watchlist_id is not None:
            predicates.append(
                "subscription_id IN (SELECT subscription_id FROM watchlist_sources WHERE watchlist_id = ?)"
            )
            params.append(watchlist_id)
        if unassigned_only:
            predicates.append(
                "NOT EXISTS (SELECT 1 FROM watchlist_sources ws WHERE ws.subscription_id = subscription_items.subscription_id)"
            )
        with self.transaction() as conn:
            rows = conn.execute(
                f"UPDATE subscription_items SET status = 'reviewed' WHERE {' AND '.join(predicates)} RETURNING id",
                tuple(params),
            ).fetchall()
        return [row[0] for row in rows]

    def restore_items_new(self, item_ids: List[int]) -> int:
        """Move the given ids back to ``new`` — but only ones still ``reviewed``.

        The undo half of `mark_all_read`. The ``status = 'reviewed'`` guard
        means an item the user has since ingested or ignored is not yanked
        back to unread.
        """
        if not item_ids:
            return 0
        placeholders = ", ".join("?" for _ in item_ids)
        with self.transaction() as conn:
            cursor = conn.execute(
                f"UPDATE subscription_items SET status = 'new' WHERE id IN ({placeholders}) AND status = 'reviewed'",
                tuple(item_ids),
            )
            return cursor.rowcount
```

Services: `LocalWatchlistsService` gains `async def mark_all_read(self, *, source_id=None, watchlist_id=None, unassigned_only=False) -> list[int]` and `async def restore_items_new(self, *, item_ids) -> int` — thin delegates (`int(source_id) if source_id is not None else None` normalization, same as `list_items`). `WatchlistScopeService` gains matching async methods that `_enforce_policy(backend, "items.update")` and reject the server backend with `ValueError` (item writes are local-only, mirroring `update_item` at :299+). The screen will call them through `WatchlistsBackendController` — check that controller for an `update_item_status` probe chain and add a `mark_all_read`/`restore_items_new` forward in the same `**kwargs` style as `list_items` (:41-47).

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest <db testfile> Tests/Watchlists/test_watchlist_scope_service.py -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/DB/Subscriptions_DB.py tldw_chatbook/Subscriptions/ tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py Tests/
git commit -m "feat(watchlists): bulk mark-all-read with undo restore (task-NNNN)"
```

---

### Task 5: Read-first IA — remove the FEEDS region, reorder tabs, land on Read

This is the structural task. The FEEDS region dies; the tab strip (which lived inside FEEDS on the Read tab) moves to the centre header on **every** tab; Read becomes tab `1` and the default section. Section **ids** do not change (`"items"` stays), so deep links (`apply_navigation_context`, :902-931) and `WATCHLISTS_NAV_CONTEXT_*` referrers keep working.

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/region_layout.py` (:18-38)
- Modify: `tldw_chatbook/UI/Watchlists_Modules/watchlists_workbench.py` (:27-57)
- Modify: `tldw_chatbook/UI/Watchlists_Modules/watchlists_tab_strip.py` (:19-27)
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (BINDINGS :385-411, reactives :413-440, `compose_content` :2359-2422, `_hidden_centre_regions` :2424-2454, `_rendered_region_layout` :2456-2508, delete `_build_list_pane` :1846-1914)
- Modify: `tldw_chatbook/css/features/_watchlists.tcss` (delete FEEDS rules)
- Test: `Tests/Watchlists/test_region_layout.py`, `test_watchlists_workbench.py`, `test_watchlists_tab_strip.py`, `test_watchlists_collections_screen.py`

- [ ] **Step 1: Update the tests to the new contract (they must fail first)**

- `test_region_layout.py`: remove every `Region.FEEDS` reference; `CENTRE_REGIONS == (Region.ITEMS, Region.CONTENT)`; solo of ITEMS collapses CONTENT and vice versa; `Region("feeds")` now raises `ValueError`.
- `test_watchlists_workbench.py`: `REGION_TITLES` has no FEEDS; a workbench built with `hidden=frozenset({Region.CONTENT})` renders LEFT_RAIL, ITEMS, RIGHT_RAIL only.
- `test_watchlists_tab_strip.py`: `SECTIONS[0] == ("items", "Read")`; clicking the first tab posts `SectionSelected("items")`.
- `test_watchlists_collections_screen.py`: default `active_section` is `"items"` on mount; binding `1` switches to items and `7` to overview; `#wl-tabs` exists on the Read tab AND on the Sources tab; `#watchlists-list-pane` does not exist anywhere (delete geometry tests pinned to `.watchlists-region-feeds` row caps — the pane is gone).

Run: `python3 -m pytest Tests/Watchlists/test_region_layout.py Tests/Watchlists/test_watchlists_workbench.py Tests/Watchlists/test_watchlists_tab_strip.py -x -q`
Expected: FAIL.

- [ ] **Step 2: `region_layout.py` — delete the region**

Remove `FEEDS = "feeds"` from `Region`, remove it from `REGION_ORDER`, and set `CENTRE_REGIONS = (Region.ITEMS, Region.CONTENT)`. Update the module docstring ("five regions" → four). No migration code: persisted `"feeds"` strings are dropped by the existing unknown-region guard in `region_layout_store.load_region_layout` (:132-137) with a debug log — record this in ADR-042 (done in Task 1).

- [ ] **Step 3: `watchlists_workbench.py` + `watchlists_tab_strip.py`**

- Delete `Region.FEEDS` from `REGION_TITLES` and `SELF_HEADED_REGIONS` (which becomes `{Region.ITEMS, Region.RIGHT_RAIL}`). Sweep the module's docstrings/comments for FEEDS references (there are several, e.g. the `SELF_HEADED_REGIONS` rationale and the `watchlists-region-sole-centre` comment mentioning the feeds cap) and rewrite them for the four-region reality — the `#watchlists-list-pane` CSS note dies with the region.
- `watchlists_tab_strip.py`: `SECTIONS = (("items", "Read"), ("sources", "Sources"), ("runs", "Runs"), ("rules", "Rules"), ("notifications", "Notifications"), ("artifacts", "Artifacts"), ("overview", "Overview"))`.

- [ ] **Step 4: The screen**

1. BINDINGS (:385-411) — remap digits, keep everything else:
   ```python
   ("1", "switch_section('items')", "Read"),
   ("2", "switch_section('sources')", "Sources"),
   ("3", "switch_section('runs')", "Runs"),
   ("4", "switch_section('rules')", "Rules"),
   ("5", "switch_section('notifications')", "Notifications"),
   ("6", "switch_section('artifacts')", "Artifacts"),
   ("7", "switch_section('overview')", "Overview"),
   ```
2. `active_section = reactive("items")` (:413) and `focused_region = reactive(Region.ITEMS)` (:440).
3. `_SECTION_DETAIL_TITLE["items"] = "Read"` (:474-481 area).
4. Delete `_build_list_pane` (:1846-1914) and its `Region.FEEDS: self._build_list_pane` entry in `compose_content` (:2405).
5. `compose_content`: the header is now unconditional — `header=self._build_centre_status_header` (delete the `None if self.active_section == "items" else ...` conditional at :2416-2420). The tab strip reaches Read through the header now.
6. `_hidden_centre_regions` (:2424-2454): off Read, return `frozenset({Region.CONTENT})`; on Read, `frozenset()`. Rewrite the docstring (it argues about FEEDS at length).
7. `_rendered_region_layout` (:2456-2508): logic unchanged (only the ITEMS adjustment remains); rewrite the docstring's FEEDS references.
8. Sweep the screen for remaining `Region.FEEDS` / `_build_list_pane` references (`grep -n "FEEDS\|_build_list_pane\|watchlists-list-pane" tldw_chatbook/UI/Screens/watchlists_collections_screen.py`) — `watch_tree_scope` (:3385-3440) refreshes the FEEDS region on scope moves; replace that call with the items reload Task 7 adds (leave a `TODO`-free clean path: for now, guard with `if self.active_section == "items": self.run_worker(self._load_items(), exclusive=True)` — Task 7 fleshes this out properly, doing both edits in one pass is fine if you keep the commits separate).
9. `tldw_chatbook/css/features/_watchlists.tcss`: delete the `.watchlists-region-feeds` rule(s), the `#watchlists-list-pane` rule, and any `.watchlist-feed-source-row` rule. `grep -n "feeds\|list-pane\|feed-source" tldw_chatbook/css/features/_watchlists.tcss` must come back empty afterwards.

- [ ] **Step 5: Run the updated tests**

Run: `python3 -m pytest Tests/Watchlists/ -x -q`
Expected: PASS. Also `python3 -m pytest Tests/UI/test_destination_shells.py Tests/UI/test_destination_visual_parity_correction.py -q` — these drive the `#wl-tabs` selectors and must stay green.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/ Tests/Watchlists/
git commit -m "feat(watchlists): remove FEEDS region, Read-first tab order and landing (task-NNNN)"
```

---

### Task 6: First-run default — inspector starts collapsed

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/region_layout_store.py` (:32-42)
- Test: `Tests/Watchlists/test_region_layout_store.py`

- [ ] **Step 1: Failing test**

```python
def test_first_run_default_collapses_right_rail(monkeypatch_config):
    # no persisted value at all
    assert load_region_layout().collapsed == frozenset({Region.RIGHT_RAIL})

def test_explicitly_empty_persisted_layout_stays_expanded(monkeypatch_config):
    # user saved [] deliberately — the None-vs-[] distinction the loader documents
    save_region_layout(RegionLayout())
    assert load_region_layout().collapsed == frozenset()
```

- [ ] **Step 2: Run to verify the first fails** — currently `RegionLayout()` (nothing collapsed).

- [ ] **Step 3: Implement**

`_FIRST_RUN_DEFAULT = RegionLayout(collapsed=frozenset({Region.RIGHT_RAIL}))` and rewrite its comment block (:32-42): the CONTENT-stub history stays (it's why the migration key exists), plus one paragraph — a new user's Read tab is a reader, not an inspector; the RIGHT_RAIL's management actions recede until `]` opens them. The loader's None-vs-`[]` machinery already makes "user deliberately expanded everything" survive, so no other change.

- [ ] **Step 4: Run** `python3 -m pytest Tests/Watchlists/test_region_layout_store.py -x -q` — PASS.

- [ ] **Step 5: Commit** `git commit -m "feat(watchlists): inspector starts collapsed for new users (task-NNNN)"`.

---

### Task 7: Scope-plumbed `_load_items` + reload on tree moves

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (`_load_items` :8109-8133, add `_items_scope_query` near `_items_status_query` :8029, `watch_tree_scope` :3385-3440)
- Test: `Tests/Watchlists/test_watchlists_collections_screen.py`

- [ ] **Step 1: Failing tests**

```python
async def test_items_reload_scopes_to_watchlist(screen):
    screen._apply_tree_scope(TreeScope(kind="watchlist", watchlist_id=1))
    await screen._load_items()
    assert all(item["source_id"] in WATCHLIST_1_SOURCE_IDS for item in screen._loaded_items)

async def test_items_reload_scopes_to_unassigned(screen): ...
async def test_items_reload_scopes_to_source(screen): ...
async def test_tree_move_triggers_items_reload_on_read_tab(screen):
    # spy on _load_items; _apply_tree_scope(...) with active_section == "items"
    ...
```

Follow the file's existing pilot/fixture patterns for building the screen with a seeded DB.

- [ ] **Step 2: Run to verify they fail** — `_load_items` ignores scope today.

- [ ] **Step 3: Implement**

Add beside `_items_status_query` (:8029):

```python
    def _items_scope_query(self) -> dict[str, Any]:
        """The tree scope as `list_items` kwargs.

        `all` passes nothing (every source). A `source` scope collapses to its
        single `source_id`; watchlist membership (many-to-many) is resolved by
        the query, not here. This is the wiring the whole phase exists for:
        before it, `_load_items` fetched the newest 100 items of ANY source
        regardless of the rail selection.
        """
        scope = self.tree_scope
        if scope.kind == "source" and scope.source_id is not None:
            return {"source_id": scope.source_id}
        if scope.kind == "watchlist" and scope.watchlist_id is not None:
            return {"watchlist_id": scope.watchlist_id}
        if scope.kind == "unassigned":
            return {"unassigned_only": True}
        return {}
```

`_load_items` (:8112-8117) — pass it through:

```python
            items = await self._controller.list_items(
                runtime_backend=self.runtime_backend,
                status=self._items_status_query(),
                limit=100,
                offset=0,
                **self._items_scope_query(),
            )
```

`watch_tree_scope` (:3385-3440): wherever it refreshed the FEEDS region for `active_section == "items"` (Task 5 removed the region), dispatch the reload instead — `self.run_worker(self._load_items(), exclusive=True)` — guarded to the Read tab, and keep the header refresh (`refresh_header_content`) since the header now carries the scope markers on every tab.

- [ ] **Step 4: Run** `python3 -m pytest Tests/Watchlists/test_watchlists_collections_screen.py -x -q` — PASS.

- [ ] **Step 5: Commit** `git commit -m "feat(watchlists): tree scope drives the items list (task-NNNN)"`.

---

### Task 8: Per-source unread badges in the tree

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/watchlist_tree.py` (`__init__` :145-173, `_watchlist_node` :424-447)
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (`__init__` near :523, `_load_tree_data` :1041-1063, `_build_tree_pane` :1498-1524)
- Test: `Tests/Watchlists/test_watchlist_tree.py`

- [ ] **Step 1: Failing test**

```python
def test_source_node_shows_unread_badge_when_positive():
    tree = WatchlistTree(
        watchlists=[{"id": 1, "name": "Brief", "tags": []}],
        counts={1: {"total": 5, "unread": 3}},
        source_counts={10: {"total": 5, "unread": 3}, 11: {"total": 2, "unread": 0}},
        source_rows_loader=lambda wid: [{"id": 10, "name": "Feed A", "type": "rss"},
                                        {"id": 11, "name": "Feed B", "type": "rss"}],
        expanded=frozenset({1}),
    )
    # source 10's button label ends with the unread count; source 11's shows no number
```

Follow the file's existing construction/pilot pattern. Badge rule: sources show the unread number **only when > 0** (roots and watchlists keep their current always-show behavior); the tooltip always uses `_unread_phrase`.

- [ ] **Step 2: Run to verify it fails** — `TypeError: unexpected keyword 'source_counts'`.

- [ ] **Step 3: Implement**

`watchlist_tree.py`:
- `__init__`: new optional param `source_counts: Mapping[int, Mapping[str, int]] | None = None`; `self._source_counts = dict(source_counts or {})`.
- `_watchlist_node`'s source loop (:431-447):
  ```python
  source_id = int(row["id"])
  source_name = escape_markup(str(row["name"]))
  unread = self._source_counts.get(source_id, {}).get("unread", 0)
  badge = f"  {unread}" if unread > 0 else ""
  source = Button(
      f"    {source_name}{badge}",
      id=f"wl-tree-node-source-{watchlist_id}-{row['id']}",
      compact=True,
      tooltip=f"Show items from {source_name}. {self._unread_phrase(unread)}.",
  )
  ```

Screen:
- `__init__` (near `self._loaded_items` :523): `self._tree_source_counts: dict[int, dict[str, int]] = {}`.
- `_load_tree_data` (:1060): right after `self._tree_counts = service.get_watchlist_item_counts()`, add `self._tree_source_counts = service.get_source_item_counts()` (and `{}` in the failure branch at :1063). Both go through the bundle service delegate from Task 3.
- `_build_tree_pane` (:1515-1524): pass `source_counts=self._tree_source_counts`. The tree's existing rebuild-on-counts-refresh path (the surface-refresh drain `_load_tree_data` publishes through) picks the badges up unchanged.

- [ ] **Step 4: Run** `python3 -m pytest Tests/Watchlists/test_watchlist_tree.py Tests/Watchlists/test_watchlists_collections_screen.py -x -q` — PASS.

- [ ] **Step 5: Commit** `git commit -m "feat(watchlists): per-source unread badges in the rail tree (task-NNNN)"`.

---

### Task 9: Collapsed-rail header shows total unread

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/watchlists_workbench.py` (`__init__` :91-153, `_region_widget` :179-206; add `set_collapsed_suffixes`)
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (`compose_content` :2359-2422, `_load_tree_data` :1041-1063)
- Test: `Tests/Watchlists/test_watchlists_workbench.py`

- [ ] **Step 1: Failing tests**

```python
def test_collapsed_header_shows_suffix():
    wb = WatchlistsWorkbench(RegionLayout(collapsed=frozenset({Region.LEFT_RAIL})),
                             collapsed_suffixes={Region.LEFT_RAIL: "12 unread"})
    # the header button's label is "▸ Watchlists  12 unread"

def test_set_collapsed_suffixes_repaints_mounted_collapsed_header(): ...
```

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement**

Workbench:
- `__init__` param `collapsed_suffixes: Mapping[Region, str] | None = None` → `self._collapsed_suffixes = dict(collapsed_suffixes or {})`. Constructor-only like `hidden` — but mutable through the setter below, since counts change while the rail stays collapsed.
- `_region_widget`'s collapsed branch (:199-206): `suffix = self._collapsed_suffixes.get(region, "")`; label `f"▸ {REGION_TITLES[region]}" + (f"  {suffix}" if suffix else "")`.
- New method:
  ```python
  def set_collapsed_suffixes(self, suffixes: Mapping[Region, str]) -> None:
      """Update collapsed-header suffixes in place (no recompose).

      Counts refresh while the rail stays collapsed; tearing the workbench
      down for a number is exactly what `refresh_region_content` exists to
      avoid for bodies. A no-op for regions not currently collapsed.
      """
      self._collapsed_suffixes = dict(suffixes)
      for region, suffix in self._collapsed_suffixes.items():
          if not self.region_layout.is_collapsed(region):
              continue
          try:
              header = self.query_one(f"#wl-header-{region.value}", Button)
          except NoMatches:
              continue
          header.label = f"▸ {REGION_TITLES[region]}" + (f"  {suffix}" if suffix else "")
  ```

Screen:
- `compose_content`: `collapsed_suffixes={Region.LEFT_RAIL: self._rail_unread_suffix()}` on the workbench.
- New helper: `def _rail_unread_suffix(self) -> str: n = self._tree_counts.get(ALL_SOURCES_BUCKET, {}).get("unread", 0); return f"{n} unread" if n else ""` (import `ALL_SOURCES_BUCKET` from `watchlist_tree` — the screen already imports the tree).
- `_load_tree_data`: after the counts land, `self.query_one(WatchlistsWorkbench).set_collapsed_suffixes({Region.LEFT_RAIL: self._rail_unread_suffix()})` inside the same `try`/`NoMatches` guard style used elsewhere.

- [ ] **Step 4: Run** `python3 -m pytest Tests/Watchlists/test_watchlists_workbench.py -x -q` — PASS.

- [ ] **Step 5: Commit** `git commit -m "feat(watchlists): collapsed rail header shows total unread (task-NNNN)"`.

---

### Task 10: Reader verbs — `m`, `space`, `a`, `u`

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (BINDINGS :385-411; new actions beside `_navigate_item` :9420-9526)
- Modify: `tldw_chatbook/UI/Watchlists_Modules/items_pane.py` (add BINDINGS + a `NextUnreadRequested` message)
- Test: `Tests/Watchlists/test_watchlists_collections_screen.py`, `Tests/Watchlists/test_watchlists_items_pane.py`

Key-dispatch facts, verified against the installed Textual 8.2.7 (do not re-litigate, do write the regression tests): the rail is `WatchlistTree(Vertical)` composed of `Button`s — there is NO Textual `Tree` on this screen, and `Button.BINDINGS` is enter-only, so a screen-level `space` binding would fire while the rail has focus and hijack it. `space` is therefore bound on **ItemsPane** (what the spec's keybinding table calls for): `DataTable` has no `space` binding, so space with the items table focused bubbles up to the pane; `Input` consumes printable keys, so typing spaces in the search box never reaches it; rail widgets are not ItemsPane descendants, so the rail is unreachable by construction. In Phase 1, space with the reader (ContentPane) focused does nothing — that affordance lands with Phase 2's article-list/reader work.

- [ ] **Step 1: Failing tests**

```python
async def test_m_toggles_read_state_on_open_item(screen): ...        # new->reviewed->new
async def test_m_refuses_on_ingested_item(screen): ...               # notify, no write
async def test_space_opens_next_unread(screen): ...                  # skips reviewed rows
async def test_space_at_end_notifies_all_caught_up(screen): ...
async def test_space_with_rail_focused_does_not_navigate(screen): ...  # regression: rail unaffected
# ^ assert "no selection change / handler not called" (spy on select_and_reveal),
#   not "binding didn't fire" — the pane binding is simply unreachable from the rail.
async def test_space_in_items_search_input_still_types(screen): ...
async def test_mark_all_read_then_undo_roundtrip(screen): ...        # a then u restores statuses
async def test_mark_all_read_scoped_to_watchlist(screen): ...
async def test_verbs_noop_off_read_tab(screen): ...
```

- [ ] **Step 2: Run to verify they fail** — actions don't exist.

- [ ] **Step 3: Implement**

Screen BINDINGS additions (after the `j`/`k` lines) — note `space` is deliberately NOT here, see the dispatch note above:

```python
        ("m", "toggle_read_selected", "Read/Unread"),
        ("a", "mark_all_read", "Mark all read"),
        ("u", "undo_mark_all_read", "Undo mark-all-read"),
```

`items_pane.py` — the `space` binding and its message (binding the key on the pane means it only exists while focus is inside the items region):

```python
class NextUnreadRequested(Message):
    """Posted when the user asks for the next unread item (`space`)."""


class ItemsPane(RecomposeCaptureGuard, Vertical):
    BINDINGS = [("space", "next_unread", "Next unread")]
    ...
    def action_next_unread(self) -> None:
        self.post_message(NextUnreadRequested())
```

`__init__`: `self._last_mark_all_read_batch: list[int] = []` (near `_loaded_items`, :523).

New actions (share `_navigate_item`'s guards — Input/editable-TextArea focus, `active_section != "items"` no-op):

```python
    def action_toggle_read_selected(self) -> None:
        """`m`: flip the open item between new and reviewed."""
        item = self._selected_content_item
        if self.active_section != "items" or item is None:
            return
        item_id = item.get("id")
        if item_id is None:
            return
        current = str(item.get("status") or "").strip().lower()
        if current == "new":
            target = "reviewed"
        elif current == "reviewed":
            target = "new"
        else:
            self.app_instance.notify(
                "Only read/unread items can be toggled.", severity="warning"
            )
            return
        self._dispatch_item_status(item_id, _ItemStatusIntent(status=target, gate=True))
        self._request_tree_counts_refresh()

    @on(NextUnreadRequested)  # import it beside `ItemSelected` from `items_pane`
    def handle_next_unread_requested(self, event: NextUnreadRequested) -> None:
        """`space` (ItemsPane binding): open the next unread item after the current one.

        No Input/Tree focus guards needed: `Input` consumes printable keys
        before the pane binding can fire, and rail widgets are not ItemsPane
        descendants, so this message can only originate from the items region.
        """
        event.stop()
        if self.active_section != "items":
            return
        try:
            pane = self.query_one("#watchlists-items-pane", ItemsPane)
        except NoMatches:
            return
        items = pane.displayed_items()
        if not items:
            return
        current = self._selected_content_item
        current_id = current.get("id") if current else None
        start = -1
        if current_id is not None:
            for position, candidate in enumerate(items):
                if candidate.get("id") == current_id:
                    start = position
                    break
        for candidate in items[start + 1:]:
            if str(candidate.get("status") or "").lower() == "new":
                pane.select_and_reveal(candidate)
                return
        self.app_instance.notify("All caught up.", severity="information")
```

Mark-all-read runs in a worker (DB write + potentially slow badge refresh), mirroring how every other write handler here defers to `run_worker`:

```python
    def action_mark_all_read(self) -> None:
        """`a`: catch the current scope up. Undoable with `u`."""
        if self.active_section != "items":
            return
        self.run_worker(self._mark_all_read_worker(), exclusive=True, group="wl-mark-all-read")

    async def _mark_all_read_worker(self) -> None:
        ids = await self._controller.mark_all_read(
            runtime_backend=self.runtime_backend, **self._items_scope_query()
        )
        if not ids:
            self.app_instance.notify("Nothing unread in this scope.")
            return
        self._last_mark_all_read_batch = list(ids)
        id_set = {int(i) for i in ids}
        for item in self._loaded_items:          # in-place patch, same contract as
            if int(item.get("id") or -1) in id_set:   # _mark_item_read_on_open's patch_item
                item["status"] = "reviewed"
        self._repaint_visible_status_cells()
        self._request_tree_counts_refresh()
        self.app_instance.notify(f"Marked {len(ids)} read — press u to undo.")

    def action_undo_mark_all_read(self) -> None:
        """`u`: restore the most recent mark-all-read batch."""
        if self.active_section != "items":
            return
        if not self._last_mark_all_read_batch:
            self.app_instance.notify("Nothing to undo.")
            return
        self.run_worker(self._undo_mark_all_read_worker(), exclusive=True, group="wl-mark-all-read")

    async def _undo_mark_all_read_worker(self) -> None:
        batch, self._last_mark_all_read_batch = self._last_mark_all_read_batch, []
        restored = await self._controller.restore_items_new(
            runtime_backend=self.runtime_backend, item_ids=batch
        )
        id_set = {int(i) for i in batch}
        for item in self._loaded_items:
            if int(item.get("id") or -1) in id_set and item.get("status") == "reviewed":
                item["status"] = "new"
        self._repaint_visible_status_cells()
        self._request_tree_counts_refresh()
        self.app_instance.notify(f"Restored {restored} to unread.")
```

`_repaint_visible_status_cells` — loop `self.query_one("#watchlists-items-pane", ItemsPane).displayed_items()`, calling the pane's existing `update_item_status_cell(item["id"], item["status"])` for each (it no-ops on unrendered rows already, `items_pane.py:238-271`).

- [ ] **Step 4: Run** `python3 -m pytest Tests/Watchlists/test_watchlists_collections_screen.py -x -q` — PASS.

- [ ] **Step 5: Commit** `git commit -m "feat(watchlists): reader verbs m/space/a/u (task-NNNN)"`.

---

### Task 11: Full suite + manual QA

- [ ] **Step 1: Full Watchlists + UI suites**

```bash
python3 -m pytest Tests/Watchlists/ Tests/UI/test_destination_shells.py Tests/UI/test_destination_visual_parity_correction.py -q
```
Expected: all green.

- [ ] **Step 2: Wider regression sweep**

```bash
python3 -m pytest Tests/ -q -k "subscription or watchlist"
```
Expected: green.

- [ ] **Step 3: Manual smoke (describe results in the task notes)**

`python3 -m tldw_chatbook.app` → Watchlists: lands on Read; rail badges on sources; tree click scopes items; `j`/`k`/`space` walk; `m` toggles; `a` catches up, `u` restores; `[`/`]` rails, collapsed left rail shows "▸ Watchlists  N unread"; tabs 1-7 in new order with Read first.

- [ ] **Step 4: Update docs + close out**

- `Docs/superpowers/specs/2026-08-05-watchlists-reader-first-design.md` — Status → "Phase 1 implemented"; note anything that deviated.
- Backlog: check off AC boxes that hold, add Implementation Notes (approach, files, trade-offs), `backlog task edit <id> -s Done` only if every Phase-1 AC passes; otherwise leave In Progress with notes.

- [ ] **Step 5: Final commit**

```bash
git commit -am "docs(watchlists): phase 1 closeout notes (task-NNNN)"
```

---

## Phase 1 done-when (from the spec)

- Picking any rail node scopes the items list.
- Per-feed unread badges render in the tree; collapsed rail shows total unread.
- Catch-up is two keys (`a`, maybe `u`); read state survives region toggles and tab switches.
- Read is the landing tab; tabs are Read, Sources, Runs, Rules, Notifications, Artifacts, Overview on `1`–`7`.
- `Tests/Watchlists/` green after updates.

## Explicitly NOT in Phase 1 (later phases, per spec §Phasing)

- Reader-style article list widget (snippets, date groups, bold-unread rows) — Phase 2.
- Star/flag wiring, `content_render.py` HTML→text, open-in-browser, action row, position footer — Phase 2.
- Smart-feed nodes (All Unread/Today/Starred), `/` FTS search, refresh-all — Phase 3.
- OPML folder mapping, polish tasks 2308/2310/2312/2313 — Phase 4.
- Per-source error markers in the tree — Phase 2 (needs a latest-run join; kept out to hold Phase 1 to the reading loop).
