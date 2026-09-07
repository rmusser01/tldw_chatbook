# Watchlists Rebuild — Phase C: Tree & Tabs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Watchlists left rail into the watchlist tree you navigate by, move section switching to tabs above the centre stack, and retire the section navigator.

**Architecture:** The tree is the primary navigation surface: two permanent roots (All sources, Unassigned) above a flat list of watchlists, each expandable to its sources. Selecting a node sets a screen-level *scope* that the Feeds and Items regions read. Section switching becomes a one-row tab strip in the centre. Panes keep working throughout — this changes navigation, not pane internals.

**Tech Stack:** Python ≥3.11, Textual ≥3.3.0, pytest. No new dependencies.

## Global Constraints

- Spec: `Docs/superpowers/specs/2026-07-25-watchlists-console-rebuild-design.md`. Read its "Starting point — corrected 2026-07-26" section before anything else.
- **No feature may be lost.** The screen has six working panes and a large existing test suite. This phase changes navigation; panes keep functioning.
- **No data-layer changes** beyond the one additive service method in Task 1. Nothing under `tldw_chatbook/DB`.
- Guard files — **no existing test body may be edited**, additions only: `Tests/UI/test_watchlists_destination_shell.py`, `Tests/UI/test_destination_shells.py`, `Tests/UI/test_console_live_work_handoffs.py`.
- The screen class name and route (`watchlists_collections`, stored as `screen_name`) are unchanged.
- **Counts come from one query.** `SubscriptionsDB.get_watchlist_item_counts()` returns every bucket in a single statement, keyed by watchlist id plus `UNASSIGNED_BUCKET = -1` and `ALL_SOURCES_BUCKET = -2`. Never count per node.
- The workbench reactive is `region_layout`, not `layout` — `Widget.layout` is a read-only Textual property.
- Workbench `content` holds **factories**, not instances: `region_layout` is `recompose=True`, so every toggle rebuilds every region. Any pane-local state that matters must be mirrored to screen state and re-seeded.
- `get_cli_setting` is a **flat** lookup; dotted sections silently return the default.
- Tests run from the venv. **Foreground, one file at a time** — anything past ~90 seconds is auto-backgrounded here and appears to hang.
- `Tests/Watchlists/test_watchlists_navigator.py::test_navigator_has_all_section_buttons` fails on dev already. Task 3 deletes that file legitimately; until then, leave it.

## What the screen looks like today

Captured live at 235×52 on merged dev, after Phase B:

```
╭─Watchlists──╮╭─Feeds───────────────────────╮╭─Inspector──╮
│    Overview ││┌───────────────────────────┐│││ Inspector ││   <- region title AND
│    Sources  │││  Sources                  ││││ State:... ││      pane title
│     Items   ││└───────────────────────────┘│││ Stage Wat ││   <- truncated at 26 cols
│      Runs   │╭─Items──────────────────────╮│││ Open curr ││
│     Rules   ││┌───────────────────────────┐││││          ││
│ Notification│││  Overview                 ││││          ││
```

Four defects this phase fixes, all measured: region titles mislabel their contents (the region called "Feeds" holds a pane titled "Sources"); every region draws a border and the pane inside draws another; the 26-column Inspector truncates every action label; and the left rail spends its width on six words.

## Target

```
╭─Watchlists──────╮╭─[Read] Sources Runs Rules Artifacts──╮╭─Inspector────────╮
│ ▸ All sources 52││ Feeds in Morning AI Brief (3)        ││ Morning AI Brief │
│ ▸ Unassigned   4││  ArXiv: AI   rss  2m   24  Healthy   ││  ▸ ArXiv: AI     │
│ ▾ Morning     24│├──────────────────────────────────────┤│    ▸ RAG Eval    │
│     ArXiv: AI   ││ Items · ArXiv: AI (24)               ││ ──────────────── │
│     HN Top      ││  1 RAG Evaluation   04-25  New       ││ [Open       o]   │
│ ▸ Security     8│├──────────────────────────────────────┤│ [Stage      s]   │
│ #daily #ai #sec ││ Content                              ││ [Ingest     i]   │
```

## File Structure

| Path | Responsibility | Task |
|---|---|---|
| `tldw_chatbook/Subscriptions/watchlist_bundle_service.py` | + `list_source_rows` — one JOIN, no N+1 | 1 |
| `tldw_chatbook/app.py` | Wire `WatchlistBundleService` — it has no production caller today | 1 |
| `tldw_chatbook/UI/Watchlists_Modules/watchlist_tree.py` | **New.** Tree: roots, watchlists, lazy sources, tag filters | 2 |
| `tldw_chatbook/UI/Watchlists_Modules/watchlists_tab_strip.py` | **New.** One-row section tab strip | 3 |
| `tldw_chatbook/UI/Watchlists_Modules/watchlists_navigator.py` | **Deleted** — replaced by the tree + tab strip | 3 |
| `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` | Service accessor, scope reactive, region mapping | 1, 3, 4, 7 |
| `tldw_chatbook/UI/Watchlists_Modules/inspector_pane.py` | Breadcrumb stack | 5 |
| `tldw_chatbook/css/features/_watchlists.tcss` | Rail widths, single-border chrome | 5, 6 |
| `Tests/Subscriptions/test_watchlist_bundle_service.py` | + membership rows | 1 |
| `Tests/Watchlists/test_watchlist_tree.py` | **New.** Tree behaviour | 2 |
| `Tests/Watchlists/test_watchlists_tab_strip.py` | **New.** Tab behaviour | 3 |
| `Tests/Watchlists/test_watchlists_navigator.py` | **Deleted** with the navigator | 3 |

---

### Task 1: Wire `WatchlistBundleService`, and give it `list_source_rows`

**Files:**
- Modify: `tldw_chatbook/Subscriptions/watchlist_bundle_service.py`
- Modify: `tldw_chatbook/app.py` — wire the service (it currently has no production caller)
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` — add the accessor
- Test: `Tests/Subscriptions/test_watchlist_bundle_service.py`

**Interfaces:**
- Consumes: the existing `watchlists` / `watchlist_sources` tables, and the `subscriptions_db` already constructed in `app.py`.
- Produces: `app.watchlist_bundle_service` (a live `WatchlistBundleService`), a screen-side accessor for it, and `WatchlistBundleService.list_source_rows(watchlist_id: int) -> list[dict[str, Any]]` with each row `{"id": int, "name": str, "type": str}`. Task 2's tree calls it on expand; Task 4 reads the service through the accessor.

**`WatchlistBundleService` is not wired into the app.** Verified: its only non-test reference in
`tldw_chatbook/` is a comment. Phase A built it and never instantiated it, exactly as it built
`backfill_items_fts` without a caller. So before the tree can read anything, the service must exist
on the app the way its sibling does — `self.watchlist_scope_service = WatchlistScopeService(...)`
at `app.py:5468`, reached from the screen via `getattr(app_instance, "watchlist_scope_service", None)`.

Wire `self.watchlist_bundle_service = WatchlistBundleService(subscriptions_db)` alongside it, using
the `subscriptions_db` already constructed there, and add a screen-side accessor following the same
`getattr(..., None)` pattern so the screen degrades rather than crashing when the service is absent.

**Why `list_source_rows` exists.** `list_sources` returns bare subscription **ids**. A tree needs names and types, and calling `get_subscription(id)` per id would be N+1 inside a render — exactly the trap `get_watchlist_item_counts` was built in one query to avoid. One JOIN instead.

- [ ] **Step 1: Write the failing test**

Append to `Tests/Subscriptions/test_watchlist_bundle_service.py`:

```python
def test_list_source_rows_returns_names_and_types(service, db):
    watchlist = service.create("Morning")
    a = db.add_subscription(name="ArXiv: AI", type="rss", source="https://a.example/f")
    b = db.add_subscription(name="anthropic.com", type="url", source="https://b.example/")
    service.add_source(watchlist["id"], a)
    service.add_source(watchlist["id"], b)

    rows = service.list_source_rows(watchlist["id"])
    assert [r["name"] for r in rows] == ["ArXiv: AI", "anthropic.com"]
    assert {r["type"] for r in rows} == {"rss", "url"}
    assert {r["id"] for r in rows} == {a, b}


def test_list_source_rows_is_empty_for_a_watchlist_with_no_sources(service):
    watchlist = service.create("Empty")
    assert service.list_source_rows(watchlist["id"]) == []


def test_list_source_rows_uses_a_single_query(service, db, monkeypatch):
    watchlist = service.create("Morning")
    for index in range(6):
        service.add_source(
            watchlist["id"],
            db.add_subscription(name=f"S{index}", type="rss", source=f"https://s{index}.example/f"),
        )

    class _Counting:
        def __init__(self, inner):
            self._inner = inner
            self.execute_count = 0

        def execute(self, *args, **kwargs):
            self.execute_count += 1
            return self._inner.execute(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    counting = _Counting(db.conn)
    monkeypatch.setattr(type(db), "conn", property(lambda self: counting))
    service.list_source_rows(watchlist["id"])
    assert counting.execute_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Subscriptions/test_watchlist_bundle_service.py -k list_source_rows -v`
Expected: FAIL — `AttributeError: 'WatchlistBundleService' object has no attribute 'list_source_rows'`.

- [ ] **Step 3: Write the implementation**

Add to `WatchlistBundleService`, after `list_sources`:

```python
    def list_source_rows(self, watchlist_id: int) -> list[dict[str, Any]]:
        """Sources in a watchlist, with the fields a tree row needs.

        ``list_sources`` returns bare ids; resolving each to a name would be
        one query per source inside a render. This joins instead, so expanding
        a watchlist costs exactly one query no matter how many sources it has.

        Args:
            watchlist_id: The watchlist whose sources to list.

        Returns:
            One dict per source with ``id``, ``name`` and ``type``, in the
            order the sources were added.
        """
        rows = self._db.conn.execute(
            """
            SELECT s.id, s.name, s.type
            FROM watchlist_sources ws
            JOIN subscriptions s ON s.id = ws.subscription_id
            WHERE ws.watchlist_id = ?
            ORDER BY ws.added_at, s.id
            """,
            (watchlist_id,),
        ).fetchall()
        return [{"id": row[0], "name": row[1], "type": row[2]} for row in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest Tests/Subscriptions/test_watchlist_bundle_service.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Subscriptions/watchlist_bundle_service.py Tests/Subscriptions/test_watchlist_bundle_service.py
git commit -m "feat(watchlists): add list_source_rows for tree rendering"
```

---

### Task 2: The watchlist tree

**Files:**
- Create: `tldw_chatbook/UI/Watchlists_Modules/watchlist_tree.py`
- Test: `Tests/Watchlists/test_watchlist_tree.py`

**Interfaces:**
- Consumes: Task 1's `list_source_rows`; `WatchlistBundleService.list_watchlists`; `SubscriptionsDB.get_watchlist_item_counts` with its `UNASSIGNED_BUCKET = -1` / `ALL_SOURCES_BUCKET = -2` sentinels.
- Produces: `WatchlistTree(Vertical)` and the message `TreeScopeChanged(scope)`, where `scope` is a `TreeScope` dataclass with `kind: Literal["all", "unassigned", "watchlist", "source"]`, `watchlist_id: int | None`, `source_id: int | None`. Tasks 4 and 5 consume both.

**Design rules that are load-bearing:**

- **Two permanent roots.** `All sources` and `Unassigned` always render, above the watchlists. Without them, deleting a watchlist orphans its sources into invisibility, and a first-run install (which has no watchlists — the folder migration is effectively a no-op) shows an empty rail.
- **Counts come from the single grouped query**, keyed by watchlist id plus the two sentinels. Never count per node.
- **Sources load lazily, on expand.** A watchlist node shows its count when collapsed and fetches its rows via `list_source_rows` only when the user expands it. That keeps the initial render to two queries regardless of how many watchlists exist.
- **Tag filters** render below the tree and narrow which watchlists are shown. Tags are comma-joined on `watchlists.tags`.

- [ ] **Step 1: Write the failing test**

Create `Tests/Watchlists/test_watchlist_tree.py`:

```python
import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import (
    TreeScope,
    TreeScopeChanged,
    WatchlistTree,
)


def _tree_data():
    return {
        "watchlists": [
            {"id": 1, "name": "Morning AI Brief", "tags": ["ai", "daily"]},
            {"id": 2, "name": "Security", "tags": ["sec"]},
        ],
        "counts": {
            1: {"total": 24, "unread": 24},
            2: {"total": 8, "unread": 3},
            -1: {"total": 4, "unread": 1},
            -2: {"total": 52, "unread": 37},
        },
    }


class _TreeApp(App):
    def __init__(self, data, source_rows=None):
        super().__init__()
        self._data = data
        self._source_rows = source_rows or {}
        self.scopes: list[TreeScope] = []

    def compose(self) -> ComposeResult:
        yield WatchlistTree(
            watchlists=self._data["watchlists"],
            counts=self._data["counts"],
            source_rows_loader=lambda wid: self._source_rows.get(wid, []),
            id="wl-tree",
        )

    def on_tree_scope_changed(self, message: TreeScopeChanged) -> None:
        self.scopes.append(message.scope)


@pytest.mark.asyncio
async def test_permanent_roots_always_render():
    app = _TreeApp(_tree_data())
    async with app.run_test():
        assert app.query("#wl-tree-node-all")
        assert app.query("#wl-tree-node-unassigned")


@pytest.mark.asyncio
async def test_roots_render_even_with_no_watchlists():
    # First run: the folder migration is effectively a no-op, so there are none.
    app = _TreeApp({"watchlists": [], "counts": {-1: {"total": 0, "unread": 0},
                                                 -2: {"total": 0, "unread": 0}}})
    async with app.run_test():
        assert app.query("#wl-tree-node-all")
        assert app.query("#wl-tree-node-unassigned")


@pytest.mark.asyncio
async def test_watchlists_render_with_their_unread_counts():
    app = _TreeApp(_tree_data())
    async with app.run_test():
        text = " ".join(str(n.renderable) for n in app.query("Static"))
        assert "Morning AI Brief" in text
        assert "24" in text
        assert "Security" in text


@pytest.mark.asyncio
async def test_selecting_a_watchlist_posts_its_scope():
    app = _TreeApp(_tree_data())
    async with app.run_test() as pilot:
        await pilot.click("#wl-tree-node-watchlist-1")
        await pilot.pause()
        assert app.scopes[-1] == TreeScope(kind="watchlist", watchlist_id=1, source_id=None)


@pytest.mark.asyncio
async def test_selecting_all_sources_posts_the_all_scope():
    app = _TreeApp(_tree_data())
    async with app.run_test() as pilot:
        await pilot.click("#wl-tree-node-all")
        await pilot.pause()
        assert app.scopes[-1].kind == "all"


@pytest.mark.asyncio
async def test_sources_load_only_when_a_watchlist_is_expanded():
    calls: list[int] = []

    class _App(_TreeApp):
        def compose(self) -> ComposeResult:
            def loader(wid):
                calls.append(wid)
                return [{"id": 10, "name": "ArXiv: AI", "type": "rss"}]

            yield WatchlistTree(
                watchlists=self._data["watchlists"],
                counts=self._data["counts"],
                source_rows_loader=loader,
                id="wl-tree",
            )

    app = _App(_tree_data())
    async with app.run_test() as pilot:
        assert calls == [], "no watchlist is expanded yet, so nothing should have loaded"
        await pilot.click("#wl-tree-expand-1")
        await pilot.pause()
        assert calls == [1]
        assert app.query("#wl-tree-node-source-10")


@pytest.mark.asyncio
async def test_selecting_a_source_posts_a_source_scope():
    app = _TreeApp(_tree_data(), source_rows={1: [{"id": 10, "name": "ArXiv: AI", "type": "rss"}]})
    async with app.run_test() as pilot:
        await pilot.click("#wl-tree-expand-1")
        await pilot.pause()
        await pilot.click("#wl-tree-node-source-10")
        await pilot.pause()
        assert app.scopes[-1] == TreeScope(kind="source", watchlist_id=1, source_id=10)


@pytest.mark.asyncio
async def test_tag_filter_narrows_which_watchlists_render():
    app = _TreeApp(_tree_data())
    async with app.run_test() as pilot:
        await pilot.click("#wl-tree-tag-sec")
        await pilot.pause()
        assert app.query("#wl-tree-node-watchlist-2")
        assert not app.query("#wl-tree-node-watchlist-1")
        # The permanent roots survive filtering — they are not watchlists.
        assert app.query("#wl-tree-node-all")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Watchlists/test_watchlist_tree.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named '...watchlist_tree'`.

- [ ] **Step 3: Write the implementation**

Create `tldw_chatbook/UI/Watchlists_Modules/watchlist_tree.py`:

```python
"""Left-rail watchlist tree: roots, watchlists, lazily-loaded sources.

This is the screen's primary navigation surface. Selecting a node sets a
*scope* that the Feeds and Items regions read, which is why the message
carries a structured `TreeScope` rather than a bare id — "watchlist 1" and
"source 10 inside watchlist 1" are different scopes with the same numbers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Static


ALL_SOURCES_BUCKET = -2
UNASSIGNED_BUCKET = -1


@dataclass(frozen=True)
class TreeScope:
    """What the user has selected, as the panes need to understand it."""

    kind: Literal["all", "unassigned", "watchlist", "source"]
    watchlist_id: int | None = None
    source_id: int | None = None


class TreeScopeChanged(Message):
    """Posted when the tree selection changes."""

    def __init__(self, scope: TreeScope) -> None:
        self.scope = scope
        super().__init__()


class WatchlistTree(Vertical):
    """Roots, watchlists with counts, lazily-expanded sources, tag filters."""

    expanded: reactive[frozenset[int]] = reactive(frozenset(), recompose=True)
    active_tag: reactive[str | None] = reactive(None, recompose=True)

    def __init__(
        self,
        watchlists: Sequence[Mapping[str, Any]],
        counts: Mapping[int, Mapping[str, int]],
        source_rows_loader: Callable[[int], Sequence[Mapping[str, Any]]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.add_class("watchlist-tree")
        self._watchlists = list(watchlists)
        self._counts = dict(counts)
        self._load_source_rows = source_rows_loader
        self._source_cache: dict[int, list[Mapping[str, Any]]] = {}

    # --- rendering ---

    def compose(self) -> ComposeResult:
        yield self._root_node("all", "All sources", ALL_SOURCES_BUCKET)
        yield self._root_node("unassigned", "Unassigned", UNASSIGNED_BUCKET)

        for watchlist in self._visible_watchlists():
            yield from self._watchlist_node(watchlist)

        tags = self._all_tags()
        if tags:
            yield Static("", classes="watchlist-tree-spacer")
            for tag in tags:
                button = Button(f"#{escape_markup(tag)}", id=f"wl-tree-tag-{tag}", compact=True)
                button.add_class("watchlist-tree-tag")
                if tag == self.active_tag:
                    button.add_class("is-active")
                yield button

    def _root_node(self, key: str, label: str, bucket: int) -> Button:
        unread = self._counts.get(bucket, {}).get("unread", 0)
        button = Button(f"{label}  {unread}", id=f"wl-tree-node-{key}", compact=True)
        button.add_class("watchlist-tree-root")
        return button

    def _watchlist_node(self, watchlist: Mapping[str, Any]) -> ComposeResult:
        watchlist_id = int(watchlist["id"])
        unread = self._counts.get(watchlist_id, {}).get("unread", 0)
        is_open = watchlist_id in self.expanded
        caret = "▾" if is_open else "▸"

        expander = Button(caret, id=f"wl-tree-expand-{watchlist_id}", compact=True)
        expander.add_class("watchlist-tree-expander")
        yield expander

        node = Button(
            f"{escape_markup(str(watchlist['name']))}  {unread}",
            id=f"wl-tree-node-watchlist-{watchlist_id}",
            compact=True,
        )
        node.add_class("watchlist-tree-watchlist")
        yield node

        if is_open:
            for row in self._source_rows(watchlist_id):
                source = Button(
                    f"  {escape_markup(str(row['name']))}",
                    id=f"wl-tree-node-source-{row['id']}",
                    compact=True,
                )
                source.add_class("watchlist-tree-source")
                yield source

    # --- data ---

    def _visible_watchlists(self) -> list[Mapping[str, Any]]:
        if self.active_tag is None:
            return self._watchlists
        return [w for w in self._watchlists if self.active_tag in (w.get("tags") or [])]

    def _all_tags(self) -> list[str]:
        seen: list[str] = []
        for watchlist in self._watchlists:
            for tag in watchlist.get("tags") or []:
                if tag not in seen:
                    seen.append(tag)
        return seen

    def _source_rows(self, watchlist_id: int) -> list[Mapping[str, Any]]:
        """Fetch a watchlist's sources once, on first expand."""
        if watchlist_id not in self._source_cache:
            self._source_cache[watchlist_id] = list(self._load_source_rows(watchlist_id))
        return self._source_cache[watchlist_id]

    def _watchlist_of_source(self, source_id: int) -> int | None:
        for watchlist_id, rows in self._source_cache.items():
            if any(int(row["id"]) == source_id for row in rows):
                return watchlist_id
        return None

    # --- interaction ---

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""

        if button_id.startswith("wl-tree-expand-"):
            event.stop()
            watchlist_id = int(button_id.rsplit("-", 1)[1])
            expanded = set(self.expanded)
            expanded.symmetric_difference_update({watchlist_id})
            self.expanded = frozenset(expanded)
            return

        if button_id.startswith("wl-tree-tag-"):
            event.stop()
            tag = button_id[len("wl-tree-tag-"):]
            self.active_tag = None if tag == self.active_tag else tag
            return

        scope: TreeScope | None = None
        if button_id == "wl-tree-node-all":
            scope = TreeScope(kind="all")
        elif button_id == "wl-tree-node-unassigned":
            scope = TreeScope(kind="unassigned")
        elif button_id.startswith("wl-tree-node-watchlist-"):
            scope = TreeScope(kind="watchlist", watchlist_id=int(button_id.rsplit("-", 1)[1]))
        elif button_id.startswith("wl-tree-node-source-"):
            source_id = int(button_id.rsplit("-", 1)[1])
            scope = TreeScope(
                kind="source",
                watchlist_id=self._watchlist_of_source(source_id),
                source_id=source_id,
            )

        if scope is not None:
            event.stop()
            self.post_message(TreeScopeChanged(scope))
```

Note the `escape_markup` on every name: watchlist and source names are user- and feed-supplied, and this repo has shipped markup-injection bugs before.

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest Tests/Watchlists/test_watchlist_tree.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/watchlist_tree.py Tests/Watchlists/test_watchlist_tree.py
git commit -m "feat(watchlists): add the left-rail watchlist tree"
```

---

### Task 3: Section tabs, and retire the navigator

**Files:**
- Create: `tldw_chatbook/UI/Watchlists_Modules/watchlists_tab_strip.py`
- Delete: `tldw_chatbook/UI/Watchlists_Modules/watchlists_navigator.py`
- Delete: `Tests/Watchlists/test_watchlists_navigator.py`
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Test: `Tests/Watchlists/test_watchlists_tab_strip.py`

**Interfaces:**
- Consumes: the screen's existing `active_section` reactive and `SectionSelected` handling.
- Produces: `WatchlistsTabStrip(Horizontal)` and `SectionSelected(section_id)` — **reuse the existing message name and shape** so the screen's current handler keeps working unchanged.

**Deleting the navigator test is legitimate here** and is the one exception to "never delete a test to make things pass": it asserts on a widget this task removes. It is also the long-standing failure on dev (asserts 6 buttons, finds 5). Say so explicitly in your report.

- [ ] **Step 1: Write the failing test**

Create `Tests/Watchlists/test_watchlists_tab_strip.py`:

```python
import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.UI.Watchlists_Modules.watchlists_tab_strip import (
    SectionSelected,
    WatchlistsTabStrip,
)


class _StripApp(App):
    def __init__(self, active="overview"):
        super().__init__()
        self._active = active
        self.selected: list[str] = []

    def compose(self) -> ComposeResult:
        yield WatchlistsTabStrip(active_section=self._active, id="wl-tabs")

    def on_section_selected(self, message: SectionSelected) -> None:
        self.selected.append(message.section_id)


@pytest.mark.asyncio
async def test_every_section_has_a_tab():
    app = _StripApp()
    async with app.run_test():
        for section in ("overview", "sources", "items", "runs", "rules", "notifications"):
            assert app.query(f"#wl-tab-{section}"), f"missing tab for {section}"


@pytest.mark.asyncio
async def test_clicking_a_tab_posts_section_selected():
    app = _StripApp()
    async with app.run_test() as pilot:
        await pilot.click("#wl-tab-runs")
        await pilot.pause()
        assert app.selected == ["runs"]


@pytest.mark.asyncio
async def test_the_active_tab_is_marked():
    app = _StripApp(active="rules")
    async with app.run_test():
        assert app.query_one("#wl-tab-rules").has_class("is-active")
        assert not app.query_one("#wl-tab-runs").has_class("is-active")


@pytest.mark.asyncio
async def test_the_strip_is_one_row_tall():
    app = _StripApp()
    async with app.run_test():
        assert app.query_one(WatchlistsTabStrip).styles.height.value == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Watchlists/test_watchlists_tab_strip.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write the tab strip**

Create `tldw_chatbook/UI/Watchlists_Modules/watchlists_tab_strip.py`:

```python
"""One-row section tab strip for the Watchlists centre.

Replaces the left-rail section navigator: the rail now holds the watchlist
tree, which is what the user actually navigates by. The message type and
shape are unchanged from the navigator so the screen's existing handler
keeps working.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Button


SECTIONS: tuple[tuple[str, str], ...] = (
    ("overview", "Overview"),
    ("sources", "Sources"),
    ("items", "Items"),
    ("runs", "Runs"),
    ("rules", "Rules"),
    ("notifications", "Notifications"),
)


class SectionSelected(Message):
    """Posted when the user selects a section."""

    def __init__(self, section_id: str) -> None:
        self.section_id = section_id
        super().__init__()


class WatchlistsTabStrip(Horizontal):
    """Compact one-row tab strip across the top of the centre stack."""

    def __init__(self, active_section: str = "overview", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_class("watchlists-tab-strip")
        self.active_section = active_section
        self.styles.height = 1
        self.styles.min_height = 1

    def compose(self) -> ComposeResult:
        for section_id, label in SECTIONS:
            tab = Button(label, id=f"wl-tab-{section_id}", compact=True)
            tab.add_class("watchlists-tab")
            if section_id == self.active_section:
                tab.add_class("is-active")
            yield tab

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        prefix = "wl-tab-"
        if not button_id.startswith(prefix):
            return
        event.stop()
        self.post_message(SectionSelected(button_id[len(prefix):]))
```

- [ ] **Step 4: Retire the navigator**

In `watchlists_collections_screen.py`, change the `SectionSelected` import to come from `watchlists_tab_strip` instead of `watchlists_navigator`, and drop the `WatchlistsNavigator` import. Then:

```bash
git rm tldw_chatbook/UI/Watchlists_Modules/watchlists_navigator.py
git rm Tests/Watchlists/test_watchlists_navigator.py
grep -rn "WatchlistsNavigator\|watchlists_navigator" tldw_chatbook Tests
```

That grep must return nothing before you continue.

The screen's `@on(SectionSelected)` handler and its `active_section` reactive stay exactly as they are — only the sender changes.

- [ ] **Step 5: Run the tests**

Run, separately: `pytest Tests/Watchlists/test_watchlists_tab_strip.py -v`, then `pytest Tests/Watchlists -v`, then `pytest Tests/UI/test_watchlists_destination_shell.py -v`.
Expected: PASS. Note that `Tests/Watchlists` should now be fully green — the long-standing navigator failure is gone because the navigator is gone.

- [ ] **Step 6: Commit**

```bash
git add -A tldw_chatbook/UI/Watchlists_Modules tldw_chatbook/UI/Screens/watchlists_collections_screen.py Tests/Watchlists
git commit -m "feat(watchlists): replace the section navigator with a centre tab strip"
```

---

### Task 4: Wire the tree and tabs into the screen

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Test: `Tests/UI/test_watchlists_destination_shell.py`

**Interfaces:**
- Consumes: `WatchlistTree`, `TreeScope`, `TreeScopeChanged` (Task 2); `WatchlistsTabStrip` (Task 3); `list_source_rows` (Task 1).
- Produces: a `selected_scope: reactive[TreeScope]` on the screen, which Phase D's panes read.

- [ ] **Step 1: Write the failing test**

Append to `Tests/UI/test_watchlists_destination_shell.py` — **additions only**:

```python
@pytest.mark.asyncio
async def test_left_rail_hosts_the_tree_not_the_navigator(watchlists_app):
    async with watchlists_app.run_test() as pilot:
        await pilot.pause()
        screen = watchlists_app.screen
        assert screen.query("#wl-tree")
        assert not screen.query("#watchlists-navigator")


@pytest.mark.asyncio
async def test_centre_hosts_the_tab_strip(watchlists_app):
    async with watchlists_app.run_test() as pilot:
        await pilot.pause()
        assert watchlists_app.screen.query("#wl-tabs")


@pytest.mark.asyncio
async def test_clicking_a_tab_switches_the_active_section(watchlists_app):
    async with watchlists_app.run_test() as pilot:
        await pilot.pause()
        screen = watchlists_app.screen
        await pilot.click("#wl-tab-runs")
        await pilot.pause()
        assert screen.active_section == "runs"


@pytest.mark.asyncio
async def test_tree_selection_sets_the_screen_scope(watchlists_app):
    from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import TreeScope, TreeScopeChanged

    async with watchlists_app.run_test() as pilot:
        await pilot.pause()
        screen = watchlists_app.screen
        screen.post_message(TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=7)))
        await pilot.pause()
        assert screen.selected_scope.kind == "watchlist"
        assert screen.selected_scope.watchlist_id == 7


@pytest.mark.asyncio
async def test_scope_survives_a_region_toggle(watchlists_app):
    from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import TreeScope, TreeScopeChanged

    async with watchlists_app.run_test() as pilot:
        await pilot.pause()
        screen = watchlists_app.screen
        screen.post_message(TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=7)))
        await pilot.pause()
        await pilot.press("[")
        await pilot.pause()
        assert screen.selected_scope.watchlist_id == 7, (
            "scope lives on the screen, so a workbench recompose must not lose it"
        )
```

- [ ] **Step 2: Run to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/UI/test_watchlists_destination_shell.py -v`
Expected: the five new tests FAIL (`#wl-tree` absent, no `selected_scope`); existing tests still pass.

- [ ] **Step 3: Add the scope reactive and tree data loading**

In the screen, add:

```python
from ..Watchlists_Modules.watchlist_tree import TreeScope, TreeScopeChanged, WatchlistTree
from ..Watchlists_Modules.watchlists_tab_strip import WatchlistsTabStrip
```

```python
    selected_scope = reactive(TreeScope(kind="all"))
```

and a loader that fetches both tree inputs in a worker, mirroring how the screen already loads section data:

```python
    @work(exclusive=True, group="wc_tree")
    async def _load_tree_data(self) -> None:
        """Load watchlists and their counts — two queries, never per node."""
        try:
            service = self._watchlist_bundle_service()
            db = self._subscriptions_db()
            self._tree_watchlists = service.list_watchlists()
            self._tree_counts = db.get_watchlist_item_counts()
        except Exception:
            logger.opt(exception=True).debug("Failed to load watchlists tree data.")
            self._tree_watchlists, self._tree_counts = [], {}
        self.refresh(recompose=True)
```

Call it from `on_mount` alongside the existing loaders. If the screen has no accessor for the bundle service yet, add one that mirrors how it reaches `watchlist_scope_service` — do not construct a service inline in a builder.

- [ ] **Step 4: Map the tree and tabs into the regions**

Change the workbench `content` mapping so `LEFT_RAIL` builds the tree, and put the tab strip at the top of the `FEEDS` region's builder:

```python
                    Region.LEFT_RAIL: self._build_tree_pane,
```

```python
    def _build_tree_pane(self) -> WatchlistTree:
        """Left rail: the watchlist tree. A factory, because the workbench
        recomposes and a widget instance can only be mounted once."""
        return WatchlistTree(
            watchlists=self._tree_watchlists,
            counts=self._tree_counts,
            source_rows_loader=self._load_source_rows_for_tree,
            id="wl-tree",
        )

    def _load_source_rows_for_tree(self, watchlist_id: int) -> list[dict[str, Any]]:
        """Synchronous, because the tree calls it during compose on expand.
        One query (Task 1's JOIN), so this is safe on the UI thread."""
        try:
            return self._watchlist_bundle_service().list_source_rows(watchlist_id)
        except Exception:
            logger.opt(exception=True).debug("Failed to load tree source rows.")
            return []
```

- [ ] **Step 5: Handle the scope message**

```python
    @on(TreeScopeChanged)
    def _on_tree_scope_changed(self, event: TreeScopeChanged) -> None:
        event.stop()
        self.selected_scope = event.scope
```

`selected_scope` lives on the screen precisely so it survives the workbench's recompose — the same reason `selected_run` and the create-form draft do.

- [ ] **Step 6: Run the tests**

Run, separately: `pytest Tests/UI/test_watchlists_destination_shell.py -v`, `pytest Tests/Watchlists -v`, `pytest Tests/UI/test_destination_shells.py -v`, `pytest Tests/UI/test_console_live_work_handoffs.py -v`.
Expected: PASS, with no existing test body modified.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/UI/Screens/watchlists_collections_screen.py Tests/UI/test_watchlists_destination_shell.py
git commit -m "feat(watchlists): wire the tree and tab strip into the screen"
```

---

### Task 5: Inspector breadcrumb stack, and stop truncating it

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/inspector_pane.py`
- Modify: `tldw_chatbook/css/features/_watchlists.tcss`
- Test: `Tests/Watchlists/test_watchlists_inspector.py`

**Interfaces:**
- Consumes: `TreeScope` (Task 2), the screen's existing `selected_entity`.
- Produces: an Inspector that renders a breadcrumb stack.

**The two defects, both measured on the live screen:**

1. At 26 columns every action label truncates — `Stage Watchlists Cont`, `Open current Watchlis`, `Console follow unavai`. Widen the right rail enough for the longest label, and let labels wrap rather than clip if they still do not fit.
2. The Inspector shows one flat level. The spec calls for a **breadcrumb stack** — watchlist ▸ source ▸ item — with the deepest selection expanded, shallower levels collapsed to one line, and **the action buttons always belonging to the deepest expanded level**. Clicking a shallower breadcrumb promotes it, swapping detail and actions together, so the actions on screen can never belong to a different level than the detail above them.

- [ ] **Step 1: Write the failing test**

Append to `Tests/Watchlists/test_watchlists_inspector.py`:

```python
@pytest.mark.asyncio
async def test_breadcrumb_shows_each_selected_level():
    from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import TreeScope

    pane = InspectorPane(id="insp")
    app = _InspectorApp(pane)
    async with app.run_test() as pilot:
        pane.scope = TreeScope(kind="source", watchlist_id=1, source_id=10)
        pane.breadcrumb_labels = ["Morning AI Brief", "ArXiv: AI"]
        await pilot.pause()
        text = " ".join(str(n.renderable) for n in app.query("Static"))
        assert "Morning AI Brief" in text
        assert "ArXiv: AI" in text


@pytest.mark.asyncio
async def test_actions_belong_to_the_deepest_level():
    from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import TreeScope

    pane = InspectorPane(id="insp")
    app = _InspectorApp(pane)
    async with app.run_test() as pilot:
        pane.scope = TreeScope(kind="watchlist", watchlist_id=1)
        await pilot.pause()
        assert app.query("#inspector-action-check-now")
        assert not app.query("#inspector-action-open"), (
            "an item action must not show while a watchlist is the deepest selection"
        )
```

Reuse whatever harness the existing tests in that file already use rather than adding a second one.

- [ ] **Step 2: Run to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Watchlists/test_watchlists_inspector.py -v`
Expected: FAIL — the pane has no `scope`/`breadcrumb_labels` and renders one flat level.

- [ ] **Step 3: Implement the breadcrumb stack**

Give `InspectorPane` a `scope: reactive[TreeScope | None]` and `breadcrumb_labels: reactive[list[str]]`, both `recompose=True`. Render one line per shallower level and the full detail plus action buttons for the deepest. Derive which action set to show from `scope.kind`, so it is impossible to render item actions under a watchlist detail.

Keep every existing action id (`#watchlists-follow-in-console`, the Console staging button, and the rest) — the guard tests in `Tests/UI/test_console_live_work_handoffs.py` and `Tests/UI/test_destination_shells.py` click them by id.

- [ ] **Step 4: Widen the rail**

In `_watchlists.tcss`, raise the right rail from its current width so the longest action label fits, and allow wrapping as a fallback:

```css
.watchlists-region-right_rail {
    width: 34;
    min-width: 0;
    height: 100%;
    border: round $ds-grid-line;
}

.watchlists-region-right_rail Button {
    width: 100%;
    text-wrap: wrap;
}
```

Verify against the real stylesheet, not a bare `App`: the production-CSS harness in `Tests/UI/test_destination_visual_parity_correction.py` is how the rail-collapse bug was caught. Add an assertion that no Inspector action label is clipped.

- [ ] **Step 5: Run the tests**

Run, separately: `pytest Tests/Watchlists/test_watchlists_inspector.py -v`, `pytest Tests/UI/test_destination_visual_parity_correction.py -v`, then the three guard files.
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/inspector_pane.py tldw_chatbook/css/features/_watchlists.tcss Tests/Watchlists/test_watchlists_inspector.py
git commit -m "feat(watchlists): give the Inspector a breadcrumb stack and room to render"
```

---

### Task 6: One border, one title per region

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/watchlists_workbench.py`
- Modify: `tldw_chatbook/css/features/_watchlists.tcss`
- Test: `Tests/Watchlists/test_watchlists_workbench.py`

**Interfaces:**
- Consumes: `REGION_TITLES` from the workbench.
- Produces: no API change — this is chrome.

**The defect, measured:** every region draws a border *and* a title, and the pane inside draws its own border and title too. The result is nested boxes and doubled headings — the region called "Feeds" renders a box titled "Feeds" containing a box titled "Sources".

Two things to decide with evidence rather than taste, and to state in your report:

1. **Which title survives.** The region title is generic and currently wrong (region "Feeds" hosts the Sources pane); the pane title is accurate. Prefer suppressing the region title where a pane supplies its own, rather than renaming regions to match panes that Phase D will replace anyway.
2. **Which border survives.** Keep exactly one. The region border is what collapse acts on, so it is the stronger candidate — but check how the panes look without their own.

- [ ] **Step 1: Write the failing test**

Append to `Tests/Watchlists/test_watchlists_workbench.py`:

```python
@pytest.mark.asyncio
async def test_a_region_with_supplied_content_does_not_double_title():
    from textual.widgets import Static

    def factory():
        return Static("Sources", classes="pane-title")

    class _App(App):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(), content={Region.FEEDS: factory}, id="wl-workbench"
            )

    app = _App()
    async with app.run_test():
        titles = [str(n.renderable) for n in app.query(".watchlists-region-title")]
        assert "Feeds" not in titles, (
            "a region whose content supplies its own heading should not add a second one"
        )
```

- [ ] **Step 2: Run to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Watchlists/test_watchlists_workbench.py -v`
Expected: FAIL — the region renders its own title alongside the supplied content's.

- [ ] **Step 3: Suppress the redundant chrome**

In `_region_widget`, omit the region title `Static` when a content factory was supplied for that region, keeping it only for regions still rendering the placeholder stub. Then remove the duplicated border in CSS so exactly one box is drawn per region.

- [ ] **Step 4: Verify against the real stylesheet**

Run: `source .venv/bin/activate && pytest Tests/UI/test_destination_visual_parity_correction.py -v`
Expected: PASS, including the existing vertical-stack geometry assertions.

- [ ] **Step 5: Confirm it in the running app**

Launch the app and capture the Watchlists screen — the recipe is in `.claude/skills/verify` (tmux, SGR clicks, scratch `TLDW_CONFIG_PATH`). Confirm by eye that each region shows one border and one heading, and paste the capture into your report. **Never point the app at the real `~/.config/tldw_cli/config.toml`.**

This step is not optional. Chrome defects are exactly what unit tests miss and a screenshot catches in seconds — that is how the four defects this phase fixes were found in the first place.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/watchlists_workbench.py tldw_chatbook/css/features/_watchlists.tcss Tests/Watchlists/test_watchlists_workbench.py
git commit -m "feat(watchlists): draw one border and one title per region"
```

---

### Task 7: Make the scope drive the Feeds region

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Test: `Tests/UI/test_watchlists_destination_shell.py`

**Interfaces:**
- Consumes: `selected_scope` (Task 4), `list_source_rows` (Task 1).
- Produces: a Feeds region whose contents follow the tree selection.

**Why this task exists.** Without it, Task 4 ships a `selected_scope` that nothing reads — the tree
would look functional and change nothing. This is the minimum that makes the navigation real. The
full feeds/items **tables** are Phase D's work alongside the reader; this task only makes the Feeds
region show the sources the current scope covers.

Scope semantics, which the spec fixes and the tests must pin:

| Scope | Feeds region shows |
|---|---|
| `all` | every source |
| `unassigned` | sources belonging to no watchlist |
| `watchlist` | that watchlist's sources |
| `source` | that one source |

- [ ] **Step 1: Write the failing test**

Append to `Tests/UI/test_watchlists_destination_shell.py` — additions only:

```python
@pytest.mark.asyncio
async def test_feeds_region_follows_the_tree_scope(watchlists_app):
    from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import TreeScope, TreeScopeChanged

    async with watchlists_app.run_test() as pilot:
        await pilot.pause()
        screen = watchlists_app.screen

        screen.post_message(TreeScopeChanged(TreeScope(kind="all")))
        await pilot.pause()
        all_rows = screen.scoped_source_rows()

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=1))
        )
        await pilot.pause()
        scoped_rows = screen.scoped_source_rows()

        assert scoped_rows != all_rows or all_rows == [], (
            "narrowing the scope to one watchlist must change what Feeds covers"
        )


@pytest.mark.asyncio
async def test_source_scope_narrows_to_exactly_one(watchlists_app):
    from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import TreeScope, TreeScopeChanged

    async with watchlists_app.run_test() as pilot:
        await pilot.pause()
        screen = watchlists_app.screen
        screen.post_message(
            TreeScopeChanged(TreeScope(kind="source", watchlist_id=1, source_id=10))
        )
        await pilot.pause()
        rows = screen.scoped_source_rows()
        assert len(rows) <= 1
        assert all(int(r["id"]) == 10 for r in rows)
```

- [ ] **Step 2: Run to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/UI/test_watchlists_destination_shell.py -k scope -v`
Expected: FAIL — the screen has no `scoped_source_rows`.

- [ ] **Step 3: Implement the scope resolution**

```python
    def scoped_source_rows(self) -> list[dict[str, Any]]:
        """Source rows the current tree scope covers.

        The Feeds region renders these, so selecting a node in the tree
        actually narrows what the centre shows rather than only recording a
        selection. Kept on the screen (not the pane) because the workbench
        recomposes and pane-local state does not survive it.
        """
        service = self._watchlist_bundle_service()
        if service is None:
            return []
        scope = self.selected_scope
        try:
            if scope.kind == "watchlist" and scope.watchlist_id is not None:
                return service.list_source_rows(scope.watchlist_id)
            if scope.kind == "source" and scope.source_id is not None:
                rows = (
                    service.list_source_rows(scope.watchlist_id)
                    if scope.watchlist_id is not None
                    else self._all_source_rows()
                )
                return [r for r in rows if int(r["id"]) == int(scope.source_id)]
            if scope.kind == "unassigned":
                return self._unassigned_source_rows()
            return self._all_source_rows()
        except Exception:
            logger.opt(exception=True).debug("Failed to resolve scoped source rows.")
            return []
```

Add the two remaining resolvers to `WatchlistBundleService` (not the screen — they are queries, and
the service owns queries), each **one** statement. Do not loop `list_source_rows` over every
watchlist; that is the N+1 this whole design keeps avoiding.

```python
    def list_all_source_rows(self) -> list[dict[str, Any]]:
        """Every source, in the shape the tree and Feeds region render.

        Returns:
            One dict per source with ``id``, ``name`` and ``type``.
        """
        rows = self._db.conn.execute(
            "SELECT id, name, type FROM subscriptions ORDER BY LOWER(name), id"
        ).fetchall()
        return [{"id": r[0], "name": r[1], "type": r[2]} for r in rows]

    def list_unassigned_source_rows(self) -> list[dict[str, Any]]:
        """Sources belonging to no watchlist.

        These would be unreachable from a watchlist-only tree, which is why
        the tree carries a permanent Unassigned root.

        Returns:
            One dict per unassigned source with ``id``, ``name`` and ``type``.
        """
        rows = self._db.conn.execute(
            """
            SELECT s.id, s.name, s.type
            FROM subscriptions s
            WHERE NOT EXISTS (
                SELECT 1 FROM watchlist_sources ws WHERE ws.subscription_id = s.id
            )
            ORDER BY LOWER(s.name), s.id
            """
        ).fetchall()
        return [{"id": r[0], "name": r[1], "type": r[2]} for r in rows]
```

Have the screen call these through its service accessor, and add a service-level test that each
issues exactly one query (reuse the counting-connection pattern from Task 1's tests).

Have `_build_list_pane` render `scoped_source_rows()` with a heading naming the scope (for example `Feeds in Morning AI Brief (3)`), and make `watch_selected_scope` refresh so the region follows the selection.

- [ ] **Step 4: Run the tests**

Run, separately: `pytest Tests/UI/test_watchlists_destination_shell.py -v`, then `pytest Tests/Watchlists -v`, then the other two guard files.
Expected: PASS, no existing test body edited.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/watchlists_collections_screen.py Tests/UI/test_watchlists_destination_shell.py
git commit -m "feat(watchlists): make the Feeds region follow the tree scope"
```

---

## Phase C Completion Checklist

- [ ] All seven tasks committed.
- [ ] `Tests/Watchlists`, `Tests/UI/test_watchlists_destination_shell.py`, `Tests/UI/test_destination_shells.py`, `Tests/UI/test_console_live_work_handoffs.py`, `Tests/UI/test_destination_visual_parity_correction.py` all pass, with no existing test body edited in the three guards.
- [ ] `grep -rn "WatchlistsNavigator\|watchlists_navigator" tldw_chatbook Tests` returns nothing.
- [ ] No file under `tldw_chatbook/DB` modified.
- [ ] Tree counts are one query; expanding a watchlist is one more; scope resolution is one. None scale with node count.
- [ ] `WatchlistBundleService` has a production caller — `grep -rn "WatchlistBundleService" tldw_chatbook | grep -v watchlist_bundle_service.py` shows the app wiring, not just a comment.
- [ ] A live capture shows: tree in the rail, tabs in the centre, one border and one heading per region, and no truncated Inspector label.
- [ ] Phase D plan written against merged Phase C code.
