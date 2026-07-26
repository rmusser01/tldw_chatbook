# Watchlists Rebuild — Phase B: Workbench & Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Watchlists screen's placeholder three-column shell with a Console-styled workbench — two collapsible rails around a vertically-stacked, independently collapsible centre — leaving the panes themselves as stubs for Phase C.

**Architecture:** A pure state machine (`region_layout.py`) owns collapse/solo/restore across five regions and is unit-testable with no Textual pilot. A purpose-built container (`watchlists_workbench.py`) renders it, because the shared `DestinationWorkbench` is a fixed equal-width `Horizontal` with no collapse, resize, or stacking. The screen shell becomes thin routing over those two, with Console handoff extracted to its own module.

**Tech Stack:** Python ≥3.11, Textual ≥3.3.0, pytest. No new dependencies.

## Global Constraints

- Spec: `Docs/superpowers/specs/2026-07-25-watchlists-console-rebuild-design.md`. Every requirement below traces to it.
- **No data-layer changes.** Phase A (PR #917) owns `SubscriptionsDB`, `item_persist.py`, and `watchlist_bundle_service.py`. Do not modify them.
- **Panes stay stubs.** Phase B builds the container and the collapse behaviour. Real tables, the tree, and the reader are Phase C/D. A stub renders its title and a placeholder line — nothing more.
- The screen class name, route (`watchlists_collections`), and existing stable widget selectors are preserved so Console handoffs and route tests keep passing.
- `DestinationModeStrip` and the `$ds-*` token set are reused. `DestinationWorkbench` is **not** — verified: fixed `width: 1fr` panes composed once from a frozen tuple, no collapse, no stacking.
- Reuse Console's vocabulary: `ConsoleRailHandle` (`Widgets/Console/console_rail_handle.py`) is the model for a collapsed rail's focusable handle.
- **Config lookups are flat, not dotted.** `get_cli_setting("watchlists.layout", …)` silently returns the default — this repo has already shipped that bug with `chat.images`. Use `get_cli_setting("watchlists", "collapsed_regions", …)`.
- Tests run from the venv: `source .venv/bin/activate && pytest …`. The `timeout` command does not exist here.
- **Run one test file at a time, in the foreground.** Anything past ~90 seconds is auto-backgrounded by this environment and appears to hang.

## Phase Map

| Phase | Scope | Status |
|---|---|---|
| A | Schema, unified persist, FTS5, bundle service, counts | **merged — PR #917** |
| **B (this plan)** | Region state machine, workbench container, shell rewrite, Console-handoff extraction | this plan |
| C | Tree, feeds pane, items pane, Inspector breadcrumb stack | next |
| D | Content pane: article + change renderers, escaping, HTML fallback | after C |
| E | Sources / Runs / Rules / Artifacts tabs | after C |

## File Structure

| Path | Responsibility | Task |
|---|---|---|
| `tldw_chatbook/UI/Watchlists_Modules/region_layout.py` | Five-region collapse/solo/restore. Pure, no Textual import | 1 |
| `tldw_chatbook/UI/Watchlists_Modules/region_layout_store.py` | Load/save collapse state to config | 2 |
| `tldw_chatbook/UI/Watchlists_Modules/watchlists_workbench.py` | Rails + stacked collapsible centre container | 3 |
| `tldw_chatbook/UI/Watchlists_Modules/watchlists_console_handoff.py` | Console staging/follow, extracted from the shell | 4 |
| `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` | Thin shell: rails, tabs, bindings, routing | 5 |
| `tldw_chatbook/css/features/_watchlists.tcss` | Workbench and region styling | 3 |
| `Tests/Watchlists/test_region_layout.py` | Pure state machine — no pilot | 1 |
| `Tests/Watchlists/test_region_layout_store.py` | Config round-trip | 2 |
| `Tests/Watchlists/test_watchlists_workbench.py` | Container rendering and collapse | 3 |
| `Tests/UI/test_watchlists_destination_shell.py` | Extended: bindings, tabs, route stability | 5 |

---

### Task 1: Region layout state machine

**Files:**
- Create: `tldw_chatbook/UI/Watchlists_Modules/region_layout.py`
- Test: `Tests/Watchlists/test_region_layout.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Region` (str enum: `LEFT_RAIL`, `FEEDS`, `ITEMS`, `CONTENT`, `RIGHT_RAIL`), `CENTRE_REGIONS` tuple, and `RegionLayout` — a frozen dataclass with `collapsed: frozenset[Region]`, `solo_region: Region | None`, and methods `is_collapsed(region) -> bool`, `toggle(region) -> RegionLayout`, `solo(region) -> RegionLayout`, `visible() -> tuple[Region, ...]`. Tasks 2, 3, and 5 all consume these.

This is a pure module so the fiddliest interaction in the screen is testable without a Textual pilot. It must not import Textual.

- [ ] **Step 1: Write the failing test**

Create `Tests/Watchlists/test_region_layout.py`:

```python
import pytest

from tldw_chatbook.UI.Watchlists_Modules.region_layout import (
    CENTRE_REGIONS,
    Region,
    RegionLayout,
)


def test_default_layout_has_everything_visible():
    layout = RegionLayout()
    assert layout.collapsed == frozenset()
    assert layout.solo_region is None
    assert layout.visible() == (
        Region.LEFT_RAIL, Region.FEEDS, Region.ITEMS, Region.CONTENT, Region.RIGHT_RAIL,
    )


def test_toggle_collapses_then_expands():
    layout = RegionLayout().toggle(Region.CONTENT)
    assert layout.is_collapsed(Region.CONTENT)
    assert Region.CONTENT not in layout.visible()

    layout = layout.toggle(Region.CONTENT)
    assert not layout.is_collapsed(Region.CONTENT)


def test_toggle_returns_a_new_instance_and_leaves_the_original_alone():
    original = RegionLayout()
    changed = original.toggle(Region.ITEMS)
    assert original.collapsed == frozenset()
    assert changed is not original


def test_rails_collapse_independently_of_the_centre():
    layout = RegionLayout().toggle(Region.LEFT_RAIL).toggle(Region.RIGHT_RAIL)
    assert layout.is_collapsed(Region.LEFT_RAIL)
    assert layout.is_collapsed(Region.RIGHT_RAIL)
    for region in CENTRE_REGIONS:
        assert not layout.is_collapsed(region)


def test_solo_collapses_the_other_centre_regions_only():
    layout = RegionLayout().solo(Region.ITEMS)
    assert layout.solo_region == Region.ITEMS
    assert not layout.is_collapsed(Region.ITEMS)
    assert layout.is_collapsed(Region.FEEDS)
    assert layout.is_collapsed(Region.CONTENT)
    # Rails are untouched by solo.
    assert not layout.is_collapsed(Region.LEFT_RAIL)
    assert not layout.is_collapsed(Region.RIGHT_RAIL)


def test_solo_twice_restores_the_prior_layout():
    before = RegionLayout().toggle(Region.FEEDS).toggle(Region.LEFT_RAIL)
    after = before.solo(Region.ITEMS).solo(Region.ITEMS)
    assert after.collapsed == before.collapsed
    assert after.solo_region is None


def test_solo_on_a_different_region_re_solos_without_stacking():
    layout = RegionLayout().solo(Region.ITEMS).solo(Region.CONTENT)
    assert layout.solo_region == Region.CONTENT
    assert layout.is_collapsed(Region.ITEMS)
    assert not layout.is_collapsed(Region.CONTENT)
    # Restoring from here returns to the ORIGINAL pre-solo layout, not to the ITEMS solo.
    restored = layout.solo(Region.CONTENT)
    assert restored.collapsed == frozenset()
    assert restored.solo_region is None


def test_manual_toggle_while_soloed_clears_solo():
    # Otherwise a later Z would "restore" a layout the user has since edited by hand.
    layout = RegionLayout().solo(Region.ITEMS).toggle(Region.FEEDS)
    assert layout.solo_region is None
    assert not layout.is_collapsed(Region.FEEDS)


def test_solo_rejects_rails():
    with pytest.raises(ValueError, match="centre region"):
        RegionLayout().solo(Region.LEFT_RAIL)


def test_all_three_centre_regions_may_collapse_at_once():
    # Legal: each collapses to a one-line header that stays clickable, so this is recoverable.
    layout = RegionLayout()
    for region in CENTRE_REGIONS:
        layout = layout.toggle(region)
    assert all(layout.is_collapsed(region) for region in CENTRE_REGIONS)
    assert layout.visible() == (Region.LEFT_RAIL, Region.RIGHT_RAIL)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Watchlists/test_region_layout.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.UI.Watchlists_Modules.region_layout'`.

- [ ] **Step 3: Write the implementation**

Create `tldw_chatbook/UI/Watchlists_Modules/region_layout.py`:

```python
"""Collapse and solo state for the Watchlists workbench's five regions.

Pure state: no Textual import, no I/O. The screen's fiddliest interaction —
five independently collapsible regions plus a solo/restore toggle — lives
here so it can be tested without a Textual pilot.

Every mutator returns a new instance; the type is frozen and hashable, so a
Textual reactive can hold it and equality comparison decides whether to
re-render.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Region(str, Enum):
    """One collapsible region of the Watchlists workbench."""

    LEFT_RAIL = "left_rail"
    FEEDS = "feeds"
    ITEMS = "items"
    CONTENT = "content"
    RIGHT_RAIL = "right_rail"


#: Display order, left rail through right rail.
REGION_ORDER: tuple[Region, ...] = (
    Region.LEFT_RAIL,
    Region.FEEDS,
    Region.ITEMS,
    Region.CONTENT,
    Region.RIGHT_RAIL,
)

#: The vertically stacked centre panes. Only these may be soloed.
CENTRE_REGIONS: tuple[Region, ...] = (Region.FEEDS, Region.ITEMS, Region.CONTENT)


@dataclass(frozen=True)
class RegionLayout:
    """Which regions are collapsed, and whether one centre pane is soloed."""

    collapsed: frozenset[Region] = frozenset()
    solo_region: Region | None = None
    _pre_solo: frozenset[Region] | None = None

    def is_collapsed(self, region: Region) -> bool:
        """Whether ``region`` is currently collapsed to its header."""
        return region in self.collapsed

    def visible(self) -> tuple[Region, ...]:
        """Expanded regions, in display order."""
        return tuple(r for r in REGION_ORDER if r not in self.collapsed)

    def toggle(self, region: Region) -> RegionLayout:
        """Collapse ``region`` if expanded, expand it if collapsed.

        A manual toggle clears any solo: the user has edited the layout by
        hand, so a later solo-restore must not resurrect a stale snapshot.
        """
        collapsed = set(self.collapsed)
        if region in collapsed:
            collapsed.discard(region)
        else:
            collapsed.add(region)
        return RegionLayout(collapsed=frozenset(collapsed))

    def solo(self, region: Region) -> RegionLayout:
        """Collapse the other centre panes around ``region``; call again to restore.

        Rails are unaffected — solo is about the centre stack only.

        Args:
            region: The centre region to isolate.

        Returns:
            A layout with the other centre regions collapsed, or the
            pre-solo layout if ``region`` is already soloed.

        Raises:
            ValueError: If ``region`` is a rail rather than a centre region.
        """
        if region not in CENTRE_REGIONS:
            raise ValueError(f"{region!r} is not a centre region; solo applies to {CENTRE_REGIONS}")

        if self.solo_region == region:
            return RegionLayout(collapsed=self._pre_solo or frozenset())

        # Re-soloing a different pane keeps the ORIGINAL pre-solo snapshot, so
        # restore always returns to what the user had before soloing at all.
        baseline = self._pre_solo if self.solo_region is not None else self.collapsed
        rails = {r for r in self.collapsed if r not in CENTRE_REGIONS}
        others = {r for r in CENTRE_REGIONS if r != region}
        return RegionLayout(
            collapsed=frozenset(rails | others),
            solo_region=region,
            _pre_solo=baseline,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest Tests/Watchlists/test_region_layout.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/region_layout.py Tests/Watchlists/test_region_layout.py
git commit -m "feat(watchlists): add pure five-region collapse/solo state machine"
```

---

### Task 2: Persist collapse state to config

**Files:**
- Create: `tldw_chatbook/UI/Watchlists_Modules/region_layout_store.py`
- Test: `Tests/Watchlists/test_region_layout_store.py`

**Interfaces:**
- Consumes: `Region`, `RegionLayout` from Task 1.
- Produces: `load_region_layout() -> RegionLayout` and `save_region_layout(layout: RegionLayout) -> None`. Task 5 calls both.

**The trap this task exists to avoid.** `get_cli_setting` performs a **flat** section lookup. Passing a dotted section like `"watchlists.layout"` silently returns the default and the setting never round-trips — this repo has already shipped that exact bug with `chat.images`. Use section `"watchlists"`, key `"collapsed_regions"`.

Solo state is deliberately **not** persisted: it is a transient view mode, and restoring "soloed" across restarts would strand a user in a layout they did not choose.

- [ ] **Step 1: Write the failing test**

Create `Tests/Watchlists/test_region_layout_store.py`:

```python
from tldw_chatbook.UI.Watchlists_Modules import region_layout_store
from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region, RegionLayout


def test_round_trips_collapsed_regions(monkeypatch):
    saved = {}

    def fake_save(section, key, value):
        saved[(section, key)] = value
        return True

    monkeypatch.setattr(region_layout_store, "save_setting_to_cli_config", fake_save)
    region_layout_store.save_region_layout(
        RegionLayout(collapsed=frozenset({Region.CONTENT, Region.RIGHT_RAIL}))
    )

    # Flat section, not "watchlists.layout" — a dotted section silently no-ops.
    assert ("watchlists", "collapsed_regions") in saved
    assert sorted(saved[("watchlists", "collapsed_regions")]) == ["content", "right_rail"]

    monkeypatch.setattr(
        region_layout_store, "get_cli_setting",
        lambda section, key, default=None: saved.get((section, key), default),
    )
    loaded = region_layout_store.load_region_layout()
    assert loaded.collapsed == frozenset({Region.CONTENT, Region.RIGHT_RAIL})


def test_load_defaults_to_everything_expanded(monkeypatch):
    monkeypatch.setattr(
        region_layout_store, "get_cli_setting", lambda section, key, default=None: default
    )
    assert region_layout_store.load_region_layout() == RegionLayout()


def test_load_ignores_unknown_region_names(monkeypatch):
    # A config hand-edited or written by a newer version must not crash the screen.
    monkeypatch.setattr(
        region_layout_store, "get_cli_setting",
        lambda section, key, default=None: ["content", "nonsense", "left_rail"],
    )
    loaded = region_layout_store.load_region_layout()
    assert loaded.collapsed == frozenset({Region.CONTENT, Region.LEFT_RAIL})


def test_load_tolerates_a_non_list_value(monkeypatch):
    monkeypatch.setattr(
        region_layout_store, "get_cli_setting", lambda section, key, default=None: "content"
    )
    assert region_layout_store.load_region_layout().collapsed == frozenset({Region.CONTENT})


def test_save_never_persists_solo(monkeypatch):
    saved = {}
    monkeypatch.setattr(
        region_layout_store, "save_setting_to_cli_config",
        lambda section, key, value: saved.__setitem__((section, key), value) or True,
    )
    region_layout_store.save_region_layout(RegionLayout().solo(Region.ITEMS))
    assert sorted(saved[("watchlists", "collapsed_regions")]) == ["content", "feeds"]
    assert ("watchlists", "solo_region") not in saved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Watchlists/test_region_layout_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named '...region_layout_store'`.

- [ ] **Step 3: Write the implementation**

Create `tldw_chatbook/UI/Watchlists_Modules/region_layout_store.py`:

```python
"""Persist Watchlists workbench collapse state to the user's config.

Collapse state is UI preference, not data, so it belongs in config rather
than SubscriptionsDB. Solo is deliberately not persisted — it is a transient
view mode, and restoring it across restarts would strand the user in a
layout they did not choose.
"""

from __future__ import annotations

from collections.abc import Sequence

from loguru import logger

from ...config import get_cli_setting, save_setting_to_cli_config
from .region_layout import Region, RegionLayout


logger = logger.bind(module="WatchlistsRegionLayoutStore")

#: Flat section. `get_cli_setting` does NOT resolve dotted sections — passing
#: "watchlists.layout" silently returns the default and the setting never
#: round-trips. This repo has shipped that bug before with "chat.images".
_SECTION = "watchlists"
_KEY = "collapsed_regions"


def load_region_layout() -> RegionLayout:
    """Read collapse state from config, defaulting to everything expanded."""
    raw = get_cli_setting(_SECTION, _KEY, [])
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, Sequence):
        logger.debug("Ignoring non-sequence watchlists collapse state: {!r}", raw)
        return RegionLayout()

    collapsed = set()
    for value in raw:
        try:
            collapsed.add(Region(str(value)))
        except ValueError:
            logger.debug("Ignoring unknown watchlists region {!r} from config.", value)
    return RegionLayout(collapsed=frozenset(collapsed))


def save_region_layout(layout: RegionLayout) -> None:
    """Write collapse state to config. Solo state is not persisted."""
    values = sorted(region.value for region in layout.collapsed)
    try:
        save_setting_to_cli_config(_SECTION, _KEY, values)
    except Exception:
        logger.opt(exception=True).debug("Failed to persist watchlists collapse state.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest Tests/Watchlists/test_region_layout_store.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/region_layout_store.py Tests/Watchlists/test_region_layout_store.py
git commit -m "feat(watchlists): persist workbench collapse state to config"
```

---

### Task 3: Workbench container

**Files:**
- Create: `tldw_chatbook/UI/Watchlists_Modules/watchlists_workbench.py`
- Modify: `tldw_chatbook/css/features/_watchlists.tcss`
- Test: `Tests/Watchlists/test_watchlists_workbench.py`

**Interfaces:**
- Consumes: `Region`, `CENTRE_REGIONS`, `RegionLayout` from Task 1.
- Produces: `WatchlistsWorkbench(Horizontal)` with constructor `WatchlistsWorkbench(layout: RegionLayout, *, id: str | None = None)`, a **`region_layout`** reactive, and the message `RegionToggled(region: Region)` posted when a collapsed header or rail handle is activated. Task 5 mounts it and handles `RegionToggled`.

**The reactive cannot be called `layout`.** `Widget.layout` already exists in Textual as a read-only property the compositor calls `.arrange()` on during every render; shadowing it with a reactive breaks rendering. The constructor parameter stays `layout` for readability; the attribute is `region_layout`.

Each region renders a titled body when expanded and a single-line header when collapsed. **A collapsed region's header stays focusable and clickable** — otherwise collapse is one-way when focus cannot return to it.

Panes are stubs in this phase: a title and one placeholder line. Phase C replaces their bodies.

- [ ] **Step 1: Write the failing test**

Create `Tests/Watchlists/test_watchlists_workbench.py`:

```python
import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region, RegionLayout
from tldw_chatbook.UI.Watchlists_Modules.watchlists_workbench import (
    RegionToggled,
    WatchlistsWorkbench,
)


class _WorkbenchApp(App):
    def __init__(self, layout: RegionLayout) -> None:
        super().__init__()
        self._layout = layout
        self.toggles: list[Region] = []

    def compose(self) -> ComposeResult:
        yield WatchlistsWorkbench(self._layout, id="wl-workbench")

    def on_region_toggled(self, message: RegionToggled) -> None:
        self.toggles.append(message.region)


@pytest.mark.asyncio
async def test_all_regions_render_expanded_by_default():
    app = _WorkbenchApp(RegionLayout())
    async with app.run_test():
        for region in Region:
            assert app.query(f"#wl-region-{region.value}")
            assert not app.query(f"#wl-header-{region.value}")


@pytest.mark.asyncio
async def test_collapsed_region_renders_a_header_instead_of_a_body():
    app = _WorkbenchApp(RegionLayout(collapsed=frozenset({Region.CONTENT})))
    async with app.run_test():
        assert app.query("#wl-header-content")
        assert not app.query("#wl-region-content")


@pytest.mark.asyncio
async def test_collapsed_header_is_focusable_so_collapse_is_not_one_way():
    app = _WorkbenchApp(RegionLayout(collapsed=frozenset({Region.ITEMS})))
    async with app.run_test():
        header = app.query_one("#wl-header-items")
        assert header.focusable, "a collapsed region must be reachable by keyboard"


@pytest.mark.asyncio
async def test_clicking_a_collapsed_header_posts_region_toggled():
    app = _WorkbenchApp(RegionLayout(collapsed=frozenset({Region.FEEDS})))
    async with app.run_test() as pilot:
        await pilot.click("#wl-header-feeds")
        await pilot.pause()
        assert app.toggles == [Region.FEEDS]


@pytest.mark.asyncio
async def test_updating_the_layout_reactive_re_renders():
    app = _WorkbenchApp(RegionLayout())
    async with app.run_test() as pilot:
        workbench = app.query_one(WatchlistsWorkbench)
        workbench.region_layout = RegionLayout(collapsed=frozenset({Region.LEFT_RAIL}))
        await pilot.pause()
        assert app.query("#wl-header-left_rail")
        assert not app.query("#wl-region-left_rail")


@pytest.mark.asyncio
async def test_every_centre_region_may_be_collapsed_at_once():
    app = _WorkbenchApp(RegionLayout(collapsed=frozenset(CENTRE := {Region.FEEDS, Region.ITEMS, Region.CONTENT})))
    async with app.run_test():
        for region in CENTRE:
            assert app.query(f"#wl-header-{region.value}")
        # The rails survive, so the screen is never empty.
        assert app.query("#wl-region-left_rail")
        assert app.query("#wl-region-right_rail")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Watchlists/test_watchlists_workbench.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named '...watchlists_workbench'`.

- [ ] **Step 3: Write the implementation**

Create `tldw_chatbook/UI/Watchlists_Modules/watchlists_workbench.py`:

```python
"""Watchlists workbench: two collapsible rails around a stacked centre.

The shared ``DestinationWorkbench`` cannot express this layout — it is a
fixed ``Horizontal`` of equal-width panes composed once from a frozen tuple,
with no collapse, resize, or vertical stacking. If the collapse behaviour
here proves useful to a second screen, it graduates into the shared widget
then; generalising ahead of a second consumer is not worth it.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Static

from .region_layout import CENTRE_REGIONS, Region, RegionLayout


#: Human-readable titles, used for both expanded bodies and collapsed headers.
REGION_TITLES: dict[Region, str] = {
    Region.LEFT_RAIL: "Watchlists",
    Region.FEEDS: "Feeds",
    Region.ITEMS: "Items",
    Region.CONTENT: "Content",
    Region.RIGHT_RAIL: "Inspector",
}

#: Placeholder body copy. Phase C and D replace these with real panes.
REGION_PLACEHOLDERS: dict[Region, str] = {
    Region.LEFT_RAIL: "Watchlist tree arrives in the next slice.",
    Region.FEEDS: "Feeds table arrives in the next slice.",
    Region.ITEMS: "Items table arrives in the next slice.",
    Region.CONTENT: "Reader arrives in the next slice.",
    Region.RIGHT_RAIL: "Inspector arrives in the next slice.",
}


class RegionToggled(Message):
    """A collapsed region's header or rail handle was activated."""

    def __init__(self, region: Region) -> None:
        super().__init__()
        self.region = region


class WatchlistsWorkbench(Horizontal):
    """Renders a :class:`RegionLayout` as rails plus a stacked centre."""

    layout: reactive[RegionLayout] = reactive(RegionLayout(), recompose=True)

    def __init__(self, layout: RegionLayout, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_class("watchlists-workbench")
        self.set_reactive(WatchlistsWorkbench.layout, layout)

    def compose(self) -> ComposeResult:
        yield self._region_widget(Region.LEFT_RAIL)

        with Vertical(id="wl-centre", classes="watchlists-centre"):
            for region in CENTRE_REGIONS:
                yield self._region_widget(region)

        yield self._region_widget(Region.RIGHT_RAIL)

    def _region_widget(self, region: Region) -> Widget:
        """Build one region: a titled body, or a focusable one-line header.

        Returns a constructed widget rather than yielding, so `compose` stays
        the single place that mounts anything. Building children positionally
        avoids the `with container: ... ; yield container` shape, which
        double-mounts — Textual's `with` already adds the container.
        """
        if self.layout.is_collapsed(region):
            # A Button, not a Static: a collapsed region must stay focusable
            # and clickable, or collapsing it is one-way.
            header = Button(
                f"▸ {REGION_TITLES[region]}",
                id=f"wl-header-{region.value}",
                compact=True,
            )
            header.add_class("watchlists-region-header")
            header.tooltip = f"Expand {REGION_TITLES[region]}"
            return header

        body = Vertical(
            Static(REGION_TITLES[region], classes="watchlists-region-title"),
            Static(REGION_PLACEHOLDERS[region], classes="watchlists-region-placeholder"),
            id=f"wl-region-{region.value}",
            classes=f"watchlists-region watchlists-region-{region.value}",
        )
        # Regions must be keyboard-reachable, or `z` cannot target them.
        body.can_focus = True
        return body

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        prefix = "wl-header-"
        if not button_id.startswith(prefix):
            return
        event.stop()
        self.post_message(RegionToggled(Region(button_id[len(prefix):])))
```

- [ ] **Step 4: Add the styling**

Append to `tldw_chatbook/css/features/_watchlists.tcss`:

```css
/* Workbench: rails around a vertically stacked centre */
.watchlists-workbench {
    width: 100%;
    height: 1fr;
    min-height: 0;
}

.watchlists-region-left_rail,
.watchlists-region-right_rail {
    width: 28;
    min-width: 0;
    height: 100%;
    border: round $ds-grid-line;
}

.watchlists-centre {
    width: 1fr;
    min-width: 0;
    height: 100%;
    min-height: 0;
}

.watchlists-region-feeds,
.watchlists-region-items,
.watchlists-region-content {
    width: 100%;
    height: 1fr;
    min-height: 0;
    border: round $ds-grid-line;
}

.watchlists-region-title {
    height: 1;
    text-style: bold;
    color: $ds-text-primary;
}

.watchlists-region-placeholder {
    color: $ds-text-muted;
}

/* Collapsed regions keep a one-line, focusable header so nothing vanishes */
.watchlists-region-header {
    height: 1;
    min-height: 1;
    width: 100%;
    color: $ds-text-muted;
}
```

If this project builds a bundled stylesheet, regenerate it rather than hand-editing the bundle — run `python -m tldw_chatbook.css.build_css` if that entry point exists, and never edit `tldw_cli_modular.tcss` directly.

- [ ] **Step 5: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest Tests/Watchlists/test_watchlists_workbench.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/watchlists_workbench.py tldw_chatbook/css/features/_watchlists.tcss Tests/Watchlists/test_watchlists_workbench.py
git commit -m "feat(watchlists): add collapsible workbench container"
```

---

### Task 4: Extract Console handoff from the shell

**Files:**
- Create: `tldw_chatbook/UI/Watchlists_Modules/watchlists_console_handoff.py`
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Test: `Tests/UI/test_watchlists_destination_shell.py` (existing — must keep passing)

**Interfaces:**
- Consumes: nothing new.
- Produces: `WatchlistsConsoleHandoff(app_instance)` with the staging and follow methods moved off the screen. Task 5's rewritten shell delegates to it.

**Why first, and separately.** The shell is 1,219 lines and Task 5 rewrites it. Extracting the Console handoff in its own commit means the rewrite is reviewable as a rewrite, rather than as a rewrite tangled with a move. It also makes Task 5's target of a thin shell achievable rather than aspirational.

**This task must not change behaviour.** It is a pure move: same logic, same messages, same stable selectors. The existing shell tests are the guard — they must pass unchanged, and you must not edit them to accommodate the move.

- [ ] **Step 1: Establish the baseline**

Run: `source .venv/bin/activate && pytest Tests/UI/test_watchlists_destination_shell.py -v`

Record the pass/fail counts. Every test passing now must still pass at the end of this task. Note any that already fail — those are pre-existing and not yours to fix.

- [ ] **Step 2: Identify exactly what moves**

In `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`, the Console handoff surface is the `@on(Button.Pressed, "#wc-attach-to-console")` handler, the `@on(Button.Pressed, "#watchlists-follow-in-console")` handler, the `@on(StageInConsoleRequested)` handler, and the private helpers they call (the `_latest_console_follow_*` state on `__init__` and the methods that populate it).

List them before editing. Moving a partial set leaves the shell holding half a feature, which is worse than not moving it.

- [ ] **Step 3: Create the module with the moved logic**

Create `tldw_chatbook/UI/Watchlists_Modules/watchlists_console_handoff.py` in this shape:

```python
"""Console staging and follow for the Watchlists screen.

Extracted from the screen shell so the shell stays thin. This is a pure
move: the logic, messages, and stable selectors are unchanged from when it
lived on the screen.
"""

from __future__ import annotations

from typing import Any

from loguru import logger


logger = logger.bind(module="WatchlistsConsoleHandoff")


class WatchlistsConsoleHandoff:
    """Owns the Watchlists screen's Console staging and follow state."""

    def __init__(self, app_instance: Any) -> None:
        self.app_instance = app_instance
        self._latest_console_follow_item_id = None
        self._latest_console_follow_item_cache = None
        self._latest_console_follow_loaded = False
        self._latest_console_follow_error_logged = False

    # Each method below is moved verbatim from the screen. Keep the bodies
    # byte-identical apart from `self.app_instance` resolution.
```

Move the bodies **verbatim** — do not rewrite logic, rename variables, reorder branches, or "improve" anything while moving. A behaviour change made during a move is indistinguishable from a bug in review, and the tests guarding this are the only thing standing between you and a silently broken Console handoff.

The screen keeps its `@on(...)` decorators — Textual needs them on the widget receiving the event — and each body becomes a one-line delegation. In the screen's `__init__`, replace the four `_latest_console_follow_*` assignments with:

```python
        self._console_handoff = WatchlistsConsoleHandoff(app_instance)
```

and each handler becomes, for example:

```python
    @on(Button.Pressed, "#watchlists-follow-in-console")
    def _on_follow_in_console(self, event: Button.Pressed) -> None:
        event.stop()
        self._console_handoff.follow_in_console()
```

- [ ] **Step 4: Verify no behaviour changed**

Run: `source .venv/bin/activate && pytest Tests/UI/test_watchlists_destination_shell.py -v`
Expected: identical results to Step 1. If a test now fails, the move was not faithful — fix the move, not the test.

Then: `source .venv/bin/activate && pytest Tests/Watchlists -v`
Expected: no new failures.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/watchlists_console_handoff.py tldw_chatbook/UI/Screens/watchlists_collections_screen.py
git commit -m "refactor(watchlists): extract Console handoff from the screen shell"
```

---

### Task 5: Rewrite the shell over the workbench

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Test: `Tests/UI/test_watchlists_destination_shell.py`

**Interfaces:**
- Consumes: `RegionLayout`, `Region` (Task 1); `load_region_layout`, `save_region_layout` (Task 2); `WatchlistsWorkbench`, `RegionToggled` (Task 3); `WatchlistsConsoleHandoff` (Task 4).
- Produces: the rebuilt screen. Phase C mounts real panes into the workbench's regions.

**What changes:** `compose_content` yields the `DestinationModeStrip`, a tab strip, and the `WatchlistsWorkbench` instead of the three placeholder columns. Collapse bindings are added. The section-navigator rail is replaced by centre tabs.

**What must not change:** the class name `WatchlistsCollectionsScreen`, the route `watchlists_collections`, the existing recovery-state behaviour, and every stable widget selector the existing tests assert on. Console handoffs and route tests must keep passing.

- [ ] **Step 1: Write the failing test**

Append to `Tests/UI/test_watchlists_destination_shell.py`, following the fixture style already in that file:

```python
@pytest.mark.asyncio
async def test_workbench_replaces_the_placeholder_columns(watchlists_app):
    async with watchlists_app.run_test() as pilot:
        await pilot.pause()
        screen = watchlists_app.screen
        assert screen.query("#wl-workbench"), "the workbench container should be mounted"
        # The literal placeholder labels from the old shell are gone.
        text = " ".join(str(node.renderable) for node in screen.query(Static))
        assert "Column 1: Watchlist List" not in text
        assert "Column 3: Status Inspector" not in text


@pytest.mark.asyncio
async def test_z_collapses_the_focused_region_and_persists(watchlists_app, monkeypatch):
    saved = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.save_region_layout",
        lambda layout: saved.append(layout),
    )
    async with watchlists_app.run_test() as pilot:
        await pilot.pause()
        screen = watchlists_app.screen
        screen.focused_region = Region.CONTENT
        await pilot.press("z")
        await pilot.pause()
        assert screen.region_layout.is_collapsed(Region.CONTENT)
        assert saved, "collapse state must persist across visits"


@pytest.mark.asyncio
async def test_shift_z_solos_and_restores(watchlists_app):
    async with watchlists_app.run_test() as pilot:
        await pilot.pause()
        screen = watchlists_app.screen
        screen.focused_region = Region.ITEMS
        await pilot.press("Z")
        await pilot.pause()
        assert screen.region_layout.is_collapsed(Region.FEEDS)
        assert screen.region_layout.is_collapsed(Region.CONTENT)
        await pilot.press("Z")
        await pilot.pause()
        assert screen.region_layout.collapsed == frozenset()


@pytest.mark.asyncio
async def test_bracket_keys_toggle_the_rails(watchlists_app):
    async with watchlists_app.run_test() as pilot:
        await pilot.pause()
        screen = watchlists_app.screen
        await pilot.press("[")
        await pilot.pause()
        assert screen.region_layout.is_collapsed(Region.LEFT_RAIL)
        await pilot.press("]")
        await pilot.pause()
        assert screen.region_layout.is_collapsed(Region.RIGHT_RAIL)


@pytest.mark.asyncio
async def test_clicking_a_collapsed_header_expands_it(watchlists_app):
    async with watchlists_app.run_test() as pilot:
        await pilot.pause()
        screen = watchlists_app.screen
        screen.region_layout = RegionLayout(collapsed=frozenset({Region.FEEDS}))
        await pilot.pause()
        await pilot.click("#wl-header-feeds")
        await pilot.pause()
        assert not screen.region_layout.is_collapsed(Region.FEEDS)


@pytest.mark.asyncio
async def test_route_and_class_name_are_unchanged(watchlists_app):
    async with watchlists_app.run_test():
        screen = watchlists_app.screen
        assert type(screen).__name__ == "WatchlistsCollectionsScreen"
        # BaseAppScreen stores the route as `screen_name` (base_app_screen.py:23),
        # not `route_name` — the screen passes "watchlists_collections" to super().
        assert screen.screen_name == "watchlists_collections"
```

These tests need `from textual.widgets import Static`, plus `Region` and `RegionLayout` from `tldw_chatbook.UI.Watchlists_Modules.region_layout`. Add them to the file's existing import block rather than a second one.

If the existing file has no `watchlists_app` fixture, reuse whatever fixture its current tests use and adapt these accordingly — do not invent a second app harness alongside an existing one.

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/UI/test_watchlists_destination_shell.py -v`
Expected: the six new tests FAIL (no `#wl-workbench`, no `region_layout` attribute); the pre-existing tests still pass.

- [ ] **Step 3: Add layout state and bindings to the screen**

In `WatchlistsCollectionsScreen`, add the imports and reactive state:

```python
from ..Watchlists_Modules.region_layout import CENTRE_REGIONS, Region, RegionLayout
from ..Watchlists_Modules.region_layout_store import load_region_layout, save_region_layout
from ..Watchlists_Modules.watchlists_workbench import RegionToggled, WatchlistsWorkbench
```

```python
    region_layout = reactive(RegionLayout())
    focused_region = reactive(Region.FEEDS)
```

Extend `BINDINGS` with the collapse keys, keeping the existing entries:

```python
        ("z", "toggle_region", "Collapse"),
        ("Z", "solo_region", "Solo"),
        ("left_square_bracket", "toggle_left_rail", "Left rail"),
        ("right_square_bracket", "toggle_right_rail", "Right rail"),
```

Load persisted state in `on_mount`, before the first render:

```python
        self.region_layout = load_region_layout()
```

- [ ] **Step 4: Implement the actions**

```python
    def _apply_layout(self, layout: RegionLayout) -> None:
        """Set the layout, push it to the workbench, and persist it."""
        self.region_layout = layout
        try:
            # The workbench's reactive is `region_layout`, NOT `layout` —
            # `Widget.layout` is an existing read-only Textual property the
            # compositor calls .arrange() on every render, so shadowing it
            # breaks rendering outright. Verified empirically in Task 3.
            self.query_one(WatchlistsWorkbench).region_layout = layout
        except Exception:
            logger.debug("Workbench not mounted yet; layout will apply on compose.")
        save_region_layout(layout)

    def action_toggle_region(self) -> None:
        """Collapse or expand whichever region currently has focus."""
        self._apply_layout(self.region_layout.toggle(self.focused_region))

    def action_solo_region(self) -> None:
        """Isolate the focused centre pane; press again to restore."""
        if self.focused_region not in CENTRE_REGIONS:
            self.notify("Solo applies to the Feeds, Items, or Content panes.")
            return
        self._apply_layout(self.region_layout.solo(self.focused_region))

    def action_toggle_left_rail(self) -> None:
        self._apply_layout(self.region_layout.toggle(Region.LEFT_RAIL))

    def action_toggle_right_rail(self) -> None:
        self._apply_layout(self.region_layout.toggle(Region.RIGHT_RAIL))

    @on(RegionToggled)
    def _on_region_toggled(self, event: RegionToggled) -> None:
        event.stop()
        self._apply_layout(self.region_layout.toggle(event.region))
```

**Track focus, or `z` is a lie.** `focused_region` must follow real focus, otherwise every `z` collapses whichever region the reactive happened to default to. Walk up from the focused widget to find its owning region:

```python
    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        """Keep `focused_region` in step with whatever actually holds focus."""
        node = event.widget
        while node is not None:
            node_id = getattr(node, "id", None) or ""
            for prefix in ("wl-region-", "wl-header-"):
                if node_id.startswith(prefix):
                    try:
                        self.focused_region = Region(node_id[len(prefix):])
                    except ValueError:
                        pass
                    return
            node = node.parent
```

Add `from textual import events` to the imports. Both prefixes are handled so that focusing a *collapsed* region's header targets that region — otherwise `z` on a collapsed header would expand some other region.

Add this test alongside the others in Step 1:

```python
@pytest.mark.asyncio
async def test_focus_drives_which_region_z_collapses(watchlists_app):
    async with watchlists_app.run_test() as pilot:
        await pilot.pause()
        screen = watchlists_app.screen
        screen.query_one("#wl-region-items").focus()
        await pilot.pause()
        assert screen.focused_region == Region.ITEMS
        await pilot.press("z")
        await pilot.pause()
        assert screen.region_layout.is_collapsed(Region.ITEMS)
        assert not screen.region_layout.is_collapsed(Region.FEEDS)
```

If `#wl-region-items` is not focusable as composed, give the region bodies `can_focus = True` in `WatchlistsWorkbench._compose_region` rather than weakening this test — a region the keyboard cannot reach cannot be collapsed by keyboard either.

- [ ] **Step 5: Replace the three placeholder columns in `compose_content`**

Replace the three-column body (the `Column 1: Watchlist List` / `Column 2: …` / `Column 3: Status Inspector` containers) with the workbench, keeping the destination header, the backend selector, and the recovery-state rendering exactly as they are:

```python
            yield WatchlistsWorkbench(self.region_layout, id="wl-workbench")
```

Delete the now-unused `WatchlistsNavigator` import and its section-rail composition — section switching becomes centre tabs in Phase C/E. Leave the `_SECTION_DETAIL_TITLE` mapping and `active_section` reactive in place; Phase E's tabs consume them.

- [ ] **Step 6: Run the tests**

Run: `source .venv/bin/activate && pytest Tests/UI/test_watchlists_destination_shell.py -v`
Expected: PASS, including every pre-existing test.

Then, one at a time: `pytest Tests/Watchlists -v`, then `pytest Tests/UI -v`.
Expected: no new failures. `Tests/Watchlists/test_watchlists_navigator.py::test_navigator_has_all_section_buttons` fails on this branch already — it asserts on the section rail this task removes, so it becomes legitimately obsolete. Delete that test file if the navigator is gone, and say so in your report rather than leaving a test asserting on deleted UI.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/UI/Screens/watchlists_collections_screen.py Tests/UI/test_watchlists_destination_shell.py
git commit -m "feat(watchlists): rebuild the screen shell over the collapsible workbench"
```

---

## Phase B Completion Checklist

- [ ] All five tasks committed.
- [ ] `pytest Tests/Watchlists` and `pytest Tests/UI` pass with no new failures.
- [ ] No data-layer file modified: `git diff --stat origin/dev -- tldw_chatbook/DB tldw_chatbook/Subscriptions` is empty.
- [ ] The literal strings `Column 1: Watchlist List` and `Column 3: Status Inspector` no longer appear anywhere in the repo.
- [ ] Collapse state survives a screen revisit and an app restart.
- [ ] Every collapsed region can be re-expanded by keyboard alone.
- [ ] Phase C plan written against merged Phase B code.
