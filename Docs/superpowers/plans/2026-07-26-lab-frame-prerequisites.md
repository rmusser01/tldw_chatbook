# Lab Frame Prerequisites (PR0 + PR1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the live bug that hides Lab's active mode label, and promote the rail widgets to a shared, Chat-free base — the two prerequisites the Lab frame builds on.

**Architecture:** PR0 adds an app-tier CSS module for Lab so the bundle's global `.is-active` border rule stops winning against `LabModeStrip.DEFAULT_CSS`. PR1 extracts a pure `DestinationRailHandle` / `DestinationRailSectionHeader` into `Widgets/destination_rail.py` and turns `ConsoleRailHandle` into a thin subclass, so all six existing consumers and the CSS bundle are untouched.

**Tech Stack:** Python ≥3.11, Textual 8.2.7, pytest, the generated TCSS bundle (`build_css.py`).

**Spec:** `Docs/superpowers/specs/2026-07-26-lab-destination-console-frame-design.md`

## Global Constraints

- Run pytest via the venv only: `.venv/bin/python -m pytest`. System python lacks the deps.
- Never hand-edit `tldw_chatbook/css/tldw_cli_modular.tcss`. It is generated; regenerate with `build_css.py`.
- Colours, borders, and status treatments belong in the **app-tier bundle**, never in widget `DEFAULT_CSS` — the bundle outranks `DEFAULT_CSS` regardless of specificity. This is the defect PR0 fixes; do not reintroduce it.
- `CSS_MODULES` in `build_css.py` is an **ordered** list. `features/` entries land after `components/` and before `utilities/`.
- Keep the existing `.console-rail-*` CSS class names. Renaming them is a deliberate deferral; PR1 must produce a **zero-diff** bundle.
- Textual 8.2.7 has no `App.export_text()`. Assert rendered styling via `widget.styles`, not by scraping text.
- Every test must be **mutation-checked**: revert the fix, confirm the test goes red, restore.
- Run pytest from the worktree with `PYTHONPATH=$(pwd)` so imports cannot silently resolve to the main checkout's editable install.

## Known baseline failures — do NOT try to fix these

Measured on this worktree at `origin/dev` + docs only, before any code change:

```
Tests/UI/test_lab_mode_strip.py, test_console_rail_sections.py, test_console_rail_title.py,
test_console_persistent_rails.py, test_console_agent_rail.py,
test_console_workspace_context_rail.py, test_home_triage_rail.py
    -> 1 failed, 168 passed
```

The one failure is **pre-existing and unrelated to this work**:

`test_console_persistent_rails.py::test_generated_console_stylesheet_includes_rail_rules`
asserts *globally* over the entire stylesheet that `"border: thick $ds-action-focus;"` appears
nowhere (line 278). Unrelated RAG-settings work (`e619c4d81`, `ffc5959cb`) later added exactly that
declaration to `.settings-rag-profile-modal` and `.settings-library-rag-starter-panel`
(`components/_agentic_terminal.tcss:4029` and `:4066`). It is an over-broad Console rail test
tripped by Settings, not a Console regression.

**Your job is to keep this at exactly 1 failure.** If the count rises, you broke something. Do not
"fix" this test, do not weaken its assertion, and do not remove the Settings rules — that is a
separate concern with its own owner. Note that PR0 also writes to the bundle; its rules use
`border: none` and must not trip this assertion.

`Tests/UI/test_library_shell.py` additionally carries **3** recorded baseline failures (measured
directly on this worktree, `-p no:randomly`, reproduced twice for determinism — not 4 as previously
recorded here):

```
Tests/UI/test_library_shell.py::test_library_shell_search_history_prefers_app_config_over_cli_config
Tests/UI/test_library_shell.py::test_library_shell_rail_preferences_prefers_app_config_over_cli_config
Tests/UI/test_library_shell.py::test_library_shell_ingest_nav_context_deeplink_reentry_resets_stale_form

3 failed, 254 passed
```

Confirm the count is unchanged rather than expecting zero.

**Caveat — this file's count is not a reliable gate.** Independent re-runs found the three above
fail deterministically, but 3 of 4 runs showed a *fourth* failure that rotated between
`test_library_shell_export_registry_failure_warns_it_wont_appear_in_artifacts` and
`test_library_shell_ingest_canvas_different_canvas_isolation` — both self-documented in the test
file as order/global-state and CPU-contention flakes, and both sensitive to machine load. Treat the
three named tests as the floor and investigate only *new* failure names, never a raw count.

---

## Sequencing correction — CONFIRMED, folded into the spec

The spec sequences PR2 as "frame + all three screens inherit", with the rail lifts following in PR3
(Models) and PR4 (Speech). **That intermediate state is not shippable.** After PR2 the frame would
render an empty left rail (spec: "first run: left rail open") while each legacy sidebar is still
alive inside its body — two navigation columns side by side, worse than today.

**Confirmed and adopted** — the spec's Sequencing section now carries this as the plan of record.
Each screen's sidebar lift is folded into that screen's own adoption PR: a screen adopts the frame
and fills its rail in one change, or it does not adopt yet.

| PR | Contents | Shippable? |
|---|---|---|
| PR0 | CSS fix (this plan) | yes — pure improvement |
| PR1 | rail widget promotion (this plan) | yes — pure refactor, no visual change |
| PR2 | frame + **Models adopts it and lifts its sidebar** | yes — one screen fully good |
| PR3 | Speech adopts + lifts + capability chip | yes |
| PR4 | Evals adopts (empty rail + honest empty state) | yes |

This plan covers PR0 and PR1 only. PR2+ is deliberately not planned here: the restructure above is
a spec change that needs sign-off, and PR2's tasks should be written against PR1's realised API
rather than an anticipated one.

---

## File Structure

| File | Responsibility |
|---|---|
| `tldw_chatbook/css/features/_lab.tcss` | **new** — app-tier Lab styling; PR0 seeds it with the mode-chip rules |
| `tldw_chatbook/css/build_css.py` | modify — register the new module in `CSS_MODULES` |
| `tldw_chatbook/css/tldw_cli_modular.tcss` | regenerated, committed |
| `tldw_chatbook/Widgets/destination_rail.py` | **new** — pure, destination-agnostic rail handle + section header |
| `tldw_chatbook/Widgets/Console/console_rail_handle.py` | modify — `ConsoleRailHandle` becomes a subclass keeping Console's vocabulary |
| `tldw_chatbook/Widgets/Console/console_rail_section.py` | modify — re-export the shared section header |
| `Tests/UI/test_lab_mode_strip.py` | modify — add the bundle-loaded chip-border regression |
| `Tests/UI/test_destination_rail.py` | **new** — base widget behaviour + Console subclass parity |

---

# PR0 — The invisible active-mode label

### Task 1: App-tier Lab mode-chip rules

**Files:**
- Create: `tldw_chatbook/css/features/_lab.tcss`
- Modify: `tldw_chatbook/css/build_css.py` (the `CSS_MODULES` list, after `features/_watchlists.tcss`)
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss` (regenerated, not hand-edited)
- Test: `Tests/UI/test_lab_mode_strip.py`

**Interfaces:**
- Consumes: `LabModeStrip` from `tldw_chatbook.UI.Screens.lab_mode_strip` (unchanged).
- Produces: `features/_lab.tcss` as the app-tier home for all Lab styling. PR2 extends this same file; it does not create another.

**Background.** `tldw_cli_modular.tcss:5356` declares a global, unscoped `.is-active { border: round $ds-action-focus; }`. `LabModeStrip.DEFAULT_CSS` tries to override it (`lab_mode_strip.py:61-73`) but loses, because app-tier CSS beats `DEFAULT_CSS` regardless of specificity. The active chip gets a round border, becomes a 3-row box inside a height-1 strip, and only its top border row renders — the label is invisible on all three Lab screens. MCP has the same strip and works because its `border: none` is app-tier at `tldw_cli_modular.tcss:5855`.

The existing `_StripHarness` in this test file mounts the strip **without** the bundle, so it cannot reproduce this. The new test must set `CSS_PATH`.

- [ ] **Step 1: Write the failing test**

Append to `Tests/UI/test_lab_mode_strip.py`:

```python
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUNDLED_STYLESHEET = _REPO_ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"


class _BundledStripHarness(App[None]):
    """Mount the strip with the production stylesheet.

    The bundle is required: the bug under test lives in the bundle's global
    `.is-active` rule, which beats LabModeStrip.DEFAULT_CSS. A harness
    without CSS_PATH passes vacuously.
    """

    CSS_PATH = str(_BUNDLED_STYLESHEET)

    def __init__(self, active_route: str) -> None:
        super().__init__()
        self._active_route = active_route

    def compose(self):
        yield LabModeStrip(active_route=self._active_route, id="lab-mode-strip")


def _has_border(widget) -> bool:
    """True when any edge declares a border style."""
    border = widget.styles.border
    return any(
        edge[0] for edge in (border.top, border.right, border.bottom, border.left)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("route", "active_chip"), [
    ("llm", "lab-mode-models"),
    ("stts", "lab-mode-speech"),
    ("evals", "lab-mode-evals"),
])
async def test_active_mode_chip_has_no_border_so_its_label_renders(route, active_chip):
    """The active chip must not gain the bundle's global `.is-active` border.

    The strip is one row tall. A bordered chip becomes a three-row box, so
    only its top border renders and the mode label disappears entirely --
    leaving no way to see which Lab mode is active.
    """
    app = _BundledStripHarness(route)
    async with app.run_test(size=(80, 6)) as pilot:
        await pilot.pause()
        chip = app.query_one(f"#{active_chip}")

        assert "is-active" in chip.classes
        assert not _has_border(chip), (
            f"{active_chip} has a border; its label is clipped by the 1-row strip"
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/UI/test_lab_mode_strip.py -k no_border -v`

Expected: 3 FAILED, each with `AssertionError: lab-mode-... has a border; its label is clipped by the 1-row strip`.

If it passes, the harness is not loading the bundle — check `CSS_PATH` resolves to a real file.

- [ ] **Step 3: Create the Lab CSS module**

Create `tldw_chatbook/css/features/_lab.tcss`:

```css
/* Lab destination (Models | Speech | Evals).
 *
 * These rules MUST live app-tier, not in LabModeStrip.DEFAULT_CSS: the
 * bundle's global `.is-active` rule (border: round) outranks DEFAULT_CSS
 * regardless of specificity, so a widget-tier override silently loses and
 * the active chip's label is clipped by the one-row strip. Mirrors
 * `.personas-mode-chip.is-active` and `#mcp-mode-strip Button.mcp-mode-chip`.
 */

#lab-mode-strip Button.lab-mode-chip {
    width: auto;
    min-width: 0;
    height: 1;
    padding: 0 1;
    border: none;
}

#lab-mode-strip .lab-mode-chip.is-active {
    border: none;
    background: $ds-focus-bg;
    color: $ds-focus-fg;
    text-style: bold underline;
}

#lab-mode-strip .lab-mode-chip.is-active:focus,
#lab-mode-strip .lab-mode-chip.is-active:hover:focus {
    outline: none;
    border: none;
    background: $ds-focus-bg;
    color: $ds-focus-fg;
    text-style: bold underline;
}
```

- [ ] **Step 4: Register the module**

In `tldw_chatbook/css/build_css.py`, add to `CSS_MODULES` immediately after `"features/_watchlists.tcss"`:

```python
    "features/_lab.tcss",
```

Order matters: it must sit inside the `features/` block, after `components/` and before the `utilities/` entries.

- [ ] **Step 5: Regenerate the bundle**

Run: `.venv/bin/python tldw_chatbook/css/build_css.py`

Then confirm the rules landed and nothing else moved:

```bash
git diff --stat tldw_chatbook/css/tldw_cli_modular.tcss
grep -n "lab-mode-chip" tldw_chatbook/css/tldw_cli_modular.tcss
```

Expected: the bundle diff is purely additive (the new block), and `grep` shows the new rules.

- [ ] **Step 6: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest Tests/UI/test_lab_mode_strip.py -v`

Expected: the 3 new tests PASS and every pre-existing test in the file still passes.

- [ ] **Step 7: Mutation-check**

Comment out the two `border: none` declarations in `features/_lab.tcss`, regenerate, and re-run. Expected: the 3 tests FAIL again. Restore, regenerate, confirm green. This proves the test is bound to the fix rather than to the harness.

- [ ] **Step 8: Confirm on the running app**

Launch and look at the strip:

```bash
tmux -L labfix kill-server 2>/dev/null
tmux -L labfix new-session -d -x 200 -y 50 \
  '.venv/bin/python -m tldw_chatbook.app'
sleep 18
```

Click the `Lab` tab (find its column in row 2 of `tmux -L labfix capture-pane -p`, then send
`$'\x1b[<0;COL;2M'` and `$'\x1b[<0;COL;2m'`). Capture row 9.

Expected: `Modes:  Models   Speech   Evals` with **`Models` visible and underlined**.
Before the fix this row reads `Modes:  ╭──────────╮  Speech    Evals`.

Then `tmux -L labfix kill-server`.

- [ ] **Step 9: Commit**

```bash
git add tldw_chatbook/css/features/_lab.tcss \
        tldw_chatbook/css/build_css.py \
        tldw_chatbook/css/tldw_cli_modular.tcss \
        Tests/UI/test_lab_mode_strip.py
git commit -m "fix(lab): render the active mode chip's label

The bundle's global .is-active rule (border: round) outranks
LabModeStrip.DEFAULT_CSS regardless of specificity, so the active chip
became a three-row box inside a one-row strip and only its top border
rendered -- leaving no way to see which Lab mode is active.

Moves the chip rules app-tier into a new features/_lab.tcss, mirroring
.personas-mode-chip.is-active and #mcp-mode-strip Button.mcp-mode-chip,
both of which already do this and both of which render correctly."
```

---

### Task 2: Resolve the `.library-collection-row` suspect

**Files:**
- Inspect: `tldw_chatbook/Widgets/Library/library_collections_panel.py:153`
- Possibly modify: `tldw_chatbook/css/features/_lab.tcss` is **not** the right home — if a fix is needed it belongs in the Library's own styling.

**Interfaces:**
- Consumes: nothing from Task 1 beyond the confirmed root cause.
- Produces: either a fix, or a documented decision that the border is intentional.

**Background.** The `.is-active` sweep cleared six consumers. `.library-collection-row` is the one open case: `library_collections_panel.py:153` assigns the class and `library_screen.py:13326` handles its press, but **no CSS rule targets it anywhere**. A selected collection row therefore inherits the same global `border: round` that broke Lab, while its unselected siblings have none. Whether that clips a label (as in Lab) or merely reads as a highlight depends on the row's height, which is unconstrained. Do not fix this blind.

- [ ] **Step 1: Reproduce**

Launch the app (same tmux recipe as Task 1 Step 8), navigate to `3 Library`, and open the
Collections view. Capture the pane and inspect the selected row.

- [ ] **Step 2: Classify the finding**

- **If the label is clipped or the row jumps height when selected** — it is the same bug. Add an app-tier rule alongside the Library's existing collection styling (near `#library-collection-actions` in the bundle source module), with a test mirroring Task 1's `_has_border` assertion.
- **If it renders as a deliberate-looking highlight box** — it is cosmetically fine. Record that in the spec's sweep table as resolved-intentional and stop. Do not "tidy" it; changing it would alter Library's appearance for no defect.

- [ ] **Step 3: Commit the outcome**

Either the fix plus its test, or a spec edit updating the sweep table's verdict from `SUSPECT` to the resolved state, with one sentence of evidence.

```bash
git add -A && git commit -m "fix(library): <or> docs: resolve the .library-collection-row is-active suspect"
```

---

# PR1 — Promote the rail widgets

### Task 3: Pure `DestinationRailHandle`

**Files:**
- Create: `tldw_chatbook/Widgets/destination_rail.py`
- Test: `Tests/UI/test_destination_rail.py`

**Interfaces:**
- Consumes: nothing from PR0.
- Produces:
  ```python
  class DestinationRailHandle(Vertical):
      def __init__(self, *, label: str, badge: str = "", button_id: str,
                   badge_id: str, side: str, open_tooltip: str | None = None,
                   **kwargs) -> None
      def sync_state(self, label: str, badge: str) -> None
      def _display_label(self) -> str      # override point
      def _display_badge(self) -> str      # override point
  ```
  Task 4 subclasses this. The Lab frame (PR2) constructs it directly.

**Background.** `ConsoleRailHandle` already has six consumers — `chat_screen`, `home_screen`, `library_screen`, `personas_screen`, `Widgets/Home/home_rail`, `Widgets/Library/library_rail` — while living in a Console-private namespace and importing `CONSOLE_RAIL_INSPECTOR_LABEL` from `tldw_chatbook.Chat.console_rail_state`. The base extracted here carries no Chat import and no Console vocabulary; Console's specifics move to the subclass in Task 4.

**Expect transitional duplication.** This task adds a `compose()` body closely matching the one still
living in `console_rail_handle.py`; Task 4 deletes that original and reduces `ConsoleRailHandle` to a
subclass. The overlap exists only between Tasks 3 and 4 and is inherent to a two-step extract. Do not
try to resolve it inside Task 3 by editing the Console file — that is Task 4's diff.

The `.console-rail-handle*` class names are kept deliberately, so the CSS bundle sees no diff. The TCSS contains no type selectors for these widgets — only class selectors — so renaming the Python types is invisible to CSS.

- [ ] **Step 1: Write the failing test**

Create `Tests/UI/test_destination_rail.py`:

```python
"""Shared destination rail widgets: the Chat-free base behind ConsoleRailHandle."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.destination_rail import DestinationRailHandle


class _HandleHarness(App[None]):
    def __init__(self, handle: DestinationRailHandle) -> None:
        super().__init__()
        self._handle = handle

    def compose(self) -> ComposeResult:
        yield self._handle


@pytest.mark.asyncio
async def test_base_handle_renders_label_and_badge_verbatim():
    """The base applies no vocabulary of its own -- Console's lives in its subclass."""
    handle = DestinationRailHandle(
        label="Catalog",
        badge="3 servers",
        button_id="lab-rail-open",
        badge_id="lab-rail-badge",
        side="left",
    )
    app = _HandleHarness(handle)
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert str(app.query_one("#lab-rail-open", Button).label) == "Catalog"
        assert str(app.query_one("#lab-rail-badge", Static).renderable) == "3 servers"


@pytest.mark.asyncio
async def test_base_handle_default_tooltip_names_the_rail():
    handle = DestinationRailHandle(
        label="Catalog",
        button_id="lab-rail-open",
        badge_id="lab-rail-badge",
        side="left",
    )
    app = _HandleHarness(handle)
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert app.query_one("#lab-rail-open", Button).tooltip == "Open Catalog rail"


@pytest.mark.asyncio
async def test_base_handle_accepts_an_explicit_tooltip():
    handle = DestinationRailHandle(
        label="Whatever",
        button_id="lab-rail-open",
        badge_id="lab-rail-badge",
        side="right",
        open_tooltip="Open Inspector rail",
    )
    app = _HandleHarness(handle)
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert app.query_one("#lab-rail-open", Button).tooltip == "Open Inspector rail"


@pytest.mark.asyncio
async def test_base_handle_keeps_the_existing_css_class_names():
    """Class names are deliberately unchanged so the CSS bundle sees no diff."""
    handle = DestinationRailHandle(
        label="Catalog",
        button_id="lab-rail-open",
        badge_id="lab-rail-badge",
        side="left",
    )
    app = _HandleHarness(handle)
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert "console-rail-handle" in handle.classes
        assert "console-rail-handle-left" in handle.classes


@pytest.mark.asyncio
async def test_base_handle_omits_the_badge_when_empty():
    handle = DestinationRailHandle(
        label="Catalog",
        button_id="lab-rail-open",
        badge_id="lab-rail-badge",
        side="left",
    )
    app = _HandleHarness(handle)
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert not app.query("#lab-rail-badge")


def test_shared_glyphs_match_the_console_originals():
    """Guard the deliberate duplication of the glyph literals.

    ``destination_rail`` redeclares these rather than importing from
    ``Chat.console_glyphs``, so the shared widget stays free of the Chat
    layer. That duplication would otherwise drift silently if either side
    changed.
    """
    from tldw_chatbook.Chat import console_glyphs
    from tldw_chatbook.Widgets.destination_rail import (
        GLYPH_COLLAPSED,
        GLYPH_EXPANDED,
    )

    assert GLYPH_EXPANDED == console_glyphs.GLYPH_EXPANDED
    assert GLYPH_COLLAPSED == console_glyphs.GLYPH_COLLAPSED
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/UI/test_destination_rail.py -v`

Expected: collection error — `ModuleNotFoundError: No module named 'tldw_chatbook.Widgets.destination_rail'`.

- [ ] **Step 3: Write the implementation**

Create `tldw_chatbook/Widgets/destination_rail.py`:

```python
"""Shared, destination-agnostic rail chrome.

Extracted from ``Widgets/Console/console_rail_handle.py`` and
``console_rail_section.py``, which had six consumers across Console, Home,
Library, and Personas while living in a Console-private namespace and
importing from the Chat layer. This module carries no Chat import and no
Console vocabulary; ``ConsoleRailHandle`` subclasses it and keeps its own.

The ``.console-rail-*`` CSS class names are retained deliberately so the
generated bundle sees no diff. The TCSS references these widgets only by
class, never by type, so the new type names are invisible to CSS. Renaming
the classes is a deferred cleanup.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static


#: Toggle-button id prefix. Unchanged from the Console original so existing
#: selectors and tests keep resolving.
RAIL_SECTION_TOGGLE_PREFIX = "console-rail-section-toggle-"

#: Default collapse/expand affordance glyphs. Literals rather than an import
#: from ``Chat.console_glyphs`` so this module stays free of the Chat layer;
#: the values match that module exactly.
GLYPH_EXPANDED = "▾"
GLYPH_COLLAPSED = "▸"


class DestinationRailHandle(Vertical):
    """Focusable compact handle for opening a collapsed destination rail."""

    def __init__(
        self,
        *,
        label: str,
        badge: str = "",
        button_id: str,
        badge_id: str,
        side: str,
        open_tooltip: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a rail handle.

        Args:
            label: Rail name shown on the handle button.
            badge: Optional secondary line under the button.
            button_id: DOM id for the open button.
            badge_id: DOM id for the badge static.
            side: ``"left"`` or ``"right"``; drives height and CSS class.
            open_tooltip: Button tooltip. Defaults to ``"Open <label> rail"``.
            kwargs: Forwarded to ``Vertical``.
        """
        super().__init__(**kwargs)
        self.label = label
        self.badge = badge
        self.button_id = button_id
        self.badge_id = badge_id
        self.side = side
        self.open_tooltip = open_tooltip or f"Open {label} rail"
        self.add_class("console-rail-handle")
        self.add_class(f"console-rail-handle-{side}")

    def compose(self) -> ComposeResult:
        """Render the open button and, when set, the badge."""
        button_width = 11
        button_height: int | str = 3 if self.side == "right" else "100%"
        button = Button(self._display_label(), id=self.button_id, compact=True)
        button.add_class("console-rail-handle-button")
        button.add_class(f"console-rail-handle-button-{self.side}")
        button.styles.width = button_width
        button.styles.min_width = 0
        button.styles.max_width = button_width
        button.styles.height = button_height
        button.styles.min_height = button_height
        button.styles.max_height = button_height
        button.tooltip = self.open_tooltip
        yield button
        if self.badge:
            badge = Static(self._display_badge(), id=self.badge_id, markup=False)
            badge.add_class("console-rail-handle-badge")
            badge.tooltip = self.badge
            yield badge

    def sync_state(self, label: str, badge: str) -> None:
        """Refresh label and badge without recomposing the whole screen."""
        if self.label == label and self.badge == badge:
            return
        self.label = label
        self.badge = badge
        self.call_later(self.recompose)

    def _display_label(self) -> str:
        """Visible button text. Override to abbreviate."""
        return self.label

    def _display_badge(self) -> str:
        """Visible badge text. Override to abbreviate."""
        return self.badge


class DestinationRailSectionHeader(Horizontal):
    """One-line rail section header with a collapse/expand toggle.

    Attributes:
        title: User-facing section title.
        section_id: Stable section id used in child widget ids.
        open: Whether the associated section body is currently visible.
    """

    def __init__(
        self,
        title: str,
        *,
        section_id: str,
        open: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(classes="console-rail-section-header", **kwargs)
        self.title = title
        self.section_id = section_id
        self.open = open

    def compose(self) -> ComposeResult:
        """Render the section title and its collapse/expand toggle."""
        title = Static(
            self.title,
            id=f"console-rail-section-title-{self.section_id}",
            classes="console-rail-section-title",
            markup=False,
        )
        title.styles.width = "1fr"
        yield title
        toggle = Button(
            self._toggle_label(),
            id=f"{RAIL_SECTION_TOGGLE_PREFIX}{self.section_id}",
            classes="console-workspace-action console-rail-section-toggle",
            compact=True,
        )
        toggle.tooltip = self._toggle_tooltip()
        toggle.styles.width = 3
        toggle.styles.min_width = 3
        toggle.styles.max_width = 3
        yield toggle

    def sync_open(self, open: bool) -> None:
        """Refresh the toggle affordance after body visibility changes."""
        self.open = open
        toggle = self.query_one(
            f"#{RAIL_SECTION_TOGGLE_PREFIX}{self.section_id}",
            Button,
        )
        toggle.label = self._toggle_label()
        toggle.tooltip = self._toggle_tooltip()

    def _toggle_label(self) -> str:
        return GLYPH_EXPANDED if self.open else GLYPH_COLLAPSED

    def _toggle_tooltip(self) -> str:
        return f"Collapse {self.title}" if self.open else f"Expand {self.title}"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest Tests/UI/test_destination_rail.py -v`

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/destination_rail.py Tests/UI/test_destination_rail.py
git commit -m "refactor(widgets): add a Chat-free destination rail base

ConsoleRailHandle and ConsoleRailSectionHeader already had six consumers
across Console, Home, Library, and Personas while living in a
Console-private namespace, with the handle importing from the Chat layer.

Adds the pure base. Console's own vocabulary moves to a subclass next.
CSS class names are unchanged so the generated bundle sees no diff."
```

---

### Task 4: `ConsoleRailHandle` becomes a subclass

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_rail_handle.py` (replace the class body)
- Modify: `tldw_chatbook/Widgets/Console/console_rail_section.py` (re-export the shared header)
- Test: `Tests/UI/test_destination_rail.py` (extend)

**Interfaces:**
- Consumes: `DestinationRailHandle`, `DestinationRailSectionHeader`, `RAIL_SECTION_TOGGLE_PREFIX` from Task 3.
- Produces: `ConsoleRailHandle` and `ConsoleRailSectionHeader` keep their existing import paths and constructor signatures, so **no consumer changes**.

**Background.** Console abbreviates badges to fit the collapsed inspector (`"1 approval"` → `"1 appr"`, `"N approvals"` → `"N appr"`, `"artifact"` → `"art"`) and renames the inspector label via `CONSOLE_RAIL_INSPECTOR_LABEL`. Its tooltips are fixed strings, not derived from the label. All of that is Console vocabulary and stays in Console.

- [ ] **Step 1: Write the failing test**

Append to `Tests/UI/test_destination_rail.py`:

```python
from tldw_chatbook.Chat.console_rail_state import CONSOLE_RAIL_INSPECTOR_LABEL
from tldw_chatbook.Widgets.Console.console_rail_handle import ConsoleRailHandle


def _console_handle(**overrides) -> ConsoleRailHandle:
    kwargs = dict(
        label="Context",
        badge="",
        button_id="console-rail-open",
        badge_id="console-rail-badge",
        side="left",
    )
    kwargs.update(overrides)
    return ConsoleRailHandle(**kwargs)


@pytest.mark.asyncio
async def test_console_handle_is_a_destination_rail_handle():
    assert issubclass(ConsoleRailHandle, DestinationRailHandle)


@pytest.mark.asyncio
@pytest.mark.parametrize(("side", "expected"), [
    ("left", "Open Context rail"),
    ("right", "Open Inspector rail"),
])
async def test_console_handle_keeps_its_fixed_tooltips(side, expected):
    """Console's tooltips are fixed strings, not derived from the label."""
    app = _HandleHarness(_console_handle(side=side, label="Anything"))
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert app.query_one("#console-rail-open", Button).tooltip == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("badge", "expected"), [
    ("1 approval", "1 appr"),
    ("3 approvals", "3 appr"),
    ("artifact", "art"),
    ("something else", "something else"),
])
async def test_console_handle_abbreviates_right_side_badges(badge, expected):
    app = _HandleHarness(_console_handle(side="right", badge=badge))
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert str(app.query_one("#console-rail-badge", Static).renderable) == expected


@pytest.mark.asyncio
async def test_console_handle_renames_the_inspector_label_on_the_right():
    app = _HandleHarness(
        _console_handle(side="right", label=CONSOLE_RAIL_INSPECTOR_LABEL)
    )
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert str(app.query_one("#console-rail-open", Button).label) == "Inspector"


@pytest.mark.asyncio
async def test_console_handle_leaves_left_side_text_alone():
    app = _HandleHarness(_console_handle(side="left", badge="1 approval"))
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert str(app.query_one("#console-rail-badge", Static).renderable) == "1 approval"


def test_console_section_header_is_the_shared_widget():
    from tldw_chatbook.Widgets.Console.console_rail_section import (
        CONSOLE_RAIL_SECTION_TOGGLE_PREFIX,
        ConsoleRailSectionHeader,
    )

    assert ConsoleRailSectionHeader is DestinationRailSectionHeader
    assert CONSOLE_RAIL_SECTION_TOGGLE_PREFIX == RAIL_SECTION_TOGGLE_PREFIX
```

Add `DestinationRailSectionHeader` and `RAIL_SECTION_TOGGLE_PREFIX` to the module's existing import from `tldw_chatbook.Widgets.destination_rail`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/UI/test_destination_rail.py -k console -v`

Expected: FAIL — `ConsoleRailHandle` is still a `Vertical`, not a `DestinationRailHandle`, and `ConsoleRailSectionHeader` is not the shared class.

- [ ] **Step 3: Rewrite `console_rail_handle.py`**

Replace the whole file with:

```python
"""Console's rail handle: the shared base plus Console's own vocabulary."""

from __future__ import annotations

from typing import Any

from tldw_chatbook.Chat.console_rail_state import CONSOLE_RAIL_INSPECTOR_LABEL
from tldw_chatbook.Widgets.destination_rail import DestinationRailHandle


class ConsoleRailHandle(DestinationRailHandle):
    """Rail handle carrying Console's fixed tooltips and badge abbreviations.

    The abbreviations exist because the collapsed inspector is eleven
    columns wide. They are Console's vocabulary, not the shared base's.
    """

    def __init__(self, *, side: str, **kwargs: Any) -> None:
        super().__init__(
            side=side,
            open_tooltip=(
                "Open Context rail" if side == "left" else "Open Inspector rail"
            ),
            **kwargs,
        )

    def _display_label(self) -> str:
        """Compact visible label; full text stays in the tooltip."""
        if self.side != "right":
            return self.label
        return "Inspector" if self.label == CONSOLE_RAIL_INSPECTOR_LABEL else self.label

    def _display_badge(self) -> str:
        """Badge copy that fits the collapsed inspector affordance."""
        if self.side != "right":
            return self.badge
        if self.badge == "1 approval":
            return "1 appr"
        if self.badge.endswith(" approvals"):
            count = self.badge.split(maxsplit=1)[0]
            return f"{count} appr"
        if self.badge == "artifact":
            return "art"
        return self.badge
```

- [ ] **Step 4: Rewrite `console_rail_section.py`**

Replace the whole file with:

```python
"""Console's rail section header.

The implementation moved to ``Widgets/destination_rail.py`` when a fourth
destination needed it. These names are retained so existing imports and
selectors keep resolving.
"""

from __future__ import annotations

from tldw_chatbook.Widgets.destination_rail import (
    RAIL_SECTION_TOGGLE_PREFIX,
    DestinationRailSectionHeader,
)

CONSOLE_RAIL_SECTION_TOGGLE_PREFIX = RAIL_SECTION_TOGGLE_PREFIX
ConsoleRailSectionHeader = DestinationRailSectionHeader

__all__ = ["CONSOLE_RAIL_SECTION_TOGGLE_PREFIX", "ConsoleRailSectionHeader"]
```

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `.venv/bin/python -m pytest Tests/UI/test_destination_rail.py -v`

Expected: all PASS.

- [ ] **Step 6: Prove the six consumers are untouched**

This is the whole point of subclassing rather than migrating. Run every suite that exercises the rail widgets:

```bash
.venv/bin/python -m pytest \
  Tests/UI/test_console_rail_sections.py \
  Tests/UI/test_console_rail_title.py \
  Tests/UI/test_console_persistent_rails.py \
  Tests/UI/test_console_agent_rail.py \
  Tests/UI/test_console_workspace_context_rail.py \
  Tests/UI/test_home_triage_rail.py \
  Tests/UI/test_personas_workbench.py \
  Tests/UI/test_library_shell.py \
  -v
```

Expected: PASS, **with no edits to any of those files**. Note the three pre-existing
`test_library_shell` failures recorded as baseline (see the corrected count above) — confirm the
count is unchanged rather than assuming zero.

If any of these need changing, the subclass approach failed; stop and report rather than editing
the tests.

- [ ] **Step 7: Prove the CSS bundle is unchanged**

```bash
.venv/bin/python tldw_chatbook/css/build_css.py
git diff --stat tldw_chatbook/css/tldw_cli_modular.tcss
```

Expected: **no diff**. Task 3's docstring promises a zero-diff bundle; this verifies it.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_rail_handle.py \
        tldw_chatbook/Widgets/Console/console_rail_section.py \
        Tests/UI/test_destination_rail.py
git commit -m "refactor(console): rail handle subclasses the shared base

ConsoleRailHandle keeps its fixed tooltips, its inspector label rename,
and its badge abbreviations -- all Console vocabulary, sized to the
eleven-column collapsed inspector. The Chat-layer import stays here
rather than in shared code.

console_rail_section re-exports the shared header. All six existing
consumers and the generated CSS bundle are unchanged."
```

---

## Follow-ups surfaced by the final review (not done here, not yet filed)

None of these block this branch. They need a backlog-ID scan and a decision on whether they fold
into PR2, so they are recorded rather than filed.

**(a) `Tests/UI/test_console_persistent_rails.py:278` is the only unscoped assertion** in a function
whose every neighbour uses `_css_block(css, selector)`. Scoping it the same way is a one-line change
and would clear the known-red test that this branch had to work around. Worth doing before PR2
leans on that suite again.

**(b) `personas_screen.py:989-998` `_sync_personas_rail_tooltips()` is now obsolete.** It exists only
to overwrite Console's hard-coded `"Open Context rail"` after compose. The `open_tooltip` parameter
this branch added makes it redundant — switching Personas' two handles to `DestinationRailHandle`
deletes the method and yields correct tooltips for free. The extraction paid off and was not cashed
in.

**(c) The glyph constants are a real trade-off, not a settled call.** `destination_rail.py`
re-declares `"▾"` / `"▸"` rather than importing them, guarded by an equality test. The final
reviewer argued this installs a hidden *bidirectional* lockstep — neither module can change its
glyphs without a test in a third file going red — and preferred inverting it (define in
`destination_rail.py`, re-export from `console_glyphs.py`). The counter-argument is that inverting
makes the Chat layer import from `Widgets/`, which is the worse direction. Decide before a second
destination adopts the base.

**(d) Four consumers still import the section header from Console's namespace** — `home_rail.py`,
`library_rail.py`, `home_screen.py`, `library_screen.py` — via the `console_rail_section.py` shim.
Migrating them is a textual swap with no behaviour change, and would let the shim carry a real
deprecation horizon. Until then, "extracted out of Console's private namespace" is true only for Lab.

## Self-Review

**Spec coverage.** This plan covers the spec's PR0 (`.lab-mode-chip.is-active` app-tier rule, rendered-label test, `.library-collection-row` live look, bundle regeneration) and PR1 (pure base in `Widgets/destination_rail.py`, `ConsoleRailHandle` as subclass, zero consumer edits, zero CSS diff). The spec's PR2–PR4 are deliberately out of scope, with the sequencing defect and proposed restructure recorded above.

**Placeholders.** None. Every code step contains the actual file content; the one investigative task (Task 2) has explicit decision criteria for both outcomes rather than "handle appropriately".

**Type consistency.** `DestinationRailHandle.__init__` keyword names (`label`, `badge`, `button_id`, `badge_id`, `side`, `open_tooltip`) match every call site in Tasks 3 and 4. `_display_label` / `_display_badge` are defined in Task 3 and overridden with identical names in Task 4. `RAIL_SECTION_TOGGLE_PREFIX` is defined in Task 3 and aliased in Task 4. `_has_border` (Task 1) and `_HandleHarness` / `_console_handle` (Tasks 3–4) are each defined once in the file that uses them.

**Known residual.** Task 3 duplicates the glyph literals `▾` / `▸` rather than importing them from `Chat.console_glyphs`, to keep the shared widget free of the Chat layer. `test_shared_glyphs_match_the_console_originals` guards the duplication so it cannot drift silently. Consolidating the glyph constants into one shared module remains the cleaner long-term fix and is out of scope here.

**Expected test counts.** Task 1 adds 3 tests; Task 3 adds 6; Task 4 adds 10. Task 4 Step 6 changes no test files at all — that is its pass condition, not an oversight.
