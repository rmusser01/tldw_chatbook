"""Contracts for the shared adaptive pane shell (Roleplay frame B0).

The shell, grip and messages moved to ``Widgets/adaptive_pane_shell.py``
unchanged in behaviour; the Library keeps thin subclasses and same-object
aliases. These tests pin the shared contract with NEUTRAL probe classes and,
in later sections, the Library compatibility seam, the CSS re-key, the
promoted search input and the ``panes`` pattern family.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Button, Static

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.Utils import adaptive_reader_state as ars
from tldw_chatbook.Widgets import adaptive_pane_shell as shared

ROOT = Path(__file__).resolve().parents[2]

#: Neutral probe destination: no stylesheet names these classes.
PROBE_CLASSES = shared.AdaptivePaneClasses(
    shell="probe-pane-shell",
    nav="probe-pane-nav",
    items="probe-pane-items",
    work="probe-pane-work",
    grip="probe-pane-grip",
)


def _layout(*, nav_open: bool = True, items_open: bool = True) -> ars.AdaptivePaneLayout:
    return ars.AdaptivePaneLayout(
        library_open=nav_open,
        items_open=items_open,
        library_width=28 if nav_open else 0,
        items_width=40 if items_open else 0,
        reader_width=82,
        priority_pane=None,
    )


class _StyledShellApp(ConsolidatedCSSApp):
    """The app's real CSS (bundle + every split sheet).

    The probe classes carry no rules, so the shell's height is pinned inline,
    as a destination sheet would do; the grips get ``h-full``/``p-0``/
    ``border-none`` from the boot utilities.
    """

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def __init__(self, painted_names: dict[str, str] | None = None) -> None:
        super().__init__()
        self.painted_names = painted_names
        self.visibility: list[tuple[str, bool]] = []
        self.toggles: list[str] = []
        self.resizes = 0

    def compose(self) -> ComposeResult:
        shell = shared.AdaptivePaneShell(
            Static("Nav"),
            Static("Items"),
            Static("Work"),
            _layout(),
            id_prefix="probe",
            library_label="Characters",
            items_label="Lore books",
            destination=PROBE_CLASSES,
            painted_names=self.painted_names,
            id="probe-shell",
        )
        shell.styles.height = 30
        yield shell

    @on(shared.PaneVisibilityChanged)
    def _visibility(self, event: shared.PaneVisibilityChanged) -> None:
        self.visibility.append((event.pane, event.open))

    @on(shared.PaneToggleRequested)
    def _toggle(self, event: shared.PaneToggleRequested) -> None:
        self.toggles.append(event.pane)

    @on(shared.AdaptivePaneShellResized)
    def _resized(self, event: shared.AdaptivePaneShellResized) -> None:
        self.resizes += 1


def _painted_column(app: App, widget: Widget) -> str:
    """The first painted character of each of ``widget``'s rows, top to bottom."""
    strips = list(app.screen._compositor.render_strips())
    column = []
    for y in range(widget.region.y, widget.region.bottom):
        text = strips[y].crop(widget.region.x, widget.region.right).text.strip()
        column.append(text[:1] or " ")
    return "".join(column)


async def test_shared_shell_puts_only_its_destination_classes_on_its_parts() -> None:
    app = _StyledShellApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        shell = app.query_one("#probe-shell", shared.AdaptivePaneShell)
        assert list(shell.children) == [
            shell.library,
            shell.library_grip,
            shell.items,
            shell.items_grip,
            shell.work,
        ]
        assert shell.has_class("probe-pane-shell")
        assert shell.library.has_class("probe-pane-nav")
        assert shell.items.has_class("probe-pane-items")
        assert shell.work.has_class("probe-pane-work")
        assert shell.library_grip.has_class("probe-pane-grip")
        assert shell.items_grip.has_class("probe-pane-grip")
        leaked = sorted(
            {
                css_class
                for node in [shell, *shell.walk_children()]
                for css_class in node.classes
                if css_class.startswith("library-")
            }
        )
        assert leaked == []


async def test_shared_shell_posts_the_neutral_messages() -> None:
    app = _StyledShellApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        shell = app.query_one("#probe-shell", shared.AdaptivePaneShell)
        assert app.resizes >= 1
        # The first sync installs a layout from nothing, so both panes report.
        assert app.visibility == [("library", True), ("items", True)]
        shell.sync_layout(_layout(nav_open=False))
        await pilot.pause()
        assert app.visibility[-1] == ("library", False)
        assert shell.library.display is False and shell.library.disabled is True
        shell.items_grip.press()
        await pilot.pause()
        assert app.toggles == ["items"]


async def test_grip_paints_the_destination_painted_name() -> None:
    app = _StyledShellApp(painted_names={"Characters": "Kinds"})
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        shell = app.query_one("#probe-shell", shared.AdaptivePaneShell)
        assert shell.library_grip.painted_name() == "Kinds"
        assert shell.library_grip.tooltip == "Collapse Characters pane"
        assert _painted_column(app, shell.items_grip).startswith("Lorebooks")


async def test_sync_label_repaints_the_painted_name_once_and_only_on_a_change() -> None:
    """``sync_open`` alone never repaints a renamed grip: nothing reactive changed."""
    app = _StyledShellApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grip = app.query_one("#probe-shell", shared.AdaptivePaneShell).items_grip
        grip.sync_label("Chat dictionaries")
        await pilot.pause()
        assert _painted_column(app, grip).startswith("Chatdictionari")
        assert grip.tooltip == "Collapse Chat dictionaries pane"
        refreshes: list[tuple] = []
        original = grip.refresh

        def counting_refresh(*args, **kwargs):
            refreshes.append(args)
            return original(*args, **kwargs)

        grip.refresh = counting_refresh
        grip.sync_label("Chat dictionaries")
        assert refreshes == []


class _FocusShellApp(App):
    def compose(self) -> ComposeResult:
        yield shared.AdaptivePaneShell(
            Vertical(Button("Nav action", id="probe-nav-action")),
            Vertical(Button("Items action", id="probe-items-action")),
            Static("Work"),
            _layout(),
            id_prefix="probe",
            library_label="Characters",
            items_label="Lore books",
            destination=PROBE_CLASSES,
            id="probe-shell",
        )


async def test_closing_a_focused_pane_moves_focus_to_its_grip() -> None:
    app = _FocusShellApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        app.query_one("#probe-nav-action", Button).focus()
        await pilot.pause()
        shell = app.query_one("#probe-shell", shared.AdaptivePaneShell)
        shell.sync_layout(_layout(nav_open=False))
        await pilot.pause()
        assert app.focused is shell.library_grip


def test_the_shared_module_carries_no_library_class_and_imports_no_library_code() -> None:
    """Roleplay imports this module: it must never pull the Library in.

    Class tokens are checked in string constants, not as a substring of the
    source: the grip ids are ``f"{id_prefix}-library-grip"``, whose constant
    piece ``-library-grip`` names the resolver's ``"library"`` pane, not a
    class, and does not start with ``library-``.
    """
    source = Path(shared.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_literals = sorted(
        token
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        for token in node.value.split()
        if token.startswith("library-")
    )
    assert class_literals == []
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    forbidden = (
        "tldw_chatbook.Widgets.Library",
        "tldw_chatbook.Library",
        "tldw_chatbook.UI.Library_Modules",
        ".Library",
    )
    assert sorted(name for name in imported if name.startswith(forbidden)) == []


def test_no_ui_ready_resident_module_imports_the_shared_shell(tmp_path: Path) -> None:
    """The UI-ready census has no headroom: residents must not import it."""
    for name in ("data", "config", "home"):
        (tmp_path / name).mkdir()
    env = {
        **os.environ,
        "TLDW_TEST_MODE": "1",
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "HOME": str(tmp_path / "home"),
        "PYTHONPATH": str(ROOT),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("TLDW_CONFIG_PATH", None)
    code = (
        "import sys\n"
        "import tldw_chatbook.Utils.adaptive_reader_state\n"
        "import tldw_chatbook.Widgets.destination_rail\n"
        "import tldw_chatbook.UI.Navigation.base_app_screen\n"
        "print('tldw_chatbook.Widgets.adaptive_pane_shell' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "False"


SHARED_WIDGET_NAMES = ["AdaptivePaneShell", "AdaptivePaneGrip"]


def test_shared_widgets_declare_no_class_level_css() -> None:
    """Every shell rule is destination-keyed in a lazy sheet: zero boot bytes."""
    for name in SHARED_WIDGET_NAMES:
        widget_class = getattr(shared, name)
        for attribute in ("DEFAULT_CSS", "CSS", "BUNDLED_CSS"):
            assert attribute not in vars(widget_class), (name, attribute)
