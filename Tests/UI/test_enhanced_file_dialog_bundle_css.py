"""Bundle-CSS regression guard for the enhanced file picker's action bar.

TASK-16478: the app bundle's bare ``Select { width: 100%; }`` rule
(``css/features/_conversations.tcss``) outranks any widget ``DEFAULT_CSS``,
so with the bundle loaded every *filtered* ``EnhancedFileDialog`` rendered
its file-type Select at full width: the filename Input was crushed to a
few columns and the Select/Cancel buttons were laid out past the dialog's
right edge, clipped out of view. ``css/components/_dialogs.tcss`` now pins
``EnhancedFileDialog InputBar Select`` to a fixed width, like the vendored
dialogs. These tests mount the dialog under the same CSS tiers the real
app registers (``app.py``'s CSS_PATH) and pin the input-bar geometry, so
a bundle rule that re-breaks the row fails here instead of in front of a
user.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult

import tldw_chatbook
from tldw_chatbook.Third_Party.textual_fspicker import Filters
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen

_CSS_DIR = Path(tldw_chatbook.__file__).parent / "css"
_DIALOG_ID = "#enhanced-file-dialog"


def _picker_filters() -> Filters:
    return Filters(
        ("Character cards (PNG/JSON)", lambda p: p.suffix.lower() in (".png", ".json")),
        ("All Files", lambda p: True),
    )


class _BundleHost(App[None]):
    """Host registering the same CSS sources as ``TldwCli`` (app.py CSS_PATH)."""

    CSS_PATH = [
        str(_CSS_DIR / "screen_css_scoped.tcss"),
        str(_CSS_DIR / "tldw_cli_modular.tcss"),
        str(_CSS_DIR / "screen_css_self.tcss"),
    ]

    def __init__(self, dialog):
        super().__init__()
        self._dialog = dialog

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        await self.push_screen(self._dialog)


@pytest.mark.asyncio
async def test_filtered_dialog_action_bar_stays_inside_dialog(tmp_path):
    """Import/Cancel sit inside the dialog; the filename input keeps flex room.

    Without the ``_dialogs.tcss`` override the bare ``Select { width: 100% }``
    rule pushed the buttons to x > 150 in a 152-wide dialog (observed:
    filename input width 6, select at x=161, cancel at x=178 -- both
    clipped).
    """
    dialog = EnhancedFileOpen(
        title="Import Character Card",
        location=tmp_path,
        filters=_picker_filters(),
        context="t_bundle_action_bar",
    )
    app = _BundleHost(dialog)

    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause()
        dialog_region = dialog.query_one(_DIALOG_ID).region
        filename = dialog.query_one("#filename-input").region
        buttons = {
            "#select": dialog.query_one("#select").region,
            "#cancel": dialog.query_one("#cancel").region,
        }

        dialog_right = dialog_region.x + dialog_region.width
        for label, region in buttons.items():
            assert region.width > 0, f"{label} has zero width"
            assert (
                region.x + region.width <= dialog_right + 1
            ), f"{label} is clipped past the dialog's right edge (x={region.x})"
        assert filename.width >= 20, (
            f"filename input crushed to {filename.width} columns; "
            "the filter Select is eating the row again"
        )


@pytest.mark.asyncio
async def test_filter_select_is_pinned_not_full_width(tmp_path):
    (tmp_path / "card.png").write_bytes(b"x")

    dialog = EnhancedFileOpen(
        title="Import Character Card",
        location=tmp_path,
        filters=_picker_filters(),
        context="t_bundle_select_width",
    )
    app = _BundleHost(dialog)

    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause()
        select_region = dialog.query_one("#file-filter").region
        dialog_region = dialog.query_one(_DIALOG_ID).region
        # Pinned (24) plus chrome slack -- never a share of the whole row.
        assert select_region.width <= 30, (
            f"filter Select is {select_region.width} wide; the bare "
            "`Select {{ width: 100% }}` bundle rule is beating the pin again"
        )
        assert select_region.width < dialog_region.width
