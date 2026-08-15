"""Personas Visual Identity pack browser contracts (TASK-16319.3 Task 12)."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from PIL import Image
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Button, Input, OptionList, Static

from tldw_chatbook.Character_Chat.visual_identity import (
    SAMIRA_EXPRESSION_KEYS,
    SAMIRA_REACTION_LABELS,
)


def _browser_module():
    """Import the wished-for browser API with a useful RED failure."""

    try:
        return importlib.import_module(
            "tldw_chatbook.Widgets.Persona_Widgets.personas_visual_identity_pack_widget"
        )
    except ModuleNotFoundError:
        pytest.fail("Personas Visual Identity pack browser is not implemented yet")


def test_visual_identity_pack_browser_module_exists():
    assert _browser_module() is not None


def _pack_api():
    module = importlib.import_module(
        "tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages"
    )
    try:
        return module.VisualIdentityAssetMetadata, module.VisualIdentityPackMetadata
    except AttributeError:
        pytest.fail("typed Visual Identity pack metadata is not implemented yet")


def _samira_pack():
    Asset, Pack = _pack_api()
    assets = tuple(
        Asset(
            asset_id=index,
            expression_key=SAMIRA_EXPRESSION_KEYS[label],
            original_label=label,
            display_label=label.title(),
            content_type="image/webp",
            is_animated=False,
        )
        for index, label in enumerate(SAMIRA_REACTION_LABELS, start=1)
    )
    return Pack(
        pack_id=10,
        pack_version_id=20,
        title="Samira Reactions",
        source_kind="builtin",
        default_expression_key="neutral",
        assets=assets,
    )


def _browser_class():
    module = _browser_module()
    try:
        return module.PersonasVisualIdentityPackWidget
    except AttributeError:
        pytest.fail("PersonasVisualIdentityPackWidget is not implemented yet")


def _text(widget: Static) -> str:
    return str(widget.renderable)


def _browser_host(pack=None):
    Browser = _browser_class()

    class BrowserHost(App):
        CSS = Browser.BUNDLED_CSS

        def __init__(self):
            super().__init__()
            self.captured = []

        def compose(self) -> ComposeResult:
            yield Browser(pack)

        def on_visual_identity_pack_preview_requested(self, message):
            self.captured.append(message)

        def on_visual_identity_pack_replace_requested(self, message):
            self.captured.append(message)

        def on_visual_identity_pack_generate_requested(self, message):
            self.captured.append(message)

        def on_visual_identity_pack_clear_requested(self, message):
            self.captured.append(message)

        def on_visual_identity_pack_save_requested(self, message):
            self.captured.append(message)

    return BrowserHost()


@pytest.mark.asyncio
async def test_browser_filters_all_31_labels_and_keeps_exact_metadata_visible():
    pack = _samira_pack()
    app = _browser_host(pack)
    async with app.run_test(size=(120, 40)) as pilot:
        browser = app.query_one(_browser_class())
        options = browser.query_one("#personas-visual-identity-results", OptionList)
        assert options.option_count == 31
        assert (
            _text(browser.query_one("#personas-visual-identity-count", Static))
            == "1 / 31"
        )
        assert (
            _text(browser.query_one("#personas-visual-identity-label", Static))
            == "Admiration"
        )
        assert (
            _text(browser.query_one("#personas-visual-identity-key", Static))
            == "custom:admiration"
        )

        for label in SAMIRA_REACTION_LABELS:
            browser.apply_filter(label)
            await pilot.pause()
            # Substring search intentionally keeps related labels too
            # ("approval" also matches "disapproval"); every canonical label
            # must remain reachable and the current/filtered count stay exact.
            assert options.option_count >= 1, label
            assert _text(
                browser.query_one("#personas-visual-identity-count", Static)
            ) == (f"1 / {options.option_count}")
            assert (
                _text(browser.query_one("#personas-visual-identity-label", Static))
                == label.title()
            )
            assert (
                _text(browser.query_one("#personas-visual-identity-key", Static))
                == SAMIRA_EXPRESSION_KEYS[label]
            )


@pytest.mark.asyncio
async def test_filter_field_matches_internal_key_and_updates_count():
    app = _browser_host(_samira_pack())
    async with app.run_test(size=(120, 40)) as pilot:
        field = app.query_one("#personas-visual-identity-filter", Input)
        field.value = "custom:nervous"
        await pilot.pause()
        assert (
            app.query_one("#personas-visual-identity-results", OptionList).option_count
            == 1
        )
        assert (
            _text(app.query_one("#personas-visual-identity-label", Static))
            == "Nervousness"
        )
        assert (
            _text(app.query_one("#personas-visual-identity-key", Static))
            == "custom:nervousness"
        )


@pytest.mark.asyncio
async def test_widget_does_not_read_paths_or_decode_images(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("browser attempted file I/O or image decode")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Image, "open", forbidden)
    app = _browser_host(_samira_pack())
    async with app.run_test(size=(120, 40)) as pilot:
        browser = app.query_one(_browser_class())
        browser.apply_filter("joy")
        await pilot.pause()
        assert (
            _text(browser.query_one("#personas-visual-identity-label", Static)) == "Joy"
        )


@pytest.mark.asyncio
async def test_selected_preview_is_screen_supplied_and_replaces_the_prior_child():
    app = _browser_host(_samira_pack())
    async with app.run_test(size=(120, 40)) as pilot:
        browser = app.query_one(_browser_class())
        holder = browser.query_one("#personas-visual-identity-preview-image", Container)

        browser.set_preview("preview one", expression_key="custom:admiration")
        await pilot.pause()
        assert len(holder.children) == 1
        assert _text(holder.children[0]) == "preview one"

        browser.set_preview("preview two", expression_key="custom:admiration")
        await pilot.pause()
        assert len(holder.children) == 1
        assert _text(holder.children[0]) == "preview two"

        browser.set_preview("stale", expression_key="happy")
        await pilot.pause()
        assert len(holder.children) == 1
        assert _text(holder.children[0]) == "preview two"


@pytest.mark.asyncio
async def test_builtin_notice_dirty_summary_and_typed_action_messages():
    messages = importlib.import_module(
        "tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages"
    )
    app = _browser_host(_samira_pack())
    async with app.run_test(size=(120, 40)) as pilot:
        browser = app.query_one(_browser_class())
        app.captured.clear()

        notice = _text(browser.query_one("#personas-visual-identity-notice", Static))
        assert "Built-in" in notice
        assert "private copy" in notice

        for button_id, message_name in (
            ("personas-visual-identity-replace", "VisualIdentityPackReplaceRequested"),
            (
                "personas-visual-identity-generate",
                "VisualIdentityPackGenerateRequested",
            ),
            ("personas-visual-identity-clear", "VisualIdentityPackClearRequested"),
        ):
            browser.query_one(f"#{button_id}", Button).press()
            await pilot.pause()
            assert isinstance(app.captured[-1], getattr(messages, message_name))
            assert app.captured[-1].asset.expression_key == "custom:admiration"

        browser.set_staged_change("custom:admiration", "replace")
        assert (
            _text(browser.query_one("#personas-visual-identity-dirty", Static))
            == "1 staged change"
        )
        assert not browser.query_one("#personas-visual-identity-save", Button).disabled
        browser.query_one("#personas-visual-identity-save", Button).press()
        await pilot.pause()
        assert isinstance(app.captured[-1], messages.VisualIdentityPackSaveRequested)
        assert app.captured[-1].pack_id == 10
        assert app.captured[-1].pack_version_id == 20


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
async def test_browser_geometry_stays_in_bounds_and_hides_preview_first(size):
    app = _browser_host(_samira_pack())
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        browser = app.query_one(_browser_class())
        assert browser.region.right <= size[0]
        assert browser.region.bottom <= size[1]
        assert browser.query_one("#personas-visual-identity-results").display
        assert browser.query_one("#personas-visual-identity-actions").display
        preview = browser.query_one("#personas-visual-identity-preview")
        assert preview.display is (size != (80, 24))
