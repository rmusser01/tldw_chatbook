"""ModelCatalogConsentModal dismisses with the user's allow/deny choice."""

import pytest
from textual.app import App

from tldw_chatbook.UI.Screens.model_catalog_consent import ModelCatalogConsentModal


class _ConsentHost(App):
    def __init__(self):
        super().__init__()
        self.results = []

    def on_mount(self):
        self.push_screen(
            ModelCatalogConsentModal(), lambda result: self.results.append(result)
        )


@pytest.mark.asyncio
async def test_allow_button_dismisses_true():
    app = _ConsentHost()
    async with app.run_test() as pilot:
        await pilot.click("#model-catalog-consent-allow")
        await pilot.pause()
    assert app.results == [True]


@pytest.mark.asyncio
async def test_deny_button_dismisses_false():
    app = _ConsentHost()
    async with app.run_test() as pilot:
        await pilot.click("#model-catalog-consent-deny")
        await pilot.pause()
    assert app.results == [False]


@pytest.mark.asyncio
async def test_escape_dismisses_false():
    app = _ConsentHost()
    async with app.run_test() as pilot:
        await pilot.press("escape")
        await pilot.pause()
    assert app.results == [False]


@pytest.mark.asyncio
async def test_app_push_suppresses_modal_in_headless_runs():
    """run_test() is headless: no user can answer, so nothing is pushed.

    This is the guard that keeps full-app UI tests (GGUF source modes,
    first-run flows, ...) free of an interleaved consent dialog.
    """
    from tldw_chatbook.app import TldwCli

    class HostApp(App):
        pass

    app = HostApp()
    async with app.run_test() as pilot:
        TldwCli._push_model_catalog_consent_modal(app)
        await pilot.pause()
        assert not isinstance(app.screen, ModelCatalogConsentModal)
