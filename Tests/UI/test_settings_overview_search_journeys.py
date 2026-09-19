"""Overview navigation and guided search retain visible keyboard recovery paths."""

import tomllib

import pytest
from textual.widgets import Button, Input, Select, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_settings_provider_keyboard_journeys import (
    _edit,
    _revert,
    _settle,
    _tab_to,
)
from Tests.UI.test_settings_web_search import SearchSettingsHarness
from Tests.UI.test_settings_web_search import setup as setup  # noqa: PLC0414
from tldw_chatbook import config
from tldw_chatbook.Web_Scraping import search_backend_settings as catalog


async def _category(host, pilot, name):
    await pilot.press("escape", "/", *name, "enter")
    await _settle(host, pilot)


def _painted(host, widget):
    region = widget.region
    strips = host.screen._compositor.render_strips()
    return "\n".join(
        strips[y].crop(region.x, region.right).text
        for y in range(max(0, region.y), min(region.bottom, len(strips)))
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@private_profile_test
async def test_overview_and_search_keyboard_recovery(
    request, setup, theme, size, monkeypatch
):
    _, path = setup
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": ""}
    host = SearchSettingsHarness(app, "settings")
    host.theme = theme
    calls = []

    def probe(backend):
        calls.append(backend)
        return catalog.ProbeResult(
            False,
            "Could not connect. Check network access and the configured endpoint.",
        )

    monkeypatch.setattr(catalog, "probe_saved_backend", probe)
    async with host.run_test(size=size) as pilot:
        await _settle(host, pilot)
        text = str(
            host.screen.query_one("#settings-overview-configuration", Static).renderable
        )
        assert "Not ready" in text
        for target, category in (
            ("providers-models", "providers-models"),
            ("storage", "storage"),
            ("privacy-security", "privacy-security"),
        ):
            button = await _tab_to(host, pilot, f"#settings-overview-open-{target}")
            assert str(button.label) in _painted(host, button)
            if target == "providers-models":
                for resized in ((80, 24), (170, 48), size):
                    await pilot.resize_terminal(*resized)
                    await _settle(host, pilot)
                    assert host.screen.focused is button
                    assert str(button.label) in _painted(host, button)
            await pilot.press("enter")
            await _settle(host, pilot)
            assert host.screen.active_category == category
            await _category(host, pilot, "Overview")
        await _category(host, pilot, "Web Search")
        model = host.screen._web_search_model()
        field_id = "#web-search-serper_search_api_key"
        await _edit(host, pilot, field_id, "synthetic-replacement")
        field = host.screen.query_one(field_id, Input)
        assert field.password
        assert "synthetic-replacement" not in _painted(host, field)
        assert host.screen.query_one("#web-search-test", Button).disabled
        before = path.read_bytes()
        await _category(host, pilot, "Overview")
        await _category(host, pilot, "Web Search")
        assert host.screen.query_one(field_id, Input).value == "synthetic-replacement"
        await _revert(host, pilot, discard=False)
        assert model.draft.is_dirty
        await _revert(host, pilot, discard=True)
        assert path.read_bytes() == before
        assert not model.draft.is_dirty
        await _edit(host, pilot, field_id, "replacement-after-revert")
        real_writer = config.apply_settings_mutation_to_cli_config

        def fail(*args, **kwargs):
            raise OSError("synthetic-sensitive-error")

        monkeypatch.setattr(config, "apply_settings_mutation_to_cli_config", fail)
        await pilot.press("escape", "s")
        await _settle(host, pilot)
        assert "Could not save" in model.save_status
        assert "synthetic-sensitive-error" not in model.save_status
        assert model.draft.is_dirty and path.read_bytes() == before
        monkeypatch.setattr(
            config, "apply_settings_mutation_to_cli_config", real_writer
        )
        await pilot.press("escape", "s")
        await _settle(host, pilot)
        assert not model.draft.is_dirty
        assert (
            tomllib.loads(path.read_text())["SearchEngines"]["serper_search_api_key"]
            == "replacement-after-revert"
        )
        assert not calls
        button = await _tab_to(host, pilot, "#web-search-test")
        assert str(button.label) in _painted(host, button)
        await pilot.press("enter")
        await _settle(host, pilot)
        assert calls == ["serper"]
        assert "Could not connect" in model.test_status
        assert not button.disabled
        await _tab_to(host, pilot, "#web-search-backend")
        await pilot.press("enter", "home", "down", "enter")
        await _settle(host, pilot)
        assert model.backend == "brave"
        assert host.screen.query_one("#web-search-default", Select).value == "serper"
        assert "Not tested" in model.test_status
        assert host.screen.query_one("#web-search-test", Button).disabled
        assert "Setup incomplete" in model.setup_status
        assert calls == ["serper"]
