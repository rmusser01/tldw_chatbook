"""Settings ▸ Providers & Models: the restored-OpenAI review row is read
again when it must be (review of #3057, R1 #3 and R3 #9).

The pending read answered "not pending" when it could not tell, the card
kept that answer for the life of the screen, and saving a changed OpenAI
connection (which can need a new review) never asked again.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.widgets import Input

from Tests.private_profile import private_profile_test
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_configuration_hub import (
    _capture_provider_settings_mutations,
)
from Tests.UI.test_settings_connect_rows import _FAKE_KEY, _SIZE, _open_providers
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from Tests.UI.test_settings_provider_keyboard_journeys import _settle
from tldw_chatbook.LLM_Calls import recovery_review
from tldw_chatbook.UI.Settings_Modules import providers_models_card as card


def _refuse_storage(*_args, **_kwargs):
    raise OSError("storage lease refused")


def _fake_screen():
    """A screen that runs the card's worker and its UI callback in place."""
    return SimpleNamespace(
        _openai_reconnect_pending=None,
        is_mounted=False,
        app=SimpleNamespace(call_from_thread=lambda callback, *args: callback(*args)),
        run_worker=lambda work, **_options: work(),
        call_after_refresh=lambda callback, *args: callback(*args),
    )


@pytest.mark.bootstrap_profile
def test_the_pending_read_says_when_it_cannot_tell(monkeypatch, tmp_path):
    monkeypatch.setattr(
        recovery_review.bootstrap,
        "effective_config_path",
        lambda: tmp_path / "config.toml",
    )
    monkeypatch.setattr(recovery_review, "acquire_storage", _refuse_storage)

    assert recovery_review.openai_reconnect_pending() is None


@pytest.mark.bootstrap_profile
def test_a_read_that_could_not_tell_is_not_kept_and_the_next_compose_asks_again(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        recovery_review.bootstrap,
        "effective_config_path",
        lambda: tmp_path / "config.toml",
    )
    monkeypatch.setattr(recovery_review, "acquire_storage", _refuse_storage)
    screen = _fake_screen()

    card.read_openai_reconnect_once(screen, "openai")
    assert screen._openai_reconnect_pending is None

    monkeypatch.setattr(recovery_review, "openai_reconnect_pending", lambda: True)
    card.read_openai_reconnect_once(screen, "openai")
    assert screen._openai_reconnect_pending is True


@pytest.mark.asyncio
@private_profile_test
async def test_saving_a_changed_openai_connection_reads_the_review_state_again(
    request, monkeypatch
):
    answers = [False]
    monkeypatch.setattr(
        recovery_review, "openai_reconnect_pending", lambda: answers[-1]
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _capture_provider_settings_mutations(monkeypatch)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": {"api_key": _FAKE_KEY}}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        row = screen.query_one("#settings-openai-reconnect-row")
        assert not row.display

        # The saved connection changes, and the restored one now needs review.
        answers.append(True)
        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        endpoint.focus()
        await pilot.pause()
        await pilot.press(*"https://proxy.example/v1", "escape", "s")
        await _settle(host, pilot)

        assert not screen._settings_drafts
        assert row.display
