"""Settings ▸ Providers & Models: a key pasted into Env var is not repeated in
another row (review of #3057, R1 #25, ruling C4).

The Env var field is not masked. The screen builds two texts from its value
that other rows print: the Test result's Key row ("... or set NAME", "from
env var NAME") and a custom endpoint's credential line ("env var NAME").
A value shaped like a key is named "that variable" there.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_connect_rows import _SIZE, _open_providers
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from Tests.UI.test_settings_provider_keyboard_journeys import _settle
from tldw_chatbook.Chat.custom_endpoint_registry import entry_for
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen, env_var_echo

#: Shaped like a key, and still a name an environment variable may have.
_PASTED = "sk_live_" + "a1B2c3D4" * 3
_NAME = "A2_ECHO_VAR"


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize(
    ("value", "echoed"),
    [
        (_NAME, _NAME),
        ("OPENAI_API_KEY", "OPENAI_API_KEY"),
        (_PASTED, "that variable"),
        ("a" * 41, "that variable"),
        ("A_" + "B" * 40, "A_" + "B" * 40),
        # Fix round 1 (findings 3 and 15): provider keys that are also valid
        # variable names and match no secret pattern. Only text written the
        # way variable names are written (upper case) is repeated.
        ("gsk_" + "a1B2c3D4e5F6g" * 4, "that variable"),  # Groq
        ("hf_" + "aB3dE6gH9jK2mN5pQ" * 2, "that variable"),  # Hugging Face
        ("a1B2c3D4" * 4, "that variable"),  # 32 characters (Mistral)
        ("a1B2c3D4" * 5, "that variable"),  # exactly 40 (Cohere)
        ("my_openai_key", "that variable"),  # a lower-case name is not echoed
    ],
)
def test_a_value_shaped_like_a_key_is_named_that_variable(value, echoed):
    assert env_var_echo(value) == echoed


@pytest.mark.bootstrap_profile
def test_a_custom_endpoint_env_var_shaped_like_a_key_is_not_in_its_credential_line(
    monkeypatch,
):
    monkeypatch.setenv(_PASTED, "endpoint-key-abcdefgh1234")
    entry = entry_for(
        {
            "custom_endpoints": {
                "box": {
                    "display_name": "Box",
                    "family": "openai_compatible",
                    "base_url": "http://192.168.1.6:9000/v1",
                    "api_key_env": _PASTED,
                    "models": ["model-a"],
                }
            }
        },
        "custom-ep:box",
    )
    assert entry is not None and entry.api_key_env == _PASTED
    screen = SimpleNamespace(
        _provider_is_registry_id=lambda _provider: True,
        _provider_registry_entry=lambda _provider: entry,
    )

    line = SettingsScreen._provider_registry_credential_text(
        screen, "custom-ep:box", SimpleNamespace(ready=True)
    )

    assert line == "env var that variable (this endpoint)"


async def _test_result(host, pilot, screen, value: str) -> str:
    """Stage ``value`` as the Env var, run Test, return the result's text."""
    screen.query_one("#settings-provider-credential-env-var", Input).value = value
    try:
        async with asyncio.timeout(10):
            while screen._provider_current_credential_source("openai") != (
                "environment"
            ):
                await pilot.pause(0.02)
    except TimeoutError as exc:
        raise AssertionError("the Env var edit was never staged") from exc
    screen.query_one("#settings-test-provider", Button).press()
    await _settle(host, pilot)
    return str(screen.query_one("#settings-provider-test-result", Static).renderable)


@pytest.mark.asyncio
@private_profile_test
async def test_the_test_result_does_not_repeat_a_key_pasted_into_env_var(
    request, monkeypatch
):
    for name in ("OPENAI_API_KEY", _NAME, _PASTED):
        monkeypatch.delenv(name, raising=False)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": {}}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)

        named = await _test_result(host, pilot, screen, _NAME)
        assert f"or set {_NAME}" in named, named

        pasted = await _test_result(host, pilot, screen, _PASTED)
        assert "or set that variable" in pasted, pasted
        shown = "\n".join(str(static.renderable) for static in screen.query(Static))
        assert _PASTED not in shown
