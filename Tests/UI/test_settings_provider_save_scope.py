"""Providers & Models copy names what a save applies to (TASK-33002.3).

ADR-095 as amended 2026-09-26 (D1): a Settings save reaches new chats and
open chats nobody has used yet; a chat with work keeps its own settings,
changed per chat in Console with Alt+M. The save result, the toast, the
State line and the Provider field's Purpose used to say none of that
("Provider settings saved.", "Shared with Console", "Console generation
defaults"). The save result and toast are pinned on the real writer in
test_settings_configuration_hub.py.
"""

from __future__ import annotations

import time

import httpx
import pytest
from textual.widgets import Input, Select, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _static_text,
    _wait_for_selector,
)
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from tldw_chatbook.state.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen


@pytest.mark.asyncio
@private_profile_test
async def test_providers_models_state_line_names_its_scope_in_one_row(request) -> None:
    """AC#2 on the rendered banner (production stylesheet, 211x44).

    Plan ruling T3 x T4: the dirty line becomes "State: {badge} · N unsaved
    | {scope}" with no Save/Revert tail (the badge and the footer's s/r
    hints carry that guidance), so the scope must leave room for the count.
    The real-edit check fails if the dirty line ever wraps, whatever T4
    makes it say. The F1 help body repeats the scope under "Scope:".
    """
    category = SettingsCategoryId.PROVIDERS_MODELS
    host = _SettingsCssHarness(_build_test_app(), "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        screen = _active_destination_screen(host)
        screen._select_category(category.value)
        await _wait_for_selector(screen, pilot, "#settings-model-value", timeout=30.0)
        banner = screen.query_one("#settings-category-state-banner", Static)
        # TASK-38963 review r10e-rev1: the banner can exist before its first
        # layout (1 run in 20 read a content_region of height 0); wait for it.
        deadline = time.monotonic() + 10
        while banner.size.width == 0 and time.monotonic() < deadline:
            await pilot.pause()
        scope = screen._category_state_scope_text(category)
        assert scope == (
            "Applies to new and unused open chats · used chats keep theirs "
            "(Console: Alt+M)"
        )
        # Capture finding 3, rewritten on purpose: at rest the banner read
        # "Draft — save with s" beside the Inspector's "No unsaved changes".
        clean = f"State: No changes · edits save with s | {scope}"
        assert _static_text(banner) == clean
        assert _static_text(
            screen.query_one("#settings-selected-category-draft-status", Static)
        ) == "No unsaved changes"
        assert banner.content_region.height == 1

        widest = f"State: Draft — save with s · 99 unsaved | {scope}"
        banner.update(widest)
        await pilot.pause()
        assert (_static_text(banner), banner.content_region.height) == (widest, 1)

        screen.query_one("#settings-model-value", Input).value = "scope-review"
        await pilot.pause()
        assert screen._category_has_unsaved_changes(category)
        assert _static_text(banner) == screen._category_state_banner_text(category)
        assert banner.content_region.height == 1

        screen.action_show_workbench_help()
        await pilot.pause()
        body = str(host.screen.query_one("#workbench-help-body", Static).render())
        assert f"Scope: {scope}" in body, body


@pytest.mark.asyncio
@private_profile_test
async def test_provider_field_purpose_names_the_new_chat_scope(request) -> None:
    screen = SettingsScreen(_build_test_app())
    screen._active_settings_field_id = "settings-provider-value"

    rows = dict(screen._provider_field_guidance_rows_base())

    assert rows["Focused setting"] == "Provider"
    assert rows["Purpose"] == (
        "Sets the provider new chats start with; open chats nobody has used "
        "yet follow it."
    )
    # Final review (Task 3 rider): the Model row kept "when Console has no
    # narrower override" framing; it names the same scope now.
    screen._active_settings_field_id = "settings-model-value"
    assert dict(screen._provider_field_guidance_rows_base())["Purpose"] == (
        "Sets the model new chats start with; open chats nobody has used yet follow it."
    )


@pytest.mark.asyncio
@private_profile_test
async def test_focusing_the_visible_provider_picker_shows_the_provider_purpose(
    request,
) -> None:
    """AC#3 live: the hidden Select was the only id the Provider rows named.

    The Provider field a user reaches is the search box and its picker list;
    focusing either used to leave the inspector on "None — Tab to a setting",
    so the Purpose row could never be read.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        screen = _active_destination_screen(host)
        screen._select_category(SettingsCategoryId.PROVIDERS_MODELS.value)
        await _wait_for_selector(
            screen, pilot, "#settings-provider-search", timeout=30.0
        )
        # TASK-33007.2, rewritten on purpose: the list under the Provider
        # control never takes focus, so the control is the one field.
        for selector in ("#settings-provider-search",):
            screen.query_one(selector).focus()
            await pilot.pause()
            guide = [
                _static_text(screen.query_one(f"#settings-provider-field-guide-{i}"))
                for i in range(2)
            ]
            assert guide[0] == "Focused setting: Provider", selector
            assert guide[1] == (
                "Purpose: Sets the provider new chats start with; open chats "
                "nobody has used yet follow it."
            ), selector


_KOBOLD_REFUSAL = (
    "KoboldCpp needs its native /api/v1/generate address. "
    "For an OpenAI-compatible server, choose Custom instead."
)


async def _save_provider_card(screen, pilot) -> str:
    """Press Save and return the provider card's result line once it is set."""
    await pilot.pause()
    screen._provider_save_result = ""
    await pilot.click("#settings-save-category")
    deadline = time.monotonic() + 10
    while not screen._provider_save_result and time.monotonic() < deadline:
        await pilot.pause(0.05)
    return screen._provider_save_result


@pytest.mark.asyncio
@private_profile_test
async def test_koboldcpp_saves_only_the_native_route_its_send_posts_to(
    request, monkeypatch
) -> None:
    """TASK-36516: Settings saved koboldcpp's suggested bare root as
    ``<root>/v1/chat/completions``, where ``chat_with_kobold`` posts KoboldAI's
    native payload, so the send failed. A bare root is now refused, naming the
    native route, and writes nothing; the placeholder is that route, it saves
    unchanged (AC#2), and a send then posts the native payload to it.
    """
    from tldw_chatbook.Chat.Chat_Functions import chat_api_call
    from tldw_chatbook.config import load_cli_config_and_ensure_existence
    from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local as local

    requests = []

    class _Session:
        def post(self, url, **kwargs):
            requests.append((url, kwargs["json"]))
            return httpx.Response(
                200,
                json={"results": [{"text": "synthetic-native-reply"}]},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(local, "create_pooled_default_session", lambda **kw: _Session())
    native = "http://127.0.0.1:5001/api/v1/generate"
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async def _save(screen, pilot, endpoint: str) -> str:
        screen.query_one("#settings-provider-endpoint-value", Input).value = endpoint
        screen.query_one("#settings-model-value", Input).value = "kobold-model"
        return await _save_provider_card(screen, pilot)

    async with host.run_test(size=(211, 44)) as pilot:
        screen = _active_destination_screen(host)
        screen._select_category(SettingsCategoryId.PROVIDERS_MODELS.value)
        await _wait_for_selector(screen, pilot, "#settings-model-value", timeout=30.0)
        provider = screen.query_one("#settings-provider-value", Select)
        provider.value = "koboldcpp"
        screen.handle_provider_value_changed(Select.Changed(provider, "koboldcpp"))
        await pilot.pause()
        saved_before = load_cli_config_and_ensure_existence(force_reload=True)
        url_before = saved_before["api_settings"]["koboldcpp"].get("api_url")

        result = await _save(screen, pilot, "http://127.0.0.1:5001")
        assert result == _KOBOLD_REFUSAL
        saved = load_cli_config_and_ensure_existence(force_reload=True)
        assert saved["api_settings"]["koboldcpp"].get("api_url") == url_before

        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        assert endpoint.placeholder == native
        result = await _save(screen, pilot, endpoint.placeholder)
        assert result.startswith("Provider settings saved"), result

    saved = load_cli_config_and_ensure_existence(force_reload=True)
    assert saved["api_settings"]["koboldcpp"]["api_url"] == native
    reply = chat_api_call(
        "koboldcpp",
        [{"role": "user", "content": "synthetic native prompt"}],
        model="kobold-model",
        streaming=False,
    )
    assert reply["choices"][0]["message"]["content"] == "synthetic-native-reply"
    assert len(requests) == 1
    url, payload = requests[0]
    assert url == native
    assert payload["prompt"] == "synthetic native prompt"
    assert "messages" not in payload


@pytest.mark.asyncio
@private_profile_test
async def test_the_endpoint_row_refuses_a_koboldcpp_root_as_it_is_typed(
    request,
) -> None:
    """TASK-38964 AC#4: Settings refused a koboldcpp root only when **s** saved
    it, while the wizard says so as the address is typed. The Endpoint row
    now says it before any save, and the save still refuses."""
    host = DestinationHarness(_build_test_app(), "settings")

    async with host.run_test(size=(211, 44)) as pilot:
        screen = _active_destination_screen(host)
        screen._select_category(SettingsCategoryId.PROVIDERS_MODELS.value)
        await _wait_for_selector(screen, pilot, "#settings-model-value", timeout=30.0)
        provider = screen.query_one("#settings-provider-value", Select)
        provider.value = "koboldcpp"
        screen.handle_provider_value_changed(Select.Changed(provider, "koboldcpp"))
        await pilot.pause()

        def row_help() -> str:
            return _static_text(screen.query_one("#settings-provider-endpoint-help"))

        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        endpoint.focus()
        await pilot.pause()
        await pilot.press("end", *["backspace"] * len(endpoint.value))
        await pilot.press(*"http://127.0.0.1:5001")
        await pilot.pause()
        assert endpoint.value == "http://127.0.0.1:5001"
        assert row_help() == _KOBOLD_REFUSAL
        assert screen._provider_save_result == ""

        await pilot.press(*"/api/v1/generate")
        await pilot.pause()
        assert row_help() == "the server's /api/v1/generate address"

        await pilot.press(*["backspace"] * len("/api/v1/generate"))
        screen.query_one("#settings-model-value", Input).value = "kobold-model"
        assert await _save_provider_card(screen, pilot) == _KOBOLD_REFUSAL
        assert row_help() == _KOBOLD_REFUSAL


@pytest.mark.asyncio
@private_profile_test
async def test_a_stored_koboldcpp_chat_route_refuses_only_saves_that_write_it(
    request,
) -> None:
    """TASK-36516 review r8e-rev1: the native-route check ran before Save knew
    what it writes, so a context-window save over a ``/v1/chat/completions``
    route the old bug had stored was refused. That save now goes through and
    leaves the route alone; a model change, which rewrites the endpoint, is
    still refused (AC#1)."""
    from tldw_chatbook.config import (
        load_cli_config_and_ensure_existence,
        save_settings_to_cli_config,
    )

    stored = "http://127.0.0.1:5001/v1/chat/completions"
    assert save_settings_to_cli_config(
        {
            "chat_defaults": {"provider": "koboldcpp", "model": "kobold-model"},
            "api_settings.koboldcpp": {"api_url": stored, "model": "kobold-model"},
        }
    )
    host = DestinationHarness(_build_test_app(), "settings")

    async with host.run_test(size=(211, 44)) as pilot:
        screen = _active_destination_screen(host)
        screen._select_category(SettingsCategoryId.PROVIDERS_MODELS.value)
        await _wait_for_selector(screen, pilot, "#settings-model-value", timeout=30.0)
        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        assert endpoint.value == stored

        screen.query_one("#settings-model-context-window", Input).value = "4096"
        result = await _save_provider_card(screen, pilot)
        assert result.startswith("Provider settings saved"), result

        screen.query_one("#settings-model-value", Input).value = "kobold-model-2"
        assert await _save_provider_card(screen, pilot) == _KOBOLD_REFUSAL

    saved = load_cli_config_and_ensure_existence(force_reload=True)
    assert saved["api_settings"]["koboldcpp"]["api_url"] == stored
    assert saved["api_settings"]["koboldcpp"]["model"] == "kobold-model"
    windows = saved["model_capabilities"]["models"]
    assert windows["kobold-model"]["context_window"] == 4096

