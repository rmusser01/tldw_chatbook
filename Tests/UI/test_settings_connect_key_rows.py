"""Settings ▸ Providers & Models ▸ Connect: the key and endpoint rows (TASK-33007.2).

Split out of test_settings_connect_rows.py so the UI Fast Lane's round-robin
shards can spread the Connect cases (each mounts the whole Settings screen,
~16 s a case on CI). The API key row's source words and Clear key, the Key
check row, an endpoint with no shipped default, and the card's result line.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_connect_rows import (
    _CLEAR_HINT,
    _CLEAR_KEY,
    _FAKE_KEY,
    _SAVED_KEY,
    _SIZE,
    _held,
    _no_provider_key_env,
    _open_providers,
    _resting_help_counts,
    _squeezed,
    _text,
    _type_filter,
    _until,
)
from Tests.UI.test_settings_narrow_layout import _region_rows, _SettingsCssHarness


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    ("setup", "word"),
    [("config", "saved in config"), ("env", "from env var"), ("none", "missing")],
)
async def test_api_key_row_says_where_the_key_comes_from(
    request, monkeypatch, setup, word
):
    """AC#4: masked, one row, the source in words, and Clear still offered."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    if setup == "env":
        monkeypatch.setenv("OPENAI_API_KEY", _FAKE_KEY)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {"api_key": _FAKE_KEY} if setup == "config" else {}
    }
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        api_key = screen.query_one("#settings-provider-api-key", Input)
        clear = screen.query_one("#settings-provider-api-key-clear", Button)
        status = screen.query_one("#settings-provider-key-status", Static)
        row = screen.query_one("#settings-provider-api-key-row")

        assert api_key.password
        assert row.region.height == 1
        assert str(status.renderable) == word
        assert clear.parent is row
        assert clear.disabled is (setup != "config")
        # Capture finding 10: Clear shows only while there is a key to clear.
        assert clear.visible is (setup == "config")
        assert _FAKE_KEY not in "\n".join(_region_rows(screen, row))
        # ADR-031 rule 4: the row names Clear's key only while it clears.
        assert (_CLEAR_HINT in _text(screen, "#settings-provider-api-key-help")) is (
            setup == "config"
        )

        env_row = screen.query_one("#settings-provider-env-var-row")
        endpoint_row = screen.query_one("#settings-provider-endpoint-row")
        assert env_row.region.height == endpoint_row.region.height == 1
        assert _text(screen, "#settings-provider-env-var-source") == (
            "set in shell" if setup == "env" else "not set"
        )
        assert _text(screen, "#settings-provider-endpoint-source") == "built-in"
        assert "safer" in _text(screen, "#settings-provider-credential-guidance")
        assert _text(screen, "#settings-provider-endpoint-help") == (
            "blank uses the provider default"
        )

        # Which key is sent when both exist: the row's help has no room for it
        # beside the two keys, so the Inspector's guide for the field says.
        api_key.focus()
        await pilot.pause()
        guide = " ".join(
            _text(screen, f"#settings-provider-field-guide-{index}")
            for index in range(4)
        )
        assert "Focused setting: API key" in guide
        assert "A saved key is used before the env var." in guide
        # ADR-031 rule 4 rests on this: while Clear is disabled its key does
        # nothing, so the row names the key only for a saved one. Clear is
        # live beside a saved key, so that case disables it by hand.
        clear.disabled = True
        await pilot.press(_CLEAR_KEY)
        await pilot.pause()
        assert not screen._settings_drafts
        assert str(status.renderable) == word


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    ("settings", "at_rest", "checked"),
    [
        pytest.param({}, "Not ready · no key", "Not ready · no key", id="no-key"),
        pytest.param(
            dict(_SAVED_KEY),
            "Ready · not tested",
            "Ready · verified",
            id="key-accepted",
        ),
    ],
)
async def test_key_check_row_ends_connect_and_its_detail_lives_in_the_inspector(
    request, monkeypatch, settings, at_rest, checked
):
    """AC#6, AC#7, AC#8: the verdict and a t action on one row, the last of
    Connect; t runs the check and the row then says its verdict (a saved key
    the provider's model listing accepts reads "Ready · verified"); the
    labelled rows go to the Inspector's Key block, so Default model does not
    move."""
    import httpx

    from Tests.UI.test_settings_provider_key_check import _Provider, _route_discovery_to

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    listing = _Provider(httpx.Response(200, json={"data": [{"id": "gpt-4.1"}]}))
    _route_discovery_to(monkeypatch, listing)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": dict(settings)}
    app.providers_models = {"OpenAI": ["gpt-4.1"]}  # the key check lists it
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        card = screen.query_one("#settings-providers-models-card")
        verdict = screen.query_one("#settings-provider-readiness", Static)
        button = screen.query_one("#settings-test-provider", Button)
        result = screen.query_one("#settings-provider-test-result", Static)
        default_model = screen.query_one("#settings-default-model-title", Static)

        assert str(verdict.renderable) == at_rest
        assert not str(result.renderable).startswith("Readiness")  # no check yet
        assert "(t)" in str(button.label)
        assert button.parent is verdict.parent
        assert verdict.parent.region.height == 1
        assert screen.query_one("#settings-impact-pane") in result.ancestors
        assert card not in result.ancestors
        assert "Provider readiness" not in [
            str(widget.renderable) for widget in card.query(".destination-section")
        ]
        assert _text(screen, "#settings-provider-source") == "new-chat default"
        assert _text(screen, "#settings-model-source") == "new-chat default"
        # The Key check row is the last shown row of Connect: the next shown
        # child of the card is Default model's header.
        shown = [child for child in card.children if child.display]
        assert shown[shown.index(verdict.parent) + 1] is default_model

        default_model_y = default_model.virtual_region.y
        # Esc releases the focused control; t then runs the check.
        await pilot.press("escape", "t")
        await _until(pilot, lambda: str(result.renderable).startswith("Readiness"))
        await host.workers.wait_for_complete()
        await _until(pilot, lambda: str(verdict.renderable).startswith(checked))

        assert str(result.renderable).startswith("Readiness")
        # A verified verdict carries the time of the check after the word.
        assert str(verdict.renderable).startswith(checked), str(result.renderable)
        assert len(listing.requests) == (1 if settings else 0)
        assert default_model.virtual_region.y == default_model_y


@pytest.mark.asyncio
@private_profile_test
async def test_ctrl_l_on_the_api_key_field_clears_the_saved_key_until_revert_or_save(
    request, monkeypatch
):
    """Owner ruling 2026-10-04: Clear is a key on the API key field. It
    stages the removal as the button does: a confirmed r undoes it, s saves."""
    from Tests.UI.test_settings_configuration_hub import (
        _capture_provider_settings_mutations,
    )
    from Tests.UI.test_settings_provider_keyboard_journeys import _revert, _settle
    from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId

    category = SettingsCategoryId.PROVIDERS_MODELS
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mutations = _capture_provider_settings_mutations(monkeypatch)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": dict(_SAVED_KEY)}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        api_key = screen.query_one("#settings-provider-api-key", Input)
        api_key.focus()
        await pilot.pause()
        await pilot.press(_CLEAR_KEY)
        await pilot.pause()

        assert host.focused is api_key
        assert _text(screen, "#settings-provider-key-status") == "cleared *"
        assert _text(screen, "#settings-provider-api-key-help") == (
            "s removes the saved key"
        )
        assert screen._settings_drafts[category].values["api_key"] == ""
        assert mutations == []
        assert app.app_config["api_settings"]["openai"]["api_key"] == _FAKE_KEY

        await _revert(host, pilot, discard=True)
        assert _text(screen, "#settings-provider-key-status") == "saved in config"
        assert not screen._category_has_unsaved_changes(category)
        assert mutations == []

        screen.query_one("#settings-provider-api-key", Input).focus()
        await pilot.pause()
        await pilot.press(_CLEAR_KEY, "escape", "s")
        await _settle(host, pilot)

        assert len(mutations) == 1
        assert "api_key" in mutations[0][1]["api_settings.openai"]
        assert "api_key" not in app.app_config["api_settings"]["openai"]


@pytest.mark.asyncio
@private_profile_test
async def test_an_emptied_api_key_field_keeps_the_saved_key_and_only_clear_removes_it(
    request, monkeypatch
):
    """Final review finding 5: typing a character and deleting it is no edit
    (Web Search's rule); it used to stage removal, so s deleted the key."""
    from Tests.UI.test_settings_configuration_hub import (
        _capture_provider_settings_mutations,
    )
    from Tests.UI.test_settings_provider_keyboard_journeys import _settle

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mutations = _capture_provider_settings_mutations(monkeypatch)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": dict(_SAVED_KEY)}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        clear = screen.query_one("#settings-provider-api-key-clear", Button)
        screen.query_one("#settings-provider-api-key", Input).focus()
        await pilot.pause()
        await pilot.press("x")
        await pilot.pause()
        assert _text(screen, "#settings-provider-key-status") == "edited *"

        await pilot.press("backspace")
        await pilot.pause()
        assert _text(screen, "#settings-provider-key-status") == "saved in config"
        assert not screen._settings_drafts
        assert not clear.disabled
        await pilot.press("escape", "s")
        await _settle(host, pilot)
        assert mutations == []
        assert app.app_config["api_settings"]["openai"]["api_key"] == _FAKE_KEY

        screen.query_one("#settings-provider-api-key", Input).focus()
        await pilot.pause()
        await pilot.press(_CLEAR_KEY)
        await pilot.pause()
        assert _text(screen, "#settings-provider-key-status") == "cleared *"

        # Checkpoint review: typing after Clear and deleting it again returns
        # to Clear's removal; the emptied-field rule must not undo a Clear.
        await pilot.press("x")
        await pilot.pause()
        assert _text(screen, "#settings-provider-key-status") == "edited *"
        await pilot.press("backspace")
        await pilot.pause()
        assert _text(screen, "#settings-provider-key-status") == "cleared *"


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("provider", ["azure", "cloudflare", "databricks"])
@pytest.mark.parametrize("where", ["config", "env"])
async def test_a_key_reads_saved_before_its_base_url_is_set(
    request, monkeypatch, provider, where
):
    """Final review finding 2 (AC#2, AC#4): readiness drops the key's source
    for these three until a base URL is set, but the key is there all the
    same -- the row says where, Clear clears a saved one, and the provider
    leads the list as Configured."""
    from tldw_chatbook.Chat.provider_readiness import default_api_key_env_var

    env_var = default_api_key_env_var(provider)
    assert env_var
    _no_provider_key_env(monkeypatch)
    if where == "env":
        monkeypatch.setenv(env_var, _FAKE_KEY)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": provider, "model": "m-1"}
    app.app_config["api_settings"] = {
        provider: dict(_SAVED_KEY) if where == "config" else {}
    }
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        clear = screen.query_one("#settings-provider-api-key-clear", Button)

        assert _text(screen, "#settings-provider-key-status") == (
            "saved in config" if where == "config" else "from env var"
        )
        assert clear.disabled is (where != "config")
        configured = next(
            group
            for group in screen._provider_picker_groups()
            if group.group_id == "configured"
        )
        assert [option.provider_id for option in configured.options] == [provider]
        assert _resting_help_counts(
            _text(screen, "#settings-provider-search-status"), 1
        )


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    ("provider", "target"),
    [
        ("azure", "resource host"),
        ("cloudflare", "account URL"),
        ("databricks", "workspace host"),
    ],
)
async def test_an_endpoint_with_no_shipped_default_reads_required(
    request, monkeypatch, provider, target
):
    """Capture 01c (fix 1): with no base URL the Key check reads "Not ready",
    yet the Endpoint row said built-in and "blank uses the provider default".
    These three ship no default, so the row says the URL is required.

    Capture round 2, finding 9: the Key check said "no URL" where the row
    asks for "your resource host"; it names the row to fill, "no endpoint".
    """
    from tldw_chatbook.Chat.provider_readiness import default_api_key_env_var

    monkeypatch.delenv(default_api_key_env_var(provider), raising=False)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": provider, "model": "m-1"}
    app.app_config["api_settings"] = {provider: dict(_SAVED_KEY)}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        help_line = f"required: your {target}"

        assert endpoint.value == ""
        assert _text(screen, "#settings-provider-readiness") == (
            "Not ready · no endpoint"
        )
        assert _text(screen, "#settings-provider-endpoint-source") == "not set"
        assert _text(screen, "#settings-provider-endpoint-help") == help_line
        assert endpoint.placeholder == f"Enter your {target}"
        painted = _region_rows(
            screen, screen.query_one("#settings-provider-endpoint-row")
        )
        assert help_line in painted[0] and endpoint.placeholder in painted[0], painted


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    ("provider", "settings", "required"),
    [
        ("azure", dict(_SAVED_KEY), "your resource host"),
        ("llama_cpp", {}, "the server's base URL"),
        ("openai", dict(_SAVED_KEY), None),
    ],
)
async def test_the_focused_endpoint_guide_says_required_as_its_row_does(
    request, provider, settings, required
):
    """TASK-33007 follow-up to fix 1: the Endpoint row reads "required: ..."
    for providers that ship no URL, yet its Inspector guide still said the
    address applies "when set". It says required, in the row's words; a
    provider with a default URL keeps "when set"."""
    from tldw_chatbook.Chat.console_provider_support import MODEL_CONFIG_FIELDS

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": provider, "model": "m-1"}
    app.app_config["api_settings"] = {provider: settings}
    host = _SettingsCssHarness(app, "settings")
    address = MODEL_CONFIG_FIELDS["endpoint"].valid_range

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        screen.query_one("#settings-provider-endpoint-value", Input).focus()
        await pilot.pause()

        assert _text(screen, "#settings-provider-field-guide-0") == (
            "Focused setting: Endpoint"
        )
        guide = _squeezed(_text(screen, "#settings-provider-field-guide-3"))
        if required is None:
            assert guide == _squeezed(f"Validation: {address} when set")
            return
        assert _text(screen, "#settings-provider-endpoint-help") == (
            f"required: {required}"
        )
        assert guide == _squeezed(f"Validation: {address}; required: {required}")


@pytest.mark.asyncio
@private_profile_test
async def test_the_card_result_line_takes_no_row_until_there_is_a_result(
    request, monkeypatch
):
    """Checkpoint review (captures): every card opened with "… have not been
    saved this session." -- a row saying nothing happened, while the Inspector
    already says "Save (s) — no changes". The line shows only once a save (s)
    has something to report.

    Capture findings 4 and 5, rewritten on purpose: a revert reported itself
    twice, on this line and in a toast, and its line stayed beside the next
    edit's "1 unsaved". The toast is the one report now, and the next edit
    clears a save's line too."""
    from Tests.UI.test_settings_configuration_hub import (
        _capture_provider_settings_mutations,
    )
    from Tests.UI.test_settings_provider_keyboard_journeys import _revert

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mutations = _capture_provider_settings_mutations(monkeypatch)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": dict(_SAVED_KEY)}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        toasts: list[str] = []
        notify = host.notify
        host.notify = lambda message, **kwargs: (
            toasts.append(str(message)),
            notify(message, **kwargs),
        )
        result = screen.query_one("#settings-provider-save-result", Static)
        assert result.display is False
        assert result.region.height == 0

        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        endpoint.focus()
        await pilot.pause()
        await pilot.press(*"https://first.invalid/v1")
        await pilot.pause()
        assert result.display is False

        await _revert(host, pilot, discard=True)
        await _until(pilot, lambda: any("reverted" in toast for toast in toasts))
        assert [toast for toast in toasts if "reverted" in toast] == [
            "Settings category changes reverted."
        ]
        assert result.display is False and result.region.height == 0
        assert mutations == []

        endpoint.focus()
        await pilot.pause()
        await pilot.press(*"https://second.invalid/v1", "escape", "s")
        await _until(pilot, lambda: len(mutations) == 1)
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert len(mutations) == 1
        assert result.display is True and result.region.height >= 1
        assert _text(screen, "#settings-provider-save-result").startswith(
            "Provider settings saved"
        )

        # The next edit makes the saved report stale: it leaves the card.
        endpoint.focus()
        await pilot.pause()
        await pilot.press("x")
        await _until(pilot, lambda: not result.display)
        assert screen._category_has_unsaved_changes(screen._active_category_id())
        assert result.display is False and result.region.height == 0


@pytest.mark.asyncio
@private_profile_test
async def test_a_provider_that_needs_no_key_offers_no_clear_and_an_optional_env_var(
    request, monkeypatch
):
    """Capture finding 10 (captures 09b, 10a-10c): llama.cpp needs no key, yet
    its API key row showed Clear and its Env var row said "safer: a
    variable's name, never the key". Clear shows only while there is a key
    to clear, and a provider that needs no key calls its Env var optional."""
    from tldw_chatbook.Chat.provider_readiness import provider_config_key
    from tldw_chatbook.UI.Settings_Modules import providers_models_card as card

    _no_provider_key_env(monkeypatch)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": dict(_SAVED_KEY),
        "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
    }
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        clear = screen.query_one("#settings-provider-api-key-clear", Button)
        row = screen.query_one("#settings-provider-api-key-row")
        guidance = "#settings-provider-credential-guidance"
        assert clear.visible
        assert _text(screen, guidance) == card.ENV_VAR_HELP_COPY

        await _type_filter(pilot, screen, "llama")
        await pilot.press("enter")
        await _until(pilot, lambda: provider_config_key(_held(screen)) == "llama_cpp")
        await pilot.pause()

        assert _text(screen, "#settings-provider-key-status") == "not required"
        assert not clear.visible
        assert "Clear" not in _region_rows(screen, row)[0]
        assert _text(screen, guidance) == card.ENV_VAR_OPTIONAL_HELP

        # A key typed for a server started with one can be cleared again.
        screen.query_one("#settings-provider-api-key", Input).focus()
        await pilot.pause()
        await pilot.press("x")
        await _until(pilot, lambda: clear.visible)
        assert clear.visible and not clear.disabled


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    ("provider", "settings", "help_line"),
    [
        ("llama_cpp", {"api_url": "http://127.0.0.1:9099"}, "the server's base URL"),
        (
            "azure",
            {**_SAVED_KEY, "api_base_url": "https://my-resource.openai.azure.com"},
            "your resource host",
        ),
    ],
)
async def test_a_filled_required_endpoint_says_what_it_holds_not_required(
    request, provider, settings, help_line
):
    """Capture round 2, finding 7 (09b): llama.cpp's Endpoint held a URL and
    still said "required: the server's base URL", which reads as an error.
    Once the field holds an address the help says what it is; "required:"
    stays for the empty field (the test above)."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": provider, "model": "m-1"}
    app.app_config["api_settings"] = {provider: settings}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)

        assert screen.query_one("#settings-provider-endpoint-value", Input).value
        assert _text(screen, "#settings-provider-endpoint-source") == (
            "saved in config"
        )
        assert _text(screen, "#settings-provider-endpoint-help") == help_line


@pytest.mark.asyncio
@private_profile_test
async def test_the_inspector_names_the_fields_a_provider_switch_leaves_unsaved(
    request,
):
    """Capture round 2, finding 7 (09b): choosing one provider read "2
    unsaved". Both are real (Save writes the provider and the model the
    switch cleared), so the Inspector's unsaved-edits note names them. It
    sits in the scrolling body: the pinned "Unsaved changes" line keeps one
    row, so Save (s) and Revert (r) never move while you edit."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": dict(_SAVED_KEY)}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        status = "#settings-selected-category-draft-status"
        assert _text(screen, status) == "No unsaved changes"

        await _type_filter(pilot, screen, "llama")
        await pilot.press("enter")
        await _until(pilot, lambda: "llama" in _held(screen).lower())
        await pilot.pause()

        banner = _text(screen, "#settings-category-state-banner")
        assert "· 2 unsaved" in banner, banner
        assert _text(screen, status) == "Unsaved changes"
        assert _text(screen, "#settings-provider-next-chat-note") == (
            "Unsaved: Provider, Model. Edits apply only after save (s)."
        )
