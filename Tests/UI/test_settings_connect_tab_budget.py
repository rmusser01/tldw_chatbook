"""Settings ▸ Providers & Models ▸ Connect: the Tab budget to Model (TASK-33007.2).

Split out of test_settings_connect_rows.py so the UI Fast Lane's round-robin
shards can spread the Connect cases: each mounts the whole Settings screen
(~16 s a case on CI), and one file of all of them outgrew a 20-min shard.
Parent AC#2: Model is at most five Tab presses from Provider, for every cloud
provider, saved key or not, and while a restored connection awaits review.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, Select

from Tests.private_profile import private_profile_test
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_connect_rows import (
    _CLEAR_HINT,
    _CLEAR_KEY,
    _SAVED_KEY,
    _SIZE,
    _anthropic_host,
    _open_providers,
    _text,
    _type_filter,
    _until,
)
from Tests.UI.test_settings_narrow_layout import _region_rows, _SettingsCssHarness

# Parent AC#2. Owner ruling 2026-10-04: Clear left the Tab chain (it is a key
# on the API key field), so a saved key adds no stop. These three providers
# each carry one conditional Connect stop and are pinned at their exact
# chains: a sixth stop, or a lost one, turns them red.
_CONNECT_STOPS = (
    "settings-provider-api-key",
    "settings-provider-credential-env-var",
    "settings-provider-endpoint-value",
)

_ANTHROPIC_STOPS = ("settings-provider-auth-source", *_CONNECT_STOPS)

#: Rewritten on purpose (TASK-33007 capture fix 3): "Review restored OpenAI
#: connection" is a stop only while a restored connection awaits review, and
#: these profiles restored nothing -- see
#: test_restored_openai_connection_row_shows_only_while_a_review_awaits.
_OPENAI_STOPS = _CONNECT_STOPS

_OPENAI_REVIEW_STOPS = (*_CONNECT_STOPS, "settings-openai-reconnect-review")

_QWENCLOUD_STOPS = (*_CONNECT_STOPS, "settings-provider-api-mode")


async def _tab_stops_to_model(host, pilot, screen) -> list[str | None]:
    """Tab from the Provider control until the Model field holds focus.

    Args:
        host: The app under test.
        pilot: Its pilot.
        screen: The Settings screen showing Providers & Models.

    Returns:
        The id focused after each press; the last is the Model field's.
    """
    # TASK-33007.3: the Model stop is the Default model picker's field (the
    # Input behind it is a hidden adapter).
    model = screen.query_one("#model-search-picker-input", Input)
    screen.query_one("#settings-provider-search", Input).focus()
    await pilot.pause()
    stops: list[str | None] = []
    while host.focused is not model and len(stops) < 10:
        await pilot.press("tab")
        await pilot.pause()
        stops.append(getattr(host.focused, "id", None))
    return stops


def _focus_chain_stops_to_model(screen) -> list[str | None]:
    """Read the stops Tab makes from the Provider control to the Model field.

    Tab walks ``Screen.focus_chain`` here (Settings takes Tab over only from
    the nav bar or with nothing focused), and the six pinned cases below press
    real keys along it. Measured over the 44 Cloud providers with and without
    a saved key: this slice and ``_tab_stops_to_model`` agree in all 88.

    Args:
        screen: The Settings screen showing Providers & Models.

    Returns:
        The id of each stop after the Provider control; the last is the Model
        field's. Empty if either is off the focus chain (disabled or hidden)
        or Model does not follow Provider, so the caller records the provider
        and goes on to the next.
    """
    chain = screen.focus_chain
    provider_control = screen.query_one("#settings-provider-search", Input)
    model = screen.query_one("#model-search-picker-input", Input)
    if provider_control not in chain or model not in chain:
        return []
    start = chain.index(provider_control)
    end = chain.index(model)
    return [widget.id for widget in chain[start + 1 : end + 1]]


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    ("provider", "settings", "stops_before_model"),
    [
        pytest.param("anthropic", {}, _ANTHROPIC_STOPS, id="anthropic-no-key"),
        pytest.param("openai", {}, _OPENAI_STOPS, id="openai-no-key"),
        pytest.param("qwencloud", {}, _QWENCLOUD_STOPS, id="qwencloud-no-key"),
        pytest.param(
            "anthropic", _SAVED_KEY, _ANTHROPIC_STOPS, id="anthropic-key-saved"
        ),
        pytest.param("openai", _SAVED_KEY, _OPENAI_STOPS, id="openai-key-saved"),
        pytest.param(
            "qwencloud", _SAVED_KEY, _QWENCLOUD_STOPS, id="qwencloud-key-saved"
        ),
    ],
)
async def test_model_is_at_most_five_tab_presses_from_provider(
    request, monkeypatch, provider, settings, stops_before_model
):
    """Parent AC#2: neither Test (t) nor Clear is a Tab stop -- a key runs
    each -- so Model stays within five presses of the Provider control, with
    QwenCloud's API mode Select or Anthropic's Sign in with Select
    (TASK-34201) in the chain. OpenAI's "Review restored OpenAI connection"
    (AC#9) is in it only while a restored connection awaits review."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": provider, "model": "m-1"}
    app.app_config["api_settings"] = {provider: dict(settings)}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        test_button = screen.query_one("#settings-test-provider", Button)
        clear = screen.query_one("#settings-provider-api-key-clear", Button)

        stops = await _tab_stops_to_model(host, pilot, screen)

        assert stops == [*stops_before_model, "model-search-picker-input"]
        assert len(stops) <= 5
        assert test_button.display and not test_button.disabled
        assert clear.display and clear.disabled == (not settings)


@pytest.mark.asyncio
@private_profile_test
async def test_an_opened_provider_id_row_is_a_sixth_stop_for_anthropic(
    request, monkeypatch
):
    """Parent AC#2's budget of five is for the card at rest. "Enter provider
    ID" opens a row for typing an id no list holds, and that row is a Tab stop
    the user opened: with Anthropic's Sign in with Select the chain to Model
    is then six presses. Accepted by the controller's ruling C10 on the review
    of #3057; pinned here so a seventh stop, or a lost one, turns this red."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    host = _anthropic_host()

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        manual = screen.query_one("#settings-provider-manual-value", Input)
        assert manual not in screen.focus_chain  # at rest the row is closed

        # No provider matches, so only "Enter provider ID" is left to choose.
        await _type_filter(pilot, screen, "zzzz")
        await pilot.press("enter")
        await _until(pilot, lambda: host.focused is manual)
        assert host.focused is manual

        stops = await _tab_stops_to_model(host, pilot, screen)

        assert stops == [manual.id, *_ANTHROPIC_STOPS, "model-search-picker-input"]
        assert len(stops) == 6


def _cloud_provider_ids() -> list[str]:
    """List the providers the Settings picker groups under Cloud.

    Returns:
        The readiness key of every catalog provider that needs an API key and
        is not a custom slot or legacy alias.
    """
    from tldw_chatbook.Chat.console_session_settings import settings_provider_catalog
    from tldw_chatbook.UI.Screens.settings_provider_view_model import (
        build_provider_picker_groups,
    )

    groups = build_provider_picker_groups(settings_provider_catalog(), "", "")
    return [
        str(option.provider_id)
        for group in groups
        if group.group_id == "cloud"
        for option in group.options
    ]


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("keyed", [False, True], ids=["no-key", "key-saved"])
async def test_every_cloud_provider_reaches_model_within_five_tab_presses(
    request, keyed
):
    """Parent AC#2 over the whole catalog: every Cloud provider, with and
    without a saved key, chosen in turn on one mounted card. The stops are
    read from the focus chain; 88 walks of real presses cost about 100 s."""
    providers = _cloud_provider_ids()
    assert len(providers) >= 40, providers
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": providers[0], "model": "m-1"}
    app.app_config["api_settings"] = {
        provider: dict(_SAVED_KEY) if keyed else {} for provider in providers
    }
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        # The hidden Select is the adapter the Provider control chooses through.
        adapter = screen.query_one("#settings-provider-value", Select)
        clear = screen.query_one("#settings-provider-api-key-clear", Button)
        over_budget: dict[str, list[str | None]] = {}
        clearable: list[str] = []
        for provider in providers:
            adapter.value = screen._provider_select_value_for_provider(provider)
            # Select.Changed and the staging it drives can outlast one pause
            # under xdist load.
            for _ in range(100):
                await pilot.pause(0.05)
                if screen._provider_setting_values_mapping()["provider"] == provider:
                    break
            assert screen._provider_setting_values_mapping()["provider"] == provider
            await pilot.pause()
            if not clear.disabled:
                clearable.append(provider)
            stops = _focus_chain_stops_to_model(screen)
            if not 1 <= len(stops) <= 5 or clear.id in stops:
                over_budget[provider] = stops

        assert not over_budget, over_budget
        # The saved-key walk means something only while Clear is still live
        # after each provider switch, not just on the provider mounted first.
        # Final review finding 2, rewritten on purpose: Azure, Cloudflare and
        # Databricks used to read as keyless until their endpoint was set.
        if keyed:
            assert clearable == providers, sorted(set(providers) - set(clearable))
        else:
            assert clearable == []


@pytest.mark.asyncio
@private_profile_test
async def test_clear_is_not_a_tab_stop_stays_clickable_and_the_row_names_its_key(
    request, monkeypatch
):
    """Owner ruling 2026-10-04: like Test (t), Clear is a visible action that
    Tab skips; the API key row's help names Clear's key in full at 211x44
    (ruling C2 on PR #3057, rewritten on purpose: "t" is not a key there, and
    C5: a saved key is "hidden", not "masked")."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": dict(_SAVED_KEY)}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        api_key = screen.query_one("#settings-provider-api-key", Input)
        clear = screen.query_one("#settings-provider-api-key-clear", Button)
        row = screen.query_one("#settings-provider-api-key-row")

        assert clear.display and not clear.disabled and not clear.can_focus
        # Test carries its key in its label; Clear's is in its tooltip.
        assert f"({_CLEAR_KEY} in the API key field)" in str(clear.tooltip)
        assert _text(screen, "#settings-provider-api-key-help") == (
            f"hidden · {_CLEAR_HINT}"
        )
        assert row.region.height == 1
        assert _CLEAR_HINT in "\n".join(_region_rows(screen, row))

        api_key.focus()
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()
        assert host.focused is screen.query_one("#settings-provider-credential-env-var")

        await pilot.click("#settings-provider-api-key-clear")
        await pilot.pause()
        assert _text(screen, "#settings-provider-key-status") == "cleared *"
        assert host.focused is not clear


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    ("provider", "stops_before_model"),
    [
        ("anthropic", _ANTHROPIC_STOPS),
        ("openai", _OPENAI_STOPS),
        ("qwencloud", _QWENCLOUD_STOPS),
    ],
)
async def test_model_stays_within_five_presses_while_a_return_is_pending(
    request, monkeypatch, provider, stops_before_model
):
    """Final review finding 4 (parent AC#2): with a Chat-settings return
    pending and an unsaved key edit, Return without saving used to be a stop
    between Endpoint and Model (6 presses); it now follows Default model."""
    from Tests.UI.test_settings_configuration_hub import (
        _stage_conversation_settings_return_intent,
    )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": provider, "model": "m-1"}
    app.app_config["api_settings"] = {provider: {}}
    _intent, target = _stage_conversation_settings_return_intent(app, provider=provider)
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        screen.apply_navigation_context(target.to_context())
        await host.workers.wait_for_complete()
        await pilot.pause()
        screen.query_one("#settings-provider-api-key", Input).value = "sk-unsaved-1234"
        await pilot.pause()
        return_without_saving = screen.query_one(
            "#settings-provider-return-without-save", Button
        )
        assert return_without_saving.display and not return_without_saving.disabled

        stops = await _tab_stops_to_model(host, pilot, screen)

        assert stops == [*stops_before_model, "model-search-picker-input"]
        chain = screen.focus_chain
        model = screen.query_one("#model-search-picker-input", Input)
        assert chain.index(return_without_saving) > chain.index(model)


def _switch_provider(screen, provider_key: str) -> None:
    """Choose ``provider_key`` through the hidden Select the control drives."""
    adapter = screen.query_one("#settings-provider-value", Select)
    adapter.value = provider_key
    screen.handle_provider_value_changed(Select.Changed(adapter, provider_key))


@pytest.mark.asyncio
@private_profile_test
async def test_restored_openai_connection_row_shows_only_while_a_review_awaits(
    request, monkeypatch
):
    """Captures 01 and 02 (fix 3): the review button showed, and took a Tab
    stop, on every OpenAI profile, a fresh one included; while it had focus
    the Inspector named no setting. It is a Connect row now, shown only while
    a restored connection awaits review: read at mount, on a provider switch
    to OpenAI, and cleared once a review is recorded."""
    from types import SimpleNamespace

    from tldw_chatbook.LLM_Calls import recovery_review
    from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "m-1"}
    app.app_config["api_settings"] = {"openai": {}, "anthropic": {}}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await host.workers.wait_for_complete()
        await pilot.pause()
        row = screen.query_one("#settings-openai-reconnect-row")
        button = screen.query_one("#settings-openai-reconnect-review", Button)

        # This profile restored nothing: no row, no stop.
        assert not row.display
        assert button not in screen.focus_chain

        monkeypatch.setattr(recovery_review, "openai_reconnect_pending", lambda: True)
        _switch_provider(screen, "anthropic")
        await pilot.pause()
        _switch_provider(screen, "openai")
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert row.display and row.region.height == 1
        assert button.parent is row
        assert _text(screen, "#settings-openai-reconnect-source") == "restored"
        painted = _region_rows(screen, row)[0]
        assert "Connection" in painted and "Review" in painted, painted
        assert "requests wait until you review it" in painted, painted
        stops = await _tab_stops_to_model(host, pilot, screen)
        assert stops == [*_OPENAI_REVIEW_STOPS, "model-search-picker-input"]

        button.focus()
        await pilot.pause()
        guide = [
            _text(screen, f"#settings-provider-field-guide-{index}")
            for index in range(4)
        ]
        assert guide[0] == "Focused setting: Restored OpenAI connection"
        assert guide[1].startswith("Purpose: This profile was restored from a backup")
        assert guide[2] == "Save: applies immediately · no Save needed"

        # Enter on the focused Review button asks, and "Allow OpenAI
        # requests" records the review; only the recovery store is stubbed.
        review = SimpleNamespace(
            endpoint="https://api.openai.com/v1",
            auth_source="config",
            config_selector="api_settings.openai",
            generations=("generation-1",),
        )
        recorded: list[object] = []
        monkeypatch.setattr(recovery_review, "prepare_openai_reconnect", lambda: review)
        monkeypatch.setattr(
            recovery_review, "confirm_openai_reconnect", recorded.append
        )
        await pilot.press("enter")
        await _until(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
        assert isinstance(host.screen, ConfirmationDialog), [
            str(note.message) for note in host._notifications
        ]
        await pilot.click("#confirm-button")
        await _until(pilot, lambda: host.screen is screen and not row.display)
        assert recorded == [review]
        assert not row.display


@pytest.mark.asyncio
@private_profile_test
async def test_tab_leaves_the_model_row_after_custom_id_and_down_still_picks(request):
    """Capture round 2, finding 24 (Phase 8's item 8): the Default model
    picker opens its results list when the field takes focus, and Tab
    stopped on that list, so leaving the Model row took an extra press
    (field, Custom ID, results, ...). The list is not a Tab stop: Tab goes
    from Custom ID to the next row; Down still enters the list and Enter
    picks from it."""
    from textual.widgets import OptionList

    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": dict(_SAVED_KEY)}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        picker = screen.query_one("#settings-model-picker")
        field = picker.query_one("#model-search-picker-input", Input)
        results = picker.query_one("#model-search-picker-results", OptionList)
        field.focus()
        await _until(pilot, lambda: results.display)
        assert results.display

        await pilot.press("tab")
        await pilot.pause()
        assert getattr(host.focused, "id", None) == "model-search-picker-custom"
        # Capture fix 2, finding 3: Custom ID closes the list (so Tab skips
        # it while a click can still focus it); Shift+Tab reopens it.
        assert not results.display
        await pilot.press("shift+tab")
        await _until(pilot, lambda: results.display)
        assert host.focused is field
        await pilot.press("tab")
        await pilot.pause()
        assert getattr(host.focused, "id", None) == "model-search-picker-custom"
        await pilot.press("tab")
        await pilot.pause()
        assert host.focused is not results
        assert host.focused is not None and picker not in host.focused.ancestors

        field.focus()
        await _until(pilot, lambda: results.display)
        await pilot.press("down")
        await pilot.pause()
        assert host.focused is results
        await pilot.press("enter")
        await pilot.pause()
        assert host.focused is field and not results.has_focus
