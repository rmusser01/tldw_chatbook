"""Settings opens when the saved default provider is a custom endpoint.

TASK-33002.12: with ``[chat_defaults] provider = "custom-ep:<slug>"`` (the
ADR-146 registry), opening Settings killed the app -- readiness canonicalized
the id to ``custom_ep:<slug>``, which ``ProviderReadiness`` rejects with
``ValueError('Provider key is invalid.')``. Readiness for a registry id now
resolves through the registry, and a dangling slug reads as an honest
not-ready state.
"""

import logging
import os
from types import SimpleNamespace

import pytest
from textual.content import Content
from textual.css.query import QueryError
from textual.widgets import Button, Input, Select, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
)
from Tests.UI.test_settings_configuration_hub import (
    _click_scrolled_settings_button,
    _open_settings_category,
    _settle_settings_mount_storm,
)
from tldw_chatbook.Chat.console_session_settings import (
    build_console_settings_readiness,
    build_default_console_session_settings,
)
from tldw_chatbook.Chat.custom_endpoint_registry import entry_for
from tldw_chatbook.Chat.provider_readiness import (
    ENDPOINT_NOT_FOUND_REASON,
    get_provider_readiness,
)
from tldw_chatbook.config import save_settings_to_cli_config
from tldw_chatbook.UI.Screens.settings_screen import (
    ENDPOINT_NOT_FOUND_SETTINGS_COPY,
    PROVIDER_MANUAL_SELECT_VALUE,
    SettingsScreen,
)

_ENTRY = {
    "display_name": "GPU box",
    "family": "llama_cpp",
    "base_url": "http://192.168.1.5:8080",
    "models": ["model-a"],
}


def _registry_default_app(provider: str, display_name: str = "GPU box"):
    app = _build_test_app()
    app.app_config["custom_endpoints"] = {
        "gpu-box": {**_ENTRY, "display_name": display_name}
    }
    app.app_config["chat_defaults"] = {"provider": provider, "model": "model-a"}
    return app


def _plain(widget: Static) -> str:
    """Rendered text, after markup parsing (what the user actually sees)."""
    return str(getattr(widget.visual, "plain", widget.visual))


def _text(screen, selector: str) -> str:
    return _plain(screen.query_one(selector, Static))


def _all_static_text(screen) -> str:
    return "\n".join(_plain(w) for w in screen.query(Static))


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    "display_name",
    [
        "GPU box",
        # Round 1 review: a no-break space is not ``isprintable()`` (the
        # readiness record rejected it), and ``[/]`` is Textual markup (the
        # Overview row raised MarkupError). Both killed Settings on open.
        "GPU\u00a0box [local] [/]",
    ],
)
async def test_settings_opens_on_a_registry_default_and_names_the_entry(
    request, display_name
):
    app = _registry_default_app("custom-ep:gpu-box", display_name)
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _settle_settings_mount_storm(pilot)
        screen = _active_destination_screen(host)
        overview = _text(screen, "#settings-overview-configuration")
        assert "Status: Ready" in overview, overview
        assert f"{display_name} / model-a" in overview, overview

        await _open_settings_category(pilot, "#settings-category-providers-models")
        await _settle_settings_mount_storm(pilot)
        rendered = _all_static_text(screen)
        assert f"Provider readiness: {display_name} / model-a" in rendered, rendered


@pytest.mark.asyncio
@private_profile_test
async def test_settings_opens_on_a_dangling_registry_default_honestly(request):
    host = DestinationHarness(_registry_default_app("custom-ep:gone"), "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _settle_settings_mount_storm(pilot)
        screen = _active_destination_screen(host)
        overview = _text(screen, "#settings-overview-configuration")
        # TASK-33005 capture checkpoint (rewritten on purpose): the Overview
        # status speaks the Console's word; the pane below names the cause.
        assert "Status: Not ready · unsupported" in overview, overview

        await _open_settings_category(pilot, "#settings-category-providers-models")
        await _settle_settings_mount_storm(pilot)
        rendered = _all_static_text(screen)
        assert ENDPOINT_NOT_FOUND_SETTINGS_COPY in rendered, rendered


# --- readiness seam (no mount) -------------------------------------------


def _config(**entry_overrides):
    return {"custom_endpoints": {"gpu-box": {**_ENTRY, **entry_overrides}}}


@pytest.mark.parametrize("spelling", ["custom-ep:gpu-box", "custom_ep:gpu_box"])
def test_registry_readiness_is_the_entry_family_readiness(spelling):
    config = _config()
    config["api_settings"] = {"llama_cpp": {"api_key_env_var": "LLAMA_CPP_API_KEY"}}
    readiness = get_provider_readiness(spelling, config, environ={})
    assert readiness.ready is True
    assert readiness.provider == "GPU box"
    assert readiness.provider_key == "llama_cpp"
    assert readiness.api_key is None
    # The family slot's env var (LLAMA_CPP_API_KEY) is not the entry's.
    assert readiness.env_var is None


@pytest.mark.parametrize(
    "display_name",
    ["GPU\u00a0box", "GPU\u3000box", "Dev \U0001f469\u200d\U0001f4bb", "GPU\tbox", "GPU\u200fbox"],
)
def test_registry_readiness_survives_a_display_name_it_cannot_carry(display_name):
    """The registry accepts names ``ProviderReadiness`` rejects; label by id."""
    ready = get_provider_readiness(
        "custom-ep:gpu-box", _config(display_name=display_name), environ={}
    )
    assert (ready.ready, ready.provider) == (True, "custom-ep:gpu-box")

    blocked = get_provider_readiness(
        "custom-ep:gpu-box",
        _config(display_name=display_name, api_key_env="GPU_KEY"),
        environ={},
    )
    assert (blocked.reason, blocked.provider) == ("Missing API key", "custom-ep:gpu-box")
    assert "custom-ep:gpu-box" in blocked.user_message


def test_registry_readiness_is_ready_when_the_declared_credential_resolves():
    readiness = get_provider_readiness(
        "custom-ep:gpu-box",
        _config(api_key_env="GPU_KEY"),
        environ={"GPU_KEY": "sk-env-entry-key"},
    )
    assert readiness.ready is True


def test_registry_readiness_never_carries_the_family_slot_credential():
    """The ``[api_settings.custom]`` key belongs to the slot, not the entry."""
    config = _config(family="openai_compatible")
    config["api_settings"] = {"custom": {"api_key": "sk-slot-key-not-the-entry"}}
    assert get_provider_readiness("custom", config, environ={}).api_key
    readiness = get_provider_readiness("custom-ep:gpu-box", config, environ={})
    assert readiness.ready is True
    assert readiness.api_key is None
    assert readiness.api_key_source is None


def test_registry_readiness_blocks_an_unresolved_declared_credential():
    readiness = get_provider_readiness(
        "custom-ep:gpu-box", _config(api_key_env="GPU_KEY"), environ={}
    )
    assert readiness.ready is False
    assert readiness.reason == "Missing API key"
    assert "GPU_KEY" in readiness.user_message


def test_dangling_registry_readiness_is_an_honest_not_ready_state():
    readiness = get_provider_readiness("custom-ep:gone", _config(), environ={})
    assert readiness.ready is False
    assert readiness.reason == ENDPOINT_NOT_FOUND_REASON
    assert readiness.configuration_issue == "endpoint_missing"
    assert "custom-ep:gone" in readiness.user_message


@pytest.mark.parametrize(
    "provider", ["custom-ep:gone[/]", "custom-ep:bad\x1b[31m", "custom-ep:" + "x" * 300]
)
def test_dangling_registry_readiness_never_echoes_an_invalid_slug(provider):
    readiness = get_provider_readiness(provider, _config(), environ={})
    assert (readiness.reason, readiness.provider) == (
        ENDPOINT_NOT_FOUND_REASON,
        "Custom endpoint",
    )


def _test_rows(provider: str, config) -> tuple[dict[str, str], str]:
    screen = SettingsScreen.__new__(SettingsScreen)
    screen.app_instance = SimpleNamespace(app_config=config)
    readiness = get_provider_readiness(provider, config, environ={})
    detail, summary, _passed = screen._build_provider_readiness_findings(
        provider, "model-a", readiness, draft_endpoint="", dirty=set()
    )
    rows = {line[:12].strip(): line[12:] for line in detail.split("\n")}
    return rows, summary


@pytest.mark.parametrize("api_key_env", [None, "GPU_KEY"])
def test_settings_test_findings_render_a_markup_like_entry_name(api_key_env):
    """Round 1 review: the name is text. Phase 2's Test rows are plain
    (``markup=False``); the toast is markup. The Endpoint row names the
    entry's URL, never the family's provider default."""
    config = _config(display_name="GPU [/]", api_key_env=api_key_env)
    rows, summary = _test_rows("custom-ep:gpu-box", config)
    assert "GPU [/]" in rows["Config"], rows
    assert "GPU [/]" in Content.from_markup(summary).plain, summary
    assert rows["Endpoint"] == "http://192.168.1.5:8080", rows
    assert "API key field" not in rows["Key"], rows


def test_settings_test_rows_state_a_dangling_registry_endpoint():
    rows, _summary = _test_rows("custom-ep:gone", _config())
    assert rows["Endpoint"].startswith("not found"), rows
    # TASK-33005.3: the Readiness word leads, then the blocking fact.
    assert list(rows)[:2] == ["Readiness", "Endpoint"], rows


@pytest.mark.parametrize("spelling", ["custom-ep:gpu-box", "custom_ep:gpu_box"])
def test_settings_registry_facts_resolve_either_spelling(spelling):
    """Readiness accepts both spellings, so the Settings facts must too."""
    screen = SettingsScreen.__new__(SettingsScreen)
    screen.app_instance = SimpleNamespace(app_config=_config())
    assert screen._provider_display_name(spelling) == "GPU box"
    assert screen._provider_endpoint_row(spelling) == (
        "Endpoint key: custom_endpoints.gpu-box.base_url"
    )
    # TASK-33007.2, rewritten on purpose: the Endpoint row's help names the
    # entry's URL (the readiness block's Endpoint line is gone).
    from tldw_chatbook.UI.Settings_Modules.providers_models_card import (
        endpoint_row_copy,
    )

    assert endpoint_row_copy(screen, spelling, "") == (
        "this endpoint",
        "http://192.168.1.5:8080 · edit in Custom endpoints",
    )


@pytest.mark.parametrize(
    ("provider", "ready", "reason"),
    [
        ("OpenAI", False, "Missing API key"),
        ("llama_cpp", True, "Ready"),
        ("not-a-provider", False, "Unknown provider"),
    ],
)
def test_ordinary_provider_readiness_is_unchanged(provider, ready, reason):
    readiness = get_provider_readiness(provider, _config(), environ={})
    assert (readiness.ready, readiness.reason) == (ready, reason)


@pytest.mark.parametrize(
    ("provider", "blocked"), [("custom-ep:gpu-box", False), ("custom-ep:gone", True)]
)
def test_home_console_readiness_seam_handles_a_registry_default(provider, blocked):
    """Home's readiness (``_home_console_provider_ready``) rides this seam."""
    config = {**_config(), "chat_defaults": {"provider": provider, "model": "model-a"}}
    settings = build_default_console_session_settings(config)
    readiness = build_console_settings_readiness(
        settings, app_config=config, environ={}, background_credentials=False
    )
    assert readiness.native_send_supported is not blocked


# --- Qodo #2876 round: malformed entries degrade; read-only registry detail --
#
# F1 / rider TASK-33002.18: a hand-edited entry must degrade to an honest
# not-ready state, never a ValueError or MarkupError. F3-F5: for a registry
# default, Providers & Models states the entry's own facts (URL, credential
# source) and offers no field whose draft Save would reject.

_INVALID_ENV_NAMES = ["gpu-key", "$GPU_KEY", "GPU KEY", "\x1b[31mGPU", "G" * 129]


@pytest.mark.parametrize("api_key_env", _INVALID_ENV_NAMES)
def test_registry_readiness_names_an_invalid_api_key_env_instead_of_raising(
    api_key_env,
):
    readiness = get_provider_readiness(
        "custom-ep:gpu-box", _config(api_key_env=api_key_env), environ={}
    )
    assert (readiness.ready, readiness.reason) == (False, "Invalid provider settings")
    assert readiness.env_var is None
    assert "credential env var name is invalid" in readiness.user_message
    # The bad value may be a pasted secret: the problem is named, not echoed.
    assert api_key_env not in readiness.user_message


def test_an_invalid_api_key_env_does_not_block_a_resolving_stored_key():
    config = _config(api_key_env="gpu-key", api_key="sk-stored-entry-key-123")
    assert get_provider_readiness("custom-ep:gpu-box", config, environ={}).ready


def test_console_readiness_seam_survives_an_invalid_api_key_env():
    config = {
        **_config(api_key_env="gpu-key"),
        "chat_defaults": {"provider": "custom-ep:gpu-box", "model": "model-a"},
    }
    settings = build_default_console_session_settings(config)
    readiness = build_console_settings_readiness(
        settings, app_config=config, environ={}, background_credentials=False
    )
    assert readiness.native_send_supported is False


def test_registry_load_flags_an_invalid_api_key_env_without_logging_it(caplog):
    secret_like = "sk-pasted-into-the-name-field-123"
    with caplog.at_level(
        logging.WARNING, logger="tldw_chatbook.Chat.custom_endpoint_registry"
    ):
        entry = entry_for(_config(api_key_env=secret_like), "custom-ep:gpu-box")
    assert entry is not None  # kept, so its readiness can name the problem
    assert "gpu-box" in caplog.text and "api_key_env" in caplog.text
    assert secret_like not in caplog.text
    # The env may hold a variable by that name; the loader cannot tell.
    assert "resolves no credential" not in caplog.text
    # Every readiness read reloads the registry: warn once, not per read.
    caplog.clear()
    with caplog.at_level(
        logging.WARNING, logger="tldw_chatbook.Chat.custom_endpoint_registry"
    ):
        entry_for(_config(api_key_env=secret_like), "custom-ep:gpu-box")
    assert "api_key_env" not in caplog.text


@pytest.mark.asyncio
@private_profile_test
async def test_settings_opens_on_an_entry_with_an_invalid_api_key_env(request):
    app = _registry_default_app("custom-ep:gpu-box")
    app.app_config["custom_endpoints"]["gpu-box"]["api_key_env"] = "gpu-key"
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _settle_settings_mount_storm(pilot)
        screen = _active_destination_screen(host)
        overview = _text(screen, "#settings-overview-configuration")
        # TASK-33005 capture checkpoint (rewritten on purpose): the Console's word.
        assert "Status: Not ready · check settings" in overview, overview

        await _open_settings_category(pilot, "#settings-category-providers-models")
        await _settle_settings_mount_storm(pilot)
        # TASK-33007.2, rewritten on purpose: the Credentials status line and
        # the readiness block's key row are one API key row; its help says it.
        status = _text(screen, "#settings-provider-api-key-help")
        assert "credential env var name is invalid" in status, status
        assert _text(screen, "#settings-provider-key-status") == "this endpoint"
        assert "gpu-key" not in _all_static_text(screen)


@pytest.mark.asyncio
@private_profile_test
async def test_providers_models_opens_on_a_dangling_markup_registry_default(request):
    host = DestinationHarness(_registry_default_app("custom-ep:gone[/]"), "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _settle_settings_mount_storm(pilot)
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-providers-models")
        await _settle_settings_mount_storm(pilot)
        # Rows fold long dotted keys at separators; compare without them.
        key_row = "".join(_text(screen, "#settings-provider-endpoint-key").split())
        assert "custom_endpoints.gone[/].base_url" in key_row, key_row

        await _click_scrolled_settings_button(screen, pilot, "#settings-test-provider")
        await _settle_settings_mount_storm(pilot)
        result = _text(screen, "#settings-provider-test-result")
        assert "gone[/]" in result and "api_settings" not in result, result


_FAMILY_SLOT_SECRET = "sk-family-slot-key-999"


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    ("entry_fields", "env", "expected"),
    [
        ({"api_key_env": "GPU_KEY"}, {"GPU_KEY": "sk-entry-env-secret-123"}, "env var GPU_KEY"),
        ({"api_key": "sk-entry-stored-secret-123"}, {}, "saved in this endpoint"),
        ({}, {}, "none required"),
    ],
)
async def test_providers_models_states_the_entry_endpoint_and_credential(
    request, monkeypatch, entry_fields, env, expected
):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    app = _registry_default_app("custom-ep:gpu-box")
    app.app_config["custom_endpoints"]["gpu-box"].update(entry_fields)
    # The family slot's own key must never be reported as the entry's.
    app.app_config.setdefault("api_settings", {})["llama_cpp"] = {
        "api_key": _FAMILY_SLOT_SECRET
    }
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _settle_settings_mount_storm(pilot)
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-providers-models")
        await _settle_settings_mount_storm(pilot)
        # TASK-33007.2, rewritten on purpose: the API key and Endpoint rows
        # say the entry's credential and URL in their help lines.
        help_text = _text(screen, "#settings-provider-api-key-help")
        assert expected in help_text, help_text
        assert "http://192.168.1.5:8080" in _text(
            screen, "#settings-provider-endpoint-help"
        )
        key_row = "".join(_text(screen, "#settings-provider-endpoint-key").split())
        assert "custom_endpoints.gpu-box.base_url" in key_row, key_row

        await _click_scrolled_settings_button(screen, pilot, "#settings-test-provider")
        await _settle_settings_mount_storm(pilot)
        result = _text(screen, "#settings-provider-test-result")
        assert "http://192.168.1.5:8080" in result, result
        assert "api_settings.custom_ep" not in result, result
        assert expected in result, result
        if entry_fields:
            assert "No API key is required" not in result, result
        rendered = _all_static_text(screen)
        assert "sk-entry" not in rendered and _FAMILY_SLOT_SECRET not in rendered


_REGISTRY_LOCKED_FIELDS = (
    "#settings-model-value",
    "#settings-provider-endpoint-value",
    "#settings-provider-api-key",
    "#settings-provider-api-key-clear",
    "#settings-provider-credential-env-var",
    "#settings-model-context-window",
    "#settings-model-context-window-reset",
    "#settings-generation-defaults",
    "#settings-discover-provider-models",
)


@pytest.mark.asyncio
@private_profile_test
async def test_providers_models_locks_a_registry_default_and_links_to_its_editor(
    request,
):
    # The Custom endpoints editor reads the on-disk registry; this is the
    # private profile's per-test sandbox config, never a real one.
    assert os.environ.get("TLDW_TEST_PRIVATE_PROFILE_NODE")
    assert save_settings_to_cli_config({"custom_endpoints.gpu-box": dict(_ENTRY)})
    app = _registry_default_app("custom-ep:gpu-box")
    # A saved context-window override would otherwise enable Reset.
    app.app_config["model_capabilities"] = {
        "models": {"model-a": {"context_window": 4096}}
    }
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _settle_settings_mount_storm(pilot)
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-providers-models")
        await _settle_settings_mount_storm(pilot)
        for selector in _REGISTRY_LOCKED_FIELDS:
            assert screen.query_one(selector).disabled, selector
        edit = screen.query_one("#settings-provider-edit-custom-endpoint", Button)
        assert edit.display

        await _click_scrolled_settings_button(
            screen, pilot, "#settings-provider-edit-custom-endpoint"
        )
        await _settle_settings_mount_storm(pilot)
        url_input = screen.query_one("#settings-cep-edit-url", Input)
        assert url_input.value == "http://192.168.1.5:8080"
        assert screen.focused is url_input
        # Live 211x44 review: focus alone left the editor below the fold.
        body = screen.query_one("#settings-detail-pane-body")
        assert body.region.overlaps(url_input.region), (body.region, url_input.region)
        assert screen._provider_draft() is None

        # Choosing an ordinary provider gives the form back.
        screen._apply_provider_value_change("llama_cpp")
        await pilot.pause()
        for selector in (
            "#settings-model-value",
            "#settings-provider-endpoint-value",
            "#settings-provider-api-key",
            "#settings-provider-credential-env-var",
            "#settings-model-context-window",
            "#settings-generation-defaults",
        ):
            assert not screen.query_one(selector).disabled, selector
        assert not edit.display


def test_provider_widget_value_reads_the_saved_provider_before_the_select_mounts(
    monkeypatch,
):
    """The 0.25s subscription poll can tick between compose and mount.

    Textual applies a Select's ``value`` only in ``_on_mount``, so a composed
    Select reads NULL until then; the poll resolved that to "" and repainted
    the credential rows with the no-provider copy ("not required for this
    provider") -- flaky for the registry facts above under load.
    """
    select = Select(
        [("Manual", PROVIDER_MANUAL_SELECT_VALUE)],
        value=PROVIDER_MANUAL_SELECT_VALUE,
        allow_blank=False,
    )
    assert select.value == Select.NULL  # not mounted yet

    def query_one(selector, expect_type=None):
        if selector == "#settings-provider-value" and expect_type is Select:
            return select
        raise QueryError(selector)

    screen = SettingsScreen.__new__(SettingsScreen)
    monkeypatch.setattr(screen, "query_one", query_one)
    monkeypatch.setattr(
        screen,
        "_provider_setting_values_mapping",
        lambda: {"provider": "custom-ep:gpu-box"},
    )
    assert screen._provider_widget_value() == "custom-ep:gpu-box"


# --- Review follow-ups: malformed hand-edited defaults; Revert re-locks ------
#
# A provider id that is neither a registry id nor a valid config key
# (``provider_config_key`` keeps its ``:``) failed ``ProviderReadiness``'s key
# check and killed Settings on open. Registry ids are case-sensitive (every
# send-path lookup matches the lowercase ``custom-ep:`` prefix exactly), so
# ``CUSTOM-EP:<slug>`` is not a registry id either: all three read "Unknown
# provider".

_MALFORMED_DEFAULTS = ["custom-ep:", "CUSTOM-EP:gpu-box", "foo:bar"]


@pytest.mark.parametrize("provider", [*_MALFORMED_DEFAULTS, "x" * 200])
def test_malformed_provider_id_reads_unknown_provider(provider):
    readiness = get_provider_readiness(provider, _config(), environ={})
    assert (readiness.ready, readiness.reason) == (False, "Unknown provider")
    assert "api_settings" not in readiness.user_message


@pytest.mark.parametrize("provider", _MALFORMED_DEFAULTS)
def test_console_readiness_seam_survives_a_malformed_default(provider):
    config = {**_config(), "chat_defaults": {"provider": provider, "model": "model-a"}}
    settings = build_default_console_session_settings(config)
    readiness = build_console_settings_readiness(
        settings, app_config=config, environ={}, background_credentials=False
    )
    assert readiness.native_send_supported is False


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("provider", _MALFORMED_DEFAULTS)
async def test_settings_opens_on_a_malformed_hand_edited_default(request, provider):
    host = DestinationHarness(_registry_default_app(provider), "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _settle_settings_mount_storm(pilot)
        screen = _active_destination_screen(host)
        overview = _text(screen, "#settings-overview-configuration")
        # TASK-33005 capture checkpoint (rewritten on purpose): the Console's word.
        assert "Status: Not ready · unsupported" in overview, overview

        await _open_settings_category(pilot, "#settings-category-providers-models")
        await _settle_settings_mount_storm(pilot)
        assert host._exception is None, host._exception


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("provider", _MALFORMED_DEFAULTS)
async def test_home_readiness_computes_a_malformed_default(request, provider):
    """Home swallowed the ValueError and read "Blocked" by accident."""
    from loguru import logger

    from Tests.UI.test_home_screen import (
        HOME_MOUNT_PAUSE,
        HOME_TEST_SIZE,
        HomeHarness,
        _active_home_screen,
    )

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        host = HomeHarness(_registry_default_app(provider))
        async with host.run_test(size=HOME_TEST_SIZE) as pilot:
            await pilot.pause(HOME_MOUNT_PAUSE)
            home = _active_home_screen(host)
            # The badge's own seam; ``#home-details-body`` mounts on a timer
            # (test_home_model_badge_reports_blocked_without_credential flakes
            # on NoMatches at HEAD).
            ready = home._home_console_provider_ready(
                background_credentials=False, allow_fresh_load=False
            )
            assert ready is False
    finally:
        logger.remove(sink_id)
    failures = [m for m in messages if "readiness check failed" in m]
    assert not failures, failures


@pytest.mark.asyncio
@private_profile_test
async def test_revert_to_a_registry_default_relocks_model_discovery(request):
    """Revert restored the registry id but left llama_cpp's Discover live."""
    from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId

    host = DestinationHarness(_registry_default_app("custom-ep:gpu-box"), "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _settle_settings_mount_storm(pilot)
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-providers-models")
        await _settle_settings_mount_storm(pilot)
        buttons = [
            screen.query_one(selector, Button)
            for selector in (
                "#settings-discover-provider-models",
                "#settings-save-discovered-provider-models",
                "#settings-clear-discovered-provider-models",
            )
        ]

        screen._apply_provider_value_change("llama_cpp")
        await pilot.pause()
        # A discovery ran for llama_cpp: every discovery action is live.
        screen._model_discovery_models = (SimpleNamespace(model_id="m.gguf"),)
        screen._refresh_model_discovery_widgets()
        assert not any(button.disabled for button in buttons)

        screen._revert_category(SettingsCategoryId.PROVIDERS_MODELS)
        await _settle_settings_mount_storm(pilot)
        assert screen._provider_widget_value() == "custom-ep:gpu-box"
        assert [button.id for button in buttons if not button.disabled] == []
