"""Red regression tests for the Console provider/endpoint persistence defects.

Filed from the 2026-08-15 user report (llama.cpp custom endpoint lost on every
boot; Provider chip flipping to a provider the user never chose). Each test
pins the DESIRED behavior and is expected to fail until its task lands:

- TASK-16473: applying Console settings with a session-scoped provider
  endpoint that nothing persisted backs must warn that it will not survive a
  restart (``test_console_settings_apply_warns_when_llamacpp_endpoint_not_persisted``).
- TASK-16474: programmatic compact-bar population (mount and sidebar
  reverse-sync) must never write the provider/model mirrors
  (``test_compact_provider_mirror_untouched_by_mount_population``) and the
  bar must not preselect an arbitrary first provider when
  ``chat_defaults.provider`` is missing
  (``test_compact_bar_selects_no_arbitrary_first_provider``).
- TASK-16475: a stale-default refresh that swaps the session provider must be
  visible to the user (``test_stale_default_refresh_swap_is_visible``).
- TASK-16476: adopting a detected local server must not overwrite a
  configured endpoint (``test_detected_server_adoption_keeps_configured_endpoint``).
"""

from dataclasses import replace

import pytest

import tldw_chatbook.UI.Screens.chat_screen as chat_screen_module
import tldw_chatbook.Widgets.compact_model_bar as compact_model_bar_module
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    unsaved_console_endpoint_warning,
)
from tldw_chatbook.Chat.local_server_discovery import DiscoveredLocalServer
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)


def test_unsaved_console_endpoint_warning_cases():
    """TASK-16473 pure helper: which session endpoints are restart-backed.

    The four AC cases plus the env/override backing and the no-endpoint
    provider, against the real restart fallback chain.
    """
    empty_config = {"chat_defaults": {}, "api_settings": {"llama_cpp": {}}}
    configured_config = {
        "chat_defaults": {},
        "api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:8080"}},
    }
    override_config = {
        "chat_defaults": {},
        "api_settings": {"llama_cpp": {}},
        "console": {"llama_cpp_base_url_override": "http://192.168.1.50:8080"},
    }

    def session(url):
        return ConsoleSessionSettings(provider="llama_cpp", model="m", base_url=url)

    # Custom endpoint, nothing persisted: the trap itself.
    assert (
        unsaved_console_endpoint_warning(
            session("http://192.168.1.50:8080"), app_config=empty_config
        )
        is not None
    )
    # Custom endpoint differing from the configured one.
    assert (
        unsaved_console_endpoint_warning(
            session("http://127.0.0.1:9099"), app_config=configured_config
        )
        is not None
    )
    # Endpoint matching the configured one: backed, no warning.
    assert (
        unsaved_console_endpoint_warning(
            session("http://127.0.0.1:8080"), app_config=configured_config
        )
        is None
    )
    # Default endpoint with nothing persisted: the boot fallback reproduces
    # it, so there is nothing to warn about.
    assert (
        unsaved_console_endpoint_warning(
            session("http://127.0.0.1:9099"), app_config=empty_config
        )
        is None
    )
    # Console override backing the session endpoint.
    assert (
        unsaved_console_endpoint_warning(
            session("http://192.168.1.50:8080"), app_config=override_config
        )
        is None
    )
    # Env override backing the session endpoint.
    assert (
        unsaved_console_endpoint_warning(
            session("http://192.168.1.50:8080"),
            app_config=empty_config,
            environ={"TLDW_CONSOLE_LLAMA_CPP_BASE_URL": "http://192.168.1.50:8080"},
        )
        is None
    )
    # Provider without an endpoint: nothing to lose.
    assert (
        unsaved_console_endpoint_warning(
            ConsoleSessionSettings(provider="openai", model="gpt-4o"),
            app_config={"api_settings": {"openai": {"api_key": DUMMY_OPENAI_API_KEY}}},
        )
        is None
    )
    # The warning copy names the consequence and the persist action.
    warning = unsaved_console_endpoint_warning(
        session("http://192.168.1.50:8080"), app_config=empty_config
    )
    assert "restart" in warning and "Save as default" in warning


# Matches the repo-wide dummy convention (self-describing, not
# token-shaped) so key-scanner rules stay quiet.
DUMMY_OPENAI_API_KEY = "DUMMY_OPENAI_API_KEY"


def _capture_notifies(app, monkeypatch) -> list[tuple[str, dict]]:
    """Capture ``(message, kwargs)`` for every ``app.notify`` call."""
    captured: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        app, "notify", lambda message, **kwargs: captured.append((str(message), kwargs))
    )
    return captured


def _discard_worker(work=None, *args, **kwargs):
    """Stand in for ``run_worker`` on an unmounted screen.

    The real seam receives an already-created coroutine; close it so the
    discarded coroutine never warns.
    """
    close = getattr(work, "close", None)
    if callable(close):
        close()


def test_console_settings_apply_warns_when_llamacpp_endpoint_not_persisted(
    monkeypatch,
):
    """TASK-16473: a session-only llama.cpp endpoint must not save silently.

    The user's report: they enter their custom llama.cpp IP:Port in Console
    settings, press Save, and re-enter it on every boot. The apply path is
    session-scoped by design; the defect is that nothing tells the user their
    endpoint has no persisted backing. Desired: a warning notification on the
    apply, naming the endpoint consequence.
    """
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "m"}
    app.app_config["api_settings"] = {"llama_cpp": {}}
    console = ChatScreen(app)
    store = console._ensure_console_chat_store()
    store.ensure_session()

    notifies = _capture_notifies(app, monkeypatch)
    monkeypatch.setattr(console, "_sync_console_identity_surfaces", lambda: None)
    monkeypatch.setattr(console, "run_worker", _discard_worker)

    current = console._session._ensure_active_console_session_settings()
    session_only = replace(
        current,
        provider="llama_cpp",
        model="m",
        base_url="http://192.168.1.50:8080",
        source="user",
    )
    console._apply_console_settings_result(session_only)

    endpoint_warnings = [
        message
        for message, kwargs in notifies
        if kwargs.get("severity") == "warning" and "endpoint" in message.lower()
    ]
    assert endpoint_warnings, (
        "Applying a llama.cpp endpoint with no persisted backing must warn "
        f"that it will not survive a restart; notifies seen: {notifies}"
    )
    warning = endpoint_warnings[0].lower()
    assert "restart" in warning or "survive" in warning or "save as default" in warning


@pytest.mark.asyncio
async def test_compact_provider_mirror_untouched_by_mount_population(monkeypatch):
    """TASK-16474: mount/population must not write the provider mirror.

    ``_console_control_provider`` outranks ``chat_defaults.provider`` when
    fresh session defaults are derived, so an ambient Select.Changed from the
    compact bar's mount population silently decides the provider of the next
    session. Desired: with zero user interaction the mirror stays unset.

    Note the bar in ``ConsoleControlBar`` is LIVE UI despite its
    ``console-hidden-control`` class (a visible-text diff at 160x48 on
    2026-08-15 showed it rendering in the header), so the fix is event
    suppression, not removal.
    """
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "m1"}
    monkeypatch.setattr(
        compact_model_bar_module,
        "get_cli_providers_and_models",
        lambda: {"llama_cpp": ["m1"]},
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-compact-model-bar")
        await pilot.pause(0.05)

        assert console._console_control_provider is None, (
            "Compact-bar mount/population wrote the provider mirror "
            f"({console._console_control_provider!r}); only a user selection "
            "may set it."
        )
        assert console._console_control_model is None, (
            "Compact-bar mount/population wrote the model mirror "
            f"({console._console_control_model!r}); only a user selection "
            "may set it."
        )

        # Programmatic reverse-sync (the sidebar->bar seam) is population
        # too: it must refresh the bar's displayed values without claiming
        # them as user selections.
        console._sync_compact_shell_controls(provider="llama_cpp", model="m1")
        await pilot.pause(0.05)
        assert console._console_control_provider is None, (
            "sync_from_sidebar wrote the provider mirror "
            f"({console._console_control_provider!r}); programmatic "
            "population must not."
        )
        assert console._console_control_model is None, (
            "sync_from_sidebar wrote the model mirror "
            f"({console._console_control_model!r}); programmatic "
            "population must not."
        )


@pytest.mark.asyncio
async def test_compact_bar_selects_no_arbitrary_first_provider(monkeypatch):
    """TASK-16474: no arbitrary first-[providers]-key fallback in the bar.

    With ``chat_defaults.provider`` missing, the bar used to preselect the
    first ``[providers]`` key in file order -- a provider nobody chose. The
    select must stay on its prompt until the user picks one.
    """
    app = _build_test_app()
    app.app_config["chat_defaults"] = {}
    monkeypatch.setattr(
        compact_model_bar_module,
        "get_cli_providers_and_models",
        lambda: {"alpha": ["a1"], "beta": ["b1"]},
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-compact-model-bar")
        await pilot.pause(0.05)

        from textual.widgets import Select as TextualSelect

        provider_select = console.query_one("#compact-api-provider", TextualSelect)
        assert provider_select.value in (
            None,
            TextualSelect.BLANK,
            TextualSelect.NULL,
        ), (
            "Compact bar preselected an arbitrary provider "
            f"({provider_select.value!r}) with no chat_defaults.provider."
        )


def test_stale_default_refresh_swap_is_visible(monkeypatch):
    """TASK-16475: a stale-default refresh that changes provider must notify.

    Setup mirrors ``test_console_stale_default_refresh_respects_user_marked_
    settings``: an untouched blocked openai session converges on the
    llama.cpp chat defaults. The swap itself is task-177 behavior and stays;
    the defect is that it is invisible. Desired: a warning names the provider
    change.
    """
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "local-model"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "local-model"},
        "openai": {"api_key": ""},
    }
    notifies = _capture_notifies(app, monkeypatch)
    console = ChatScreen(app)
    store = console._ensure_console_chat_store()

    stale_derived = ConsoleSessionSettings(provider="openai", model="gpt-4o")
    store.create_session(
        settings=stale_derived,
        canonical_settings_baseline=stale_derived,
    )

    refreshed = console._session._ensure_active_console_session_settings()
    # Sanity guard against a vacuous pass: the swap itself must still happen
    # (task-177 convergence), so the notice requirement is exercised for real.
    assert refreshed.provider == "llama_cpp"

    swap_notices = [
        message
        for message, kwargs in notifies
        if kwargs.get("severity") == "warning" and "provider" in message.lower()
    ]
    assert swap_notices, (
        "The stale-default refresh replaced the session provider "
        f"(openai -> llama_cpp) without any user-visible notice; "
        f"notifies seen: {notifies}"
    )
    notice = swap_notices[0].lower()
    assert "openai" in notice and "llama_cpp" in notice


def test_detected_server_adoption_keeps_configured_endpoint(monkeypatch):
    """TASK-16476: adoption must not clobber a configured endpoint.

    The user's llama.cpp endpoint is persisted at
    ``api_settings.llama_cpp.api_url``. Adopting a detected loopback server at
    a different port must keep the configured endpoint in config (fill only
    when absent) while still applying the detected endpoint to the session so
    the adoption remains effective immediately.
    """
    configured_endpoint = "http://127.0.0.1:8080"
    detected_endpoint = "http://127.0.0.1:9099"
    app = _build_test_app()
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": configured_endpoint, "model": "user-model"}
    }
    console = ChatScreen(app)
    store = console._ensure_console_chat_store()
    store.ensure_session()

    captured_sections: dict = {}

    def fake_save(section_values):
        captured_sections.update(section_values)
        return True

    monkeypatch.setattr(
        chat_screen_module, "save_settings_to_cli_config", fake_save
    )
    monkeypatch.setattr(console, "_sync_console_transcript_guidance", lambda: None)
    monkeypatch.setattr(console, "run_worker", _discard_worker)
    console._session._sync_chat_core_state_fn = lambda: None
    console._session._sync_settings_summary_fn = lambda: None

    console._console_detected_local_server = DiscoveredLocalServer(
        provider_key="llama_cpp",
        base_url=detected_endpoint,
        model_ids=("detected-model",),
    )
    console._apply_detected_local_server()

    provider_write = captured_sections.get("api_settings.llama_cpp", {})
    assert provider_write.get("api_url") != detected_endpoint, (
        "Adoption overwrote the configured llama.cpp endpoint "
        f"({configured_endpoint}) with the detected one ({detected_endpoint})."
    )
    active_settings = store.session_settings(store.active_session_id)
    assert active_settings.base_url == detected_endpoint, (
        "Adoption must apply the detected endpoint to the active session so "
        "'Use detected ...' stays effective without the config write; got "
        f"{active_settings.base_url!r}."
    )


def test_detected_server_adoption_treats_trailing_slash_as_same_endpoint(
    monkeypatch,
):
    """TASK-16476 / Qodo review (PR #1720): identity, not raw-string, compare.

    A configured endpoint differing from the detected one only by a trailing
    slash is the same server: adoption must not warn about "keeping" it and
    the canonicalizing ``api_url`` write still happens.
    """
    detected_endpoint = "http://127.0.0.1:8080"
    app = _build_test_app()
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:8080/", "model": "user-model"}
    }
    console = ChatScreen(app)
    store = console._ensure_console_chat_store()
    store.ensure_session()

    notifies = _capture_notifies(app, monkeypatch)
    captured_sections: dict = {}

    def fake_save(section_values):
        captured_sections.update(section_values)
        return True

    monkeypatch.setattr(
        chat_screen_module, "save_settings_to_cli_config", fake_save
    )
    monkeypatch.setattr(console, "_sync_console_transcript_guidance", lambda: None)
    monkeypatch.setattr(console, "run_worker", _discard_worker)
    console._session._sync_chat_core_state_fn = lambda: None
    console._session._sync_settings_summary_fn = lambda: None

    console._console_detected_local_server = DiscoveredLocalServer(
        provider_key="llama_cpp",
        base_url=detected_endpoint,
        model_ids=("detected-model",),
    )
    console._apply_detected_local_server()

    assert not [message for message, _kwargs in notifies if "Keeping" in message], (
        f"Same-server endpoints (trailing slash) must not warn; saw: {notifies}"
    )
    provider_write = captured_sections.get("api_settings.llama_cpp", {})
    assert provider_write.get("api_url") == detected_endpoint, (
        "Same-server adoption should still write the canonical api_url; got "
        f"{provider_write.get('api_url')!r}."
    )
