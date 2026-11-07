"""The Console 'Get started' card really looks for a local server (TASK-34100.6).

Review finding entry-exit-handoff-21: the card has a detected-server path,
but live it never probed -- a server started after setup was never offered.
Root cause, measured live on 2026-10-04 with a real llama-server on :9099:
the card's discovery ran on the UI loop, where the recovery guard around
``discover_local_servers`` refused it with ``ProviderReconnectRequired``, and
the card swallowed that at debug level. The first-run wizard ran the same
scan off the loop and found the server. The card now runs it off the loop
too, uses the wizard's candidates (LM Studio's :1234 among them), offers
'Look again' instead of polling, and says 'No model connected' rather than
'no key' when a local server is the way in.

These drive the real card mount and record what the probe asked for at its
one network chokepoint (``_get_models_payload``), so nothing leaves the
process.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.private_profile import private_profile_test

from Tests.UI.test_console_local_server_probe_isolation import (
    ConsoleHarness,
    _blocked_provider_app,
    _wait_for,
)
from tldw_chatbook.Chat import local_server_discovery
from tldw_chatbook.Widgets.Console.console_setup_modal import ConsoleSetupModal

pytestmark = pytest.mark.local_server_probe


def _record_probes(monkeypatch, answering: set[int]) -> list[str]:
    seen: list[str] = []

    async def payload(_client, url, _timeout, display):
        seen.append(url)
        if any(f":{port}/" in url for port in answering):
            return ("qwen2.5-0.5b-instruct",), ""
        return None, f"No models endpoint at {display}."

    monkeypatch.setattr(local_server_discovery, "_get_models_payload", payload)
    return seen


def _modal(console) -> ConsoleSetupModal:
    return console.query_one("#console-setup-modal", ConsoleSetupModal)


@pytest.mark.asyncio
@private_profile_test
async def test_the_card_mount_probes_and_offers_the_local_server(monkeypatch, request):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen = _record_probes(monkeypatch, {9099})
    host = ConsoleHarness(_blocked_provider_app())
    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for(lambda: console._console_detected_local_server is not None, pilot)
        assert any(url.startswith("http://127.0.0.1:9099/") for url in seen)
        assert any(url.startswith("http://127.0.0.1:1234/") for url in seen), (
            "the card uses the wizard's candidates, LM Studio's :1234 among them"
        )
        modal = _modal(console)
        assert modal.detected_server_action is not None
        await pilot.pause()
        step_one = modal._card_state.steps[0]
        # Dev's capture finding 12 names the chat's provider before its word.
        assert step_one.detail.endswith("No model connected")
        assert "no key" not in step_one.detail


@pytest.mark.asyncio
@private_profile_test
async def test_look_again_finds_a_server_started_after_the_card_showed(monkeypatch, request):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    answering: set[int] = set()
    seen = _record_probes(monkeypatch, answering)
    host = ConsoleHarness(_blocked_provider_app())
    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        modal = _modal(console)
        look_again = modal.query_one("#console-setup-modal-look-again", Button)
        await _wait_for(lambda: bool(seen) and look_again.display, pilot)
        assert console._console_detected_local_server is None
        probes = len(seen)

        answering.add(9099)  # the user starts llama-server now
        look_again.press()
        await _wait_for(lambda: console._console_detected_local_server is not None, pilot)
        assert len(seen) > probes
        assert not look_again.display


@pytest.mark.asyncio
@private_profile_test
async def test_the_card_scans_off_the_ui_loop_where_the_guard_refused_it(
    monkeypatch, request
):
    """The measured root cause: on the UI loop the recovery guard around
    ``discover_local_servers`` raised ``ProviderReconnectRequired`` (live,
    2026-10-04), and the card swallowed it.

    Review round 1, TE-07: the earlier stand-in replaced the scan, guard
    included, with a fake that refused on the main thread -- it encoded the
    hypothesis instead of exercising it. This keeps the real, guarded
    ``discover_local_servers`` (recorded and called through; the network
    stays at the ``_get_models_payload`` chokepoint) and asserts every scan
    the card starts runs off the UI thread, where the guard admits it, and
    finds the server.
    """
    import threading

    from tldw_chatbook.UI.Screens import chat_screen

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _record_probes(monkeypatch, {9099})
    on_ui_thread: list[bool] = []
    real_scan = local_server_discovery.discover_local_servers

    async def recorded_scan(app_config, **kwargs):
        on_ui_thread.append(threading.current_thread() is threading.main_thread())
        return await real_scan(app_config, **kwargs)

    monkeypatch.setattr(local_server_discovery, "discover_local_servers", recorded_scan)
    # base imported it by name; head scans through the off-loop entry
    monkeypatch.setattr(chat_screen, "discover_local_servers", recorded_scan, raising=False)
    host = ConsoleHarness(_blocked_provider_app())
    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for(lambda: console._console_detected_local_server is not None, pilot)
        detected = console._console_detected_local_server
        assert detected.provider_key == "llama_cpp"
        assert detected.base_url == "http://127.0.0.1:9099"
    assert on_ui_thread, "the card scanned"
    assert not any(on_ui_thread), "every card scan ran off the UI thread"


@pytest.mark.asyncio
@private_profile_test
async def test_set_up_provider_opens_on_the_provider_whose_key_is_exported(
    monkeypatch, request
):
    """new-coverage-power-03: only ANTHROPIC_API_KEY exported -- Settings
    opens on Anthropic, never on the template OpenAI with a missing key."""
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "a" * 40)
    _record_probes(monkeypatch, set())
    host = ConsoleHarness(_blocked_provider_app())
    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for(lambda: console._console_setup_modal_blocking(), pilot)
        posted: list[object] = []
        original = console.post_message
        monkeypatch.setattr(
            console, "post_message", lambda message: posted.append(message) or original(message)
        )
        await console._open_console_provider_recovery()
        navigations = [m for m in posted if isinstance(m, NavigateToScreen)]
        assert navigations, f"no navigation posted: {posted!r}"
        assert navigations[-1].screen_context["provider"] == "anthropic"
        # Dev's open_provider_setup (TASK-33008.1) always names the model
        # slot; None routes no model, so OpenAI's is never carried over.
        assert navigations[-1].screen_context.get("model") is None
