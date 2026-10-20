"""TASK-34100.6 slice E: local detection on the Provider step.

Review findings provider-02 (a detected server was forgotten on any pick, and
Enter picked OpenAI), provider-13 (LM Studio was never looked for),
provider-14 (no server found was a dead end), provider-21 (vague found-server
copy, 'model(s)') and enhancement E3 (a server started mid-setup should
appear unprompted).

The localhost scan is replaced at its module seam
(``local_server_discovery.discover_local_servers``) with canned results; the
real probe is covered by ``Tests/Chat/test_local_server_discovery_engines.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.widgets import Button, Collapsible, Input

from Tests.Wizards.provider_step_harness import (
    SIZE,
    WizardHost,
    current_step,
    pick_provider,
    provider_list,
    resumed_on_provider,
    static_text,
    wait_for_step,
    wait_until,
)
from tldw_chatbook.Chat import local_server_discovery
from tldw_chatbook.Chat.local_server_discovery import DiscoveredLocalServer
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    ModelDiscoveryError,
    ModelDiscoveryResult,
)
from tldw_chatbook.UI.Wizards import first_run_model_discovery as model_discovery
from tldw_chatbook.UI.Wizards.first_run_setup_state import STEP_PROVIDER

pytestmark = pytest.mark.bootstrap_profile

_LLAMA = DiscoveredLocalServer("llama_cpp", "http://127.0.0.1:9099", ("a", "b", "c"))
_LM_STUDIO = DiscoveredLocalServer(
    "custom", "http://127.0.0.1:1234", ("qwen2.5-7b-instruct",), engine="LM Studio"
)


def _scan_returns(monkeypatch, *results):
    """Each scan returns the next result (the last one repeats); counts calls."""
    calls: list[int] = []

    async def scan(_app_config, **_kwargs):
        calls.append(1)
        return results[min(len(calls), len(results)) - 1]

    monkeypatch.setattr(local_server_discovery, "discover_local_servers", scan)
    return calls


def _ids(step) -> list[str]:
    choices = provider_list(step)
    return [str(choices.get_option_at_index(i).id) for i in range(choices.option_count)]


def _prompt(step, option_id: str) -> str:
    option = provider_list(step).get_option(option_id)
    return str(getattr(option.prompt, "plain", option.prompt))


def _endpoint(step) -> str:
    return step.query_one("#setup-provider-endpoint", Input).value


async def _on_provider(pilot, wizard):
    await wait_for_step(pilot, wizard, STEP_PROVIDER)
    _container, step = current_step(wizard)
    return step


@pytest.mark.asyncio
async def test_a_detected_server_is_the_first_highlighted_row_and_enter_uses_it(
    monkeypatch,
):
    """provider-02: detection is the first row, so Enter means 'use it'."""
    _scan_returns(monkeypatch, (_LLAMA,))
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        await wait_until(pilot, lambda: "found-0" in _ids(step))
        ids = _ids(step)
        assert ids[:2] == ["group-ready-on-this-machine", "found-0"]
        assert _prompt(step, "found-0") == (
            "llama.cpp on this computer · 127.0.0.1:9099 · 3 models"
        )
        choices = provider_list(step)
        assert choices.highlighted == 1
        assert step.selected_provider_key == "", "highlighting never picks"

        choices.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert step.selected_provider_key == "llama_cpp"
        assert _endpoint(step) == "http://127.0.0.1:9099"
        assert static_text(step, "#setup-provider-detected") == (
            "✓ Using the llama.cpp server found at 127.0.0.1:9099 — 3 models."
        )
        assert not step.query("#setup-provider-use-detected"), "no second affordance"


@pytest.mark.asyncio
async def test_switching_providers_never_forgets_the_detected_server(monkeypatch):
    """provider-02: pick OpenAI, then llama.cpp by hand: the found port is used."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _scan_returns(monkeypatch, (_LLAMA,))
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        await wait_until(pilot, lambda: "found-0" in _ids(step))
        await pick_provider(pilot, step, "openai")
        assert "found-0" in _ids(step)

        await pick_provider(pilot, step, "llama_cpp")
        assert _endpoint(step) == "http://127.0.0.1:9099"
        assert "found-0" in _ids(step)


@pytest.mark.asyncio
async def test_lm_studio_on_1234_sets_up_the_custom_row_named_lm_studio(monkeypatch):
    """provider-13: the row names the engine; Enter uses its port."""
    _scan_returns(monkeypatch, (_LM_STUDIO,))
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        await wait_until(pilot, lambda: "found-0" in _ids(step))
        assert _prompt(step, "found-0") == (
            "LM Studio on this computer · 127.0.0.1:1234 · 1 model"
        )
        provider_list(step).focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert step.selected_provider_key == "custom"
        assert _endpoint(step) == "http://127.0.0.1:1234"


@pytest.mark.asyncio
async def test_no_local_server_lists_the_addresses_checked_and_how_to_go_on(
    monkeypatch,
):
    """provider-14: what was checked, Check again, and a collapsed how-to."""
    calls = _scan_returns(monkeypatch, ())
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        block = step.query_one("#setup-provider-no-server")
        await wait_until(pilot, lambda: block.display)
        text = static_text(step, "#setup-provider-no-server-text")
        assert text.startswith("Looked on this computer at 127.0.0.1:8080, :9099, :11434")
        assert ":1234" in text and ":1337" in text
        assert text.endswith("— nothing answered.")
        help_box = step.query_one("#setup-provider-no-server-help", Collapsible)
        assert help_box.title == "Don't have a local server yet?"
        assert help_box.collapsed
        help_text = " ".join(
            str(node.render()) for node in help_box.query(".setup-no-server-line")
        )
        assert "User Guide" in help_text
        assert "Set it up now, start it later" in help_text
        # E3 / AC#30; review round 1, TE-11(d): the copyable pull command.
        assert "Popular models to run: llama3.2 (ollama pull llama3.2)." in help_text

        await pilot.click("#setup-provider-check-again")
        await wait_until(pilot, lambda: len(calls) >= 2)


@pytest.mark.asyncio
async def test_a_server_started_mid_setup_appears_without_asking(monkeypatch):
    """E3: a slow loopback re-check while the step is visible, stopped on leave."""
    calls = _scan_returns(monkeypatch, (), (_LLAMA,))
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        await wait_until(pilot, lambda: step.query_one("#setup-provider-no-server").display)
        assert step._recheck_timer is not None
        step._quiet_recheck()  # what the timer does every RECHECK_SECONDS
        await wait_until(pilot, lambda: "found-0" in _ids(step))
        assert len(calls) == 2
        assert not step.query_one("#setup-provider-no-server").display
        step.on_hide()
        assert step._recheck_timer is None


@pytest.mark.asyncio
async def test_a_server_started_mid_setup_appears_beside_ones_already_found(monkeypatch):
    """Review v3r1, G6-BV3-04 (live: a llama-server started on :1337 while
    :9099 and :1234 were already listed never appeared until Check again): the
    quiet re-check ran only while nothing had been found. It now runs while
    the step shows, and adds what it finds -- a scan that misses a server
    already listed (a slow answer under load) never drops its row."""
    jan = DiscoveredLocalServer("custom", "http://127.0.0.1:1337", ("m",), engine="Jan")
    calls = _scan_returns(monkeypatch, (_LLAMA,), (_LLAMA, jan), ())
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        await wait_until(pilot, lambda: "found-0" in _ids(step))
        assert "found-1" not in _ids(step)
        step._quiet_recheck()  # what the timer does every RECHECK_SECONDS
        await wait_until(pilot, lambda: "found-1" in _ids(step))
        assert len(calls) == 2
        assert "Jan on this computer · 127.0.0.1:1337" in _prompt(step, "found-1")
        step._quiet_recheck()  # this scan answers nothing
        await wait_until(pilot, lambda: len(calls) == 3)
        await pilot.pause()
        assert {"found-0", "found-1"} <= set(_ids(step)), "a quiet scan only adds"


@pytest.mark.asyncio
async def test_nothing_answering_names_the_address_and_the_engine(monkeypatch):
    """provider-14: the selected engine's own reason, never a bare 'couldn't'."""

    async def refused(**kwargs):
        return ModelDiscoveryResult(
            provider=kwargs.get("provider", "ollama"),
            provider_list_key=None,
            endpoint_fingerprint=None,
            status="error",
            error=ModelDiscoveryError(
                kind="request_failed",
                message="Model discovery request failed.",
                recovery_hint="",
                category="connection_refused",
            ),
        )

    _scan_returns(monkeypatch, ())
    wizard = resumed_on_provider()
    wizard.app_instance.llm_provider_catalog_scope_service = SimpleNamespace(
        discover_models=refused
    )
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        await pick_provider(pilot, step, "ollama")
        status = "#setup-provider-probe-status"
        await wait_until(pilot, lambda: "Nothing is answering" in static_text(step, status))
        # The shipped template's Ollama row is http://localhost:11434/v1/...;
        # the line names the server's address, not its chat route.
        assert static_text(step, status) == (
            "! Nothing is answering at http://localhost:11434 — start Ollama, "
            "then Check again."
        )
        assert step.query_one(status).has_class("-warn")  # provider-16
        assert step.query_one("#setup-provider-check-again", Button).display


def test_the_model_step_says_the_certificate_is_not_trusted():
    """coverage-19 on Model: the server answered; Settings ▸ Network fixes it."""
    result = ModelDiscoveryResult(
        provider="custom",
        provider_list_key="custom",
        endpoint_fingerprint=None,
        status="error",
        error=ModelDiscoveryError(
            kind="request_failed",
            message="x",
            recovery_hint="y",
            category="tls_untrusted",
        ),
    )
    _models, state, category = model_discovery._model_discovery_ui_outcome(result)
    assert state == "connection_failed"
    text = model_discovery.connection_failed_text(
        "custom", category, "https://127.0.0.1:8443/v1/chat/completions"
    )
    # Live finding: the one-line row was cut off before the fix it names.
    assert text == (
        "https://127.0.0.1:8443 answered, but its certificate isn't trusted — "
        "add your CA in Settings ▸ Network."
    )
    plain = model_discovery.connection_failed_text("custom", "plain HTTP", "")
    assert "http:// instead of https://" in plain
    assert model_discovery.connection_failed_text("ollama", "request failed", "") == (
        "Ollama isn't running. Start it (ollama serve), then Retry — or enter a "
        "model ID below."
    )


@pytest.mark.asyncio
async def test_use_a_popular_model_now_and_start_the_server_later(monkeypatch):
    """E3 / provider-14: pick the engine now; Model starts with the model."""
    from tldw_chatbook.UI.Wizards.first_run_setup_state import STEP_MODEL

    _scan_returns(monkeypatch, ())
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        await wait_until(pilot, lambda: step.query_one("#setup-provider-no-server").display)
        step.query_one("#setup-provider-no-server-help", Collapsible).collapsed = False
        await pilot.pause()
        later = step.query_one("#setup-provider-use-later", Button)
        assert str(later.label) == "Use llama3.2 with Ollama — I'll start it later"
        later.press()
        await pilot.pause()
        assert step.selected_provider_key == "ollama"
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        _c, model = current_step(wizard)
        assert model.query_one("#setup-model-custom", Input).value == "llama3.2"


@pytest.mark.asyncio
async def test_start_later_says_what_it_set_when_ollama_is_already_picked(monkeypatch):
    """Review v3r1, G6-BV3-06 (live): with Ollama already selected, the
    start-later button only took focus -- re-picking the shown provider is a
    no-op -- so the first press looked dead until Next showed llama3.2."""
    _scan_returns(monkeypatch, ())
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        await pick_provider(pilot, step, "ollama")
        await wait_until(pilot, lambda: step.query_one("#setup-provider-no-server").display)
        step.query_one("#setup-provider-no-server-help", Collapsible).collapsed = False
        await pilot.pause()
        step.query_one("#setup-provider-use-later", Button).press()
        await pilot.pause()
        assert static_text(step, "#setup-provider-probe-status") == (
            "Ollama with llama3.2 is set — press Next, and start Ollama before "
            "your first chat."
        )


@pytest.mark.asyncio
async def test_a_step_shown_with_a_provider_already_picked_still_looks(monkeypatch):
    """Live finding: a resumed run opens Provider with its provider picked, and
    detection waited for the 15 s re-check -- the found rows came late.

    Review round 1, TE-10: resumed from a real SetupDraft whose Provider
    values name OpenAI (the path 'Continue setup?' takes), not by poking the
    step's private state after on_hide.
    """
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import FirstRunSetupWizard
    from tldw_chatbook.UI.Wizards.first_run_setup_state import (
        SETUP_DRAFT_VERSION,
        STEP_WELCOME,
        TRACK_QUICK,
        SetupDraft,
    )
    from Tests.Wizards.provider_step_harness import app_instance

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls = _scan_returns(monkeypatch, (_LLAMA,))
    draft = SetupDraft(
        version=SETUP_DRAFT_VERSION,
        track=TRACK_QUICK,
        active_step_id=STEP_PROVIDER,
        values={
            STEP_WELCOME: {"track": TRACK_QUICK},
            STEP_PROVIDER: {"provider_key": "openai", "provider_value": "openai"},
        },
    )
    wizard = FirstRunSetupWizard(app_instance(), resume_draft=draft)
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        assert step.selected_provider_key == "openai", "resumed with OpenAI picked"
        # Within a second -- the 15 s re-check never gets a chance.
        await wait_until(pilot, lambda: "found-0" in _ids(step), tries=50)
        assert len(calls) >= 1
        assert step.selected_provider_key == "openai", "the scan never re-picks"


@pytest.mark.asyncio
async def test_at_80x24_a_found_row_that_arrives_late_is_scrolled_on_screen(
    monkeypatch,
):
    """Live finding at 80x24: the list had focus, the step sat at its top, and
    the found rows arrived after -- the pre-highlighted row stayed below the
    fold. The scan is held until the step is at its top, as it was live.

    Review round 1, TE-06: at this harness's 80x24 the row was on screen
    anyway, so the test guarded nothing. The live app is shorter there (its
    'Small terminal' notice and the wrapped filter line take three rows the
    harness doesn't have), so the harness runs at 80x21: the list starts at
    the step's last row, as live. RED with both scrolls removed (the
    found-rows scroll and the browse guard); either one alone passes."""
    import threading

    release = threading.Event()

    async def held_scan(_app_config, **_kwargs):
        release.wait(5)
        return (_LLAMA,)

    monkeypatch.setattr(local_server_discovery, "discover_local_servers", held_scan)
    from Tests.Wizards.provider_step_harness import app_instance
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import FirstRunSetupWizard
    from tldw_chatbook.UI.Wizards.first_run_setup_state import STEP_WELCOME

    wizard = FirstRunSetupWizard(app_instance())  # the live path: Welcome, Enter
    app = WizardHost(wizard)
    async with app.run_test(size=(80, 21)) as pilot:
        await wait_for_step(pilot, wizard, STEP_WELCOME)
        await pilot.press("enter")
        step = await _on_provider(pilot, wizard)
        choices = provider_list(step)
        await pilot.pause(0.3)
        # Review round 1, TE-06: live, the step sat at its top with the list
        # focused when the rows arrived. Pilot's focus scroll had already
        # moved the step, so the test never reproduced it; put it there.
        assert app.focused is choices
        step.scroll_home(animate=False, immediate=True)
        await pilot.pause()
        release.set()
        await wait_until(pilot, lambda: "found-0" in _ids(step))
        await pilot.pause(0.3)
        row = choices.region.y + 1 + 1  # border, then the Ready heading
        assert step.region.y <= row < step.region.bottom, (
            f"the found row (y={row}) is outside the step viewport {step.region}"
        )


@pytest.mark.asyncio
async def test_typing_another_address_drops_the_using_line(monkeypatch):
    """Live finding: 'Using the LM Studio server found at …:1234' stayed after
    the address was edited to another server (cross-cutting-07)."""
    _scan_returns(monkeypatch, (_LM_STUDIO,))
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        await wait_until(pilot, lambda: "found-0" in _ids(step))
        provider_list(step).focus()
        await pilot.press("enter")
        await pilot.pause()
        banner = step.query_one("#setup-provider-detected")
        assert "Using the LM Studio server" in str(banner.render())
        endpoint = step.query_one("#setup-provider-endpoint", Input)
        endpoint.focus()
        await pilot.press("end", "backspace")
        await pilot.pause()
        assert str(banner.render()) == "" and banner.has_class("hidden")


@pytest.mark.asyncio
async def test_a_hand_pick_keeps_a_user_configured_endpoint(monkeypatch):
    """Review round 1, F3: on a rerun with llama.cpp configured at a LAN
    address, hand-picking llama.cpp replaced it with the loopback server the
    scan found (and Next would have saved that over the user's config). Only
    the template's own default, or the found server itself, gives way."""
    _scan_returns(monkeypatch, (_LLAMA,))
    wizard = resumed_on_provider(
        lambda config: config.setdefault("api_settings", {})
        .setdefault("llama_cpp", {})
        .update(api_url="http://192.168.1.50:8080")
    )
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        await wait_until(pilot, lambda: "found-0" in _ids(step))
        await pick_provider(pilot, step, "llama_cpp")
        assert _endpoint(step) == "http://192.168.1.50:8080"
        assert step.query_one("#setup-provider-detected").has_class("hidden")
        draft = step._effective_provider_draft()
        assert "192.168.1.50" in draft.endpoint

        # The found row itself still means 'use the server that was found'.
        provider_list(step).focus()
        provider_list(step).highlighted = _ids(step).index("found-0")
        await pilot.press("enter")
        await pilot.pause()
        assert _endpoint(step) == "http://127.0.0.1:9099"


@pytest.mark.asyncio
async def test_a_quiet_rescan_never_reveals_the_old_detected_list(monkeypatch):
    """Review round 1, F7: the 15 s re-check (and on_show) un-hid the old
    'Detected endpoints / No local endpoints found.' list under a local row's
    endpoint, beside the new 'Looked on this computer … nothing answered'."""
    calls = _scan_returns(monkeypatch, ())
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        await pick_provider(pilot, step, "ollama")
        results = step.query_one("#setup-provider-detection-results")
        await wait_until(pilot, lambda: len(calls) >= 1)
        before = len(calls)
        step._quiet_recheck()
        await wait_until(pilot, lambda: len(calls) > before)
        await pilot.pause()
        assert results.has_class("hidden")
        assert step.query_one("#setup-provider-no-server").display

        # 'Find local servers' is the one action that asks for that list.
        step.query_one("#setup-provider-detect", Button).press()
        await wait_until(pilot, lambda: not results.has_class("hidden"))



@pytest.mark.asyncio
async def test_two_spellings_of_one_port_are_one_found_row(monkeypatch):
    """provider-22 / AC#25; review round 1, TE-08: the wizard's own merge
    (one_row_per_port) had no test -- the stubbed scans were already unique.
    The spelling that answered first is the one kept."""
    localhost = DiscoveredLocalServer("llama_cpp", "http://localhost:9099", ("a",))
    _scan_returns(monkeypatch, (_LLAMA, localhost))
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        await wait_until(pilot, lambda: "found-0" in _ids(step))
        found = [option_id for option_id in _ids(step) if option_id.startswith("found-")]
        assert found == ["found-0"]
        assert "127.0.0.1:9099" in _prompt(step, "found-0")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("category", "help_shown"),
    [("tls_plain_http", False), ("tls_untrusted", False), ("connection_refused", True)],
)
async def test_the_no_server_help_shows_only_when_nothing_answered(
    monkeypatch, category, help_shown
):
    """Review round 2, G6-VR2-09: when the selected local server answered but
    failed (plain HTTP on an https:// address, an untrusted certificate),
    'Don't have a local server yet?' still showed beside 'This server speaks
    plain HTTP'. Check again stays (it re-checks after the fix); the how-to
    is for nothing answering."""

    async def failed(**kwargs):
        return ModelDiscoveryResult(
            provider=kwargs.get("provider", "custom"),
            provider_list_key=None,
            endpoint_fingerprint=None,
            status="error",
            error=ModelDiscoveryError(
                kind="request_failed", message="x", recovery_hint="", category=category
            ),
        )

    _scan_returns(monkeypatch, ())
    wizard = resumed_on_provider(
        lambda config: config.setdefault("api_settings", {})
        .setdefault("custom", {})
        .update(api_url="https://127.0.0.1:9099/v1/chat/completions")
    )
    wizard.app_instance.llm_provider_catalog_scope_service = SimpleNamespace(
        discover_models=failed
    )
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        await pick_provider(pilot, step, "custom")
        status = "#setup-provider-probe-status"
        await wait_until(pilot, lambda: static_text(step, status).startswith("!"))
        await pilot.pause()
        assert step.query_one("#setup-provider-no-server").display  # Check again
        how_to = step.query_one("#setup-provider-no-server-help", Collapsible)
        assert how_to.display is help_shown
        if not help_shown:
            text = step.query_one("#setup-provider-no-server-text")
            assert not text.display, "'nothing answered' beside a server that answered"


async def _plain_http_after_a_request_line(reader, writer) -> None:
    """llama-server's shape: nothing is sent until a full request arrives, so
    an https ClientHello stalls the handshake while plain HTTP is answered."""
    import asyncio

    try:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
        await writer.drain()
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
        pass
    finally:
        writer.close()


@pytest.mark.loopback_network
@pytest.mark.asyncio
@pytest.mark.parametrize(("scheme", "wanted"), [
    ("https", "! This server speaks plain HTTP — use http:// instead of https://."),
    ("http", "didn't answer at http://127.0.0.1:"),
])
async def test_a_check_the_wizard_times_out_still_names_its_cause(monkeypatch, scheme, wanted):
    """Review round 2, R2-07: llama-server stalls an https handshake. httpx's
    own ConnectTimeout (10 s) -- where plain HTTP is confirmed -- never fired
    before the wizard's 8 s wait, so Provider said only 'Couldn't discover
    models'. A wizard timeout is now a typed 'timeout' result, and on an https
    loopback address one plain GET names plain HTTP. Real socket, real GET."""
    import asyncio

    monkeypatch.setattr(model_discovery, "MODEL_DISCOVERY_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(local_server_discovery, "PLAIN_HTTP_CONFIRM_TIMEOUT_SECONDS", 0.5)
    server = await asyncio.start_server(_plain_http_after_a_request_line, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    never = asyncio.Event()

    async def stalled_listing(**_kwargs):
        await never.wait()  # the handshake that never completes

    _scan_returns(monkeypatch, ())
    wizard = resumed_on_provider(
        lambda config: config.setdefault("api_settings", {})
        .setdefault("custom", {})
        .update(api_url=f"{scheme}://127.0.0.1:{port}/v1/chat/completions")
    )
    wizard.app_instance.llm_provider_catalog_scope_service = SimpleNamespace(
        discover_models=stalled_listing
    )
    app = WizardHost(wizard)
    async with server:
        async with app.run_test(size=SIZE) as pilot:
            step = await _on_provider(pilot, wizard)
            await pick_provider(pilot, step, "custom")
            status = "#setup-provider-probe-status"
            await wait_until(
                pilot, lambda: static_text(step, status).startswith("!"), tries=250
            )
            assert wanted in static_text(step, status)
            [outcome] = step._selected_provider_outcomes.values()
            assert outcome.error.category == ("tls_plain_http" if scheme == "https" else "timeout")


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_only_an_https_loopback_listing_gives_up_time_for_the_plain_http_check(
    monkeypatch,
):
    """Review v3r1, R1-08: R2-07 cut every listing's wait to leave room for the
    plain-HTTP confirmation, which only an https loopback address can use. A
    cloud listing that took longer than the cut (OpenAI or OpenRouter on a slow
    link) used to succeed inside the full wait and now ended as a 'timeout':
    Provider '!', Model 'Couldn't reach the server'."""
    import asyncio

    monkeypatch.setattr(model_discovery, "MODEL_DISCOVERY_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(local_server_discovery, "PLAIN_HTTP_CONFIRM_TIMEOUT_SECONDS", 0.5)

    async def slow_listing(**kwargs):
        await asyncio.sleep(1.4)  # past the cut wait (1.0 s), inside the full 2.0 s
        provider = kwargs["provider"]
        return ModelDiscoveryResult(
            provider=provider,
            provider_list_key=provider,
            endpoint_fingerprint=None,
            status="success",
        )

    def staged(provider: str, endpoint: str) -> dict:
        from tldw_chatbook.Chat.provider_setup_persistence import provider_endpoint_key

        return {"api_settings": {provider: {provider_endpoint_key(provider): endpoint}}}

    cloud = await model_discovery.discover_within_timeout(
        slow_listing, "openai", staged("openai", "https://api.openai.com/v1")
    )
    assert cloud.status == "success", "a cloud listing keeps the full wait"
    loopback = await model_discovery.discover_within_timeout(
        slow_listing, "custom", staged("custom", "https://127.0.0.1:9/v1/chat/completions")
    )
    assert loopback.status == "error" and loopback.error.category == "timeout"


@pytest.mark.asyncio
async def test_a_pick_before_the_first_scan_returns_keeps_what_it_finds(monkeypatch):
    """Review round 2, G6-VR2-02 (live: type 'openai', Enter, as the step
    paints): a provider switch cancelled the arrival scan and left its state
    'in_progress', so the quiet re-check never ran either. 'Ready on this
    machine' never appeared and a hand-pick of llama.cpp got the template
    :8080 -- another session's server on the dev machine. Only Back/Next
    brought the rows back."""
    import asyncio
    import threading

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    release = threading.Event()  # the scan runs on its own thread's loop
    calls: list[int] = []

    async def slow_scan(_app_config, **_kwargs):
        calls.append(1)
        while not release.is_set():
            await asyncio.sleep(0.01)
        return (_LLAMA,)

    monkeypatch.setattr(local_server_discovery, "discover_local_servers", slow_scan)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        step = await _on_provider(pilot, wizard)
        await wait_until(pilot, lambda: bool(calls))
        await pick_provider(pilot, step, "openai")
        release.set()
        await wait_until(pilot, lambda: "found-0" in _ids(step))
        assert step._local_discovery_state == "complete"
        await pick_provider(pilot, step, "llama_cpp")
        assert _endpoint(step) == "http://127.0.0.1:9099"
