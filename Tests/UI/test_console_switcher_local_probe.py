"""Switch model probes local servers when it opens (TASK-33005.5).

The real popover, the real Console probe seam and the real readiness
builder over a shared evidence owner; only the network under
``probe_settings_endpoint`` is replaced, by servers keyed on port.
"""

from __future__ import annotations

import errno
import inspect
import threading

import httpx
import pytest
from textual.widgets import Input, Static

import tldw_chatbook.UI.Screens.settings_endpoint_probe as probe_module
import tldw_chatbook.Widgets.Console.console_model_popover as popover_module
from Tests.private_profile import private_profile_test
from Tests.UI.test_console_model_switcher import (
    SwitcherHarness,
    _draft,
    _rebase,
    _settle,
    line_with,
    list_lines,
)
from tldw_chatbook.Chat.console_session_settings import (
    build_console_settings_readiness,
    build_target_default_console_session_settings,
    console_send_connection,
)
from tldw_chatbook.Chat.console_settings_apply import ConsoleSettingsOrigin
from tldw_chatbook.Chat.provider_test_evidence import (
    ProviderProbeResult,
    ProviderTestEvidenceStore,
    shared_connection_evidence,
)
from tldw_chatbook.UI.Console_Modules import connection_probe
from tldw_chatbook.UI.Console_Modules.connection_probe import (
    SWITCHER_PROBES_IN_FLIGHT,
    switcher_connection_prober,
)
from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover

pytestmark = pytest.mark.asyncio

LLAMA, OLLAMA = 9099, 11434
CONFIG = {
    "api_settings": {
        "llama_cpp": {"api_url": f"http://127.0.0.1:{LLAMA}", "model": "model-a"},
        "ollama": {"api_url": f"http://127.0.0.1:{OLLAMA}", "model": "qwen3:4b"},
        # Keyed and public: listed, never contacted (AC#4).
        "vllm": {"api_url": "http://127.0.0.1:8000", "api_key": "sk-local"},
        "koboldcpp": {"api_url": "http://8.8.8.8:5001"},
        "openai": {"api_key": "sk-cloud"},
    }
}
PROVIDERS_MODELS = {
    "llama_cpp": ["model-a"],
    "ollama": ["qwen3:4b"],
    "vllm": ["vendor/model:b"],
    "koboldcpp": ["kobold-model"],
    "openai": ["gpt-5.1"],
}


class Servers:
    """Local servers by port; every other port refuses. Records each request."""

    def __init__(self, *up: int) -> None:
        self.up = set(up)
        self.hold: threading.Event | None = None
        self.requests: list[tuple[str, int, str | None, int]] = []
        self.in_flight = self.max_in_flight = 0
        self._lock = threading.Lock()

    def handle(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.requests.append(
                (
                    request.url.host,
                    request.url.port,
                    request.headers.get("authorization"),
                    threading.get_ident(),
                )
            )
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.hold is not None:
                self.hold.wait(10)
            if request.url.port not in self.up:
                raise httpx.ConnectError("refused") from OSError(
                    errno.ECONNREFUSED, "refused"
                )
            return httpx.Response(200, json={"data": [{"id": "model-a"}]})
        finally:
            with self._lock:
                self.in_flight -= 1

    @property
    def ports(self) -> list[int]:
        return [port for _host, port, _auth, _thread in self.requests]


def _stub_network(monkeypatch, fake: Servers) -> None:
    real = probe_module.probe_settings_endpoint

    async def probe(base_url, **kwargs):
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handle)) as client:
            return await real(base_url, http_client=client, **kwargs)

    monkeypatch.setattr(probe_module, "probe_settings_endpoint", probe)


@pytest.fixture
def servers(monkeypatch) -> Servers:
    """The fake servers, with the egress policy let through.

    The policy reads config, which this suite's shared sandbox refuses
    (ADR-126); with it open, only the switcher's own filter keeps a host out
    (AC#4). The Console test at the end runs the real policy in a private
    profile.
    """

    async def allow(*_args, **_kwargs) -> None:
        return None

    fake = Servers()
    _stub_network(monkeypatch, fake)
    monkeypatch.setattr(probe_module, "check_url_or_raise_async", allow)
    yield fake
    if fake.hold is not None:
        fake.hold.set()  # Never leave a probe thread parked.


def build(
    app, *, config=CONFIG, providers_models=PROVIDERS_MODELS, prober=True
) -> ConsoleModelPopover:
    """The switcher as Alt+M opens it, minus the screen, on an Ollama chat
    (this chat's own row shows CURRENT, not a hint)."""

    def readiness(provider, model):  # ChatScreen._console_default_readiness
        return build_console_settings_readiness(
            build_target_default_console_session_settings(config, provider, model),
            app_config=config,
            connection_evidence=shared_connection_evidence(lambda: app),
        )

    switcher = ConsoleModelPopover(
        origin=ConsoleSettingsOrigin("session-a", None, 0),
        app_config=config,
        initial_draft=_draft("ollama", "qwen3:4b"),
        providers_models=providers_models,
        scope_copy="Applies to: this chat only",
        durability_copy="Temporary until this chat is promoted",
        draft_rebaser=_rebase,
        live_committer=lambda submission: None,
        default_readiness_resolver=readiness,
        setup_opener=lambda provider, model: app.setup.append((provider, model)),
        connection_prober=switcher_connection_prober(app, config) if prober else None,
    )
    return switcher


def _harness() -> SwitcherHarness:
    app = SwitcherHarness()
    app.setup = []
    return app


async def _wait(pilot, predicate) -> None:
    for _ in range(100):
        if predicate():
            return
        await pilot.pause(0.02)
    pytest.fail("never happened")


def _evidence(app, port: int):
    owner = shared_connection_evidence(lambda: app)
    provider = "llama_cpp" if port == LLAMA else "ollama"
    identity = console_send_connection(
        build_target_default_console_session_settings(CONFIG, provider, None),
        app_config=CONFIG,
    )
    return identity, owner.evidence_for(identity)


async def test_opening_probes_local_servers_without_waiting_for_them(servers):
    """AC#1, AC#6, AC#9: the list paints and takes keys while every probe is
    held; probes run off the UI thread; then a refused llama.cpp and a running
    Ollama read their words in the rows."""
    servers.up = {OLLAMA}
    servers.hold = threading.Event()
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        ui_thread = threading.get_ident()
        switcher = build(app)
        await app.push_screen(switcher)
        await _wait(pilot, lambda: len(servers.requests) >= 2)

        # Both probes are parked in the network, and the switcher is live.
        assert "Ready · not tested" in line_with(list_lines(app, switcher), "qwen3:4b")
        await pilot.press(*"qwen")
        await pilot.pause()
        assert switcher.query_one("#console-popover-find", Input).value == "qwen"
        assert switcher.highlighted_row().model == "qwen3:4b"
        await pilot.press(*["backspace"] * 4)

        servers.hold.set()
        await _settle(app, pilot)
        lines = list_lines(app, switcher)
        _identity, reached = _evidence(app, OLLAMA)
        ollama = line_with(lines, "qwen3:4b")
        assert f"Ready · reachable {reached.observed_at.astimezone():%H:%M}" in ollama
        # Mockup (a): the refused server leads NEEDS SETUP, ahead of the
        # keyless cloud rows the row cap would otherwise hide it behind.
        setup = lines.index("NEEDS SETUP · Enter opens the fix")
        llama = lines[setup + 1]
        assert "(any model)" in llama and "llama.cpp" in llama
        assert "Not ready · refused :9099" in llama
        assert "start it; rechecked on open" in llama  # AC#7

    assert sorted(servers.ports) == [LLAMA, OLLAMA]
    assert all(thread != ui_thread for *_rest, thread in servers.requests)


async def test_no_credential_and_no_cloud_or_public_host_is_ever_contacted(servers):
    """AC#4: only the two keyless local servers were asked, with no key."""
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        await app.push_screen(build(app))
        await _settle(app, pilot)

    assert sorted(servers.ports) == [LLAMA, OLLAMA]
    assert {host for host, *_rest in servers.requests} == {"127.0.0.1"}
    assert [auth for _host, _port, auth, _thread in servers.requests] == [None, None]


async def test_reopening_inside_the_window_reuses_the_result(servers, monkeypatch):
    """AC#3: two opens inside the window send one probe per endpoint; a fresh
    explicit result counts too; past the window the next open re-checks."""
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        for _open in range(2):
            switcher = build(app)
            await app.push_screen(switcher)
            await _settle(app, pilot)
            await switcher.action_dismiss_popover()
            await _settle(app, pilot)
        assert sorted(servers.ports) == [LLAMA, OLLAMA]

        monkeypatch.setattr(connection_probe, "SWITCHER_PROBE_CACHE_SECONDS", 0.0)
        await app.push_screen(build(app))
        await _settle(app, pilot)
        assert sorted(servers.ports) == [LLAMA, LLAMA, OLLAMA, OLLAMA]


async def test_a_probe_still_waiting_on_a_server_is_not_sent_again(servers):
    """AC#3: close and reopen while both probes wait on slow servers; the
    reopen sends nothing, as no result exists yet to reuse."""
    servers.hold = threading.Event()
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(app)
        await app.push_screen(switcher)
        await _wait(pilot, lambda: len(servers.requests) == 2)
        await switcher.action_dismiss_popover()
        await _wait(pilot, lambda: not switcher.is_attached)
        again = build(app)
        await app.push_screen(again)
        await _wait(pilot, lambda: bool(again._probes_offered))
        await pilot.pause(0.3)
        servers.hold.set()
        await _settle(app, pilot)

    assert sorted(servers.ports) == [LLAMA, OLLAMA]


async def test_a_fresh_explicit_test_is_not_probed_again(servers):
    """Pair 4-5 ruling: a just-settled Settings 't' or Retry is a cached result."""
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        identity, _none = _evidence(app, LLAMA)
        store = ProviderTestEvidenceStore(lambda: app)
        store.settle(store.begin(identity), ProviderProbeResult("reachable", ("model-a",)))
        await app.push_screen(build(app))
        await _settle(app, pilot)

    assert servers.ports == [OLLAMA]


async def test_one_open_keeps_a_small_fixed_number_of_probes_in_flight(servers):
    """AC#2: six local servers, at most SWITCHER_PROBES_IN_FLIGHT at a time."""
    ports = range(8101, 8107)
    config = {"api_settings": {}}
    providers_models = {}
    for port, provider in zip(
        ports, ("llama_cpp", "ollama", "vllm", "koboldcpp", "tabbyapi", "aphrodite")
    ):
        config["api_settings"][provider] = {"api_url": f"http://127.0.0.1:{port}"}
        providers_models[provider] = [f"{provider}-model"]
    servers.hold = threading.Event()
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        await app.push_screen(
            build(app, config=config, providers_models=providers_models)
        )
        await _wait(pilot, lambda: len(servers.requests) >= SWITCHER_PROBES_IN_FLIGHT)
        await pilot.pause(0.2)
        assert len(servers.requests) == SWITCHER_PROBES_IN_FLIGHT
        servers.hold.set()
        await _settle(app, pilot)

    assert sorted(servers.ports) == list(ports)
    assert servers.max_in_flight == SWITCHER_PROBES_IN_FLIGHT


async def test_enter_on_a_server_that_did_not_answer_stays_in_place(servers):
    """AC#7: the fix is outside the app, so Enter explains and navigates nowhere."""
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(app)
        await app.push_screen(switcher)
        await _settle(app, pilot)
        await pilot.press(*"model-a")
        await pilot.pause()
        row = switcher.highlighted_row()
        assert (row.kind, row.provider) == ("setup", "llama_cpp")
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen is switcher
        error = str(switcher.query_one("#console-popover-error", Static).render())
        assert "start it" in error and "rechecked" in error

    assert app.setup == []


async def test_closing_mid_probe_raises_nothing_and_a_late_result_loses(servers):
    """AC#8: Esc while a probe is parked, a newer test settles, then the parked
    probe answers late: nothing raises and the newer result stands."""
    servers.hold = threading.Event()
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(app)
        await app.push_screen(switcher)
        await _wait(pilot, lambda: LLAMA in servers.ports)
        await switcher.action_dismiss_popover()
        await _wait(pilot, lambda: not switcher.is_attached)  # popped

        identity, _none = _evidence(app, LLAMA)
        store = ProviderTestEvidenceStore(lambda: app)
        store.settle(store.begin(identity), ProviderProbeResult("reachable", ("model-a",)))
        servers.hold.set()  # The parked probe now settles "refused", late.
        await _wait(pilot, lambda: servers.in_flight == 0)
        await pilot.pause(0.2)
        assert _evidence(app, LLAMA)[1].endpoint == "reachable"
    assert app.return_code in {None, 0}


async def test_the_switcher_widget_makes_no_network_call_itself(servers):
    """AC#5 (ADR-011): probes only through the injected, screen-owned seam."""
    source = inspect.getsource(popover_module)
    for name in ("httpx", "probe_settings_endpoint", "Console_Modules", "UI.Screens"):
        assert name not in source, name
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        await app.push_screen(build(app, prober=False))
        await _settle(app, pilot)
    assert servers.requests == []


@pytest.mark.asyncio
@private_profile_test
async def test_a_refused_active_llama_cpp_reaches_the_console_status_row(
    request, monkeypatch
):
    """AC#6: opened from the real Console (Alt+M), the probe of this chat's
    stopped llama.cpp turns the header status Not ready · refused :9099."""
    from Tests.UI.test_console_endpoint_discovery import _console_settled, _rail_text
    from Tests.UI.test_console_provider_apply_defaults_flow import (
        _ConsoleFlowHarness,
        _open_provider_popover,
        _persisted_console_app,
    )
    from Tests.UI.test_destination_shells import _wait_for_selector

    servers = Servers()
    _stub_network(monkeypatch, servers)
    # Tests/UI/conftest.py shuts this seam for every other Alt+M test.
    monkeypatch.setattr(
        connection_probe, "switcher_connection_prober", switcher_connection_prober
    )
    harness = _ConsoleFlowHarness(_persisted_console_app())
    async with harness.run_test(size=(211, 44)) as pilot:
        console = harness.screen
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        assert _rail_text(console, "#workbench-header-status") == "Ready · not tested"

        switcher = await _open_provider_popover(console, harness, pilot)
        await _settle(harness, pilot)
        refused = "Not ready · refused :9099"
        assert refused in line_with(list_lines(harness, switcher), "model-a")
        await switcher.action_dismiss_popover()
        await _console_settled(console, pilot, lambda r: r.blocker is not None)
        assert _rail_text(console, "#workbench-header-status") == refused
        assert _rail_text(console, "#console-model-section-recovery") == refused

    assert LLAMA in servers.ports
    assert all(auth is None for _host, _port, auth, _thread in servers.requests)


@pytest.mark.asyncio
@private_profile_test
async def test_a_probe_settling_mid_poll_still_reaches_the_console(request):
    """Switcher probes settle off the UI thread, so one can land while the
    idle poll builds readiness. The poll reads the evidence version before
    the build, so the next tick still refreshes (TASK-33005.2 review M-3)."""
    from Tests.UI.test_console_endpoint_discovery import _rail_text
    from Tests.UI.test_console_provider_apply_defaults_flow import (
        _ConsoleFlowHarness,
        _persisted_console_app,
    )
    from Tests.UI.test_destination_shells import _wait_for_selector

    harness = _ConsoleFlowHarness(_persisted_console_app())
    async with harness.run_test(size=(211, 44)) as pilot:
        console = harness.screen
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        console._stop_console_credential_poll_timer()
        console._poll_console_credential_readiness()
        settings, _readiness = console._active_console_settings_readiness()
        identity = console_send_connection(
            settings, app_config=console._provider_readiness_app_config()
        )
        build = console._active_console_settings_readiness_uncached

        def build_then_settle():
            built = build()  # The poll's readiness, from before the result.
            store = ProviderTestEvidenceStore(lambda: harness)
            store.settle(
                store.begin(identity),
                ProviderProbeResult("unreachable", (), "connection_refused"),
            )
            return built

        console._active_console_settings_readiness_uncached = build_then_settle
        console._poll_console_credential_readiness()
        console._active_console_settings_readiness_uncached = build
        console._poll_console_credential_readiness()  # The next tick.
        await pilot.pause()

        assert _rail_text(console, "#workbench-header-status") == (
            "Not ready · refused :9099"
        )
