"""Switch model probes local servers when it opens (TASK-33005.5).

The real popover, the real Console probe seam and the real readiness
builder over a shared evidence owner; only the network under
``probe_settings_endpoint`` is replaced, by servers keyed on port.
"""

from __future__ import annotations

import asyncio
import errno
import inspect
import threading
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from textual.widgets import Button, Input, OptionList, Static

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
from tldw_chatbook.UI.Console_Modules.model_switcher import _switcher_sources
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
        #: Port -> HTTP status a server answers its listing with (401, 403).
        self.status: dict[int, int] = {}
        #: Port -> the model ids a running server lists (default: model-a).
        self.listing: dict[int, list[str]] = {}
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
            if request.url.port in self.status:
                return httpx.Response(self.status[request.url.port])
            if request.url.port not in self.up:
                raise httpx.ConnectError("refused") from OSError(
                    errno.ECONNREFUSED, "refused"
                )
            listed = self.listing.get(request.url.port, ["model-a"])
            return httpx.Response(200, json={"data": [{"id": m} for m in listed]})
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
    app,
    *,
    config=CONFIG,
    providers_models=PROVIDERS_MODELS,
    prober=True,
    pair=("ollama", "qwen3:4b"),
    served_models=None,
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
        initial_draft=_draft(*pair),
        providers_models=providers_models,
        scope_copy="Applies to: this chat only",
        durability_copy="Temporary until this chat is promoted",
        draft_rebaser=_rebase,
        live_committer=lambda submission: None,
        default_readiness_resolver=readiness,
        setup_opener=lambda provider, model: app.setup.append((provider, model)),
        connection_prober=switcher_connection_prober(app, config) if prober else None,
        served_models=served_models,
    )
    return switcher


def _harness() -> SwitcherHarness:
    app = SwitcherHarness()
    app.setup = []
    return app


async def _wait(pilot, predicate, tries: int = 100) -> None:
    for _ in range(tries):
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
        setup = lines.index("NEEDS SETUP · Enter opens the fix or explains it")
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


async def test_one_open_on_a_moved_endpoint_keeps_one_in_flight_cap(
    servers, monkeypatch
):
    """AC#2 with TASK-35190's second prober (review round 3): a llama.cpp chat
    on :8200 probes it and the configured :8101, beside two other local
    servers; Switch model's own prober keeps the four under one cap."""
    monkeypatch.setattr(  # Tests/conftest.py shuts this seam.
        connection_probe, "switcher_connection_prober", switcher_connection_prober
    )
    config = {
        "api_settings": {
            provider: {"api_url": f"http://127.0.0.1:{port}"}
            for provider, port in (
                ("llama_cpp", 8101),
                ("ollama", 8102),
                ("koboldcpp", 8103),
            )
        }
    }
    draft = _draft()
    draft = replace(
        draft, settings=replace(draft.settings, base_url="http://127.0.0.1:8200")
    )
    servers.hold = threading.Event()
    app = _harness()
    screen = SimpleNamespace(
        app=app,
        is_attached=False,
        _ensure_console_chat_store=lambda: None,
        _providers_models=lambda: PROVIDERS_MODELS,
        _console_default_readiness=lambda *_args, **_kwargs: None,  # Never read.
        _ensure_console_chat_controller=lambda: SimpleNamespace(
            rebase_console_settings_draft=_rebase
        ),
    )
    async with app.run_test(size=(211, 44)) as pilot:
        sources = _switcher_sources(
            screen, "session-a", ("llama_cpp", "model-a"), config, draft=draft
        )
        targets = dict.fromkeys(config["api_settings"])
        probing = asyncio.ensure_future(
            sources["connection_prober"](targets, lambda _provider: None)
        )
        await _wait(pilot, lambda: len(servers.requests) >= SWITCHER_PROBES_IN_FLIGHT)
        await pilot.pause(0.2)
        assert len(servers.requests) == SWITCHER_PROBES_IN_FLIGHT
        servers.hold.set()
        await probing

    assert sorted(servers.ports) == [8101, 8102, 8103, 8200]
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
        # Round 2 finding 4: the key row says what Enter does on this row.
        apply = switcher.query_one("#console-popover-apply", Button)
        assert str(apply.label) == "Enter explain"
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen is switcher
        error = str(switcher.query_one("#console-popover-error", Static).render())
        assert "start it" in error and "rechecked" in error

    assert app.setup == []


@private_profile_test
async def test_closing_mid_probe_raises_nothing_and_a_late_result_loses(
    servers, request, monkeypatch
):
    """AC#8: Esc while a probe is parked, a newer test settles, then the parked
    probe answers late: nothing raises and the newer result stands.

    TASK-33005.10 AC#4: the probe swallows a raise (it logs and returns) and
    its worker does not exit on error, so "newer stands" also held when the
    late probe crashed before settling. The late result must be offered to
    the owner and lose there, and no probe worker may end in error."""
    from textual.worker import WorkerState

    servers.hold = threading.Event()
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        owner = shared_connection_evidence(lambda: app)
        offered: list[tuple[str, bool]] = []
        publish = owner.publish

        def recording_publish(evidence, **kwargs):
            changed = publish(evidence, **kwargs)
            if evidence.identity.provider_key == "llama_cpp":  # Ollama is probed too.
                offered.append((evidence.endpoint, changed))
            return changed

        monkeypatch.setattr(owner, "publish", recording_publish)
        switcher = build(app)
        await app.push_screen(switcher)
        await _wait(pilot, lambda: LLAMA in servers.ports)
        probes = [w for w in app.workers if w.group == "console-switcher-probes"]
        assert probes
        await switcher.action_dismiss_popover()
        await _wait(pilot, lambda: not switcher.is_attached)  # popped

        identity, _none = _evidence(app, LLAMA)
        store = ProviderTestEvidenceStore(lambda: app)
        store.settle(store.begin(identity), ProviderProbeResult("reachable", ("model-a",)))
        servers.hold.set()  # The parked probe now settles "refused", late.
        await _wait(pilot, lambda: servers.in_flight == 0)
        await _wait(pilot, lambda: ("unreachable", False) in offered)
        await pilot.pause(0.2)
        assert _evidence(app, LLAMA)[1].endpoint == "reachable"
        assert ("unreachable", True) not in offered
        assert [w.state for w in probes if w.state is WorkerState.ERROR] == []
    assert app.return_code in {None, 0}


TABBY, APHRODITE = 8080, 2242  # Their shipped localhost defaults.
SHIPPED_MODELS = {
    **PROVIDERS_MODELS,
    "tabbyapi": ["tabby-model"],
    "aphrodite": ["aphrodite-engine"],
}


def _with_shipped(*providers: str) -> dict:
    """CONFIG plus template tables exactly as every profile carries them."""
    from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML

    config = deepcopy(CONFIG)
    for provider in providers:
        config["api_settings"][provider] = deepcopy(
            DEFAULT_CONFIG_FROM_TOML["api_settings"][provider]
        )
    return config


def _group(switcher: ConsoleModelPopover, header: str) -> list:
    rows = switcher._rows
    start = next(
        (i for i, row in enumerate(rows) if row.kind == "header" and row.text.startswith(header)),
        None,
    )
    if start is None:
        return []
    end = next(
        (i for i in range(start + 1, len(rows)) if rows[i].kind == "header"), len(rows)
    )
    return rows[start + 1 : end]


async def test_an_untouched_shipped_default_that_refuses_is_quietly_not_running(
    servers,
):
    """TASK-33005.6 AC#3 (owner ruling 2026-10-02): shipped localhost defaults
    are still probed, so a server already running is found; one the user never
    set up or used that refuses is listed once as not running, never as a
    "refused :PORT" NEEDS SETUP row. A configured one keeps its row."""
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(
            app,
            config=_with_shipped("tabbyapi", "aphrodite"),
            providers_models=SHIPPED_MODELS,
        )
        await app.push_screen(switcher)
        await _settle(app, pilot)
        setup = {row.provider for row in _group(switcher, "NEEDS SETUP")}
        quiet = _group(switcher, "NOT RUNNING")
        lines = list_lines(app, switcher)

    assert {TABBY, APHRODITE} <= set(servers.ports)  # Still probed.
    assert "llama_cpp" in setup  # Edited to :9099, refused: unchanged.
    assert not {"tabbyapi", "aphrodite"} & setup, setup
    assert [row.kind for row in quiet] == ["info"], quiet
    assert "TabbyAPI" in quiet[0].text and "Aphrodite Engine" in quiet[0].text
    assert not any(f"refused :{TABBY}" in line or f"refused :{APHRODITE}" in line for line in lines)


@pytest.mark.parametrize("owned", (False, True), ids=("untouched", "set-up"))
async def test_an_untouched_mlx_lm_is_not_running_not_ready(servers, owned):
    """Capture finding 11: a plain Alt+M listed "MLX-LM (Apple silicon) · no
    models reported · type its name and an id" under READY PROVIDERS on a
    profile that never set it up. MLX-LM is a server (chat_with_mlx_lm posts
    to it), not an in-process engine, but no models route lets the switcher
    probe it, so the shipped template's endpoint and placeholder model read
    Ready. Still at the template and never used, it is listed as not running,
    as a probed shipped default that refuses is; set up, it stays READY."""
    config = _with_shipped("local_mlx_lm")
    if owned:
        config["api_settings"]["local_mlx_lm"]["model"] = "my-mlx-model"
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(
            app,
            config=config,
            providers_models={**SHIPPED_MODELS, "local_mlx_lm": ["None"]},
        )
        await app.push_screen(switcher)
        await _settle(app, pilot)
        ready = " ".join(row.text for row in _group(switcher, "READY PROVIDERS"))
        quiet = " ".join(row.text for row in _group(switcher, "NOT RUNNING"))
        placeholder = switcher.query_one("#console-popover-find", Input).placeholder
        ready_keys = [
            key
            for key in switcher._provider_order
            if popover_module._is_ready(switcher._readiness.get(key))
        ]

    assert ("MLX-LM" in ready) is owned, ready
    # Round 2 finding 3: Find counts only the providers READY PROVIDERS lists.
    assert "local_mlx_lm" in ready_keys
    listed = len(ready_keys) - (0 if owned else 1)
    noun = "provider" if listed == 1 else "providers"
    assert f" in {listed} {noun} " in placeholder, placeholder
    # Capture fix round 1: nothing checked it, so the line says so rather
    # than claim it is not running.
    assert ("MLX-LM (Apple silicon) (not checked)" in quiet) is not owned, quiet
    assert 5678 not in servers.ports  # Never contacted either way.


_UNTOUCHED = (
    *("ollama", "tabbyapi", "aphrodite", "vllm"),
    *("custom", "custom_2", "local_mlx_lm"),
)
#: Without Ollama the names make one 132-cell row in the 134 cells the list
#: has at 211, and two rows in the 114 it has at 120.
_UNTOUCHED_BUT_OLLAMA = _UNTOUCHED[1:]


@pytest.mark.parametrize(
    ("opened", "size", "shipped", "rows"),
    (
        ((211, 44), (211, 44), _UNTOUCHED, 2),
        ((120, 44), (120, 44), _UNTOUCHED, 2),
        ((211, 44), (120, 44), _UNTOUCHED_BUT_OLLAMA, 2),
        ((120, 44), (211, 44), _UNTOUCHED_BUT_OLLAMA, 1),
    ),
    ids=("211x44", "120x44", "211x44-shrunk-to-120x44", "120x44-widened-to-211x44"),
)
async def test_the_not_running_line_wraps_so_every_name_shows_whole(
    servers, opened, size, shipped, rows
):
    """Round 2 finding 2: seven untouched servers on one NOT RUNNING row were cut
    at the dialog's edge to "MLX-LM (Apple silicon) (not ch…" at both full
    sizes. The names are packed into as many rows as the list's width needs.
    Capture fix 2 round 1 finding 2: they were packed only when the rows were
    built, so a list opened at 211 and shrunk to 120 cut them again, and one
    widened kept more rows than it needed; a resize repacks them."""
    app = _harness()
    async with app.run_test(size=opened) as pilot:
        switcher = build(
            app,
            pair=("openai", "gpt-5.1"),  # 05d: an OpenAI chat
            config=_with_shipped(*shipped),
            providers_models={
                **SHIPPED_MODELS,
                **{key: [] for key in shipped if key not in SHIPPED_MODELS},
            },
        )
        await app.push_screen(switcher)
        await _settle(app, pilot)
        if size != opened:
            await pilot.resize_terminal(*size)
            await _settle(app, pilot)
        quiet = [row.text for row in _group(switcher, "NOT RUNNING")]
        lines = list_lines(app, switcher)

    joined = " · ".join(text.strip() for text in quiet)
    names = ("TabbyAPI", "vLLM", "MLX-LM (Apple silicon) (not checked)")
    for name in names + ("Ollama",) * ("ollama" in shipped):
        assert name in joined, quiet
    assert len(quiet) == rows, quiet
    for text in quiet:
        assert line_with(lines, text.strip()), text  # painted whole, no "…"
    assert not any(line.endswith("…") for line in lines), lines


async def test_the_list_draws_no_blank_row_when_a_layout_ran_between_answers(servers):
    """Capture findings 5 (1): one fresh launch of 05d's profile at 211x44 drew
    a blank row under "MLX-LM (Apple silicon) (not checked)", gone after a
    resize. Each probe answer rebuilds the rows, and a server that leaves READY
    for NOT RUNNING's packed row shortens the list by one. When another widget
    in the dialog got its layout to the screen while one answer's layout was
    still pending on the list, the list's box was measured and cached at that
    length, and the next answer's refresh(layout=True) did not invalidate it
    (Textual bumps the cache key only when no layout is pending). The rows
    were packed against the laid-out width all along. Made deterministic:
    three answers behind and laid out, one answer, a sibling's layout first,
    then the other two. (A list that grows shows its scrollbar for a frame,
    and the scrollbar's own layout refresh happens to clear the stale box:
    the shrinking steps are the case.)"""
    servers.hold = threading.Event()
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(
            app,
            pair=("openai", "gpt-5.1"),  # 05d: an OpenAI chat
            config=_with_shipped(*_UNTOUCHED),
            providers_models={
                **SHIPPED_MODELS,
                **{key: [] for key in _UNTOUCHED if key not in SHIPPED_MODELS},
            },
        )
        await app.push_screen(switcher)
        await _wait(pilot, lambda: switcher._first_readiness_done and servers.requests)
        behind = dict(switcher._readiness)  # every local server Ready, unprobed
        servers.hold.set()
        await _settle(app, pilot)
        answered = dict(switcher._readiness)
        pairs = switcher.query_one("#console-popover-pairs", OptionList)
        settled = pairs.option_count

        late = ("aphrodite", "ollama", "tabbyapi")
        switcher._readiness.update({key: behind[key] for key in late})
        switcher._rebuild_rows()  # three READY rows, four names on one row
        await _settle(app, pilot)
        assert pairs.option_count == settled + 2
        switcher._readiness["tabbyapi"] = answered["tabbyapi"]
        switcher._rebuild_rows()  # one answer: a row shorter, its layout pending
        # Another widget in the dialog reaches the screen with its layout first.
        label = switcher.query_one("#console-popover-values-label", Static)
        label.refresh(layout=True)
        label._check_refresh()
        switcher._refresh_layout()
        switcher._readiness.update(answered)
        switcher._rebuild_rows()  # the other two: a row shorter again
        await _settle(app, pilot)
        lines = list_lines(app, switcher)

    assert lines[-1] == "  MLX-LM (Apple silicon) (not checked)", lines
    assert all(line.strip() for line in lines), lines
    assert len(lines) == settled


async def test_find_counts_one_listed_provider_in_the_singular(servers):
    """Round 2 finding 3, seen live after its fix: with MLX-LM out of the count
    a profile with one cloud key read "14 models in 1 providers"."""
    from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML

    config = {
        "api_settings": {
            "openai": {"api_key": "sk-cloud"},
            "local_mlx_lm": deepcopy(
                DEFAULT_CONFIG_FROM_TOML["api_settings"]["local_mlx_lm"]
            ),
        }
    }
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(
            app,
            pair=("openai", "gpt-5.1"),
            config=config,
            providers_models={"openai": ["gpt-5.1"], "local_mlx_lm": []},
        )
        await app.push_screen(switcher)
        await _settle(app, pilot)
        placeholder = switcher.query_one("#console-popover-find", Input).placeholder

    assert placeholder.startswith("type to search 1 model in 1 provider · "), placeholder


@private_profile_test
async def test_find_filters_not_running_by_provider_name_as_every_group(
    servers, request
):
    """TASK-33005.15 AC#3: typing a provider name in Find filters the NOT
    RUNNING names as it filters the other groups: the named provider stays,
    every other name goes, and a name no server matches leaves no NOT
    RUNNING group at all, while READY PROVIDERS keeps the typed provider's
    rows only."""
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(
            app,
            config=_with_shipped("tabbyapi", "aphrodite"),
            providers_models=SHIPPED_MODELS,
        )
        await app.push_screen(switcher)
        await _settle(app, pilot)
        find = switcher.query_one("#console-popover-find", Input)

        def groups() -> tuple[str, set[str]]:
            quiet = " ".join(row.text for row in _group(switcher, "NOT RUNNING"))
            ready = {row.provider for row in _group(switcher, "READY PROVIDERS")}
            return quiet, ready - {""}

        seen: dict[str, tuple[str, set[str]]] = {"": groups()}
        for typed in ("tabby", "aphro", "openai"):
            find.value = typed
            await _settle(app, pilot)
            seen[typed] = groups()

    quiet, ready = seen[""]
    assert "TabbyAPI" in quiet and "Aphrodite Engine" in quiet, quiet
    assert {"openai", "vllm"} <= ready, ready
    quiet, _ready = seen["tabby"]
    assert "TabbyAPI" in quiet and "Aphrodite" not in quiet, quiet
    quiet, _ready = seen["aphro"]
    assert "Aphrodite Engine" in quiet and "TabbyAPI" not in quiet, quiet
    quiet, ready = seen["openai"]
    assert quiet == "", quiet
    assert ready == {"openai"}, ready


@pytest.mark.parametrize("owned_by", ("chat default", "edit"))
async def test_a_configured_default_that_refuses_still_reads_refused(
    servers, owned_by
):
    """TASK-33005.6 AC#3: the chat default, or a table the user changed (here
    only its model), is configured, so its refusal stays a NEEDS SETUP row."""
    config = _with_shipped("tabbyapi")
    if owned_by == "chat default":
        config["chat_defaults"] = {"provider": "tabbyapi", "model": "tabby-model"}
    else:
        config["api_settings"]["tabbyapi"]["model"] = "my-own-tabby-model"
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(app, config=config, providers_models=SHIPPED_MODELS)
        await app.push_screen(switcher)
        await _settle(app, pilot)
        setup = {row.provider for row in _group(switcher, "NEEDS SETUP")}
        word = popover_module.switcher_readiness_words(switcher._readiness["tabbyapi"])

    assert "tabbyapi" in setup
    assert word == f"Not ready · refused :{TABBY}"


async def test_an_untouched_shipped_default_that_runs_reads_reachable(servers):
    """TASK-33005.6 AC#3: the reason to keep probing shipped defaults."""
    servers.up = {TABBY}
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(
            app, config=_with_shipped("tabbyapi"), providers_models=SHIPPED_MODELS
        )
        await app.push_screen(switcher)
        await _settle(app, pilot)
        ready = {row.provider for row in _group(switcher, "READY PROVIDERS")}
        word = popover_module.switcher_readiness_words(switcher._readiness["tabbyapi"])

    assert "tabbyapi" in ready
    assert word.startswith("Ready · reachable ")


@pytest.mark.parametrize(
    ("status", "word", "blocked"),
    [(401, "Not ready · key rejected", True), (403, "Ready · not tested", False)],
    ids=["401", "403"],
)
async def test_a_listing_403_never_blocks_but_a_401_does(servers, status, word, blocked):
    """Qodo #2958 follow-up (owner ruling): llama.cpp answers its listing with
    a 401 or a 403. The switcher row and the Console's readiness for the same
    connection agree; only the 401 rejects the key and blocks sending."""
    servers.status = {LLAMA: status}
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(app)
        await app.push_screen(switcher)
        await _settle(app, pilot)
        row = line_with(list_lines(app, switcher), "llama.cpp")
        console = build_console_settings_readiness(
            build_target_default_console_session_settings(CONFIG, "llama_cpp", "model-a"),
            app_config=CONFIG,
            connection_evidence=shared_connection_evidence(lambda: app),
        )

    assert LLAMA in servers.ports
    assert console.connection is not None  # The Console read the probe's result.
    assert popover_module.switcher_readiness_words(console) == word
    assert word in row
    assert (console.operability != "ready_to_send") is blocked


@private_profile_test
async def test_rows_read_verified_only_for_the_models_the_key_check_listed(request):
    """TASK-33005.11 AC#2/#3/#4 (captures 10 and 12): one accepted OpenAI key
    check listed gpt-4o only. Every OpenAI row read "Ready · verified"; now the
    listed row does (the control) and the unlisted row keeps a weaker word.
    For each model the switcher row, the Console's status row and rail word
    (the readiness presentation) and Settings' Readiness row agree. Nothing
    is probed: the evidence is the key check's, published as Settings does."""
    from datetime import datetime

    from tldw_chatbook.Chat.provider_readiness import get_provider_readiness
    from tldw_chatbook.Chat.provider_test_evidence import ProviderTestEvidence
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
    from tldw_chatbook.Widgets.Console.console_settings_summary import (
        build_console_readiness_presentation,
    )

    models = ("gpt-4o", "gpt-5.6-terra")
    words = {
        "gpt-4o": "Ready · verified 09:07",
        "gpt-5.6-terra": "Ready · reachable 09:07",
    }
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        owner = shared_connection_evidence(lambda: app)
        identity = console_send_connection(
            build_target_default_console_session_settings(CONFIG, "openai", None),
            app_config=CONFIG,
        )
        evidence = ProviderTestEvidence(
            identity,
            "reachable",
            ("gpt-4o",),
            credential="listing_accepted",
            observed_at=datetime(2026, 10, 1, 9, 7).astimezone(),
        )
        owner.publish(evidence, order=1)
        switcher = build(
            app,
            providers_models={**PROVIDERS_MODELS, "openai": list(models)},
            prober=False,
            pair=("openai", "gpt-4o"),
        )
        await app.push_screen(switcher)
        await _settle(app, pilot)
        lines = list_lines(app, switcher)

    readiness = get_provider_readiness("openai", CONFIG, environ={})
    for model in models:
        console = build_console_settings_readiness(
            build_target_default_console_session_settings(CONFIG, "openai", model),
            app_config=CONFIG,
            connection_evidence=owner,
        )
        settings_rows = dict(
            SettingsScreen._provider_test_rows(
                readiness,
                display_name="OpenAI",
                model=model,
                endpoint="",
                evidence=evidence,
            )
        )
        assert words[model] in line_with(lines, f" {model} ", "OpenAI"), lines
        presentation = build_console_readiness_presentation(console)
        assert presentation.primary_label == words[model]
        assert settings_rows["Readiness"] == words[model]
    assert "not listed by the server" in settings_rows["Model"]


@pytest.mark.parametrize(
    "server",
    ["lists", "lists-this-chat", "lists-past-the-bound", "not-probed", "refused"],
)
async def test_a_local_servers_rows_are_the_models_its_probe_listed(servers, server):
    """TASK-33005.12: Ollama's rows came from its saved list (qwen3:4b,
    gemma3:12b) although the running server listed only qwen2.5:0.5b, which
    no row offered. Once the probe's listing is in the shared evidence, the
    rows are the listed models, and an Ollama chat's own model stays first
    (ADR-020); with no listing (not probed, or refused) they stay the saved
    ones. A listing at the probe's bound (100 of a server's 150) may stop
    short, so the saved models past it stay offered (round-1 review: the
    saved m120 dropped out, and Find offered it as a typed id), after the
    models a Chat settings listing found (served-y), as under the bound (Wave 6
    integration review round 2: served-y came last). The rows read the evidence
    the probe recorded: one request per local server, as before."""
    servers.up = set() if server == "refused" else {OLLAMA}
    bound = [f"m{index:03d}" for index in range(150)]
    servers.listing = {OLLAMA: bound if "bound" in server else ["qwen2.5:0.5b"]}
    saved = ["m120"] if "bound" in server else ["qwen3:4b", "gemma3:12b"]
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(
            app,
            providers_models={**PROVIDERS_MODELS, "ollama": saved},
            prober=server != "not-probed",
            pair=("ollama", "qwen3:4b")
            if server == "lists-this-chat"
            else ("llama_cpp", "model-a"),
            served_models={"ollama": ["served-y"]} if "bound" in server else None,
        )
        await app.push_screen(switcher)
        await _settle(app, pilot)
        ready = [
            row.model
            for row in _group(switcher, "READY PROVIDERS")
            if row.kind == "pair" and row.provider == "ollama"
        ]
        models = switcher._models_for("ollama")
        lines = list_lines(app, switcher)

    if server == "lists":
        assert ready == ["qwen2.5:0.5b"]
        _identity, reached = _evidence(app, OLLAMA)
        word = f"Ready · reachable {reached.observed_at.astimezone():%H:%M}"
        assert word in line_with(lines, "qwen2.5:0.5b")
        assert not any("qwen3:4b" in line or "gemma3:12b" in line for line in lines)
    elif server == "lists-this-chat":
        assert ready == ["qwen3:4b", "qwen2.5:0.5b"]
        assert popover_module.CURRENT_MARK in line_with(lines, "qwen3:4b")
        assert not any("gemma3:12b" in line for line in lines)
    elif server == "lists-past-the-bound":
        assert ready == bound[:3]  # the first three, then "… 99 more"
    elif server == "not-probed":
        assert ready == saved
    else:
        assert ready == []  # refused: not offered as ready at all
    if "bound" in server:
        assert models == (*bound[:100], "served-y", "m120")
    else:
        assert models == (("qwen2.5:0.5b",) if "lists" in server else tuple(saved))
    assert sorted(servers.ports) == ([] if server == "not-probed" else [LLAMA, OLLAMA])


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
    # Tests/conftest.py shuts this seam for every other test that opens it.
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
        # TASK-33005 final review I-5 (TASK-30011 AC#6): the header under the
        # open switcher says so too, not "Ready · not tested" until Esc.
        assert harness.screen is switcher
        assert _rail_text(console, "#workbench-header-status") == refused
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
        # TASK-34956 follow-up: the poll now carries a cheap pre-gate whose
        # version component reopens it, so the settle is driven directly on
        # the evidence store (as a real off-thread settle would be) instead
        # of as a side effect of the build hook -- the pre-gate would skip
        # the build entirely, and the simulated settle would never fire.
        store = ProviderTestEvidenceStore(lambda: harness)
        store.settle(
            store.begin(identity),
            ProviderProbeResult("unreachable", (), "connection_refused"),
        )
        console._poll_console_credential_readiness()  # The next tick.
        await pilot.pause()

        assert _rail_text(console, "#workbench-header-status") == (
            "Not ready · refused :9099"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("category", "word", "hint"),
    (
        (
            "connection_refused",
            "Not ready · refused :9099",
            "Send blocked — retry the connection to continue",
        ),
        (
            "unauthorized",
            "Not ready · key rejected",
            "Send blocked — key rejected; update the API key",
        ),
    ),
    ids=("refused", "key-rejected"),
)
@private_profile_test
async def test_the_inspector_setup_row_and_send_hint_read_the_readiness_word(
    request, category, word, hint
):
    """TASK-33005.7 AC#2/#3/#5: the Inspector's setup line read "Provider
    configuration required" for any blocker, and a rejected key's composer
    hint said "add an API key". Both now say what the status row says for
    the same readiness: the line is its word, and the hint names the
    rejection."""
    from Tests.UI.test_console_endpoint_discovery import _rail_text
    from Tests.UI.test_console_provider_apply_defaults_flow import (
        _ConsoleFlowHarness,
        _persisted_console_app,
    )
    from Tests.UI.test_destination_shells import _wait_for_selector
    from tldw_chatbook.Chat.console_session_settings import readiness_words

    harness = _ConsoleFlowHarness(_persisted_console_app())
    async with harness.run_test(size=(211, 44)) as pilot:
        console = harness.screen
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        # The poll returns at its attach gate until the Console's attach
        # reconcile ends; the selector only proves the summary mounted
        # (a poll there refreshed nothing in about half the local runs).
        await _wait(
            pilot,
            lambda: console._console_attach_reconciled
            and not console._console_attach_reconcile_running,
            tries=500,
        )
        console._stop_console_credential_poll_timer()
        settings, _readiness = console._active_console_settings_readiness()
        identity = console_send_connection(
            settings, app_config=console._provider_readiness_app_config()
        )
        store = ProviderTestEvidenceStore(lambda: harness)
        store.settle(
            store.begin(identity), ProviderProbeResult("unreachable", (), category)
        )
        console._poll_console_credential_readiness()
        await pilot.pause()

        _settings, readiness = console._active_console_settings_readiness()
        inspector = console._build_console_inspector_state(None)
        rows = {row.label: row.value for row in inspector.rows}
        composer = console._console_composer_or_none()
        status = _rail_text(console, "#workbench-header-status")

    assert readiness_words(readiness) == word == status
    assert rows["Setup"] == word
    assert composer._send_disabled_reason == hint
    assert "add an API key" not in hint


async def test_a_cloud_row_whose_key_check_timed_out_opens_settings(servers):
    """TASK-33005 final review I-4: OpenAI cannot be "started" or rechecked
    on open (D2), so its row neither says so nor dead-ends: Enter opens its
    Settings, where 't' tests it again."""
    from tldw_chatbook.Chat.console_session_settings import console_send_connection

    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        identity = console_send_connection(
            build_target_default_console_session_settings(CONFIG, "openai", None),
            app_config=CONFIG,
        )
        store = ProviderTestEvidenceStore(lambda: app)
        store.settle(store.begin(identity), ProviderProbeResult("unreachable", (), "timeout"))
        switcher = build(app)
        await app.push_screen(switcher)
        await _settle(app, pilot)
        openai = line_with(list_lines(app, switcher), "OpenAI")
        assert "Not ready · timed out" in openai
        assert "Enter: open Settings" in openai and "start it" not in openai
        await pilot.press(*"gpt-5.1")
        await pilot.pause()
        row = switcher.highlighted_row()
        assert (row.kind, row.provider) == ("setup", "openai")
        await pilot.press("enter")
        await pilot.pause()

    assert app.setup == [("openai", "gpt-5.1")]
    assert "api.openai.com" not in {host for host, *_rest in servers.requests}
