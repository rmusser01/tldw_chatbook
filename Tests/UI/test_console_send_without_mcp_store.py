"""TASK-40714: an ordinary Console Send must not need a bound MCP store.

The MCP store binds to backup custody only at the profile's canonical path
and only with pinned I/O. Every supported platform has pinned I/O (Windows
through `Utils/windows_files.WindowsOS`), so TASK-40714's "never on Windows"
was wrong (TASK-41308): a store outside the profile's data folder is the
unbound case, and a bind with pinned I/O switched off is kept as a second way
to build one. Unbound, the stock screen refused every Send with "Console
preparation source unavailable; Send again."
The tests press the real Send button on a mounted ChatScreen and wait for the
gateway's reply in the transcript and in the store; check that a replaced MCP
reader is still refused; and rerun the draft-snapshot sends that take the
other capture routes (a queued send and a received spoken send) without a
bound store.

TASK-41308: the Inspector's MCP row says MCP is off, and why, whenever the
store is unbound, so a Send without MCP tools is never silent.

TASK-41900: the ordinary send also runs with the agent runtime on (the
Console default). Unbound, the turn's native sources refused it with
`mcp_source_not_installed`; a bound store keeps its source proofs.
TASK-42016: so do the draft-snapshot sends.

TASK-42018: the launcher for the probes that show an unseen receipt never
reaches the next bootstrap-profile test lives here, in a laned file. With
that clear in ``Tests/conftest.py``, TASK-42138 removed this file's own two
receipt workarounds (marking earlier tests' receipts seen before mounting,
and waiting for a test's own to be seen before leaving).
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Static

from Tests.UI import test_console_send_acknowledgement as acknowledgement
from Tests.UI import test_console_send_draft_snapshot as draft_snapshot
from Tests.UI.test_console_native_chat_flow import (
    CapturingGateway,
    _build_console_send_test_app,
    _select_llamacpp_console,
    _wait_for_reply,
)
from Tests.UI.test_console_internals_decomposition import _open_console_inspector
from Tests.UI.test_console_send_admission_off_pump import _record_notices
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)
from tldw_chatbook import config as config_module
from tldw_chatbook.Agents import native_source_storage
from tldw_chatbook.Backup_Recovery import mcp_source_participants, raw_participants
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Chat import console_configuration_preparation as preparation
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.MCP.console_snapshot import (
    UNBOUND_MCP_STORE_REASON,
    _source_identity,
)
from tldw_chatbook.Widgets.Console import ConsoleComposerBar

pytestmark = pytest.mark.bootstrap_profile


class _ToolRecordingGateway(CapturingGateway):
    """Record the tools each provider call offered, beside its messages."""

    def __init__(self) -> None:
        super().__init__()
        self.offered_tools: list = []

    async def stream_chat(self, resolution, messages, **kwargs):
        self.offered_tools.append(kwargs.get("tools"))
        async for chunk in super().stream_chat(resolution, messages, **kwargs):
            yield chunk


def _record_captures(monkeypatch) -> list:
    """Record the configuration snapshot each owned turn capture froze."""
    captured: list = []
    real = preparation.capture_console_turn_configuration

    def capture(*args, **kwargs):
        result = real(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(preparation, "capture_console_turn_configuration", capture)
    return captured


def _bind_without_pinned_io(monkeypatch) -> None:
    """Bind MCP stores with pinned I/O switched off, so they stay unbound.

    Scoped to the bind: the profile's other sources registered with pinned I/O
    before the test, and switching it off under them is refused by design.
    """
    real_bind = mcp_source_participants.bind

    def bind(source):
        with monkeypatch.context() as patch:
            patch.setattr(raw_participants, "_pinned_io_available", lambda: False)
            return real_bind(source)

    monkeypatch.setattr(mcp_source_participants, "bind", bind)


async def _wait_for_saved_reply(console, pilot, reply: str = "accepted") -> None:
    """Wait up to 30 s for ``reply`` to be saved or shown.

    An agent-runtime turn took about 4.5 s from press to reply under I/O
    load, past the shared ``_wait_for_reply``'s 4 s (TASK-41900 review).
    """
    from tldw_chatbook.Chat.console_display_state import QUEUE_REASON_PREPARING

    store = console._ensure_console_chat_store()

    def arrived() -> bool:
        saved = any(
            m.role is ConsoleMessageRole.ASSISTANT and m.content == reply
            for m in store.messages_for_session(store.active_session_id)
        )
        return saved or reply in _visible_text(console).replace(
            QUEUE_REASON_PREPARING, ""
        )

    deadline = time.monotonic() + 30
    while not arrived():
        if time.monotonic() > deadline:
            raise AssertionError(f"Reply not found: {reply!r}")
        await pilot.pause(0.05)


def _turn_on_agent_runtime(app, monkeypatch) -> list:
    """Run the agent runtime (the builder turns it off) and record its sources."""
    assert config_module.save_setting_to_cli_config("console", "agent_runtime", True)
    app.app_config = config_module.load_settings(force_reload=True)
    built: list = []
    real = native_source_storage._AgentSources.__init__

    def init(self, *args, **kwargs):
        real(self, *args, **kwargs)
        built.append(self)

    monkeypatch.setattr(native_source_storage._AgentSources, "__init__", init)
    return built


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime", ["gateway", "agent"])
@pytest.mark.parametrize("profile", ["bound", "no-pinned-io", "non-default-data-dir"])
async def test_ordinary_send_reaches_the_provider_and_saves_the_reply(
    monkeypatch, tmp_path, profile, runtime
):
    kwargs = {}
    if profile == "no-pinned-io":
        _bind_without_pinned_io(monkeypatch)
    elif profile == "non-default-data-dir":
        kwargs["user_data_dir"] = tmp_path
    gateway = _ToolRecordingGateway()
    captures = _record_captures(monkeypatch)
    app = _build_console_send_test_app(**kwargs)
    sources = (
        _turn_on_agent_runtime(app, monkeypatch) if runtime == "agent" else []
    )
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    bound = mcp_source_participants._BINDINGS.get(app.local_mcp_store) is not None
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        notices = _record_notices(console)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello without mcp")
        console.query_one("#console-send-message", Button).press()
        try:
            await _wait_for_saved_reply(console, pilot)
        except AssertionError:
            raise AssertionError(f"no reply; notices: {notices}") from None
        store = console._ensure_console_chat_store()
        replies = [
            m.content
            for m in store.messages_for_session(store.active_session_id)
            if m.role is ConsoleMessageRole.ASSISTANT
        ]

    assert bound is (profile == "bound")
    assert gateway.sent_messages[-1][-1]["content"] == "hello without mcp"
    assert replies == ["accepted"]
    # A bound store freezes its MCP definitions (the built-in server's tools);
    # an unbound one freezes none, and the provider is offered no tools.
    assert len(captures) == 1
    assert bool(captures[0].mcp_definition_maximum) is (profile == "bound")
    assert gateway.offered_tools == [None]
    # The agent runtime really ran the turn: it pinned the turn's sources.
    assert len(sources) == (runtime == "agent")
    for source in sources:
        assert source.permission is app.unified_mcp_service._permission_store
        assert source.prospective_mcp_identity is None
        assert (
            mcp_source_participants._BINDINGS.get(source.permission) is not None
        ) is bound


def _agent_sources(app) -> native_source_storage._AgentSources:
    """Pin one turn's native sources for ``app``, as a Console submit does."""
    store = SimpleNamespace(persistence=SimpleNamespace(db=None))
    controller = SimpleNamespace(app=app, store=store, _scratch_spaces=None)
    return native_source_storage._AgentSources(
        controller,
        None,
        SimpleNamespace(_db=None),
        SimpleNamespace(workspace_id=None, incarnation_id="incarnation"),
        SimpleNamespace(scratch_space=None),
    )


@pytest.mark.parametrize("profile", ["bound", "no-pinned-io", "replaced-reader"])
def test_turn_sources_prove_a_bound_store_and_select_an_unbound_ones_permissions(
    monkeypatch, profile
):
    """TASK-41900: a turn pins its MCP permission store before it runs.

    A bound store whose permission store does not exist yet still gives the
    turn its prospective identity, exactly as before. An unbound stock store
    cannot be proven, so the turn selects the service's own permission store
    instead: the turn's kill-switch and tool-state readers would otherwise
    create it after the pin and fail the turn. A replaced reader is not stock
    and still refuses.
    """
    if profile != "bound":
        _bind_without_pinned_io(monkeypatch)
    app = _build_console_send_test_app()
    service = app.unified_mcp_service
    store = service.local_service.store
    if profile == "replaced-reader":
        original = service.get_kill_switch
        monkeypatch.setattr(
            service, "get_kill_switch", lambda *a, **k: original(*a, **k), raising=False
        )
    assert service._permission_store is None

    if profile == "replaced-reader":
        with pytest.raises(RecoveryRequired, match="mcp_source_not_installed"):
            _agent_sources(app)
        assert service._permission_store is None
        return
    source = _agent_sources(app)

    if profile == "bound":
        assert service._permission_store is None
        assert source.permission is None
        assert source.prospective_mcp_store is store
        assert source.prospective_mcp_identity == _source_identity(store)
        assert source.prospective_permission_path == store.path.with_name(
            "mcp_permissions.json"
        )
    else:
        with pytest.raises(RecoveryRequired, match="mcp_source_not_installed"):
            _source_identity(store)
        assert source.permission is service._permission_store is not None
        assert source.permission_path == store.path.with_name("mcp_permissions.json")
        assert source.prospective_mcp_store is None
        assert source.prospective_mcp_identity is None
        service.get_kill_switch()  # a turn's lazy reader finds the pinned store
        assert service._permission_store is source.permission


@pytest.mark.asyncio
async def test_unbound_store_with_a_replaced_mcp_reader_is_still_refused(monkeypatch):
    """Only a stock MCP service runs without MCP tools; a replaced reader refuses."""
    _bind_without_pinned_io(monkeypatch)
    gateway = CapturingGateway()
    app = _build_console_send_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    service = app.unified_mcp_service
    original = service.get_kill_switch
    monkeypatch.setattr(
        service, "get_kill_switch", lambda *a, **k: original(*a, **k), raising=False
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        notices = _record_notices(console)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello without mcp")
        console.query_one("#console-send-message", Button).press()
        for _ in range(100):
            if notices:
                break
            await pilot.pause(0.05)

    assert notices == ["Console preparation source unavailable; Send again."]
    assert gateway.sent_messages == []


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime", ["gateway", "agent"])
@pytest.mark.parametrize("profile", ["no-pinned-io", "non-default-data-dir"])
@pytest.mark.parametrize(
    "scenario",
    [
        "test_console_active_run_queues_without_a_screen_worker",
        "test_console_armed_unknown_tokenless_send_snapshots_before_skill_await",
        "test_spoken_send_sends_its_capture_and_keeps_text_typed_after_it",
    ],
)
async def test_draft_snapshot_sends_without_a_bound_mcp_store(
    monkeypatch, tmp_path, profile, scenario, runtime
):
    """TASK-42016: also with the agent runtime on, as TASK-41900 did for the
    ordinary send."""
    if profile == "no-pinned-io":
        _bind_without_pinned_io(monkeypatch)
    build = _build_console_send_test_app
    sources: list = []

    def build_for_profile(*args, **kwargs):
        if profile == "non-default-data-dir":
            kwargs["user_data_dir"] = tmp_path
        app = build(*args, **kwargs)
        if runtime == "agent":
            sources.append(_turn_on_agent_runtime(app, monkeypatch))
        return app

    for module in (draft_snapshot, acknowledgement):
        monkeypatch.setattr(module, "_build_console_send_test_app", build_for_profile)
    captures = _record_captures(monkeypatch)
    await getattr(draft_snapshot, scenario)(monkeypatch)

    # The send's capture froze the runtime as set: saved after the build
    # without a reload, it never reaches the Console (TASK-41900's lesson).
    flags = {c.tool_configuration.get("agent_runtime_enabled") for c in captures}
    assert flags == {runtime == "agent"}
    # Only the spoken send runs its turn here (the queued one waits, the armed
    # one's acceptance is replaced); with the runtime on, it pinned its sources.
    ran = scenario == "test_spoken_send_sends_its_capture_and_keeps_text_typed_after_it"
    assert [len(built) for built in sources] == ([int(ran)] if runtime == "agent" else [])


def _mcp_row_text(console) -> str | None:
    """The MCP row's text when it is on screen: mounted, outside the closed
    "More" disclosure, with a painted region. ``None`` when it is absent."""
    rows = list(console.query("#console-inspector-mcp").results(Static))
    if not rows:
        return None
    row = rows[0]
    more = list(console.query("#console-inspector-more-body"))
    assert not any(body in row.ancestors for body in more), "MCP row under More"
    assert row.region.width > 0 and row.region.height > 0, "MCP row not painted"
    return str(row.renderable)


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["bound", "no-pinned-io", "non-default-data-dir"])
async def test_inspector_mcp_row_says_off_and_why_when_the_store_is_unbound(
    monkeypatch, tmp_path, profile
):
    """TASK-41308: before and after a Send, an unbound store shows the row.

    The off row makes the Tools group actionable, so it sits above "More"
    where the user sees it without opening anything.
    """
    kwargs = {}
    if profile == "no-pinned-io":
        _bind_without_pinned_io(monkeypatch)
    elif profile == "non-default-data-dir":
        kwargs["user_data_dir"] = tmp_path
    gateway = _ToolRecordingGateway()
    app = _build_console_send_test_app(**kwargs)
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)
    off = (
        None
        if profile == "bound"
        else f"MCP: off · {UNBOUND_MCP_STORE_REASON}"
    )

    async with host.run_test(size=(196, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _select_llamacpp_console(console)
        await _open_console_inspector(console, pilot)
        if off is not None:
            await _wait_for_selector(console, pilot, "#console-inspector-mcp")
        before = _mcp_row_text(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello without mcp")
        console.query_one("#console-send-message", Button).press()
        await _wait_for_reply(console, pilot)
        await pilot.pause()
        after = _mcp_row_text(console)

    assert before == off
    assert after == off
    assert gateway.offered_tools == [None]


@pytest.mark.asyncio
async def test_inspector_keeps_the_unbound_reason_when_the_kill_switch_is_on(tmp_path):
    """The store reason wins over the user's MCP kill switch (TASK-41308).

    Turning the switch off would not bring MCP back while the store cannot be
    used, so the row names the cause that stays. Nothing may show the switch
    instead: a mount-time read is a UI-thread disk read, and the row's reason
    comes from the store's binding, which wins over any kill-switch read a
    Send makes (with the agent runtime on, its tools read the switch;
    TASK-41900).
    """
    app = _build_console_send_test_app(user_data_dir=tmp_path)
    app.unified_mcp_service.set_kill_switch(True)
    assert app.unified_mcp_service.get_kill_switch() is True
    host = ConsoleHarness(app)

    async with host.run_test(size=(196, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _open_console_inspector(console, pilot)
        await _wait_for_selector(console, pilot, "#console-inspector-mcp")
        text = _mcp_row_text(console)

    assert text == f"MCP: off · {UNBOUND_MCP_STORE_REASON}"


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["bound", "no-pinned-io", "non-default-data-dir"])
async def test_live_work_mcp_row_names_the_unbound_store_reason(
    monkeypatch, tmp_path, profile
):
    """TASK-41631: the live-work card's MCP row agrees with the Inspector row.

    Both writers are read: the card the builder makes, and the row the sync
    tick rewrites. A bound store keeps "Not wired".
    """
    kwargs = {}
    if profile == "no-pinned-io":
        _bind_without_pinned_io(monkeypatch)
    elif profile == "non-default-data-dir":
        kwargs["user_data_dir"] = tmp_path
    app = _build_console_send_test_app(**kwargs)
    host = ConsoleHarness(app)
    expected = (
        "MCP: Not wired - MCP servers."
        if profile == "bound"
        else f"MCP: Off - {UNBOUND_MCP_STORE_REASON}"
    )

    async with host.run_test(size=(196, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_selector(console, pilot, "#console-live-work-source-mcp")
        card = console._build_console_live_work_source_readiness_card()
        # The built card is not mounted, so its children are only readable
        # through Textual's `_pending_children`, as six other test files do.
        built = next(
            str(child.renderable)
            for child in card._pending_children
            if child.id == "console-live-work-source-mcp"
        )
        row = console.query_one("#console-live-work-source-mcp", Static)
        row.update("stale")
        console._sync_console_live_work_readiness_rows()
        await pilot.pause()
        ticked = str(row.renderable)

    assert built == expected
    assert ticked == expected


@pytest.mark.asyncio
async def test_unbound_live_work_mcp_row_fits_the_section_at_full_screen(
    tmp_path, monkeypatch
):
    """TASK-41631: the reason wraps the MCP row, and the card stays in its cap.

    Measured with the Inspector open: the row's content is 36 columns at
    235x52, 29 at 200x60 and 23 at 160x45. The unbound row takes 2, 2 and 3
    lines, and the card 17, 18 and 19 rows (bound: 16, 17, 18). All stay
    within the live-work section's 20-row cap, so no "more" hint appears.
    """
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    # The measured heights below assume "RAG: Ready" (one line). Without the
    # embeddings extras (the CI UI lane) the RAG row wraps to two lines and the
    # card measures 18, so pin the extras rather than the runner's installs.
    monkeypatch.setattr(ChatScreen, "_console_rag_extras_available", lambda self: True)
    app = _build_console_send_test_app(user_data_dir=tmp_path)
    host = ConsoleHarness(app)

    async with host.run_test(size=(235, 52)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _open_console_inspector(console, pilot)
        await _wait_for_selector(console, pilot, "#console-live-work-source-mcp")
        row = console.query_one("#console-live-work-source-mcp", Static)
        card = console.query_one("#console-live-work-source-readiness")
        bounded = console.query_one("#console-bounded-section-live-work")
        for _ in range(100):
            if row.region.height == 2 and bounded.desired_content_lines == (
                card.region.height
            ):
                break
            await pilot.pause(0.05)
        text = str(row.renderable)
        row_lines = row.region.height
        card_rows = card.region.height
        demand = bounded.desired_content_lines
        cap = bounded.max_content_lines
        hint_shown = bounded.hint.display

    assert text == f"MCP: Off - {UNBOUND_MCP_STORE_REASON}"
    assert row_lines == 2
    assert card_rows == demand == 17
    assert demand <= cap == 20
    assert hint_shown is False


def test_an_unseen_console_receipt_never_reaches_the_next_bootstrap_test(tmp_path):
    """TASK-42018: the shared database's unseen receipts are cleared between
    bootstrap-profile tests, as their config is (``Tests/conftest.py``), and
    a file at that path that is not a database is left alone, with no
    teardown error for that test or the next.

    Runs TASK-36478's child pytest process, so each planting probe and the
    reading probe after it share one process in file order.
    """
    from Tests.test_bootstrap_profile_config_reset import _run_probes

    result = _run_probes(tmp_path, "probe_g or probe_h or probe_i or probe_j")

    output = result.stdout + result.stderr
    assert result.returncode == 0, output[-6000:]
    assert "4 passed" in output, output[-6000:]
    assert "error" not in output.strip().splitlines()[-1], output[-6000:]
