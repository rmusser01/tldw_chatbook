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
"""

from __future__ import annotations

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
)
from tldw_chatbook.Backup_Recovery import mcp_source_participants, raw_participants
from tldw_chatbook.Chat import console_configuration_preparation as preparation
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
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


def _record_mcp_maximum(monkeypatch) -> list:
    """Record the MCP definition maximum each owned turn capture froze."""
    captured: list = []
    real = preparation.capture_console_turn_configuration

    def capture(*args, **kwargs):
        result = real(*args, **kwargs)
        captured.append(dict(result.mcp_definition_maximum))
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


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["bound", "no-pinned-io", "non-default-data-dir"])
async def test_ordinary_send_reaches_the_provider_and_saves_the_reply(
    monkeypatch, tmp_path, profile
):
    kwargs = {}
    if profile == "no-pinned-io":
        _bind_without_pinned_io(monkeypatch)
    elif profile == "non-default-data-dir":
        kwargs["user_data_dir"] = tmp_path
    gateway = _ToolRecordingGateway()
    maximums = _record_mcp_maximum(monkeypatch)
    app = _build_console_send_test_app(**kwargs)
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
            await _wait_for_reply(console, pilot)
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
    assert len(maximums) == 1
    assert bool(maximums[0]) is (profile == "bound")
    assert gateway.offered_tools == [None]


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
    monkeypatch, tmp_path, profile, scenario
):
    if profile == "no-pinned-io":
        _bind_without_pinned_io(monkeypatch)
    else:
        build = _build_console_send_test_app

        def build_elsewhere(*args, **kwargs):
            return build(*args, user_data_dir=tmp_path, **kwargs)

        for module in (draft_snapshot, acknowledgement):
            monkeypatch.setattr(module, "_build_console_send_test_app", build_elsewhere)
    await getattr(draft_snapshot, scenario)(monkeypatch)


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
        else "MCP: off · this profile's MCP settings file can't be used here"
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
    instead: a mount-time read is a UI-thread disk read, and a Send never
    reads it for an unbound store (the preparation drops MCP first), so no
    published state carries it.
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

    assert text == "MCP: off · this profile's MCP settings file can't be used here"
