"""TASK-40714: an ordinary Console Send must not need a bound MCP store.

The MCP store binds to backup custody only with POSIX pinned I/O (never on
Windows) and only at the profile's canonical path. Unbound, the stock screen
refused every Send with "Console preparation source unavailable; Send again."
The first test presses the real Send button on a mounted ChatScreen and waits
for the gateway's reply in the transcript and in the store. The second reruns
the draft-snapshot sends that take the other capture routes (a queued send and
a received spoken send) without a bound store.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI import test_console_send_acknowledgement as acknowledgement
from Tests.UI import test_console_send_draft_snapshot as draft_snapshot
from Tests.UI.test_console_native_chat_flow import (
    CapturingGateway,
    _build_console_send_test_app,
    _select_llamacpp_console,
    _wait_for_reply,
)
from Tests.UI.test_console_send_admission_off_pump import _record_notices
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Backup_Recovery import mcp_source_participants, raw_participants
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Widgets.Console import ConsoleComposerBar

pytestmark = pytest.mark.bootstrap_profile


def _bind_as_on_windows(monkeypatch) -> None:
    """Bind MCP stores without POSIX pinned I/O, as every Windows install does.

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
@pytest.mark.parametrize("profile", ["bound", "windows", "non-default-data-dir"])
async def test_ordinary_send_reaches_the_provider_and_saves_the_reply(
    monkeypatch, tmp_path, profile
):
    kwargs = {}
    if profile == "windows":
        _bind_as_on_windows(monkeypatch)
    elif profile == "non-default-data-dir":
        kwargs["user_data_dir"] = tmp_path
    gateway = CapturingGateway()
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


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["windows", "non-default-data-dir"])
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
    if profile == "windows":
        _bind_as_on_windows(monkeypatch)
    else:
        build = _build_console_send_test_app

        def build_elsewhere(*args, **kwargs):
            return build(*args, user_data_dir=tmp_path, **kwargs)

        for module in (draft_snapshot, acknowledgement):
            monkeypatch.setattr(module, "_build_console_send_test_app", build_elsewhere)
    await getattr(draft_snapshot, scenario)(monkeypatch)

