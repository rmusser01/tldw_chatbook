"""Status text derives from current activity without promising remote receipt."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Console_Modules.agent import console_send_activity_copy

# UI autouse fixtures import the app, which owns its collection-time profile.
pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize(
    "status,started,expected",
    [
        ("validating", False, "Sending…"),
        ("streaming", False, "Waiting for reply…"),
        ("streaming", True, "Streaming reply…"),
        ("checking_citations", False, "Checking citations."),
        ("retrying", False, "Retrying."),
        ("idle", False, ""),
        ("blocked", False, ""),
        ("failed", False, ""),
        ("stopped", False, ""),
        ("completed", False, ""),
    ],
)
def test_send_copy_tracks_current_stage_without_a_bridge(status, started, expected):
    copy = {"checking_citations": "Checking citations.", "retrying": "Retrying."}.get(
        status, ""
    )
    assert (
        console_send_activity_copy(
            SimpleNamespace(status=status, visible_copy=copy), reply_started=started
        )
        == expected
    )


def test_setup_and_live_primary_tool_keep_specific_copy():
    from tldw_chatbook.Agents.agent_models import AGENT_KIND_PRIMARY, STEP_TOOL_CALL

    state = SimpleNamespace(status="streaming", visible_copy="Agent running.")
    assert (
        console_send_activity_copy(state, snapshot=SimpleNamespace(status="setup"))
        == "Connecting tools…"
    )
    tool = SimpleNamespace(
        agent_kind=AGENT_KIND_PRIMARY, kind=STEP_TOOL_CALL, text="fs_read"
    )
    snapshot = SimpleNamespace(status="running", steps=(tool,))
    assert console_send_activity_copy(state, snapshot=snapshot) == "⚙ fs_read"
    snapshot.status = "completed"
    assert console_send_activity_copy(state, snapshot=snapshot) == "Waiting for reply…"


def test_live_output_read_is_bounded_and_never_materializes_or_snapshots(monkeypatch):
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="old answer",
        persist=False,
    )
    current = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        persist=False,
    )

    current_node = store._nodes_by_session[session.id][current.id]
    current_node.status = "pending"

    def unexpected(*args, **kwargs):
        raise AssertionError(
            "progress must not copy history or materialize/persist chunks"
        )

    monkeypatch.setattr(store, "_snapshot", unexpected)
    monkeypatch.setattr(store, "_materialize_stream_buffer", unexpected)
    assert not store.has_live_reply_output(session.id)
    store._stream_chunks_by_message[current.id] = [" ", "new delta"]
    assert store.has_live_reply_output(session.id)
    assert current_node.content == ""
    current_node.status = "complete"
    assert not store.has_live_reply_output(session.id)
    current_node.status = "streaming"
    current_node.generation_projection_quarantined = True
    assert not store.has_live_reply_output(session.id)
