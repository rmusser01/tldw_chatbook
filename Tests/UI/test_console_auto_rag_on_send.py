"""Task 13 ownership checks for the former screen-owned auto-RAG path."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.UI.Console_Modules import retrieval as retrieval_module
from tldw_chatbook.UI.Console_Modules.retrieval import ConsoleRetrievalController


def _controller(*, launch=None):
    state = SimpleNamespace(launch=launch, sent_notice="old", released=[])
    app = SimpleNamespace(library_rag_search_service=SimpleNamespace())
    controller = ConsoleRetrievalController(
        app_instance=app,
        active_native_session=lambda: None,
        current_conversation_id=lambda: None,
        clear_evidence_sent_notice=lambda: setattr(state, "sent_notice", None),
        consume_pending_launch=lambda: state.launch,
        release_consumed_launch=lambda *args: state.released.append(args),
        is_mounted=lambda: True,
        sync_retrieval_scope_row=lambda: None,
        sync_control_bar=lambda: None,
        request_control_bar_sync=lambda: None,
        dictionary_scope_service=lambda: None,
        set_library_rag_source_scope=lambda _value: None,
        set_library_rag_query=lambda _value: None,
        run_library_rag_action=lambda: None,
        push_screen=lambda *_args, **_kwargs: None,
        library_rag_source_scope=lambda: ("media",),
        library_rag_top_k=lambda: 99,
        pending_launch=lambda: state.launch,
        set_pending_launch=lambda value: setattr(state, "launch", value),
        set_pending_auto_open=lambda _value: None,
        set_evidence_sent_notice=lambda value: setattr(state, "sent_notice", value),
        sync_pending_launch_surfaces=lambda: None,
        refresh_screen=lambda: None,
        has_staged_evidence=lambda: state.launch is not None,
    )
    return controller, state


def test_mounted_retrieval_controller_has_no_automatic_send_owner():
    """Standing policy/config, spend, timeout, and fail-open copy moved out."""

    source = inspect.getsource(retrieval_module)

    assert "_maybe_auto_retrieve_for_send" not in source
    assert 'get_cli_setting("chat_defaults", "rag_auto_retrieve_on_send"' not in source
    assert "Message sent without Library evidence" not in source
    assert "CONSOLE_AUTO_RAG_FAILED_NOTICE" not in source
    assert not hasattr(ConsoleRetrievalController, "_maybe_auto_retrieve_for_send")


def test_future_automatic_default_does_not_restore_a_send_path_owner():
    """The setting seeds new policy holders; retrieval still owns no policy."""

    assert "rag_auto_retrieve_on_send" not in inspect.getsource(retrieval_module)


@pytest.mark.asyncio
async def test_capture_seam_consumes_only_explicit_manual_evidence(monkeypatch):
    launch = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Manually staged",
        payload={"query": "manual", "evidence_bundle": {"bundle_id": "manual"}},
        status="staged",
    )
    controller, state = _controller(launch=launch)
    captured = SimpleNamespace(context="manual context")
    capture = AsyncMock(return_value=captured)
    monkeypatch.setattr(
        retrieval_module,
        "capture_console_staged_evidence_for_chat",
        capture,
    )

    result = await controller._capture_console_staged_rag("exact draft")

    assert result is captured
    capture.assert_awaited_once_with(
        controller.app_instance,
        launch,
        user_message="exact draft",
    )
    assert state.released == [(launch, captured)]
    assert state.sent_notice is None


@pytest.mark.asyncio
async def test_capture_seam_never_reads_manual_filters_for_automatic_work(monkeypatch):
    controller, _state = _controller(launch=None)
    capture = AsyncMock(return_value=SimpleNamespace(context=None))
    monkeypatch.setattr(
        retrieval_module,
        "capture_console_staged_evidence_for_chat",
        capture,
    )
    controller._library_rag_source_scope = lambda: (_ for _ in ()).throw(
        AssertionError("manual source filters were read")
    )
    controller._library_rag_top_k = lambda: (_ for _ in ()).throw(
        AssertionError("manual top-k was read")
    )

    await controller._capture_console_staged_rag("plain text")

    capture.assert_awaited_once()
