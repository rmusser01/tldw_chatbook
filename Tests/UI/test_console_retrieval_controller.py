"""No-mount contracts for Console retrieval ownership."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.UI.Console_Modules.retrieval import ConsoleRetrievalController


def _controller() -> tuple[ConsoleRetrievalController, SimpleNamespace]:
    """Build the real controller with observable no-mount boundary doubles."""
    state = SimpleNamespace(
        pending=None,
        pending_auto_open=False,
        sent_notice=3,
        sync_result=True,
        sync_calls=0,
        refresh_calls=0,
    )

    def sync_pending_launch_surfaces() -> bool:
        state.sync_calls += 1
        return state.sync_result

    controller = ConsoleRetrievalController(
        app_instance=SimpleNamespace(),
        active_native_session=lambda: None,
        current_conversation_id=lambda: None,
        clear_evidence_sent_notice=lambda: None,
        consume_pending_launch=lambda: None,
        release_consumed_launch=lambda _launch, _result: None,
        is_mounted=lambda: False,
        sync_retrieval_scope_row=lambda: None,
        sync_control_bar=lambda: None,
        request_control_bar_sync=lambda: None,
        dictionary_scope_service=lambda: None,
        set_library_rag_source_scope=lambda _scope: None,
        set_library_rag_query=lambda _query: None,
        run_library_rag_action=lambda: None,
        library_rag_source_scope=lambda: ("notes", "media", "conversations"),
        library_rag_top_k=lambda: 5,
        pending_launch=lambda: state.pending,
        set_pending_launch=lambda value: setattr(state, "pending", value),
        set_pending_auto_open=lambda value: setattr(state, "pending_auto_open", value),
        set_evidence_sent_notice=lambda value: setattr(state, "sent_notice", value),
        sync_pending_launch_surfaces=sync_pending_launch_surfaces,
        refresh_screen=lambda: setattr(state, "refresh_calls", state.refresh_calls + 1),
        has_staged_evidence=lambda: False,
    )
    return controller, state


@pytest.mark.unit
def test_retrieval_controller_initializes_owned_cache_state() -> None:
    """The six extracted state fields start with the historical defaults."""
    controller, _state = _controller()

    assert controller._console_retrieval_scope_cache == {}
    assert controller._console_effective_scope_cache == {}
    assert controller._active_dictionaries_summary is None
    assert controller._last_console_dictionary_scope_ids is None
    assert controller._active_world_books_summary is None
    assert controller._last_console_world_book_scope_ids is None


@pytest.mark.unit
def test_retrieval_status_vocabulary_is_controller_owned() -> None:
    """Status projection stays usable without a mounted ChatScreen."""
    controller, _state = _controller()
    searching = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Library retrieval",
        status="searching",
    )

    assert controller._console_rag_source_status(None) == "not staged"
    assert (
        controller._console_rag_source_status(None, sent_source_count=2)
        == "sent with the last message · 2 sources"
    )
    assert (
        controller._console_rag_source_status(searching)
        == "retrieving from Library Search/RAG"
    )


@pytest.mark.unit
def test_staging_updates_state_before_sync_and_recomposes_only_as_fallback() -> None:
    """The controller stages once and keeps full recompose as a fallback."""
    controller, state = _controller()
    launch = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Library retrieval",
        status="staged",
    )

    controller._stage_console_library_rag_launch(launch)

    assert state.pending is launch
    assert state.sent_notice is None
    assert state.sync_calls == 1
    assert state.refresh_calls == 0

    state.sync_result = False
    controller._stage_console_library_rag_launch(launch)

    assert state.sync_calls == 2
    assert state.refresh_calls == 1


@pytest.mark.unit
def test_retrieval_controller_owns_no_automatic_placeholder_cleanup() -> None:
    """Automatic preparation placeholders belong to controller/store state."""
    controller, _state = _controller()

    assert not hasattr(controller, "_clear_console_auto_rag_placeholder")
