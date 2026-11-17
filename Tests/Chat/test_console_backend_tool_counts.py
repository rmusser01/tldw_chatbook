"""Actual backend count publications retain original computation and absence."""

from dataclasses import FrozenInstanceError
import sys

import pytest

from Tests.private_profile import private_profile_test


@private_profile_test
@pytest.mark.asyncio
async def test_original_backend_tool_publications_and_foreign_owner_refusal(
    request, monkeypatch
):
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook.Agents import plain_chat_tool_surface as surface
    from tldw_chatbook.Chat import console_backend_tool_counts as count_source
    from tldw_chatbook.Chat.console_backend_tool_counts import (
        ConsoleBackendToolCounts,
        backend_tool_counts,
    )
    from tldw_chatbook.Chat.console_execution_backend import ConsoleExecutionBackend
    from tldw_chatbook.Chat.console_execution_profile import open_console_backend_profile

    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(
        expected_config_path=path, expected_data_directory=directory
    )
    backend = ConsoleExecutionBackend(
        profile, profile_epoch="profile", service_epoch="counts"
    )
    try:
        backend.start()
        counts = backend.presentation.tool_counts
        assert type(counts) is ConsoleBackendToolCounts
        controller = backend.controller
        session = backend.store.create_session(title="Original counts")
        assert backend_tool_counts(controller) is counts
        empty = counts.snapshot(controller, session.id)
        assert not empty.published
        assert empty.builtin_count is empty.mcp_count is None
        assert empty.not_connected_count is empty.selected_count is None

        controller._publish_selected_tool_count(session.id, 7)
        selected = counts.snapshot(controller, session.id)
        assert not selected.published and selected.selected_count == 7
        calls = []

        def original_entry(frame):
            calls.append(
                (
                    frame.f_locals["bridge"],
                    frame.f_locals["session_id"],
                    frame.f_locals["composed_switch"],
                )
            )

        with _observe(
            surface.offered_builtin_tool_count.__code__,
            sys.monitoring.events.PY_START,
            original_entry,
        ):
            controller._publish_mcp_inspector_counts(3, None, session.id)
            published = counts.snapshot(controller, session.id)
            assert published.published and type(published.builtin_count) is int
            assert published.mcp_count == 3 and published.not_connected_count is None
            assert published.selected_count is None
            assert calls == [(controller._agent_bridge, session.id, None)]
            controller._publish_selected_tool_count(session.id, 0)
            assert counts.snapshot(controller, session.id).selected_count == 0
            controller._publish_mcp_inspector_counts(None, None, session.id)
            absent = counts.snapshot(controller, session.id)
            assert absent.published and absent.mcp_count is None
            assert absent.not_connected_count is absent.selected_count is None
            assert len(calls) == 2  # Snapshots never recompute the catalog/count.

        with pytest.raises(FrozenInstanceError):
            published.mcp_count = 99
        with pytest.raises(RuntimeError, match="definitions changed"):
            counts.publish_selected(object(), session.id, 99)
        with pytest.raises(RuntimeError, match="session changed"):
            counts.publish_selected(controller, "foreign-session", 99)
        foreign = ConsoleBackendToolCounts(backend.presentation)
        with pytest.raises(RuntimeError, match="presentation changed"):
            foreign.publish_mcp(controller, 99, 99, session.id)
        assert counts.snapshot(controller, session.id) == absent
        counts.snapshot = lambda *_: absent
        try:
            with pytest.raises(RuntimeError, match="definitions changed"):
                backend_tool_counts(controller)
        finally:
            # Restore descriptor lookup, not an instance-bound method shadow.
            del counts.snapshot
        with monkeypatch.context() as changed:
            changed.setattr(
                count_source, "backend_tool_counts",
                lambda *_: pytest.fail("Changed resolver was invoked"),
            )
            with pytest.raises(RuntimeError, match="resolver changed"):
                controller._publish_selected_tool_count(session.id, 99)
        controller._publish_selected_tool_count(session.id, 5)
        backend.store.close_session(session.id)
        assert session.id not in backend.store._sessions
        assert counts.snapshot(controller, None).selected_count is None
        assert session.id not in counts._selected
    finally:
        await backend.aclose()
