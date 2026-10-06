"""Real projection entry fact restores across nested body/error/cancellation."""

import asyncio

import pytest

from Tests.private_profile import private_profile_test

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("fault", [None, "error", "cancel"])
async def test_actual_projection_nested_entry_fact_retires(tmp_path, request, fault):
    from Tests.UI.test_console_checked_display_scope import _screen
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend

    database, _store, _controller, screen, tasks = _screen(tmp_path)
    projection = spend.ConsoleReadinessConfigProjection.for_screen(screen)
    try:
        assert projection.run(lambda: None) is False
        await asyncio.gather(*tasks)
        assert projection._display_proof in spend._checked_display_proofs
        assert not projection.pending and projection._settled.is_set()
        assert getattr(projection, "_display_refresh_pending_at_entry", None) is None
        before_active = getattr(screen, "_console_readiness_projection_active", None)
        await asyncio.sleep(1.02)
        scheduled_before = len(tasks)
        stages = []

        def inner():
            assert projection._display_refresh_pending_at_entry is True
            assert screen._console_readiness_projection_active[2] is projection
            stages.append("inner")
            if fault == "error":
                raise ValueError("entry fact body error")
            if fault == "cancel":
                raise asyncio.CancelledError()
            return False

        def outer():
            # This actual direct run creates the queued original refresh.
            assert projection.pending and projection._read_request is None
            assert projection._display_refresh_pending_at_entry is False
            parent_active = screen._console_readiness_projection_active
            try:
                assert projection.run(inner) is False
            finally:
                assert projection._display_refresh_pending_at_entry is False
                assert screen._console_readiness_projection_active is parent_active
                stages.append("outer_restored")
            return True

        if fault is None:
            assert projection.run(outer) is True
        else:
            error = ValueError if fault == "error" else asyncio.CancelledError
            with pytest.raises(error):
                projection.run(outer)
        assert stages == ["inner", "outer_restored"]
        assert projection._display_refresh_pending_at_entry is None
        assert screen._console_readiness_projection_active is before_active
        assert len(tasks) == scheduled_before + 1
        assert not tasks[scheduled_before].done() and projection.pending
        await tasks[scheduled_before]
        assert not projection.pending and projection._settled.is_set()
        assert projection._display_refresh_pending_at_entry is None
        request.node.user_properties.append(
            ("actual_projection_nested_restoration", fault or "normal")
        )
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()
        with storage._lock:
            assert not storage._pending_acquisitions
            assert not storage._operations
            assert not storage._raw_operations
            assert not storage._retiring_holds
            assert not raw._states
            assert not (storage._live_leases - set(storage._startups.values()))
