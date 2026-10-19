"""Coordinator contention defers presentation without native fallback."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from Tests.private_profile import private_profile_test

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("edge", ["entry", "exit", "pricing", "capacity"])
async def test_checked_display_defers_contended_coordinator(tmp_path, request, edge):
    from Tests.UI import test_console_checked_display_scope as fixture
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
    from tldw_chatbook.UI.Console_Modules.config_sync import (
        run_console_presentation_sync,
    )

    database, _store, _controller, screen, tasks = fixture._screen(tmp_path)
    held, release, timed_out = threading.Event(), threading.Event(), []
    retries, rendered, after_helper = [], [], []
    screen._request_console_control_bar_sync = lambda **_: retries.append(True)

    def hold():
        with storage._lock:
            held.set()
            if not release.wait(2):
                timed_out.append(True)

    holder = threading.Thread(target=hold)

    def contend():
        holder.start()
        assert held.wait(2)

    def forbidden_fallback(*args):
        pytest.fail("contended checked display entered native/custom fallback")

    def render():
        rendered.append(True)
        if edge != "entry":
            contend()
        if edge == "pricing":
            spend.pricing_catalog_for_display(screen, forbidden_fallback)
            after_helper.append(True)
        elif edge == "capacity":
            spend.cached_context_window_for_display(
                screen, SimpleNamespace(cached_context_window=forbidden_fallback), None
            )
            after_helper.append(True)

    try:
        await fixture._warm(screen, tasks)
        if edge == "entry":
            contend()
        with fixture._actual_calls() as calls:
            assert run_console_presentation_sync(screen, render) is False
        assert not timed_out, "presentation waited for the coordinator"
        assert rendered == ([] if edge == "entry" else [True])
        assert retries and not after_helper
        assert calls["main_scopes"] == calls["main_opens"] == 0, calls
        release.set()
        holder.join(2)
        assert not holder.is_alive()
        assert run_console_presentation_sync(screen, lambda: after_helper.append(True))
        assert after_helper == [True]
        # A later close is still refused; lock availability grants no authority.
        screen._closing = True
        assert not run_console_presentation_sync(screen, forbidden_fallback)
    finally:
        release.set()
        if holder.ident is not None:
            holder.join(3)
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("helper", ["pricing", "capacity"])
async def test_expired_native_display_retires_before_contention_retry(
    tmp_path, request, helper
):
    from Tests.UI import test_console_checked_display_scope as fixture
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
    from tldw_chatbook.UI.Console_Modules.config_sync import (
        run_console_presentation_sync,
    )

    database, _store, _controller, screen, tasks = fixture._screen(tmp_path)
    held, release = threading.Event(), threading.Event()
    operations, retries = [], []

    def retry(**_):
        assert operations and operations[0] not in raw._states
        retries.append(True)

    screen._request_console_control_bar_sync = retry

    def hold():
        with storage._lock:
            held.set()
            release.wait(2)

    holder = threading.Thread(target=hold)

    def forbidden_fallback(*args):
        pytest.fail("contended display helper tried synchronous fallback")

    def render():
        operations.append(getattr(raw._local, "operation", None))
        assert operations[0] in raw._states
        holder.start()
        assert held.wait(2)
        try:
            if helper == "pricing":
                spend.pricing_catalog_for_display(screen, forbidden_fallback)
            else:
                spend.cached_context_window_for_display(
                    screen,
                    SimpleNamespace(cached_context_window=forbidden_fallback),
                    None,
                )
        finally:
            release.set()  # Native retirement must acquire the coordinator.
            holder.join(2)

    try:
        projection = await fixture._warm(screen, tasks)
        await asyncio.sleep(projection.max_age + 0.02)
        assert not projection.pending
        assert run_console_presentation_sync(screen, render) is False
        assert retries == [True]
        assert operations[0] not in raw._states
        await asyncio.gather(*tasks)
        assert run_console_presentation_sync(screen, lambda: None)
    finally:
        release.set()
        if holder.ident is not None:
            holder.join(3)
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()
