"""Readiness publication shares the original poll's presentation ownership.

These are scheduling controls, using the existing cadence screen double and
the real finite readiness reader. Mounted config/currentness/lifetime controls
remain in test_console_readiness_config_projection.py.
"""

import asyncio
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from Tests.UI.test_console_config_sync_cadence_inputs import _Screen
from Tests.UI.test_console_readiness_config_projection import _read_result
from tldw_chatbook.UI.Console_Modules import config_sync
from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


def _publication_screen():
    screen = _Screen()
    owners = [
        SimpleNamespace(id=value, workspace_id=None)
        for value in screen._console_chat_store.session_ids
    ]
    screen._console_chat_store.sessions = lambda: owners
    screen.app_instance.app_config = {"configured": True}
    screen.app_instance.chachanotes_db = object()
    screen._console_derivation_scope = nullcontext
    screen.set_timer = lambda *_args, **_kwargs: None
    tasks, publications = [], []

    def schedule(work, **options):
        if options.get("group") == "console-readiness-publication":
            # As the existing idle-publication test does, observe the queued
            # original callback before worker admission. Do not execute an
            # additional FULL behind the awaiting owner's back.
            publications.append((work, options))
            return None
        task = asyncio.create_task(work() if callable(work) else work)
        tasks.append(task)
        return task

    screen.run_worker = schedule
    return screen, tasks, publications


async def test_changed_readiness_publication_shares_awaiting_full_and_settlement():
    screen, tasks, publications = _publication_screen()
    screen._console_transcript_sync_timer = object()
    entered, release, retired = (threading.Event() for _ in range(3))
    values, reads, rendered = {"version": 1}, [], []

    def read():
        captured = dict(values)
        reads.append(threading.get_ident())
        if len(reads) == 1:
            try:
                entered.set()
                assert release.wait(3)
            finally:
                retired.set()
        return _read_result(captured)

    projection = spend.ConsoleReadinessConfigProjection(screen, read_current=read)
    screen._console_readiness_config_projection = projection
    original_render = screen._sync_console_rail_and_controls

    def render():
        rendered.append(dict(projection.value))
        return original_render()

    screen._sync_console_rail_and_controls = render

    async def awaiting_full():
        # This is the original FULL's warm-then-project dependency, without
        # claiming this decision-level double is a mounted ChatScreen.
        screen._console_sync_in_progress = True
        try:
            assert await projection.warm()
            assert config_sync.project_console_tick(screen)
        finally:
            screen._console_sync_in_progress = False

    full = asyncio.create_task(awaiting_full())
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        assert not full.done() and screen._console_sync_in_progress
        assert projection.pending and not projection._settled.is_set()
        assert rendered == [] and publications == []
        release.set()
        await asyncio.wait_for(full, 3)
        await asyncio.gather(*tasks)
        assert retired.is_set() and not projection._reads
        assert projection._settled.is_set() and not projection.pending
        assert reads == [reads[0]] and reads[0] != threading.get_ident()
        assert projection.value == {"version": 1}
        assert rendered == [{"version": 1}]
        # Original RED: all successful read/retirement/render prerequisites
        # precede the redundant FULL-enqueue oracle.
        assert publications == [], "the awaiting FULL already consumed this read"

        # A later changed publication is still an input, even at the same
        # cadence clock. Stopping the poll must render the unserved tail.
        values["version"] = 2
        await projection._refresh(projection._key())
        assert projection.value == {"version": 2}
        assert len(reads) == 2 and not projection._reads
        assert publications == [] and len(rendered) == 1
        screen._console_transcript_sync_timer = None
        config_sync.settle_stopped_poll(screen, lambda item: item.torn_down)
        assert rendered == [{"version": 1}, {"version": 2}]
        assert not config_sync._cadence(screen).owed

        # Subsequent independent decision state must still reconcile through
        # the same original cadence, rather than being covered by old data.
        screen._console_transcript_sync_timer = object()
        screen._console_chat_controller.rounds["s1"] = {"approval": 1}
        await config_sync.run_periodic_console_sync(screen._sync_native_console_chat_ui)
        assert len(rendered) == 3
    finally:
        release.set()
        await asyncio.gather(full, *tasks, return_exceptions=True)


async def test_idle_readiness_publication_keeps_original_custom_full_callback():
    screen, tasks, publications = _publication_screen()
    entered = []

    async def custom_full():
        entered.append(projection.value)

    screen._sync_native_console_chat_ui = custom_full
    projection = spend.ConsoleReadinessConfigProjection(
        screen, read_current=lambda: _read_result({"version": 1})
    )
    try:
        await projection._refresh(projection._key())
        assert projection.value == {"version": 1} and not projection._reads
        assert projection._settled.is_set() and entered == []
        assert publications == [
            (
                custom_full,
                {"exclusive": False, "group": "console-readiness-publication"},
            )
        ]
        await publications[0][0]()
        assert entered == [{"version": 1}]
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
