"""Guardian observations belong to the captured draft, through delayed work."""
import asyncio
import threading

import pytest

from Tests.Guardian.test_check_pipeline import FIXED_NOW, _fresh_db
from Tests.Guardian.test_dispatch_seam import (
    _calls,
    _FakeChatController,
    _seam_controller,
)
from tldw_chatbook.Guardian import settings
from tldw_chatbook.Guardian.check_pipeline import GuardianChecker


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["notify", "block"])
async def test_delayed_guardian_notice_keeps_captured_chat(tmp_path, monkeypatch, action):
    monkeypatch.setattr(settings, "guardian_setting", lambda key, default=None: True if key == "enabled" else default)
    db = _fresh_db(tmp_path, [{"name":"private topic", "topic":"private", "pattern":"private phrase", "action":action}])
    controller = _FakeChatController()
    rows = []
    calls = _calls()
    async def append_visible(text):
        rows.append((controller.store.active_session_id, text))
    async def append_captured(text, session_id):
        rows.append((session_id, text))
    checker = GuardianChecker(db_getter=lambda:db, session_id_getter=lambda:controller.store.active_session_id, notify=lambda *args:None, append_system_row=append_visible, now=lambda:FIXED_NOW)
    checker._append_captured_system_row = append_captured
    seam = _seam_controller(lambda:checker, calls)
    seam._chat_controller_accessor = lambda:controller
    seam._append_captured_system_message = append_captured
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    original = checker._record_matches
    def paused(*args):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        return original(*args)
    monkeypatch.setattr(checker, "_record_matches", paused)
    try:
        task = asyncio.create_task(seam.dispatch("private phrase", session_id="session-a"))
        await asyncio.wait_for(started.wait(), 3)
        controller.store.active_session_id = "session-b"
        release.set()
        await asyncio.wait_for(task, 3)
        assert rows and all(owner == "session-a" for owner, text in rows)
        assert calls["notified"] == []
        assert calls["focused"] == []
        with db.connection() as connection:
            assert {row[0] for row in connection.execute("SELECT session_id FROM guardian_alerts")} == {"session-a"}
    finally:
        release.set()
        db.close()


@pytest.mark.asyncio
async def test_cached_checker_disable_reads_no_store_again(tmp_path, monkeypatch):
    enabled = [True]
    monkeypatch.setattr(settings, "guardian_setting", lambda key, default=None: enabled[0] if key == "enabled" else default)
    db = _fresh_db(tmp_path, [])
    reads = []
    checker = GuardianChecker(db_getter=lambda:(reads.append(True) or db), session_id_getter=lambda:"session-a", notify=lambda *args:None, append_system_row=lambda text:None, now=lambda:FIXED_NOW)
    try:
        await checker.check("first")
        enabled[0] = False
        result = await checker.check("second")
        assert result["action"] == "allow"
        assert reads == [True]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_cancelled_check_is_drained_before_captured_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "guardian_setting", lambda key, default=None: True if key == "enabled" else default)
    db = _fresh_db(tmp_path, [{"name": "private topic", "topic": "private", "pattern": "private phrase"}])
    main_thread = threading.get_ident()
    getter_threads = []
    visible = ["session-a"]
    rows = []
    checker = GuardianChecker(db_getter=lambda: (getter_threads.append(threading.get_ident()) or db),
        session_id_getter=lambda: visible[0], notify=lambda *args: None,
        append_system_row=rows.append, now=lambda: FIXED_NOW)
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    original = checker._record_matches
    def paused(*args):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        return original(*args)
    monkeypatch.setattr(checker, "_record_matches", paused)
    try:
        checking = asyncio.create_task(checker.check("private phrase", session_id="session-a"))
        await asyncio.wait_for(started.wait(), 3)
        checking.cancel()
        with pytest.raises(asyncio.CancelledError):
            await checking
        visible[0] = "session-b"
        finalizing = asyncio.create_task(checker.finalize_visit_async())
        await asyncio.sleep(0)
        assert not finalizing.done()
        assert checker._pending_store_work
        release.set()
        payload = await asyncio.wait_for(finalizing, 3)
        assert payload["session_id"] == "session-a"
        assert payload["per_topic_counts"] == {"private": 1}
        assert rows == []
        assert not checker._pending_store_work
        assert set(getter_threads) == {main_thread}
    finally:
        release.set()
        await checker.drain()
        db.close()


@pytest.mark.asyncio
async def test_one_screen_visits_keep_each_chat_summary_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "guardian_setting", lambda key, default=None: True if key == "enabled" else default)
    db = _fresh_db(tmp_path, [{"name": "private topic", "topic": "private", "pattern": "private phrase"}])
    checker = GuardianChecker(db_getter=lambda: db, session_id_getter=lambda: "other",
        notify=lambda *args: None, append_system_row=lambda text: None, now=lambda: FIXED_NOW)
    try:
        await checker.check("private phrase", session_id="session-a")
        await checker.check("private phrase", session_id="session-b")
        payload = await checker.finalize_visit_async()
        summaries = payload["session_summaries"]
        assert {s["session_id"] for s in summaries} == {"session-a", "session-b"}
        assert len({s["visit_id"] for s in summaries}) == 2
        assert all(s["per_topic_counts"] == {"private": 1} for s in summaries)
    finally:
        db.close()
