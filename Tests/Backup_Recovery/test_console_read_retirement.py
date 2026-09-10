"""Observed Console/cleanup worker reads retire native handles on return."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio, threading
from pathlib import Path
from types import SimpleNamespace
import sys
from loguru import logger
from tldw_chatbook.app import TldwCli
from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases

route, outcome = sys.argv[1:]
root = Path.home()
if route == "history":
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from Tests.UI.test_console_workspace_controller import _workspace_controller
    db = CharactersRAGDB(root / "chat.db", "retire-test")
    service = ChatConversationService(db)
    service.create_conversation(title="retained")
    controller = _workspace_controller(app_instance=SimpleNamespace(local_chat_conversation_service=service))
    target, method = db, "search_conversations_page"
    call = controller._persisted_console_browser_rows
elif route == "changes":
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.UI.Screens.change_review_screen import AgentRunsChangeReviewProvider
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    db = AgentRunsDB(root / "runs.db")
    provider = AgentRunsChangeReviewProvider(db=db, service=None, conversation_id="none", diff_display_max_lines=50)
    target, method = db, "change_snapshots_for_conversation"
    async def call():
        jobs = []
        host = SimpleNamespace(_console_change_review_provider=lambda:provider,
            _console_changed_files_row_cache={},
            _land_console_changed_files=lambda *a:None,
            _land_console_changed_files_empty=lambda *a:None,
            app=SimpleNamespace(call_from_thread=lambda f,*a:f(*a)),
            run_worker=lambda f,**kw:jobs.append(asyncio.create_task(asyncio.to_thread(f))))
        ChatScreen._dispatch_console_changed_files_worker(host, "none")
        await jobs[0]
else:
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    db = MediaDatabase(root / "media.db", client_id="retire-test")
    target, method = db, "get_deletion_candidates"
    host = SimpleNamespace(media_db=db, loguru_logger=logger, notify=lambda *a,**kw:None)
    call = lambda:TldwCli.perform_media_cleanup(host)

entered, release, exited = threading.Event(), threading.Event(), threading.Event()
original = getattr(target, method)
def read(*args, **kwargs):
    try:
        result = original(*args, **kwargs)
        entered.set()
        if outcome == "cancel":
            assert release.wait(5)
        if outcome == "error":
            raise ValueError("injected read failure")
        return result
    finally:
        exited.set()
setattr(target, method, read)
async def main():
    task = asyncio.create_task(call())
    try:
        if outcome == "cancel":
            assert await asyncio.to_thread(entered.wait, 3)
            assert worker_leases(db)
            task.cancel()
            try: await task
            except asyncio.CancelledError: pass
            assert worker_leases(db)
            release.set()
            assert await asyncio.to_thread(exited.wait, 3)
            for _ in range(100):
                if not worker_leases(db): break
                await asyncio.sleep(.01)
        else:
            await task
        assert entered.is_set()
        assert not worker_leases(db), "worker retained lease"
        setattr(target, method, original)
        await call()
        assert not worker_leases(db)
        print("retired and reopened")
    finally:
        release.set()
        if hasattr(db,"close_connection"): db.close_connection()
        else: db.close()
asyncio.run(main())
"""


@pytest.mark.parametrize("route", ["history", "changes", "cleanup"])
@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
def test_console_read_retires_actual_worker_database(tmp_path, route, outcome):
    _run(tmp_path, route, outcome, script=_SCRIPT)
