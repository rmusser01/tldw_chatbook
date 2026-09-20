"""Artifact controller reads release native caches at the finite worker boundary."""

import asyncio
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from Tests.private_profile import private_profile_test
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Library.library_artifacts_catalog import LibraryArtifactsCatalog
from tldw_chatbook.Library.library_artifacts_state import ArtifactKey, ArtifactScope
from tldw_chatbook.UI.Library_Modules.library_artifacts_controller import (
    LibraryArtifactsController,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["page", "locate", "detail"])
@pytest.mark.parametrize("outcome", ["success", "error", "cancel", "borrowed"])
@private_profile_test
async def test_artifact_controller_retires_only_owned_worker_cache(
    tmp_path, request, monkeypatch, operation, outcome
):
    db = CharactersRAGDB(tmp_path / "artifacts.sqlite", client_id="artifact-worker")
    saved = db.create_kept_briefing(
        source_briefing_id=1,
        watchlist_name="Saved report",
        body_markdown="# Complete saved report",
        origin="manual",
    )
    main_connection = db.get_connection()
    key = ArtifactKey("kept_report", saved)
    scope = ArtifactScope(kept_only=True)
    catalog = LibraryArtifactsCatalog(subscriptions_db=None, chachanotes_db=db)
    method = {"page": "read_page", "locate": "locate", "detail": "read_detail"}[
        operation
    ]
    original = getattr(catalog, method)
    entered, release = threading.Event(), threading.Event()
    observed = []

    def read(*args, **kwargs):
        result = original(*args, **kwargs)
        observed.append(db.get_connection())
        entered.set()
        if outcome == "error":
            raise ValueError("injected artifact read failure")
        if outcome == "cancel":
            assert release.wait(10)
        return result

    monkeypatch.setattr(catalog, method, read)
    controller = LibraryArtifactsController.__new__(LibraryArtifactsController)
    controller.screen = SimpleNamespace(
        _run_library_service_call=LibraryScreen._run_library_service_call
    )
    # Publication is irrelevant here; exercise actual controller dispatch and storage.
    controller.disposed = True
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=1)
    previous_executor = loop._default_executor
    loop.set_default_executor(pool)
    try:
        if outcome == "borrowed":
            await loop.run_in_executor(pool, db.get_connection)
        if operation == "detail":
            operation_task = controller._load_detail(catalog, ((), scope, key, "", 0))
        else:
            operation_task = controller._load_page(
                catalog,
                0,
                (),
                scope,
                None,
                "after",
                key if operation == "locate" else None,
            )
        task = asyncio.create_task(operation_task)
        try:
            if outcome == "cancel":
                for _ in range(200):
                    if entered.is_set():
                        break
                    await asyncio.sleep(0.01)
                assert entered.is_set()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert not release.is_set()
            else:
                await task
        finally:
            release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        def check_on_owner_thread():
            assert len(observed) == 1
            connection = observed[0]
            if outcome == "borrowed":
                assert connection.execute("SELECT 1").fetchone()[0] == 1
                db.close_connection()
            else:
                with pytest.raises(sqlite3.ProgrammingError):
                    connection.execute("SELECT 1")

        # A queued job on the same one-thread pool joins the cancelled worker too.
        await loop.run_in_executor(pool, check_on_owner_thread)
        assert main_connection.execute("SELECT 1").fetchone()[0] == 1
        db.close_connection()
        participant = db._maintenance_participant
        assert not participant.connections
        participant.close_admission()
        try:
            assert participant.drain(time.monotonic() + 0.1)
        finally:
            participant.resume()
    finally:
        release.set()
        await loop.run_in_executor(pool, db.close_connection)
        loop._default_executor = previous_executor
        pool.shutdown(wait=True)
        db.close_connection()
