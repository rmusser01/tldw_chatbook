"""task-15467: thread local media-reading service calls off the event loop.

Evidence for the audit finding in `Docs/Design/2026-08-11-input-latency-audit.md`
("Blocking I/O in click paths. Media hub: `run_worker(coroutine)` != a thread;
every search/filter/item click runs sync SQLite on the loop
(`MediaWindow_v2.py:2387/:1188`)"): `MediaReadingScopeService` routed every
LOCAL-mode call straight through `_maybe_await` to a plain synchronous
`LocalMediaReadingService` method. `run_worker(coroutine)` does not leave the
event loop, so those sync sqlite calls ran inline on the loop whenever a
search/filter/pagination/item-click worker fired.

Two kinds of evidence, mirroring task-283's `Tests/Chat/test_search_off_loop.py`
and task-15463's `Tests/Subscriptions/test_watchlists_db_instance_and_off_loop.py`:

  * Thread-affinity doubles: a file-backed local service's leaf call records
    `threading.get_ident()` and must NOT match the caller/event-loop thread;
    a `:memory:`-backed local service (and every server-mode call, sync or
    async) must stay on the caller thread -- threading a `:memory:` DB would
    hand a worker thread an empty, unmigrated database (the task-283 hazard).

  * A real `MediaDatabase` + `sqlite3.Connection.set_trace_callback` bound to
    the calling (event-loop) thread's own connection: since `MediaDatabase`
    keeps thread-local connections, any statement this callback observes is
    by construction a statement that ran on the calling thread -- no timing,
    no sampling, no flake. Covers the exact two audited call chains: a
    search/browse call, and the full item-click detail chain (detail +
    reading progress + highlights + document versions).
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService
from tldw_chatbook.Media.media_reading_scope_service import MediaReadingScopeService


# ---------------------------------------------------------------------------
# Thread-affinity doubles
# ---------------------------------------------------------------------------


class _RecordingLocalService:
    """Plain-sync local service double recording the thread each leaf ran on."""

    def __init__(self, *, is_memory_db: bool):
        self.media_db = SimpleNamespace(is_memory_db=is_memory_db)
        self.thread_idents: dict[str, list[int]] = {}
        self.calls: list[tuple[str, Any]] = []

    def _record(self, name: str, *args: Any) -> None:
        self.thread_idents.setdefault(name, []).append(threading.get_ident())
        self.calls.append((name, args))

    def search_media(self, *, query=None, limit=20, offset=0, **kwargs):
        self._record("search_media", query)
        return {"items": [], "total": 0, "offset": offset, "limit": limit}

    def list_library_media_types(self):
        self._record("list_library_media_types")
        return ["article"]

    def get_media_detail(self, media_id):
        self._record("get_media_detail", media_id)
        return {"id": media_id, "title": "Detail", "type": "article"}

    def get_reading_progress(self, media_id):
        self._record("get_reading_progress", media_id)
        return {"media_id": media_id, "current_page": 1, "total_pages": 5}

    def list_highlights(self, item_id):
        self._record("list_highlights", item_id)
        return []

    def list_document_versions(self, media_id, include_deleted=False):
        self._record("list_document_versions", media_id)
        return []

    def delete_media(self, media_id):
        self._record("delete_media", media_id)
        return True

    def undelete_media(self, media_id):
        self._record("undelete_media", media_id)
        return True

    # task-15768: the scope service dispatches the unprefixed highlight leaf
    # names -- the only ones the real LocalMediaReadingService implements.
    def create_highlight(self, item_id, **kwargs):
        self._record("create_highlight", item_id)
        return {"id": 5, "item_id": item_id, "quote": kwargs.get("quote", "")}

    def update_highlight(self, highlight_id, **changes):
        self._record("update_highlight", highlight_id)
        return {"id": highlight_id, "item_id": 1, "quote": "q"}

    def delete_highlight(self, highlight_id):
        self._record("delete_highlight", highlight_id)
        return True

    def save_analysis_version(
        self, media_id, *, content, analysis_content, prompt=None
    ):
        self._record("save_analysis_version", media_id)
        return {"media_id": media_id}

    def overwrite_analysis_version(
        self, media_id, *, content, analysis_content, prompt=None
    ):
        self._record("overwrite_analysis_version", media_id)
        return {"media_id": media_id}

    def delete_analysis_version(self, version_uuid):
        self._record("delete_analysis_version", version_uuid)
        return True

    def save_to_read_it_later(self, media_id):
        self._record("save_to_read_it_later", media_id)
        return True

    def remove_from_read_it_later(self, media_id):
        self._record("remove_from_read_it_later", media_id)
        return True

    def update_media_metadata(self, media_id, **metadata):
        self._record("update_media_metadata", media_id)
        return {"id": media_id, **metadata}


class _RecordingServerService:
    """Async server double -- server mode must never thread."""

    def __init__(self):
        self.thread_idents: dict[str, list[int]] = {}

    def _record(self, name: str) -> None:
        self.thread_idents.setdefault(name, []).append(threading.get_ident())

    async def search_media(self, *, query=None, limit=20, offset=0, **kwargs):
        self._record("search_media")
        return {"items": [], "total": 0, "offset": offset, "limit": limit}

    async def get_media_detail(self, media_id):
        self._record("get_media_detail")
        return {"id": media_id, "title": "Detail"}

    async def get_reading_progress(self, media_id):
        self._record("get_reading_progress")
        return None


def _scope(*, is_memory_db: bool) -> tuple[MediaReadingScopeService, _RecordingLocalService]:
    local = _RecordingLocalService(is_memory_db=is_memory_db)
    scope = MediaReadingScopeService(local_service=local, server_service=None)
    return scope, local


# Each case: (label, coroutine factory taking the scope service, expected leaf
# call names that must have run -- usually one, two for get_media_detail
# which also fetches reading progress inline).
LOCAL_LEAF_CASES: list[tuple[str, Any, list[str]]] = [
    (
        "search_media",
        lambda s: s.search_media(mode="local", query="q"),
        ["search_media"],
    ),
    (
        "list_library_media_types",
        lambda s: s.list_library_media_types(mode="local"),
        ["list_library_media_types"],
    ),
    (
        "list_read_it_later",
        lambda s: s.list_read_it_later(mode="local", query="q"),
        ["search_media"],
    ),
    (
        "get_media_detail",
        lambda s: s.get_media_detail(mode="local", media_id=1),
        ["get_media_detail", "get_reading_progress"],
    ),
    (
        "get_reading_progress",
        lambda s: s.get_reading_progress(mode="local", media_id=1),
        ["get_reading_progress"],
    ),
    (
        "list_reading_highlights",
        lambda s: s.list_reading_highlights(mode="local", media_id=1),
        ["list_highlights"],
    ),
    (
        "list_document_versions",
        lambda s: s.list_document_versions(mode="local", media_id=1),
        ["list_document_versions"],
    ),
    (
        "delete_media",
        lambda s: s.delete_media(mode="local", media_id=1),
        ["delete_media"],
    ),
    (
        "undelete_media",
        lambda s: s.undelete_media(mode="local", media_id=1),
        ["undelete_media"],
    ),
    (
        "create_reading_highlight",
        lambda s: s.create_reading_highlight(mode="local", media_id=1, quote="hi"),
        ["create_highlight"],
    ),
    (
        "update_reading_highlight",
        lambda s: s.update_reading_highlight(mode="local", highlight_id=5, note="hi"),
        ["update_highlight"],
    ),
    (
        "delete_reading_highlight",
        lambda s: s.delete_reading_highlight(mode="local", highlight_id=5),
        ["delete_highlight"],
    ),
    (
        "save_analysis_version",
        lambda s: s.save_analysis_version(
            mode="local", media_id=1, content="c", analysis_content="a"
        ),
        ["save_analysis_version"],
    ),
    (
        "overwrite_analysis_version",
        lambda s: s.overwrite_analysis_version(
            mode="local", media_id=1, content="c", analysis_content="a"
        ),
        ["overwrite_analysis_version"],
    ),
    (
        "delete_analysis_version",
        lambda s: s.delete_analysis_version(mode="local", version_uuid="v1"),
        ["delete_analysis_version"],
    ),
    (
        "save_to_read_it_later",
        lambda s: s.save_to_read_it_later(mode="local", media_id=1),
        ["save_to_read_it_later"],
    ),
    (
        "remove_from_read_it_later",
        lambda s: s.remove_from_read_it_later(mode="local", media_id=1),
        ["remove_from_read_it_later"],
    ),
    (
        "update_media_metadata",
        lambda s: s.update_media_metadata(mode="local", media_id=1, title="t"),
        ["update_media_metadata"],
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("label,call,leaves", LOCAL_LEAF_CASES, ids=[c[0] for c in LOCAL_LEAF_CASES])
async def test_local_leaf_threads_off_the_calling_thread_when_file_backed(
    label, call, leaves
):
    scope, local = _scope(is_memory_db=False)
    caller_thread = threading.get_ident()

    await call(scope)

    for leaf in leaves:
        idents = local.thread_idents.get(leaf)
        assert idents, f"{label}: leaf {leaf!r} was never called"
        assert all(ident != caller_thread for ident in idents), (
            f"{label}: leaf {leaf!r} ran on the calling (event-loop) thread "
            f"instead of a worker thread"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("label,call,leaves", LOCAL_LEAF_CASES, ids=[c[0] for c in LOCAL_LEAF_CASES])
async def test_local_leaf_stays_inline_when_memory_backed(label, call, leaves):
    scope, local = _scope(is_memory_db=True)
    caller_thread = threading.get_ident()

    await call(scope)

    for leaf in leaves:
        idents = local.thread_idents.get(leaf)
        assert idents, f"{label}: leaf {leaf!r} was never called"
        assert all(ident == caller_thread for ident in idents), (
            f"{label}: leaf {leaf!r} ran off the calling thread for a "
            f":memory:-backed DB -- this would hand a worker thread an "
            f"empty, unmigrated database"
        )


@pytest.mark.asyncio
async def test_local_leaf_threads_an_unrecognized_local_double_with_no_media_db():
    """Positive-confirmation predicate (task-283 lesson): a local service the
    seam cannot positively confirm as memory-backed still threads -- there is
    no negative branch that silently keeps an unrecognized shape inline.
    """

    class _BareLocalDouble:
        def __init__(self):
            self.thread_idents: list[int] = []

        def search_media(self, *, query=None, limit=20, offset=0, **kwargs):
            self.thread_idents.append(threading.get_ident())
            return {"items": [], "total": 0}

    local = _BareLocalDouble()
    scope = MediaReadingScopeService(local_service=local, server_service=None)
    caller_thread = threading.get_ident()

    await scope.search_media(mode="local", query="q")

    assert local.thread_idents
    assert local.thread_idents[0] != caller_thread


@pytest.mark.asyncio
async def test_server_mode_never_threads_even_for_a_sync_double():
    """The threading gate is mode == LOCAL, not merely "is this sync?" --
    mirrors ChatConversationScopeService's server-mode carve-out.
    """

    class _SyncServerDouble:
        def __init__(self):
            self.thread_idents: list[int] = []

        def search_media(self, *, query=None, limit=20, offset=0, **kwargs):
            self.thread_idents.append(threading.get_ident())
            return {"items": [], "total": 0}

    server = _SyncServerDouble()
    scope = MediaReadingScopeService(local_service=None, server_service=server)
    caller_thread = threading.get_ident()

    await scope.search_media(mode="server", query="q")

    assert server.thread_idents == [caller_thread]


@pytest.mark.asyncio
async def test_server_mode_async_double_runs_inline_as_before():
    server = _RecordingServerService()
    scope = MediaReadingScopeService(local_service=None, server_service=server)
    caller_thread = threading.get_ident()

    await scope.get_media_detail(mode="server", media_id=41)

    assert server.thread_idents["get_media_detail"] == [caller_thread]


# ---------------------------------------------------------------------------
# Real MediaDatabase + sqlite trace callback: the "15463 evidence-suite
# pattern" -- no sampling, no timing, no flake.
# ---------------------------------------------------------------------------


@pytest.fixture
def real_local_scope(tmp_path):
    db = MediaDatabase(str(tmp_path / "media_off_loop.db"), client_id="off_loop_test")
    media_id, _, _ = db.add_media_with_keywords(
        title="Offloop Fixture",
        content="Body text for the offloop probe.",
        media_type="article",
        keywords=[],
    )
    local_service = LocalMediaReadingService(db)
    scope = MediaReadingScopeService(local_service=local_service, server_service=None)
    try:
        yield scope, db, media_id
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_search_media_runs_no_sqlite_on_the_event_loop(real_local_scope):
    """AC#1: the search/browse/pagination/keyword-filter leaf.

    `db.get_connection()` returns the connection belonging to THIS thread --
    the thread the event loop runs on. `MediaDatabase` connections are
    thread-local, so any statement this callback sees is, by construction, a
    statement that ran on the event-loop thread; the worker's own
    thread-local connection is invisible to it.
    """
    scope, db, media_id = real_local_scope
    loop_statements: list[str] = []
    db.get_connection().set_trace_callback(loop_statements.append)
    try:
        result = await scope.search_media(mode="local", query="Offloop")
    finally:
        db.get_connection().set_trace_callback(None)

    assert not loop_statements, (
        "media search ran SQL on the event-loop thread: "
        f"{loop_statements[:5]}"
    )
    assert result["total"] == 1
    assert result["items"][0]["title"] == "Offloop Fixture"


@pytest.mark.asyncio
async def test_library_media_types_run_no_sqlite_on_the_event_loop(real_local_scope):
    scope, db, _media_id = real_local_scope
    loop_statements: list[str] = []
    db.get_connection().set_trace_callback(loop_statements.append)
    try:
        media_types = await scope.list_library_media_types(mode="local")
    finally:
        db.get_connection().set_trace_callback(None)

    assert loop_statements == []
    assert media_types == ["article"]


@pytest.mark.asyncio
async def test_item_click_detail_chain_runs_no_sqlite_on_the_event_loop(
    real_local_scope,
):
    """AC#1: the item-click chain -- detail (which internally also fetches
    reading progress), reading highlights, and document versions -- the
    sequential queries the audit named at `MediaWindow_v2.py:1188-1191`.

    ``list_reading_highlights`` is exercised against the real
    ``LocalMediaReadingService`` since task-15768 fixed the scope service's
    highlight dispatch to the unprefixed leaf names the local service
    actually implements (it previously AttributeError'd here, which is why
    an older revision of this test had to leave it out).
    """
    scope, db, media_id = real_local_scope
    loop_statements: list[str] = []
    db.get_connection().set_trace_callback(loop_statements.append)
    try:
        detail = await scope.get_media_detail(mode="local", media_id=media_id)
        highlights = await scope.list_reading_highlights(
            mode="local", media_id=media_id
        )
        versions = await scope.list_document_versions(
            mode="local", media_id=media_id
        )
    finally:
        db.get_connection().set_trace_callback(None)

    assert not loop_statements, (
        "the item-click detail chain ran SQL on the event-loop thread: "
        f"{loop_statements[:5]}"
    )
    # Never vacuous: the chain really did fetch real data, including the
    # reading-progress leaf `get_media_detail` fetches internally, and the
    # document version `add_media_with_keywords` creates automatically.
    assert detail["title"] == "Offloop Fixture"
    assert "reading_progress" in detail
    assert highlights == []
    assert len(versions) == 1
    assert versions[0]["media_id"] == media_id


@pytest.mark.asyncio
async def test_memory_backed_real_db_stays_on_the_calling_thread(tmp_path):
    """A `:memory:` MediaDatabase must NOT be threaded -- it is only visible
    to the thread that created it (the task-283 hazard). This positively
    proves the opposite of the two tests above for the one case where
    staying inline is correct.
    """
    db = MediaDatabase(":memory:", client_id="off_loop_memory_test")
    try:
        media_id, _, _ = db.add_media_with_keywords(
            title="Memory Fixture",
            content="Body",
            media_type="article",
            keywords=[],
        )
        local_service = LocalMediaReadingService(db)
        scope = MediaReadingScopeService(
            local_service=local_service, server_service=None
        )
        caller_thread = threading.get_ident()

        detail = await scope.get_media_detail(mode="local", media_id=media_id)

        assert detail["title"] == "Memory Fixture"
        # A threaded call against a :memory: DB would hit a fresh, empty
        # connection and raise/return nothing -- getting the real row back
        # at all is itself evidence the call stayed on this thread, and the
        # thread-affinity parametrized test above pins the mechanism.
        assert threading.get_ident() == caller_thread
    finally:
        db.close_connection()
