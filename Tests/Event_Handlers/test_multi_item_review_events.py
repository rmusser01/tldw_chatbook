"""Multi-Item Review batch-analysis handlers: off-loop DB/LLM calls.

task-16194 coverage. Mirrors ``Tests/Event_Handlers/test_collections_tag_events.py``
(task-15471), which found and fixed the identical bug in
``collections_tag_events.py``. Every threaded call in
``multi_item_review_events.py`` used ``app.run_in_thread`` -- a method
neither Textual 8.x's ``App`` nor ``TldwCli`` defines -- so all three paths
below died before doing any real work:

1. ``generate_single_analysis`` -- the LLM call (line 172 pre-fix) raised
   ``AttributeError`` inside its own inner ``except``, surfacing as an
   "Error generating analysis: ..." string instead of the real analysis.
2. ``save_analysis_to_db`` -- the ``execute_query`` call (line 256 pre-fix)
   raised the same ``AttributeError``, caught by the function's own
   ``except`` and returned as ``False``. Even patched to a working
   off-loop call, the *second* dead call right after it (``app.media_db
   .commit``, line 262 pre-fix -- ``MediaDatabase`` has no ``commit``
   method) would still have kept the UPDATE uncommitted; the fix folds
   both into a single ``execute_query(..., commit=True)`` call.
3. ``load_existing_analyses`` -- the ``execute_query`` call (line 296
   pre-fix) raised the same ``AttributeError``, caught by the function's
   own ``except`` and returned as ``{}``.

The success-path assertions below fail against the pre-task-16194 code.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Event_Handlers.multi_item_review_events import (
    generate_single_analysis,
    load_existing_analyses,
    save_analysis_to_db,
)


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeMediaDB:
    # False on purpose: exercises the real asyncio.to_thread path.
    is_memory_db = False

    def __init__(self):
        self.executed: list[tuple[str, object, bool]] = []
        self.rows: dict[int, str] = {}

    def execute_query(self, query, params=None, *, commit=False):
        self.executed.append((query, params, commit))
        if "UPDATE Media" in query:
            analysis, _last_modified, media_id = params
            self.rows[media_id] = analysis
            return _FakeCursor([])
        if "SELECT id, analysis_content" in query:
            rows = [
                {"id": media_id, "analysis_content": self.rows.get(media_id)}
                for media_id in params
            ]
            return _FakeCursor(rows)
        raise AssertionError(f"unexpected query: {query}")


class _FakeLLMClient:
    def __init__(self, response: str):
        self._response = response
        self.calls: list[dict] = []

    def chat_with_model(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeApp:
    """Duck-typed TldwCli stand-in: media_db + llm_api_client + notify.

    Deliberately has no ``run_in_thread`` -- the real app never had one
    either, which is exactly what this file pins.
    """

    def __init__(self, media_db, llm_api_client=None):
        self.media_db = media_db
        self.llm_api_client = llm_api_client
        self.llm_model_var = "test-model"
        self.llm_temperature_var = 0.7
        self.llm_context_size_var = 4096
        self.notifications: list[tuple[str, str]] = []
        self.posted: list[object] = []

    def notify(self, message, severity="information"):
        self.notifications.append((str(message), severity))

    def post_message(self, message):
        self.posted.append(message)


@pytest.mark.asyncio
async def test_generate_single_analysis_calls_llm_without_run_in_thread():
    llm = _FakeLLMClient("The generated analysis body.")
    app = _FakeApp(media_db=_FakeMediaDB(), llm_api_client=llm)
    item = {"id": 1, "title": "Doc", "content": "Some content to analyze"}

    result = await generate_single_analysis(app, item, "Summarize this")

    assert result is not None
    assert "The generated analysis body." in result
    assert llm.calls, "the LLM client must actually be invoked"
    assert llm.calls[0]["model"] == "test-model"
    assert llm.calls[0]["stream"] is False


@pytest.mark.asyncio
async def test_save_analysis_to_db_persists_in_one_committed_call():
    db = _FakeMediaDB()
    app = _FakeApp(media_db=db)

    ok = await save_analysis_to_db(app, 42, "analysis text")

    assert ok is True
    # Exactly one execute_query call, committed inline -- no separate
    # (nonexistent) `.commit()` follow-up call.
    assert len(db.executed) == 1
    query, params, commit = db.executed[0]
    assert "UPDATE Media" in query
    assert commit is True
    assert params[0] == "analysis text"
    assert params[2] == 42
    assert db.rows[42] == "analysis text"


@pytest.mark.asyncio
async def test_load_existing_analyses_returns_rows_without_run_in_thread():
    db = _FakeMediaDB()
    db.rows[1] = "existing analysis"
    app = _FakeApp(media_db=db)

    result = await load_existing_analyses(app, [1, 2])

    assert result == {1: "existing analysis", 2: None}


@pytest.mark.asyncio
async def test_save_analysis_to_db_off_loop_guard_still_works_for_memory_db():
    db = _FakeMediaDB()
    db.is_memory_db = True
    app = _FakeApp(media_db=db)

    ok = await save_analysis_to_db(app, 7, "in-memory analysis")

    assert ok is True
    assert db.rows[7] == "in-memory analysis"
