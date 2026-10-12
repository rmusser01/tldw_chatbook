"""Live master-switch admission after real awaited budget/provider work."""
import pytest

from Tests.Dreams.test_cycle_service import (
    _deps, _fake_chat, _fake_perform, _seed_topics, dreams_db, settings,
)
from Tests.Dreams.test_track_service import (
    _DispatchRecorder, _FakeChat, _FakeSearch, _SET_A, _TEMPLATE, _track_deps,
)
from tldw_chatbook.Dreams.cycle_service import run_cycle
from tldw_chatbook.Dreams.track_service import run_track_check, track_question


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["provider", "synthesis", "search", "story"])
async def test_cycle_rechecks_disabled_after_awaited_admission(
    dreams_db, settings, monkeypatch, boundary,
):
    _seed_topics(dreams_db)
    calls = []
    llm_reservations = 0
    original = dreams_db.usage_reserve

    def reserve(*args, **kwargs):
        nonlocal llm_reservations
        result = original(*args, **kwargs)
        counter = kwargs["counter"]
        if counter == "llm_calls":
            llm_reservations += 1
        if ((boundary == "search" and counter == "searches")
            or (boundary == "synthesis" and llm_reservations == 1)
            or (boundary == "story" and llm_reservations == 2)):
            settings["enabled"] = False
        return result

    def chat(**kwargs):
        calls.append(("model", settings["enabled"]))
        return _fake_chat(**kwargs)

    def perform(engine, query, **kwargs):
        calls.append(("search", settings["enabled"]))
        return _fake_perform(engine, query, **kwargs)

    def resolve():
        if boundary == "provider":
            settings["enabled"] = False
        return chat

    monkeypatch.setattr(dreams_db, "usage_reserve", reserve)
    deps = _deps(dreams_db, perform=perform)
    deps.chat_getter = resolve
    result = await run_cycle(deps, trigger="scheduled")
    assert result["status"] == "disabled"
    assert all(enabled for _kind, enabled in calls), calls
    if boundary in {"provider", "synthesis"}:
        assert calls == []
    elif boundary == "search":
        assert [kind for kind, _enabled in calls] == ["model"]
    else:
        assert sum(kind == "model" for kind, _enabled in calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["search", "provider", "judge", "alert"])
async def test_track_rechecks_disabled_after_awaited_admission(
    dreams_db, settings, monkeypatch, boundary,
):
    item_id = await track_question(dreams_db, query_template=_TEMPLATE, intent="deal")
    dreams_db.insert_track_run(item_id, status="baseline", digest_hash="older")
    search, chat, recorder = _FakeSearch(_SET_A), _FakeChat(), _DispatchRecorder()
    original = dreams_db.usage_reserve

    def reserve(*args, **kwargs):
        result = original(*args, **kwargs)
        if ((boundary == "search" and kwargs["counter"] == "searches")
            or (boundary == "judge" and kwargs["counter"] == "llm_calls")):
            settings["enabled"] = False
        return result

    def resolve():
        if boundary == "provider":
            settings["enabled"] = False
        return chat

    def dispatch_getter():
        if boundary == "alert":
            settings["enabled"] = False
        return recorder()

    monkeypatch.setattr(dreams_db, "usage_reserve", reserve)
    deps = _track_deps(dreams_db, perform_search=search, chat=chat,
                       dispatch_getter=dispatch_getter)
    deps.chat_getter = resolve
    result = await run_track_check(deps, item_id)
    assert result == {"status": "disabled", "notified": False}
    assert recorder.dispatched == []
    assert [row["status"] for row in dreams_db.list_recent_track_runs(item_id)] == ["baseline"]
    if boundary == "search":
        assert search.calls == []
    if boundary != "alert":
        assert chat.calls == []
