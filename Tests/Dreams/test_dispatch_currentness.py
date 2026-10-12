"""Live master-switch admission after real awaited budget/provider work."""
import pytest

from Tests.Dreams.test_cycle_service import (
    _deps,
    _fake_chat,
    _fake_perform,
    _seed_topics,
)
from Tests.Dreams.test_cycle_service import (
    dreams_db as _dreams_db_fixture,
)
from Tests.Dreams.test_cycle_service import (
    settings as _settings_fixture,
)
from Tests.Dreams.test_track_service import (
    _SET_A,
    _TEMPLATE,
    _DispatchRecorder,
    _FakeChat,
    _FakeSearch,
    _track_deps,
)
from tldw_chatbook.Dreams.cycle_service import run_cycle
from tldw_chatbook.Dreams.track_service import run_track_check, track_question

dreams_db = _dreams_db_fixture
settings = _settings_fixture


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


async def _disable_before_native_callback(monkeypatch, settings, select, revoke=None):
    """Pause the actual worker immediately before its callback, then revoke."""
    import asyncio
    import threading

    original = asyncio.to_thread
    crossed = []

    async def delayed(function, *args, **kwargs):
        if crossed or not select(function):
            return await original(function, *args, **kwargs)
        crossed.append(function)
        reached, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()

        def invoke():
            loop.call_soon_threadsafe(reached.set)
            assert release.wait(5), "native callback was not released"
            return function(*args, **kwargs)

        pending = asyncio.create_task(original(invoke))
        try:
            await asyncio.wait_for(reached.wait(), 5)
            if revoke is None:
                settings["enabled"] = False
            else:
                revoke()
        finally:
            release.set()
        return await asyncio.wait_for(pending, 5)

    monkeypatch.setattr(asyncio, "to_thread", delayed)
    return crossed


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["synthesis", "search", "story"])
async def test_cycle_rechecks_on_actual_queued_worker(
    dreams_db, settings, monkeypatch, boundary,
):
    _seed_topics(dreams_db)
    calls = []

    def chat(**kwargs):
        calls.append(("model", settings["enabled"]))
        return _fake_chat(**kwargs)

    def perform(engine, query, **kwargs):
        calls.append(("search", settings["enabled"]))
        return _fake_perform(engine, query, **kwargs)

    names = {"synthesis": "chat", "search": "perform", "story": "generate_story"}
    crossed = await _disable_before_native_callback(
        monkeypatch, settings, lambda function: getattr(function, "__name__", "") == names[boundary])
    result = await run_cycle(_deps(dreams_db, perform=perform, chat=chat), trigger="scheduled")
    assert len(crossed) == 1
    assert all(enabled for _kind, enabled in calls), calls
    assert result == {"collection_id": None, "status": "disabled", "stories": 0}
    with dreams_db.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM dreams_collections").fetchone()[0] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["search", "judge", "alert"])
async def test_track_rechecks_on_actual_queued_worker(
    dreams_db, settings, monkeypatch, boundary,
):
    from types import SimpleNamespace

    item_id = await track_question(dreams_db, query_template=_TEMPLATE, intent="deal")
    dreams_db.insert_track_run(item_id, status="baseline", digest_hash="older")
    search, model = _FakeSearch(_SET_A), _FakeChat()
    calls = []

    def perform(*args, **kwargs):
        calls.append(("search", settings["enabled"]))
        return search(*args, **kwargs)

    def chat(**kwargs):
        calls.append(("model", settings["enabled"]))
        return model(**kwargs)

    def dispatch(**kwargs):
        calls.append(("alert", settings["enabled"]))

    recorder = SimpleNamespace(store=SimpleNamespace(is_memory_db=False), dispatch=dispatch)
    names = {"search": "perform", "judge": "chat", "alert": "dispatch"}
    crossed = await _disable_before_native_callback(
        monkeypatch, settings, lambda function: getattr(function, "__name__", "") == names[boundary])
    result = await run_track_check(_track_deps(
        dreams_db, perform_search=perform, chat=chat, dispatch_getter=lambda: recorder), item_id)
    assert len(crossed) == 1
    assert all(enabled for _kind, enabled in calls), calls
    assert result == {"status": "disabled", "notified": False}
    assert [row["status"] for row in dreams_db.list_recent_track_runs(item_id)] == ["baseline"]


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["synthesis", "search", "story"])
async def test_guardian_revocation_on_actual_queued_worker(
    tmp_path, dreams_db, settings, monkeypatch, boundary,
):
    from tldw_chatbook.DB.Guardian_DB import GuardianDB
    from tldw_chatbook.Guardian import settings as guardian_settings

    monkeypatch.setattr(guardian_settings, "guardian_setting",
                        lambda key, default=None: True if key == "enabled" else default)
    guardian = GuardianDB(tmp_path / "queued-consent.db")
    rule = guardian.upsert_rule(name="private topic", topic="rust tui", pattern="rust", feeds_discovery=1)
    guardian.insert_alert(rule_id=rule, session_id="a", visit_id="v",
                          topic="rust tui", message_digest="a" * 64)
    revoked, calls = [], []

    def chat(**kwargs):
        calls.append(("model", bool(revoked)))
        return _fake_chat(**kwargs)

    def perform(engine, query, **kwargs):
        calls.append(("search", bool(revoked)))
        return _fake_perform(engine, query, **kwargs)

    def revoke():
        guardian.upsert_rule(id=rule, feeds_discovery=0)
        revoked.append(True)

    names = {"synthesis": "chat", "search": "perform", "story": "generate_story"}
    crossed = await _disable_before_native_callback(
        monkeypatch, settings, lambda function: getattr(function, "__name__", "") == names[boundary], revoke)
    deps = _deps(dreams_db, perform=perform, chat=chat)
    deps.guardian_db_getter = lambda: guardian
    try:
        result = await run_cycle(deps, trigger="scheduled")
        assert len(crossed) == 1
        assert settings["enabled"]
        assert not any(was_revoked for _kind, was_revoked in calls), calls
        assert result == {"collection_id": None, "status": "disabled", "stories": 0}
    finally:
        guardian.close()
