"""Pooled egress sessions close on eviction; the app drains the pool at shutdown."""

from unittest.mock import patch

import pytest

from tldw_chatbook.Utils import egress


def test_pool_eviction_closes_the_evicted_session(monkeypatch):
    monkeypatch.setattr(egress, "_SESSION_POOL", {})
    monkeypatch.setattr(egress, "_SESSION_POOL_MAX", 2)
    # Config reads stay out of this unit's scope (function-local imports
    # pull from their source modules, so patch there).
    from tldw_chatbook.Utils import tls_trust

    monkeypatch.setattr(tls_trust, "requests_verify", lambda: True)
    monkeypatch.setattr(egress, "default_session_timeout", lambda: 30.0)
    closed = []
    monkeypatch.setattr(
        egress.requests.Session, "close", staticmethod(closed.append), raising=True
    )

    def _bare_session(self, default_timeout=None):
        self.hooks = {"response": []}
        self.verify = True
        self.adapters = {}

    with patch.object(
        egress.PooledDefaultTimeoutSession, "__init__", _bare_session
    ):
        first = egress.create_pooled_default_session(timeout=10.0)
        egress.create_pooled_default_session(timeout=11.0)
        third = egress.create_pooled_default_session(timeout=12.0)

    assert first is not third
    assert len(closed) == 1, "evicting the oldest entry must CLOSE it, not drop it"
    assert closed[0] is first


def test_close_pooled_default_sessions_drains_under_concurrency():
    sentinel = object()
    egress._SESSION_POOL[("k",)] = sentinel
    # A hostile close that raises must not strand the loop or the lock.
    with patch.object(
        egress.requests.Session, "close", side_effect=OSError("boom")
    ):
        try:
            egress.close_pooled_default_sessions()
        except OSError:
            pass  # close() propagates per the existing contract
    assert not egress._SESSION_POOL
    # The lock is still usable afterwards.
    with egress._SESSION_POOL_LOCK:
        assert True


def test_shutdown_lifecycle_closes_the_pool():
    import inspect

    from tldw_chatbook.app_lifecycle import LifecycleMixin

    body = inspect.getsource(LifecycleMixin._shutdown_app_owned_lifecycles)
    assert "close_pooled_default_sessions()" in body, (
        "the app shutdown sequence must drain the pooled egress sessions"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("primary", [None, "pool", "recovery", "lab", "workflow"])
async def test_pool_closes_after_later_owners_without_replacing_primary_error(
    monkeypatch, primary
):
    import asyncio
    from types import SimpleNamespace

    from tldw_chatbook import app_lifecycle
    from tldw_chatbook.app_lifecycle import LifecycleMixin

    # Exercise the inherited cleanup methods without importing the app/profile.
    monkeypatch.setattr(app_lifecycle, "TldwCli", LifecycleMixin)
    events = []
    pool_error = OSError("pool close failed")
    retained = asyncio.CancelledError() if primary == "recovery" else RuntimeError()

    async def idle():
        return None

    async def recovery():
        return retained if primary == "recovery" else None

    async def fail():
        raise retained

    app = SimpleNamespace(
        _shutdown_actor_pack_recovery=recovery,
        audio_cpp_model_install_owner=SimpleNamespace(shutdown=idle),
        _meeting_session_owner=None,
    )
    for name in (
        "workflow_session", "collections_capture_runtime", "notes_sync_runtime",
        "actor_pack_import", "actor_pack_export", "raw_cli_runtime",
        "terminal_session_manager", "console_settings_durability",
        "console_runtime", "persona_buddy", "console_image_edits",
        "file_notes_session_owner",
    ):
        async def drain(label=name):
            events.append(label)
        setattr(app, f"_shutdown_{name}", drain)
    if primary == "lab":
        app._chunking_lab_coordinator = SimpleNamespace(close=fail)
    elif primary == "workflow":
        app._workflow_authoring = SimpleNamespace(close=fail)

    session = egress.requests.Session()
    monkeypatch.setattr(egress, "_SESSION_POOL", {("owned",): session})

    def close(actual):
        assert actual is session
        events.append("pool")
        if primary is not None:
            raise pool_error

    monkeypatch.setattr(egress.requests.Session, "close", close)
    if primary is None:
        await LifecycleMixin._shutdown_app_owned_lifecycles(app)
    else:
        expected = pool_error if primary == "pool" else retained
        with pytest.raises(type(expected)) as caught:
            await LifecycleMixin._shutdown_app_owned_lifecycles(app)
        assert caught.value is expected
    assert events[-2:] == ["file_notes_session_owner", "pool"]
    assert events.index("console_runtime") < events.index("pool")
    assert not egress._SESSION_POOL


def test_pool_close_attempts_remaining_sessions_after_one_close_fails(monkeypatch):
    first, failing = egress.requests.Session(), egress.requests.Session()
    monkeypatch.setattr(egress, "_SESSION_POOL", {("first",): first, ("last",): failing})
    original_close = egress.requests.Session.close
    closed = []
    failure = OSError("adapter close failed")

    def close(session):
        original_close(session)
        closed.append(session)
        if session is failing:
            raise failure

    monkeypatch.setattr(egress.requests.Session, "close", close)
    with pytest.raises(OSError) as caught:
        egress.close_pooled_default_sessions()
    assert caught.value is failure
    assert closed == [failing, first]
    assert not egress._SESSION_POOL
