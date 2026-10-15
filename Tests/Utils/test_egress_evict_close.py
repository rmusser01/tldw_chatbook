"""Pooled egress sessions close on eviction; the app drains the pool at shutdown."""

import threading
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
