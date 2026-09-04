"""Capability-store security and lifecycle tests for the native Canvas gateway."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tldw_chatbook.Canvas.capabilities import (
    CanvasCapabilityError,
    CanvasCapabilityScope,
    CanvasCapabilityStore,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


def _scope(**changes: str) -> CanvasCapabilityScope:
    base = CanvasCapabilityScope(
        browser_session_id="browser-a",
        load_id="load-a",
        conversation_session_id="conversation-session-a",
        canvas_id="canvas-a",
        revision_id="revision-a",
        action="render_plan",
    )
    return replace(base, **changes)


def test_capabilities_are_random_hashed_at_rest_and_redacted() -> None:
    store = CanvasCapabilityStore()

    first = store.issue(_scope(), ttl_seconds=10)
    second = store.issue(_scope(load_id="load-b"), ttl_seconds=10)

    assert first.token != second.token
    assert first.token not in repr(first)
    assert first.token not in repr(store)
    assert first.token.encode() not in b"".join(store._records)
    assert all(len(digest) == 32 for digest in store._records)

    with pytest.raises(CanvasCapabilityError, match="lifetime"):
        store.issue(_scope(load_id="load-nan"), ttl_seconds=float("nan"))


def test_capability_is_single_use_exact_scope_and_monotonic_expiry() -> None:
    clock = _Clock()
    store = CanvasCapabilityStore(clock=clock)
    grant = store.issue(_scope(), ttl_seconds=5)

    with pytest.raises(CanvasCapabilityError, match="scope"):
        store.consume(grant.token, expected_scope=_scope(revision_id="revision-b"))
    assert store.consume(grant.token, expected_scope=_scope()) == _scope()
    with pytest.raises(CanvasCapabilityError, match="unavailable"):
        store.consume(grant.token, expected_scope=_scope())

    expired = store.issue(_scope(load_id="load-expired"), ttl_seconds=5)
    clock.value += 5
    with pytest.raises(CanvasCapabilityError, match="expired"):
        store.consume(
            expired.token,
            expected_scope=_scope(load_id="load-expired"),
        )

    secret = "do-not-leak-\N{SNOWMAN}"
    with pytest.raises(CanvasCapabilityError) as error:
        store.consume(secret, expected_scope=_scope())
    assert secret not in str(error.value)
    assert error.value.__cause__ is None


def test_exact_revocation_covers_reload_scope_change_close_and_shutdown() -> None:
    store = CanvasCapabilityStore()
    old_load = store.issue(_scope(), ttl_seconds=10)
    new_load = store.issue(_scope(load_id="load-b"), ttl_seconds=10)
    sibling_browser = store.issue(
        _scope(browser_session_id="browser-b"), ttl_seconds=10
    )

    assert store.revoke_load("browser-a", "load-a") == 1
    with pytest.raises(CanvasCapabilityError):
        store.consume(old_load.token, expected_scope=_scope())
    assert store.consume(
        new_load.token, expected_scope=_scope(load_id="load-b")
    ) == _scope(load_id="load-b")

    changed_revision = store.issue(_scope(load_id="load-c"), ttl_seconds=10)
    assert (
        store.revoke_selection(
            browser_session_id="browser-a",
            conversation_session_id="conversation-session-a",
            canvas_id="canvas-a",
            revision_id="revision-a",
        )
        == 1
    )
    with pytest.raises(CanvasCapabilityError):
        store.consume(changed_revision.token, expected_scope=_scope(load_id="load-c"))
    assert store.consume(
        sibling_browser.token,
        expected_scope=_scope(browser_session_id="browser-b"),
    ) == _scope(browser_session_id="browser-b")

    browser_close = store.issue(_scope(load_id="load-d"), ttl_seconds=10)
    assert store.revoke_browser_session("browser-a") == 1
    with pytest.raises(CanvasCapabilityError):
        store.consume(browser_close.token, expected_scope=_scope(load_id="load-d"))

    shutdown = store.issue(_scope(load_id="load-e"), ttl_seconds=10)
    store.close()
    with pytest.raises(CanvasCapabilityError, match="closed"):
        store.consume(shutdown.token, expected_scope=_scope(load_id="load-e"))
    with pytest.raises(CanvasCapabilityError, match="closed"):
        store.issue(_scope(), ttl_seconds=10)
