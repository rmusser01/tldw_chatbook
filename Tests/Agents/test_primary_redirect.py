"""Service-side redirect surface: mailbox + abort flag, one atomic unit.

TASK-26000. `redirect_primary` posts a STEERING_SOURCE_REDIRECT entry into the
SAME mailbox steering uses and sets the run's abort flag under the same lock;
the drain clears the flag when it consumes a redirect entry. The flag is what
the bridge composes into its STREAM-cancel predicate (aborting only the
in-flight model request) and what the loop's `has_pending_redirect` probe
reads -- one source of truth, no second mailbox to desync.
"""

from __future__ import annotations

from types import SimpleNamespace

from tldw_chatbook.Agents.agent_models import (
    MAX_STEERING_CHARS,
    STEERING_SOURCE_REDIRECT,
    STEERING_SOURCE_USER,
)
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry


def _service(**kwargs):
    return AgentService(
        db=SimpleNamespace(), registry=ToolCatalogRegistry(), **kwargs
    )


def test_redirecting_an_unknown_run_is_refused_honestly():
    service = _service()

    refusal = service.redirect_primary("no-such-run", "do Y instead")

    assert refusal is not None
    assert "not running" in refusal.lower()


def test_redirecting_a_finished_run_is_refused():
    service = _service()
    service._register_primary_mailbox("run-1")
    service._unregister_primary_mailbox("run-1")

    assert service.redirect_primary("run-1", "too late") is not None


def test_empty_and_overlong_corrections_are_refused():
    service = _service()
    service._register_primary_mailbox("run-1")

    assert service.redirect_primary("run-1", "   ") is not None
    over = service.redirect_primary("run-1", "x" * (MAX_STEERING_CHARS + 1))
    assert over is not None
    assert str(MAX_STEERING_CHARS) in over


def test_accepted_redirect_posts_entry_and_raises_the_abort_flag():
    service = _service()
    drain = service._register_primary_mailbox("run-1")

    assert service.redirect_primary("run-1", "  no — the YAML parser  ") is None
    assert service._primary_redirect_pending("run-1") is True

    entries = drain()
    assert entries == [(STEERING_SOURCE_REDIRECT, "no — the YAML parser")]
    # consuming the redirect entry lowers the flag -- the next model call
    # must not be aborted by a redirect already delivered
    assert service._primary_redirect_pending("run-1") is False


def test_plain_steering_never_raises_the_abort_flag():
    service = _service()
    drain = service._register_primary_mailbox("run-1")

    assert service.steer_primary("run-1", "gentle nudge") is None
    assert service._primary_redirect_pending("run-1") is False

    entries = drain()
    assert entries == [(STEERING_SOURCE_USER, "gentle nudge")]


def test_mixed_drain_clears_the_flag_and_keeps_order():
    service = _service()
    drain = service._register_primary_mailbox("run-1")

    service.steer_primary("run-1", "first")
    service.redirect_primary("run-1", "second")

    assert service._primary_redirect_pending("run-1") is True
    assert drain() == [
        (STEERING_SOURCE_USER, "first"),
        (STEERING_SOURCE_REDIRECT, "second"),
    ]
    assert service._primary_redirect_pending("run-1") is False


def test_on_primary_redirect_ready_hands_working_callables():
    captured = {}

    def ready(redirect_fn, abort_probe):
        captured["redirect"] = redirect_fn
        captured["probe"] = abort_probe

    service = _service(on_primary_redirect_ready=ready)
    drain = service._register_primary_mailbox("run-1")

    assert set(captured) == {"redirect", "probe"}
    assert captured["probe"]() is False
    assert captured["redirect"]("switch to plan B") is None
    assert captured["probe"]() is True
    assert drain() == [(STEERING_SOURCE_REDIRECT, "switch to plan B")]
    assert captured["probe"]() is False

    service._unregister_primary_mailbox("run-1")
    assert captured["redirect"]("gone") is not None
    assert captured["probe"]() is False
