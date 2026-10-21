"""TASK-33620.15.1.3: one session's workspace roots, read without a turn snapshot.

``resolve_session_workspace_roots`` replaces
``controller.resolve_turn_execution_context(session_id).workspace_roots`` for
the Console Environment panel and the Change Review opener. These pin that it
reads exactly what that snapshot carried -- the consent-admitted roots of the
session's OWN workspace -- that it builds no snapshot for the runtime-owned
controller, and that it keeps the old answers for the edge cases the panel
tells apart (an unknown session raises; a consent failure is "no roots").
The equality against the real controller's full snapshot is pinned through the
mounted Console in ``Tests/UI/test_console_environment_refresh_off_loop.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_turn_context import resolve_session_workspace_roots


class _Store:
    def __init__(self, workspaces: dict[str, str]) -> None:
        self._workspaces = workspaces

    def session_workspace_id(self, session_id: str) -> str:
        try:
            return self._workspaces[session_id]
        except KeyError as exc:
            raise KeyError(f"Unknown Console chat session: {session_id}") from exc


def _controller(*, consent, provider=None, workspaces=None):
    def _no_snapshot(_session_id):
        raise AssertionError("the narrow read must not build a turn snapshot")

    return SimpleNamespace(
        _turn_context_provider=provider,
        store=_Store(workspaces or {"s1": "ws-1", "s2": "ws-2"}),
        app=SimpleNamespace(change_review_consent_service=consent),
        resolve_turn_execution_context=_no_snapshot,
    )


def _consent(roots_by_workspace: dict[str, tuple[str, ...]], asked: list[str]):
    def admit_turn(workspace_id):
        asked.append(workspace_id)
        return SimpleNamespace(
            ready_roots=roots_by_workspace.get(workspace_id, ()),
            ready_aliases=(),
            skipped_roots=(),
        )

    return SimpleNamespace(admit_turn=admit_turn)


def test_reads_the_sessions_own_workspace_admitted_roots():
    asked: list[str] = []
    controller = _controller(
        consent=_consent({"ws-1": ("/w/one", "/w/two"), "ws-2": ("/w/other",)}, asked)
    )

    assert resolve_session_workspace_roots(controller, "s1") == ("/w/one", "/w/two")
    assert resolve_session_workspace_roots(controller, "s2") == ("/w/other",)
    assert asked == ["ws-1", "ws-2"]


def test_no_consent_service_or_a_raising_one_is_no_roots():
    assert resolve_session_workspace_roots(_controller(consent=None), "s1") == ()

    def _boom(_workspace_id):
        raise RuntimeError("consent read failed")

    raising = SimpleNamespace(admit_turn=_boom)
    assert resolve_session_workspace_roots(_controller(consent=raising), "s1") == ()


def test_an_unknown_session_raises_as_the_full_build_did():
    """The Environment panel reads a raise as UNKNOWN_ROOT, not as UNBOUND."""
    controller = _controller(consent=_consent({"ws-1": ("/w/one",)}, []))

    with pytest.raises(KeyError):
        resolve_session_workspace_roots(controller, "gone")


def test_an_installed_turn_context_provider_still_owns_the_roots():
    """A controller built with a provider answers from the provider's snapshot."""
    asked: list[str] = []
    controller = _controller(
        consent=_consent({"ws-1": ("/w/consent",)}, asked),
        provider=lambda session_id: None,
    )
    controller.resolve_turn_execution_context = lambda session_id: SimpleNamespace(
        workspace_roots=("/w/provider", session_id)
    )

    assert resolve_session_workspace_roots(controller, "s1") == ("/w/provider", "s1")
    assert asked == []
