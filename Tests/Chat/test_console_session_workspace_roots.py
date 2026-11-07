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


def test_roots_are_strings_as_the_snapshot_coerces_them():
    """Equal to the snapshot's ``workspace_roots`` by construction, not by luck.

    ``ConsoleTurnConfigurationSnapshot`` stores ``str(root)`` for every
    admitted root; the panel compares the root it asked for against the one
    it landed, so a ``Path`` here and a ``str`` there would read as a change.
    """
    from pathlib import PurePosixPath

    controller = _controller(
        consent=_consent({"ws-1": (PurePosixPath("/w/one"), "/w/two")}, [])
    )

    roots = resolve_session_workspace_roots(controller, "s1")

    assert roots == ("/w/one", "/w/two")
    assert all(type(root) is str for root in roots)


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


def test_an_installed_turn_context_provider_is_not_asked_for_the_roots():
    """TASK-33620.15.1.5 AC#3: no caller builds a turn snapshot to get the roots.

    The runtime-owned controller never has a provider (``ConsoleRuntime``
    forces it to ``None``); a directly built one used to answer from its
    provider's whole snapshot. The roots are the consent admission's either
    way, so they are read from it either way.
    """
    asked: list[str] = []
    controller = _controller(
        consent=_consent({"ws-1": ("/w/consent",)}, asked),
        provider=lambda session_id: None,
    )

    assert resolve_session_workspace_roots(controller, "s1") == ("/w/consent",)
    assert asked == ["ws-1"]


def test_capture_change_review_admission_returns_string_roots():
    """TASK-33620.15.1.5 AC#2: every reader gets ``str`` roots from the source."""
    from pathlib import PurePosixPath

    from tldw_chatbook.Chat.console_turn_context import (
        capture_change_review_admission,
    )

    app = SimpleNamespace(
        change_review_consent_service=_consent(
            {"ws-1": (PurePosixPath("/w/one"), "/w/two")}, []
        )
    )

    roots, _aliases, _skipped = capture_change_review_admission(app, "ws-1")

    assert roots == ("/w/one", "/w/two")
    assert all(type(root) is str for root in roots)


def test_the_admission_stamp_moves_with_each_in_app_input_and_reads_nothing():
    """TASK-33620.15.1.6: the Environment guard's UI-loop check, from memory.

    ``session_change_review_admission_stamp`` names everything
    ``read_session_change_review_admission`` reads -- the session's
    workspace, the registry's binding generation, and which consent service
    answers at which admission revision -- without asking the registry or
    the consent service anything, so the UI loop can tell that a session's
    roots may have changed after a worker read them.
    """
    from tldw_chatbook.Chat.console_turn_context import (
        session_change_review_admission_stamp,
    )

    def _no_read(*_args, **_kwargs):
        raise AssertionError("the stamp must not read the registry or consent")

    consent = SimpleNamespace(admission_revision=0, admit_turn=_no_read)
    registry = SimpleNamespace(mutation_generation=0, list_folder_bindings=_no_read)
    app = SimpleNamespace(
        change_review_consent_service=consent, workspace_registry_service=registry
    )
    workspaces = {"s1": "ws-1", "s2": "ws-2"}
    store = _Store(workspaces)

    def stamp(session_id: str = "s1"):
        return session_change_review_admission_stamp(app, store, session_id)

    first = stamp()
    assert stamp() == first, "nothing changed, so the stamp holds"
    hash(first)

    assert stamp("s2") != first  # another session's workspace
    workspaces["s1"] = "ws-2"
    assert stamp() != first  # the session moved to another workspace
    workspaces["s1"] = "ws-1"
    assert stamp() == first

    registry.mutation_generation += 1  # a folder bound or unbound
    assert stamp() != first
    moved = stamp()
    consent.admission_revision += 1  # consent toggled, or a root became ready
    assert stamp() != moved
    moved = stamp()
    app.change_review_consent_service = SimpleNamespace(
        admission_revision=consent.admission_revision, admit_turn=_no_read
    )
    assert stamp() != moved  # another service answers now

    assert stamp("gone") is not None  # an unknown session does not raise
