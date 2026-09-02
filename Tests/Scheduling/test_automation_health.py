"""Read-time health tests for local automation definitions (schedules-handoff
PR-2 Task 6).

`compute_local_health` is pure and read-time-only: it never touches the DB
or persists anything. Fake ``app`` objects stand in for the running app;
``resolve_execution_target`` (lazily imported inside the health module and
exposed as a monkeypatchable module attribute -- see
``automation_health.py``'s own docstring) is patched directly so these tests
never need the real Library RAG seams importable.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Scheduling import automation_health

pytestmark = pytest.mark.unit

DEFINITION_ROW: dict = {"id": "def-1", "input": {}}


def test_capability_unavailable_when_app_has_no_rag_service():
    app = SimpleNamespace()  # no library_rag_search_service attribute at all

    health, reason = automation_health.compute_local_health(app, DEFINITION_ROW)

    assert health == "capability_unavailable"
    assert reason


def test_capability_unavailable_when_rag_service_is_none():
    app = SimpleNamespace(library_rag_search_service=None)

    health, reason = automation_health.compute_local_health(app, DEFINITION_ROW)

    assert health == "capability_unavailable"
    assert reason


def _fake_search_service() -> SimpleNamespace:
    """A minimal stand-in with a callable `search` -- what a real
    `LibraryLocalRagSearchService` (or any capable seam) actually exposes."""
    return SimpleNamespace(search=lambda *args, **kwargs: None)


def _readable_app(**overrides: object) -> SimpleNamespace:
    """A fake app with a capable RAG service and both scoped-source DBs
    present -- the baseline every source-readable test starts from and
    overrides pieces of."""
    attrs: dict[str, object] = {
        "library_rag_search_service": _fake_search_service(),
        "media_db": object(),
        "chachanotes_db": object(),
    }
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def test_permission_required_when_no_provider_resolves(monkeypatch):
    app = _readable_app()
    monkeypatch.setattr(
        automation_health,
        "resolve_execution_target",
        lambda row: {"provider": None, "model": None, "max_tokens": 1000},
    )

    health, reason = automation_health.compute_local_health(app, DEFINITION_ROW)

    assert health == "permission_required"
    assert reason


def test_ready_when_rag_service_present_and_provider_resolves(monkeypatch):
    app = _readable_app()
    monkeypatch.setattr(
        automation_health,
        "resolve_execution_target",
        lambda row: {"provider": "openai", "model": "gpt-4", "max_tokens": 1000},
    )

    health, reason = automation_health.compute_local_health(app, DEFINITION_ROW)

    assert health == "ready"
    assert reason == ""


def test_capability_check_short_circuits_before_resolving_provider(monkeypatch):
    """capability_unavailable must win even if resolve_execution_target would
    otherwise blow up -- the function must not be called at all."""
    app = SimpleNamespace(library_rag_search_service=None)

    def _boom(row):
        raise AssertionError("resolve_execution_target must not be called")

    monkeypatch.setattr(automation_health, "resolve_execution_target", _boom)

    health, _reason = automation_health.compute_local_health(app, DEFINITION_ROW)

    assert health == "capability_unavailable"


def test_capability_unavailable_when_service_has_no_callable_search(monkeypatch):
    """Finding H: a `library_rag_search_service` that is present but does
    not expose a callable `search` (e.g. a stub/misconfigured object) is
    not a capable seam -- it must not be reported `ready` or
    `permission_required`, either of which would tell a caller it is safe
    to dispatch."""
    app = SimpleNamespace(library_rag_search_service=object())

    def _boom(row):
        raise AssertionError("resolve_execution_target must not be called")

    monkeypatch.setattr(automation_health, "resolve_execution_target", _boom)

    health, reason = automation_health.compute_local_health(app, DEFINITION_ROW)

    assert health == "capability_unavailable"
    assert reason


# --- Sources-readable check (plan Task 6) -----------------------------------
#
# Default scope (no `config`, or `config.scope` omitted) normalizes to
# "all_searchable_library" over all three server source names
# (`media_db`/`notes`/`chats`), which `library_source_types` maps onto the
# Library's plural vocabulary (`media`/`notes`/`conversations`) -- the same
# mapping `automation_execution.py` uses for retrieval.

SCOPED_TO_CHATS_ONLY: dict = {
    "id": "def-scoped",
    "input": {},
    "config": {"scope": {"mode": "sources", "sources": ["chats"]}},
}


def test_ready_when_default_scope_sources_all_readable(monkeypatch):
    """Every default-scope source (media/notes/conversations) resolves to a
    live DB attribute -- the new check must not block a `ready` app."""
    app = _readable_app()
    monkeypatch.setattr(
        automation_health,
        "resolve_execution_target",
        lambda row: {"provider": "openai", "model": "gpt-4", "max_tokens": 1000},
    )

    health, reason = automation_health.compute_local_health(app, DEFINITION_ROW)

    assert health == "ready"
    assert reason == ""


def test_capability_unavailable_when_a_scoped_source_db_is_missing(monkeypatch):
    """`chachanotes_db` missing (no notes/chats DB) blocks health even
    though the provider would otherwise resolve -- and the reason names the
    unreadable source."""
    app = _readable_app(chachanotes_db=None)

    def _boom(row):
        raise AssertionError("resolve_execution_target must not be called")

    monkeypatch.setattr(automation_health, "resolve_execution_target", _boom)

    health, reason = automation_health.compute_local_health(app, DEFINITION_ROW)

    assert health == "capability_unavailable"
    # First unreadable Library source type in iteration order (media, notes,
    # conversations) -- media_db is present, notes is the first miss.
    assert "notes" in reason


def test_ready_ignores_a_missing_db_for_a_source_outside_the_scope(monkeypatch):
    """A scope that only asks for `chats` must not be blocked by a missing
    `media_db` -- readability is scoped, not blanket."""
    app = _readable_app(media_db=None)
    monkeypatch.setattr(
        automation_health,
        "resolve_execution_target",
        lambda row: {"provider": "openai", "model": "gpt-4", "max_tokens": 1000},
    )

    health, reason = automation_health.compute_local_health(app, SCOPED_TO_CHATS_ONLY)

    assert health == "ready"
    assert reason == ""


def test_capability_unavailable_when_the_in_scope_source_db_is_missing(monkeypatch):
    """The inverse of the previous test: `chats` IS in scope and its DB is
    missing -- must block, naming `conversations` (the Library vocabulary
    `chats` maps onto)."""
    app = _readable_app(chachanotes_db=None)

    def _boom(row):
        raise AssertionError("resolve_execution_target must not be called")

    monkeypatch.setattr(automation_health, "resolve_execution_target", _boom)

    health, reason = automation_health.compute_local_health(app, SCOPED_TO_CHATS_ONLY)

    assert health == "capability_unavailable"
    assert "conversations" in reason


def test_deps_missing_takes_precedence_over_unreadable_source(monkeypatch):
    """RAG-deps unavailability (no `library_rag_search_service`) must win
    over a scoped source being unreadable -- the source-specific reason
    must never leak when the seam itself is the actual blocker."""
    app = SimpleNamespace(library_rag_search_service=None, media_db=None, chachanotes_db=None)

    def _boom(row):
        raise AssertionError("resolve_execution_target must not be called")

    monkeypatch.setattr(automation_health, "resolve_execution_target", _boom)

    health, reason = automation_health.compute_local_health(app, DEFINITION_ROW)

    assert health == "capability_unavailable"
    assert reason == automation_health._CAPABILITY_UNAVAILABLE_REASON
