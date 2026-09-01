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


def test_permission_required_when_no_provider_resolves(monkeypatch):
    app = SimpleNamespace(library_rag_search_service=object())
    monkeypatch.setattr(
        automation_health,
        "resolve_execution_target",
        lambda row: {"provider": None, "model": None, "max_tokens": 1000},
    )

    health, reason = automation_health.compute_local_health(app, DEFINITION_ROW)

    assert health == "permission_required"
    assert reason


def test_ready_when_rag_service_present_and_provider_resolves(monkeypatch):
    app = SimpleNamespace(library_rag_search_service=object())
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
