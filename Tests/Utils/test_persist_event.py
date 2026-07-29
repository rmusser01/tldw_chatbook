"""TASK-1240: persist_event is the single admitted path to the persistent log."""

from __future__ import annotations

import logging

import pytest

from tldw_chatbook.Utils.persistent_diagnostics import (
    PersistentDiagnosticFilter,
    persist_event,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def sink(tmp_path):
    """A real file handler behind the real filter, as the app installs it."""
    path = tmp_path / "app.log"
    handler = logging.FileHandler(path)
    handler.addFilter(PersistentDiagnosticFilter())
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    root.addHandler(handler)
    previous = root.level
    root.setLevel(logging.INFO)
    yield path, handler
    handler.flush()
    root.removeHandler(handler)
    root.setLevel(previous)
    handler.close()


def test_persist_event_reaches_the_sink(sink):
    path, handler = sink
    persist_event("scheduling", "scheduler_configured", item_count=2, status="ok")
    handler.flush()
    written = path.read_text()
    assert "event=scheduler_configured" in written
    assert "component=scheduling" in written
    assert "item_count=2" in written


def test_ordinary_logging_is_still_rejected(sink):
    """The boundary must not widen: only marked records are admitted."""
    path, handler = sink
    logging.getLogger("tldw_chatbook.diagnostics.scheduling").info(
        "an ordinary line that must not persist"
    )
    handler.flush()
    assert path.read_text() == ""


def test_component_cannot_be_passed_twice(sink):
    with pytest.raises(TypeError):
        persist_event("app", "app_started", component="somewhere_else")


def test_unknown_fields_are_still_rejected(sink):
    """The schema is the guarantee; persist_event must not bypass it."""
    with pytest.raises(ValueError):
        persist_event("app", "app_started", prompt="secret user text")
