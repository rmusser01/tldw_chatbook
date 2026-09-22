"""A buddy that fails to load must say why somewhere.

TASK-32901 (tier-2 S16 P2): ``Persona_Buddy/controller.py`` swallowed eight
exceptions to ``None``/``False`` with zero ``logger`` calls in the whole
1,444-line module -- no loguru or stdlib logging import at all. A repository
bug, a schema mismatch or a corrupted visual pack therefore rendered as
"this persona has no avatar", with no log line, no notification and no
diagnostic field to work from.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from loguru import logger

from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController


@pytest.fixture()
def captured():
    records: list = []
    sink_id = logger.add(lambda record: records.append(record), level="DEBUG")
    try:
        yield records
    finally:
        logger.remove(sink_id)


def _rendered(records) -> str:
    return "".join(str(record.record["message"]) for record in records)


def test_graph_read_failure_is_recorded(captured):
    def _explode(_db):
        raise RuntimeError("visual pack schema mismatch")

    controller = PersonaBuddyController(
        profile_db=object(),
        repository_factory=_explode,
    )

    assert controller._read_graph("persona-1") is None
    assert "visual pack schema mismatch" in _rendered(captured)


def test_runtime_resolution_failure_is_recorded(captured):
    controller = PersonaBuddyController(
        profile_db=object(),
        profile_root="/nonexistent/profile/root",
        repository_factory=lambda _db: (_ for _ in ()).throw(
            RuntimeError("corrupted pack")
        ),
    )

    assert controller._read_graph(None, buddy_id="buddy-1") is None
    assert "corrupted pack" in _rendered(captured)


def _is_blanket(handler: ast.ExceptHandler) -> bool:
    """A handler that catches everything, as opposed to a named failure."""
    if handler.type is None:
        return True
    return isinstance(handler.type, ast.Name) and handler.type.id == "Exception"


def test_every_blanket_handler_in_the_module_records_something():
    """No ``except Exception`` body in this module may stay silent.

    The point of the finding is the module-wide silence, not one call site.
    Handlers that hand a failure category back to their caller already report
    it; the rest must log. Named-exception handlers are deliberate control
    flow and are out of scope.
    """
    source = Path(
        *PersonaBuddyController.__module__.replace(".", "/").split("/")
    ).with_suffix(".py")
    tree = ast.parse(source.read_text(encoding="utf-8"))

    silent: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler) or not _is_blanket(node):
            continue
        dumped = ast.dump(node)
        if "logger" not in dumped and "error_category" not in dumped:
            silent.append(node.lineno)

    assert silent == []
