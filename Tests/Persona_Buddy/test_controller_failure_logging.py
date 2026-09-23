"""A buddy that fails to load must say why somewhere -- but only so much.

TASK-32901 (tier-2 S16 P2): ``Persona_Buddy/controller.py`` swallowed eight
exceptions to ``None``/``False`` with zero ``logger`` calls in the whole
1,444-line module -- no loguru or stdlib logging import at all. A repository
bug, a schema mismatch or a corrupted visual pack therefore rendered as
"this persona has no avatar", with no log line, no notification and no
diagnostic field to work from.

The first cut of that fix used ``logger.opt(exception=True)``, which attaches
the full traceback and the exception's own message. That is wrong *here*:
``_read_local_buddy`` is built over ``self._profile_root`` and its handler has
carried ``# noqa: BLE001 - reject invalid private library authority without
paths`` since before this change, and the repo already keeps a bounded-error
discipline at that boundary (``PrivatePathError`` deliberately exposes a
bounded result and drops the original exception -- see
``Tests/Utils/test_private_paths.py``). An OSError or PrivatePathError raised
under that call renders the user's private profile path into its message.

So the contract these tests pin is two-sided: the failure *site* and the
exception *class* are recorded (diagnosable), and the exception's own message
and traceback are not (bounded). Widening it back to ``exception=True`` fails
the canary assertions below.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from loguru import logger

from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController
from tldw_chatbook.Persona_Buddy.preferences import BuddySelection


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


def _attached_tracebacks(records) -> list:
    """Every record that carries loguru's structured exception payload."""
    return [record for record in records if record.record["exception"] is not None]


def test_graph_read_failure_is_recorded(captured):
    def _explode(_db):
        raise RuntimeError("visual pack schema mismatch")

    controller = PersonaBuddyController(
        profile_db=object(),
        repository_factory=_explode,
    )

    assert controller._read_graph("persona-1") is None
    rendered = _rendered(captured)
    # Diagnosable: the site and the failure class both land in the message.
    assert "visual graph unreadable" in rendered
    assert "RuntimeError" in rendered
    # Bounded: the exception's own text and traceback do not.
    assert "visual pack schema mismatch" not in rendered
    assert _attached_tracebacks(captured) == []


def test_runtime_resolution_failure_is_recorded(captured):
    controller = PersonaBuddyController(
        profile_db=object(),
        profile_root="/nonexistent/profile/root",
        repository_factory=lambda _db: (_ for _ in ()).throw(
            RuntimeError("corrupted pack")
        ),
    )

    assert controller._read_graph(None, buddy_id="buddy-1") is None
    rendered = _rendered(captured)
    assert "RuntimeError" in rendered
    assert "corrupted pack" not in rendered
    assert _attached_tracebacks(captured) == []


def test_private_library_boundary_never_logs_its_exception_payload(
    captured, monkeypatch
):
    """The one handler whose comment promises no paths must keep that promise."""

    def _explode(*_args, **_kwargs):
        raise OSError(2, "No such file", "/Users/someone/private/buddies.db")

    monkeypatch.setattr(
        "tldw_chatbook.Persona_Buddy.library.BuddyLibrary", _explode, raising=True
    )

    controller = PersonaBuddyController(
        profile_db=object(),
        profile_root="/Users/someone/private/profile",
    )

    assert controller._read_local_buddy(BuddySelection(buddy_id="buddy-1")) is None
    rendered = _rendered(captured)
    assert "private library authority rejected" in rendered
    assert "/Users/someone/private" not in rendered
    assert _attached_tracebacks(captured) == []


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
