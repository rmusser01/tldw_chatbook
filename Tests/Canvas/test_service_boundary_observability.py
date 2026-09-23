"""A sanitized Canvas boundary failure must still leave a record.

TASK-32901 (tier-2 S13 P2): ``Canvas/`` had 167 ``except`` clauses and zero
``logger`` calls. Six ladders in ``Canvas/service.py`` map an unexpected
dependency failure to a generic ``operation_failed`` and re-raise it *after*
the handler, so ``__context__`` is unset too -- a "Canvas won't open" report
left nothing anywhere in the process to act on.

ADR-121 forbids logging arbitrary exception representations, so the record is
the exception *class* plus the stable refusal code, never its message. These
tests pin both halves: something is logged, and the payload is not.
"""

from __future__ import annotations

import pytest
from loguru import logger

from tldw_chatbook.Canvas import service as canvas_service


@pytest.fixture()
def captured():
    records: list = []
    sink_id = logger.add(lambda record: records.append(record), level="DEBUG")
    try:
        yield records
    finally:
        logger.remove(sink_id)


def test_unexpected_boundary_failure_is_recorded_by_class_not_payload(captured):
    secret = "user-private-path-/Users/someone/notes.md"
    error = canvas_service._boundary_error(RuntimeError(secret), "operation_failed")

    assert isinstance(error, canvas_service.CanvasServiceError)
    assert error.code == "operation_failed"

    rendered = "".join(str(record.record["message"]) for record in captured)
    assert "RuntimeError" in rendered
    assert "operation_failed" in rendered
    assert secret not in rendered


def test_storage_failures_are_recorded_too(captured):
    error = canvas_service._boundary_error(ValueError("boom"), "storage_failure")

    assert error.code == "storage_failure"
    rendered = "".join(str(record.record["message"]) for record in captured)
    assert "ValueError" in rendered
    assert "boom" not in rendered
