"""TASK-34100.5 AC#7 (entry-exit-handoff-06), diagnostics half.

The first send after some setup exits failed with 'Trace capture blocked'
(trace_reservation failed, trace_revision_unavailable) in 2 of 5 live runs,
and nothing recorded WHICH revision the reservation could not find, so the
trigger stayed unidentified. The refusal now logs the missing opaque revision
ids (no content), which of them is the active turn's, and how many were found.
"""

from __future__ import annotations

import pytest
from loguru import logger

from Tests.Chat.test_console_trace_runtime import (
    _saved_message,
    _semantic_request,
)
from Tests.Chat.test_console_trace_runtime import (
    make_database as _database_fixture,
)
from Tests.Chat.test_console_trace_runtime import (
    make_gateway as _gateway_fixture,
)
from tldw_chatbook.Chat.console_prepared_request import (
    prepare_provider_request,
    resolve_request_capacity,
)
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    SavedRevisionTraceProvenance,
)
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory

make_database = _database_fixture
make_gateway = _gateway_fixture


def test_a_missing_revision_row_is_named_in_the_diagnostic(
    tmp_path, make_database
) -> None:
    database = make_database(tmp_path / "trace-missing.sqlite", "trace-missing")
    conversation_id = database.add_conversation({"title": "missing revision"})
    assert conversation_id is not None
    _prior_id, prior = _saved_message(database, conversation_id, "prior")
    ghost = SavedRevisionTraceProvenance(new_opaque_id())
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    semantic = _semantic_request(
        [
            {"role": "user", "content": "prior"},
            {"role": "user", "content": "active turn whose row is missing"},
        ],
        [prior, ghost],
        policy,
    )
    prepared = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        provider="openai",
        model="gpt-test",
        capacity=resolve_request_capacity(context_window_tokens=None),
        apply_safety_window=False,
    )
    records: list[str] = []
    sink = logger.add(lambda message: records.append(str(message)), level="WARNING")
    try:
        with pytest.raises(ValueError, match="trace_revision_unavailable"):
            ConsoleTraceBoundaryFactory(database)(
                prepared, None, ConsoleRequestRoute.FRESH
            )
    finally:
        logger.remove(sink)

    named = [line for line in records if "no saved revision row" in line]
    assert len(named) == 1, records
    assert ghost.revision_id in named[0]
    assert "active_missing=['" + ghost.revision_id in named[0]
    assert prior.revision_id not in named[0].split("present=")[0]
    assert "active turn whose row is missing" not in named[0]


@pytest.mark.asyncio
async def test_a_refusal_inside_the_reservation_is_typed_not_established(
    tmp_path, make_database, make_gateway
) -> None:
    """TASK-33621.6: the refusal rolls back, so no call can have been reserved.

    It used to reach the gateway as a plain ValueError and read reservation
    status "unknown": Cancel send refused ("Trace reservation status is
    unavailable") and Retry capture opened a false unknown-delivery panel.
    """
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderResolution,
        ConsoleProviderStreamSignals,
    )
    from tldw_chatbook.Chat.console_trace_errors import (
        TraceCallPersistenceError,
        TraceReservationRefused,
    )
    from tldw_chatbook.Chat.console_trace_provenance import ConsoleTraceCaptureMode

    database = make_database(tmp_path / "trace-refused.sqlite", "trace-refused")
    conversation_id = database.add_conversation({"title": "refused reservation"})
    _prior_id, prior = _saved_message(database, conversation_id, "prior")
    ghost = SavedRevisionTraceProvenance(new_opaque_id())
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    semantic = _semantic_request(
        [
            {"role": "user", "content": "prior"},
            {"role": "user", "content": "active turn whose row is missing"},
        ],
        [prior, ghost],
        policy,
    )
    factory = ConsoleTraceBoundaryFactory(database)
    prepared = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        provider="openai",
        model="gpt-test",
        capacity=resolve_request_capacity(context_window_tokens=None),
        apply_safety_window=False,
    )
    with pytest.raises(TraceReservationRefused) as refused:
        factory(prepared, None, ConsoleRequestRoute.FRESH)
    assert refused.value.args == ("trace_revision_unavailable",)
    assert isinstance(refused.value.__cause__, ValueError)
    with database.transaction() as cursor:
        assert cursor.execute("SELECT COUNT(*) FROM console_trace_calls").fetchone()[0] == 0

    gateway = make_gateway(trace_call_boundary_factory=factory)
    resolution = ConsoleProviderResolution(
        provider="openai",
        model="test-model",
        ready=True,
        base_url="https://api.openai.com/v1",
        execution_key="openai",
        streaming=False,
    )
    request = gateway.prepare_chat_request(
        resolution,
        semantic,
        route=ConsoleRequestRoute.FRESH,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )
    with pytest.raises(TraceCallPersistenceError) as caught:
        gateway._reserve_trace_call(
            request,
            resolution,
            ConsoleRequestRoute.FRESH,
            signals=ConsoleProviderStreamSignals(),
        )
    assert caught.value.reservation_status == "not_established"


@pytest.mark.asyncio
async def test_a_refused_reservation_logs_the_refusal_and_its_code(
    tmp_path, make_database, make_gateway
) -> None:
    """TASK-42471: the send diagnostic names the typed refusal and its code.

    The cause walk recorded only the innermost exception, the plain
    ``ValueError`` that ``TraceReservationRefused`` wraps, so the line never
    said the Send was refused (and rolled back) at reservation.
    """
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderResolution,
        ConsoleProviderStreamSignals,
    )
    from tldw_chatbook.Chat.console_send_diagnostics import send_diagnostic_scope
    from tldw_chatbook.Chat.console_trace_errors import TraceCallPersistenceError
    from tldw_chatbook.Chat.console_trace_provenance import ConsoleTraceCaptureMode

    database = make_database(tmp_path / "trace-logged.sqlite", "trace-logged")
    conversation_id = database.add_conversation({"title": "logged refusal"})
    _prior_id, prior = _saved_message(database, conversation_id, "prior")
    ghost = SavedRevisionTraceProvenance(new_opaque_id())
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    semantic = _semantic_request(
        [
            {"role": "user", "content": "prior"},
            {"role": "user", "content": "PRIVATE-DRAFT-42471"},
        ],
        [prior, ghost],
        policy,
    )
    gateway = make_gateway(
        trace_call_boundary_factory=ConsoleTraceBoundaryFactory(database)
    )
    resolution = ConsoleProviderResolution(
        provider="openai",
        model="test-model",
        ready=True,
        base_url="https://api.openai.com/v1",
        execution_key="openai",
        streaming=False,
    )
    request = gateway.prepare_chat_request(
        resolution,
        semantic,
        route=ConsoleRequestRoute.FRESH,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )
    records: list[dict[str, object]] = []
    monitor = SimpleNamespace(
        record_diagnostic=lambda _component, _event, **fields: records.append(fields)
    )
    async with send_diagnostic_scope("controller_submit", monitor):
        with pytest.raises(TraceCallPersistenceError):
            gateway._reserve_trace_call(
                request,
                resolution,
                ConsoleRequestRoute.FRESH,
                signals=ConsoleProviderStreamSignals(),
            )
    (failed,) = [
        fields
        for fields in records
        if (fields.get("phase"), fields.get("status"))
        == ("trace_reservation", "failed")
    ]
    assert failed["exception_type"] == "TraceReservationRefused"
    assert failed["error_category"] == "trace_revision_unavailable"
    assert "PRIVATE-DRAFT-42471" not in repr(records)
    assert ghost.revision_id not in repr(records)


def test_every_coded_trace_refusal_has_a_send_diagnostic_category() -> None:
    """TASK-42471: a new refusal code cannot ship without a diagnostic category.

    The category set knew 15 of the 86 codes these two modules raise, so
    TASK-42412's CI failure logged only ``ValueError``/``validation``. The set
    stays an exact, code-owned list (no token is ever derived from error
    text); this scan is what keeps it complete. A ``ValueError`` whose
    argument is not a literal fails here too, because nothing could check it.
    """
    import ast
    from pathlib import Path

    from tldw_chatbook.Chat import console_trace_runtime, console_trace_service
    from tldw_chatbook.Chat.console_send_diagnostics import _TRACE_FAILURE_CATEGORIES

    raised: set[str] = set()
    unchecked: list[str] = []
    for module in (console_trace_service, console_trace_runtime):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        # Its one ValueError(field_name) is fed the literals scanned below.
        exempt = {
            id(node)
            for function in ast.walk(tree)
            if isinstance(function, ast.FunctionDef)
            and function.name == "_required_string"
            for node in ast.walk(function)
        }
        for node in ast.walk(tree):
            if (
                id(node) in exempt
                or not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Name)
            ):
                continue
            if node.func.id == "ValueError":
                arg = node.args[0] if len(node.args) == 1 else None
            elif node.func.id == "_required_string":
                arg = node.args[1] if len(node.args) == 2 else None
            else:
                continue
            if isinstance(arg, ast.Constant) and type(arg.value) is str:
                raised.add(arg.value)
            else:
                unchecked.append(f"{module.__name__}:{node.lineno}")
    assert len(raised) > 80  # the scan found the codes, not nothing
    assert unchecked == []
    assert sorted(raised - _TRACE_FAILURE_CATEGORIES) == []
