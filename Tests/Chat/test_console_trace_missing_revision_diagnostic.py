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


#: Trace-module raise sites whose argument is not a literal code, each pinned
#: to its argument expression, and why each stays ``validation`` (TASK-42471).
#: Another non-literal argument, in these functions or elsewhere, fails the
#: scan below, and so does a row no raise site uses any more.
_UNCODED_TRACE_VALUE_ERRORS = {
    # Fed literal field names; its callers' literals are scanned as codes.
    ("console_trace_service", "_required_string"): "field_name",
    # Each raises the name of a malformed field of an internal record. Some
    # callers pass the name through a variable, so no scan can prove it.
    ("console_trace_maintenance", "_validate_utc_timestamp"): "field_name",
    ("console_trace_repository", "_validate_token"): "field_name",
    ("console_trace_repository", "_nonempty"): "field_name",
    ("console_trace_repository", "_json_object"): "field_name",
    ("console_trace_repository", "_canonical_json_value"): "field_name",
    # A constructor bound's name, taken from a literal (value, name) table.
    ("console_trace_redaction", "__init__"): "name",
    ("console_trace_redaction", "redact_pii_value"): "name",
    ("console_trace_regex_worker", "__post_init__"): "name",
    # A legacy capture row's field, read back for display, not for a Send.
    ("console_trace_projection", "_decode_legacy_row"): "capture_error",
}
#: Raise sites that carry no code, counted so a new one is reviewed before
#: it ships (measured 2026-11-23): formatted messages, prose messages (a log
#: token cannot hold spaces; 42 distinct messages at 45 sites), and
#: argument-less raises.
_UNCODED_TRACE_RAISE_COUNTS = {"formatted": 9, "prose": 45, "bare": 2}


def _trace_refusal_codes() -> tuple[set[str], list[str], set, dict[str, int]]:
    """Scan every Chat/console_trace_*.py for the codes its refusals raise.

    Returns:
        The code-shaped literals raised as ``ValueError``,
        ``TraceReservationRefused`` or ``TraceSurfaceChangeRefused`` (and
        ``_required_string``'s field names); the raise sites whose argument
        is neither a literal nor its pinned row above; the rows used; and
        the counts of raises that carry no code.
    """
    import ast
    import re
    from pathlib import Path

    from tldw_chatbook.Chat import console_send_diagnostics

    token = re.compile(r"[a-z][a-z0-9_]*")
    raisers = {"ValueError", "TraceReservationRefused", "TraceSurfaceChangeRefused"}
    raised: set[str] = set()
    unchecked: list[str] = []
    used: set[tuple[str, str]] = set()
    counts = dict.fromkeys(_UNCODED_TRACE_RAISE_COUNTS, 0)
    chat = Path(console_send_diagnostics.__file__).parent
    for path in sorted(chat.glob("console_trace_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants = {
            target.id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and type(node.value.value) is str
            for target in node.targets
            if isinstance(target, ast.Name)
        }

        def literal(value: ast.AST, constants: dict[str, str] = constants) -> str | None:
            if isinstance(value, ast.Constant) and type(value.value) is str:
                return value.value
            return constants.get(value.id) if isinstance(value, ast.Name) else None

        def scan(node: ast.AST, function: str, stem: str = path.stem) -> None:
            for child in ast.iter_child_nodes(node):
                scan(
                    child,
                    child.name
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    else function,
                )
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Name):
                counts["bare"] += node.exc.id in raisers
                return
            if not isinstance(node, ast.Call):
                return
            callee = node.func
            name = (
                callee.id
                if isinstance(callee, ast.Name)
                else callee.attr
                if isinstance(callee, ast.Attribute)
                and isinstance(callee.value, ast.Name)
                and callee.value.id == "builtins"
                else None
            )
            if name == "_required_string" and len(node.args) == 2:
                arg = node.args[1]
            elif name in raisers and not node.args:
                # TraceSurfaceChangeRefused() supplies its own code.
                counts["bare"] += name != "TraceSurfaceChangeRefused"
                return
            elif name in raisers:
                arg = node.args[0] if len(node.args) == 1 else None
            else:
                return
            if isinstance(arg, ast.JoinedStr):
                counts["formatted"] += 1
                return
            # `code or "fallback"` (_rolled_back_refusal): only a Name or a
            # string Constant, in that order; the Constant is the code.
            if (
                isinstance(arg, ast.BoolOp)
                and isinstance(arg.op, ast.Or)
                and len(arg.values) == 2
                and isinstance(arg.values[0], ast.Name)
                and isinstance(arg.values[1], ast.Constant)
                and type(arg.values[1].value) is str
            ):
                found = arg.values[1].value
            else:
                found = None if arg is None else literal(arg)
            if found is not None:
                if token.fullmatch(found):
                    raised.add(found)
                else:
                    counts["prose"] += 1
            elif arg is not None and _UNCODED_TRACE_VALUE_ERRORS.get(
                (stem, function)
            ) == ast.unparse(arg):
                used.add((stem, function))
            else:
                unchecked.append(f"{stem}:{node.lineno}")

        scan(tree, "<module>")
    return raised, unchecked, used, counts


def test_every_coded_trace_refusal_has_a_send_diagnostic_category() -> None:
    """TASK-42471: a new refusal code cannot ship without a diagnostic category.

    The category set knew 15 of the 86 codes two trace modules raise, so
    TASK-42412's CI failure logged only ``ValueError``/``validation``. The set
    stays an exact, code-owned list (no token is ever derived from error
    text); this scan of every trace module is what keeps it complete.
    """
    from tldw_chatbook.Chat.console_send_diagnostics import _TRACE_FAILURE_CATEGORIES

    raised, unchecked, used, counts = _trace_refusal_codes()
    assert len(raised) >= 210  # measured 2026-11-24 (dev #3670 added one); the scan found the codes
    assert unchecked == []
    assert used == set(_UNCODED_TRACE_VALUE_ERRORS)
    assert counts == _UNCODED_TRACE_RAISE_COUNTS
    assert sorted(raised - _TRACE_FAILURE_CATEGORIES) == []
