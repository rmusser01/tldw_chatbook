"""Frozen census for provider calls that can enter Console preparation."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest

from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
from tldw_chatbook.Chat.console_trace_provenance import (
    CONSOLE_GATEWAY_CALLSITE_CENSUS,
    CONSOLE_REQUEST_ROUTE_CENSUS,
    ConsoleRequestRoute,
    ConsoleRouteCaptureDisposition,
    OmittedTraceProvenance,
    ProviderArtifactTraceProvenance,
    TraceProvenanceAlignmentError,
    TraceOmissionReason,
    TraceProvenanceSource,
    rag_provenance_for_route,
    request_route_provenance,
)


REPOSITORY_ROOT = Path(__file__).parents[2]

# TASK-42552: both censuses are keyed by function, not line number. Their line
# pins were stale from the day they landed (c78f641adf), and any edit above a
# pinned line broke them, so no merge-blocking lane could run this file.


class _GatewayCallVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.function_stack: list[str] = []
        self.calls: list[tuple[str, str, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"stream_chat", "complete_auxiliary"}
            and self.function_stack
        ):
            # ``route=`` wins; a ``**splat`` forwards it; "" binds none.
            route = next(
                (
                    ast.unparse(keyword.value)
                    for keyword in node.keywords
                    if keyword.arg == "route"
                ),
                next(
                    (
                        "**" + ast.unparse(keyword.value)
                        for keyword in node.keywords
                        if keyword.arg is None
                    ),
                    "",
                ),
            )
            self.calls.append((self.function_stack[-1], node.func.attr, route))
        self.generic_visit(node)


def _discover_gateway_callsites() -> Counter[tuple[str, str, str, str]]:
    discovered: Counter[tuple[str, str, str, str]] = Counter()
    source_root = REPOSITORY_ROOT / "tldw_chatbook"
    for path in source_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        # Only a file naming a gateway method can call one; skipping the rest
        # before parsing cuts the walk from about 14 s to under 1 s.
        if "stream_chat" not in source and "complete_auxiliary" not in source:
            continue
        visitor = _GatewayCallVisitor()
        visitor.visit(ast.parse(source, filename=str(path)))
        module = path.relative_to(REPOSITORY_ROOT).as_posix()
        discovered.update(
            (module, function, gateway, route)
            for function, gateway, route in visitor.calls
        )
    return discovered


def _policy() -> FrozenTracePolicy:
    return FrozenTracePolicy(
        policy_id=new_opaque_id(),
        credential_filter_version="credentials-v1",
        pii_redaction_enabled=False,
        pii_ruleset_revision_id=None,
    )


def test_logical_route_census_is_closed_complete_and_actor_aware() -> None:
    assert {item.route for item in CONSOLE_REQUEST_ROUTE_CENSUS} == set(
        ConsoleRequestRoute
    )
    actor_routes = {
        item.route for item in CONSOLE_REQUEST_ROUTE_CENSUS if item.actor_chain_required
    }
    assert actor_routes == {
        ConsoleRequestRoute.AGENT_FIRST,
        ConsoleRequestRoute.TOOL_LOOP,
    }
    assert (
        next(
            item
            for item in CONSOLE_REQUEST_ROUTE_CENSUS
            if item.route is ConsoleRequestRoute.AUTO_COMPACTION
        ).capture
        is ConsoleRouteCaptureDisposition.CAPTURE_OFF
    )
    # Manual summary goes through complete_auxiliary, like automatic
    # compaction; it always admits capture off and refuses a traced route, so
    # manual summary has never been traced. The other six carry no turn
    # preparation or capture mode, so they dispatch Capture Off today;
    # test_console_route_census_capture.py drives them (TASK-42552).
    capture_off_routes = {
        ConsoleRequestRoute.AUTO_COMPACTION,
        ConsoleRequestRoute.MANUAL_SUMMARY,
        ConsoleRequestRoute.RETRY,
        ConsoleRequestRoute.CONTINUE,
        ConsoleRequestRoute.REGENERATE,
        ConsoleRequestRoute.EDIT,
        ConsoleRequestRoute.IMPERSONATE,
        ConsoleRequestRoute.CITATION_REPAIR,
    }
    assert all(
        item.capture is ConsoleRouteCaptureDisposition.CAPTURE_OFF
        for item in CONSOLE_REQUEST_ROUTE_CENSUS
        if item.route in capture_off_routes
    )
    assert all(
        item.capture is ConsoleRouteCaptureDisposition.CONVERSATION_TRACE
        for item in CONSOLE_REQUEST_ROUTE_CENSUS
        if item.route not in capture_off_routes
    )
    assert all(
        item.source_marker and item.predicate for item in CONSOLE_REQUEST_ROUTE_CENSUS
    )
    assert len({item.source_marker for item in CONSOLE_REQUEST_ROUTE_CENSUS}) == len(
        CONSOLE_REQUEST_ROUTE_CENSUS
    )
    for item in CONSOLE_REQUEST_ROUTE_CENSUS:
        path = REPOSITORY_ROOT / item.source_module
        source = path.read_text(encoding="utf-8")
        source_lines = source.splitlines()
        tree = ast.parse(source, filename=str(path))
        marker_lines = [
            line
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == item.source_function
            for line in source_lines[node.lineno - 1 : node.end_lineno]
            if line.strip() == item.source_marker
        ]
        assert marker_lines, (item.route, item.source_function)


def test_gateway_callsite_census_matches_ast_discovery_bidirectionally() -> None:
    documented = Counter(
        (item.module, item.function, item.gateway, item.route_binding)
        for item in CONSOLE_GATEWAY_CALLSITE_CENSUS
    )
    assert documented == _discover_gateway_callsites()
    excluded = {
        item.module
        for item in CONSOLE_GATEWAY_CALLSITE_CENSUS
        if item.owner == "excluded"
    }
    assert {
        "tldw_chatbook/Chat/console_side_chat.py",
        "tldw_chatbook/Chat/console_visual_evaluation.py",
        "tldw_chatbook/Prompt_Management/prompt_improvement_service.py",
    } <= excluded
    assert all(
        item.route_binding is not None for item in CONSOLE_GATEWAY_CALLSITE_CENSUS
    )
    # A traced callsite must bind a route; only an untraced one may bind none.
    assert all(
        item.route_binding not in {"", "None"}
        for item in CONSOLE_GATEWAY_CALLSITE_CENSUS
        if item.owner == ConsoleRouteCaptureDisposition.CONVERSATION_TRACE.value
    )
    # An excluded callsite carries no route; a Capture-Off one only routes the
    # census calls Capture Off; a traced one carries at least one traced route.
    route_capture = {item.route: item.capture for item in CONSOLE_REQUEST_ROUTE_CENSUS}
    for item in CONSOLE_GATEWAY_CALLSITE_CENSUS:
        captures = {route_capture[route] for route in item.routes}
        if item.owner == ConsoleRouteCaptureDisposition.EXCLUDED.value:
            assert not item.routes, item
        elif item.owner == ConsoleRouteCaptureDisposition.CAPTURE_OFF.value:
            assert captures <= {ConsoleRouteCaptureDisposition.CAPTURE_OFF}, item
        else:
            assert ConsoleRouteCaptureDisposition.CONVERSATION_TRACE in captures, item
    routed = [
        route for item in CONSOLE_GATEWAY_CALLSITE_CENSUS for route in item.routes
    ]
    assert set(routed) == set(ConsoleRequestRoute)
    # Speculative duplex voice (b7dd8e53f2) dispatches a Capture-On fresh turn
    # from its own callsite; every other route has exactly one.
    assert [route for route, count in Counter(routed).items() if count > 1] == [
        ConsoleRequestRoute.FRESH
    ]


def test_route_provenance_carries_predicate_and_required_actor_chain() -> None:
    fresh = request_route_provenance(ConsoleRequestRoute.FRESH)
    assert fresh.predicate == "fresh_submit"
    assert fresh.actor_id is None
    assert fresh.chain_id is None

    with pytest.raises(TraceProvenanceAlignmentError, match="actor"):
        request_route_provenance(ConsoleRequestRoute.AGENT_FIRST)

    actor_id = new_opaque_id()
    chain_id = new_opaque_id()
    agent = request_route_provenance(
        ConsoleRequestRoute.AGENT_FIRST,
        actor_id=actor_id,
        chain_id=chain_id,
    )
    assert agent.predicate == "agent_first_wake"
    assert agent.actor_id == actor_id
    assert agent.chain_id == chain_id
    assert actor_id not in repr(agent)
    assert chain_id not in repr(agent)

    with pytest.raises(TraceProvenanceAlignmentError, match="opaque"):
        request_route_provenance(
            ConsoleRequestRoute.AGENT_FIRST,
            actor_id="RAW PRIVATE USER TEXT",
            chain_id="sk-secret-value",
        )


def test_fresh_retry_and_agent_rag_absence_is_explicit() -> None:
    assert rag_provenance_for_route(ConsoleRequestRoute.FRESH, None) == (
        OmittedTraceProvenance(
            TraceProvenanceSource.RAG_CONTEXT,
            TraceOmissionReason.FRESH_RAG_NOT_SELECTED,
        )
    )
    assert rag_provenance_for_route(ConsoleRequestRoute.RETRY, None) == (
        OmittedTraceProvenance(
            TraceProvenanceSource.RAG_CONTEXT,
            TraceOmissionReason.RETRY_RAG_NOT_REPLAYED,
        )
    )
    assert rag_provenance_for_route(ConsoleRequestRoute.AGENT_FIRST, None) == (
        OmittedTraceProvenance(
            TraceProvenanceSource.RAG_CONTEXT,
            TraceOmissionReason.AGENT_WAKE_RAG_SKIPPED,
        )
    )
    selected = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.RAG_CONTEXT,
        _policy(),
    )
    assert rag_provenance_for_route(ConsoleRequestRoute.FRESH, selected) is selected
