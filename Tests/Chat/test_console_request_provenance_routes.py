"""Frozen census for provider calls that can enter Console preparation."""

from __future__ import annotations

import ast
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


class _GatewayCallVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.function_stack: list[str] = []
        self.calls: list[tuple[str, str, int, str | None]] = []

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
            route = next(
                (
                    ast.unparse(keyword.value)
                    for keyword in node.keywords
                    if keyword.arg == "route"
                ),
                None,
            )
            self.calls.append(
                (self.function_stack[-1], node.func.attr, node.lineno, route)
            )
        self.generic_visit(node)


def _discover_gateway_callsites() -> set[tuple[str, str, str, int, str | None]]:
    discovered: set[tuple[str, str, str, int, str | None]] = set()
    source_root = REPOSITORY_ROOT / "tldw_chatbook"
    for path in source_root.rglob("*.py"):
        visitor = _GatewayCallVisitor()
        visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        module = path.relative_to(REPOSITORY_ROOT).as_posix()
        discovered.update(
            (module, function, gateway, line, route)
            for function, gateway, line, route in visitor.calls
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
    assert all(
        item.capture is ConsoleRouteCaptureDisposition.CONVERSATION_TRACE
        for item in CONSOLE_REQUEST_ROUTE_CENSUS
        if item.route is not ConsoleRequestRoute.AUTO_COMPACTION
    )
    assert all(
        item.source_marker and item.predicate for item in CONSOLE_REQUEST_ROUTE_CENSUS
    )
    assert len({item.source_marker for item in CONSOLE_REQUEST_ROUTE_CENSUS}) == len(
        CONSOLE_REQUEST_ROUTE_CENSUS
    )
    for item in CONSOLE_REQUEST_ROUTE_CENSUS:
        path = REPOSITORY_ROOT / item.source_module
        source_lines = path.read_text(encoding="utf-8").splitlines()
        assert source_lines[item.source_line - 1].strip() == item.source_marker
        tree = ast.parse("\n".join(source_lines), filename=str(path))
        owning_functions = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.lineno <= item.source_line <= (node.end_lineno or node.lineno)
        }
        assert item.source_function in owning_functions


def test_gateway_callsite_census_matches_ast_discovery_bidirectionally() -> None:
    documented = {
        (
            item.module,
            item.function,
            item.gateway,
            item.source_line,
            item.route_binding,
        )
        for item in CONSOLE_GATEWAY_CALLSITE_CENSUS
    }
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
    identities = [
        (item.module, item.function, item.gateway, item.source_line)
        for item in CONSOLE_GATEWAY_CALLSITE_CENSUS
    ]
    assert len(identities) == len(set(identities))
    assert all(
        item.route_binding is not None for item in CONSOLE_GATEWAY_CALLSITE_CENSUS
    )
    assert all(
        bool(item.routes)
        == (item.owner == ConsoleRouteCaptureDisposition.CONVERSATION_TRACE.value)
        for item in CONSOLE_GATEWAY_CALLSITE_CENSUS
        if ConsoleRequestRoute.AUTO_COMPACTION not in item.routes
    )
    routed = [
        route for item in CONSOLE_GATEWAY_CALLSITE_CENSUS for route in item.routes
    ]
    assert set(routed) == set(ConsoleRequestRoute)
    assert len(routed) == len(set(routed))


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
