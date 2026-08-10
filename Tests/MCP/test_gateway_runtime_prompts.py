"""Strict prompt registration, argument, and result tests for the MCP gateway."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, Optional

import pytest

gateway = pytest.importorskip(
    "mcp_unified.gateway", reason="mcp-unified extra not installed"
)
GatewayApplicationError = gateway.GatewayApplicationError
GatewayRequestContext = gateway.GatewayRequestContext

from tldw_chatbook.MCP.gateway_runtime import ChatbookGatewayRuntime  # noqa: E402
from tldw_chatbook.MCP.server import _describe_local_prompts  # noqa: E402


PROMPT_NAMES = [
    "summarize_conversation",
    "generate_document",
    "analyze_media",
    "search_and_synthesize",
    "character_writing",
]
_DEFAULT_RESULT = object()

EXPECTED_PROMPT_DESCRIPTORS = [
    {
        "name": "summarize_conversation",
        "description": "Generate a prompt to summarize a conversation.",
        "arguments": [
            {"name": "conversation_id", "required": True},
            {"name": "style", "required": False},
            {"name": "focus", "required": False},
        ],
    },
    {
        "name": "generate_document",
        "description": "Generate a prompt to create a document from a conversation.",
        "arguments": [
            {"name": "conversation_id", "required": True},
            {"name": "doc_type", "required": False},
            {"name": "format", "required": False},
        ],
    },
    {
        "name": "analyze_media",
        "description": "Generate a prompt to analyze ingested media.",
        "arguments": [
            {"name": "media_id", "required": True},
            {"name": "analysis_type", "required": False},
            {"name": "detail_level", "required": False},
        ],
    },
    {
        "name": "search_and_synthesize",
        "description": "Generate a prompt to search RAG and synthesize results.",
        "arguments": [
            {"name": "query", "required": True},
            {"name": "num_sources", "required": False},
            {"name": "synthesis_type", "required": False},
        ],
    },
    {
        "name": "character_writing",
        "description": "Generate a prompt for character-based writing.",
        "arguments": [
            {"name": "character_id", "required": True},
            {"name": "writing_type", "required": False},
            {"name": "context", "required": False},
            {"name": "style_notes", "required": False},
        ],
    },
]


def _context() -> GatewayRequestContext:
    return GatewayRequestContext(request_id="prompt-test")


def _runtime() -> ChatbookGatewayRuntime:
    return ChatbookGatewayRuntime(
        name="tldw_chatbook",
        version="0.1.0",
        tool_descriptors=[],
    )


def _register_prompts(
    runtime: ChatbookGatewayRuntime,
    *,
    summarize_result: object | Callable[..., object] = _DEFAULT_RESULT,
) -> dict[str, Any]:
    calls: list[tuple[int, str, Optional[str]]] = []

    @runtime.prompt()
    async def summarize_conversation(
        conversation_id: int,
        style: str = "concise",
        focus: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """Generate a prompt to summarize a conversation."""
        calls.append((conversation_id, style, focus))
        if callable(summarize_result):
            return summarize_result(conversation_id, style, focus)  # type: ignore[return-value]
        if summarize_result is not _DEFAULT_RESULT:
            return summarize_result  # type: ignore[return-value]
        return [
            {
                "role": "user",
                "content": f"{conversation_id}|{style}|{focus}",
            }
        ]

    @runtime.prompt()
    async def generate_document(
        conversation_id: int,
        doc_type: str = "summary",
        format: str = "markdown",
    ) -> list[dict[str, str]]:
        """Generate a prompt to create a document from a conversation."""
        return [{"role": "user", "content": str(conversation_id)}]

    @runtime.prompt()
    async def analyze_media(
        media_id: int,
        analysis_type: str = "summary",
        detail_level: str = "medium",
    ) -> list[dict[str, str]]:
        """Generate a prompt to analyze ingested media."""
        return [{"role": "user", "content": str(media_id)}]

    @runtime.prompt()
    async def search_and_synthesize(
        query: str,
        num_sources: int = 5,
        synthesis_type: str = "overview",
    ) -> list[dict[str, str]]:
        """Generate a prompt to search RAG and synthesize results."""
        return [{"role": "user", "content": query}]

    @runtime.prompt()
    async def character_writing(
        character_id: int,
        writing_type: str = "response",
        context: Optional[str] = None,
        style_notes: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """Generate a prompt for character-based writing."""
        return [{"role": "user", "content": str(character_id)}]

    return {
        "calls": calls,
        "summarize": summarize_conversation,
    }


def _ready_runtime(
    *, summarize_result: object | Callable[..., object] = _DEFAULT_RESULT
) -> tuple[ChatbookGatewayRuntime, dict[str, Any]]:
    runtime = _runtime()
    state = _register_prompts(runtime, summarize_result=summarize_result)
    runtime.finalize()
    return runtime, state


def _assert_prompt_error(
    exc_info: pytest.ExceptionInfo[GatewayApplicationError],
    *,
    message: str,
    reason_code: str,
) -> None:
    assert exc_info.value.public_message == message
    assert exc_info.value.reason_code == reason_code
    assert exc_info.value.kind == "prompt"


def test_ast_prompt_descriptors_have_exact_names_arguments_and_no_annotations() -> None:
    descriptors = _describe_local_prompts()

    assert descriptors == EXPECTED_PROMPT_DESCRIPTORS
    assert [descriptor["name"] for descriptor in descriptors] == PROMPT_NAMES
    assert all(
        set(argument) <= {"name", "description", "required"}
        for descriptor in descriptors
        for argument in descriptor["arguments"]
    )


def test_runtime_lists_exact_prompt_descriptors_as_defensive_ordered_copies() -> None:
    runtime, _state = _ready_runtime()

    first = asyncio.run(runtime.list_prompts(_context()))
    first[0]["name"] = "mutated"
    first[0]["arguments"][0]["name"] = "mutated"
    second = asyncio.run(runtime.list_prompts(_context()))

    assert second == EXPECTED_PROMPT_DESCRIPTORS
    assert [descriptor["name"] for descriptor in second] == PROMPT_NAMES


def test_runtime_requires_all_five_chatbook_prompts() -> None:
    runtime = _runtime()

    @runtime.prompt()
    async def summarize_conversation(conversation_id: int) -> list[dict[str, str]]:
        return [{"role": "user", "content": str(conversation_id)}]

    with pytest.raises(ValueError, match="prompt"):
        runtime.finalize()


def test_runtime_rejects_duplicate_prompt_names() -> None:
    runtime = _runtime()

    @runtime.prompt(name="summarize_conversation")
    async def first(conversation_id: int) -> list[dict[str, str]]:
        return [{"role": "user", "content": str(conversation_id)}]

    with pytest.raises(ValueError, match="duplicate prompt"):

        @runtime.prompt(name="summarize_conversation")
        async def second(conversation_id: int) -> list[dict[str, str]]:
            return [{"role": "user", "content": str(conversation_id)}]


def test_runtime_rejects_unbounded_prompt_argument_names() -> None:
    runtime = _runtime()
    argument_name = "x" * 129

    async def summarize_conversation(value: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": value}]

    summarize_conversation.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [
            inspect.Parameter(
                argument_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=str,
            )
        ]
    )
    summarize_conversation.__annotations__[argument_name] = str

    with pytest.raises(ValueError, match="prompt argument name"):
        runtime.prompt()(summarize_conversation)


@pytest.mark.parametrize("annotation", [float, list[str], Optional[int]])
def test_runtime_rejects_unsupported_prompt_parameter_types(annotation: object) -> None:
    runtime = _runtime()

    async def summarize_conversation(value: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": value}]

    summarize_conversation.__annotations__["value"] = annotation

    with pytest.raises(ValueError, match="prompt parameter type"):
        runtime.prompt()(summarize_conversation)


def test_prompt_arguments_accept_json_primitives_coerce_only_ints_and_use_defaults() -> (
    None
):
    runtime, state = _ready_runtime()

    explicit = asyncio.run(
        runtime.get_prompt(
            "summarize_conversation",
            {"conversation_id": 7, "style": "0007", "focus": None},
            _context(),
        )
    )
    coerced = asyncio.run(
        runtime.get_prompt(
            "summarize_conversation",
            {"conversation_id": "42"},
            _context(),
        )
    )

    assert explicit["messages"][0]["content"]["text"] == "7|0007|None"
    assert coerced["messages"][0]["content"]["text"] == "42|concise|None"
    assert state["calls"] == [(7, "0007", None), (42, "concise", None)]


def test_absent_optional_arguments_are_omitted_so_python_defaults_apply() -> None:
    runtime, state = _ready_runtime()
    handler = state["summarize"]
    handler.__defaults__ = ("changed-python-default", None)

    result = asyncio.run(
        runtime.get_prompt(
            "summarize_conversation",
            {"conversation_id": 9},
            _context(),
        )
    )

    assert result["messages"][0]["content"]["text"] == ("9|changed-python-default|None")


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"conversation_id": 1, "unknown": "value"},
        {"conversation_id": True},
        {"conversation_id": "4.2"},
        {"conversation_id": " 4"},
        {"conversation_id": 4.0},
        {"conversation_id": None},
        {"conversation_id": 1, "style": 2},
        [],
        "not-an-object",
        None,
    ],
)
def test_invalid_prompt_arguments_fail_before_the_handler(arguments: object) -> None:
    runtime, state = _ready_runtime()

    with pytest.raises(GatewayApplicationError) as exc_info:
        asyncio.run(
            runtime.get_prompt(
                "summarize_conversation",
                arguments,  # type: ignore[arg-type]
                _context(),
            )
        )

    _assert_prompt_error(
        exc_info,
        message="Invalid prompt arguments.",
        reason_code="invalid_prompt_arguments",
    )
    assert state["calls"] == []


def test_unknown_prompt_name_has_a_bounded_non_reflective_error() -> None:
    runtime, _state = _ready_runtime()

    with pytest.raises(GatewayApplicationError) as exc_info:
        asyncio.run(
            runtime.get_prompt(
                "secret-unknown-prompt",
                {},
                _context(),
            )
        )

    _assert_prompt_error(
        exc_info,
        message="Prompt not found.",
        reason_code="prompt_not_found",
    )
    assert "secret-unknown-prompt" not in str(exc_info.value)


def test_prompt_result_passes_user_and_assistant_through_as_text_blocks() -> None:
    raw = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Draft"},
    ]
    runtime, _state = _ready_runtime(summarize_result=raw)

    result = asyncio.run(
        runtime.get_prompt("summarize_conversation", {"conversation_id": 1}, _context())
    )

    assert result == {
        "messages": [
            {"role": "user", "content": {"type": "text", "text": "Question"}},
            {
                "role": "assistant",
                "content": {"type": "text", "text": "Draft"},
            },
        ]
    }


def test_prompt_result_folds_only_the_contiguous_leading_system_block() -> None:
    raw = [
        {"role": "system", "content": "system one"},
        {"role": "system", "content": "system two"},
        {"role": "user", "content": "original user text"},
        {"role": "assistant", "content": "Draft"},
    ]
    runtime, _state = _ready_runtime(summarize_result=raw)

    result = asyncio.run(
        runtime.get_prompt("summarize_conversation", {"conversation_id": 1}, _context())
    )

    assert result == {
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": (
                        "System instructions:\nsystem one\n\nsystem two"
                        "\n\nUser request:\noriginal user text"
                    ),
                },
            },
            {
                "role": "assistant",
                "content": {"type": "text", "text": "Draft"},
            },
        ]
    }


@pytest.mark.parametrize(
    "raw_result",
    [
        [],
        None,
        {},
        "not-a-list",
        ["not-a-dict"],
        [{}],
        [{"role": "user"}],
        [{"content": "missing role"}],
        [{"role": "user", "content": 1}],
        [{"role": "tool", "content": "unknown role"}],
        [{"role": "system", "content": "trailing"}],
        [
            {"role": "system", "content": "instructions"},
            {"role": "assistant", "content": "not the first user"},
        ],
        [
            {"role": "user", "content": "first"},
            {"role": "system", "content": "mid-stream"},
        ],
    ],
)
def test_invalid_prompt_results_fail_closed_without_reflecting_content(
    raw_result: object,
) -> None:
    runtime, _state = _ready_runtime(summarize_result=raw_result)

    with pytest.raises(GatewayApplicationError) as exc_info:
        asyncio.run(
            runtime.get_prompt(
                "summarize_conversation", {"conversation_id": 1}, _context()
            )
        )

    _assert_prompt_error(
        exc_info,
        message="Prompt handler returned an invalid result.",
        reason_code="invalid_prompt_result",
    )
    assert "mid-stream" not in str(exc_info.value)


def test_search_and_synthesize_awaits_keyword_search(monkeypatch) -> None:
    from tldw_chatbook.MCP.prompts import MCPPrompts
    from tldw_chatbook.RAG_Search.simplified import search_service

    calls: list[tuple[str, int, object]] = []

    class AsyncSearch:
        def __init__(self, media_db: object) -> None:
            assert media_db is sentinel_db

        async def keyword_search(
            self,
            query: str,
            limit: int = 10,
            media_types: Optional[list[str]] = None,
        ) -> list[dict[str, str]]:
            calls.append((query, limit, media_types))
            return [
                {
                    "title": "Awaited source",
                    "media_type": "document",
                    "content": "awaited search content",
                }
            ]

    sentinel_db = object()
    monkeypatch.setattr(search_service, "SimplifiedRAGSearchService", AsyncSearch)
    prompts = MCPPrompts(object(), sentinel_db)  # type: ignore[arg-type]

    result = asyncio.run(
        prompts.search_and_synthesize_prompt(
            query="await me", num_sources=3, synthesis_type="overview"
        )
    )

    assert calls == [("await me", 3, None)]
    assert "awaited search content" in result[0]["content"]
