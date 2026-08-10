"""Strict prompt registration, argument, and result tests for the MCP gateway."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from typing import Any, Optional

import pytest
from loguru import logger

gateway = pytest.importorskip(
    "mcp_unified.gateway", reason="mcp-unified extra not installed"
)
GatewayApplicationError = gateway.GatewayApplicationError
GatewayLimits = gateway.GatewayLimits
GatewayRequestContext = gateway.GatewayRequestContext

from tldw_chatbook.MCP import gateway_runtime as gateway_runtime_module  # noqa: E402
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
_PRIVATE_SENTINEL = "/Users/private/.config/tldw api_key=sk-review-sentinel"
_PUBLIC_PROMPT_FAILURE = [{"role": "user", "content": "Unable to create prompt."}]

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


def test_prompt_signature_accepts_pep604_optional_and_keyword_only_arguments() -> None:
    runtime = _runtime()

    @runtime.prompt(name="summarize_conversation")
    async def handler(
        value: str | None = None, *, style: str = "concise"
    ) -> list[dict[str, str]]:
        return [{"role": "user", "content": f"{value}|{style}"}]

    assert runtime._prompt_descriptors["summarize_conversation"]["arguments"] == [
        {"name": "value", "required": False},
        {"name": "style", "required": False},
    ]


def test_prompt_signature_rejects_invalid_primitive_defaults() -> None:
    async def bool_int(value: int = True) -> list[dict[str, str]]:
        return [{"role": "user", "content": str(value)}]

    async def null_string(value: str = None) -> list[dict[str, str]]:  # type: ignore[assignment]
        return [{"role": "user", "content": str(value)}]

    async def int_optional_string(
        value: Optional[str] = 1,  # type: ignore[assignment]
    ) -> list[dict[str, str]]:
        return [{"role": "user", "content": str(value)}]

    for handler in (bool_int, null_string, int_optional_string):
        with pytest.raises(ValueError, match="prompt parameter default"):
            _runtime().prompt(name="summarize_conversation")(handler)


@pytest.mark.parametrize("kind", ["positional_only", "args", "kwargs"])
def test_prompt_signature_rejects_non_keyword_dispatch_parameters(kind: str) -> None:
    async def positional_only(value: str, /) -> list[dict[str, str]]:
        return [{"role": "user", "content": value}]

    async def args(*values: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": str(values)}]

    async def kwargs(**values: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": str(values)}]

    handler = {
        "positional_only": positional_only,
        "args": args,
        "kwargs": kwargs,
    }[kind]
    with pytest.raises(ValueError, match="prompt parameter kind"):
        _runtime().prompt(name="summarize_conversation")(handler)


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


@pytest.mark.parametrize(
    ("wire_value", "expected"),
    [
        ("-7", -7),
        ("0", 0),
        ("-0", 0),
        ("1" * 128, int("1" * 128)),
    ],
)
def test_prompt_integer_string_grammar_accepts_pinned_forms(
    wire_value: str, expected: int
) -> None:
    runtime, state = _ready_runtime()

    asyncio.run(
        runtime.get_prompt(
            "summarize_conversation",
            {"conversation_id": wire_value},
            _context(),
        )
    )

    assert state["calls"][0][0] == expected


@pytest.mark.parametrize("wire_value", ["+1", "00", "01", "-01", "1" * 129])
def test_prompt_integer_string_grammar_rejects_unpinned_forms(
    wire_value: str,
) -> None:
    runtime, state = _ready_runtime()

    with pytest.raises(GatewayApplicationError) as exc_info:
        asyncio.run(
            runtime.get_prompt(
                "summarize_conversation",
                {"conversation_id": wire_value},
                _context(),
            )
        )

    _assert_prompt_error(
        exc_info,
        message="Invalid prompt arguments.",
        reason_code="invalid_prompt_arguments",
    )
    assert state["calls"] == []


def test_prompt_integer_subclasses_are_rejected_without_conversion() -> None:
    class IntegerSubclass(int):
        def __int__(self) -> int:
            raise RuntimeError(_PRIVATE_SENTINEL)

    runtime, state = _ready_runtime()

    with pytest.raises(GatewayApplicationError) as exc_info:
        asyncio.run(
            runtime.get_prompt(
                "summarize_conversation",
                {"conversation_id": IntegerSubclass(4)},
                _context(),
            )
        )
    _assert_prompt_error(
        exc_info,
        message="Invalid prompt arguments.",
        reason_code="invalid_prompt_arguments",
    )
    assert _PRIVATE_SENTINEL not in str(exc_info.value)
    assert state["calls"] == []


@pytest.mark.parametrize(
    "wire_value",
    [
        10**gateway_runtime_module._MAX_PROMPT_INTEGER_CHARS - 1,
        -(10 ** (gateway_runtime_module._MAX_PROMPT_INTEGER_CHARS - 1) - 1),
    ],
    ids=["positive-128-digits", "negative-128-characters"],
)
def test_prompt_exact_integers_accept_the_string_domain_limits(wire_value: int) -> None:
    runtime, state = _ready_runtime()

    asyncio.run(
        runtime.get_prompt(
            "summarize_conversation",
            {"conversation_id": wire_value},
            _context(),
        )
    )

    assert state["calls"][0][0] == wire_value


@pytest.mark.parametrize(
    "wire_value",
    [
        10**gateway_runtime_module._MAX_PROMPT_INTEGER_CHARS,
        -(10 ** (gateway_runtime_module._MAX_PROMPT_INTEGER_CHARS - 1)),
        10**10_000,
    ],
    ids=["positive-one-over", "negative-one-over", "very-large-no-conversion"],
)
def test_prompt_exact_integers_reject_values_outside_the_string_domain(
    wire_value: int,
) -> None:
    runtime, state = _ready_runtime()

    with pytest.raises(GatewayApplicationError) as exc_info:
        asyncio.run(
            runtime.get_prompt(
                "summarize_conversation",
                {"conversation_id": wire_value},
                _context(),
            )
        )

    _assert_prompt_error(
        exc_info,
        message="Invalid prompt arguments.",
        reason_code="invalid_prompt_arguments",
    )
    assert state["calls"] == []


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


def test_prompt_handler_exception_becomes_a_context_free_bounded_error(
    capsys,
) -> None:
    def fail(*_args: object) -> object:
        raise RuntimeError(_PRIVATE_SENTINEL)

    runtime, _state = _ready_runtime(summarize_result=fail)
    records: list[tuple[str, object]] = []
    sink_id = logger.add(
        lambda message: records.append((str(message), message.record["exception"]))
    )
    try:
        with pytest.raises(GatewayApplicationError) as exc_info:
            asyncio.run(
                runtime.get_prompt(
                    "summarize_conversation",
                    {"conversation_id": 1},
                    _context(),
                )
            )
    finally:
        logger.remove(sink_id)

    _assert_prompt_error(
        exc_info,
        message="Prompt handler returned an invalid result.",
        reason_code="invalid_prompt_result",
    )
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    captured = capsys.readouterr()
    assert all(exception is None for _message, exception in records)
    assert all(
        _PRIVATE_SENTINEL not in value
        for value in [
            str(exc_info.value),
            *(message for message, _exception in records),
            captured.out,
            captured.err,
        ]
    )


def test_prompt_handler_cancellation_propagates() -> None:
    def cancel(*_args: object) -> object:
        raise asyncio.CancelledError

    runtime, _state = _ready_runtime(summarize_result=cancel)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            runtime.get_prompt(
                "summarize_conversation",
                {"conversation_id": 1},
                _context(),
            )
        )


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


def test_prompt_result_folds_many_leading_system_messages_in_order() -> None:
    system_text = [f"instruction {index}" for index in range(2_000)]
    raw = [
        *({"role": "system", "content": text} for text in system_text),
        {"role": "user", "content": "request"},
    ]
    runtime, _state = _ready_runtime(summarize_result=raw)

    result = asyncio.run(
        runtime.get_prompt("summarize_conversation", {"conversation_id": 1}, _context())
    )

    assert result["messages"] == [
        {
            "role": "user",
            "content": {
                "type": "text",
                "text": (
                    "System instructions:\n"
                    + "\n\n".join(system_text)
                    + "\n\nUser request:\nrequest"
                ),
            },
        }
    ]


def _serialized_prompt_bytes(text: str) -> int:
    result = {"messages": [{"role": "user", "content": {"type": "text", "text": text}}]}
    return len(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def test_prompt_result_accepts_exact_byte_budget_and_rejects_one_over() -> None:
    limit = GatewayLimits().max_result_bytes
    overhead = _serialized_prompt_bytes("")
    exact_text = "x" * (limit - overhead)
    exact_runtime, _state = _ready_runtime(
        summarize_result=[{"role": "user", "content": exact_text}]
    )
    over_runtime, _state = _ready_runtime(
        summarize_result=[{"role": "user", "content": exact_text + "x"}]
    )

    exact = asyncio.run(
        exact_runtime.get_prompt(
            "summarize_conversation", {"conversation_id": 1}, _context()
        )
    )
    assert _serialized_prompt_bytes(exact["messages"][0]["content"]["text"]) == limit

    with pytest.raises(GatewayApplicationError) as exc_info:
        asyncio.run(
            over_runtime.get_prompt(
                "summarize_conversation", {"conversation_id": 1}, _context()
            )
        )
    _assert_prompt_error(
        exc_info,
        message="Prompt handler returned an invalid result.",
        reason_code="invalid_prompt_result",
    )


@pytest.mark.parametrize(
    ("unit", "serialized_unit_bytes"),
    [("é", 2), ("\x00", 6)],
    ids=["multibyte-utf8", "json-escaped-control"],
)
def test_prompt_result_budget_uses_compact_utf8_json_bytes(
    unit: str, serialized_unit_bytes: int
) -> None:
    limit = GatewayLimits().max_result_bytes
    overhead = _serialized_prompt_bytes("")
    assert _serialized_prompt_bytes(unit) - overhead == serialized_unit_bytes
    repetitions, remainder = divmod(limit - overhead, serialized_unit_bytes)
    exact_text = unit * repetitions + "x" * remainder
    exact_runtime, _state = _ready_runtime(
        summarize_result=[{"role": "user", "content": exact_text}]
    )
    over_runtime, _state = _ready_runtime(
        summarize_result=[{"role": "user", "content": exact_text + "x"}]
    )

    exact = asyncio.run(
        exact_runtime.get_prompt(
            "summarize_conversation", {"conversation_id": 1}, _context()
        )
    )
    assert _serialized_prompt_bytes(exact["messages"][0]["content"]["text"]) == limit

    with pytest.raises(GatewayApplicationError) as exc_info:
        asyncio.run(
            over_runtime.get_prompt(
                "summarize_conversation", {"conversation_id": 1}, _context()
            )
        )
    _assert_prompt_error(
        exc_info,
        message="Prompt handler returned an invalid result.",
        reason_code="invalid_prompt_result",
    )


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
        [{"role": "user", "content": float("nan")}],
        [{"role": "tool", "content": "unknown role"}],
        [{"role": "user", "content": "extra", "extra": "field"}],
        [{"role": "user", "content": "\ud800"}],
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


@pytest.mark.parametrize(
    ("method_name", "arguments"),
    [
        ("summarize_conversation_prompt", {"conversation_id": 1}),
        ("generate_document_prompt", {"conversation_id": 1}),
        ("analyze_media_prompt", {"media_id": 1}),
        ("search_and_synthesize_prompt", {"query": "private query"}),
        ("character_writing_prompt", {"character_id": 1}),
    ],
)
def test_prompt_fallbacks_never_expose_internal_exception_text(
    method_name: str,
    arguments: dict[str, object],
    monkeypatch,
    capsys,
) -> None:
    from tldw_chatbook.MCP.prompts import MCPPrompts
    from tldw_chatbook.RAG_Search.simplified import search_service

    search_calls: list[tuple[str, int, object]] = []

    class ExplodingConversationDB:
        def get_conversation_by_id(self, _identifier: object) -> object:
            raise RuntimeError(_PRIVATE_SENTINEL)

        def get_character_card_by_id(self, _identifier: object) -> object:
            raise RuntimeError(_PRIVATE_SENTINEL)

    class ExplodingMediaDB:
        def get_media_by_id(self, _identifier: object) -> object:
            raise RuntimeError(_PRIVATE_SENTINEL)

    class ExplodingSearch:
        def __init__(self, _media_db: object) -> None:
            pass

        async def keyword_search(
            self, query: str, limit: int = 10, media_types=None
        ) -> list[dict[str, str]]:
            search_calls.append((query, limit, media_types))
            raise RuntimeError(_PRIVATE_SENTINEL)

    monkeypatch.setattr(search_service, "SimplifiedRAGSearchService", ExplodingSearch)
    prompts = MCPPrompts(  # type: ignore[arg-type]
        ExplodingConversationDB(), ExplodingMediaDB()
    )
    records: list[tuple[str, object]] = []
    sink_id = logger.add(
        lambda message: records.append((str(message), message.record["exception"]))
    )
    try:
        result = asyncio.run(getattr(prompts, method_name)(**arguments))
    finally:
        logger.remove(sink_id)

    captured = capsys.readouterr()
    assert result == _PUBLIC_PROMPT_FAILURE
    assert records
    if method_name == "search_and_synthesize_prompt":
        assert search_calls == [("private query", 5, None)]
    else:
        assert search_calls == []
    assert all(exception is None for _message, exception in records)
    assert all(
        _PRIVATE_SENTINEL not in value
        for value in [
            json.dumps(result),
            *(message for message, _exception in records),
            captured.out,
            captured.err,
        ]
    )
    assert all("Traceback" not in message for message, _exception in records)


def test_real_keyword_search_failure_stays_private_through_prompt_gateway(
    tmp_path, monkeypatch, capsys
) -> None:
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.MCP.prompts import MCPPrompts
    from tldw_chatbook.MCP.server import TldwMCPServer
    from tldw_chatbook.RAG_Search.simplified import search_service

    private_fragments = ("SENTINEL", "/private/path", "API_KEY", "secret")
    private_sentinel = "SENTINEL /private/path API_KEY=secret"
    calls: list[tuple[str, object, int]] = []
    media_db = MediaDatabase(tmp_path / "prompt-search.sqlite", client_id="test")

    def fail_search(
        *, search_query: str, media_types: object, results_per_page: int
    ) -> tuple[list[dict[str, object]], int]:
        calls.append((search_query, media_types, results_per_page))
        raise RuntimeError(private_sentinel)

    monkeypatch.setattr(media_db, "search_media_db", fail_search)
    monkeypatch.setattr(search_service, "create_rag_service", lambda **_kwargs: None)

    runtime = _runtime()
    server = TldwMCPServer.__new__(TldwMCPServer)
    server.mcp = runtime
    server.prompts = MCPPrompts(object(), media_db)  # type: ignore[arg-type]
    server._register_prompts()
    runtime.finalize()

    records: list[tuple[str, object]] = []
    sink_id = logger.add(
        lambda message: records.append((str(message), message.record["exception"]))
    )
    try:
        result = asyncio.run(
            runtime.get_prompt(
                "search_and_synthesize",
                {"query": "private query", "num_sources": 5},
                _context(),
            )
        )
    finally:
        logger.remove(sink_id)
        media_db.close_connection()

    captured = capsys.readouterr()
    assert calls == [("private query", None, 5)]
    assert result == {
        "messages": [
            {
                "role": "user",
                "content": {"type": "text", "text": "Unable to create prompt."},
            }
        ]
    }
    assert records
    assert all(exception is None for _message, exception in records)
    public_values = [
        json.dumps(result),
        *(message for message, _exception in records),
        captured.out,
        captured.err,
    ]
    assert all(
        fragment not in value
        for fragment in private_fragments
        for value in public_values
    )
