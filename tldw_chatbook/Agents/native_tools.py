# tldw_chatbook/Agents/native_tools.py
"""Native provider tool-calls: capability check, conversion, parsing.

The fence-first text protocol (``agent_runtime.render_tool_protocol``)
remains the fallback for every provider not listed here — see the
vertical-slice spec and the task-231 tool-call flow review (opportunity 1).

A provider earns a place in ``NATIVE_TOOLS_PROVIDERS`` only when ALL of:

1. ``PROVIDER_PARAM_MAP`` forwards ``tools`` (and the handler accepts it),
2. the handler returns (or normalizes to) the OpenAI-compatible response
   dict with ``choices[0].message.tool_calls`` intact — raw passthrough for
   the OpenAI-compatible providers; full conversion for anthropic
   (task-263), google (task-266), and cohere (task-267, via the v2 /chat
   migration), and
3. the provider accepts OpenAI-shape ``role: "tool"`` history messages.

Pure module: no I/O, no provider imports.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_chatbook.Chat.local_reasoning import ReasoningReplayPolicy

from .agent_models import ToolCall, ToolSchema

#: TASK-33621.1: JSON-Schema keywords a provider refuses at the TOP level of a
#: function's parameters, before the model ever runs. OpenAI: "schema must
#: have type 'object' and not have 'oneOf'/'anyOf'/'allOf'/'enum'/'const'/
#: 'not' at the top level"; Anthropic: "input_schema does not support oneOf,
#: allOf, or anyOf at the top level". One such schema fails the WHOLE request,
#: whatever the prompt or model. Nested use inside ``properties`` is accepted
#: by both and is never touched.
PROVIDER_FORBIDDEN_TOP_LEVEL_SCHEMA_KEYS = (
    "anyOf",
    "oneOf",
    "allOf",
    "enum",
    "const",
    "not",
)

NATIVE_TOOLS_PROVIDERS = frozenset(
    {
        "openai",
        "groq",
        "openrouter",
        "mistral",
        "deepseek",
        "moonshot",
        "zai",
        # TASK-3771: both QwenCloud wire modes normalize function calls to
        # OpenAI shape and translate canonical assistant/tool continuation.
        # Joined Console tests exercise the real dispatcher and HTTP boundary.
        "qwencloud",
        # ADR-179 Phase 2 Task 6: the whole custom execution family (the
        # legacy slots and the swapped engine key) forwards OpenAI tools
        # and returns the raw OpenAI-compatible response shape. Kept as
        # literals (not the shared custom-keys constant) because this
        # module is deliberately pure -- no provider imports; membership
        # is guarded by the registry parity test
        # (NATIVE_TOOLS_PROVIDERS == provider_registry.NATIVE_TOOLS_KEYS).
        "custom-openai-api",
        "custom-openai-api-2",
        "custom-hosted",
        # task-263: chat_with_anthropic converts OpenAI tools/tool-history to
        # Anthropic blocks and normalizes tool_use (non-streaming + streaming)
        # back to OpenAI shape — live-gated against the real API 2026-07-17
        # (Docs/superpowers/qa/anthropic-native-2026-07/).
        "anthropic",
        # task-266: chat_with_google wraps functionDeclarations, converts
        # functionCall/functionResponse history (incl. Gemini 3 thought-
        # signature round-trip), and emits streamed functionCall parts as
        # OpenAI fragments — live-gated 2026-07-17
        # (Docs/superpowers/qa/google-native-2026-07/).
        "google",
        # task-267: chat_with_cohere migrated to the v2 /chat API (OpenAI-
        # shaped messages/tools end-to-end), parses message.tool_calls on both
        # response paths, and round-trips tool_plan as the cohere_tool_plan
        # extra — live-gated 2026-07-17
        # (Docs/superpowers/qa/cohere-native-2026-07/).
        "cohere",
        # ADR-179 Task 12: the hosted engine preset forwards ``tools`` and
        # returns the raw OpenAI-compatible response shape (Databricks AI
        # Gateway external models); confirmed (or honestly disabled) at the
        # Task 14 live gate.
        "databricks",
        # ADR-179 Phase 2 Task 5: the inference-cloud engine presets
        # (together/fireworks/cerebras) advertise OpenAI-shaped tools and
        # tool-history through the same strict engine closure; confirmed (or
        # honestly disabled) at the Task 7 live probes.
        "together",
        "fireworks",
        "cerebras",
        # TASK-33201: the doc-derived presets; each provider documents
        # OpenAI-shaped tools/tool_choice on Chat Completions.
        "sambanova",
        "nvidia",
        "deepinfra",
        "nebius",
        "novita",
        "minimax",
        # TASK-33350: StepFun ships tools OFF (finish "stop" alongside tool_calls).
        "mimo",
        "tokenhub",
        "byteplus",
        # TASK-33351 gateway/host presets (Nous ships tools off: schema unread).
        "vercel",
        "zenmux",
        "kilo",
        "siliconflow",
        "baseten",
        "gmi",
        "ollama_cloud",
        "upstage",
        "arcee",
        "qianfan",
        "venice",
        "meta",
        # TASK-33505..33509 follow-up presets.
        "azure",
        "wandb",
        "cloudflare",
        "opencode_zen",
        "commandcode",
    }
)


def provider_supports_native_tools(
    api_endpoint: str | None, *, reasoning_replay: ReasoningReplayPolicy | None = None
) -> bool:
    """Return whether ``api_endpoint`` supports native tool-calls end-to-end.

    Args:
        api_endpoint: The ``chat_api_call`` provider key. The Console passes
            ``ConsoleProviderResolution.execution_key`` — the key
            ``PROVIDER_PARAM_MAP`` is indexed by.
        reasoning_replay: Frozen local endpoint facts; native support remains
            independent of the optional reasoning replay mode.

    Returns:
        True when the provider forwards ``tools=`` AND returns the raw
        OpenAI-compatible response shape (see module docstring).
    """
    provider = str(api_endpoint or "").strip().lower()
    if provider in {
        "llama_cpp",
        "local_llamacpp",
        "vllm",
        "local_vllm",
        "ollama",
        "local_ollama",
    }:
        return bool(reasoning_replay is not None and reasoning_replay.native_tools)
    return provider in NATIVE_TOOLS_PROVIDERS


def provider_conformant_parameters(parameters: object) -> object:
    """Return a function parameter schema every native provider accepts.

    A conformant schema is returned as the same object. Otherwise a shallow
    copy is returned with the top-level keywords in
    ``PROVIDER_FORBIDDEN_TOP_LEVEL_SCHEMA_KEYS`` removed and ``type`` set to
    ``"object"``; ``properties`` (nested combinators included), ``required``
    and ``additionalProperties`` are kept verbatim, and the input is never
    mutated. This is the single projection seam for EVERY tool source, so a
    third-party MCP server's schema cannot fail every send (TASK-33621.1).
    A removed rule is not lost for enforcement: the tool's own handler (or
    MCP server) still validates the call it receives.

    Args:
        parameters: The disclosed ``ToolSchema.parameters`` value.

    Returns:
        The parameters to send; an empty or non-mapping value becomes the
        minimal valid object schema (providers reject ``{}``).
    """
    if not isinstance(parameters, Mapping) or not parameters:
        # A fresh literal per call: a shared module-level default would leak
        # downstream mutations across conversions through its nested
        # "properties" dict (PR #648 review).
        return {"type": "object", "properties": {}}
    if parameters.get("type") == "object" and not any(
        key in parameters for key in PROVIDER_FORBIDDEN_TOP_LEVEL_SCHEMA_KEYS
    ):
        return parameters
    conformant = {
        key: value
        for key, value in parameters.items()
        if key not in PROVIDER_FORBIDDEN_TOP_LEVEL_SCHEMA_KEYS
    }
    conformant["type"] = "object"
    return conformant


def schemas_to_openai_tools(schemas: list[ToolSchema]) -> list[dict]:
    """Convert ``ToolSchema`` entries to the OpenAI ``tools=`` wire format.

    Args:
        schemas: Disclosed tool schemas (runtime + active), in order.

    Returns:
        One ``{"type": "function", "function": {...}}`` entry per schema,
        its ``parameters`` made provider-conformant by
        ``provider_conformant_parameters``.
    """
    tools = []
    for schema in schemas:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": schema.name,
                    "description": schema.description,
                    "parameters": provider_conformant_parameters(schema.parameters),
                },
            }
        )
    return tools


#: A provider 400 that names a tool by NAME (OpenAI: "Invalid schema for
#: function 'todo_update': ..."). Only a name that exactly matches a tool the
#: request actually sent is ever reported, so provider text never reaches copy.
_REJECTED_TOOL_NAME = re.compile(
    r"\b(?:function|tool)\s+['\"`]([^'\"`\s]{1,256})['\"`]"
)
#: ...or by its POSITION in the request's tools list, most specific first:
#: Gemini "tools[0].function_declarations[3]", OpenAI "tools[34].function",
#: Anthropic "tools.34.custom.input_schema".
_REJECTED_TOOL_INDEX = (
    re.compile(r"function_declarations\[(\d{1,4})\]"),
    re.compile(r"\btools\[(\d{1,4})\]"),
    re.compile(r"\btools\.(\d{1,4})\."),
)


def rejected_tool_name(
    provider_message: str, tools: Sequence[Mapping[str, object]] | None
) -> str | None:
    """Name the sent tool a provider's bad-request message blames, if any.

    Args:
        provider_message: The provider's error text (untrusted).
        tools: The OpenAI-shape ``tools`` list the request actually sent.

    Returns:
        The rejected tool's name exactly as sent, or None when the message
        names no tool from ``tools`` (a position is honoured only when every
        entry is a named function tool, so it maps to the same index the
        provider adapter sent).
    """
    names: list[str] = []
    for tool in tools or ():
        function = tool.get("function") if isinstance(tool, Mapping) else None
        name = function.get("name") if isinstance(function, Mapping) else None
        if not isinstance(name, str) or not name:
            names = []
            break
        names.append(name)
    if not names:
        return None
    text = str(provider_message or "")
    for match in _REJECTED_TOOL_NAME.finditer(text):
        if match.group(1) in names:
            return match.group(1)
    for pattern in _REJECTED_TOOL_INDEX:
        match = pattern.search(text)
        if match is not None:
            index = int(match.group(1))
            return names[index] if index < len(names) else None
    return None


#: Provider phrasing that pins a bad request on a tool DEFINITION even when no
#: sent tool can be named (OpenAI's error code, Anthropic's schema field, a
#: positional tools path).
_TOOL_DEFINITION_MARKERS = re.compile(
    r"invalid_function_parameters|input_schema|function_declarations"
    r"|\btools(?:\[\d{1,4}\]|\.\d{1,4}\.)"
)


def blames_tool_definition(provider_message: str) -> bool:
    """Return whether a provider bad-request message blames a tool definition.

    Args:
        provider_message: The provider's error text (untrusted; only tested).

    Returns:
        True when the text carries a tool-definition marker.
    """
    return bool(_TOOL_DEFINITION_MARKERS.search(str(provider_message or "")))


def ensure_tool_call_ids(raw_calls: list | None) -> list:
    """Return tool-call entries with every dict entry carrying an id.

    Some OpenAI-compatible servers omit tool-call ids. An id-less call would
    split the history convention — the assistant echo carries the id-less
    entry while its result falls back to a fence-style user-role line —
    which strict providers reject on the next request (PR #648 review).
    Missing ids get a synthesized ``call_<position>`` so the echo and its
    ``role="tool"`` reply always pair; the caller must use the SAME
    normalized list for both the echo and parsing.

    Args:
        raw_calls: The raw ``message.tool_calls`` list (or None).

    Returns:
        A new list where every dict entry has a non-empty ``id``; entries
        with ids and non-dict junk pass through untouched.
    """
    normalized = []
    for position, raw in enumerate(raw_calls or []):
        if isinstance(raw, dict) and not raw.get("id"):
            raw = {**raw, "id": f"call_{position}"}
        normalized.append(raw)
    return normalized


def parse_native_tool_calls(message: dict | None) -> tuple[ToolCall, ...]:
    """Parse OpenAI-shape ``message.tool_calls`` into ``ToolCall`` entries.

    Malformed ``arguments`` JSON yields ``args={}`` rather than dropping
    the call: the downstream tool's own validation error is echoed back to
    the model as a normal tool result, so it can retry with corrected
    arguments. Entries without a function name are dropped.

    Args:
        message: The ``choices[0].message`` dict from a provider response
            (or anything — junk yields no calls).

    Returns:
        Parsed calls in provider order, each carrying its ``call_id``.
    """
    if not isinstance(message, dict):
        return ()
    calls = []
    for raw in message.get("tool_calls") or []:
        if not isinstance(raw, dict):
            continue
        function = raw.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        raw_args = function.get("arguments")
        args: dict = {}
        if isinstance(raw_args, dict):
            args = raw_args
        elif isinstance(raw_args, str) and raw_args.strip():
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                args = parsed
        calls.append(
            ToolCall(
                name=name,
                args=args,
                call_id=str(raw.get("id") or ""),
                raw_arguments=raw_args if isinstance(raw_args, str) else "",
            )
        )
    return tuple(calls)
