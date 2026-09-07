"""Small adapter from Chatbook's decorators to the strict MCP gateway."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import inspect
import json
import math
import re
import sys
import types
from collections.abc import Awaitable, Callable, Iterable
from binascii import Error as Base64Error
from typing import (
    TYPE_CHECKING,
    Any,
    NoReturn,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit

from jsonschema import Draft7Validator, Draft202012Validator
from jsonschema.exceptions import SchemaError
from mcp_unified.gateway import (
    GatewayApplicationError,
    GatewayJSONValue,
    GatewayLimits,
    GatewayRequestContext,
    GatewayToolExecutionError,
)

from tldw_chatbook.Agents.agent_models import ToolResult
from tldw_chatbook.Agents.local_tool_provider import (
    LOCAL_DENY_REFUSAL,
    LOCAL_GATE_ERROR_REFUSAL,
    LOCAL_KILL_SWITCH_REFUSAL,
    LOCAL_TIMEOUT_REFUSAL,
)
from tldw_chatbook.MCP.local_server_tools import EXTERNAL_NO_CALLBACK_REFUSAL

if TYPE_CHECKING:
    from tldw_chatbook.MCP.local_server_tools import LocalToolRegistration


_TOOL_NAME = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")
_GATEWAY_LIMITS = GatewayLimits()
_ToolHandler = Callable[..., Awaitable[GatewayJSONValue]]
_LocalToolHandler = Callable[[dict[str, Any]], ToolResult]
_ResourceHandler = Callable[..., Awaitable[dict[str, Any]]]
_ResourceListHandler = Callable[[], Awaitable[list[dict[str, Any]]]]
_PromptHandler = Callable[..., Awaitable[list[dict[str, str]]]]

MAX_RESOURCE_CHUNK_BYTES = 256 * 1024
CONTINUATION_QUERY_KEY = "tldw_continue"

_CONTINUATION_VERSION = 1
_MAX_CONTINUATION_TOKEN_CHARS = 512
_MAX_RESOURCE_URI_CHARS = 2_048
_BAD_PERCENT_ENCODING = re.compile(r"%(?![0-9A-Fa-f]{2})")
_CONTINUATION_TOKEN = re.compile(r"[A-Za-z0-9_-]{1,512}\Z")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
_URI_UNRESERVED = "-._~"
_RESOURCE_TEMPLATE_VARIABLES = {
    "conversation://{conversation_id}": "conversation_id",
    "note://{note_id}": "note_id",
    "character://{character_id}": "character_id",
    "media://{media_id}": "media_id",
    "rag-chunk://{chunk_uuid}": "chunk_uuid",
}
_RESOURCE_TEMPLATE_MATCHERS = {
    template: re.compile(
        rf"{re.escape(template.split(':', 1)[0])}://(?P<{variable}>[^/?#]+)\Z"
    )
    for template, variable in _RESOURCE_TEMPLATE_VARIABLES.items()
}
_PROMPT_NAMES = (
    "summarize_conversation",
    "generate_document",
    "analyze_media",
    "search_and_synthesize",
    "character_writing",
)
_PROMPT_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
_MAX_PROMPT_INTEGER_CHARS = 128
_MAX_PROMPT_INTEGER_VALUE = 10**_MAX_PROMPT_INTEGER_CHARS - 1
_MIN_PROMPT_INTEGER_VALUE = -(10 ** (_MAX_PROMPT_INTEGER_CHARS - 1) - 1)


def _encode_continuation_state(
    *, offset: int, base_digest: str, content_digest: str
) -> str:
    payload = json.dumps(
        {
            "b": base_digest,
            "c": content_digest,
            "o": offset,
            "v": _CONTINUATION_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


_MAX_EMITTED_CONTINUATION_TOKEN_CHARS = len(
    _encode_continuation_state(
        offset=sys.maxsize,
        base_digest="0" * 64,
        content_digest="0" * 64,
    )
)
_CONTINUATION_QUERY_PREFIX = f"?{CONTINUATION_QUERY_KEY}="
_MAX_RESOURCE_BASE_URI_CHARS = (
    _MAX_RESOURCE_URI_CHARS
    - len(_CONTINUATION_QUERY_PREFIX)
    - _MAX_EMITTED_CONTINUATION_TOKEN_CHARS
)

_OPERATOR_APPROVAL_ERROR = (
    "operator_approval_required",
    "Operator approval is required for this local tool.",
)
_LOCAL_FAILURES = {
    EXTERNAL_NO_CALLBACK_REFUSAL: _OPERATOR_APPROVAL_ERROR,
    LOCAL_TIMEOUT_REFUSAL: _OPERATOR_APPROVAL_ERROR,
    LOCAL_DENY_REFUSAL: (
        "tool_permission_denied",
        "This local tool is disabled by operator policy.",
    ),
    LOCAL_KILL_SWITCH_REFUSAL: (
        "local_tools_disabled",
        "Local tools are disabled.",
    ),
    LOCAL_GATE_ERROR_REFUSAL: (
        "permission_state_unavailable",
        "Local tool permission state is unavailable.",
    ),
}
_GENERIC_LOCAL_FAILURE = ("local_tool_failed", "Local tool execution failed.")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate cursor keys instead of silently taking the last value."""
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("duplicate JSON key")
    return value


def _is_finite_json_structure(value: object, *, max_depth: int) -> bool:
    """Match the gateway's finite JSON container and depth rules."""
    active: set[int] = set()
    stack: list[tuple[object, int, bool]] = [(value, 1, False)]
    while stack:
        current, depth, leaving = stack.pop()
        if leaving:
            active.remove(id(current))
            continue
        if isinstance(current, (dict, list)):
            if depth > max_depth or id(current) in active:
                return False
            active.add(id(current))
            stack.append((current, depth, True))
            children: Iterable[object]
            if isinstance(current, dict):
                if any(not isinstance(key, str) for key in current):
                    return False
                children = current.values()
            else:
                children = current
            stack.extend((child, depth + 1, False) for child in children)
            continue
        if current is None or isinstance(current, (bool, int, str)):
            continue
        if isinstance(current, float) and math.isfinite(current):
            continue
        return False
    return True


class ChatbookGatewayRuntime:
    """Register and dispatch Chatbook's standalone built-in tools.

    Args:
        name: Bounded server identity advertised by the gateway.
        version: Bounded Chatbook runtime version.
        tool_descriptors: Canonical built-in tool descriptors to bind before
            finalization.
    """

    def __init__(
        self,
        *,
        name: str,
        version: str,
        tool_descriptors: list[dict[str, Any]],
    ) -> None:
        self.name = self._bounded_identity(name)
        self.version = self._bounded_identity(version)
        if not isinstance(tool_descriptors, list):
            raise ValueError("tool_descriptors must be a list")

        self._tool_descriptors: dict[str, dict[str, Any]] = {}
        for descriptor in tool_descriptors:
            descriptor_copy = self._validated_descriptor(descriptor)
            descriptor_name = descriptor_copy["name"]
            if descriptor_name in self._tool_descriptors:
                raise ValueError(f"duplicate tool descriptor: {descriptor_name}")
            self._tool_descriptors[descriptor_name] = descriptor_copy

        self._tool_handlers: dict[str, _ToolHandler] = {}
        self._local_tool_handlers: dict[str, _LocalToolHandler] = {}
        self._resource_handlers: dict[
            str, tuple[str, re.Pattern[str], _ResourceHandler]
        ] = {}
        self._resource_template_descriptors: dict[str, dict[str, str]] = {}
        self._resource_list_handler: _ResourceListHandler | None = None
        self._prompt_handlers: dict[str, _PromptHandler] = {}
        self._prompt_descriptors: dict[str, dict[str, Any]] = {}
        self._prompt_parameter_kinds: dict[str, dict[str, tuple[str, bool]]] = {}
        self._finalized = False

    @staticmethod
    def _bounded_identity(value: object) -> str:
        if not isinstance(value, str) or not value or len(value) > 512:
            raise ValueError("runtime name and version must be bounded strings")
        return value

    @staticmethod
    def _validated_descriptor(descriptor: object) -> dict[str, Any]:
        if not isinstance(descriptor, dict):
            raise ValueError("tool descriptor must be a dictionary")
        name = descriptor.get("name")
        if not isinstance(name, str) or _TOOL_NAME.fullmatch(name) is None:
            raise ValueError("tool descriptor name is invalid")
        description = descriptor.get("description")
        if (
            not isinstance(description, str)
            or not description
            or len(description) > 4_096
        ):
            raise ValueError("tool descriptor description must be a bounded string")
        schema = descriptor.get("inputSchema")
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ValueError("tool inputSchema must have type object")
        if schema.get("additionalProperties") is not False:
            raise ValueError("tool inputSchema additionalProperties must be false")
        if not isinstance(schema.get("properties"), dict) or not isinstance(
            schema.get("required"), list
        ):
            raise ValueError("tool inputSchema properties and required are invalid")
        return copy.deepcopy(descriptor)

    def tool(
        self, *, name: str | None = None
    ) -> Callable[[_ToolHandler], _ToolHandler]:
        """Return a decorator that records one async built-in handler.

        Args:
            name: Optional descriptor name override. The handler's function
                name is used when omitted.

        Returns:
            A decorator that validates and records an async tool handler.
        """
        if self._finalized:
            raise RuntimeError("runtime is finalized")

        def decorator(handler: _ToolHandler) -> _ToolHandler:
            if self._finalized:
                raise RuntimeError("runtime is finalized")
            handler_name = getattr(handler, "__name__", None) if name is None else name
            if (
                not isinstance(handler_name, str)
                or _TOOL_NAME.fullmatch(handler_name) is None
            ):
                raise ValueError("tool handler name is invalid")
            if not inspect.iscoroutinefunction(handler):
                raise ValueError("tool handler must be async")
            if handler_name in self._tool_handlers:
                raise ValueError(f"duplicate tool handler: {handler_name}")
            self._tool_handlers[handler_name] = handler
            return handler

        return decorator

    def register_local_tools(
        self, registrations: Iterable[LocalToolRegistration]
    ) -> None:
        """Validate and publish one complete local-tool registration set.

        Args:
            registrations: Local provider registrations to validate and add
                atomically to the standalone catalog.
        """
        if self._finalized:
            raise RuntimeError("runtime is finalized")

        descriptors: dict[str, dict[str, Any]] = {}
        handlers: dict[str, _LocalToolHandler] = {}
        for registration in registrations:
            try:
                name = registration.name
                description = registration.description
                parameters = registration.parameters
                handler = registration.handler
            except Exception:
                raise ValueError("local tool registration is invalid") from None

            if not isinstance(name, str) or _TOOL_NAME.fullmatch(name) is None:
                raise ValueError("local tool name is invalid")
            if name in self._tool_descriptors or name in descriptors:
                raise ValueError("local tool name collides with another tool")
            if (
                not isinstance(description, str)
                or not description
                or len(description) > 4_096
            ):
                raise ValueError("local tool description must be a bounded string")
            if not isinstance(parameters, dict) or parameters.get("type") != "object":
                raise ValueError("local tool parameters must have type object")
            if not _is_finite_json_structure(
                parameters, max_depth=_GATEWAY_LIMITS.max_schema_depth
            ):
                raise ValueError("local tool parameters must be valid JSON Schema")
            try:
                serialized_parameters = json.dumps(
                    parameters,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            except (RecursionError, TypeError, ValueError, UnicodeEncodeError):
                serialized_parameters = None
            if (
                serialized_parameters is None
                or len(serialized_parameters) > _GATEWAY_LIMITS.max_schema_bytes
            ):
                raise ValueError("local tool parameters must be valid JSON Schema")
            if "$schema" in parameters:
                raise ValueError(
                    "local tool parameters must not declare a schema dialect"
                )
            parameters_are_valid = True
            for validator in (Draft7Validator, Draft202012Validator):
                try:
                    validator.check_schema(parameters)
                except SchemaError:
                    parameters_are_valid = False
                    break
            if not parameters_are_valid:
                raise ValueError("local tool parameters must be valid JSON Schema")
            if not callable(handler):
                raise ValueError("local tool handler must be callable")

            descriptors[name] = {
                "name": name,
                "description": description,
                "inputSchema": copy.deepcopy(parameters),
            }
            handlers[name] = handler

        self._tool_descriptors.update(descriptors)
        self._local_tool_handlers.update(handlers)

    def resource(self, template: str) -> Callable[[_ResourceHandler], _ResourceHandler]:
        """Return a decorator for one supported resource template.

        Args:
            template: Canonical Chatbook resource URI template.

        Returns:
            A decorator that validates and records an async resource handler.
        """
        if self._finalized:
            raise RuntimeError("runtime is finalized")
        variable = _RESOURCE_TEMPLATE_VARIABLES.get(template)
        if variable is None:
            raise ValueError("resource template is not supported")
        if template in self._resource_handlers:
            raise ValueError("duplicate resource template")
        matcher = _RESOURCE_TEMPLATE_MATCHERS[template]

        def decorator(handler: _ResourceHandler) -> _ResourceHandler:
            if self._finalized:
                raise RuntimeError("runtime is finalized")
            if template in self._resource_handlers:
                raise ValueError("duplicate resource template")
            if not inspect.iscoroutinefunction(handler):
                raise ValueError("resource handler must be async")
            parameters = inspect.signature(handler).parameters
            if list(parameters) != [variable] or parameters[variable].kind not in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }:
                raise ValueError("resource identifier and handler parameter must match")
            name = getattr(handler, "__name__", None)
            if not isinstance(name, str) or _TOOL_NAME.fullmatch(name) is None:
                raise ValueError("resource handler name is invalid")
            descriptor = {"uriTemplate": template, "name": name}
            doc = inspect.getdoc(handler)
            if doc:
                descriptor["description"] = doc.splitlines()[0].strip()
            self._resource_handlers[template] = (variable, matcher, handler)
            self._resource_template_descriptors[template] = descriptor
            return handler

        return decorator

    @overload
    def list_resources(
        self, context: None = None
    ) -> Callable[[_ResourceListHandler], _ResourceListHandler]: ...

    @overload
    def list_resources(
        self, context: GatewayRequestContext
    ) -> Awaitable[list[dict[str, Any]]]: ...

    def list_resources(
        self, context: GatewayRequestContext | None = None
    ) -> (
        Callable[[_ResourceListHandler], _ResourceListHandler]
        | Awaitable[list[dict[str, Any]]]
    ):
        """Register the dynamic catalog or return its detached current value.

        Args:
            context: Gateway request context when serving a catalog request;
                omit it when using this method as a registration decorator.

        Returns:
            A registration decorator when ``context`` is omitted, otherwise
            an awaitable that resolves to detached resource descriptors.
        """
        if context is not None:
            return self._list_resources(context)
        if self._finalized:
            raise RuntimeError("runtime is finalized")

        def decorator(handler: _ResourceListHandler) -> _ResourceListHandler:
            if self._finalized:
                raise RuntimeError("runtime is finalized")
            if self._resource_list_handler is not None:
                raise ValueError("duplicate resource list handler")
            if not inspect.iscoroutinefunction(handler):
                raise ValueError("resource list handler must be async")
            if inspect.signature(handler).parameters:
                raise ValueError("resource list handler must not accept arguments")
            self._resource_list_handler = handler
            return handler

        return decorator

    def prompt(
        self, *, name: str | None = None
    ) -> Callable[[_PromptHandler], _PromptHandler]:
        """Return a decorator that records one typed async prompt handler.

        Args:
            name: Optional prompt name override. The handler's function name
                is used when omitted.

        Returns:
            A decorator that validates and records an async prompt handler.
        """
        if self._finalized:
            raise RuntimeError("runtime is finalized")

        def decorator(handler: _PromptHandler) -> _PromptHandler:
            if self._finalized:
                raise RuntimeError("runtime is finalized")
            handler_name = getattr(handler, "__name__", None) if name is None else name
            if (
                not isinstance(handler_name, str)
                or _TOOL_NAME.fullmatch(handler_name) is None
            ):
                raise ValueError("prompt handler name is invalid")
            if handler_name in self._prompt_handlers:
                raise ValueError(f"duplicate prompt handler: {handler_name}")
            if not inspect.iscoroutinefunction(handler):
                raise ValueError("prompt handler must be async")

            descriptor, parameter_kinds = self._describe_prompt(handler_name, handler)
            self._prompt_handlers[handler_name] = handler
            self._prompt_descriptors[handler_name] = descriptor
            self._prompt_parameter_kinds[handler_name] = parameter_kinds
            return handler

        return decorator

    @classmethod
    def _describe_prompt(
        cls, name: str, handler: _PromptHandler
    ) -> tuple[dict[str, Any], dict[str, tuple[str, bool]]]:
        try:
            type_hints = get_type_hints(handler)
        except Exception:
            raise ValueError("prompt parameter type is invalid") from None
        arguments: list[dict[str, Any]] = []
        parameter_kinds: dict[str, tuple[str, bool]] = {}
        for parameter in inspect.signature(handler).parameters.values():
            if _TOOL_NAME.fullmatch(parameter.name) is None:
                raise ValueError("prompt argument name is invalid")
            if parameter.kind not in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }:
                raise ValueError("prompt parameter kind is invalid")
            kind = cls._prompt_parameter_kind(type_hints.get(parameter.name))
            required = parameter.default is inspect.Parameter.empty
            if not required and not cls._valid_prompt_default(parameter.default, kind):
                raise ValueError("prompt parameter default is invalid")
            arguments.append({"name": parameter.name, "required": required})
            parameter_kinds[parameter.name] = (kind, required)

        descriptor: dict[str, Any] = {"name": name}
        doc = inspect.getdoc(handler)
        if doc:
            description = doc.splitlines()[0].strip()
            if not description or len(description) > 4_096:
                raise ValueError("prompt description is invalid")
            descriptor["description"] = description
        descriptor["arguments"] = arguments
        return descriptor, parameter_kinds

    @staticmethod
    def _prompt_parameter_kind(annotation: object) -> str:
        if annotation is str:
            return "str"
        if annotation is int:
            return "int"
        if get_origin(annotation) in {Union, types.UnionType} and set(
            get_args(annotation)
        ) == {str, type(None)}:
            return "optional_str"
        raise ValueError("prompt parameter type is unsupported")

    @staticmethod
    def _valid_prompt_default(value: object, kind: str) -> bool:
        if kind == "str":
            return isinstance(value, str)
        if kind == "int":
            return isinstance(value, int) and not isinstance(value, bool)
        return value is None or isinstance(value, str)

    async def list_prompts(
        self, context: GatewayRequestContext
    ) -> list[dict[str, Any]]:
        """Return detached prompt descriptors in registration order.

        Args:
            context: Gateway request context supplied by ``mcp-unified``.

        Returns:
            Defensive copies of the registered prompt descriptors.
        """
        self._require_finalized()
        return copy.deepcopy(list(self._prompt_descriptors.values()))

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        """Validate arguments, invoke one prompt, and map its messages.

        Args:
            name: Registered prompt name.
            arguments: JSON prompt arguments received from the gateway.
            context: Gateway request context supplied by ``mcp-unified``.

        Returns:
            A canonical MCP prompt result containing ordered text messages.
        """
        self._require_finalized()
        handler = self._prompt_handlers.get(name)
        if handler is None:
            raise GatewayApplicationError(
                "Prompt not found.",
                reason_code="prompt_not_found",
                kind="prompt",
            ) from None
        normalized = self._normalize_prompt_arguments(
            arguments, self._prompt_parameter_kinds[name]
        )
        raw_result: object = None
        handler_failed = False
        try:
            raw_result = await handler(**normalized)
        except asyncio.CancelledError:
            raise
        except Exception:
            handler_failed = True
        if handler_failed:
            self._raise_invalid_prompt_result()
        return self._canonical_prompt_result(raw_result)

    @classmethod
    def _normalize_prompt_arguments(
        cls,
        arguments: object,
        parameter_kinds: dict[str, tuple[str, bool]],
    ) -> dict[str, object]:
        if not isinstance(arguments, dict) or any(
            not isinstance(key, str) for key in arguments
        ):
            cls._raise_invalid_prompt_arguments()
        if not set(arguments) <= set(parameter_kinds):
            cls._raise_invalid_prompt_arguments()
        if any(
            required and name not in arguments
            for name, (_kind, required) in parameter_kinds.items()
        ):
            cls._raise_invalid_prompt_arguments()

        normalized: dict[str, object] = {}
        for name, value in arguments.items():
            kind = parameter_kinds[name][0]
            if kind == "str":
                if not isinstance(value, str):
                    cls._raise_invalid_prompt_arguments()
                normalized[name] = value
            elif kind == "optional_str":
                if value is not None and not isinstance(value, str):
                    cls._raise_invalid_prompt_arguments()
                normalized[name] = value
            elif isinstance(value, bool):
                cls._raise_invalid_prompt_arguments()
            elif type(value) is int:
                if not _MIN_PROMPT_INTEGER_VALUE <= value <= _MAX_PROMPT_INTEGER_VALUE:
                    cls._raise_invalid_prompt_arguments()
                normalized[name] = value
            elif (
                isinstance(value, str)
                and len(value) <= _MAX_PROMPT_INTEGER_CHARS
                and _PROMPT_INTEGER.fullmatch(value) is not None
            ):
                normalized[name] = int(value)
            else:
                cls._raise_invalid_prompt_arguments()
        return normalized

    @classmethod
    def _canonical_prompt_result(cls, raw_result: object) -> dict[str, Any]:
        if not isinstance(raw_result, list) or not raw_result:
            cls._raise_invalid_prompt_result()

        messages: list[tuple[str, str]] = []
        for raw_message in raw_result:
            if not isinstance(raw_message, dict) or set(raw_message) != {
                "role",
                "content",
            }:
                cls._raise_invalid_prompt_result()
            role = raw_message["role"]
            content = raw_message["content"]
            if role not in {"system", "user", "assistant"} or not isinstance(
                content, str
            ):
                cls._raise_invalid_prompt_result()
            messages.append((role, content))

        leading_system_count = 0
        while (
            leading_system_count < len(messages)
            and messages[leading_system_count][0] == "system"
        ):
            leading_system_count += 1
        if leading_system_count:
            if (
                leading_system_count == len(messages)
                or messages[leading_system_count][0] != "user"
            ):
                cls._raise_invalid_prompt_result()
            leading_system = [
                content for _role, content in messages[:leading_system_count]
            ]
            user_text = messages[leading_system_count][1]
            messages = [
                (
                    "user",
                    "System instructions:\n"
                    + "\n\n".join(leading_system)
                    + "\n\nUser request:\n"
                    + user_text,
                ),
                *messages[leading_system_count + 1 :],
            ]
        if any(role == "system" for role, _content in messages):
            cls._raise_invalid_prompt_result()

        result = {
            "messages": [
                {
                    "role": role,
                    "content": {"type": "text", "text": content},
                }
                for role, content in messages
            ]
        }
        try:
            serialized_result = json.dumps(
                result,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (RecursionError, TypeError, ValueError, UnicodeEncodeError):
            cls._raise_invalid_prompt_result()
        if len(serialized_result) > _GATEWAY_LIMITS.max_result_bytes:
            cls._raise_invalid_prompt_result()
        return result

    @staticmethod
    def _raise_invalid_prompt_arguments() -> NoReturn:
        raise GatewayApplicationError(
            "Invalid prompt arguments.",
            reason_code="invalid_prompt_arguments",
            kind="prompt",
        ) from None

    @staticmethod
    def _raise_invalid_prompt_result() -> NoReturn:
        raise GatewayApplicationError(
            "Prompt handler returned an invalid result.",
            reason_code="invalid_prompt_result",
            kind="prompt",
        ) from None

    async def _list_resources(
        self, context: GatewayRequestContext
    ) -> list[dict[str, Any]]:
        self._require_finalized()
        if self._resource_list_handler is None:
            return []
        resources = await self._resource_list_handler()
        if not isinstance(resources, list):
            self._raise_invalid_resource_result()
        return [self._canonical_resource_descriptor(item) for item in resources]

    async def list_resource_templates(
        self, context: GatewayRequestContext
    ) -> list[dict[str, Any]]:
        """Return detached resource templates in their accepted order.

        Args:
            context: Gateway request context supplied by ``mcp-unified``.

        Returns:
            Defensive copies of the registered resource template descriptors.
        """
        self._require_finalized()
        return copy.deepcopy(
            [
                self._resource_template_descriptors[template]
                for template in _RESOURCE_TEMPLATE_VARIABLES
                if template in self._resource_template_descriptors
            ]
        )

    async def read_resource(
        self,
        uri: str,
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        """Route and map one bounded canonical resource chunk.

        Args:
            uri: Requested canonical or continuation resource URI.
            context: Gateway request context supplied by ``mcp-unified``.

        Returns:
            A bounded canonical resource result with continuation metadata
            when more content remains.
        """
        self._require_finalized()
        base_uri, token = self._parse_resource_uri(uri)
        variable, handler, identifier, canonical_base_uri = self._match_resource(
            base_uri
        )
        if len(canonical_base_uri) > _MAX_RESOURCE_BASE_URI_CHARS:
            self._raise_invalid_resource_uri()
        state = self._decode_continuation(token) if token is not None else None
        base_digest = self._digest(canonical_base_uri)
        if state is not None and state["b"] != base_digest:
            self._raise_invalid_resource_uri()

        raw_result = await handler(**{variable: identifier})
        return await asyncio.to_thread(
            self._project_resource_result,
            raw_result,
            canonical_base_uri,
            base_digest,
            state,
        )

    @classmethod
    def _project_resource_result(
        cls,
        raw_result: object,
        canonical_base_uri: str,
        base_digest: str,
        state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build one result without monopolizing the event loop.

        Continuations intentionally remain stateless: every read rematerializes and
        scans O(total content) in bounded slices so changed content cannot be hidden
        by a stale cache.
        """
        content, mime_type, metadata = cls._canonical_resource_result(
            raw_result, expected_uri=canonical_base_uri
        )
        content_digest = hashlib.sha256()
        total_bytes = 0
        try:
            for position in range(0, len(content), MAX_RESOURCE_CHUNK_BYTES):
                encoded = content[
                    position : min(position + MAX_RESOURCE_CHUNK_BYTES, len(content))
                ].encode("utf-8")
                content_digest.update(encoded)
                total_bytes += len(encoded)
        except UnicodeEncodeError:
            cls._raise_invalid_resource_result()
        content_digest_hex = content_digest.hexdigest()
        start = 0
        if state is not None:
            if state["c"] != content_digest_hex:
                raise GatewayApplicationError(
                    "Resource changed; restart from the base URI.",
                    reason_code="resource_changed",
                    kind="resource",
                ) from None
            start = state["o"]
            if start <= 0 or start >= len(content):
                cls._raise_invalid_resource_uri()

        window_end = min(start + MAX_RESOURCE_CHUNK_BYTES, len(content))
        try:
            raw_window = content[start:window_end].encode("utf-8")
            chunk = raw_window[:MAX_RESOURCE_CHUNK_BYTES].decode(
                "utf-8", errors="ignore"
            )
        except UnicodeEncodeError:
            cls._raise_invalid_resource_result()
        maximum_end = start + len(chunk)

        maximum_result = cls._build_resource_result(
            content=content,
            mime_type=mime_type,
            metadata=metadata,
            base_uri=canonical_base_uri,
            base_digest=base_digest,
            content_digest=content_digest_hex,
            start=start,
            end=maximum_end,
            total_bytes=total_bytes,
        )
        if (
            cls._serialized_result_size(maximum_result)
            <= _GATEWAY_LIMITS.max_result_bytes
        ):
            return maximum_result

        low = start + 1
        high = maximum_end - 1
        best: dict[str, Any] | None = None
        while low <= high:
            end = (low + high) // 2
            candidate = cls._build_resource_result(
                content=content,
                mime_type=mime_type,
                metadata=metadata,
                base_uri=canonical_base_uri,
                base_digest=base_digest,
                content_digest=content_digest_hex,
                start=start,
                end=end,
                total_bytes=total_bytes,
            )
            if (
                cls._serialized_result_size(candidate)
                <= _GATEWAY_LIMITS.max_result_bytes
            ):
                best = candidate
                low = end + 1
            else:
                high = end - 1
        if best is None:
            cls._raise_invalid_resource_result()
        return best

    @classmethod
    def _build_resource_result(
        cls,
        *,
        content: str,
        mime_type: str,
        metadata: dict[str, Any] | None,
        base_uri: str,
        base_digest: str,
        content_digest: str,
        start: int,
        end: int,
        total_bytes: int,
    ) -> dict[str, Any]:
        chunk = content[start:end]
        try:
            returned_bytes = len(chunk.encode("utf-8"))
        except UnicodeEncodeError:
            cls._raise_invalid_resource_result()
        has_more = end < len(content)
        next_uri = None
        if has_more:
            next_token = cls._encode_continuation(
                offset=end,
                base_digest=base_digest,
                content_digest=content_digest,
            )
            next_uri = f"{base_uri}?{urlencode({CONTINUATION_QUERY_KEY: next_token})}"
            if len(next_uri) > _MAX_RESOURCE_URI_CHARS:
                cls._raise_invalid_resource_result()

        result_meta: dict[str, Any] = {
            "tldw.chatbook/continuation": {
                "startChar": start,
                "endChar": end,
                "totalChars": len(content),
                "totalBytes": total_bytes,
                "returnedBytes": returned_bytes,
                "hasMore": has_more,
                "nextUri": next_uri,
            }
        }
        if metadata:
            result_meta["tldw.chatbook/resource"] = metadata
        return {
            "contents": [{"uri": base_uri, "mimeType": mime_type, "text": chunk}],
            "_meta": result_meta,
        }

    @staticmethod
    def _serialized_result_size(result: dict[str, Any]) -> int:
        try:
            return len(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
        except (RecursionError, TypeError, ValueError, UnicodeEncodeError):
            ChatbookGatewayRuntime._raise_invalid_resource_result()

    @staticmethod
    def _canonical_resource_descriptor(descriptor: object) -> dict[str, Any]:
        if not isinstance(descriptor, dict):
            ChatbookGatewayRuntime._raise_invalid_resource_result()
        uri = descriptor.get("uri")
        name = descriptor.get("name")
        mime_type = descriptor.get("mimeType")
        if (
            not isinstance(uri, str)
            or not uri
            or len(uri) > _MAX_RESOURCE_URI_CHARS
            or any(character.isspace() or ord(character) < 32 for character in uri)
            or not isinstance(name, str)
            or not name
            or len(name) > 512
            or not isinstance(mime_type, str)
            or not mime_type
            or len(mime_type) > 255
        ):
            ChatbookGatewayRuntime._raise_invalid_resource_result()
        canonical = {"uri": uri, "name": name, "mimeType": mime_type}
        if "description" in descriptor:
            description = descriptor["description"]
            if not isinstance(description, str) or len(description) > 4_096:
                ChatbookGatewayRuntime._raise_invalid_resource_result()
            canonical["description"] = description
        return canonical

    @staticmethod
    def _canonical_resource_result(
        result: object,
        *,
        expected_uri: str,
    ) -> tuple[str, str, dict[str, Any] | None]:
        if not isinstance(result, dict):
            ChatbookGatewayRuntime._raise_invalid_resource_result()
        result_uri = result.get("uri")
        name = result.get("name")
        mime_type = result.get("mimeType")
        content = result.get("content")
        if (
            not isinstance(result_uri, str)
            or not result_uri
            or len(result_uri) > _MAX_RESOURCE_URI_CHARS
            or not isinstance(name, str)
            or not name
            or len(name) > 512
            or not isinstance(mime_type, str)
            or not mime_type
            or len(mime_type) > 255
            or not isinstance(content, str)
        ):
            ChatbookGatewayRuntime._raise_invalid_resource_result()
        canonical_result_uri = ChatbookGatewayRuntime._canonical_resource_uri(
            result_uri
        )
        if canonical_result_uri != expected_uri:
            ChatbookGatewayRuntime._raise_invalid_resource_result()
        if "metadata" not in result:
            return content, mime_type, None
        metadata = result["metadata"]
        if not isinstance(metadata, dict):
            ChatbookGatewayRuntime._raise_invalid_resource_result()
        if not metadata:
            return content, mime_type, None
        if not _is_finite_json_structure(
            metadata, max_depth=_GATEWAY_LIMITS.max_json_depth - 2
        ):
            ChatbookGatewayRuntime._raise_invalid_resource_result()
        try:
            serialized = json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            normalized = json.loads(serialized)
        except (RecursionError, TypeError, ValueError, UnicodeEncodeError):
            ChatbookGatewayRuntime._raise_invalid_resource_result()
        if not isinstance(normalized, dict) or not normalized:
            ChatbookGatewayRuntime._raise_invalid_resource_result()
        return content, mime_type, normalized

    @staticmethod
    def _parse_resource_uri(uri: object) -> tuple[str, str | None]:
        if (
            not isinstance(uri, str)
            or not uri
            or len(uri) > _MAX_RESOURCE_URI_CHARS
            or any(character.isspace() or ord(character) < 32 for character in uri)
            or _BAD_PERCENT_ENCODING.search(uri) is not None
            or "#" in uri
        ):
            ChatbookGatewayRuntime._raise_invalid_resource_uri()
        try:
            parsed = urlsplit(uri)
        except ValueError:
            ChatbookGatewayRuntime._raise_invalid_resource_uri()
        if parsed.fragment:
            ChatbookGatewayRuntime._raise_invalid_resource_uri()
        token = None
        if parsed.query:
            try:
                pairs = parse_qsl(
                    parsed.query,
                    keep_blank_values=True,
                    strict_parsing=True,
                    max_num_fields=2,
                )
            except ValueError:
                ChatbookGatewayRuntime._raise_invalid_resource_uri()
            if (
                len(pairs) != 1
                or pairs[0][0] != CONTINUATION_QUERY_KEY
                or not pairs[0][1]
            ):
                ChatbookGatewayRuntime._raise_invalid_resource_uri()
            token = pairs[0][1]
        base_uri = urlunsplit(
            (parsed.scheme.lower(), parsed.netloc, parsed.path, "", "")
        )
        return base_uri, token

    def _match_resource(self, base_uri: str) -> tuple[str, _ResourceHandler, str, str]:
        for template in _RESOURCE_TEMPLATE_VARIABLES:
            registration = self._resource_handlers.get(template)
            if registration is None:
                continue
            variable, matcher, handler = registration
            matched = matcher.fullmatch(base_uri)
            if matched is None:
                continue
            try:
                identifier = unquote(matched.group(variable), errors="strict")
            except UnicodeDecodeError:
                self._raise_invalid_resource_uri()
            if not identifier:
                self._raise_invalid_resource_uri()
            try:
                canonical_identifier = quote(identifier, safe=_URI_UNRESERVED)
            except UnicodeEncodeError:
                self._raise_invalid_resource_uri()
            canonical_base_uri = f"{base_uri.split(':', 1)[0]}://{canonical_identifier}"
            return variable, handler, identifier, canonical_base_uri
        self._raise_invalid_resource_uri()

    @staticmethod
    def _canonical_resource_uri(uri: object) -> str:
        if (
            not isinstance(uri, str)
            or not uri
            or len(uri) > _MAX_RESOURCE_URI_CHARS
            or any(character.isspace() or ord(character) < 32 for character in uri)
            or _BAD_PERCENT_ENCODING.search(uri) is not None
            or "?" in uri
            or "#" in uri
        ):
            ChatbookGatewayRuntime._raise_invalid_resource_result()
        try:
            parsed = urlsplit(uri)
        except ValueError:
            ChatbookGatewayRuntime._raise_invalid_resource_result()
        base_uri = urlunsplit(
            (parsed.scheme.lower(), parsed.netloc, parsed.path, "", "")
        )
        for template, matcher in _RESOURCE_TEMPLATE_MATCHERS.items():
            variable = _RESOURCE_TEMPLATE_VARIABLES[template]
            scheme = template.split(":", 1)[0]
            matched = matcher.fullmatch(base_uri)
            if matched is None:
                continue
            try:
                identifier = unquote(matched.group(variable), errors="strict")
            except UnicodeDecodeError:
                ChatbookGatewayRuntime._raise_invalid_resource_result()
            if not identifier:
                ChatbookGatewayRuntime._raise_invalid_resource_result()
            try:
                canonical_identifier = quote(identifier, safe=_URI_UNRESERVED)
            except UnicodeEncodeError:
                ChatbookGatewayRuntime._raise_invalid_resource_result()
            return f"{scheme}://{canonical_identifier}"
        ChatbookGatewayRuntime._raise_invalid_resource_result()

    @staticmethod
    def _digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _encode_continuation(
        *, offset: int, base_digest: str, content_digest: str
    ) -> str:
        return _encode_continuation_state(
            offset=offset,
            base_digest=base_digest,
            content_digest=content_digest,
        )

    @staticmethod
    def _decode_continuation(token: str) -> dict[str, Any]:
        if (
            len(token) > _MAX_CONTINUATION_TOKEN_CHARS
            or _CONTINUATION_TOKEN.fullmatch(token) is None
        ):
            ChatbookGatewayRuntime._raise_invalid_resource_uri()
        try:
            padding = b"=" * (-len(token) % 4)
            decoded = base64.b64decode(
                token.encode("ascii") + padding,
                altchars=b"-_",
                validate=True,
            )
            state = json.loads(decoded, object_pairs_hook=_unique_json_object)
        except (Base64Error, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            ChatbookGatewayRuntime._raise_invalid_resource_uri()
        if (
            not isinstance(state, dict)
            or set(state) != {"b", "c", "o", "v"}
            or not isinstance(state["v"], int)
            or state["v"] != _CONTINUATION_VERSION
            or isinstance(state["v"], bool)
            or isinstance(state["o"], bool)
            or not isinstance(state["o"], int)
            or state["o"] <= 0
            or not isinstance(state["b"], str)
            or _SHA256_HEX.fullmatch(state["b"]) is None
            or not isinstance(state["c"], str)
            or _SHA256_HEX.fullmatch(state["c"]) is None
        ):
            ChatbookGatewayRuntime._raise_invalid_resource_uri()
        return state

    @staticmethod
    def _raise_invalid_resource_uri() -> NoReturn:
        raise GatewayApplicationError(
            "Invalid resource URI.",
            reason_code="invalid_resource_uri",
            kind="resource",
        ) from None

    @staticmethod
    def _raise_invalid_resource_result() -> NoReturn:
        raise GatewayApplicationError(
            "Resource handler returned an invalid result.",
            reason_code="invalid_resource_result",
            kind="resource",
        ) from None

    def finalize(self) -> None:
        """Validate exact descriptor/handler identity and freeze registration."""
        handler_names = set(self._tool_handlers) | set(self._local_tool_handlers)
        if set(self._tool_descriptors) != handler_names:
            raise ValueError("tool descriptor and handler names must match exactly")
        resource_surface_present = bool(
            self._resource_handlers or self._resource_list_handler
        )
        if resource_surface_present and (
            set(self._resource_handlers) != set(_RESOURCE_TEMPLATE_VARIABLES)
            or self._resource_list_handler is None
        ):
            raise ValueError(
                "resource templates and dynamic resource catalog must be registered together"
            )
        if self._prompt_handlers and set(self._prompt_handlers) != set(_PROMPT_NAMES):
            raise ValueError("all five Chatbook prompts must be registered together")
        self._finalized = True

    def _require_finalized(self) -> None:
        if not self._finalized:
            raise RuntimeError("runtime must be finalized before serving")

    async def list_tools(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Return detached descriptors in their constructor order.

        Args:
            context: Gateway request context supplied by ``mcp-unified``.

        Returns:
            Defensive copies of the finalized tool descriptors.
        """
        self._require_finalized()
        return copy.deepcopy(list(self._tool_descriptors.values()))

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> GatewayJSONValue:
        """Call a registered tool and return its application value raw.

        Args:
            name: Registered tool name.
            arguments: JSON arguments received from the gateway.
            context: Gateway request context supplied by ``mcp-unified``.

        Returns:
            The tool's JSON-compatible application value.
        """
        self._require_finalized()
        local_handler = self._local_tool_handlers.get(name)
        if local_handler is not None:
            try:
                result = await asyncio.to_thread(local_handler, arguments)
            except Exception:
                result = None
            if not isinstance(result, ToolResult):
                self._raise_local_failure()
            if result.ok:
                return result.content
            reason_code, public_message = _LOCAL_FAILURES.get(
                result.error, _GENERIC_LOCAL_FAILURE
            )
            raise GatewayToolExecutionError(
                public_message, reason_code=reason_code
            ) from None
        handler = self._tool_handlers.get(name)
        if handler is None:
            raise GatewayToolExecutionError(
                "Tool not found",
                reason_code="tool_not_found",
            )
        return await handler(**arguments)

    @staticmethod
    def _raise_local_failure() -> NoReturn:
        reason_code, public_message = _GENERIC_LOCAL_FAILURE
        raise GatewayToolExecutionError(
            public_message, reason_code=reason_code
        ) from None
