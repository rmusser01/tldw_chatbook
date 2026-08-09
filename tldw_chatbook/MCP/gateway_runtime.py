"""Small adapter from Chatbook's decorators to the strict MCP gateway."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import re
from collections.abc import Awaitable, Callable, Iterable
from typing import TYPE_CHECKING, Any, NoReturn

from jsonschema import Draft7Validator, Draft202012Validator
from jsonschema.exceptions import SchemaError
from mcp_unified.gateway import (
    GatewayJSONValue,
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
_ToolHandler = Callable[..., Awaitable[GatewayJSONValue]]
_LocalToolHandler = Callable[[dict[str, Any]], ToolResult]

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


class ChatbookGatewayRuntime:
    """Register and dispatch Chatbook's standalone built-in tools."""

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
        """Return a decorator that records one async built-in handler."""
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
        """Validate and publish one complete local-tool registration set."""
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
            try:
                serialized_parameters = json.dumps(parameters, allow_nan=False)
                roundtripped_parameters = json.loads(serialized_parameters)
            except (RecursionError, TypeError, ValueError):
                parameters_are_finite_json = False
            else:
                parameters_are_finite_json = roundtripped_parameters == parameters
            if not parameters_are_finite_json:
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

    def finalize(self) -> None:
        """Validate exact descriptor/handler identity and freeze registration."""
        handler_names = set(self._tool_handlers) | set(self._local_tool_handlers)
        if set(self._tool_descriptors) != handler_names:
            raise ValueError("tool descriptor and handler names must match exactly")
        self._finalized = True

    def _require_finalized(self) -> None:
        if not self._finalized:
            raise RuntimeError("runtime must be finalized before serving")

    async def list_tools(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Return detached descriptors in their constructor order."""
        self._require_finalized()
        return copy.deepcopy(list(self._tool_descriptors.values()))

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> GatewayJSONValue:
        """Call a registered tool and return its application value raw."""
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
