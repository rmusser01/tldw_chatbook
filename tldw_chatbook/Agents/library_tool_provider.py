"""Descriptor-backed Console ToolProvider over `LocalLibraryToolService`.

task-1337 / ADR-030/031/032. Synchronous `ToolProvider`: it runs on the
agent worker thread and delegates to the shared synchronous
`LocalLibraryToolService`. Result mapping follows the spec's Console/MCP
parity rule: a successful payload is JSON-serialized into
`ToolResult.content`, while the service's structured error object is
JSON-serialized into `ToolResult.error` with `ok=False` -- after JSON
decoding, Console and MCP expose the same Library payload/error shape.

This module stays free of Textual and MCP imports; the bridge registers it
per run after `BuiltinToolProvider` and before skills/MCP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from uuid import uuid4

from loguru import logger

from tldw_chatbook.Agents.agent_models import (
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
)
from tldw_chatbook.Library.library_tool_contract import (
    ERROR_INVALID_ARGUMENT,
    ERROR_STORAGE_ERROR,
    LIBRARY_TOOL_DESCRIPTORS,
    LibraryToolError,
    json_dumps_compact,
)


def _error_result(error: LibraryToolError) -> ToolResult:
    """Serialize a structured Library error into the Console result model."""
    return ToolResult(ok=False, error=json_dumps_compact(error.to_payload()))


@dataclass(frozen=True, slots=True)
class BuiltinLibraryAuthority:
    """Live, credential-free authority issued by one built-in provider."""

    provider_instance_id: str
    reserved_names: frozenset[str]
    assistant_access: ConsoleAssistantLibraryAccess


class _BuiltinLibraryAuthorityIssuer:
    """Private instance-identity capability shared by the two built-in providers."""

    def _initialize_builtin_authority_issuer(self) -> None:
        self._builtin_library_provider_instance_id = uuid4().hex
        self._builtin_library_authority: BuiltinLibraryAuthority | None = None

    @property
    def builtin_authority(self) -> BuiltinLibraryAuthority | None:
        """Return this provider's currently issued live authority object."""
        return self._builtin_library_authority

    def issue_builtin_authority(
        self,
        *,
        reserved_names: frozenset[str],
        assistant_access: ConsoleAssistantLibraryAccess,
    ) -> BuiltinLibraryAuthority:
        """Replace and return this provider instance's live authority object."""
        authority = BuiltinLibraryAuthority(
            provider_instance_id=self._builtin_library_provider_instance_id,
            reserved_names=reserved_names,
            assistant_access=assistant_access,
        )
        self._builtin_library_authority = authority
        return authority

    def authenticates_builtin_authority(
        self, authority: object
    ) -> bool:
        """Authenticate only the exact currently issued object for this instance."""
        return (
            authority is self._builtin_library_authority
            and isinstance(authority, BuiltinLibraryAuthority)
            and authority.provider_instance_id
            == self._builtin_library_provider_instance_id
        )


class LibraryToolProvider(_BuiltinLibraryAuthorityIssuer):
    """Exposes the 18 descriptor-backed ``library_*`` tools to Console agents.

    Catalog entries and schemas are derived from ``LIBRARY_TOOL_DESCRIPTORS``
    (never hand-maintained here), so the Console catalog can never drift from
    the contract the MCP surface registers. Tool IDs carry the provider's own
    ``library:<name>`` source prefix.
    """

    SOURCE = "library"

    def __init__(self, service: Any) -> None:
        """Bind the shared synchronous Library service (duck-typed ``invoke``)."""
        self._initialize_builtin_authority_issuer()
        self._service = service

    def _tool_id(self, name: str) -> str:
        return f"{self.SOURCE}:{name}"

    @staticmethod
    def _name_from_tool_id(tool_id: str) -> str:
        return tool_id.split(":", 1)[1] if ":" in tool_id else tool_id

    def list_catalog(self) -> list[ToolCatalogEntry]:
        return [
            ToolCatalogEntry(
                id=self._tool_id(descriptor.name),
                name=descriptor.name,
                one_line_description=descriptor.description,
                source=self.SOURCE,
            )
            for descriptor in LIBRARY_TOOL_DESCRIPTORS.values()
        ]

    def load_schema(self, tool_id: str) -> ToolSchema:
        descriptor = LIBRARY_TOOL_DESCRIPTORS[self._name_from_tool_id(tool_id)]
        return ToolSchema(
            id=tool_id,
            name=descriptor.name,
            description=descriptor.description,
            parameters=descriptor.input_schema,
        )

    def invoke(self, tool_id: str, args: dict) -> ToolResult:
        name = self._name_from_tool_id(tool_id)
        if name not in LIBRARY_TOOL_DESCRIPTORS:
            return _error_result(
                LibraryToolError(
                    ERROR_INVALID_ARGUMENT,
                    "Unknown Library tool; use one of the descriptor-backed "
                    "library_* names.",
                )
            )
        arguments: Mapping[str, Any] = args if isinstance(args, Mapping) else {}
        try:
            payload = self._service.invoke(name, arguments)
        except Exception:  # noqa: BLE001 — scrubbed; never escapes into the loop
            logger.opt(exception=True).warning(
                f"LibraryToolProvider: backend invoke raised for {name}"
            )
            payload = LibraryToolError(
                ERROR_STORAGE_ERROR,
                "The local Library store could not complete the read.",
                retryable=True,
            ).to_payload()
        text = json_dumps_compact(payload)
        if isinstance(payload, Mapping) and "error" in payload:
            return ToolResult(ok=False, error=text)
        return ToolResult(ok=True, content=text)


__all__ = ["BuiltinLibraryAuthority", "LibraryToolProvider"]
