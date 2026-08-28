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
import threading
from typing import Any, Mapping
from uuid import uuid4
import weakref

from loguru import logger

from tldw_chatbook.Agents.agent_models import (
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
)
from tldw_chatbook.Chat.library_activity import (
    LibraryActivityCandidate,
    LibraryActivitySink,
    minimize_library_activity,
)
from tldw_chatbook.Agents.run_context import current_run_actor
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


@dataclass(frozen=True, slots=True, weakref_slot=True)
class BuiltinLibraryAuthority:
    """Live, credential-free authority issued by one built-in provider."""

    provider_instance_id: str
    reserved_names: frozenset[str]
    assistant_access: ConsoleAssistantLibraryAccess


class _BuiltinLibraryAuthorityIssuer:
    """Private instance-identity capability shared by the two built-in providers."""

    def _initialize_builtin_authority_issuer(self) -> None:
        self._builtin_library_provider_instance_id = uuid4().hex
        self._builtin_library_authorities: dict[
            int, weakref.ReferenceType[BuiltinLibraryAuthority]
        ] = {}
        self._builtin_library_authority_lock = threading.RLock()

    def issue_builtin_authority(
        self,
        *,
        reserved_names: frozenset[str],
        assistant_access: ConsoleAssistantLibraryAccess,
    ) -> BuiltinLibraryAuthority:
        """Issue one independently live authority for an owning run registry."""
        authority = BuiltinLibraryAuthority(
            provider_instance_id=self._builtin_library_provider_instance_id,
            reserved_names=reserved_names,
            assistant_access=assistant_access,
        )
        authority_key = id(authority)
        issuer_ref = weakref.ref(self)

        def _discard(
            dead_ref: weakref.ReferenceType[BuiltinLibraryAuthority],
            *,
            key: int = authority_key,
            owner_ref: weakref.ReferenceType[_BuiltinLibraryAuthorityIssuer] = issuer_ref,
        ) -> None:
            owner = owner_ref()
            if owner is None:
                return
            with owner._builtin_library_authority_lock:
                if owner._builtin_library_authorities.get(key) is dead_ref:
                    owner._builtin_library_authorities.pop(key, None)

        authority_ref = weakref.ref(authority, _discard)
        with self._builtin_library_authority_lock:
            self._builtin_library_authorities[authority_key] = authority_ref
        return authority

    def authenticates_builtin_authority(
        self, authority: object
    ) -> bool:
        """Authenticate only the exact currently issued object for this instance."""
        if (
            not isinstance(authority, BuiltinLibraryAuthority)
            or authority.provider_instance_id
            != self._builtin_library_provider_instance_id
        ):
            return False
        with self._builtin_library_authority_lock:
            authority_ref = self._builtin_library_authorities.get(id(authority))
            return authority_ref is not None and authority_ref() is authority


class LibraryToolProvider(_BuiltinLibraryAuthorityIssuer):
    """Exposes the 18 descriptor-backed ``library_*`` tools to Console agents.

    Catalog entries and schemas are derived from ``LIBRARY_TOOL_DESCRIPTORS``
    (never hand-maintained here), so the Console catalog can never drift from
    the contract the MCP surface registers. Tool IDs carry the provider's own
    ``library:<name>`` source prefix.
    """

    SOURCE = "library"

    def __init__(
        self,
        service: Any,
        *,
        activity_attempt_id: str | None = None,
        activity_sink: LibraryActivitySink | None = None,
    ) -> None:
        """Bind the shared synchronous Library service (duck-typed ``invoke``)."""
        self._initialize_builtin_authority_issuer()
        self._service = service
        self._activity_attempt_id = activity_attempt_id
        self._activity_sink = activity_sink

    @staticmethod
    def _capture_failure() -> ToolResult:
        return _error_result(
            LibraryToolError(
                ERROR_STORAGE_ERROR,
                "Library result withheld because activity could not be recorded.",
                retryable=True,
                details={"category": "review_capture_failed"},
            )
        )

    def _capture_activity(
        self, name: str, arguments: Mapping[str, Any], payload: object
    ) -> bool:
        if self._activity_sink is None:
            return True
        actor = current_run_actor()
        if actor is None or not self._activity_attempt_id:
            logger.warning(
                "Library activity capture failed; result withheld "
                "category=review_capture_failed"
            )
            return False
        try:
            event = minimize_library_activity(
                LibraryActivityCandidate(
                    attempt_id=self._activity_attempt_id,
                    actor_kind=actor.kind,
                    run_id=actor.run_id,
                    parent_run_id=actor.parent_run_id,
                    library_provider="direct",
                    operation=name,
                    arguments=arguments,
                    structured_result=payload,
                    failure_code=None,
                )
            )
            self._activity_sink(event)
        except Exception:  # noqa: BLE001 - payload/exception text must not log
            logger.warning(
                "Library activity capture failed; result withheld "
                "category=review_capture_failed"
            )
            return False
        return True

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
        if not self._capture_activity(name, arguments, payload):
            return self._capture_failure()
        text = json_dumps_compact(payload)
        if isinstance(payload, Mapping) and "error" in payload:
            return ToolResult(ok=False, error=text)
        return ToolResult(ok=True, content=text)


__all__ = ["BuiltinLibraryAuthority", "LibraryToolProvider"]
