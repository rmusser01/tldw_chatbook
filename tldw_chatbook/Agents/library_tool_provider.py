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

import copy
from dataclasses import dataclass, field
import threading
from typing import TYPE_CHECKING, Any, Mapping
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
from tldw_chatbook.Agents.run_context import (
    CurrentRunActor,
    current_run_actor,
    current_tool_call_id,
)
from tldw_chatbook.Library.library_tool_contract import (
    ERROR_INVALID_ARGUMENT,
    ERROR_STORAGE_ERROR,
    LIBRARY_TOOL_DESCRIPTORS,
    LibraryToolError,
    json_dumps_compact,
    make_public_id,
)
if TYPE_CHECKING:
    from tldw_chatbook.Notes.agent_lessons import AgentLessonClassification


class AgentLessonPreflightError(RuntimeError):
    """Content-free failure to classify one immutable lesson-save call."""

    def __init__(self) -> None:
        super().__init__("agent_lesson_classification_failed")


@dataclass(frozen=True, slots=True)
class AgentLessonSavePreflight:
    """Content-free snapshot bound to one reviewed Library save call."""

    call_id: str
    call_digest: str
    content_digest: str
    operation: str
    note_id: str | None
    title: str
    classification: AgentLessonClassification
    expected_version: int | None
    expected_organization_version: str | None
    observed_note_version: int | None
    observed_organization_version: str | None
    receipt_state: str | None
    receipt_version: str | None
    requested_marker: bool
    current_marker: bool
    receipt_requested_marker: bool


@dataclass(frozen=True, slots=True)
class _AgentLessonApprovalAuthority:
    """Opaque issuer-bound approval retained only until transaction handoff."""

    provider_instance_id: str = field(repr=False)
    token: str = field(repr=False)
    run_id: str
    preflight: AgentLessonSavePreflight


@dataclass(frozen=True, slots=True)
class _AgentLessonMutationContext:
    """Private handoff from this provider to the Notes transaction."""

    issuer: object = field(repr=False)
    authority: object | None = field(repr=False)
    actor: CurrentRunActor | None
    call_id: str


class _AgentLessonAuthorityRefusal(RuntimeError):
    """Content-free transaction refusal returned by the issuer."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


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
        self._agent_lesson_provider_instance_id = uuid4().hex
        self._agent_lesson_approvals: dict[
            tuple[str, str, str], _AgentLessonApprovalAuthority
        ] = {}
        self._agent_lesson_approval_lock = threading.RLock()

    def preflight_agent_lesson_save(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        call_id: str,
    ) -> AgentLessonSavePreflight | None:
        """Classify one immutable save without retaining or exposing note text.

        ``None`` means the call is an ordinary Note save. Any unreadable or
        incomplete update snapshot raises one stable content-free error so the
        review hook can fail closed without putting private content on a card.
        """

        if tool_name != "library_save_note":
            return None
        if not isinstance(arguments, Mapping):
            raise AgentLessonPreflightError()
        requested = arguments.get("ensure_keywords") or ()
        if isinstance(requested, (str, bytes)) or not isinstance(
            requested, (list, tuple)
        ):
            raise AgentLessonPreflightError()
        if any(not isinstance(item, str) for item in requested):
            raise AgentLessonPreflightError()

        note_id = arguments.get("note_id")
        current_keywords: tuple[str, ...] = ()
        receipt_state: str | None = None
        receipt_version: str | None = None
        receipt_requested_marker = False
        observed_note_version: int | None = None
        observed_organization_version: str | None = None
        if note_id is not None:
            if not isinstance(note_id, str) or not note_id:
                raise AgentLessonPreflightError()
            try:
                snapshot_reader = getattr(
                    self._service, "agent_lesson_preflight_snapshot"
                )
                snapshot = snapshot_reader(note_id)
            except Exception:  # noqa: BLE001 - never retain payload text
                raise AgentLessonPreflightError() from None
            if not isinstance(snapshot, Mapping):
                raise AgentLessonPreflightError()
            if snapshot.get("public_note_id") != note_id:
                raise AgentLessonPreflightError()
            metadata = snapshot.get("keywords")
            if isinstance(metadata, (str, bytes)) or not isinstance(
                metadata, (list, tuple)
            ):
                raise AgentLessonPreflightError()
            if any(not isinstance(name, str) for name in metadata):
                raise AgentLessonPreflightError()
            current_keywords = tuple(metadata)
            note_version = snapshot.get("note_version")
            organization_version = snapshot.get("organization_version")
            if (
                isinstance(note_version, bool)
                or not isinstance(note_version, int)
                or note_version < 1
                or not isinstance(organization_version, str)
                or not organization_version
            ):
                raise AgentLessonPreflightError()
            observed_note_version = note_version
            observed_organization_version = organization_version
            state = snapshot.get("receipt_state")
            if state in ("pending_organization", "placement_review"):
                receipt_state = state
            elif state is not None:
                raise AgentLessonPreflightError()
            if receipt_state is not None:
                receipt_note_version = snapshot.get("receipt_note_version")
                receipt_organization_version = snapshot.get(
                    "receipt_organization_version"
                )
                if (
                    isinstance(receipt_note_version, bool)
                    or not isinstance(receipt_note_version, int)
                    or receipt_note_version < 1
                    or not isinstance(receipt_organization_version, str)
                    or not receipt_organization_version
                ):
                    raise AgentLessonPreflightError()
                receipt_version = (
                    f"{receipt_note_version}:{receipt_organization_version}"
                )
                receipt_keywords = snapshot.get("receipt_requested_keywords") or ()
                if isinstance(receipt_keywords, (str, bytes)) or not isinstance(
                    receipt_keywords, (list, tuple)
                ):
                    raise AgentLessonPreflightError()
                if any(not isinstance(name, str) for name in receipt_keywords):
                    raise AgentLessonPreflightError()
                receipt_requested_marker = "agent-lesson" in receipt_keywords

        from tldw_chatbook.Notes.agent_lessons import (
            canonical_call_digest,
            classify_agent_lesson,
            lesson_content_digest,
        )

        classification = classify_agent_lesson(
            requested_keywords=tuple(requested),
            current_keywords=current_keywords,
            receipt_state=receipt_state,
        )
        if not classification.is_agent_lesson:
            return None
        normalized_call_id = str(call_id or "")
        if not normalized_call_id:
            raise AgentLessonPreflightError()
        title = arguments.get("title")
        content = arguments.get("content")
        if not isinstance(title, str) or not isinstance(content, str):
            raise AgentLessonPreflightError()
        expected_version = arguments.get("expected_version")
        if expected_version is not None and (
            isinstance(expected_version, bool) or not isinstance(expected_version, int)
        ):
            raise AgentLessonPreflightError()
        expected_organization_version = arguments.get(
            "expected_organization_version"
        )
        if expected_organization_version is not None and not isinstance(
            expected_organization_version, str
        ):
            raise AgentLessonPreflightError()
        try:
            call_digest = canonical_call_digest(tool_name, arguments)
            content_digest = lesson_content_digest(content)
        except (TypeError, ValueError):
            raise AgentLessonPreflightError() from None
        return AgentLessonSavePreflight(
            call_id=normalized_call_id,
            call_digest=call_digest,
            content_digest=content_digest,
            operation="update" if note_id is not None else "create",
            note_id=note_id,
            title=title[:160],
            classification=classification,
            expected_version=expected_version,
            expected_organization_version=expected_organization_version,
            observed_note_version=observed_note_version,
            observed_organization_version=observed_organization_version,
            receipt_state=receipt_state,
            receipt_version=receipt_version,
            requested_marker="agent-lesson" in requested,
            current_marker="agent-lesson" in current_keywords,
            receipt_requested_marker=receipt_requested_marker,
        )

    def clear_agent_lesson_approvals(self, run_id: str) -> None:
        """Discard every unconsumed lesson approval held for one run."""

        normalized = str(run_id or "")
        with self._agent_lesson_approval_lock:
            stale = [key for key in self._agent_lesson_approvals if key[0] == normalized]
            for key in stale:
                self._agent_lesson_approvals.pop(key, None)

    def issue_agent_lesson_approval(
        self, run_id: str, preflight: AgentLessonSavePreflight
    ) -> _AgentLessonApprovalAuthority:
        """Hold one opaque approve-once authority for the trusted primary run."""

        actor = current_run_actor()
        if (
            actor is None
            or actor.kind != "primary"
            or actor.run_id != run_id
            or not isinstance(preflight, AgentLessonSavePreflight)
        ):
            raise AgentLessonPreflightError()
        authority = _AgentLessonApprovalAuthority(
            provider_instance_id=self._agent_lesson_provider_instance_id,
            token=uuid4().hex,
            run_id=run_id,
            preflight=preflight,
        )
        key = (run_id, preflight.call_id, preflight.call_digest)
        with self._agent_lesson_approval_lock:
            self._agent_lesson_approvals[key] = authority
        return authority

    def peek_agent_lesson_approval(
        self, run_id: str, call_id: str, call_digest: str
    ) -> _AgentLessonApprovalAuthority | None:
        """Return the exact held authority without consuming it (Task 6 does that)."""

        with self._agent_lesson_approval_lock:
            return self._agent_lesson_approvals.get((run_id, call_id, call_digest))

    def _agent_lesson_approval_for_call(
        self, run_id: str, call_id: str
    ) -> _AgentLessonApprovalAuthority | None:
        """Find a reviewed call before recomputing its possibly changed digest."""

        with self._agent_lesson_approval_lock:
            matches = [
                authority
                for (held_run, held_call, _digest), authority in (
                    self._agent_lesson_approvals.items()
                )
                if held_run == run_id and held_call == call_id
            ]
            return matches[0] if matches else None

    def agent_lesson_approval_count(self, run_id: str) -> int:
        """Expose only a content-free count for lifecycle tests and diagnostics."""

        with self._agent_lesson_approval_lock:
            return sum(key[0] == run_id for key in self._agent_lesson_approvals)

    def _consume_agent_lesson_approval(
        self,
        context: object,
        *,
        raw_arguments: Mapping[str, Any],
        note_id: str | None,
        classification: AgentLessonClassification,
        requested_marker: bool,
        current_marker: bool,
        receipt_requested_marker: bool,
        observed_note_version: int | None,
        observed_organization_version: str | None,
        receipt_state: str | None,
        receipt_version: str | None,
    ) -> None:
        """Consume one exact authority after a complete locked comparison."""

        if type(context) is not _AgentLessonMutationContext:
            raise _AgentLessonAuthorityRefusal("approval_required")
        actor = context.actor
        if actor is not None and actor.kind != "primary":
            raise _AgentLessonAuthorityRefusal("foreground_required")
        authority = context.authority
        if (
            actor is None
            or not context.call_id
            or type(authority) is not _AgentLessonApprovalAuthority
            or context.issuer is not self
        ):
            raise _AgentLessonAuthorityRefusal("approval_required")
        from tldw_chatbook.Notes.agent_lessons import (
            canonical_call_digest,
            lesson_content_digest,
        )

        try:
            call_digest = canonical_call_digest("library_save_note", raw_arguments)
            content_digest = lesson_content_digest(raw_arguments.get("content"))
        except (TypeError, ValueError):
            raise _AgentLessonAuthorityRefusal("approval_required") from None
        key = (actor.run_id, context.call_id, call_digest)
        with self._agent_lesson_approval_lock:
            held = self._agent_lesson_approvals.get(key)
            if (
                held is not authority
                or authority.provider_instance_id
                != self._agent_lesson_provider_instance_id
                or authority.run_id != actor.run_id
            ):
                raise _AgentLessonAuthorityRefusal("approval_required")
            reviewed = authority.preflight
            public_note_id = make_public_id("note", note_id) if note_id else None
            if (
                reviewed.call_id != context.call_id
                or reviewed.call_digest != call_digest
                or reviewed.note_id != public_note_id
                or reviewed.operation != ("update" if note_id else "create")
                or reviewed.expected_version != raw_arguments.get("expected_version")
                or reviewed.expected_organization_version
                != raw_arguments.get("expected_organization_version")
                or reviewed.requested_marker != requested_marker
                or reviewed.current_marker != current_marker
                or reviewed.receipt_requested_marker != receipt_requested_marker
                or reviewed.classification != classification
                or reviewed.receipt_state != receipt_state
            ):
                raise _AgentLessonAuthorityRefusal("approval_required")
            if (
                reviewed.content_digest != content_digest
                or reviewed.observed_note_version != observed_note_version
            ):
                raise _AgentLessonAuthorityRefusal("content_changed")
            if (
                reviewed.observed_organization_version
                != observed_organization_version
                or reviewed.receipt_version != receipt_version
            ):
                raise _AgentLessonAuthorityRefusal("organization_changed")
            self._agent_lesson_approvals.pop(key)

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
            parameters=copy.deepcopy(descriptor.input_schema),
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
            if name == "library_save_note" and hasattr(
                self._service, "_invoke_with_agent_lesson_context"
            ):
                actor = current_run_actor()
                call_id = current_tool_call_id()
                authority = None
                if actor is not None and call_id:
                    authority = self._agent_lesson_approval_for_call(
                        actor.run_id, call_id
                    )
                payload = self._service._invoke_with_agent_lesson_context(
                    name,
                    arguments,
                    _AgentLessonMutationContext(
                        issuer=self,
                        authority=authority,
                        actor=actor,
                        call_id=call_id,
                    ),
                )
            else:
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


__all__ = [
    "AgentLessonPreflightError",
    "AgentLessonSavePreflight",
    "BuiltinLibraryAuthority",
    "LibraryToolProvider",
]
