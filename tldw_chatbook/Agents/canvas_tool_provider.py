"""Conversation-scoped Canvas tools with source-safe record projections."""

from __future__ import annotations

import json
import re
import weakref
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable
from uuid import UUID, uuid4

from tldw_chatbook.Canvas.limits import (
    MAX_CANVAS_TITLE_BYTES,
    MAX_CANVASES_PER_CONVERSATION,
    MAX_DURABLE_SOURCE_BYTES_PER_REVISION,
    MAX_REVISIONS_PER_CANVAS,
    CanvasLimitError,
    sha256_utf8,
    validate_opaque_identifier,
    validate_utf8_text,
)
from tldw_chatbook.Canvas.models import (
    CanvasCompatibilityIssue,
    CanvasConflictResult,
    CanvasCreateResult,
    CanvasListItem,
    CanvasMutationResult,
    CanvasReadResult,
    CanvasScope,
)

from .agent_models import (
    ToolCall,
    ToolCatalogEntry,
    ToolProjectionAudience,
    ToolRecordProjection,
    ToolResult,
    ToolSchema,
)
from .run_context import current_run_id, current_tool_call_id

CANVAS_SOURCE = "canvas"
CANVAS_TOOL_NAMES = frozenset(
    {"canvas_list", "canvas_read", "canvas_create", "canvas_update"}
)
CANVAS_MUTATION_TOOL_NAMES = frozenset({"canvas_create", "canvas_update"})
_SAFE_CODE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


class CanvasApprovalClassification(StrEnum):
    """Nominal approval category owned exclusively by the Canvas provider."""

    REVERSIBLE_CONVERSATION_LOCAL = "canvas_reversible_conversation_local"


CANVAS_MUTATION_APPROVAL_CLASSIFICATION = (
    CanvasApprovalClassification.REVERSIBLE_CONVERSATION_LOCAL
)


@runtime_checkable
class CanvasToolCoordinator(Protocol):
    """Server-owned execution port for one captured Console Canvas scope.

    Task 3.3 supplies the implementation that routes durable and temporary
    mutations into turn staging.  Keeping that lifecycle behind this port
    prevents the provider from inventing message, branch, or persistence
    authority.
    """

    def is_scope_current(self, scope: CanvasScope) -> bool: ...

    def list_canvases(self, scope: CanvasScope) -> tuple[CanvasListItem, ...]: ...

    def read_canvas(self, scope: CanvasScope, canvas_id: str) -> CanvasReadResult: ...

    def create_canvas(
        self,
        scope: CanvasScope,
        *,
        tool_call_id: str,
        title: str,
        html: str,
    ) -> CanvasMutationResult | CanvasCreateResult: ...

    def update_canvas(
        self,
        scope: CanvasScope,
        *,
        tool_call_id: str,
        canvas_id: str,
        expected_parent_revision_id: str,
        html: str,
    ) -> CanvasMutationResult | CanvasConflictResult: ...


@dataclass(frozen=True, slots=True, weakref_slot=True)
class CanvasToolRegistrationAuthority:
    """Live issuer-bound capability for one exact scoped provider instance."""

    provider_instance_id: str = field(repr=False)
    session_id: str
    run_id: str
    classification: CanvasApprovalClassification


_SCHEMAS: dict[str, ToolSchema] = {
    "canvas_list": ToolSchema(
        id="canvas:canvas_list",
        name="canvas_list",
        description="List Canvases reachable on this conversation branch.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    ),
    "canvas_read": ToolSchema(
        id="canvas:canvas_read",
        name="canvas_read",
        description="Read the complete selected reachable Canvas revision.",
        parameters={
            "type": "object",
            "properties": {
                "canvas_id": {
                    "type": "string",
                    "format": "uuid",
                    "maxLength": 36,
                }
            },
            "required": ["canvas_id"],
            "additionalProperties": False,
        },
    ),
    "canvas_create": ToolSchema(
        id="canvas:canvas_create",
        name="canvas_create",
        description="Stage a new Canvas from one complete HTML document.",
        parameters={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_CANVAS_TITLE_BYTES,
                },
                "html": {
                    "type": "string",
                    "maxLength": MAX_DURABLE_SOURCE_BYTES_PER_REVISION,
                },
            },
            "required": ["title", "html"],
            "additionalProperties": False,
        },
    ),
    "canvas_update": ToolSchema(
        id="canvas:canvas_update",
        name="canvas_update",
        description=(
            "Stage a full replacement document from the exact expected Canvas revision."
        ),
        parameters={
            "type": "object",
            "properties": {
                "canvas_id": {
                    "type": "string",
                    "format": "uuid",
                    "maxLength": 36,
                },
                "expected_parent_revision_id": {
                    "type": "string",
                    "format": "uuid",
                    "maxLength": 36,
                },
                "html": {
                    "type": "string",
                    "maxLength": MAX_DURABLE_SOURCE_BYTES_PER_REVISION,
                },
            },
            "required": ["canvas_id", "expected_parent_revision_id", "html"],
            "additionalProperties": False,
        },
    ),
}

_DESCRIPTIONS = {name: schema.description for name, schema in _SCHEMAS.items()}
_ERROR_MESSAGES = {
    "canvas_scope_unavailable": "Canvas is unavailable for this tool call.",
    "invalid_arguments": "Canvas tool arguments are invalid.",
    "invalid_canvas_id": "Canvas identifier is invalid.",
    "invalid_expected_parent": "Expected Canvas revision identifier is invalid.",
    "invalid_title": "Canvas title must not be empty.",
    "title_bytes": "Canvas title exceeds its UTF-8 byte limit.",
    "revision_source_bytes": "Canvas HTML exceeds its UTF-8 byte limit.",
    "operation_failed": "Canvas operation failed.",
}


class _ArgumentError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class CanvasToolProvider:
    """Expose four Canvas tools bound to one immutable server-owned scope."""

    SOURCE = CANVAS_SOURCE

    def __init__(
        self,
        coordinator: CanvasToolCoordinator,
        *,
        scope: CanvasScope,
        enabled: bool = True,
    ) -> None:
        if not isinstance(scope, CanvasScope):
            raise TypeError("scope must be a CanvasScope")
        if not isinstance(coordinator, CanvasToolCoordinator):
            raise TypeError("coordinator must implement CanvasToolCoordinator")
        if type(enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        self._coordinator = coordinator
        self._scope = scope
        self._enabled = enabled
        self._provider_instance_id = uuid4().hex
        self._authorities: dict[
            int, weakref.ReferenceType[CanvasToolRegistrationAuthority]
        ] = {}

    def issue_registration_authority(self) -> CanvasToolRegistrationAuthority:
        """Issue a live capability which only this provider can authenticate."""

        authority = CanvasToolRegistrationAuthority(
            provider_instance_id=self._provider_instance_id,
            session_id=self._scope.session_id,
            run_id=self._scope.run_id,
            classification=CANVAS_MUTATION_APPROVAL_CLASSIFICATION,
        )
        self._authorities[id(authority)] = weakref.ref(authority)
        return authority

    def authenticates_registration_authority(self, authority: object) -> bool:
        """Return whether *authority* is the exact live object this instance issued."""

        if type(authority) is not CanvasToolRegistrationAuthority:
            return False
        assert isinstance(authority, CanvasToolRegistrationAuthority)
        if (
            authority.provider_instance_id != self._provider_instance_id
            or authority.session_id != self._scope.session_id
            or authority.run_id != self._scope.run_id
            or authority.classification is not CANVAS_MUTATION_APPROVAL_CLASSIFICATION
        ):
            return False
        reference = self._authorities.get(id(authority))
        return reference is not None and reference() is authority

    def scope_is_current(self) -> bool:
        """Fail closed on coordinator errors or a stale session/run binding."""

        try:
            return self._enabled and bool(
                self._coordinator.is_scope_current(self._scope)
            )
        except Exception:  # noqa: BLE001 - availability checks retain no payload
            return False

    def list_catalog(self) -> list[ToolCatalogEntry]:
        if not self.scope_is_current():
            return []
        return [
            ToolCatalogEntry(
                id=schema.id,
                name=name,
                one_line_description=_DESCRIPTIONS[name],
                source=CANVAS_SOURCE,
            )
            for name, schema in _SCHEMAS.items()
        ]

    def load_schema(self, tool_id: str) -> ToolSchema:
        name = _name_from_id(tool_id)
        try:
            return _SCHEMAS[name]
        except KeyError:
            raise KeyError(f"Unknown Canvas tool id: {tool_id}") from None

    def approval_classification_for(
        self, tool_id: str
    ) -> CanvasApprovalClassification | None:
        """Classify only Canvas create/update as reversible local mutations."""

        try:
            name = _name_from_id(tool_id)
        except (TypeError, ValueError):
            return None
        if name not in CANVAS_MUTATION_TOOL_NAMES or not self.scope_is_current():
            return None
        return CANVAS_MUTATION_APPROVAL_CLASSIFICATION

    def invoke(self, tool_id: str, args: dict) -> ToolResult:
        try:
            name = _name_from_id(tool_id)
        except (TypeError, ValueError):
            return _error("invalid_arguments")
        if name not in CANVAS_TOOL_NAMES:
            return _error("invalid_arguments")
        if not self._invocation_context_is_current():
            return _error("canvas_scope_unavailable")
        try:
            checked = _validate_arguments(name, args)
            if name == "canvas_list":
                result: object = self._coordinator.list_canvases(self._scope)
            elif name == "canvas_read":
                result = self._coordinator.read_canvas(
                    self._scope, checked["canvas_id"]
                )
            elif name == "canvas_create":
                result = self._coordinator.create_canvas(
                    self._scope,
                    tool_call_id=current_tool_call_id(),
                    title=checked["title"],
                    html=checked["html"],
                )
            else:
                result = self._coordinator.update_canvas(
                    self._scope,
                    tool_call_id=current_tool_call_id(),
                    canvas_id=checked["canvas_id"],
                    expected_parent_revision_id=checked["expected_parent_revision_id"],
                    html=checked["html"],
                )
            return _serialize_result(name, result)
        except _ArgumentError as exc:
            return _error(exc.code)
        except Exception as exc:  # noqa: BLE001 - coordinator failures are sanitized
            return _dependency_error(exc)

    def project_tool_record(
        self,
        audience: ToolProjectionAudience,
        call: ToolCall,
        result: ToolResult | None,
    ) -> ToolRecordProjection:
        """Remove HTML from every durable/display/cycle/continuation record."""

        if audience not in {"display", "log", "cycle", "continuation"}:
            raise ValueError("unsupported Canvas projection audience")
        if call.name not in CANVAS_TOOL_NAMES:
            raise ValueError("projection requested for an unknown Canvas tool")
        arguments = _project_arguments(call)
        projected_content, projected_error = _project_result(result)
        return ToolRecordProjection(
            arguments=arguments,
            content=projected_content,
            error=projected_error,
            ok=result.ok if result is not None else None,
            error_category=("" if result is None or result.ok else "CanvasToolError"),
        )

    def _invocation_context_is_current(self) -> bool:
        return bool(
            current_run_id()
            and current_run_id() == self._scope.run_id
            and current_tool_call_id()
            and self.scope_is_current()
        )


def _name_from_id(tool_id: str) -> str:
    if not isinstance(tool_id, str) or not tool_id.startswith("canvas:"):
        raise ValueError("invalid Canvas tool id")
    return tool_id.removeprefix("canvas:")


def _validate_arguments(name: str, args: object) -> dict[str, str]:
    if type(args) is not dict:
        raise _ArgumentError("invalid_arguments")
    expected = {
        "canvas_list": frozenset(),
        "canvas_read": frozenset({"canvas_id"}),
        "canvas_create": frozenset({"title", "html"}),
        "canvas_update": frozenset(
            {"canvas_id", "expected_parent_revision_id", "html"}
        ),
    }[name]
    if frozenset(args) != expected:
        raise _ArgumentError("invalid_arguments")
    checked = dict(args)
    if "canvas_id" in checked:
        checked["canvas_id"] = _uuid(checked["canvas_id"], "invalid_canvas_id")
    if "expected_parent_revision_id" in checked:
        checked["expected_parent_revision_id"] = _uuid(
            checked["expected_parent_revision_id"], "invalid_expected_parent"
        )
    if "title" in checked:
        title = checked["title"]
        if not isinstance(title, str) or not title.strip():
            raise _ArgumentError("invalid_title")
        try:
            validate_utf8_text(
                title, limit=MAX_CANVAS_TITLE_BYTES, field_name="Canvas title"
            )
        except CanvasLimitError:
            raise _ArgumentError("title_bytes") from None
    if "html" in checked:
        try:
            validate_utf8_text(
                checked["html"],
                limit=MAX_DURABLE_SOURCE_BYTES_PER_REVISION,
                field_name="Canvas HTML",
            )
        except (CanvasLimitError, TypeError):
            raise _ArgumentError("revision_source_bytes") from None
    return checked


def _uuid(value: object, code: str) -> str:
    if type(value) is not str:
        raise _ArgumentError(code)
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError):
        raise _ArgumentError(code) from None
    if str(parsed) != value:
        raise _ArgumentError(code)
    return value


def _serialize_result(name: str, result: object) -> ToolResult:
    if name == "canvas_list":
        if type(result) is not tuple or len(result) > MAX_CANVASES_PER_CONVERSATION:
            return _error("operation_failed")
        if not all(isinstance(item, CanvasListItem) for item in result):
            return _error("operation_failed")
        payload = {
            "status": "ok",
            "count": len(result),
            "canvases": [_list_item_payload(item) for item in result],
        }
    elif name == "canvas_read":
        if not isinstance(result, CanvasReadResult):
            return _error("operation_failed")
        source_bytes = validate_utf8_text(
            result.source,
            limit=MAX_DURABLE_SOURCE_BYTES_PER_REVISION,
            field_name="Canvas HTML",
        )
        if (
            source_bytes != result.revision.source_bytes
            or sha256_utf8(result.source) != result.revision.content_sha256
        ):
            return _error("operation_failed")
        payload = {
            "status": "ok",
            "canvas": _validated_revision_payload(result.revision),
            "html": result.source,
        }
    elif isinstance(result, CanvasConflictResult):
        payload = {"status": "conflict", "conflict": _conflict_payload(result)}
    elif isinstance(result, (CanvasMutationResult, CanvasCreateResult)):
        payload = {
            "status": "staged",
            "canvas": _validated_revision_payload(result.revision),
            "compatibility_issues": _validated_issues(result.compatibility_issues),
        }
    else:
        return _error("operation_failed")
    return ToolResult(ok=True, content=_json(payload))


def _revision_payload(revision: Any) -> dict[str, object]:
    return {
        "canvas_id": revision.canvas_id,
        "revision_id": revision.revision_id,
        "parent_revision_id": revision.parent_revision_id,
        "title": revision.title,
        "runtime_profile": revision.runtime_profile,
        "content_sha256": revision.content_sha256,
        "source_bytes": revision.source_bytes,
        "sequence": revision.sequence,
        "origin": asdict(revision.origin),
    }


def _validated_revision_payload(revision: Any) -> dict[str, object]:
    _uuid(revision.canvas_id, "operation_failed")
    _uuid(revision.revision_id, "operation_failed")
    if revision.parent_revision_id is not None:
        _uuid(revision.parent_revision_id, "operation_failed")
    _validated_title(revision.title)
    if revision.runtime_profile != "canvas-v1":
        raise _ArgumentError("operation_failed")
    _validated_digest(revision.content_sha256)
    if (
        type(revision.source_bytes) is not int
        or not 0 <= revision.source_bytes <= MAX_DURABLE_SOURCE_BYTES_PER_REVISION
    ):
        raise _ArgumentError("operation_failed")
    _validated_sequence(revision.sequence)
    _validated_origin(revision.origin)
    return _revision_payload(revision)


def _validated_title(title: object) -> None:
    if not isinstance(title, str) or not title.strip():
        raise _ArgumentError("operation_failed")
    try:
        validate_utf8_text(
            title, limit=MAX_CANVAS_TITLE_BYTES, field_name="Canvas title"
        )
    except CanvasLimitError:
        raise _ArgumentError("operation_failed") from None


def _validated_digest(digest: object) -> None:
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise _ArgumentError("operation_failed")


def _validated_sequence(sequence: object) -> None:
    if type(sequence) is not int or not 1 <= sequence <= MAX_REVISIONS_PER_CANVAS:
        raise _ArgumentError("operation_failed")


def _validated_origin(origin: object) -> None:
    try:
        validate_opaque_identifier(origin.message_id, field_name="origin message ID")  # type: ignore[attr-defined]
        validate_opaque_identifier(origin.run_id, field_name="origin run ID")  # type: ignore[attr-defined]
    except (AttributeError, CanvasLimitError, TypeError):
        raise _ArgumentError("operation_failed") from None


def _validated_issues(issues: object) -> list[dict[str, object]]:
    if type(issues) is not tuple or len(issues) > 16:
        raise _ArgumentError("operation_failed")
    if not all(isinstance(issue, CanvasCompatibilityIssue) for issue in issues):
        raise _ArgumentError("operation_failed")
    return [asdict(issue) for issue in issues]


def _list_item_payload(item: CanvasListItem) -> dict[str, object]:
    if (
        type(item.is_selected) is not bool
        or type(item.is_historical_selection) is not bool
    ):
        raise _ArgumentError("operation_failed")
    return {
        **_validated_revision_payload(item),
        "is_selected": item.is_selected,
        "is_historical_selection": item.is_historical_selection,
    }


def _conflict_payload(conflict: CanvasConflictResult) -> dict[str, object]:
    _uuid(conflict.canvas_id, "operation_failed")
    _uuid(conflict.current_revision_id, "operation_failed")
    _validated_title(conflict.title)
    _validated_digest(conflict.content_sha256)
    _validated_sequence(conflict.sequence)
    _validated_origin(conflict.origin)
    if conflict.code != "stale_parent":
        raise _ArgumentError("operation_failed")
    return {
        "code": conflict.code,
        "canvas_id": conflict.canvas_id,
        "current_revision_id": conflict.current_revision_id,
        "content_sha256": conflict.content_sha256,
        "title": conflict.title,
        "sequence": conflict.sequence,
        "origin": asdict(conflict.origin),
    }


def _project_arguments(call: ToolCall) -> dict[str, object]:
    args = call.args if type(call.args) is dict else {}
    projected: dict[str, object] = {}
    for key in ("canvas_id", "expected_parent_revision_id", "title"):
        value = args.get(key)
        if isinstance(value, str):
            projected[key] = value[: MAX_CANVAS_TITLE_BYTES if key == "title" else 256]
    html = args.get("html")
    if isinstance(html, str):
        try:
            projected["content_sha256"] = sha256_utf8(html)
            projected["source_bytes"] = len(html.encode("utf-8", errors="strict"))
        except (CanvasLimitError, UnicodeEncodeError):
            projected["content_sha256"] = "invalid-source"
    return projected


def _project_result(result: ToolResult | None) -> tuple[str, str]:
    if result is None:
        return "", ""
    raw = result.content if result.ok else result.error
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise TypeError("result is not an object")
        safe = _source_free_payload(payload)
        encoded = _json(safe)
    except Exception:  # noqa: BLE001 - malformed source-bearing values fail closed
        encoded = _json({"code": "canvas_projection_unavailable"})
    return (encoded, "") if result.ok else ("", encoded)


def _source_free_payload(payload: dict[str, object]) -> dict[str, object]:
    safe: dict[str, object] = {}
    status = payload.get("status")
    if isinstance(status, str) and status in {"ok", "staged", "conflict"}:
        safe["status"] = status
    count = payload.get("count")
    if type(count) is int and 0 <= count <= MAX_CANVASES_PER_CONVERSATION:
        safe["count"] = count
    canvas = payload.get("canvas")
    if isinstance(canvas, dict):
        safe["canvas"] = _safe_canvas_metadata(canvas)
    canvases = payload.get("canvases")
    if isinstance(canvases, list):
        safe["canvases"] = [
            _safe_canvas_metadata(item)
            for item in canvases[:MAX_CANVASES_PER_CONVERSATION]
            if isinstance(item, dict)
        ]
    conflict = payload.get("conflict")
    if isinstance(conflict, dict):
        safe["conflict"] = _safe_conflict_metadata(conflict)
    issues = payload.get("compatibility_issues")
    if isinstance(issues, list):
        safe["compatibility_issues"] = [
            _safe_issue(item) for item in issues[:16] if isinstance(item, dict)
        ]
    code = payload.get("code")
    if isinstance(code, str) and _SAFE_CODE.fullmatch(code):
        safe["code"] = code
    if not safe:
        safe["code"] = "canvas_projection_unavailable"
    return safe


def _safe_canvas_metadata(value: dict[str, object]) -> dict[str, object]:
    keys = (
        "canvas_id",
        "revision_id",
        "parent_revision_id",
        "title",
        "runtime_profile",
        "content_sha256",
        "source_bytes",
        "sequence",
        "origin",
        "is_selected",
        "is_historical_selection",
    )
    return {key: value[key] for key in keys if key in value}


def _safe_conflict_metadata(value: dict[str, object]) -> dict[str, object]:
    keys = (
        "code",
        "canvas_id",
        "current_revision_id",
        "content_sha256",
        "title",
        "sequence",
        "origin",
    )
    return {key: value[key] for key in keys if key in value}


def _safe_issue(value: dict[str, object]) -> dict[str, object]:
    return {
        key: value.get(key) for key in ("code", "message", "location") if key in value
    }


def _safe_dependency_code(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    return (
        code
        if isinstance(code, str) and _SAFE_CODE.fullmatch(code)
        else "operation_failed"
    )


def _dependency_error(exc: Exception) -> ToolResult:
    code = _safe_dependency_code(exc)
    issues = getattr(exc, "issues", ())
    if issues:
        try:
            return _error(code, compatibility_issues=_validated_issues(issues))
        except Exception:  # noqa: BLE001 - malformed diagnostics fail closed
            return _error("operation_failed")
    return _error(code)


def _error(
    code: str,
    *,
    compatibility_issues: list[dict[str, object]] | None = None,
) -> ToolResult:
    safe_code = (
        code
        if code in _ERROR_MESSAGES
        else code
        if _SAFE_CODE.fullmatch(code)
        else "operation_failed"
    )
    payload: dict[str, object] = {
        "code": safe_code,
        "message": _ERROR_MESSAGES.get(
            safe_code, "Canvas operation could not be completed."
        ),
    }
    if compatibility_issues is not None:
        payload["compatibility_issues"] = compatibility_issues
    return ToolResult(
        ok=False,
        error=_json(payload),
    )


def _json(payload: object) -> str:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


__all__ = [
    "CANVAS_MUTATION_APPROVAL_CLASSIFICATION",
    "CANVAS_MUTATION_TOOL_NAMES",
    "CANVAS_TOOL_NAMES",
    "CanvasApprovalClassification",
    "CanvasToolCoordinator",
    "CanvasToolProvider",
    "CanvasToolRegistrationAuthority",
]
