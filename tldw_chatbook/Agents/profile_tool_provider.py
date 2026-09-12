"""Run-scoped Personal Context tools for Console agents."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Iterator

from tldw_profile_core import (
    AgentVisibility,
    ProfileGetRequest,
    ProfilePromoteRequest,
    ProfileProposeRequest,
    ProfileSearchRequest,
    ProfileUpdateRequest,
    ProfileToolResult,
    RecordState,
    ToolOperation,
    ToolResultStatus,
)
from pydantic import ValidationError

from tldw_chatbook.Personal_Context.proposal_service import (
    PrivateDuplicateReviewRequired,
    ProfileProposalQuota,
    ProposalQuotaExceeded,
)
from tldw_chatbook.Personal_Context.repository import ProposalLimitExceededError
from tldw_chatbook.Personal_Context.runtime_policy import (
    AgentAuthority,
    PersonalContextAuthorityError,
)
from tldw_chatbook.Personal_Context.service import (
    PersonalContextService,
    ProfileConflictError,
    ProfileKeyCollisionError,
)

from .agent_models import ToolCatalogEntry, ToolResult, ToolSchema


@dataclass(frozen=True, slots=True)
class ProfileToolRunScope:
    """Immutable authority and evidence captured for one root run tree."""

    run_id: str
    session_id: str
    profile_id: str
    scope_id: str
    authority: AgentAuthority
    generation: int
    authority_revision: str
    current_user_message_id: str | None = None
    current_user_text: str | None = None


_GLOBAL_PROPOSAL_QUOTA = ProfileProposalQuota()
_SCHEMA_MODELS = {
    "profile_search": ProfileSearchRequest,
    "profile_get": ProfileGetRequest,
    "profile_propose": ProfileProposeRequest,
    "profile_update": ProfileUpdateRequest,
    "profile_promote": ProfilePromoteRequest,
}
_DESCRIPTIONS = {
    "profile_search": "Search profile context visible to this agent.",
    "profile_get": "Get one visible profile record by ID.",
    "profile_propose": "Propose a reviewable profile change.",
    "profile_update": "Apply an explicit current-user profile correction.",
    "profile_promote": "Propose promoting workspace context to the global profile.",
}
_CATALOGS = {
    AgentAuthority.READ_ONLY: ("profile_search", "profile_get"),
    AgentAuthority.PROPOSE: (
        "profile_search",
        "profile_get",
        "profile_propose",
        "profile_promote",
    ),
    AgentAuthority.DIRECT_WRITE: tuple(_SCHEMA_MODELS),
}
_OPERATIONS = {
    "profile_search": ToolOperation.SEARCH,
    "profile_get": ToolOperation.GET,
    "profile_propose": ToolOperation.PROPOSE,
    "profile_update": ToolOperation.UPDATE,
    "profile_promote": ToolOperation.PROMOTE,
}
_MESSAGES = {
    ToolResultStatus.APPLIED: "Personal Context operation applied.",
    ToolResultStatus.PROPOSAL_CREATED: "Profile proposal created for review.",
}
_PREVIEW_MESSAGE_ID = (
    "9f3a61c7d0e84b2fa5c9137ed6b04821"
    "c781d5a942ef30b86a14f9dc375e802b"
    "e42c7a190d5fb86371ae924c6f08d35b"
    "a730e19cf4826bd057ac3e918d64f20b"
)


class ProfileToolProvider:
    """Expose only profile tools authorized by one immutable run scope."""

    SOURCE = "personal-context"

    def __init__(
        self,
        service: PersonalContextService,
        *,
        run_scope: ProfileToolRunScope,
        kill_switch: Callable[[], bool] = lambda: False,
        quota: ProfileProposalQuota = _GLOBAL_PROPOSAL_QUOTA,
        reserve_direct_update_schema: bool = False,
    ) -> None:
        self._service = service
        self._base_scope = run_scope
        self._scope_override: ContextVar[ProfileToolRunScope | None] = ContextVar(
            f"profile-tool-scope-{id(self)}", default=None
        )
        self._kill_switch = kill_switch
        self._reserve_direct_update_schema = reserve_direct_update_schema
        self._proposals = service.proposal_service(quota=quota)

    @property
    def _scope(self) -> ProfileToolRunScope:
        return self._scope_override.get() or self._base_scope

    @contextmanager
    def stamp_scope(self, run_id: str, scope: ProfileToolRunScope) -> Iterator[None]:
        """Temporarily override a scope for compatibility tests and nesting."""

        if run_id != scope.run_id:
            raise ValueError("run scope identity mismatch")
        token = self._scope_override.set(scope)
        try:
            yield
        finally:
            self._scope_override.reset(token)

    @staticmethod
    def _name(tool_id: str) -> str:
        return tool_id.split(":", 1)[1] if ":" in tool_id else tool_id

    def _live_scope(self) -> ProfileToolRunScope | None:
        try:
            if self._kill_switch():
                return None
            scope = self._scope
            manifest = self._service.get_manifest()
            current_authority = self._service.get_scope_authority(scope.scope_id)
            canonical_scope = next(
                item
                for item in self._service.list_scopes()
                if item.scope_id == scope.scope_id
            )
            view = self._service.authorized_context_view(
                active_workspace_scope_id=(
                    scope.scope_id
                    if canonical_scope.kind.value == "workspace"
                    else None
                )
            )
        except Exception:
            return None
        if (
            manifest.profile_id != scope.profile_id
            or manifest.purge_generation != scope.generation
            or current_authority is not scope.authority
            or view.authority_revision != scope.authority_revision
            or (
                canonical_scope.kind.value == "workspace"
                and view.workspace_scope_id != scope.scope_id
            )
        ):
            return None
        return scope

    def list_catalog(self) -> list[ToolCatalogEntry]:
        scope = self._live_scope()
        if scope is None:
            return []
        names = self._catalog_names(scope)
        return [
            ToolCatalogEntry(
                id=f"{self.SOURCE}:{name}",
                name=name,
                one_line_description=_DESCRIPTIONS[name],
                source=self.SOURCE,
            )
            for name in names
        ]

    def _catalog_names(self, scope: ProfileToolRunScope) -> tuple[str, ...]:
        names = _CATALOGS[scope.authority]
        if not self._scope_is_workspace(scope.scope_id):
            names = tuple(name for name in names if name != "profile_promote")
        if (
            scope.authority is AgentAuthority.DIRECT_WRITE
            and not (scope.current_user_message_id and scope.current_user_text)
            and not self._reserve_direct_update_schema
        ):
            return tuple(name for name in names if name != "profile_update")
        return names

    def _scope_is_workspace(self, scope_id: str) -> bool:
        try:
            return any(
                scope.scope_id == scope_id and scope.kind.value == "workspace"
                for scope in self._service.list_scopes()
            )
        except Exception:
            return False

    def load_schema(self, tool_id: str) -> ToolSchema:
        name = self._name(tool_id)
        model = _SCHEMA_MODELS[name]
        parameters = model.model_json_schema()
        if name == "profile_update":
            message_id = self._scope.current_user_message_id
            if message_id is None and self._reserve_direct_update_schema:
                message_id = _PREVIEW_MESSAGE_ID
            if message_id is not None:
                parameters["properties"]["current_user_message_id"]["const"] = (
                    message_id
                )
        return ToolSchema(
            id=f"{self.SOURCE}:{name}",
            name=name,
            description=_DESCRIPTIONS[name],
            parameters=parameters,
        )

    def invoke(self, tool_id: str, args: dict) -> ToolResult:
        """Invoke one tool after rechecking every captured authority fence."""

        name = self._name(tool_id)
        scope = self._live_scope()
        if (
            scope is None
            or name not in _SCHEMA_MODELS
            or name not in self._catalog_names(scope)
        ):
            return self._failure(
                ToolResultStatus.PROFILE_LOCKED
                if self._service.status().locked
                else ToolResultStatus.PERMISSION_DENIED
            )
        try:
            request = _SCHEMA_MODELS[name].model_validate(args)
        except (TypeError, ValidationError, ValueError):
            status = (
                ToolResultStatus.REVIEW_REQUIRED
                if name in {"profile_propose", "profile_update", "profile_promote"}
                else ToolResultStatus.PERMISSION_DENIED
            )
            return self._failure(status)
        try:
            if name == "profile_search":
                return self._search(request)
            if name == "profile_get":
                return self._get(request)
            if name == "profile_propose":
                evidence_reference, evidence_hash = self._proposal_evidence(
                    request, scope
                )
                if request.evidence_span is not None and evidence_hash is None:
                    return self._failure(ToolResultStatus.REVIEW_REQUIRED)
                self._proposals.create(
                    request,
                    profile_id=scope.profile_id,
                    scope_id=scope.scope_id,
                    turn_id=scope.run_id,
                    session_id=scope.session_id,
                    evidence_reference=evidence_reference,
                    evidence_hash=evidence_hash,
                )
                return self._success(
                    ToolOperation.PROPOSE,
                    ToolResultStatus.PROPOSAL_CREATED,
                )
            if name == "profile_promote":
                self._proposals.create(
                    request,
                    profile_id=scope.profile_id,
                    scope_id=scope.scope_id,
                    turn_id=scope.run_id,
                    session_id=scope.session_id,
                )
                return self._success(
                    ToolOperation.PROMOTE,
                    ToolResultStatus.PROPOSAL_CREATED,
                )
            return self._update(request, scope)
        except PrivateDuplicateReviewRequired:
            return self._failure(ToolResultStatus.REVIEW_REQUIRED)
        except (ProposalQuotaExceeded, ProposalLimitExceededError):
            return self._failure(ToolResultStatus.QUOTA_EXCEEDED)
        except (ProfileConflictError, ProfileKeyCollisionError):
            return self._failure(ToolResultStatus.CONFLICT)
        except PersonalContextAuthorityError as exc:
            return self._failure(
                ToolResultStatus.PROFILE_LOCKED
                if exc.reason_code == "profile_locked"
                else ToolResultStatus.REVIEW_REQUIRED
                if exc.reason_code == "record_ineligible"
                else ToolResultStatus.PERMISSION_DENIED
            )
        except Exception:
            return self._failure(ToolResultStatus.REVIEW_REQUIRED)

    @staticmethod
    def _proposal_evidence(request, scope: ProfileToolRunScope):
        evidence_span = request.evidence_span
        if evidence_span is None:
            return None, None
        if (
            scope.current_user_message_id is None
            or scope.current_user_text is None
            or evidence_span not in scope.current_user_text
        ):
            return None, None
        return (
            scope.current_user_message_id,
            hashlib.sha256(evidence_span.encode("utf-8")).hexdigest(),
        )

    def _eligible_records(self):
        scope = self._scope
        canonical_scope = next(
            item
            for item in self._service.list_scopes()
            if item.scope_id == scope.scope_id
        )
        view = self._service.authorized_context_view(
            active_workspace_scope_id=(
                scope.scope_id if canonical_scope.kind.value == "workspace" else None
            )
        )
        now = self._service.clock()
        conflicted = set(view.conflicted_record_ids)
        return tuple(
            record
            for record in view.records
            if record.state is RecordState.ACTIVE
            and (record.expires_at is None or record.expires_at > now)
            and record.controls.agent_visibility is AgentVisibility.AGENT_VISIBLE
            and record.record_id not in conflicted
        )

    def _search(self, request: ProfileSearchRequest) -> ToolResult:
        query = request.query.casefold()
        matches = [
            record
            for record in self._eligible_records()
            if query
            in json.dumps(record.model_dump(mode="json"), sort_keys=True).casefold()
        ][: request.limit]
        return self._success(
            ToolOperation.SEARCH,
            ToolResultStatus.APPLIED,
            {"records": [record.model_dump(mode="json") for record in matches]},
        )

    def _get(self, request: ProfileGetRequest) -> ToolResult:
        record = next(
            (
                record
                for record in self._eligible_records()
                if record.record_id == request.record_id
            ),
            None,
        )
        if record is None:
            return self._failure(ToolResultStatus.PERMISSION_DENIED)
        return self._success(
            ToolOperation.GET,
            ToolResultStatus.APPLIED,
            {"record": record.model_dump(mode="json")},
        )

    def _update(
        self, request: ProfileUpdateRequest, scope: ProfileToolRunScope
    ) -> ToolResult:
        if (
            request.current_user_message_id != scope.current_user_message_id
            or scope.current_user_text is None
            or request.evidence_span not in scope.current_user_text
        ):
            return self._failure(ToolResultStatus.REVIEW_REQUIRED)
        evidence_hash = hashlib.sha256(
            request.evidence_span.encode("utf-8")
        ).hexdigest()
        record = self._proposals.apply_direct_update(
            request,
            profile_id=scope.profile_id,
            scope_id=scope.scope_id,
            evidence_hash=evidence_hash,
        )
        return self._success(
            ToolOperation.UPDATE,
            ToolResultStatus.APPLIED,
            {"record": record.model_dump(mode="json")},
        )

    @staticmethod
    def _failure(status: ToolResultStatus) -> ToolResult:
        return ToolResult(ok=False, error=status.value)

    @staticmethod
    def _success(
        operation: ToolOperation,
        status: ToolResultStatus,
        data: dict | None = None,
    ) -> ToolResult:
        result = ProfileToolResult(
            operation=operation,
            status=status,
            message=_MESSAGES[status],
        ).model_dump(mode="json")
        if data is not None:
            result["data"] = data
        return ToolResult(
            ok=True,
            content=json.dumps(result, sort_keys=True, separators=(",", ":")),
        )


__all__ = ["ProfileToolProvider", "ProfileToolRunScope"]
