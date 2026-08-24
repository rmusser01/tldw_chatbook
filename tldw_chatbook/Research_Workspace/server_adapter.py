"""Fail-closed Server adapter for Research Workspace lifecycle operations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeVar
from uuid import uuid4

from tldw_chatbook.Notes.server_notes_workspace_service import (
    ServerNotesWorkspaceService,
)
from tldw_chatbook.runtime_policy.server_context import ServerContextError
from tldw_chatbook.runtime_policy.server_event_scope import (
    event_principal_id_from_active_context,
)
from tldw_chatbook.runtime_policy.types import PolicyDeniedError
from tldw_chatbook.tldw_api.exceptions import (
    APIConnectionError,
    APIResponseError,
    AuthenticationError,
)

from .contracts import (
    CapabilityUnavailableError,
    QualifiedWorkspaceRef,
    ResearchCapability,
    ResearchWorkspaceSummary,
    WorkspaceDataSource,
    require_capability,
)


_AUDITED_SERVICE_METHODS = {
    "list": "list_workspaces",
    "get": "list_workspaces",
    "create": "save_workspace",
    "update": "save_workspace",
    "duplicate": "save_workspace",
    "archive": "save_workspace",
    "restore": "save_workspace",
    "delete": "delete_workspace",
}
_AUDITED_CAPABILITY_REVISION = "server-notes-workspace-service-v1"
_RECOVERY_BY_REASON = {
    "server_not_configured": "Configure a server.",
    "server_profile_missing": "Choose or configure a server profile.",
    "server_unavailable": "Retry or change the selected server.",
    "auth_required": "Reauthenticate with the selected server.",
    "stale_authorization": "Reauthenticate with the selected server.",
    "server_credentials_unavailable": "Restore server credentials and retry.",
    "credential_store_unavailable": "Restore secure credential storage and retry.",
}
_ServerResult = TypeVar("_ServerResult")


class ServerResearchWorkspaceAdapter:
    """Use only the selected server context and server workspace service."""

    def __init__(
        self,
        service: ServerNotesWorkspaceService,
        server_context_provider: Any,
        *,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._service = service
        self._context_provider = server_context_provider
        self._id_factory = id_factory or (lambda: f"workspace-{uuid4().hex}")

    async def list_workspaces(
        self, *, include_archived: bool = False
    ) -> tuple[ResearchWorkspaceSummary, ...]:
        context, profile_id, principal_id = self._active_identity()
        require_capability(self._capabilities_for_context(context), "list")
        rows = await self._server_call(
            self._service.list_workspaces(), context=context
        )
        summaries = tuple(
            self._summary(row, profile_id=profile_id, principal_id=principal_id)
            for row in rows
        )
        if include_archived:
            return summaries
        return tuple(summary for summary in summaries if not summary.archived)

    async def get_workspace(
        self, ref: QualifiedWorkspaceRef
    ) -> ResearchWorkspaceSummary | None:
        self._require_server_ref(ref)
        context = self._context_for_ref(ref)
        require_capability(self._capabilities_for_context(context), "get")
        rows = await self._server_call(
            self._service.list_workspaces(), context=context
        )
        for row in rows:
            if str(row.get("id") or "").strip() == ref.workspace_id:
                return self._matching_summary(ref, row)
        return None

    async def create_workspace(
        self, *, name: str, description: str = "", template_id: str = ""
    ) -> ResearchWorkspaceSummary:
        context, profile_id, principal_id = self._active_identity()
        require_capability(self._capabilities_for_context(context), "create")
        if description.strip() or template_id.strip():
            capability = ResearchCapability(
                available=False,
                reason_code="server_field_unavailable",
                user_message=(
                    "The selected server cannot create workspace descriptions or templates."
                ),
                owner="server",
                recovery_action="Create a workspace with a name only.",
                capability_revision=self._capability_revision(context),
            )
            raise CapabilityUnavailableError(capability)
        row = await self._server_call(
            self._service.save_workspace(
                workspace_id=self._id_factory(), name=name
            ),
            context=context,
        )
        return self._summary(row, profile_id=profile_id, principal_id=principal_id)

    async def update_workspace(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        name: str | None = None,
        expected_version: int | None = None,
    ) -> ResearchWorkspaceSummary:
        context = self._context_for_ref(ref)
        require_capability(self._capabilities_for_context(context), "update")
        version = self._require_version(expected_version, context)
        row = await self._server_call(
            self._service.save_workspace(
                workspace_id=ref.workspace_id, name=name, version=version
            ),
            context=context,
        )
        return self._matching_summary(ref, row)

    async def duplicate_workspace(
        self, ref: QualifiedWorkspaceRef, *, name: str
    ) -> ResearchWorkspaceSummary:
        context = self._context_for_ref(ref)
        require_capability(self._capabilities_for_context(context), "duplicate")
        row = await self._server_call(
            self._service.save_workspace(
                workspace_id=self._id_factory(), name=name
            ),
            context=context,
        )
        return self._summary(
            row,
            profile_id=ref.server_profile_id,
            principal_id=ref.principal_id,
        )

    async def archive_workspace(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        expected_version: int | None = None,
    ) -> ResearchWorkspaceSummary:
        return await self._set_archived(
            ref, archived=True, expected_version=expected_version
        )

    async def restore_workspace(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        expected_version: int | None = None,
    ) -> ResearchWorkspaceSummary:
        return await self._set_archived(
            ref, archived=False, expected_version=expected_version
        )

    async def delete_workspace(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        expected_version: int | None = None,
    ) -> bool:
        context = self._context_for_ref(ref)
        require_capability(self._capabilities_for_context(context), "delete")
        if expected_version is not None:
            raise CapabilityUnavailableError(
                ResearchCapability(
                    available=False,
                    reason_code="version_precondition_unavailable",
                    user_message=(
                        "The selected server cannot enforce a delete version precondition."
                    ),
                    owner="server",
                    recovery_action="Reload the workspace and delete without a version.",
                    capability_revision=self._capability_revision(context),
                )
            )
        result = await self._server_call(
            self._service.delete_workspace(ref.workspace_id), context=context
        )
        if isinstance(result, Mapping):
            return bool(result.get("deleted", True))
        return True

    async def capabilities(
        self, ref: QualifiedWorkspaceRef
    ) -> Mapping[str, ResearchCapability]:
        context = self._context_for_ref(ref)
        return self._capabilities_for_context(context)

    async def _set_archived(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        archived: bool,
        expected_version: int | None,
    ) -> ResearchWorkspaceSummary:
        operation = "archive" if archived else "restore"
        context = self._context_for_ref(ref)
        require_capability(self._capabilities_for_context(context), operation)
        version = self._require_version(expected_version, context)
        row = await self._server_call(
            self._service.save_workspace(
                workspace_id=ref.workspace_id,
                archived=archived,
                version=version,
            ),
            context=context,
        )
        return self._matching_summary(ref, row)

    def _active_identity(self) -> tuple[Any, str, str]:
        try:
            context = self._context_provider.get_active_context()
        except ServerContextError as exc:
            raise CapabilityUnavailableError(
                self._context_failure_capability(exc)
            ) from exc
        profile_id = str(getattr(context, "active_server_id", "") or "").strip()
        if not profile_id:
            raise CapabilityUnavailableError(
                ResearchCapability(
                    available=False,
                    reason_code="server_profile_missing",
                    user_message="Active server profile is unavailable.",
                    owner="server",
                    recovery_action="Choose or configure a server profile.",
                )
            )
        principal_id = event_principal_id_from_active_context(context) or ""
        return context, profile_id, principal_id

    def _context_for_ref(self, ref: QualifiedWorkspaceRef) -> Any:
        self._require_server_ref(ref)
        context, profile_id, principal_id = self._active_identity()
        if profile_id != ref.server_profile_id or principal_id != ref.principal_id:
            raise CapabilityUnavailableError(
                ResearchCapability(
                    available=False,
                    reason_code="server_context_changed",
                    user_message="The selected server identity changed.",
                    owner="server",
                    recovery_action="Reload the selected server workspace.",
                    capability_revision=self._capability_revision(context),
                )
            )
        return context

    def _capabilities_for_context(
        self, context: Any
    ) -> Mapping[str, ResearchCapability]:
        revision = self._capability_revision(context)
        unavailable = self._context_health_unavailable(context, revision=revision)
        result: dict[str, ResearchCapability] = {}
        for operation, service_method in _AUDITED_SERVICE_METHODS.items():
            if unavailable is not None:
                result[operation] = unavailable
            elif callable(getattr(self._service, service_method, None)):
                result[operation] = ResearchCapability(
                    available=True,
                    reason_code="available",
                    user_message="Available on the selected server.",
                    owner="server",
                    capability_revision=revision,
                )
            else:
                result[operation] = ResearchCapability(
                    available=False,
                    reason_code="server_capability_unavailable",
                    user_message=(
                        f"The selected server service cannot perform workspace {operation}."
                    ),
                    owner="server",
                    recovery_action="Choose another action or update the server service.",
                    capability_revision=revision,
                )
        return result

    @staticmethod
    def _context_health_unavailable(
        context: Any, *, revision: str
    ) -> ResearchCapability | None:
        capabilities = getattr(context, "capabilities", {})
        if not isinstance(capabilities, Mapping):
            capabilities = {}
        if capabilities.get("server_configured") is False:
            return ResearchCapability(
                available=False,
                reason_code="server_not_configured",
                user_message="A server is not configured.",
                owner="server",
                recovery_action="Configure a server.",
                capability_revision=revision,
            )
        if capabilities.get("reachability") == "unreachable":
            return ResearchCapability(
                available=False,
                reason_code="server_unavailable",
                user_message="The selected server is unavailable.",
                owner="server",
                recovery_action="Retry or change the selected server.",
                capability_revision=revision,
            )
        auth_state = capabilities.get("auth_state")
        if auth_state in {"auth_required", "session_invalid"}:
            stale = auth_state == "session_invalid"
            return ResearchCapability(
                available=False,
                reason_code="stale_authorization" if stale else "auth_required",
                user_message=(
                    "Authorization with the selected server is stale."
                    if stale
                    else "Authentication with the selected server is required."
                ),
                owner="server",
                recovery_action="Reauthenticate with the selected server.",
                capability_revision=revision,
            )
        return None

    async def _server_call(
        self, operation: Awaitable[_ServerResult], *, context: Any
    ) -> _ServerResult:
        try:
            return await operation
        except ServerContextError as exc:
            raise CapabilityUnavailableError(
                self._context_failure_capability(exc)
            ) from exc
        except AuthenticationError as exc:
            raise CapabilityUnavailableError(
                ResearchCapability(
                    available=False,
                    reason_code="auth_required",
                    user_message="Authentication with the selected server is required.",
                    owner="server",
                    recovery_action="Reauthenticate with the selected server.",
                    capability_revision=self._capability_revision(context),
                )
            ) from exc
        except PolicyDeniedError as exc:
            raise CapabilityUnavailableError(
                ResearchCapability(
                    available=False,
                    reason_code=exc.reason_code,
                    user_message=exc.user_message,
                    owner=exc.authority_owner,
                    recovery_action="Review server permissions and retry.",
                    capability_revision=self._capability_revision(context),
                )
            ) from exc
        except APIResponseError as exc:
            permission_denied = exc.status_code == 403
            capability_missing = exc.status_code in {404, 405, 501}
            raise CapabilityUnavailableError(
                ResearchCapability(
                    available=False,
                    reason_code=(
                        "server_permission_denied"
                        if permission_denied
                        else (
                            "server_capability_unavailable"
                            if capability_missing
                            else "server_request_failed"
                        )
                    ),
                    user_message=(
                        "The selected server denied this workspace action."
                        if permission_denied
                        else (
                            "The selected server does not expose this workspace action."
                            if capability_missing
                            else "The selected server could not complete this action."
                        )
                    ),
                    owner="server",
                    recovery_action=(
                        "Review server permissions and retry."
                        if permission_denied
                        else (
                            "Update the selected server or choose another action."
                            if capability_missing
                            else "Retry or review server diagnostics."
                        )
                    ),
                    capability_revision=self._capability_revision(context),
                )
            ) from exc
        except (APIConnectionError, ConnectionError, OSError, TimeoutError) as exc:
            raise CapabilityUnavailableError(
                ResearchCapability(
                    available=False,
                    reason_code="server_unavailable",
                    user_message="The selected server is unavailable.",
                    owner="server",
                    recovery_action="Retry or change the selected server.",
                    capability_revision=self._capability_revision(context),
                )
            ) from exc

    @staticmethod
    def _capability_revision(context: Any) -> str:
        capabilities = getattr(context, "capabilities", {})
        if not isinstance(capabilities, Mapping):
            capabilities = {}
        reachability = str(capabilities.get("reachability") or "unknown").strip()
        auth_state = str(capabilities.get("auth_state") or "unknown").strip()
        return f"{_AUDITED_CAPABILITY_REVISION}:{reachability}:{auth_state}"

    @staticmethod
    def _context_failure_capability(exc: ServerContextError) -> ResearchCapability:
        reason_code = str(getattr(exc, "reason_code", "server_unavailable"))
        return ResearchCapability(
            available=False,
            reason_code=reason_code,
            user_message=str(exc),
            owner="server",
            recovery_action=_RECOVERY_BY_REASON.get(
                reason_code, "Retry or change the selected server."
            ),
        )

    @staticmethod
    def _require_server_ref(ref: QualifiedWorkspaceRef) -> None:
        if ref.data_source is not WorkspaceDataSource.SERVER:
            raise ValueError("Server adapter requires a Server workspace ref")

    @staticmethod
    def _require_version(expected_version: int | None, context: Any) -> int:
        if type(expected_version) is int and expected_version >= 0:
            return expected_version
        raise CapabilityUnavailableError(
            ResearchCapability(
                available=False,
                reason_code="version_required",
                user_message="Reload this server workspace before changing it.",
                owner="server",
                recovery_action="Reload the workspace and retry.",
                capability_revision=ServerResearchWorkspaceAdapter._capability_revision(
                    context
                ),
            )
        )

    @staticmethod
    def _summary(
        row: Mapping[str, Any], *, profile_id: str, principal_id: str
    ) -> ResearchWorkspaceSummary:
        workspace_id = str(row.get("id") or "").strip()
        ref = QualifiedWorkspaceRef(
            WorkspaceDataSource.SERVER,
            workspace_id,
            server_profile_id=profile_id,
            principal_id=principal_id,
        )
        version_value = row.get("version")
        version = (
            int(version_value)
            if type(version_value) in {int, str}
            and str(version_value).strip().isdigit()
            else None
        )
        return ResearchWorkspaceSummary(
            ref=ref,
            name=str(row.get("name") or "").strip(),
            description=str(row.get("description") or "").strip(),
            archived=bool(row.get("archived", False)),
            version=version,
            updated_at=str(row.get("updated_at") or "").strip(),
        )

    def _matching_summary(
        self, expected_ref: QualifiedWorkspaceRef, row: Mapping[str, Any]
    ) -> ResearchWorkspaceSummary:
        summary = self._summary(
            row,
            profile_id=expected_ref.server_profile_id,
            principal_id=expected_ref.principal_id,
        )
        if summary.ref != expected_ref:
            raise ValueError("Adapter returned a mismatched workspace ref")
        return summary
