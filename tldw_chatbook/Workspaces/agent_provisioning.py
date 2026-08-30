"""Convenience auto-create of a workspace's default agent persona + profile.

Task-8 of the workspace-assistant-defaults plan: every explicit workspace
gets a reference-backed default assistant (a local persona named
``"<workspace> Agent"`` plus an MCP permission profile ``ws-<workspace_id>``)
without the user wiring anything by hand. Provisioning is strictly
best-effort: any failure logs a warning and yields ``None`` so neither
``create_workspace`` nor the startup backfill can fail because of it.
"""

from __future__ import annotations

from typing import Protocol

from loguru import logger

from .models import (
    DEFAULT_WORKSPACE_ID,
    WorkspaceAssistantDefaults,
    WorkspaceRecord,
)

SEED_SYSTEM_PROMPT = (
    "You are the default assistant for the \"{name}\" workspace. "
    "Help the user with work in this workspace; be direct and grounded in "
    "workspace sources when they are provided."
)


class _PersonaServiceLike(Protocol):
    """Minimal surface the provisioner needs from a persona service."""

    def create_persona_profile(self, payload: dict) -> dict:  # pragma: no cover
        ...


class _PermissionStoreLike(Protocol):
    """Minimal surface the provisioner needs from a permission store."""

    def ensure_profile(self, profile_id: str) -> None:  # pragma: no cover
        ...


class WorkspaceAgentProvisioner:
    """Convenience auto-create of a workspace's default agent.

    Creates the ``"<name> Agent"`` persona via the persona service, ensures
    the ``ws-<workspace_id>`` MCP permission profile exists, and returns the
    reference-backed ``WorkspaceAssistantDefaults`` tying the workspace to
    both. The built-in Default workspace is never provisioned -- it is the
    capability-free fallback, matching the backfill's skip rule.
    """

    def __init__(
        self,
        persona_service: _PersonaServiceLike,
        permission_store: _PermissionStoreLike,
    ) -> None:
        self._personas = persona_service
        self._permissions = permission_store

    def provision(self, workspace: WorkspaceRecord) -> WorkspaceAssistantDefaults | None:
        """Provision the workspace's default agent, or ``None`` on failure.

        Never raises: any exception (persona service down, store unwritable,
        malformed response) is logged with the workspace id and swallowed.

        Args:
            workspace: The workspace record to provision for.

        Returns:
            The defaults referencing the created persona and permission
            profile, or ``None`` when provisioning failed or was skipped.
        """
        if workspace.workspace_id == DEFAULT_WORKSPACE_ID:
            return None
        try:
            record = self._personas.create_persona_profile(
                {
                    "name": f"{workspace.name} Agent",
                    "description": f"Default agent persona for workspace {workspace.name}.",
                    "system_prompt": SEED_SYSTEM_PROMPT.format(name=workspace.name),
                    "mode": "session_scoped",
                    "is_active": True,
                }
            )
            profile_id = f"ws-{workspace.workspace_id}"
            self._permissions.ensure_profile(profile_id)
            return WorkspaceAssistantDefaults(
                assistant_id=str(record["id"]), tool_policy_profile_id=profile_id
            )
        except Exception:
            logger.opt(exception=True).warning("Workspace agent provisioning failed")
            return None


class _BackfillDBLike(Protocol):
    """Minimal DB surface the backfill needs (WorkspaceDB backfill flags)."""

    def is_agent_backfill_complete(self) -> bool:  # pragma: no cover
        ...

    def mark_agent_backfill_complete(self) -> None:  # pragma: no cover
        ...


class _BackfillRegistryLike(Protocol):
    """Minimal registry surface the backfill consumes."""

    @property
    def db(self) -> _BackfillDBLike:  # pragma: no cover
        ...

    def list_workspaces(  # pragma: no cover
        self, *, include_archived: bool = False
    ) -> tuple[WorkspaceRecord, ...]: ...

    def set_assistant_defaults(  # pragma: no cover
        self,
        workspace_id: str,
        defaults: WorkspaceAssistantDefaults,
        *,
        confirm_read_write: bool = False,
    ) -> WorkspaceRecord: ...


def run_workspace_agent_backfill(
    *,
    registry: _BackfillRegistryLike,
    provisioner: WorkspaceAgentProvisioner,
) -> int:
    """Provision agents for pre-existing workspaces, once per database.

    Iterates explicit non-archived non-Default workspaces whose
    ``assistant_defaults`` are still NULL, provisions each, and marks the
    backfill complete via ``registry.db.mark_agent_backfill_complete()``.
    Idempotent: once the completion flag is set, re-runs return 0. The
    completion flag is only set when every eligible workspace either
    already had defaults or was provisioned and persisted successfully;
    any failed attempt leaves the flag unset so the next startup retries
    (successful updates are skipped idempotently).

    Args:
        registry: The workspace registry to read/update.
        provisioner: The provisioner creating persona + permission profile.

    Returns:
        The number of workspaces provisioned by this run.
    """
    if registry.db.is_agent_backfill_complete():
        return 0
    count = 0
    failed = False
    for record in registry.list_workspaces():
        if record.archived or record.workspace_id == DEFAULT_WORKSPACE_ID:
            continue
        if record.assistant_defaults is not None:
            continue
        defaults = provisioner.provision(record)
        if defaults is None:
            failed = True
            continue
        try:
            registry.set_assistant_defaults(
                record.workspace_id, defaults, confirm_read_write=True
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Workspace agent backfill could not persist defaults"
            )
            failed = True
            continue
        count += 1
    if failed:
        logger.warning(
            "Workspace agent backfill had failures; completion flag left unset "
            "so the next startup retries"
        )
        return count
    try:
        registry.db.mark_agent_backfill_complete()
    except Exception:
        logger.opt(exception=True).warning(
            "Workspace agent backfill completion flag could not be stored"
        )
    return count
