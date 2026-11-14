"""Exact run-captured project-index scope preparation for Console."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from .console_chat_controller import ConsoleChatController


def prepare_project_index_scopes(self: ConsoleChatController, admitted_roots):
    """Retire registry handles opened by scope preparation, including guards."""
    from tldw_chatbook.Workspaces.project_index.runtime import get_project_index_service

    try:
        service = get_project_index_service(self.app)
        with service.registry_connection_scope():
            return _prepare_project_index_scopes_owned(self, admitted_roots)
    except Exception:
        logger.debug("Project index preparation unavailable")
        return None, {}


def _prepare_project_index_scopes_owned(self: ConsoleChatController, admitted_roots):
    """Read compatible immutable scopes off-loop after exact run capture."""
    from types import MappingProxyType
    from tldw_chatbook.Tools.remote_root_types import is_remote, local_root_path
    from tldw_chatbook.Workspaces.project_index.runtime import (
        get_project_index_service,
        project_index_key,
    )

    scopes = {}
    service = None
    for authority in admitted_roots:
        if is_remote(authority.root):
            continue
        try:
            if not authority.guard(False):
                continue
            root = local_root_path(
                authority.root, site="project index scope preparation"
            )
            key = project_index_key(
                self.app, authority.workspace_id, authority.binding_id, root
            )
            service = get_project_index_service(self.app)
            status = service.status(key)
            compatible = status.state in {"Ready", "Partial"}
            if status.state == "Building" and status.indexed > 0:
                from tldw_chatbook.Utils.sensitive_paths import SensitiveExclusion

                paths = (
                    authority.exclusions_provider()
                    if authority.exclusions_provider
                    else ()
                )
                floor = tuple(
                    SensitiveExclusion(
                        "subtree", Path(path).relative_to(root).as_posix()
                    )
                    for path in paths
                )
                compatible = service.building_coverage(status.scope, floor)
            if compatible and authority.guard(False):
                scopes[authority.alias] = status.scope
        except Exception:  # unavailable optional capability grants no authority
            logger.debug("Project index unavailable for this run binding")
    return service, MappingProxyType(scopes)
