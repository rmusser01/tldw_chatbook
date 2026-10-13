"""Outputs_Panel operation names resolve on the real Interop scope service.

The panel duck-calls operations by name through ``_run_operation`` (it does
``getattr(self.scope_service, operation_name)(mode="server", **kwargs)``).
The panel originally targeted the pre-Interop twin's method names; that twin
was deleted in task-34892 and the seam was repointed at
``OutputsScopeService`` -- whose API is ``list_templates``/``create_artifact``
rather than ``list_output_templates``/``create_output``. This test pins every
operation name the panel passes against the REAL scope service so name drift
between the panel and the Interop API cannot silently regress the seam again
(review finding on the deletion wave, 2026-10-12).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tldw_chatbook.Outputs_Interop.outputs_scope_service import OutputsScopeService


class _RecordingServerService:
    """Fake server service mirroring the real one's return shapes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def _record(self, op: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((op, args, kwargs))
        if op.startswith("list"):
            return {"items": [{"id": 5}], "total": 1}
        return {"id": 5}

    async def list_templates(self, **kwargs: Any) -> dict[str, Any]:
        return self._record("list_templates", **kwargs)

    async def get_template(self, template_id: int) -> dict[str, Any]:
        return self._record("get_template", template_id)

    async def create_template(self, **kwargs: Any) -> dict[str, Any]:
        return self._record("create_template", **kwargs)

    async def update_template(self, template_id: int, **payload: Any) -> dict[str, Any]:
        return self._record("update_template", template_id, **payload)

    async def delete_template(self, template_id: int) -> dict[str, Any]:
        return self._record("delete_template", template_id)

    async def preview_template(self, template_id: int, **kwargs: Any) -> dict[str, Any]:
        return self._record("preview_template", template_id, **kwargs)

    async def list_artifacts(self, **kwargs: Any) -> dict[str, Any]:
        return self._record("list_artifacts", **kwargs)

    async def list_deleted_artifacts(self, **kwargs: Any) -> dict[str, Any]:
        return self._record("list_deleted_artifacts", **kwargs)

    async def get_artifact(self, output_id: int) -> dict[str, Any]:
        return self._record("get_artifact", output_id)

    async def create_artifact(self, **kwargs: Any) -> dict[str, Any]:
        return self._record("create_artifact", **kwargs)

    async def update_artifact(self, output_id: int, **payload: Any) -> dict[str, Any]:
        return self._record("update_artifact", output_id, **payload)

    async def delete_artifact(
        self, output_id: int, *, hard: bool = False, delete_file: bool = False
    ) -> dict[str, Any]:
        return self._record(
            "delete_artifact", output_id, hard=hard, delete_file=delete_file
        )


class _AllowAllPolicyEnforcer:
    def require_allowed(self, *, action_id: str) -> None:
        return None


# (operation name the panel passes, kwargs the panel builds for it) -- one
# entry per _run_operation call site in UI/Outputs_Panel.py.
PANEL_OPERATIONS: list[tuple[str, dict[str, Any]]] = [
    (
        "list_templates",
        {"q": None, "limit": 50, "offset": 0},
    ),
    (
        "create_template",
        {
            "name": "Newsletter",
            "type": "newsletter_markdown",
            "format": "md",
            "body": "# Hello",
            "description": None,
            "is_default": False,
        },
    ),
    ("get_template", {"template_id": 1}),
    (
        "update_template",
        {
            "template_id": 1,
            "name": None,
            "type": "newsletter_markdown",
            "format": "md",
            "body": None,
            "description": None,
            "is_default": False,
        },
    ),
    ("delete_template", {"template_id": 1}),
    (
        "preview_template",
        {"template_id": 1, "item_ids": [1, 2], "limit": 50},
    ),
    (
        "list_artifacts",
        {"page": 1, "size": 50, "run_id": 2, "type": "digest", "workspace_tag": "w"},
    ),
    ("list_deleted_artifacts", {"page": 1, "size": 50}),
    ("get_artifact", {"output_id": 1}),
    (
        "create_artifact",
        {
            "template_id": 1,
            "item_ids": [1],
            "title": None,
            "workspace_tag": None,
            "ingest_to_media_db": False,
        },
    ),
    ("update_artifact", {"output_id": 1, "title": "Digest", "format": "md"}),
    (
        "delete_artifact",
        {"output_id": 1, "hard": False, "delete_file": False},
    ),
]


@pytest.mark.parametrize(("operation_name", "panel_kwargs"), PANEL_OPERATIONS)
@pytest.mark.asyncio
async def test_panel_operation_name_resolves_on_the_interop_scope_service(
    operation_name, panel_kwargs
):
    server = _RecordingServerService()
    scope = OutputsScopeService(
        local_service=None,
        server_service=server,
        policy_enforcer=_AllowAllPolicyEnforcer(),
    )

    # Exactly what Outputs_Panel._run_operation does.
    operation = getattr(scope, operation_name)
    result = await operation(mode="server", **panel_kwargs)

    assert isinstance(result, dict)
    assert server.calls and server.calls[-1][0] == operation_name
