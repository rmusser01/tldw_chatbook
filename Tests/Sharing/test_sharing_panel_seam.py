"""Sharing_Panel operation names resolve on the real Interop scope service.

Deletion-wave review finding (task-34896): the panel duck-calls operations
by name through ``_run_operation`` (it does
``getattr(self.scope_service, operation_name)(mode="server", **kwargs)``),
and its link/token section still used the pre-Interop twin's names
(create_share_token / list_share_tokens / revoke_share_token /
preview_public_share / verify_public_share_password / import_public_share)
while SharingScopeService exposes the *_link / *_public_link names -- in
server mode every one of those operations would have raised AttributeError.
This test pins every operation name the panel passes against the REAL scope
service, mirroring Tests/Outputs/test_outputs_panel_seam.py, so name drift
between the panel and the Interop API cannot silently regress the seam.
"""

from __future__ import annotations

from typing import Any

import pytest

from tldw_chatbook.Sharing_Interop.sharing_scope_service import SharingScopeService


class _RecordingServerService:
    """Fake server service mirroring the real one's return shapes.

    Every Interop operation records and returns a generic shape; the
    assertions care about NAME resolution and kwarg acceptance, not
    per-method behavior (the dedicated Interop suites pin those).
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def __getattr__(self, name: str):
        async def record(*args: Any, **kwargs: Any) -> dict[str, Any]:
            self.calls.append((name, args, kwargs))
            if name.startswith("list"):
                return {"items": [], "tokens": [], "total": 0}
            return {"id": 7}

        return record


class _AllowAllPolicyEnforcer:
    def require_allowed(self, *, action_id: str) -> None:
        return None


# (operation name the panel passes, kwargs the panel builds for it) -- one
# entry per _run_operation call site in UI/Sharing_Panel.py.
PANEL_OPERATIONS: list[tuple[str, dict[str, Any]]] = [
    (
        "create_link",
        {
            "resource_type": "workspace",
            "resource_id": "ws-1",
            "access_level": "view_chat",
            "allow_clone": False,
            "password": None,
            "max_uses": None,
            "expires_at": None,
        },
    ),
    ("list_links", {}),
    ("revoke_link", {"token_id": 1}),
    ("inspect_public_link", {"token": "tok-1"}),
    ("verify_public_link_password", {"token": "tok-1", "password": "pw"}),
    ("import_public_link", {"token": "tok-1"}),
    # Workspace-share side (already Interop-named; pinned so it stays that
    # way -- these were verified correct when the six link ops were fixed).
    (
        "share_workspace",
        {
            "workspace_id": "ws-1",
            "share_scope_type": "team",
            "share_scope_id": 1,
            "access_level": "view_chat",
            "allow_clone": False,
        },
    ),
    ("list_workspace_shares", {"workspace_id": "ws-1", "include_revoked": False}),
    (
        "update_share",
        {"share_id": 1, "access_level": "view_chat", "allow_clone": False},
    ),
    ("revoke_share", {"share_id": 1}),
    ("list_shared_with_me", {}),
    ("get_shared_workspace", {"share_id": 1}),
    (
        "clone_shared_workspace",
        {"share_id": 1, "new_name": "Clone", "idempotency_key": "k-1"},
    ),
    ("list_shared_workspace_sources", {"share_id": 1}),
    ("get_shared_workspace_media", {"share_id": 1, "media_id": 1}),
    (
        "chat_with_shared_workspace",
        {"share_id": 1, "query": "summary?", "top_k": 5},
    ),
]


@pytest.mark.parametrize(("operation_name", "panel_kwargs"), PANEL_OPERATIONS)
@pytest.mark.asyncio
async def test_panel_operation_name_resolves_on_the_interop_scope_service(
    operation_name, panel_kwargs
):
    server = _RecordingServerService()
    scope = SharingScopeService(
        server_service=server,
        policy_enforcer=_AllowAllPolicyEnforcer(),
    )

    # Exactly what Sharing_Panel._run_operation does.
    operation = getattr(scope, operation_name)
    result = await operation(mode="server", **panel_kwargs)

    assert isinstance(result, dict)
    assert server.calls and server.calls[-1][0] == operation_name


def test_seam_list_is_bound_to_the_panel_call_sites():
    """Review hardening: the PANEL_OPERATIONS list must be exactly the set
    of operation names the panel actually passes to _run_operation -- a
    hand-copied list would stay green while a panel edit reintroduced the
    very drift this suite exists to catch (same guard as the Outputs seam
    test)."""
    import ast
    from pathlib import Path

    panel_source = Path("tldw_chatbook/UI/Sharing_Panel.py").read_text(encoding="utf-8")
    tree = ast.parse(panel_source)
    called: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_run_operation"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            called.add(node.args[1].value)

    listed = {name for name, _kwargs in PANEL_OPERATIONS}
    assert called == listed, (
        f"panel call sites and seam-test list diverged: "
        f"only-in-panel={sorted(called - listed)} "
        f"only-in-test={sorted(listed - called)}"
    )
    for name in sorted(called):
        assert hasattr(SharingScopeService, name), name
