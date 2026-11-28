"""Protected Workspace imports send the password to the qualified Server."""

import json

import httpx
import pytest

from tldw_chatbook.Sharing_Interop.server_sharing_service import ServerSharingService
from tldw_chatbook.Sharing_Interop.sharing_scope_service import SharingScopeService
from tldw_chatbook.tldw_api.client import TLDWAPIClient


@pytest.mark.asyncio
async def test_protected_workspace_link_verify_and_import_use_same_transport():
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-API-KEY"] == "fixture-account"
        assert json.loads(request.content) == {"password": "fixture-password"}
        if request.url.path == "/api/v1/sharing/public/sample-token/verify":
            return httpx.Response(200, json={"verified": True})
        assert request.url.path == "/api/v1/sharing/public/sample-token/import"
        return httpx.Response(
            200,
            json={
                "resource_type": "workspace",
                "resource_id": "shared-workspace",
                "access_level": "view_chat",
                "owner_user_id": 4,
            },
        )

    url = "https://sharing-fixture.invalid"
    root = TLDWAPIClient(url, token="fixture-account")
    transport = httpx.MockTransport(respond)
    async with httpx.AsyncClient(
        transport=transport, base_url=url, headers={"X-API-KEY": "fixture-account"}
    ) as http:
        root._client = http
        scope = SharingScopeService(server_service=ServerSharingService(root))
        verified = await scope.verify_public_link_password(
            "sample-token", mode="server", password="fixture-password"
        )
        assert verified["verified"] is True
        imported = await scope.import_public_link(
            "sample-token", mode="server", password="fixture-password"
        )
        assert imported["resource_id"] == "shared-workspace"
