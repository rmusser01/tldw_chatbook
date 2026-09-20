"""Keep the real Sharing panel's clone admission identity across uncertain replies."""

from types import SimpleNamespace

import pytest
from textual.app import App
from textual.widgets import Button, Input

from Tests.tldw_api.test_sharing_release_contracts import http_client, operation
from tldw_chatbook.Sharing import ServerSharingScopeService, ServerSharingService
from tldw_chatbook.UI.Sharing_Panel import SharingPanel


@pytest.mark.asyncio
async def test_panel_retry_and_remount_replay_same_clone_until_explicit_new_request():
    import httpx

    keys = []

    def handler(request):
        keys.append(request.headers.get("Idempotency-Key"))
        if len(keys) == 1:
            raise httpx.ReadTimeout("accepted response lost", request=request)
        return httpx.Response(202, json=operation())

    async with http_client(handler) as client:
        owner = SimpleNamespace(
            current_runtime_backend="server",
            server_context_provider=AuthorityProvider(),
            server_sharing_scope_service=ServerSharingScopeService(
                ServerSharingService(client)
            ),
        )

        class PanelApp(App):
            def compose(self):
                yield SharingPanel(owner)

        app = PanelApp()
        async with app.run_test(size=(160, 80)) as pilot:
            await app.workers.wait_for_complete()
            panel = app.query_one(SharingPanel)
            panel.query_one("#sharing-share-id", Input).value = "7"
            panel.query_one("#sharing-clone-name", Input).value = "Copy"
            panel.query_one("#sharing-clone-btn", Button).press()
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert len(keys) == 1
            await panel.remove()
            panel = SharingPanel(owner)
            await app.mount(panel)
            await app.workers.wait_for_complete()
            panel.query_one("#sharing-share-id", Input).value = "7"
            panel.query_one("#sharing-clone-name", Input).value = "Copy"
            panel.query_one("#sharing-clone-btn", Button).press()
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert len(keys) == 2
            assert keys[0] is not None
            assert keys[0] == keys[1]
            panel.query_one("#sharing-new-clone-btn", Button).press()
            await pilot.pause()
            panel.query_one("#sharing-clone-btn", Button).press()
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert len(keys) == 3
            assert keys[2] != keys[1]


class AuthorityProvider:
    """Model the provider's stable account identity independently of credentials."""

    def __init__(self):
        self.account = "alice"
        self.server = "server-a"

    def get_active_context(self):
        return SimpleNamespace(
            active_server_id=self.server, base_url=f"https://{self.server}"
        )

    def capture_character_authority_context(self, *, expected_server_id):
        return (expected_server_id, self.account)

    async def resolve_character_authority_id(
        self, *, expected_server_id, context_capture
    ):
        return f"{expected_server_id}:{context_capture[1]}"

    def is_character_authority_context_current(self, capture):
        return capture == (self.server, self.account)


@pytest.mark.asyncio
async def test_panel_clone_keys_isolate_authority_normalize_names_and_keep_quota_entries(
    monkeypatch,
):
    import httpx

    import tldw_chatbook.UI.Sharing_Panel as panel_module

    monkeypatch.setattr(panel_module, "MAX_RETAINED_CLONE_REQUESTS", 2, raising=False)
    calls = []

    def handler(request):
        calls.append((request.headers["Idempotency-Key"], request.content))
        return httpx.Response(202, json=operation())

    async with http_client(handler) as client:
        provider = AuthorityProvider()
        owner = SimpleNamespace(
            current_runtime_backend="server",
            server_context_provider=provider,
            server_sharing_scope_service=ServerSharingScopeService(
                ServerSharingService(client)
            ),
        )

        class PanelApp(App):
            def compose(self):
                yield SharingPanel(owner)

        app = PanelApp()
        async with app.run_test(size=(160, 80)) as pilot:
            await app.workers.wait_for_complete()
            panel = app.query_one(SharingPanel)
            panel.query_one("#sharing-share-id", Input).value = "7"
            panel.query_one("#sharing-clone-name", Input).value = "  My   Copy  "
            await panel.clone_shared_workspace()
            panel.query_one("#sharing-clone-name", Input).value = "My Copy"
            await panel.clone_shared_workspace()
            assert calls[0] == calls[1]
            provider.account = "bob"
            await panel.clone_shared_workspace()
            assert calls[2][0] != calls[0][0]
            panel.query_one("#sharing-new-clone-btn", Button).press()
            await pilot.pause()
            await panel.clone_shared_workspace()
            assert calls[3][0] != calls[2][0]
            provider.account = "alice"
            await panel.clone_shared_workspace()
            assert calls[4][0] == calls[0][0]
            provider.server = "server-b"
            await panel.clone_shared_workspace()
            assert len(calls) == 5
            provider.server = "server-a"
            await panel.clone_shared_workspace()
            assert calls[5][0] == calls[0][0]
            retained = dict(owner._sharing_clone_request_keys)
            panel.query_one("#sharing-share-id", Input).value = "invalid"
            panel.query_one("#sharing-new-clone-btn", Button).press()
            await pilot.pause()
            assert owner._sharing_clone_request_keys == retained
