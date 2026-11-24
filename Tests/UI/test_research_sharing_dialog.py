"""Mounted Research sharing uses scoped Server actions and explicit confirmation."""

import pytest
from textual.widgets import Button, Checkbox, Input, Select, Static

from Tests.app_module_patches import patch_app_global
from Tests.UI.library_quit_guard_support import _settings_without_splash, _until
from Tests.UI.test_research_chat_region import build_research_app, open_chat
from tldw_chatbook.Research_Workspace.contracts import QualifiedWorkspaceRef
from tldw_chatbook.Sharing_Interop.clone_receipts import SharingCloneReceipts
from tldw_chatbook.UI.Research_Workspace_Modules.sharing_dialog import (
    ResearchSharingDialog,
)


pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]
CLONE_OPERATION = "d290f1ee-6c54-4b01-90e6-d701748f0851"


class _SharingOwner:
    def __init__(self):
        self.calls = []
        self.active_shares = []
        self.tokens = []
        self.share_available = True

    async def list_workspace_shares(self, workspace_id, *, mode):
        assert workspace_id == "shared-ws" and mode == "server"
        return {"shares": list(self.active_shares)}

    async def share_workspace(self, **kwargs):
        self.calls.append(("share", kwargs))
        created = {"id": 3, **kwargs}
        self.active_shares.append(created)
        return created

    async def update_share(self, share_id, *, mode, **updates):
        self.calls.append(("update", share_id, mode, updates))
        self.active_shares[0].update(updates)
        return self.active_shares[0]

    async def create_link(self, **fields):
        self.calls.append(("create-link", fields))
        self.tokens.append({"id": 8, "token_prefix": "new-link", **fields})
        return {"raw_token": "new-link-fixture"}

    async def list_links(self, *, mode):
        assert mode == "server"
        return {"tokens": self.tokens}

    async def revoke_link(self, token_id, *, mode):
        self.calls.append(("revoke-link", token_id, mode))
        self.tokens.clear()
        return {"revoked": True}

    async def list_shared_with_me(self, *, mode):
        assert mode == "server"
        if not self.share_available:
            return {"items": []}
        return {
            "items": [
                {
                    "share_id": 27,
                    "workspace_id": "alpha",
                    "workspace_name": "Reading",
                    "allow_clone": True,
                }
            ]
        }

    async def get_shared_workspace(self, share_id, *, mode):
        assert share_id == 27 and mode == "server"
        return {
            "share": {
                "id": 27,
                "workspace_id": "alpha",
                "access_level": "view_chat",
                "allow_clone": True,
            }
        }

    async def clone_shared_workspace(self, share_id, *, mode, idempotency_key):
        self.calls.append(("clone", share_id, mode, idempotency_key))
        return {"operation_id": CLONE_OPERATION, "status": "pending"}

    async def get_shared_workspace_clone_operation(
        self, share_id, operation_id, *, mode
    ):
        assert (share_id, operation_id, mode) == (27, CLONE_OPERATION, "server")
        return {"operation_id": operation_id, "status": "completed"}

    async def inspect_public_link(self, token, *, mode):
        assert token == "protected-token" and mode == "server"
        return {
            "resource_type": "workspace",
            "resource_name": "Databases 101",
            "is_password_protected": True,
            "access_level": "view_chat",
        }

    async def verify_public_link_password(self, token, *, mode, password):
        self.calls.append(("verify", token, mode, password))
        return {"verified": True}

    async def import_public_link(self, token, *, mode, password):
        self.calls.append(("import", token, mode, password))
        return {"resource_type": "workspace", "resource_id": "imported-ws"}


async def _open_sharing(app, pilot, owner, *, receipts=None):
    await open_chat(app, pilot)
    ref = QualifiedWorkspaceRef("server", "shared-ws", "server-profile", "account")
    dialog = ResearchSharingDialog(
        ref,
        owner,
        current=lambda: True,
        server_base_url="https://sharing-fixture.invalid",
        clone_receipts=receipts,
    )
    await app.push_screen(dialog)
    await _until(
        pilot, lambda: bool(dialog.query("#research-sharing-section")), "sharing form"
    )
    return dialog


async def test_sharing_requires_confirmation_and_uses_exact_server_workspace(
    tmp_path, monkeypatch
):
    app, _, _ = build_research_app(tmp_path, monkeypatch)
    owner = _SharingOwner()
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(60, 20)) as pilot:
            dialog = await _open_sharing(app, pilot, owner)
            dialog.query_one("#research-share-scope", Select).value = "team"
            dialog.query_one("#research-share-target", Input).value = "17"
            dialog.query_one("#research-share-access", Select).value = "view_chat_add"
            dialog.query_one("#research-share-clone", Checkbox).value = True
            dialog.query_one("#research-share-create", Button).press()
            await _until(
                pilot,
                lambda: bool(app.screen.query("#confirm-button")),
                "share confirmation",
            )
            assert owner.calls == []
            app.screen.query_one("#confirm-button", Button).press()
            await _until(pilot, lambda: bool(owner.calls), "qualified Server share")
            assert owner.calls == [
                (
                    "share",
                    {
                        "mode": "server",
                        "workspace_id": "shared-ws",
                        "share_scope_type": "team",
                        "share_scope_id": 17,
                        "access_level": "view_chat_add",
                        "allow_clone": True,
                    },
                )
            ]

            shares = dialog.query_one("#research-share-existing", Select)
            await _until(
                pilot,
                lambda: (
                    "access granted"
                    in str(
                        dialog.query_one("#research-sharing-status", Static).render()
                    )
                ),
                "active share loaded",
            )
            shares.value = 3
            await pilot.pause()
            dialog.query_one("#research-share-access", Select).value = "view_chat"
            dialog.query_one("#research-share-clone", Checkbox).value = False
            dialog.query_one("#research-share-update", Button).press()
            await _until(
                pilot,
                lambda: bool(app.screen.query("#confirm-button")),
                "updated access confirmation",
            )
            assert len(owner.calls) == 1
            app.screen.query_one("#confirm-button", Button).press()
            await _until(pilot, lambda: len(owner.calls) == 2, "updated access saved")
            assert owner.calls[-1] == (
                "update",
                3,
                "server",
                {"access_level": "view_chat", "allow_clone": False},
            )


async def test_protected_link_preview_verify_and_import(tmp_path, monkeypatch):
    app, _, _ = build_research_app(tmp_path, monkeypatch)
    owner = _SharingOwner()
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(80, 24)) as pilot:
            dialog = await _open_sharing(app, pilot, owner)
            dialog.query_one("#research-sharing-section", Select).value = "import"
            dialog.query_one(
                "#research-import-token", Input
            ).value = "https://sharing-fixture.invalid/share/protected-token"
            dialog.query_one("#research-import-preview", Button).press()
            status = dialog.query_one("#research-sharing-status", Static)
            await _until(
                pilot,
                lambda: "Databases 101" in str(status.render()),
                "Server link preview",
            )
            importer = dialog.query_one("#research-import-apply", Button)
            assert importer.disabled
            dialog.query_one(
                "#research-import-password", Input
            ).value = "example-password"
            verifier = dialog.query_one("#research-import-verify", Button)
            await _until(pilot, lambda: not verifier.disabled, "password entry ready")
            verifier.press()
            await _until(
                pilot, lambda: not importer.disabled, "verified protected link"
            )
            assert owner.calls == [
                ("verify", "protected-token", "server", "example-password")
            ]
            importer.press()
            await _until(
                pilot,
                lambda: bool(app.screen.query("#confirm-button")),
                "import confirmation",
            )
            app.screen.query_one("#confirm-button", Button).press()
            await _until(
                pilot, lambda: len(owner.calls) == 2, "protected link imported"
            )
            assert owner.calls[-1] == (
                "import",
                "protected-token",
                "server",
                "example-password",
            )


async def test_share_link_constraints_copy_and_revoke(tmp_path, monkeypatch):
    app, _, _ = build_research_app(tmp_path, monkeypatch)
    owner = _SharingOwner()
    clipboard = []
    monkeypatch.setattr(app, "copy_to_clipboard", clipboard.append)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(100, 30)) as pilot:
            dialog = await _open_sharing(app, pilot, owner)
            dialog.query_one("#research-sharing-section", Select).value = "links"
            await _until(
                pilot,
                lambda: dialog.query_one("#research-sharing-links").display,
                "link options",
            )
            dialog.query_one("#research-link-access", Select).value = "full_edit"
            dialog.query_one("#research-link-limit", Input).value = "9"
            dialog.query_one(
                "#research-link-expiry", Input
            ).value = "2090-01-01T12:00:00Z"
            dialog.query_one("#research-link-password", Input).value = "demo-link-pass"
            dialog.query_one("#research-link-create", Button).press()
            await _until(
                pilot,
                lambda: bool(app.screen.query("#confirm-button")),
                "link confirmation",
            )
            assert owner.calls == []
            app.screen.query_one("#confirm-button", Button).press()
            button = dialog.query_one("#research-link-copy", Button)
            await _until(pilot, lambda: not button.disabled, "share link issued")
            assert owner.calls == [
                (
                    "create-link",
                    {
                        "mode": "server",
                        "resource_type": "workspace",
                        "resource_id": "shared-ws",
                        "access_level": "full_edit",
                        "allow_clone": False,
                        "password": "demo-link-pass",
                        "expires_at": "2090-01-01T12:00:00+00:00",
                        "max_uses": 9,
                    },
                )
            ]
            button.press()
            await _until(pilot, lambda: bool(clipboard), "link copied")
            assert clipboard == [
                "https://sharing-fixture.invalid/share/new-link-fixture"
            ]
            dialog.query_one("#research-link-list", Button).press()
            status = dialog.query_one("#research-sharing-status", Static)
            await _until(
                pilot, lambda: "1 active" in str(status.render()), "active link loaded"
            )
            dialog.query_one("#research-link-existing", Select).value = 8
            dialog.query_one("#research-link-revoke", Button).press()
            await _until(
                pilot,
                lambda: bool(app.screen.query("#confirm-button")),
                "revoke confirmation",
            )
            app.screen.query_one("#confirm-button", Button).press()
            await _until(
                pilot,
                lambda: "Workspace link revoked." in str(status.render()),
                "link revoked",
            )
            assert owner.calls[-1] == ("revoke-link", 8, "server")


async def test_received_workspace_clone_retains_its_operation(tmp_path, monkeypatch):
    app, _, _ = build_research_app(tmp_path, monkeypatch)
    owner = _SharingOwner()
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(84, 24)) as pilot:
            path = tmp_path / "sharing"
            receipts = SharingCloneReceipts(path, "server-profile", "account")
            dialog = await _open_sharing(app, pilot, owner, receipts=receipts)
            dialog.query_one("#research-sharing-section", Select).value = "received"
            await _until(
                pilot,
                lambda: dialog.query_one("#research-sharing-received").display,
                "received workspaces",
            )
            dialog.query_one("#research-received-list", Button).press()
            status = dialog.query_one("#research-sharing-status", Static)
            await _until(
                pilot,
                lambda: "1 workspaces shared" in str(status.render()),
                "shared workspace loaded",
            )
            dialog.query_one("#research-received-select", Select).value = 27
            dialog.query_one("#research-received-preview", Button).press()
            await _until(
                pilot,
                lambda: "Cloning allowed" in str(status.render()),
                "workspace preview",
            )
            dialog.query_one("#research-received-clone", Button).press()
            await _until(
                pilot,
                lambda: bool(app.screen.query("#confirm-button")),
                "clone confirmation",
            )
            assert not owner.calls
            app.screen.query_one("#confirm-button", Button).press()
            await _until(
                pilot,
                lambda: "Clone · pending" in str(status.render()),
                "clone admitted",
            )
            assert owner.calls[0][:3] == ("clone", 27, "server")
            assert receipts.get(27).operation_id == CLONE_OPERATION
            dialog.query_one("#research-sharing-close", Button).press()
            await _until(pilot, lambda: app.screen is not dialog, "sharing closed")
            owner.share_available = False
            reopened = SharingCloneReceipts(path, "server-profile", "account")
            dialog = await _open_sharing(app, pilot, owner, receipts=reopened)
            dialog.query_one("#research-sharing-section", Select).value = "received"
            await _until(
                pilot,
                lambda: dialog.query_one("#research-sharing-received").display,
                "received workspaces reopened",
            )
            dialog.query_one("#research-received-list", Button).press()
            status = dialog.query_one("#research-sharing-status", Static)
            await _until(
                pilot,
                lambda: "0 workspaces shared" in str(status.render()),
                "retained clone loaded after revocation",
            )
            dialog.query_one("#research-received-select", Select).value = 27
            dialog.query_one("#research-received-status", Button).press()
            await _until(
                pilot,
                lambda: "Clone · completed" in str(status.render()),
                "clone settled",
            )
            assert len(owner.calls) == 1


async def test_share_actions_can_receive_focus_across_terminal_sizes(
    tmp_path, monkeypatch
):
    app, _, _ = build_research_app(tmp_path, monkeypatch)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(160, 40)) as pilot:
            dialog = await _open_sharing(app, pilot, _SharingOwner())
            control = dialog.query_one("#research-share-create", Button)
            for width, height in (
                (160, 40),
                (120, 30),
                (100, 30),
                (84, 24),
                (80, 24),
                (60, 20),
            ):
                await pilot.resize_terminal(width, height)
                control.focus()
                control.scroll_visible(animate=False, immediate=True)
                await pilot.pause()
                assert app.focused is control
                assert 0 <= control.region.y < height
                assert control.region.x >= 0 and control.region.right <= width
