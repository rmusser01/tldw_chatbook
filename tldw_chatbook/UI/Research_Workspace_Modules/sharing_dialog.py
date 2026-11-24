"""Manage workspace sharing on one captured Server profile and account."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from urllib.parse import urlsplit
from uuid import uuid4

from loguru import logger
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Select, Static

from ...Research_Workspace.contracts import (
    CapabilityUnavailableError,
    QualifiedWorkspaceRef,
    WorkspaceDataSource,
)
from ...Widgets.confirmation_dialog import ConfirmationDialog
from ...Widgets.modal_dismissal import SafeModalDismissMixin
from ...runtime_policy.types import PolicyDeniedError
from ...tldw_api.exceptions import APIResponseError
from ...tldw_api.sharing_schemas import (
    AccessLevel,
    CreateTokenRequest,
    ShareScopeType,
    ShareWorkspaceRequest,
)


_PERMISSIONS = {
    AccessLevel.VIEW_CHAT: "View and chat",
    AccessLevel.VIEW_CHAT_ADD: "View, chat, and add",
    AccessLevel.FULL_EDIT: "Full edit",
}
_DEVICE_DISCLOSURE = "Device-only folders, annotations, pane preferences, and drafts stay on this device."
_SECTIONS = (
    ("Workspace access", "access"),
    ("Share links", "links"),
    ("Shared with me", "received"),
    ("Import a link", "import"),
)


def _number(raw: str, *, label: str, maximum: int) -> int:
    if not raw.isascii() or not raw.isdecimal() or len(raw) > len(str(maximum)):
        raise ValueError(f"Enter a valid {label}.")
    value = int(raw)
    if not 1 <= value <= maximum:
        raise ValueError(f"Enter a valid {label}.")
    return value


@dataclass(frozen=True, slots=True)
class WorkspaceShareDraft:
    """A reviewed form to carry across Copy without submitting a share."""

    section: str
    access_level: AccessLevel = AccessLevel.VIEW_CHAT
    allow_clone: bool = False
    scope_type: ShareScopeType = ShareScopeType.TEAM
    scope_id: int | None = None
    password: str = field(default="", repr=False)
    expires_at: str = ""
    max_uses: int | None = None


class ResearchSharingDialog(
    SafeModalDismissMixin, ModalScreen[WorkspaceShareDraft | None]
):
    """Prepare or confirm Server shares under the original workspace authority."""

    BINDINGS = [Binding("escape", "request_safe_cancel", "Close", show=False)]
    SAFE_MODAL_CONTENT = "#research-sharing-dialog"

    def __init__(
        self,
        ref: QualifiedWorkspaceRef,
        scope,
        *,
        current,
        draft=None,
        prepare_only=False,
        server_base_url: str = "",
        clone_receipts=None,
    ) -> None:
        super().__init__()
        if ref.data_source is not WorkspaceDataSource.SERVER:
            raise ValueError("Sharing requires a Server workspace.")
        self.ref = ref
        self.scope = scope
        self.current = current
        self.draft = draft
        self.prepare_only = prepare_only
        self.server_base_url = server_base_url.rstrip("/")
        self.clone_receipts = clone_receipts
        self._busy = False
        self._workspace_shares = {}
        self._links = {}
        self._received = {}
        self._clones = {}
        self._clone_keys = {}
        self._public_preview = None
        self._verified = None
        self._generated_link = ""

    def compose(self) -> ComposeResult:
        sections = _SECTIONS[:2] if self.prepare_only else _SECTIONS
        section = self.draft.section if self.draft else "access"
        with Vertical(
            id="research-sharing-dialog",
            classes="research-transfer-dialog research-transfer-scroll-dialog",
        ):
            yield Static(
                "Share settings for this Copy"
                if self.prepare_only
                else f"Share · {self.ref.workspace_id}",
                classes="research-pane-title",
                markup=False,
            )
            yield Select(
                sections,
                value=section,
                allow_blank=False,
                id="research-sharing-section",
            )
            with VerticalScroll(classes="research-transfer-scroll"):
                with Vertical(
                    id="research-sharing-access", classes="research-share-page"
                ):
                    yield from self._compose_access()
                with Vertical(
                    id="research-sharing-links", classes="research-share-page"
                ):
                    yield from self._compose_links()
                if not self.prepare_only:
                    with Vertical(
                        id="research-sharing-received", classes="research-share-page"
                    ):
                        yield from self._compose_received()
                    with Vertical(
                        id="research-sharing-import", classes="research-share-page"
                    ):
                        yield from self._compose_import()
            yield Static("", id="research-sharing-status", markup=False)
            yield Button("Close", id="research-sharing-close", compact=True)

    def _settings(
        self, prefix: str, draft: WorkspaceShareDraft | None
    ) -> ComposeResult:
        yield Static("Access level", markup=False)
        yield Select(
            [(label, key.value) for key, label in _PERMISSIONS.items()],
            value=(draft.access_level if draft else AccessLevel.VIEW_CHAT).value,
            allow_blank=False,
            id=f"research-{prefix}-access",
        )
        yield Checkbox(
            "Allow cloning",
            value=bool(draft and draft.allow_clone),
            id=f"research-{prefix}-clone",
        )

    def _compose_access(self) -> ComposeResult:
        draft = self.draft if self.draft and self.draft.section == "access" else None
        yield Static("Share with", markup=False)
        yield Select(
            [("Team", "team"), ("Organization", "org")],
            value=draft.scope_type.value if draft else "team",
            allow_blank=False,
            id="research-share-scope",
        )
        yield Input(
            str(draft.scope_id) if draft and draft.scope_id else "",
            placeholder="Team or organization ID",
            id="research-share-target",
        )
        yield from self._settings("share", draft)
        yield Button(
            "Save settings" if self.prepare_only else "Review share",
            id="research-share-create",
            compact=True,
            variant="primary",
        )
        if not self.prepare_only:
            yield Button(
                "Refresh active shares", id="research-share-list", compact=True
            )
            yield Select([], prompt="Active shares", id="research-share-existing")
            yield Button(
                "Review access changes", id="research-share-update", compact=True
            )
            yield Button(
                "Revoke selected share", id="research-share-revoke", compact=True
            )

    def _compose_links(self) -> ComposeResult:
        draft = self.draft if self.draft and self.draft.section == "links" else None
        yield from self._settings("link", draft)
        yield Input(
            draft.password if draft else "",
            placeholder="Password (optional)",
            password=True,
            id="research-link-password",
        )
        yield Input(
            draft.expires_at if draft else "",
            placeholder="Expiry with timezone (optional)",
            id="research-link-expiry",
        )
        yield Input(
            str(draft.max_uses) if draft and draft.max_uses else "",
            placeholder="Maximum uses (optional)",
            id="research-link-limit",
        )
        yield Button(
            "Save settings" if self.prepare_only else "Review share link",
            id="research-link-create",
            compact=True,
            variant="primary",
        )
        if not self.prepare_only:
            yield Static("", id="research-link-result", markup=False)
            yield Button(
                "Copy link", id="research-link-copy", compact=True, disabled=True
            )
            yield Button("Refresh active links", id="research-link-list", compact=True)
            yield Select([], prompt="Active links", id="research-link-existing")
            yield Button(
                "Revoke selected link", id="research-link-revoke", compact=True
            )

    def _compose_received(self) -> ComposeResult:
        yield Button(
            "Refresh shared workspaces", id="research-received-list", compact=True
        )
        yield Select([], prompt="Shared workspaces", id="research-received-select")
        yield Button("Preview workspace", id="research-received-preview", compact=True)
        yield Button("Review clone", id="research-received-clone", compact=True)
        yield Button("Check clone status", id="research-received-status", compact=True)

    def _compose_import(self) -> ComposeResult:
        yield Input(placeholder="Workspace link token", id="research-import-token")
        yield Button("Preview link", id="research-import-preview", compact=True)
        yield Input(
            placeholder="Link password",
            password=True,
            id="research-import-password",
            disabled=True,
        )
        yield Button(
            "Verify password", id="research-import-verify", compact=True, disabled=True
        )
        yield Button(
            "Review import",
            id="research-import-apply",
            compact=True,
            disabled=True,
            variant="primary",
        )

    def on_mount(self) -> None:
        self._sync_section()

    @on(Select.Changed, "#research-sharing-section")
    def _sync_section(self) -> None:
        current = self.query_one("#research-sharing-section", Select).value
        for _, name in _SECTIONS:
            pages = self.query(f"#research-sharing-{name}")
            page = pages.first(Vertical) if pages else None
            if page is not None:
                page.display = name == current

    def _text(self, name: str, *, strip: bool = True) -> str:
        value = self.query_one(f"#{name}", Input).value
        return value.strip() if strip else value

    def _status(self, message: str) -> None:
        if self.is_mounted:
            self.query_one("#research-sharing-status", Static).update(message)

    def _require_current(self) -> None:
        if not self.is_mounted or not self.current():
            raise ValueError("The workspace or Server account changed. Reopen Sharing.")

    async def _call(self, name: str, *args, **kwargs):
        self._require_current()
        method = getattr(self.scope, name, None)
        if not callable(method):
            raise ValueError(
                "This Server sharing action is unavailable. Check the Server profile and permissions."
            )
        result = await method(*args, mode="server", **kwargs)
        self._require_current()
        return result

    async def _confirm(self, title: str, message: str, label: str) -> bool:
        self._require_current()
        result = await self.app.push_screen_wait(
            ConfirmationDialog(
                title=title, message=message, confirm_label=label, cancel_label="Cancel"
            )
        )
        self._require_current()
        return result is True

    def _read_draft(self, section: str) -> WorkspaceShareDraft:
        prefix = "share" if section == "access" else "link"
        access_level = AccessLevel(
            self.query_one(f"#research-{prefix}-access", Select).value
        )
        allow_clone = self.query_one(f"#research-{prefix}-clone", Checkbox).value
        if section == "access":
            scope = ShareScopeType(
                self.query_one("#research-share-scope", Select).value
            )
            target = _number(
                self._text("research-share-target"),
                label="team or organization ID",
                maximum=2**63 - 1,
            )
            return WorkspaceShareDraft(
                section, access_level, allow_clone, scope_type=scope, scope_id=target
            )
        password = self._text("research-link-password", strip=False)
        if password and not 4 <= len(password) <= 128:
            raise ValueError("A link password must have 4 to 128 characters.")
        expires_at = self._text("research-link-expiry")
        if expires_at:
            try:
                expiry = datetime.fromisoformat(expires_at)
            except ValueError:
                raise ValueError(
                    "Enter an expiry such as 2026-12-31T23:59:00Z."
                ) from None
            if expiry.tzinfo is None or expiry <= datetime.now(UTC):
                raise ValueError("Enter a future expiry with a timezone.")
            expires_at = expiry.astimezone(UTC).isoformat()
        raw_uses = self._text("research-link-limit")
        max_uses = (
            _number(raw_uses, label="use limit", maximum=10000) if raw_uses else None
        )
        return WorkspaceShareDraft(
            section,
            access_level,
            allow_clone,
            password=password,
            expires_at=expires_at,
            max_uses=max_uses,
        )

    async def _create(self, section: str) -> None:
        draft = self._read_draft(section)
        if self.prepare_only:
            self.dismiss(draft)
            return
        permission = _PERMISSIONS[draft.access_level]
        clone = "Cloning allowed." if draft.allow_clone else "Cloning disabled."
        if section == "access":
            target = (
                "team" if draft.scope_type is ShareScopeType.TEAM else "organization"
            )
            if not await self._confirm(
                "Grant workspace access?",
                f"{self.ref.workspace_id} · {target} {draft.scope_id} · {permission}. {clone} {_DEVICE_DISCLOSURE}",
                "Grant access",
            ):
                return
            fields = ShareWorkspaceRequest(
                share_scope_type=draft.scope_type,
                share_scope_id=draft.scope_id,
                access_level=draft.access_level,
                allow_clone=draft.allow_clone,
            )
            await self._call(
                "share_workspace",
                workspace_id=self.ref.workspace_id,
                **fields.model_dump(mode="json"),
            )
            await self._list_access()
            self._status("Workspace access granted.")
            return
        protection = (
            "Password required."
            if draft.password
            else "Anyone with the token can use it."
        )
        limits = f"Expires {draft.expires_at or 'on revocation'} · {draft.max_uses or 'unlimited'} uses."
        if not await self._confirm(
            "Create workspace share link?",
            f"{self.ref.workspace_id} · {permission}. {clone} {protection} {limits} {_DEVICE_DISCLOSURE}",
            "Create link",
        ):
            return
        request = CreateTokenRequest(
            resource_type="workspace",
            resource_id=self.ref.workspace_id,
            access_level=draft.access_level,
            allow_clone=draft.allow_clone,
            password=draft.password or None,
            expires_at=draft.expires_at or None,
            max_uses=draft.max_uses,
        )
        result = await self._call(
            "create_link", **request.model_dump(mode="json", exclude_none=True)
        )
        self.query_one("#research-link-password", Input).value = ""
        raw_token = str(result.get("raw_token") or "")
        self._generated_link = (
            f"{self.server_base_url}/share/{raw_token}"
            if self.server_base_url and raw_token
            else raw_token
        )
        self.query_one(
            "#research-link-copy", Button
        ).disabled = not self._generated_link
        self.query_one("#research-link-result", Static).update(
            self._generated_link or "Link created. Refresh active links to manage it."
        )
        self._status("Share link created. Copy the link to save it.")

    def _fill(self, selector: str, rows, *, key: str, label) -> dict:
        records = {
            row[key]: row for row in rows if type(row.get(key)) is int and row[key] > 0
        }
        self.query_one(selector, Select).set_options(
            (Text(label(row)), identifier) for identifier, row in records.items()
        )
        return records

    async def _list_access(self) -> None:
        result = await self._call("list_workspace_shares", self.ref.workspace_id)
        self._workspace_shares = self._fill(
            "#research-share-existing",
            (
                row
                for row in result["shares"]
                if row.get("workspace_id") == self.ref.workspace_id
                and not row.get("is_revoked")
            ),
            key="id",
            label=lambda row: (
                f"{row['share_scope_type']} {row['share_scope_id']} · {row['access_level']}"
            ),
        )
        self._status(f"{len(self._workspace_shares)} active workspace shares.")

    async def _list_links(self) -> None:
        result = await self._call("list_links")
        self._links = self._fill(
            "#research-link-existing",
            (
                row
                for row in result["tokens"]
                if row.get("resource_type") == "workspace"
                and row.get("resource_id") == self.ref.workspace_id
                and not row.get("is_revoked")
            ),
            key="id",
            label=lambda row: (
                f"{row.get('token_prefix') or 'Link'} · {row.get('access_level')}"
            ),
        )
        self._status(f"{len(self._links)} active workspace links.")

    def _selected_id(self, selector: str, choices) -> int:
        selected = self.query_one(selector, Select).value
        if type(selected) is not int or selected not in choices:
            raise ValueError("Select an item from the Server list.")
        return selected

    @on(Select.Changed, "#research-share-existing")
    def _load_selected_access(self, event: Select.Changed) -> None:
        row = self._workspace_shares.get(event.value)
        if row and row.get("access_level") in _PERMISSIONS:
            self.query_one("#research-share-access", Select).value = row["access_level"]
            self.query_one("#research-share-clone", Checkbox).value = bool(
                row.get("allow_clone")
            )

    async def _update_access(self) -> None:
        selected = self._selected_id("#research-share-existing", self._workspace_shares)
        level = AccessLevel(self.query_one("#research-share-access", Select).value)
        clone = self.query_one("#research-share-clone", Checkbox).value
        target = self._workspace_shares[selected]
        details = (
            f"{self.ref.workspace_id} · {target['share_scope_type']} {target['share_scope_id']} "
            f"· {_PERMISSIONS[level]}. "
            + ("Cloning allowed." if clone else "Cloning disabled.")
        )
        if await self._confirm("Update workspace access?", details, "Update access"):
            await self._call(
                "update_share", selected, access_level=level.value, allow_clone=clone
            )
            await self._list_access()
            self._status("Workspace access updated.")

    async def _revoke(self, *, link: bool) -> None:
        selector = "#research-link-existing" if link else "#research-share-existing"
        selected = self._selected_id(
            selector, self._links if link else self._workspace_shares
        )
        noun = "link" if link else "share"
        if not await self._confirm(
            f"Revoke selected {noun}?",
            f"{self.ref.workspace_id} · {noun} {selected}",
            "Revoke",
        ):
            return
        await self._call("revoke_link" if link else "revoke_share", selected)
        await (self._list_links() if link else self._list_access())
        self._status(f"Workspace {noun} revoked.")

    async def _list_received(self) -> None:
        result = await self._call("list_shared_with_me")
        workspaces = list(result["items"])
        rows = list(workspaces)
        if self.clone_receipts is not None:
            available = {item["share_id"] for item in workspaces}
            saved = await asyncio.to_thread(self.clone_receipts.list_receipts)
            rows.extend(
                {
                    "share_id": share_id,
                    "workspace_name": f"Clone receipt for share {share_id}",
                    "receipt_only": True,
                }
                for share_id in sorted(saved.keys() - available)
            )
        if not self.current():
            return
        self._received = self._fill(
            "#research-received-select",
            rows,
            key="share_id",
            label=lambda row: str(row.get("workspace_name") or row["workspace_id"]),
        )
        self._status(f"{len(workspaces)} workspaces shared with this account.")

    async def _received_action(self, action: str) -> None:
        selected = self._selected_id("#research-received-select", self._received)
        item = self._received[selected]
        name = item.get("workspace_name") or item["workspace_id"]
        if action == "preview":
            if item.get("receipt_only"):
                raise ValueError(
                    "This share is unavailable. Check its retained Clone status."
                )
            reply = await self._call("get_shared_workspace", selected)
            share = reply.get("share") or {}
            if (
                share.get("id") != selected
                or share.get("workspace_id") != item["workspace_id"]
            ):
                raise ValueError("The Server returned a different shared workspace.")
            clone = (
                "Cloning allowed." if share.get("allow_clone") else "Cloning disabled."
            )
            self._status(f"{name} · {share.get('access_level')} · {clone}")
            return
        stored = (
            await asyncio.to_thread(self.clone_receipts.get, selected)
            if self.clone_receipts is not None
            else None
        )
        if action == "clone":
            pending = stored is not None and not stored.operation_id
            if not item.get("allow_clone") and not pending:
                raise ValueError("The owner disabled cloning for this workspace.")
            if not await self._confirm("Clone shared workspace?", str(name), "Clone"):
                return
            if self.clone_receipts is not None:
                stored = await asyncio.to_thread(self.clone_receipts.prepare, selected)
                key = stored.idempotency_key
            else:
                key = self._clone_keys.setdefault(selected, str(uuid4()))
            result = await self._call(
                "clone_shared_workspace", selected, idempotency_key=key
            )
        else:
            operation_id = stored.operation_id if stored else self._clones.get(selected)
            if not operation_id:
                if stored is not None:
                    raise ValueError(
                        "The clone response was interrupted. Use Clone to recover the same operation."
                    )
                raise ValueError("Start a clone before checking its operation.")
            result = await self._call(
                "get_shared_workspace_clone_operation", selected, operation_id
            )
        confirmed = result.get("operation_id")
        if confirmed:
            if stored is not None:
                await asyncio.to_thread(
                    self.clone_receipts.remember,
                    selected,
                    stored.idempotency_key,
                    confirmed,
                )
            else:
                self._clones[selected] = confirmed
        elif self.clone_receipts is not None:
            raise ValueError(
                "The Server returned no durable clone receipt. Check its workspace list for the result."
            )
        status = str(result.get("status") or "pending").replace("_", " ")
        self._status(f"Clone · {status}")

    def _token(self) -> str:
        raw = self._text("research-import-token")
        token = raw
        if "://" in raw:
            pasted = urlsplit(raw)
            selected = urlsplit(self.server_base_url)
            if (
                not selected.hostname
                or (pasted.scheme, pasted.netloc) != (selected.scheme, selected.netloc)
                or not pasted.path.startswith(selected.path.rstrip("/") + "/share/")
            ):
                raise ValueError("Open Sharing on the Server named in the link.")
            token = pasted.path.rsplit("/", 1)[-1]
        if (
            not 1 <= len(token) <= 200
            or not token.isascii()
            or any(not (char.isalnum() or char in "_-") for char in token)
        ):
            raise ValueError("Enter a valid workspace link token.")
        return token

    @on(Input.Changed, "#research-import-token, #research-import-password")
    def _sync_import_controls(self) -> None:
        if self.prepare_only or not self.is_mounted:
            return
        try:
            token = self._token()
        except ValueError:
            token = ""
        preview = self._public_preview
        same = preview is not None and token == preview[0]
        protected = same and bool(preview[1].get("is_password_protected"))
        password = self._text("research-import-password", strip=False)
        self.query_one("#research-import-password", Input).disabled = not protected
        self.query_one("#research-import-verify", Button).disabled = (
            not protected or not password
        )
        verifier = (token, sha256(password.encode()).digest())
        self.query_one("#research-import-apply", Button).disabled = not same or (
            protected and verifier != self._verified
        )

    async def _public_action(self, action: str) -> None:
        token = self._token()
        if action == "preview":
            reply = await self._call("inspect_public_link", token)
            if reply.get("resource_type") != "workspace":
                raise ValueError("Choose a link for a workspace.")
            self._public_preview = (token, reply)
            self._verified = None
            self._sync_import_controls()
            access = reply.get("access_level") or "shared access"
            self._status(
                f"{reply.get('resource_name') or 'Shared workspace'} · {access}"
            )
            return
        if self._public_preview is None or self._public_preview[0] != token:
            raise ValueError("Preview the current workspace link.")
        preview = self._public_preview[1]
        password = (
            self._text("research-import-password", strip=False)
            if preview.get("is_password_protected")
            else None
        )
        if action == "verify":
            if not password:
                raise ValueError("Enter the link password.")
            verified = await self._call(
                "verify_public_link_password", token, password=password
            )
            self._verified = (
                (token, sha256(password.encode()).digest())
                if verified.get("verified") is True
                else None
            )
            self._sync_import_controls()
            self._status(
                "Password verified."
                if self._verified
                else "The link password was rejected."
            )
            return
        if (
            password is not None
            and (token, sha256(password.encode()).digest()) != self._verified
        ):
            raise ValueError("Verify the current link password.")
        if not await self._confirm(
            "Import this workspace link?",
            str(preview.get("resource_name") or "Shared workspace"),
            "Import",
        ):
            return
        result = await self._call("import_public_link", token, password=password)
        if result.get("resource_type") != "workspace":
            raise ValueError("The Server returned a different resource type.")
        self._public_preview = None
        self._verified = None
        self.query_one("#research-import-password", Input).value = ""
        self._sync_import_controls()
        self._status(f"Server granted access to workspace {result.get('resource_id')}.")

    async def _copy_created_link(self) -> None:
        self._require_current()
        if self._generated_link:
            self.app.copy_to_clipboard(self._generated_link)
            self._status("Share link copied.")

    @on(Button.Pressed)
    def share_action(self, event: Button.Pressed) -> None:
        event.stop()
        button = event.button.id
        if button == "research-sharing-close":
            self.dismiss(None)
            return
        if self._busy:
            return
        actions = {
            "research-share-create": (self._create, ("access",)),
            "research-link-create": (self._create, ("links",)),
            "research-share-list": (self._list_access, ()),
            "research-share-update": (self._update_access, ()),
            "research-link-list": (self._list_links, ()),
            "research-link-copy": (self._copy_created_link, ()),
            "research-received-list": (self._list_received, ()),
        }
        if button in actions:
            method, args = actions[button]
            action = method(*args)
        elif button in {"research-share-revoke", "research-link-revoke"}:
            action = self._revoke(link=button == "research-link-revoke")
        elif button and button.startswith("research-received-"):
            action = self._received_action(button.removeprefix("research-received-"))
        elif button and button.startswith("research-import-"):
            action = self._public_action(button.removeprefix("research-import-"))
        else:
            return
        self._busy = True
        self.run_worker(
            self._run_action(action), group="research-sharing", exit_on_error=False
        )

    async def _run_action(self, action) -> None:
        try:
            await action
        except PolicyDeniedError as exc:
            self._status(exc.user_message)
        except CapabilityUnavailableError as exc:
            self._status(
                f"{exc.capability.user_message} {exc.capability.recovery_action}".strip()
            )
        except APIResponseError as exc:
            self._status(
                f"The Server rejected this sharing action ({exc.status_code}). Refresh its permissions and try again."
            )
        except ValueError as exc:
            self._status(
                str(exc)
                if type(exc) is ValueError
                else "Check the sharing settings and try again."
            )
        except Exception as exc:
            logger.warning(
                "Workspace sharing failed (exception_category={})", type(exc).__name__
            )
            self._status(
                "Sharing could not finish. Refresh the Server list before retrying."
            )
        finally:
            self._busy = False
