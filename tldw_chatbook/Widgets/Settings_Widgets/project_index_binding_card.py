"""Explicit folder keyword-index controls; all authority and storage work is threaded."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button, Input, Static

from tldw_chatbook.Workspaces.project_index.lifetime import LifetimeBusy
from tldw_chatbook.Workspaces.project_index.models import (
    ProjectIndexPolicy,
    ProjectIndexStatus,
)
from tldw_chatbook.Workspaces.project_index.runtime import (
    get_project_index_service,
    project_index_key,
)

DEFAULT_POLICY = ProjectIndexPolicy((), ())


def index_status_copy(
    status: ProjectIndexStatus, requires_upgrade: bool = False
) -> str:
    """Format an immutable snapshot without looking up authority or storage."""
    scan = status.last_scan_at or "never"
    return (
        f"Keyword index: {'upgrade required' if requires_upgrade else status.state} · {status.discovered:,} discovered · "
        f"{status.indexed:,} indexed · ~{status.cache_bytes / 1024:.0f} KiB · "
        f"Last scan activity: {scan}"
    )


class ProjectIndexBindingCard(Vertical):
    """Manage exactly the folder row supplied by Settings, never a global selection."""

    class ActionRequested(Message):
        """A user intent captured before dispatch, including durable scope/job CAS."""

        def __init__(
            self,
            binding_id: str,
            workspace_id: str,
            status: ProjectIndexStatus,
            action: str,
            policy: ProjectIndexPolicy,
        ) -> None:
            super().__init__()
            self.binding_id = binding_id
            self.workspace_id = workspace_id
            self.expected_scope = status.scope
            self.job_id = status.job_id
            self.job_revision = status.job_revision
            self.action = action
            self.policy = policy

    def __init__(
        self,
        binding_id: str,
        *,
        workspace_id: str,
        root: str,
        status: ProjectIndexStatus | None = None,
        policy: ProjectIndexPolicy = DEFAULT_POLICY,
        available: bool = True,
        supported: bool = True,
    ) -> None:
        super().__init__(classes="h-auto")
        self.binding_id = binding_id
        self.workspace_id = workspace_id
        self.root = root
        self.status = status
        self.requires_upgrade = False
        self.policy = policy
        self.available = available
        self.supported = supported
        self._visit = 0
        self._request = 0
        self._pending = 0
        self._painted_choices = (
            "; ".join(p or "." for p in policy.include_prefixes),
            "; ".join(policy.text_suffixes),
        )

    def compose(self) -> ComposeResult:
        yield Static(
            "Keyword index: checking…",
            classes="settings-detail-row project-index-status",
            markup=False,
        )
        yield Static(
            "", classes="settings-detail-row project-index-feedback", markup=False
        )
        yield Static(
            "Included folders (semicolon-separated; blank means all)",
            classes="settings-detail-row",
        )
        yield Input(
            "; ".join(p or "." for p in self.policy.include_prefixes),
            placeholder="All folders; use . for the binding root",
            classes="settings-compact-input project-index-prefixes",
        )
        yield Static(
            "Text suffixes (semicolon-separated; blank means all)",
            classes="settings-detail-row",
        )
        yield Input(
            "; ".join(self.policy.text_suffixes),
            placeholder="All eligible UTF-8 text",
            classes="settings-compact-input project-index-suffixes",
        )
        # The Settings detail pane can be only 34 columns at 80x24.
        # One compact action per line remains fully painted at that width.
        for action in (
            "enable",
            "refresh",
            "pause",
            "resume",
            "apply",
            "disable",
            "clear",
            "reload",
        ):
            label = {"apply": "Apply choices", "reload": "Check status"}.get(
                action, action.title()
            )
            yield Button(
                label, name=action, classes="project-index-action", compact=True
            )

    def on_mount(self) -> None:
        self._visit += 1
        self._paint()
        self.call_after_refresh(self.reload_status)
        self.set_interval(2, self._poll)

    def on_unmount(self) -> None:
        self._visit += 1
        self._request += 1

    def _poll(self) -> None:
        if (
            self.status
            and self.status.state == "Building"
            and not self.requires_upgrade
            and not self._pending
        ):
            self.reload_status()

    def _paint(self) -> None:
        status = self.status
        if not self.supported:
            text = "Keyword index unavailable: SSH folders are not supported."
        elif not self.available:
            text = "Keyword index unavailable. Restore folder access in Settings."
        else:
            text = (
                index_status_copy(status, self.requires_upgrade)
                if status
                else "Keyword index: checking…"
            )
        self.query_one(".project-index-status", Static).update(text)
        state = status.state if status else None
        actions = {"reload"} if self.available and self.supported else set()
        if status and self.available and self.supported:
            actions |= {"clear"}
            if self.requires_upgrade:
                actions.add("enable" if state == "Off" else "refresh")
            elif state == "Off":
                actions.add("enable")
            else:
                actions.add("disable")
                if state == "Building":
                    actions.add("pause")
                else:
                    actions |= {"refresh", "apply"}
                    if state in {"Paused", "Interrupted"} and status.job_id:
                        actions.add("resume")
        for button in self.query(Button):
            button.display = button.name in actions
        for field in self.query(Input):
            field.disabled = (
                not self.available
                or not self.supported
                or (state == "Building" and not self.requires_upgrade)
            )

    def _feedback(self, text: str) -> None:
        self.query_one(".project-index-feedback", Static).update(text)

    def _load(self, app: object) -> tuple[ProjectIndexStatus, ProjectIndexPolicy, bool]:
        service = get_project_index_service(app)
        key = project_index_key(
            app, self.workspace_id, self.binding_id, Path(self.root)
        )
        return service.management_snapshot(key)

    def reload_status(self) -> None:
        if self.is_mounted and self.available and self.supported and not self._pending:
            self._launch(None)

    @on(Button.Pressed, ".project-index-action")
    def _pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.name == "reload":
            self.reload_status()
            return
        if self.status is None:
            return
        policy = self.policy
        if event.button.name in {"enable", "apply"}:
            try:
                prefixes = tuple(
                    p.strip()
                    for p in self.query_one(
                        ".project-index-prefixes", Input
                    ).value.split(";")
                    if p.strip()
                )
                suffixes = tuple(
                    p.strip()
                    for p in self.query_one(
                        ".project-index-suffixes", Input
                    ).value.split(";")
                    if p.strip()
                )
                policy = ProjectIndexPolicy(prefixes, suffixes)
            except ValueError:
                self._feedback(
                    "Use relative folders and suffixes starting with a dot. Separate choices with semicolons."
                )
                return
        self.post_message(
            self.ActionRequested(
                self.binding_id,
                self.workspace_id,
                self.status,
                event.button.name,
                policy,
            )
        )

    @on(ActionRequested)
    def _requested(self, event: ActionRequested) -> None:
        event.stop()
        if (event.binding_id, event.workspace_id) != (
            self.binding_id,
            self.workspace_id,
        ):
            return
        self._launch(event)

    def _execute(
        self, app: object, action: ActionRequested
    ) -> tuple[ProjectIndexStatus, ProjectIndexPolicy, bool]:
        # Never replace the captured scope with a fresh one when dispatch is delayed.
        if (action.binding_id, action.workspace_id) != (
            self.binding_id,
            self.workspace_id,
        ):
            raise ValueError("stale_scope")
        service = get_project_index_service(app)
        key = action.expected_scope.key
        kwargs = {"expected_scope": action.expected_scope}
        if action.action in {"enable", "apply"}:
            service.enable(key, action.policy, **kwargs)
        elif action.action in {"disable", "clear"}:
            getattr(service, action.action)(key, **kwargs)
        elif action.action == "refresh":
            service.refresh(key, expected_job_revision=action.job_revision, **kwargs)
        elif action.action in {"pause", "resume"}:
            getattr(service, action.action)(
                action.job_id, expected_job_revision=action.job_revision, **kwargs
            )
        else:
            raise ValueError("unknown_index_action")
        return service.management_snapshot(key)

    def _launch(self, action: ActionRequested | None) -> None:
        self._request += 1
        token = (self._visit, self._request, self.workspace_id, self.binding_id)
        self._pending += 1
        if action:
            self._feedback(f"{action.action.title()} requested…")
        app = self.app
        submitted_choices = tuple(field.value for field in self.query(Input))

        async def run():
            try:

                def owned_call():
                    service = get_project_index_service(app)
                    with service.registry_connection_scope():
                        return self._execute(app, action) if action else self._load(app)

                result = await asyncio.to_thread(owned_call)
                error = None
            except Exception as exc:
                result, error = None, exc
            finally:
                self._pending -= 1
            if not self.is_mounted or token != (
                self._visit,
                self._request,
                self.workspace_id,
                self.binding_id,
            ):
                return
            if result:
                self.status, self.policy, self.requires_upgrade = result
                choices = tuple(field.value for field in self.query(Input))
                applied = action is not None and action.action in {"enable", "apply"}
                if choices == (submitted_choices if applied else self._painted_choices):
                    self.query_one(".project-index-prefixes", Input).value = "; ".join(
                        p or "." for p in self.policy.include_prefixes
                    )
                    self.query_one(".project-index-suffixes", Input).value = "; ".join(
                        self.policy.text_suffixes
                    )
                    self._painted_choices = tuple(
                        field.value for field in self.query(Input)
                    )
                guidance = {
                    "Off": "Off. Enable saves choices; Refresh starts a scan. Clear removes cached content.",
                    "Paused": "Paused. Refresh applies current choices; Resume continues an interrupted scan.",
                    "Interrupted": "Interrupted. Resume explicitly to continue, or Refresh to scan again.",
                    "Partial": "Partial coverage: some paths or limits prevented a complete scan. Review choices, then Refresh.",
                    "Error": "Scan failed. Check folder access and available storage, then Refresh.",
                    "Ready": "Ready. Refresh explicitly after files change.",
                    "Building": "Building. Counts describe discovered entries; total size is not yet known.",
                }
                self._feedback(
                    "Cache upgrade required. Refresh explicitly to upgrade and scan, or Clear cached content. Free storage if the upgrade cannot fit."
                    if self.requires_upgrade
                    else guidance[self.status.state]
                )
            elif isinstance(error, LifetimeBusy) or "busy" in str(error):
                self._feedback(
                    "Index is busy with reads or maintenance. Retry after current work finishes."
                )
            elif str(error) == "index_cache_limit":
                self._feedback(
                    "The cache upgrade or scan needs more storage. Free space and retry, or Clear cached content."
                )
            elif str(error) in {"stale_scope", "stale_job"}:
                self._feedback(
                    "The index changed before this action. Check status, then retry."
                )
            else:
                self.status = None
                self.query_one(".project-index-status", Static).update(
                    "Keyword index unavailable"
                )
                text = "Could not access the index. Check folder access and storage, then Check status."
                if action and action.action in {"clear", "disable"}:
                    text += " Cleanup may be incomplete."
                self._feedback(text)
            self._paint()
            if error and result is None and self.status is None:
                self.query_one(".project-index-status", Static).update(
                    "Keyword index unavailable"
                )

        self.run_worker(run(), group=f"project-index-{id(self)}", exit_on_error=False)
