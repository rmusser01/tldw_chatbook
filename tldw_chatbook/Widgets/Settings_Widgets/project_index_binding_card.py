"""Explicit folder keyword-index controls; all authority and storage work is threaded."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from tldw_chatbook.Workspaces.project_index.lifetime import LifetimeBusy
from tldw_chatbook.Workspaces.project_index.models import (
    ProjectIndexPolicy,
    ProjectIndexStatus,
    ProjectIndexRetirementUnproven,
    ProjectSemanticState,
)
from tldw_chatbook.Workspaces.project_index.runtime import (
    get_project_index_service,
    project_index_key,
)

DEFAULT_POLICY = ProjectIndexPolicy((), ())
DEFAULT_SEMANTIC_MODEL = "all-MiniLM-L6-v2"


class SemanticDisclosureDialog(ModalScreen[bool]):
    """Approve one inspected model destination, or dismiss without consent."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, destination: str) -> None:
        super().__init__()
        self.destination = destination

    def compose(self) -> ComposeResult:
        with Vertical(classes="settings-detail-card h-auto"):
            yield Static("Approve project text disclosure?", classes="settings-detail-row")
            yield Static(
                f"File text will be embedded by {self.destination}. Approve this folder and destination only.",
                classes="settings-detail-row", markup=False,
            )
            yield Button("Cancel", id="project-semantic-cancel", compact=True)
            yield Button("Approve destination", id="project-semantic-confirm", compact=True)

    def action_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#project-semantic-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(False)

    @on(Button.Pressed, "#project-semantic-confirm")
    def _confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(True)


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

    class SemanticActionRequested(Message):
        """Semantic intent with the observed keyword epoch and semantic revision."""

        def __init__(
            self, binding_id: str, workspace_id: str, status: ProjectIndexStatus,
            semantic: ProjectSemanticState, action: str, model_id: str,
        ) -> None:
            super().__init__()
            self.binding_id = binding_id
            self.workspace_id = workspace_id
            self.key = status.scope.key
            self.expected_epoch = status.scope.epoch
            self.expected_semantic_revision = semantic.semantic_revision
            self.settings = semantic.settings
            self.action = action
            self.model_id = model_id

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
        self.semantic_status: ProjectSemanticState | None = None
        self._semantic_read_failed: str | None = None
        self._semantic_cleanup_owner: tuple[object, object] | None = None
        self._semantic_destination = ""
        self._painted_semantic_model = DEFAULT_SEMANTIC_MODEL
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
        yield Static("Semantic index: checking…", classes="settings-detail-row project-semantic-status", markup=False)
        yield Static("Embedding model ID (per folder)", classes="settings-detail-row")
        yield Input(
            DEFAULT_SEMANTIC_MODEL, classes="settings-compact-input project-semantic-model",
        )
        yield Static("", classes="settings-detail-row project-semantic-destination", markup=False)
        for action, label in (
            ("semantic-stage", "Choose semantic destination"),
            ("semantic-approve", "Approve destination"),
            ("semantic-refresh", "Build semantic index"),
            ("semantic-disable", "Disable semantic index"),
            ("semantic-retry-cleanup", "Retry semantic cleanup"),
        ):
            yield Button(label, name=action, classes="project-semantic-action", compact=True)
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
            and (self._semantic_read_failed or self.status.state == "Building" or (
                self.semantic_status is not None and self.semantic_status.state == "Building"
            ))
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
        for button in self.query(".project-index-action"):
            button.display = button.name in actions
        for field in self._keyword_inputs():
            field.disabled = (
                not self.available
                or not self.supported
                or (state == "Building" and not self.requires_upgrade)
            )
        semantic = self.semantic_status
        if not self.supported or not self.available:
            semantic_text = "Semantic index unavailable. Restore folder access."
        elif self._semantic_read_failed:
            last_known = f" (last known {semantic.state})" if semantic else ""
            if self._semantic_read_failed == "busy":
                semantic_text = f"Semantic index temporarily unavailable{last_known}. Check status to retry."
            else:
                semantic_text = f"Semantic status unavailable after a read failure{last_known}. Check status to retry."
            if self._semantic_cleanup_owner:
                semantic_text += " · cleanup pending; retry semantic cleanup."
        elif semantic is None:
            semantic_text = "Semantic index: checking…"
        else:
            semantic_text = f"Semantic index: {semantic.state}"
            if semantic.coverage is not None:
                semantic_text += f" · {semantic.coverage:,} chunks"
            if semantic.reason and semantic.reason.startswith("previous_index_refresh_failed"):
                semantic_text += " · previous index available; refresh failed"
            elif semantic.reason == "destination_approval_required":
                semantic_text += " · destination approval required"
            elif semantic.reason:
                semantic_text += f" · {semantic.reason.replace('_', ' ')}"
            if semantic.close_pending or self._semantic_cleanup_owner:
                semantic_text += " · cleanup pending"
        self.query_one(".project-semantic-status", Static).update(semantic_text)
        self.query_one(".project-semantic-destination", Static).update(
            self._semantic_destination or (
                f"Saved: {semantic.settings.provider_ref}, {semantic.settings.config_ref}. Inspect before approval."
                if semantic and semantic.settings else "Choose a model to inspect its destination."
            )
        )
        semantic_actions = set()
        if self.available and self.supported and status and semantic and not self._semantic_read_failed:
            semantic_actions.add("semantic-stage")
            if semantic.enabled:
                semantic_actions.add("semantic-disable")
                if semantic.settings and semantic.settings.approved_destination_fingerprint is None:
                    semantic_actions.add("semantic-approve")
                if semantic.settings and semantic.settings.approved_destination_fingerprint == semantic.settings.destination_fingerprint and status.state in {"Ready", "Partial"}:
                    semantic_actions.add("semantic-refresh")
            if semantic.close_pending or self._semantic_cleanup_owner:
                semantic_actions.add("semantic-retry-cleanup")
        elif self.available and self.supported and self._semantic_cleanup_owner:
            semantic_actions.add("semantic-retry-cleanup")
        for button in self.query(".project-semantic-action"):
            button.display = button.name in semantic_actions
        self.query_one(".project-semantic-model", Input).disabled = (
            not self.available or not self.supported or semantic is None or self._semantic_read_failed
        )

    def _keyword_inputs(self) -> tuple[Input, Input]:
        return (
            self.query_one(".project-index-prefixes", Input),
            self.query_one(".project-index-suffixes", Input),
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

    @on(Button.Pressed, ".project-semantic-action")
    def _semantic_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self.status is None or self.semantic_status is None or self._semantic_read_failed:
            if event.button.name != "semantic-retry-cleanup" or not self._semantic_cleanup_owner:
                return
        if self.status is None or self.semantic_status is None:
            return
        model_id = self.query_one(".project-semantic-model", Input).value.strip()
        if event.button.name == "semantic-approve":
            if not self.semantic_status.settings or model_id != self.semantic_status.settings.model_id:
                self._feedback("The model choice changed. Choose its destination first.")
                return
            self.run_worker(self._confirm_semantic(), group=f"semantic-consent-{id(self)}", exclusive=True)
            return
        if event.button.name == "semantic-stage" and not model_id:
            self._feedback("Enter a configured embedding model ID.")
            return
        self.post_message(self.SemanticActionRequested(
            self.binding_id, self.workspace_id, self.status,
            self.semantic_status, event.button.name.removeprefix("semantic-"), model_id,
        ))

    async def _confirm_semantic(self) -> None:
        semantic = self.semantic_status
        status = self.status
        if semantic is None or status is None or semantic.settings is None:
            return
        try:
            selected, destination = await asyncio.to_thread(
                self._inspect_semantic_destination, semantic.settings.model_id
            )
        except Exception:
            self._feedback("Could not inspect the current destination. Check model configuration.")
            return
        if selected != replace(semantic.settings, approved_destination_fingerprint=None):
            self._feedback("The model destination changed. Stage the current destination first.")
            return
        approved = await self.app.push_screen_wait(SemanticDisclosureDialog(destination))
        if approved and self.is_mounted:
            self.post_message(self.SemanticActionRequested(
                self.binding_id, self.workspace_id, status, semantic,
                "approve", semantic.settings.model_id,
            ))

    @staticmethod
    def _inspect_semantic_destination(model_id: str) -> tuple[object, str]:
        from tldw_chatbook.Workspaces.project_index.semantic import _resolved_spec, _settings_for_spec

        spec, destination = _resolved_spec(model_id)
        selected = _settings_for_spec(model_id, spec, destination)
        if spec.provider == "huggingface":
            return selected, f"local model {spec.model_name_or_path} (revision {spec.revision or 'configured default'})"
        return selected, f"{spec.provider} model {spec.model_name_or_path} at {destination}"

    @on(SemanticActionRequested)
    def _semantic_requested(self, event: SemanticActionRequested) -> None:
        event.stop()
        if (event.binding_id, event.workspace_id) == (self.binding_id, self.workspace_id):
            self._launch(None, event)

    def _execute_semantic(self, app: object, action: SemanticActionRequested) -> tuple[ProjectSemanticState, str]:
        if (action.binding_id, action.workspace_id) != (self.binding_id, self.workspace_id):
            raise ValueError("stale_scope")
        service = get_project_index_service(app)
        kwargs = dict(expected_epoch=action.expected_epoch, expected_semantic_revision=action.expected_semantic_revision)
        if action.action == "stage":
            selected, destination = self._inspect_semantic_destination(action.model_id)
            return service.enable_semantic(action.key, selected, **kwargs), destination
        if action.action == "approve":
            if action.settings is None or action.settings.model_id != action.model_id:
                raise ValueError("stale_semantic_revision")
            selected = replace(
                action.settings,
                approved_destination_fingerprint=action.settings.destination_fingerprint,
            )
            return service.enable_semantic(action.key, selected, **kwargs), ""
        if action.action == "disable":
            return service.disable_semantic(action.key, **kwargs), ""
        if action.action == "refresh":
            service.refresh_semantic(action.key, **kwargs)
            return service.semantic_state(action.key), ""
        if action.action == "retry-cleanup":
            service.retry_semantic_close(action.key)
            return service.semantic_state(action.key), ""
        raise ValueError("unknown_semantic_action")

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

    def _launch(self, action: ActionRequested | None, semantic_action: SemanticActionRequested | None = None) -> None:
        self._request += 1
        token = (self._visit, self._request, self.workspace_id, self.binding_id)
        self._pending += 1
        if action or semantic_action:
            self._feedback(f"{(action or semantic_action).action.title()} requested…")
        app = self.app
        submitted_choices = tuple(field.value for field in self._keyword_inputs())
        submitted_model = self.query_one(".project-semantic-model", Input).value

        async def run():
            service_ref = None
            try:

                if semantic_action and semantic_action.action == "retry-cleanup" and self._semantic_cleanup_owner:
                    service_ref, key = self._semantic_cleanup_owner
                    await asyncio.to_thread(service_ref.retry_semantic_close, key)
                    pending = await asyncio.to_thread(service_ref.semantic_close_pending, key)
                    if not pending and getattr(service_ref, "_closed", False):
                        await service_ref.aclose()
                    semantic = None
                    semantic_read_failed = "pending" if pending else "closed"
                    if not pending and not getattr(service_ref, "_closed", False):
                        def read_retired_owner():
                            with service_ref.registry_connection_scope():
                                return service_ref.semantic_state(key)

                        try:
                            semantic = await asyncio.to_thread(read_retired_owner)
                            semantic_read_failed = None
                        except LifetimeBusy:
                            semantic_read_failed = "busy"
                        except Exception:
                            semantic_read_failed = "failed"
                    if not pending:
                        self._semantic_cleanup_owner = None
                    result = (
                        (self.status, self.policy, self.requires_upgrade),
                        semantic,
                        "",
                        semantic_read_failed,
                    )
                    error = None
                else:

                    def owned_call():
                        nonlocal service_ref
                        service_ref = get_project_index_service(app)
                        with service_ref.registry_connection_scope():
                            destination = ""
                            if semantic_action:
                                _, destination = self._execute_semantic(app, semantic_action)
                            keyword = self._execute(app, action) if action else self._load(app)
                            try:
                                semantic = service_ref.semantic_state(keyword[0].scope.key)
                                semantic_read_failed = None
                            except LifetimeBusy:
                                semantic = None
                                semantic_read_failed = "busy"
                            except Exception:
                                semantic = None
                                semantic_read_failed = "failed"
                            return keyword, semantic, destination, semantic_read_failed

                    result = await asyncio.to_thread(owned_call)
                    error = None
            except Exception as exc:
                result, error = None, exc
                selected_key = semantic_action.key if semantic_action else (
                    action.expected_scope.key if action else None
                )
                if (
                    isinstance(exc, ProjectIndexRetirementUnproven)
                    and service_ref is not None and selected_key is not None
                    and await asyncio.to_thread(service_ref.semantic_close_pending, selected_key)
                ):
                    self._semantic_cleanup_owner = (service_ref, selected_key)
                    self._semantic_read_failed = "pending"
                    if self.semantic_status is not None:
                        self.semantic_status = replace(self.semantic_status, close_pending=True)
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
                keyword, semantic, destination, semantic_read_failed = result
                self._semantic_read_failed = semantic_read_failed
                if semantic is not None:
                    self.semantic_status = semantic
                self.status, self.policy, self.requires_upgrade = keyword
                if destination:
                    self._semantic_destination = destination
                elif self.semantic_status and not self.semantic_status.enabled:
                    self._semantic_destination = ""
                if self.semantic_status and self.semantic_status.settings:
                    model_input = self.query_one(".project-semantic-model", Input)
                    if model_input.value == (submitted_model if semantic_action and semantic_action.action == "stage" else self._painted_semantic_model):
                        model_input.value = self.semantic_status.settings.model_id
                        self._painted_semantic_model = model_input.value
                elif semantic_action and semantic_action.action == "disable":
                    self._semantic_destination = ""
                choices = tuple(field.value for field in self._keyword_inputs())
                applied = action is not None and action.action in {"enable", "apply"}
                if choices == (submitted_choices if applied else self._painted_choices):
                    self.query_one(".project-index-prefixes", Input).value = "; ".join(
                        p or "." for p in self.policy.include_prefixes
                    )
                    self.query_one(".project-index-suffixes", Input).value = "; ".join(
                        self.policy.text_suffixes
                    )
                    self._painted_choices = tuple(field.value for field in self._keyword_inputs())
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
                    "Semantic index temporarily unavailable. Check status to retry; keyword search remains available."
                    if self._semantic_read_failed == "busy"
                    else "Semantic status read failed. Check status to retry; keyword search remains available."
                    if self._semantic_read_failed
                    else "Cache upgrade required. Refresh explicitly to upgrade and scan, or Clear cached content. Free storage if the upgrade cannot fit."
                    if self.requires_upgrade
                    else guidance[self.status.state]
                )
            elif semantic_action:
                self._feedback(
                    "Semantic cleanup pending on the existing index owner. Retry cleanup."
                    if self._semantic_cleanup_owner else
                    "Semantic action could not finish. Keyword search remains available. "
                    "Check status before retrying."
                    if str(error) not in {"stale_scope", "stale_semantic_revision"}
                    else "The semantic choice changed. Check status before retrying."
                )
            elif isinstance(error, LifetimeBusy) or "busy" in str(error):
                self._feedback(
                    "Index is busy with reads or maintenance. Retry after current work finishes."
                )
            elif str(error) == "index_cache_limit":
                self._feedback(
                    "The cache upgrade or scan needs more storage. Free space and retry, or Clear cached content."
                )
            elif str(error) in {"stale_scope", "stale_job", "stale_semantic_revision"}:
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
