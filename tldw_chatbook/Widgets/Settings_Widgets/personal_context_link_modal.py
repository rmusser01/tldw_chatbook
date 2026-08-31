"""Reviewed, content-free Personal Context first-link Settings modal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from ..modal_dismissal import SafeModalDismissMixin


@dataclass(frozen=True, slots=True)
class PersonalContextLinkReviewResult:
    """Bounded exact decisions returned to the link coordinator."""

    plan_id: str
    decisions: Mapping[str, str]
    unlinked_remote_scope_ids: tuple[str, ...]
    retry: bool = False

    @property
    def collision_decisions(self) -> Mapping[str, str]:
        """Compatibility name for callers that only handled collision rows."""

        return self.decisions


def _count_label(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else (plural or singular + 's')}"


class PersonalContextLinkModal(
    SafeModalDismissMixin,
    ModalScreen[PersonalContextLinkReviewResult | None],
):
    """Require explicit review before any canonical profile apply or upload."""

    SAFE_MODAL_CONTENT = "#personal-context-link-modal"
    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel")]

    def __init__(self, plan: Any, *, retry_callback: Any | None = None) -> None:
        super().__init__()
        self.plan = plan
        self._retry_callback = retry_callback
        self._decisions: dict[str, str] = {}
        self._busy = False

    def compose(self) -> ComposeResult:
        with Vertical(id="personal-context-link-modal"):
            yield Static("Link My Profile", classes="profile-interview-title")
            yield Static(
                "Review what will join the same canonical profile before anything is uploaded.",
                classes="profile-interview-copy",
            )
            yield Static(
                "An authorized home server can read syncable profile content. Device-only records stay here.",
                classes="profile-interview-disclosure",
            )
            if self.plan.attention_codes:
                yield Static(
                    "Review cannot continue because the server snapshot needs attention. Retry after the server or profile state is corrected.",
                    id="personal-context-link-attention",
                    classes="personal-context-review-warning",
                )
            with VerticalScroll(id="personal-context-link-review-list"):
                yield Static(
                    "Adopt the server profile identity for this device: "
                    f"{self.plan.profile_adoption[0]} → "
                    f"{self.plan.profile_adoption[1]}",
                    id="personal-context-link-profile-adoption",
                    classes="settings-inline-guidance",
                )
                yield Static(
                    "Map the global profile scope to the server global scope: "
                    f"{self.plan.global_scope_mapping[0]} → "
                    f"{self.plan.global_scope_mapping[1]}",
                    id="personal-context-link-global-mapping",
                    classes="settings-inline-guidance",
                )
                schema_outcome = str(self.plan.schema_outcome).replace(
                    "compatible:", "compatible (version "
                )
                if schema_outcome.startswith("compatible (version "):
                    schema_outcome += ")"
                quota_outcome = (
                    "Server quotas satisfy required minimums"
                    if self.plan.quota_outcome == "minimums_satisfied"
                    else "Server quotas need attention"
                )
                purge_outcome = str(self.plan.purge_outcome).replace(
                    "generation_matches:", "matches ("
                )
                if purge_outcome.startswith("matches ("):
                    purge_outcome += ")"
                yield Static(
                    f"Schema: {schema_outcome} · {quota_outcome} · "
                    f"Purge generation: {purge_outcome}",
                    id="personal-context-link-contract-outcomes",
                    classes="profile-interview-state",
                )
                for quota, required, server, satisfied in getattr(
                    self.plan, "quota_outcomes", ()
                ):
                    yield Static(
                        f"{quota} · required {required} · server {server} · "
                        f"{'satisfied' if satisfied else 'insufficient'}",
                        classes="settings-inline-guidance",
                    )
                for record_id, outcome, local_version, server_version in getattr(
                    self.plan, "record_outcomes", ()
                ):
                    yield Static(
                        f"{record_id} · {outcome.replace('_', ' ')} · "
                        f"local {local_version or 'none'} · "
                        f"server {server_version or 'none'}",
                        classes="settings-inline-guidance",
                    )
                for proposal_id, outcome in getattr(
                    self.plan, "proposal_outcomes", ()
                ):
                    yield Static(
                        f"{proposal_id} · {outcome.replace('_', ' ')}",
                        classes="settings-inline-guidance",
                    )
                yield Static(
                    " · ".join(
                        (
                            _count_label(len(self.plan.exact_record_ids), "exact match"),
                            _count_label(len(self.plan.local_only_record_ids), "local addition"),
                            _count_label(len(self.plan.remote_only_record_ids), "server addition"),
                            _count_label(len(self.plan.key_collisions), "collision"),
                            _count_label(
                                len(getattr(self.plan, "exact_proposal_ids", ())),
                                "exact proposal",
                            ),
                            _count_label(
                                len(getattr(self.plan, "local_only_proposal_ids", ())),
                                "local proposal",
                            ),
                            _count_label(
                                len(getattr(self.plan, "remote_only_proposal_ids", ())),
                                "server proposal",
                            ),
                            _count_label(
                                len(self.plan.unlinked_remote_scope_ids),
                                "unlinked workspace",
                                "unlinked workspaces",
                            ),
                            _count_label(
                                len(self.plan.device_only_record_ids),
                                "device-only record",
                            ),
                        )
                    ),
                    id="personal-context-link-counts",
                    classes="profile-interview-state",
                )
                if self.plan.unlinked_remote_scope_ids:
                    yield Static(
                        "Incoming workspace context will stay unlinked and unavailable to agents until you map it in Settings.",
                        id="personal-context-link-unlinked-copy",
                        classes="personal-context-review-warning",
                    )
                for local_scope_id, new_scope_id in getattr(
                    self.plan, "workspace_new_scope_ids", ()
                ):
                    yield Static(
                        f"{local_scope_id} → new scope {new_scope_id}",
                        classes="settings-inline-guidance",
                    )
                for conflict in getattr(
                    self.plan, "workspace_mapping_conflicts", ()
                ):
                    yield Static(
                        f"{conflict.local_scope_id} → "
                        f"{conflict.remote_scope_id} unavailable: "
                        f"{conflict.record_ids[0]} conflicts with "
                        f"{conflict.record_ids[1]}",
                        classes="personal-context-review-warning",
                    )
                for index, collision in enumerate(self.plan.key_collisions):
                    with Vertical(classes="personal-context-link-decision-row"):
                        yield Static(
                            f"Collision {index + 1}: choose which canonical record remains active.",
                            classes="settings-inline-guidance",
                        )
                        with Horizontal(classes="profile-interview-actions"):
                            yield Button(
                                "Keep this device",
                                id=f"personal-context-link-collision-{index}-keep-local",
                            )
                            yield Button(
                                "Keep server",
                                id=f"personal-context-link-collision-{index}-keep-server",
                            )
                        yield Static(
                            "Decision required",
                            id=f"personal-context-link-collision-{index}-status",
                            classes="settings-inline-guidance",
                        )
                for index, conflict in enumerate(self.plan.version_conflicts):
                    with Vertical(classes="personal-context-link-decision-row"):
                        yield Static(
                            f"Version conflict {index + 1}: choose which exact lineage to keep.",
                            classes="settings-inline-guidance",
                        )
                        with Horizontal(classes="profile-interview-actions"):
                            yield Button(
                                "Keep this device",
                                id=f"personal-context-link-version-{index}-keep-local",
                            )
                            yield Button(
                                "Keep server",
                                id=f"personal-context-link-version-{index}-keep-server",
                            )
                        yield Static(
                            "Decision required",
                            id=f"personal-context-link-version-{index}-status",
                            classes="settings-inline-guidance",
                        )
                for index, _scope_id in enumerate(
                    getattr(self.plan, "local_workspace_scope_ids", ())
                ):
                    blocked_mappings = {
                        (item.local_scope_id, item.remote_scope_id)
                        for item in getattr(
                            self.plan, "workspace_mapping_conflicts", ()
                        )
                    }
                    with Vertical(classes="personal-context-link-decision-row"):
                        yield Static(
                            f"Local workspace {index + 1}: choose its canonical scope.",
                            classes="settings-inline-guidance",
                        )
                        with Horizontal(classes="profile-interview-actions"):
                            yield Button(
                                "Keep separate",
                                id=f"personal-context-link-workspace-{index}-new",
                            )
                            yield Button(
                                "Leave unlinked",
                                id=f"personal-context-link-workspace-{index}-unlinked",
                            )
                            for remote_index, _remote_id in enumerate(
                                self.plan.unlinked_remote_scope_ids
                            ):
                                yield Button(
                                    f"Map to server workspace {remote_index + 1}",
                                    id=(
                                        f"personal-context-link-workspace-{index}-"
                                        f"map-{remote_index}"
                                    ),
                                    disabled=(
                                        (_scope_id, _remote_id)
                                        in blocked_mappings
                                    ),
                                )
                        yield Static(
                            "Decision required",
                            id=f"personal-context-link-workspace-{index}-status",
                            classes="settings-inline-guidance",
                        )
            yield Static(
                "",
                id="personal-context-link-status",
                classes="settings-inline-guidance",
            )
            with Horizontal(classes="profile-interview-actions"):
                yield Button("Cancel", id="personal-context-link-cancel")
                if self.plan.attention_codes and self._retry_callback is not None:
                    yield Button("Retry snapshot", id="personal-context-link-retry")
                yield Button(
                    "Approve and link",
                    id="personal-context-link-approve",
                    variant="primary",
                    disabled=not self._can_approve(),
                )

    def _can_approve(self) -> bool:
        required = set(self.plan.required_decision_ids)
        workspace_decisions = {
            decision_id.removeprefix("workspace:"): choice
            for decision_id, choice in self._decisions.items()
            if decision_id.startswith("workspace:")
        }
        remote_scope_ids = set(self.plan.unlinked_remote_scope_ids)
        mapped_remote_ids = [
            choice
            for choice in workspace_decisions.values()
            if choice in remote_scope_ids
        ]
        blocked_mappings = {
            (item.local_scope_id, item.remote_scope_id)
            for item in getattr(self.plan, "workspace_mapping_conflicts", ())
        }
        return (
            not self._busy
            and not self.plan.attention_codes
            and required.issubset(self._decisions)
            and len(mapped_remote_ids) == len(set(mapped_remote_ids))
            and not any(
                (scope_id, choice) in blocked_mappings
                for scope_id, choice in workspace_decisions.items()
            )
        )

    @on(Button.Pressed)
    def handle_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if not button_id.startswith("personal-context-link-"):
            return
        event.stop()
        if self._busy:
            return
        if button_id == "personal-context-link-cancel":
            self.dismiss(None)
            return
        if button_id == "personal-context-link-retry":
            self._retry()
            return
        if button_id == "personal-context-link-approve":
            if not self._can_approve():
                return
            self.dismiss(
                PersonalContextLinkReviewResult(
                    plan_id=str(self.plan.plan_id),
                    decisions=dict(self._decisions),
                    unlinked_remote_scope_ids=tuple(
                        self.plan.unlinked_remote_scope_ids
                    ),
                )
            )
            return
        if button_id.startswith("personal-context-link-collision-"):
            choice = button_id.removeprefix("personal-context-link-collision-")
            try:
                index_text, decision = choice.split("-keep-", 1)
                index = int(index_text)
                decision_id = self.plan.key_collisions[index].decision_id
            except (IndexError, ValueError):
                return
            if decision not in {"local", "server"}:
                return
            status_id = f"#personal-context-link-collision-{index}-status"
        elif button_id.startswith("personal-context-link-version-"):
            choice = button_id.removeprefix("personal-context-link-version-")
            try:
                index_text, decision = choice.split("-keep-", 1)
                index = int(index_text)
                decision_id = self.plan.version_conflicts[index].decision_id
            except (IndexError, ValueError):
                return
            if decision not in {"local", "server"}:
                return
            status_id = f"#personal-context-link-version-{index}-status"
        elif button_id.startswith("personal-context-link-workspace-"):
            choice = button_id.removeprefix("personal-context-link-workspace-")
            try:
                index_text, decision = choice.split("-", 1)
                index = int(index_text)
                scope_id = self.plan.local_workspace_scope_ids[index]
            except (IndexError, ValueError):
                return
            decision_id = f"workspace:{scope_id}"
            if decision == "new":
                decision = "new"
            elif decision == "unlinked":
                decision = "unlinked"
            elif decision.startswith("map-"):
                try:
                    remote_index = int(decision.removeprefix("map-"))
                    decision = self.plan.unlinked_remote_scope_ids[remote_index]
                except (IndexError, ValueError):
                    return
            else:
                return
            status_id = f"#personal-context-link-workspace-{index}-status"
        else:
            return
        self._decisions[decision_id] = decision
        self.query_one(status_id, Static).update(
            "This device" if decision == "local" else (
                "Server" if decision == "server" else "Mapped"
            )
        )
        self.query_one("#personal-context-link-approve", Button).disabled = (
            not self._can_approve()
        )

    def _retry(self) -> None:
        if self._retry_callback is None:
            return
        self._busy = True
        self.query_one("#personal-context-link-status", Static).update(
            "Retrying server snapshot…"
        )
        for button in self.query(Button):
            button.disabled = True
        self.dismiss(
            PersonalContextLinkReviewResult(
                plan_id=str(self.plan.plan_id),
                decisions={},
                unlinked_remote_scope_ids=(),
                retry=True,
            )
        )

    def action_request_safe_cancel(self) -> None:
        if not self._busy:
            self.dismiss(None)


__all__ = ["PersonalContextLinkModal", "PersonalContextLinkReviewResult"]
