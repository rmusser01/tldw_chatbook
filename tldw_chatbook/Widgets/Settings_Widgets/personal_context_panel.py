"""Focused My Profile editor for canonical Settings."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.css.query import QueryError
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Select, Static
from tldw_profile_core import (
    AgentVisibility,
    ConstraintPayload,
    ConventionPayload,
    CorrectionPayload,
    GoalPayload,
    IdentityPayload,
    LegacyUnclassifiedPayload,
    PreferencePayload,
    ProfileControls,
    ProfileRecord,
    RecordState,
    RelationshipPayload,
    SemanticKey,
    SyncMode,
    WorkingContextPayload,
)

from ...Personal_Context.export_service import ExportRequest, RecoveryExportRequest
from ...Personal_Context.runtime_policy import AgentAuthority
from ...Personal_Context.service import (
    PersonalContextService,
    PersonalContextSettingsSnapshot,
    ProfileConflictError,
    ProfileKeyCollisionError,
    ProfileOperationalState,
    RecordMutation,
)
from ..confirmation_dialog import ConfirmationDialog
from .personal_context_review_modal import (
    PersonalContextProposalReviewModal,
    ProposalReviewResult,
)


_STATE_LABELS = {
    ProfileOperationalState.ABSENT: "Empty",
    ProfileOperationalState.REMOVED: "Removed",
    ProfileOperationalState.LOCKED: "Locked",
    ProfileOperationalState.DISABLED: "Disabled",
    ProfileOperationalState.READY: "Available",
}

_DISABLED_REASON_COPY = {
    "personal_context_disabled": "Agent use is disabled. Your profile remains available for manual editing.",
    "runtime_policy_invalid": "Agent use is disabled because the local runtime policy is invalid.",
    "agent_authority_denied": "Agent use is disabled because runtime authority could not be verified.",
}

_KIND_OPTIONS = (
    ("Preference", "preference"),
    ("Identity", "identity"),
    ("Relationship", "relationship"),
    ("Correction", "correction"),
    ("Constraint", "constraint"),
    ("Goal", "goal"),
    ("Convention", "convention"),
    ("Working context", "working_context"),
    ("Legacy note", "legacy_unclassified"),
)

_ALL_SCOPES = "__all__"
_COLLISION_COPY = (
    "A record with the same kind and subject is already active in this scope. "
    "Change the kind or subject, or archive the other record."
)


def _bounded_label(value: object, *, limit: int = 80) -> str:
    text = str(value)
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


class RecoveryPassphraseDialog(ModalScreen[str | None]):
    """Collect and confirm a recovery passphrase without exposing its text."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def compose(self) -> ComposeResult:
        with Container(id="personal-context-recovery-dialog"):
            yield Static("Protect recovery copy", classes="dialog-title")
            yield Static(
                "Enter the passphrase twice. It cannot be recovered if lost.",
                classes="dialog-message",
            )
            yield Static("Recovery passphrase", classes="settings-input-label")
            yield Input(
                password=True,
                placeholder="Passphrase",
                id="personal-context-recovery-passphrase",
            )
            yield Static("Confirm recovery passphrase", classes="settings-input-label")
            yield Input(
                password=True,
                placeholder="Confirm passphrase",
                id="personal-context-recovery-passphrase-confirm",
            )
            yield Static("", id="personal-context-recovery-error")
            with Horizontal(classes="button-container"):
                yield Button("Cancel", id="personal-context-recovery-cancel")
                yield Button(
                    "Export recovery copy",
                    id="personal-context-recovery-confirm",
                    variant="primary",
                )

    @on(Button.Pressed)
    def handle_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "personal-context-recovery-cancel":
            self.dismiss(None)
            return
        if event.button.id != "personal-context-recovery-confirm":
            return
        passphrase = self.query_one(
            "#personal-context-recovery-passphrase", Input
        ).value
        confirmation = self.query_one(
            "#personal-context-recovery-passphrase-confirm", Input
        ).value
        if len(passphrase) < 12 or passphrase != confirmation:
            self.query_one("#personal-context-recovery-error", Static).update(
                "Passphrases must match and contain at least 12 characters."
            )
            return
        self.dismiss(passphrase)

    def action_cancel(self) -> None:
        self.dismiss(None)


class PersonalContextSettingsPanel(Vertical):
    """My Profile editor; every read and mutation delegates to the service."""

    BINDINGS = [
        Binding("a", "add_record", "Add", show=False),
        Binding("e", "edit_record", "Edit", show=False),
        Binding("d", "delete_record", "Delete", show=False),
        Binding("x", "export_profile", "Export", show=False),
    ]

    snapshot: reactive[PersonalContextSettingsSnapshot | None] = reactive(
        None, recompose=True
    )
    load_failed = reactive(False, recompose=True)
    editor_mode = reactive("", recompose=True)
    selected_record_id = reactive("", recompose=True)
    selected_scope_id = reactive(_ALL_SCOPES, recompose=True)
    interview_mode = reactive("fixed", recompose=True)

    def __init__(
        self,
        service: PersonalContextService | Callable[..., PersonalContextService],
        interview_launcher: Callable[[str, str, str], Any] | None = None,
        link_launcher: Callable[[], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("id", "personal-context-settings-panel")
        super().__init__(**kwargs)
        if callable(service):
            self._service: PersonalContextService | None = None
            self._service_factory: Callable[..., PersonalContextService] | None = (
                service
            )
        else:
            self._service = service
            self._service_factory = None
        self._interview_launcher = interview_launcher
        self._link_launcher = link_launcher
        self._load_generation = 0

    def on_mount(self) -> None:
        self.load_records(retry_locked=True)

    def load_records(self, *, retry_locked: bool = False) -> None:
        """Fetch one content snapshot without blocking Textual's event loop."""

        self._load_generation += 1
        self._load_records_in_worker(self._load_generation, retry_locked)

    @work(thread=True, exclusive=True, group="personal-context-settings-load")
    def _load_records_in_worker(self, generation: int, retry_locked: bool) -> None:
        try:
            snapshot = self._require_service(
                retry_locked=retry_locked
            ).settings_snapshot()
        except Exception:
            self.app.call_from_thread(self._apply_load_error, generation)
            return
        self.app.call_from_thread(self._apply_snapshot, generation, snapshot)

    def _require_service(self, *, retry_locked: bool = False) -> PersonalContextService:
        if self._service is None:
            if self._service_factory is None:
                raise RuntimeError("Personal Context service is unavailable")
            self._service = self._service_factory(retry_locked=retry_locked)
        elif retry_locked and self._service_factory is not None:
            status = getattr(self._service, "status", None)
            if callable(status) and status().state is ProfileOperationalState.LOCKED:
                self._service = self._service_factory(retry_locked=True)
        return self._service

    def _apply_load_error(self, generation: int) -> None:
        if generation != self._load_generation:
            return
        self.load_failed = True
        self.snapshot = None
        self._refresh_settings_shortcuts()

    def _apply_snapshot(
        self, generation: int, snapshot: PersonalContextSettingsSnapshot
    ) -> None:
        if generation != self._load_generation:
            return
        self.load_failed = False
        self.snapshot = snapshot
        scope_ids = {scope.scope.scope_id for scope in snapshot.scopes}
        if self.selected_scope_id not in scope_ids | {_ALL_SCOPES}:
            self.selected_scope_id = _ALL_SCOPES
        visible_records = self._visible_records(snapshot)
        record_ids = {record.record_id for record in visible_records}
        if self.selected_record_id not in record_ids:
            self.selected_record_id = (
                visible_records[0].record_id if visible_records else ""
            )
        self._refresh_settings_shortcuts()

    def _refresh_settings_shortcuts(self) -> None:
        try:
            refresh = getattr(self.screen, "_register_footer_shortcuts", None)
        except Exception:
            return
        if callable(refresh):
            refresh()

    def compose(self) -> ComposeResult:
        yield Static("My Profile", classes="destination-section settings-column-title")
        if self.load_failed:
            yield Static(
                "Error", id="personal-context-status", classes="settings-status-badge"
            )
            yield Static(
                "Profile status could not be loaded. Try again; private details were not displayed.",
                classes="settings-inline-guidance",
            )
            yield Button("Try again", id="personal-context-reload")
            return
        if self.snapshot is None:
            yield Static(
                "Loading", id="personal-context-status", classes="settings-status-badge"
            )
            return

        status = self.snapshot.status
        label = _STATE_LABELS[status.state]
        if status.state is ProfileOperationalState.READY and not self.snapshot.records:
            label = "Empty"
        yield Static(
            label, id="personal-context-status", classes="settings-status-badge"
        )

        if status.state is ProfileOperationalState.LOCKED:
            yield Static(
                "Your encrypted profile is locked. Unlock secure key storage and try again.",
                classes="settings-inline-guidance",
            )
            yield Button("Try again", id="personal-context-reload")
            return
        if status.state is ProfileOperationalState.REMOVED:
            yield Static(
                "The local encrypted profile was removed. Nothing will be recreated automatically.",
                classes="settings-inline-guidance",
            )
            yield Static(
                "Finish secure removal to retry deletion of the old encryption keys without creating a new profile.",
                classes="settings-inline-guidance",
            )
            yield Button("Finish secure removal", id="personal-context-finish-removal")
            yield Button(
                "Start Fresh", id="personal-context-start-fresh", variant="primary"
            )
            return
        if status.state is ProfileOperationalState.ABSENT:
            yield Static(
                "Create an encrypted local profile when you are ready.",
                classes="settings-inline-guidance",
            )
            yield Button(
                "Create profile", id="personal-context-create", variant="primary"
            )
            return

        if status.state is ProfileOperationalState.DISABLED:
            yield Static(
                _DISABLED_REASON_COPY.get(
                    status.reason_code,
                    "Agent use is disabled because runtime authority is unavailable.",
                ),
                id="personal-context-disabled-reason",
                classes="settings-inline-guidance",
            )

        runtime_label = (
            "Disable agent use" if status.runtime_enabled else "Enable agent use"
        )
        visible_records = self._visible_records()
        selected_record = self._selected_record()
        selected_mutable = self._record_is_linked(selected_record)
        archive_label = (
            "Restore"
            if selected_record is not None
            and selected_record.state is RecordState.ARCHIVED
            else "Archive"
        )
        with Horizontal(classes="personal-context-toolbar"):
            yield Button(runtime_label, id="personal-context-runtime")
            yield Button("Add", id="personal-context-add", variant="primary")
            yield Button(
                "Edit",
                id="personal-context-edit",
                disabled=not selected_mutable,
            )
            yield Button(
                archive_label,
                id="personal-context-archive-restore",
                disabled=not selected_mutable,
            )
            yield Button(
                "Delete",
                id="personal-context-delete",
                variant="error",
                disabled=not selected_mutable,
            )

        yield Static("Proposed changes", classes="destination-section")
        if not self.snapshot.proposals:
            yield Static(
                "No agent-proposed changes are waiting for review.",
                id="personal-context-proposals-empty",
                classes="settings-inline-guidance",
            )
        else:
            yield Static(
                "Nothing changes until you accept it. Rejecting removes the proposed content.",
                classes="settings-inline-guidance",
            )
            with Vertical(id="personal-context-proposal-list"):
                for index, proposal in enumerate(self.snapshot.proposals):
                    yield Button(
                        self._proposal_label(proposal),
                        id=f"personal-context-proposal-{index}",
                        classes="personal-context-record-row",
                    )

        yield Static("Agent authority by scope", classes="destination-section")
        for index, scope in enumerate(self.snapshot.scopes):
            with Horizontal(classes="settings-input-row personal-context-scope-row"):
                yield Static(scope.label, classes="settings-input-label")
                yield Select(
                    (
                        ("Read only", AgentAuthority.READ_ONLY.value),
                        ("Propose changes", AgentAuthority.PROPOSE.value),
                        ("Direct write", AgentAuthority.DIRECT_WRITE.value),
                    ),
                    value=scope.authority.value,
                    allow_blank=False,
                    compact=True,
                    disabled=not scope.linked,
                    id=f"personal-context-authority-{index}",
                )

        yield Static("Profile records", classes="destination-section")
        with Horizontal(classes="settings-input-row personal-context-scope-row"):
            yield Static("Show", classes="settings-input-label")
            yield Select(
                (("All scopes", _ALL_SCOPES),)
                + tuple(
                    (scope.label, scope.scope.scope_id)
                    for scope in self.snapshot.scopes
                ),
                value=self.selected_scope_id,
                allow_blank=False,
                compact=True,
                id="personal-context-scope-filter",
            )
        if not visible_records:
            yield Static(
                (
                    "No profile records yet. Add one manually."
                    if not self.snapshot.records
                    else "No profile records in this scope."
                ),
                id="personal-context-empty",
                classes="settings-inline-guidance",
            )
        else:
            with VerticalScroll(id="personal-context-record-list"):
                for index, record in enumerate(visible_records):
                    classes = "personal-context-record-row"
                    if record.record_id == self.selected_record_id:
                        classes += " personal-context-record-selected"
                    yield Button(
                        self._record_label(record),
                        id=f"personal-context-record-{index}",
                        classes=classes,
                    )

        if self.editor_mode:
            yield from self._compose_editor()

        yield Static("Interview", classes="destination-section")
        interview_target = self._selected_interview_scope()
        with Horizontal(classes="settings-input-row personal-context-scope-row"):
            yield Static("Question style", classes="settings-input-label")
            yield Select(
                (
                    ("Fixed local questions", "fixed"),
                    ("Adaptive provider questions", "adaptive"),
                ),
                value=self.interview_mode,
                allow_blank=False,
                compact=True,
                id="personal-context-interview-mode",
            )
        yield Button(
            "Run interview again",
            id="personal-context-run-interview",
            disabled=interview_target is None,
        )
        if interview_target is None:
            yield Static(
                "Select a linked global or workspace scope to run an interview.",
                classes="settings-inline-guidance",
            )

        yield Static("Server sync", classes="destination-section")
        yield Static(
            "Review profile, collision, and workspace outcomes before linking to your home server.",
            classes="settings-inline-guidance",
        )
        yield Button(
            "Link to home server",
            id="personal-context-link-server",
            disabled=not self._link_launcher_available(),
        )
        if not self._link_launcher_available():
            yield Static(
                "Profile linking is unavailable until an authenticated home server is active.",
                classes="settings-inline-guidance",
            )

        yield Static("Export and local data", classes="destination-section")
        export_scope_label = self._selected_export_scope()[1]
        yield Static(
            f"Plaintext scope: {export_scope_label}",
            id="personal-context-export-scope",
            classes="settings-inline-guidance",
        )
        with Horizontal(classes="personal-context-toolbar"):
            yield Button(
                f"Export plaintext: {export_scope_label}",
                id="personal-context-export-plaintext",
            )
            yield Button("Export recovery copy", id="personal-context-export-recovery")
            yield Button(
                "Remove local profile",
                id="personal-context-remove-local",
                variant="error",
            )

    def _record_label(self, record: ProfileRecord) -> str:
        payload = record.payload
        subject = getattr(payload, "subject", "note") if payload is not None else "note"
        value = (
            getattr(payload, "value", None)
            or getattr(payload, "outcome", None)
            or getattr(payload, "text", None)
            or ""
        )
        scope_label = next(
            (
                row.label
                for row in (self.snapshot.scopes if self.snapshot else ())
                if row.scope.scope_id == record.scope_id
            ),
            "Unknown scope",
        )
        return (
            f"{scope_label} · {record.kind.value.replace('_', ' ').title()} · "
            f"{subject}: {value} · "
            f"{record.state.value} · {record.controls.sync_mode.value.replace('_', ' ')} · "
            f"{record.controls.agent_visibility.value.replace('_', ' ')}"
        )

    def _visible_records(
        self, snapshot: PersonalContextSettingsSnapshot | None = None
    ) -> tuple[ProfileRecord, ...]:
        current = self.snapshot if snapshot is None else snapshot
        if current is None:
            return ()
        if self.selected_scope_id == _ALL_SCOPES:
            return current.records
        return tuple(
            record
            for record in current.records
            if record.scope_id == self.selected_scope_id
        )

    def _scope_snapshot(self, scope_id: str):
        return next(
            (
                row
                for row in (self.snapshot.scopes if self.snapshot else ())
                if row.scope.scope_id == scope_id
            ),
            None,
        )

    def _record_is_linked(self, record: ProfileRecord | None) -> bool:
        if record is None:
            return False
        scope = self._scope_snapshot(record.scope_id)
        return scope is not None and scope.linked

    def _selected_export_scope(self) -> tuple[tuple[str, ...] | None, str]:
        if self.selected_scope_id == _ALL_SCOPES:
            return None, "All scopes"
        scope = self._scope_snapshot(self.selected_scope_id)
        return (self.selected_scope_id,), scope.label if scope else "Unknown scope"

    def available_shortcuts(self) -> tuple[tuple[str, str], ...]:
        """Return only profile shortcuts that work in the current panel state."""

        snapshot = self.snapshot
        if self.load_failed or snapshot is None:
            return ()
        if snapshot.status.state not in {
            ProfileOperationalState.READY,
            ProfileOperationalState.DISABLED,
        }:
            return ()
        shortcuts: list[tuple[str, str]] = []
        if any(scope.linked for scope in snapshot.scopes):
            shortcuts.append(("a", "add record"))
        if self._record_is_linked(self._selected_record()):
            shortcuts.extend((("e", "edit record"), ("d", "delete record")))
        shortcuts.append(("x", "export profile"))
        return tuple(shortcuts)

    def _selected_record(self) -> ProfileRecord | None:
        if self.snapshot is None:
            return None
        return next(
            (
                record
                for record in self._visible_records()
                if record.record_id == self.selected_record_id
            ),
            None,
        )

    def _compose_editor(self) -> ComposeResult:
        record = self._selected_record() if self.editor_mode == "edit" else None
        payload = record.payload if record is not None else None
        linked_scopes = tuple(
            scope
            for scope in (self.snapshot.scopes if self.snapshot else ())
            if scope.linked
        )
        kind = record.kind.value if record is not None else "preference"
        subject = getattr(payload, "subject", "") if payload is not None else ""
        value = (
            getattr(payload, "value", None)
            or getattr(payload, "outcome", None)
            or getattr(payload, "text", None)
            or ""
        )
        scope_value = (
            record.scope_id
            if record is not None
            else (linked_scopes[0].scope.scope_id if linked_scopes else Select.BLANK)
        )
        if record is not None and record.expires_at is not None:
            retention = "preserve"
        elif record is not None and record.no_expiry:
            retention = "no_expiry"
        else:
            retention = "30_days" if kind == "working_context" else "not_applicable"
        with Vertical(id="personal-context-editor", classes="settings-focus-card"):
            yield Static(
                "Edit record" if record is not None else "Add record",
                classes="destination-section",
            )
            yield Static("Kind", classes="settings-input-label")
            yield Select(
                _KIND_OPTIONS,
                value=kind,
                allow_blank=False,
                disabled=record is not None,
                id="personal-context-kind",
            )
            yield Static("Scope", classes="settings-input-label")
            yield Select(
                tuple((scope.label, scope.scope.scope_id) for scope in linked_scopes),
                value=scope_value,
                allow_blank=False,
                disabled=record is not None,
                id="personal-context-scope",
            )
            if record is not None:
                yield Static(
                    "Scope is fixed while editing. Create a new record to use another scope.",
                    classes="settings-inline-guidance",
                )
            yield Static("Subject", classes="settings-input-label")
            yield Input(
                value=subject, placeholder="Subject", id="personal-context-subject"
            )
            yield Static("Value", classes="settings-input-label")
            yield Input(
                value=str(value),
                placeholder="Value or outcome",
                id="personal-context-value",
            )
            yield Static("Polarity", classes="settings-input-label")
            yield Select(
                (("Like", "like"), ("Dislike", "dislike")),
                value=getattr(payload, "polarity", "like"),
                allow_blank=False,
                disabled=kind != "preference",
                id="personal-context-polarity",
            )
            yield Static("Syncability", classes="settings-input-label")
            yield Select(
                (
                    ("Syncable", SyncMode.SYNCABLE.value),
                    ("Device only", SyncMode.DEVICE_ONLY.value),
                ),
                value=(
                    record.controls.sync_mode.value
                    if record is not None
                    else SyncMode.SYNCABLE.value
                ),
                allow_blank=False,
                id="personal-context-sync-mode",
            )
            yield Static(
                "An authorized home server can read syncable content. Device-only content stays on this device.",
                id="personal-context-sync-privacy",
                classes="settings-inline-guidance",
            )
            yield Static("Visibility", classes="settings-input-label")
            yield Select(
                (
                    ("Agent visible", AgentVisibility.AGENT_VISIBLE.value),
                    ("User only", AgentVisibility.USER_ONLY.value),
                ),
                value=(
                    record.controls.agent_visibility.value
                    if record is not None
                    else AgentVisibility.AGENT_VISIBLE.value
                ),
                allow_blank=False,
                id="personal-context-visibility",
            )
            yield Static("Retention", classes="settings-input-label")
            yield Select(
                (
                    ("Preserve current expiry", "preserve"),
                    ("Expire after 30 days", "30_days"),
                    ("No expiry", "no_expiry"),
                    ("Not applicable", "not_applicable"),
                ),
                value=retention,
                allow_blank=False,
                disabled=kind != "working_context",
                id="personal-context-retention",
            )
            yield Static(
                "", id="personal-context-editor-error", classes="settings-inline-error"
            )
            with Horizontal(classes="personal-context-toolbar"):
                yield Button("Save", id="personal-context-save", variant="primary")
                yield Button("Cancel", id="personal-context-cancel")

    @on(Button.Pressed)
    def handle_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if not button_id.startswith("personal-context-"):
            return
        event.stop()
        if button_id == "personal-context-reload":
            self.load_records(retry_locked=True)
        elif button_id == "personal-context-runtime":
            self._toggle_runtime()
        elif button_id == "personal-context-add":
            self.action_add_record()
        elif button_id == "personal-context-edit":
            self.action_edit_record()
        elif button_id == "personal-context-delete":
            self.action_delete_record()
        elif button_id == "personal-context-archive-restore":
            self._archive_or_restore()
        elif button_id == "personal-context-cancel":
            self.editor_mode = ""
        elif button_id == "personal-context-save":
            self._save_editor()
        elif button_id == "personal-context-create":
            self._run_mutation(
                self._require_service().create_profile, "Profile created."
            )
        elif button_id == "personal-context-start-fresh":
            self._confirm_start_fresh()
        elif button_id == "personal-context-finish-removal":
            self._run_mutation(
                self._require_service().finish_secure_removal,
                "Secure removal finished.",
            )
        elif button_id == "personal-context-remove-local":
            self._confirm_remove_local()
        elif button_id == "personal-context-export-plaintext":
            self.action_export_profile()
        elif button_id == "personal-context-export-recovery":
            self._start_recovery_export()
        elif button_id == "personal-context-run-interview":
            self.action_run_interview()
        elif button_id == "personal-context-link-server":
            self.action_link_server()
        elif button_id.startswith("personal-context-proposal-"):
            self._review_proposal_index(button_id)
        elif button_id.startswith("personal-context-record-"):
            self._select_record_index(button_id)

    def _proposal_label(self, proposal) -> str:
        payload = (
            proposal.proposed_record.payload
            if proposal.proposed_record is not None
            else None
        )
        target = self._proposal_target(proposal)
        subject = _bounded_label(
            getattr(
                payload
                if payload is not None
                else (target.payload if target else None),
                "subject",
                "target unavailable",
            )
        )
        scope_label = next(
            (
                scope.label
                for scope in (self.snapshot.scopes if self.snapshot else ())
                if scope.scope.scope_id == proposal.scope_id
            ),
            "Unknown scope",
        )
        return f"{proposal.operation.value.title()} · {subject} · {scope_label}"

    def _proposal_target(self, proposal) -> ProfileRecord | None:
        """Resolve only the exact agent-visible target named by a proposal."""

        if self.snapshot is None or proposal.proposed_record is not None:
            return None
        return next(
            (
                record
                for record in self.snapshot.records
                if record.record_id == proposal.target_record_id
                and record.version_id == proposal.base_version_id
                and record.scope_id == proposal.scope_id
                and record.controls.agent_visibility is AgentVisibility.AGENT_VISIBLE
            ),
            None,
        )

    def _review_proposal_index(self, button_id: str) -> None:
        if self.snapshot is None:
            return
        try:
            index = int(button_id.rsplit("-", 1)[1])
            proposal = self.snapshot.proposals[index]
        except (IndexError, ValueError):
            return
        scope_label = next(
            (
                scope.label
                for scope in self.snapshot.scopes
                if scope.scope.scope_id == proposal.scope_id
            ),
            "Unknown scope",
        )
        try:
            proposal_service = self._require_service().proposal_service()
        except Exception:
            self.notify(
                "Proposal review is unavailable in this Settings session.",
                severity="warning",
            )
            return
        self.app.push_screen(
            PersonalContextProposalReviewModal(
                proposal_service,
                proposal=proposal,
                scope_label=scope_label,
                target_record=self._proposal_target(proposal),
            ),
            callback=self._proposal_review_finished,
        )

    def _proposal_review_finished(self, result: ProposalReviewResult | None) -> None:
        self.load_records()
        if result is None:
            return
        copy = (
            "Proposal accepted." if result.state == "accepted" else "Proposal rejected."
        )
        self.notify(copy, severity="information")

    def _select_record_index(self, button_id: str) -> None:
        if self.snapshot is None:
            return
        try:
            index = int(button_id.rsplit("-", 1)[1])
            self.selected_record_id = self._visible_records()[index].record_id
        except (IndexError, ValueError):
            return

    @on(Select.Changed)
    def handle_select_changed(self, event: Select.Changed) -> None:
        select_id = event.select.id or ""
        if select_id == "personal-context-scope-filter":
            if event.value is Select.BLANK:
                return
            try:
                current_filter = self.query_one(
                    "#personal-context-scope-filter", Select
                )
            except QueryError:
                return
            if event.select is not current_filter:
                return
            self.selected_scope_id = str(event.value)
            visible_records = self._visible_records()
            visible_ids = {record.record_id for record in visible_records}
            if self.selected_record_id not in visible_ids:
                self.selected_record_id = (
                    visible_records[0].record_id if visible_records else ""
                )
            self._refresh_settings_shortcuts()
            return
        if select_id == "personal-context-interview-mode":
            if event.value is not Select.BLANK:
                self.interview_mode = str(event.value)
            return
        if select_id == "personal-context-kind":
            if event.value is Select.BLANK:
                return
            try:
                polarity = self.query_one("#personal-context-polarity", Select)
                retention = self.query_one("#personal-context-retention", Select)
            except QueryError:
                return
            kind = str(event.value)
            polarity.disabled = kind != "preference"
            retention.disabled = kind != "working_context"
            if self.editor_mode != "edit":
                retention.value = (
                    "30_days" if kind == "working_context" else "not_applicable"
                )
            return
        if not select_id.startswith("personal-context-authority-"):
            return
        if self.snapshot is None or event.value is Select.BLANK:
            return
        try:
            index = int(select_id.rsplit("-", 1)[1])
            scope_id = self.snapshot.scopes[index].scope.scope_id
            authority = AgentAuthority(str(event.value))
        except (IndexError, ValueError):
            return
        if self.snapshot.scopes[index].authority is authority:
            return
        expected_policy_version_id = self.snapshot.scopes[index].policy_version_id
        self._run_mutation(
            lambda: self._require_service().set_scope_authority(
                scope_id,
                authority,
                expected_policy_version_id=expected_policy_version_id,
            ),
            "Scope authority updated.",
        )

    def action_add_record(self) -> None:
        if self.snapshot is None:
            self.notify("Create a profile before adding records.", severity="warning")
            return
        if not any(scope.linked for scope in self.snapshot.scopes):
            self.notify(
                "No linked scope is available. Relink a workspace before adding records.",
                severity="warning",
            )
            return
        self.editor_mode = "add"

    def _selected_interview_scope(self):
        if self.snapshot is None:
            return None
        if self.selected_scope_id != _ALL_SCOPES:
            scope = self._scope_snapshot(self.selected_scope_id)
            return scope if scope is not None and scope.linked else None
        selected = self._selected_record()
        if selected is not None:
            scope = self._scope_snapshot(selected.scope_id)
            return scope if scope is not None and scope.linked else None
        return next(
            (
                scope
                for scope in self.snapshot.scopes
                if scope.linked and scope.scope.kind.value == "global"
            ),
            None,
        )

    def action_run_interview(self) -> None:
        """Launch a re-interview for the currently selected eligible scope."""

        scope = self._selected_interview_scope()
        if scope is None:
            self.notify(
                "Select a linked global or workspace scope before running an interview.",
                severity="warning",
            )
            return
        launcher = self._interview_launcher
        if launcher is None:
            launcher = getattr(self.app, "launch_personal_context_interview", None)
        if not callable(launcher):
            self.notify(
                "Interview setup is unavailable in this Settings session.",
                severity="warning",
            )
            return
        try:
            selected_mode = self.query_one(
                "#personal-context-interview-mode", Select
            ).value
        except QueryError:
            selected_mode = self.interview_mode
        mode = (
            str(selected_mode)
            if selected_mode is not Select.BLANK
            else self.interview_mode
        )
        kind = "global" if scope.scope.kind.value == "global" else "workspace"
        if kind == "global":
            kind = "personal"
        try:
            launcher(kind, scope.scope.scope_id, mode)
        except Exception:
            self.notify(
                "Interview setup is unavailable in this Settings session.",
                severity="warning",
            )

    def _link_launcher_available(self) -> bool:
        if callable(self._link_launcher):
            return True
        try:
            return callable(getattr(self.app, "launch_personal_context_link", None))
        except Exception:
            return False

    def action_link_server(self) -> None:
        """Open the app-owned reviewed first-link flow."""

        launcher = self._link_launcher
        if launcher is None:
            launcher = getattr(self.app, "launch_personal_context_link", None)
        if not callable(launcher):
            self.notify(
                "Profile linking requires an authenticated home server.",
                severity="warning",
            )
            return
        try:
            launcher()
        except Exception:
            self.notify(
                "Profile linking could not start. Check the active server and try again.",
                severity="error",
            )

    def action_edit_record(self) -> None:
        record = self._selected_record()
        if record is None:
            self.notify("Select a profile record first.", severity="warning")
            return
        scope = self._scope_snapshot(record.scope_id)
        if scope is None or not scope.linked:
            self.notify(
                "This record's workspace is no longer linked, so it is browse and export only. Relink it before editing.",
                severity="warning",
            )
            return
        self.editor_mode = "edit"

    def action_delete_record(self) -> None:
        record = self._selected_record()
        if record is None:
            self.notify("Select a profile record first.", severity="warning")
            return
        if not self._record_is_linked(record):
            self.notify(
                "This unlinked workspace record is browse and export only. Relink it before deleting.",
                severity="warning",
            )
            return
        self.run_worker(
            self._delete_after_confirmation(record),
            group="personal-context-delete-confirm",
            exclusive=True,
            exit_on_error=False,
        )

    def action_export_profile(self) -> None:
        self.run_worker(
            self._choose_plaintext_export(),
            group="personal-context-export-picker",
            exclusive=True,
            exit_on_error=False,
        )

    def _toggle_runtime(self) -> None:
        if self.snapshot is None:
            return
        enabled = not self.snapshot.status.runtime_enabled
        self._run_mutation(
            lambda: self._require_service().set_runtime_enabled(enabled),
            "Agent use enabled." if enabled else "Agent use disabled.",
        )

    def _archive_or_restore(self) -> None:
        record = self._selected_record()
        if record is None:
            return
        if not self._record_is_linked(record):
            self.notify(
                "This unlinked workspace record is browse and export only. Relink it before changing its state.",
                severity="warning",
            )
            return
        if record.state is RecordState.ARCHIVED:

            def operation() -> Any:
                return self._require_service().restore_record(
                    record.record_id, expected_version_id=record.version_id
                )

            message = "Record restored."
        else:

            def operation() -> Any:
                return self._require_service().archive_record(
                    record.record_id, expected_version_id=record.version_id
                )

            message = "Record archived."
        self._run_mutation(operation, message)

    def _save_editor(self) -> None:
        current = self._selected_record() if self.editor_mode == "edit" else None
        try:
            kind = str(self.query_one("#personal-context-kind", Select).value)
            scope_id = str(self.query_one("#personal-context-scope", Select).value)
            subject = self.query_one("#personal-context-subject", Input).value.strip()
            value = self.query_one("#personal-context-value", Input).value.strip()
            polarity = str(self.query_one("#personal-context-polarity", Select).value)
            sync_mode = SyncMode(
                str(self.query_one("#personal-context-sync-mode", Select).value)
            )
            visibility = AgentVisibility(
                str(self.query_one("#personal-context-visibility", Select).value)
            )
            retention = str(self.query_one("#personal-context-retention", Select).value)
            payload = self._payload(kind, subject, value, polarity)
        except (QueryError, TypeError, ValueError):
            self._show_editor_error("Enter a valid subject and value.")
            return
        if current is not None and kind != current.kind.value:
            self._show_editor_error(
                "Kind cannot be changed while editing. Create a new record for another kind."
            )
            return
        if current is not None and scope_id != current.scope_id:
            self._show_editor_error(
                "Scope cannot be changed while editing. Create a new record in the other scope."
            )
            return
        scope = self._scope_snapshot(current.scope_id if current else scope_id)
        if scope is None or not scope.linked:
            self._show_editor_error(
                "This workspace is no longer linked. Relink it before saving."
            )
            return
        controls = ProfileControls(sync_mode=sync_mode, agent_visibility=visibility)
        semantic_key = (
            None
            if kind == "legacy_unclassified"
            else SemanticKey(namespace=kind, subject=subject)
        )
        expires_at = None
        no_expiry: bool | None = None
        if kind == "working_context":
            if retention == "no_expiry":
                no_expiry = True
            elif retention == "30_days" and current is not None:
                expires_at = self._require_service().clock() + timedelta(days=30)
        if current is None:

            def operation() -> Any:
                return self._require_service().create_manual_record(
                    scope_id=scope_id,
                    payload=payload,
                    semantic_key=semantic_key,
                    controls=controls,
                    expires_at=expires_at,
                    no_expiry=no_expiry is True,
                )

        else:
            mutation = RecordMutation(
                payload=payload,
                semantic_key=semantic_key,
                clear_semantic_key=kind == "legacy_unclassified",
                controls=controls,
                expires_at=expires_at,
                no_expiry=no_expiry,
            )

            def operation() -> Any:
                return self._require_service().update_record(
                    current.record_id,
                    mutation,
                    expected_version_id=current.version_id,
                )

        self._run_mutation(operation, "Profile record saved.", close_editor=True)

    @staticmethod
    def _payload(kind: str, subject: str, value: str, polarity: str):
        if kind == "legacy_unclassified":
            return LegacyUnclassifiedPayload(text=value)
        if kind == "preference":
            return PreferencePayload(subject=subject, value=value, polarity=polarity)
        payload_types = {
            "identity": IdentityPayload,
            "relationship": RelationshipPayload,
            "correction": CorrectionPayload,
            "constraint": ConstraintPayload,
            "goal": GoalPayload,
            "convention": ConventionPayload,
            "working_context": WorkingContextPayload,
        }
        payload_type = payload_types[kind]
        key = "outcome" if kind == "goal" else "value"
        return payload_type(**{"subject": subject, key: value})

    def _show_editor_error(self, message: str) -> None:
        try:
            self.query_one("#personal-context-editor-error", Static).update(message)
        except QueryError:
            pass

    def _run_mutation(
        self,
        operation: Callable[[], Any],
        success_message: str,
        *,
        close_editor: bool = False,
        reload_removed_on_failure: bool = False,
    ) -> None:
        self.run_worker(
            lambda: self._perform_mutation(
                operation,
                success_message,
                close_editor,
                reload_removed_on_failure,
            ),
            thread=True,
            group="personal-context-mutation",
            exclusive=True,
            exit_on_error=False,
        )

    def _perform_mutation(
        self,
        operation: Callable[[], Any],
        success_message: str,
        close_editor: bool,
        reload_removed_on_failure: bool,
    ) -> None:
        try:
            operation()
        except ProfileKeyCollisionError:
            self.app.call_from_thread(self._mutation_key_collision)
        except ProfileConflictError:
            self.app.call_from_thread(self._mutation_conflict)
        except Exception:
            if reload_removed_on_failure:
                try:
                    snapshot = self._require_service().settings_snapshot()
                except Exception:
                    snapshot = None
                if (
                    snapshot is not None
                    and snapshot.status.state is ProfileOperationalState.REMOVED
                ):
                    self.app.call_from_thread(self._removal_incomplete, snapshot)
                    return
            self.app.call_from_thread(self._mutation_failed)
        else:
            self.app.call_from_thread(
                self._mutation_succeeded, success_message, close_editor
            )

    def _mutation_key_collision(self) -> None:
        self._show_editor_error(_COLLISION_COPY)
        self.notify(_COLLISION_COPY, severity="warning")

    def _mutation_conflict(self) -> None:
        self.editor_mode = ""
        self.notify(
            "Profile changed elsewhere; reloaded the latest version. Review and try again.",
            severity="warning",
        )
        self.load_records()

    def _mutation_failed(self) -> None:
        self.notify(
            "The profile change could not be saved. Private details were not displayed.",
            severity="error",
        )

    def _removal_incomplete(self, snapshot: PersonalContextSettingsSnapshot) -> None:
        self._load_generation += 1
        self.editor_mode = ""
        self._apply_snapshot(self._load_generation, snapshot)
        self.notify(
            "Local profile content was removed, but secure key deletion is incomplete. Finish secure removal to retry.",
            severity="warning",
        )

    def _mutation_succeeded(self, message: str, close_editor: bool) -> None:
        if close_editor:
            self.editor_mode = ""
        self.notify(message)
        self.load_records()

    async def _delete_after_confirmation(self, record: ProfileRecord) -> None:
        confirmed = await self.app.push_screen_wait(
            ConfirmationDialog(
                title="Delete profile record?",
                message=(
                    f"Delete {self._record_target_label(record)}? This removes the "
                    "record from this profile generation."
                ),
                confirm_label="Delete record",
            )
        )
        if confirmed:
            self._run_mutation(
                lambda: self._require_service().delete_record(
                    record.record_id, expected_version_id=record.version_id
                ),
                "Record deleted.",
            )

    def _record_target_label(self, record: ProfileRecord) -> str:
        scope = self._scope_snapshot(record.scope_id)
        scope_label = scope.label if scope is not None else "Unknown scope"
        payload = record.payload
        subject = getattr(payload, "subject", "note") if payload is not None else "note"
        kind = record.kind.value.replace("_", " ").title()
        return f"{scope_label} · {kind} · {subject}"

    def _confirm_remove_local(self) -> None:
        self.run_worker(
            self._remove_after_confirmation(),
            group="personal-context-remove-confirm",
            exclusive=True,
            exit_on_error=False,
        )

    async def _remove_after_confirmation(self) -> None:
        confirmed = await self.app.push_screen_wait(
            ConfirmationDialog(
                title="Remove local profile?",
                message=(
                    "This is the only local copy. It permanently destroys encrypted "
                    "profile content and keys. This is not Delete Everywhere."
                ),
                confirm_label="Remove local copy",
            )
        )
        if confirmed:
            self._run_mutation(
                lambda: self._require_service().remove_local_profile(
                    confirm_only_copy=True
                ),
                "Local profile removed.",
                reload_removed_on_failure=True,
            )

    def _confirm_start_fresh(self) -> None:
        self.run_worker(
            self._start_fresh_after_confirmation(),
            group="personal-context-fresh-confirm",
            exclusive=True,
            exit_on_error=False,
        )

    async def _start_fresh_after_confirmation(self) -> None:
        confirmed = await self.app.push_screen_wait(
            ConfirmationDialog(
                title="Start a fresh profile?",
                message="Create new encrypted storage and a new profile generation now?",
                confirm_label="Start Fresh",
            )
        )
        if confirmed:
            self._run_mutation(
                self._require_service().start_fresh_profile,
                "Fresh profile created.",
            )

    async def _choose_plaintext_export(self) -> None:
        from ..enhanced_file_picker import EnhancedFileSave

        scope_ids, scope_label = self._selected_export_scope()
        selected = await self.app.push_screen_wait(
            EnhancedFileSave(
                location=str(Path.home()),
                title="Export plaintext profile",
                default_filename="personal-context.json",
                context="personal_context_plaintext_export",
            )
        )
        if selected is None:
            return
        destination = Path(selected)
        confirm_overwrite = destination.exists()
        overwrite_copy = (
            " A file already exists at the selected destination and will be replaced."
            if confirm_overwrite
            else ""
        )
        confirmed = await self.app.push_screen_wait(
            ConfirmationDialog(
                title="Export readable profile data?",
                message=(
                    "Plaintext export is not encrypted and may contain sensitive "
                    "preferences or personal context. Protect the selected file. "
                    f"Export scope: {scope_label}.{overwrite_copy}"
                ),
                confirm_label=(
                    "Replace and export" if confirm_overwrite else "Export plaintext"
                ),
            )
        )
        if confirmed:
            self._run_export(
                lambda: self._require_service().export_plaintext(
                    ExportRequest(
                        destination=destination,
                        confirm_plaintext=True,
                        scope_ids=scope_ids,
                        confirm_overwrite=confirm_overwrite,
                    )
                ),
                "Plaintext profile exported.",
            )

    def _start_recovery_export(self) -> None:
        self.run_worker(
            self._choose_recovery_export(),
            group="personal-context-recovery-picker",
            exclusive=True,
            exit_on_error=False,
        )

    async def _choose_recovery_export(self) -> None:
        from ..enhanced_file_picker import EnhancedFileSave

        selected = await self.app.push_screen_wait(
            EnhancedFileSave(
                location=str(Path.home()),
                title="Export encrypted recovery copy",
                default_filename="personal-context-recovery.json",
                context="personal_context_recovery_export",
            )
        )
        if selected is None:
            return
        destination = Path(selected)
        confirm_overwrite = destination.exists()
        if confirm_overwrite:
            confirmed = await self.app.push_screen_wait(
                ConfirmationDialog(
                    title="Replace existing recovery copy?",
                    message=(
                        "A file already exists at the selected destination and will be replaced."
                    ),
                    confirm_label="Replace file",
                )
            )
            if not confirmed:
                return
        passphrase = await self.app.push_screen_wait(RecoveryPassphraseDialog())
        if passphrase:
            self._run_export(
                lambda: self._require_service().export_recovery(
                    RecoveryExportRequest(
                        destination=destination,
                        passphrase=passphrase,
                        confirm_overwrite=confirm_overwrite,
                    )
                ),
                "Encrypted recovery copy exported.",
            )

    def _run_export(self, operation: Callable[[], Any], message: str) -> None:
        self.run_worker(
            lambda: self._perform_export(operation, message),
            thread=True,
            group="personal-context-export-write",
            exclusive=True,
            exit_on_error=False,
        )

    def _perform_export(self, operation: Callable[[], Any], message: str) -> None:
        try:
            operation()
        except Exception:
            self.app.call_from_thread(
                self.notify,
                "Export failed. Check the selected destination and try again.",
                severity="error",
            )
        else:
            self.app.call_from_thread(self.notify, message)
