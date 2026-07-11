"""Skills destination shell for Agent Skills packs."""

from __future__ import annotations

import inspect
import asyncio
from collections.abc import Mapping
from typing import Any

from loguru import logger
from rich.markup import escape as escape_markup
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Rule, Static

from ...Chat.chat_handoff_models import ChatHandoffPayload
from ...runtime_policy.types import PolicyDeniedError
from ...Utils.input_validation import sanitize_string, validate_text_input
from ...Widgets.destination_workbench import DestinationModeStrip
from ..Navigation.base_app_screen import BaseAppScreen
from .destination_recovery import DestinationRecoveryState, policy_denied_recovery_state


logger = logger.bind(module="SkillsScreen")
SKILLS_LOCAL_PAGE_SIZE = 25
SKILLS_SERVICE_ERROR_COPY = "Skills service unavailable; retry Skills later."
SKILLS_SERVICE_UNAVAILABLE_COPY = "Skills service is unavailable in this runtime."
SKILLS_POLICY_DENIED_FALLBACK_COPY = "Local Skills are blocked by the current runtime policy."
SKILLS_TRUST_REVIEWABLE_STATUSES = {
    "quarantined_modified",
    "quarantined_added",
    "quarantined_deleted",
}
SKILL_TEXT_LIMITS = {
    "name": 64,
    "skill_name": 64,
    "description": 1000,
    "argument_hint": 500,
    "record_id": 256,
    "backend": 32,
    "policy_message": 500,
    "trust_status": 64,
    "trust_reason_code": 128,
}


class SkillTrustPassphraseModal(ModalScreen[str | None]):
    """Prompt for the local skill trust passphrase without logging it."""

    DEFAULT_CSS = """
    SkillTrustPassphraseModal {
        align: center middle;
    }

    #skill-trust-passphrase-modal {
        width: 64;
        height: auto;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    #skill-trust-passphrase-message {
        margin: 1 0;
    }

    #skill-trust-passphrase-input {
        width: 100%;
    }

    #skill-trust-passphrase-error {
        height: auto;
        min-height: 1;
        color: red;
    }

    #skill-trust-passphrase-actions {
        height: 3;
        min-height: 3;
        margin: 1 0 0 0;
        align-horizontal: right;
    }

    #skill-trust-passphrase-cancel,
    #skill-trust-passphrase-submit {
        width: 10;
        min-width: 10;
        height: 3;
        min-height: 3;
    }
    """

    BINDINGS = [("escape", "dismiss", "Cancel")]

    def __init__(self, *, confirm_bootstrap: bool) -> None:
        super().__init__()
        self._confirm_bootstrap = confirm_bootstrap

    def compose(self) -> ComposeResult:
        title = "Bootstrap Local Skill Trust" if self._confirm_bootstrap else "Unlock Local Skill Trust"
        message = (
            "Current local skill files will become the trusted baseline. "
            "Enter the local skill trust passphrase to continue."
            if self._confirm_bootstrap
            else "Enter the local skill trust passphrase to unlock trust checks for this session."
        )
        with Vertical(id="skill-trust-passphrase-modal"):
            yield Static(title, classes="destination-section")
            yield Static(message, id="skill-trust-passphrase-message")
            yield Input(
                password=True,
                id="skill-trust-passphrase-input",
                placeholder="Trust passphrase",
            )
            yield Static("", id="skill-trust-passphrase-error", markup=False)
            with Horizontal(id="skill-trust-passphrase-actions"):
                yield Button("Cancel", id="skill-trust-passphrase-cancel")
                yield Button("Submit", id="skill-trust-passphrase-submit", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#skill-trust-passphrase-input", Input).focus()

    def action_dismiss(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#skill-trust-passphrase-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, "#skill-trust-passphrase-submit")
    def _submit_button(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit()

    @on(Input.Submitted, "#skill-trust-passphrase-input")
    def _submit_input(self, event: Input.Submitted) -> None:
        event.stop()
        self._submit()

    def _submit(self) -> None:
        passphrase = self.query_one("#skill-trust-passphrase-input", Input).value
        if not passphrase:
            self.query_one("#skill-trust-passphrase-error", Static).update(
                "Passphrase cannot be blank."
            )
            return
        self.dismiss(passphrase)


class SkillsScreen(BaseAppScreen):
    """Agent Skills packs, discovery, validation, and attachments."""

    def __init__(self, app_instance: Any, **kwargs: Any) -> None:
        super().__init__(app_instance, "skills", **kwargs)
        self._local_skill_records: tuple[Mapping[str, Any], ...] = ()
        self._skills_lookup_error: str | None = None
        self._skills_lookup_recovery_state: DestinationRecoveryState | None = None
        self._skills_loaded = False
        self._selected_skill_index: int | None = None
        self._active_trust_review: dict[str, Any] | None = None

    def on_mount(self) -> None:
        super().on_mount()
        self._refresh_local_skills_context()

    @work(exclusive=True)
    async def _refresh_local_skills_context(self) -> None:
        records, lookup_error, recovery_state = await self._list_local_skills()
        self._apply_local_skills_context(records, lookup_error, recovery_state)

    def _apply_local_skills_context(
        self,
        records: tuple[Mapping[str, Any], ...],
        lookup_error: str | None = None,
        recovery_state: DestinationRecoveryState | None = None,
    ) -> None:
        self._local_skill_records = records
        self._skills_lookup_error = lookup_error
        self._skills_lookup_recovery_state = recovery_state
        self._skills_loaded = True
        self._ensure_selected_skill()
        self._reconcile_active_trust_review()
        if self.is_mounted:
            self.refresh(recompose=True)

    async def _list_local_skills(
        self,
    ) -> tuple[tuple[Mapping[str, Any], ...], str | None, DestinationRecoveryState | None]:
        service = getattr(self.app_instance, "skills_scope_service", None)
        list_skills = getattr(service, "list_skills", None)
        if not callable(list_skills):
            return (), SKILLS_SERVICE_UNAVAILABLE_COPY, None
        try:
            list_kwargs = {
                "mode": "local",
                "limit": SKILLS_LOCAL_PAGE_SIZE,
                "offset": 0,
                "include_hidden": False,
            }
            if inspect.iscoroutinefunction(list_skills):
                # Async services are expected to cooperate with the loop;
                # to_thread would only offload coroutine *creation*.
                result = await list_skills(**list_kwargs)
            else:
                # Sync service calls may block (local DB); keep them off the
                # UI loop. A sync wrapper may still hand back an awaitable.
                result = await asyncio.to_thread(list_skills, **list_kwargs)
                if inspect.isawaitable(result):
                    result = await result
        except PolicyDeniedError as exc:
            logger.info(
                "Runtime policy denied local Agent Skills listing.",
                action_id=exc.action_id,
                reason_code=exc.reason_code,
            )
            policy_message = self._safe_skill_text(
                exc.user_message,
                fallback=SKILLS_POLICY_DENIED_FALLBACK_COPY,
                max_length=SKILL_TEXT_LIMITS["policy_message"],
            )
            recovery_state = policy_denied_recovery_state(
                exc,
                unavailable_what="Attach local Skills to Console",
                stable_selector="skills-service-error",
                policy_message=policy_message,
            )
            return (), recovery_state.visible_copy, recovery_state
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to load local Agent Skills for Skills destination.",
            )
            return (), SKILLS_SERVICE_ERROR_COPY, None

        skills = result.get("skills") if isinstance(result, Mapping) else None
        records = tuple(record for record in tuple(skills or ()) if isinstance(record, Mapping))
        return records, None, None

    @staticmethod
    def _safe_skill_text(value: Any, fallback: str = "", *, max_length: int = 1000) -> str:
        text = sanitize_string(str(value or ""), max_length=max_length).strip()
        if not text:
            return fallback
        if validate_text_input(text, max_length=max_length, allow_html=False):
            return text
        return fallback

    @classmethod
    def _skill_field(cls, record: Mapping[str, Any], key: str, fallback: str = "") -> str:
        value = record.get(key)
        max_length = SKILL_TEXT_LIMITS.get(key, 1000)
        return cls._safe_skill_text(value, fallback=fallback, max_length=max_length)

    def _skill_name(self, record: Mapping[str, Any]) -> str:
        return self._skill_field(record, "name") or self._skill_field(record, "skill_name", "Untitled skill")

    def _skill_record_id(self, record: Mapping[str, Any]) -> str:
        name = self._skill_name(record)
        return self._skill_field(record, "id") or name

    def _skill_target_id(self, record: Mapping[str, Any]) -> str:
        record_id = self._skill_field(record, "record_id")
        if record_id.startswith("local:") or record_id.startswith("server:"):
            return record_id
        return f"local:skill:{self._skill_record_id(record)}"

    def _skill_validation_errors(self, record: Mapping[str, Any]) -> list[str]:
        errors = record.get("validation_errors")
        if not isinstance(errors, list):
            return []
        return [
            self._safe_skill_text(error, max_length=300)
            for error in errors
            if self._safe_skill_text(error, max_length=300)
        ]

    def _skill_validation_status(self, record: Mapping[str, Any]) -> str:
        status = self._skill_field(record, "validation_status", "valid").lower()
        return "invalid" if status == "invalid" else "valid"

    @staticmethod
    def _plain_text(value: str) -> Text:
        return Text(value)

    def _skill_trust_status(self, record: Mapping[str, Any]) -> str:
        return self._skill_field(record, "trust_status", "trusted").lower() or "trusted"

    def _skill_trust_reason(self, record: Mapping[str, Any]) -> str:
        return self._skill_field(record, "trust_reason_code")

    @staticmethod
    def _skill_trust_blocked(record: Mapping[str, Any]) -> bool:
        return bool(record.get("trust_blocked"))

    def _skill_trust_changed_files(self, record: Mapping[str, Any]) -> list[str]:
        files = record.get("trust_changed_files")
        if not isinstance(files, (list, tuple)):
            return []
        changed_files = []
        for file_name in files:
            safe_file_name = self._safe_skill_text(file_name, max_length=100)
            if safe_file_name:
                changed_files.append(safe_file_name)
        return changed_files

    def _skill_trust_copy(self, record: Mapping[str, Any]) -> str:
        status = self._skill_trust_status(record)
        trust_copy = {
            "trusted": "Trust: trusted baseline",
            "trust_uninitialized": "Trust: not initialized",
            "trust_locked": "Trust: locked",
            "quarantined_modified": "Trust: changed since trusted baseline",
            "quarantined_added": "Trust: new untrusted file",
            "quarantined_deleted": "Trust: trusted file missing",
            "quarantined_manifest_error": "Trust: manifest cannot be verified",
            "quarantined_unsupported_path": "Trust: unsupported file path",
        }
        return trust_copy.get(status, "Trust: blocked")

    def _skill_trust_blocked_copy(self, record: Mapping[str, Any]) -> str:
        status = self._skill_trust_status(record)
        blocked_copy = {
            "trust_uninitialized": (
                "local skill trust has not been initialized. "
                "Next: Bootstrap Trust after reviewing current local skill files."
            ),
            "trust_locked": (
                "local skill trust is locked. "
                "Next: Unlock Trust with the trust passphrase."
            ),
            "quarantined_modified": (
                "local skill files changed since the trusted baseline. "
                "Next: Review Diff, then Trust Reviewed Version."
            ),
            "quarantined_added": (
                "local skill files include new untrusted files. "
                "Next: Review Diff, then Trust Reviewed Version."
            ),
            "quarantined_deleted": (
                "trusted local skill files are missing. "
                "Next: Review Diff, then Trust Reviewed Version."
            ),
            "quarantined_manifest_error": (
                "local skill trust manifest cannot be verified. "
                "Next: resolve local trust state before staging."
            ),
            "quarantined_unsupported_path": (
                "local skill files include unsupported paths. "
                "Next: remove unsupported paths before staging."
            ),
        }
        return blocked_copy.get(status, "local skill trust blocks execution. Next: resolve trust state before staging.")

    def _can_review_skill_trust(self, record: Mapping[str, Any] | None) -> bool:
        if record is None:
            return False
        return (
            self._skill_validation_status(record) == "valid"
            and self._skill_trust_blocked(record)
            and self._skill_trust_status(record) in SKILLS_TRUST_REVIEWABLE_STATUSES
        )

    def _has_trust_status(self, trust_status: str) -> bool:
        return any(self._skill_trust_status(record) == trust_status for record in self._local_skill_records)

    @staticmethod
    def _review_file_label(value: Any) -> str:
        text = str(value or "").replace("\\", "/").strip()
        if not text:
            return ""
        if text.startswith("/") or text.startswith("~") or ":" in text:
            text = text.rsplit("/", 1)[-1]
        text = text.replace("..", "").strip("/ ")
        return sanitize_string(text, max_length=100).strip()

    def _active_review_changed_files(self) -> list[str]:
        if self._active_trust_review is None:
            return []
        files = self._active_trust_review.get("changed_files")
        if not isinstance(files, (list, tuple)):
            return []
        changed_files = []
        for file_name in files:
            safe_file_name = self._review_file_label(file_name)
            if safe_file_name and validate_text_input(safe_file_name, max_length=100, allow_html=False):
                changed_files.append(safe_file_name)
        return changed_files

    def _active_review_matches_selected(self, metadata: Mapping[str, Any] | None = None) -> bool:
        if self._active_trust_review is None:
            return False
        metadata = metadata or self._selected_skill_metadata()
        if not metadata:
            return False
        if not metadata.get("trust_reviewable"):
            return False
        return (
            self._active_trust_review.get("selected_skill_name") == metadata.get("selected_skill_name")
            and self._active_trust_review.get("selected_record_id") == metadata.get("selected_record_id")
            and self._active_trust_review.get("selected_target_id") == metadata.get("selected_target_id")
        )

    def _reconcile_active_trust_review(self) -> None:
        if self._active_trust_review is None:
            return
        if self._skills_lookup_error or not self._local_skill_records:
            self._active_trust_review = None
            return
        if not self._active_review_matches_selected():
            self._active_trust_review = None

    def _is_skill_valid(self, record: Mapping[str, Any]) -> bool:
        return self._skill_validation_status(record) == "valid" and not self._skill_trust_blocked(record)

    def _ensure_selected_skill(self) -> None:
        if not self._local_skill_records or self._skills_lookup_error:
            self._selected_skill_index = None
            return
        if self._selected_skill_index is not None and 0 <= self._selected_skill_index < len(self._local_skill_records):
            return
        for index, record in enumerate(self._local_skill_records):
            if self._is_skill_valid(record):
                self._selected_skill_index = index
                return
        self._selected_skill_index = 0

    def _selected_skill_record(self) -> Mapping[str, Any] | None:
        self._ensure_selected_skill()
        if self._selected_skill_index is None:
            return None
        if not (0 <= self._selected_skill_index < len(self._local_skill_records)):
            return None
        return self._local_skill_records[self._selected_skill_index]

    def _selected_skill_metadata(self) -> dict[str, Any]:
        record = self._selected_skill_record()
        if record is None:
            return {}
        trust_changed_files = self._skill_trust_changed_files(record)
        return {
            "selected_skill_name": self._skill_name(record),
            "selected_record_id": self._skill_record_id(record),
            "selected_target_id": self._skill_target_id(record),
            "validation_status": self._skill_validation_status(record),
            "validation_errors": self._skill_validation_errors(record),
            "stageable": self._is_skill_valid(record),
            "trust_status": self._skill_trust_status(record),
            "trust_reason_code": self._skill_trust_reason(record),
            "trust_blocked": self._skill_trust_blocked(record),
            "trust_changed_files": trust_changed_files,
            "trust_reviewable": self._can_review_skill_trust(record),
        }

    def _skill_body(self, records: tuple[Mapping[str, Any], ...] | None = None) -> str:
        records = records if records is not None else self._local_skill_records
        lines = [
            "Local Agent Skills available to stage into Console:",
            "",
        ]
        for index, record in enumerate(records, start=1):
            name = self._skill_name(record)
            description = self._skill_field(record, "description")
            argument_hint = self._skill_field(record, "argument_hint")
            record_id = self._skill_field(record, "record_id")
            backend = self._skill_field(record, "backend", "local")
            validation_status = self._skill_validation_status(record)

            lines.append(f"{index}. {name}")
            if description:
                lines.append(f"   description: {description}")
            if argument_hint:
                lines.append(f"   argument hint: {argument_hint}")
            if record_id:
                lines.append(f"   record id: {record_id}")
            lines.append(f"   backend: {backend}")
            lines.append(f"   validation: {validation_status}")
            lines.append(f"   {self._skill_trust_copy(record)}")
            trust_reason = self._skill_trust_reason(record)
            if trust_reason:
                lines.append(f"   reason code: {trust_reason}")
            changed_files = ", ".join(self._skill_trust_changed_files(record))
            if changed_files:
                lines.append(f"   changed files: {changed_files}")
            lines.append("")
        return "\n".join(lines).strip()

    def _skill_names(self, records: tuple[Mapping[str, Any], ...] | None = None) -> list[str]:
        records = records if records is not None else self._local_skill_records
        return [
            self._skill_name(record)
            for record in records
        ]

    def _apply_selected_skill_widgets(self) -> None:
        metadata = self._selected_skill_metadata()
        if not metadata:
            return
        selected_name = metadata["selected_skill_name"]
        target_id = metadata["selected_target_id"]
        stageable = bool(metadata["stageable"])
        reason = "; ".join(metadata["validation_errors"]) or "Selected skill is not valid."
        if metadata["validation_status"] == "valid" and metadata["trust_blocked"]:
            reason = self._skill_trust_blocked_copy(self._selected_skill_record() or {})
        updates = {
            "#skills-selected-context": f"Selected: {selected_name}",
            "#skills-selected-runtime-target": f"Runtime target: {target_id}",
            "#skills-execution-readiness": (
                "Execution: ready to stage in Console" if stageable else "Execution: blocked"
            ),
            "#skills-execution-blocked-reason": "" if stageable else f"Reason: {reason}",
            "#skills-selected-trust-reason-code": (
                f"Reason code: {metadata['trust_reason_code']}"
                if metadata["trust_blocked"] and metadata["trust_reason_code"]
                else ""
            ),
        }
        for selector, text in updates.items():
            for widget in self.query(selector):
                if isinstance(widget, Static):
                    widget.update(self._plain_text(text))
                    widget.display = bool(text)
        for button in self.query("#skills-attach-to-console"):
            if isinstance(button, Button):
                button.disabled = not stageable
                button.tooltip = (
                    "Stage selected valid Agent Skill in Console."
                    if stageable
                    else "Resolve SKILL.md validation or local trust blocks before staging this skill in Console."
                )
        for button in self.query("#skills-review-diff"):
            if isinstance(button, Button):
                button.disabled = not bool(metadata["trust_reviewable"])
                button.tooltip = (
                    "Capture the selected skill files for review against the trusted baseline."
                    if metadata["trust_reviewable"]
                    else "Review is available only for metadata-valid local skills blocked by changed files."
                )
        for button in self.query("#skills-trust-reviewed-version"):
            if isinstance(button, Button):
                review_bound = self._active_review_matches_selected(metadata)
                button.disabled = not review_bound
                button.tooltip = (
                    "Trust the captured reviewed skill snapshot."
                    if review_bound
                    else "Capture a trust review before approving this skill version."
                )
        review_bound = self._active_review_matches_selected(metadata)
        review_files = ", ".join(self._active_review_changed_files()) or "selected files"
        review_updates = {
            "#skills-trust-review-title": "Trust Review" if review_bound else "",
            "#skills-trust-review-skill": (
                f"Reviewed skill: {selected_name}" if review_bound else ""
            ),
            "#skills-trust-review-summary": (
                f"Review captured: {review_files}. Confirm these current files should become the trusted baseline."
                if review_bound
                else ""
            ),
        }
        for selector, text in review_updates.items():
            for widget in self.query(selector):
                if isinstance(widget, Static):
                    widget.update(self._plain_text(text))
                    widget.display = bool(text)

    @staticmethod
    def _column_divider(identifier: str) -> Rule:
        divider = Rule(orientation="vertical", id=identifier)
        divider.add_class("destination-pane-divider")
        return divider

    def compose_content(self) -> ComposeResult:
        local_skills_service = getattr(self.app_instance, "local_skills_service", None)
        skills_dir = getattr(local_skills_service, "skills_dir", None)
        skills_dir_label = str(skills_dir) if skills_dir is not None else "Local skills directory unavailable."
        selected_metadata = self._selected_skill_metadata()

        with Vertical(id="skills-shell"):
            yield Static(
                "Skills | Agent Skills packs, validation, Console attachments | Local",
                id="skills-title",
                classes="ds-destination-header",
            )
            with DestinationModeStrip(id="skills-mode-strip", classes="destination-mode-strip"):
                yield Static(
                    "Mode: Installed / Validate / Attach | Source: local SKILL.md directories",
                    id="skills-mode-label",
                    classes="destination-section",
                )
            with Horizontal(id="skills-workbench", classes="ds-panel destination-workbench"):
                with Vertical(id="skills-list-pane", classes="destination-workbench-pane"):
                    yield Static("Skill Library", classes="destination-section skills-column-title")
                    yield Static("Installed", classes="destination-section")
                    yield Static("Discover/Import", classes="destination-section")
                    yield Static("Validate", classes="destination-section")
                    yield Static("Scripts", classes="destination-section")
                    yield Static("References", classes="destination-section")
                    yield Static("Assets", classes="destination-section")
                    yield Static("Attachments", classes="destination-section")
                    yield Static(f"Local skills directory: {skills_dir_label}", id="skills-local-directory")
                yield self._column_divider("skills-list-detail-divider")
                with Vertical(id="skills-detail-pane", classes="destination-workbench-pane"):
                    yield Static(
                        "Skill Detail",
                        classes="destination-section skills-column-title",
                    )
                    if not self._skills_loaded:
                        yield Static(
                            "Loading local Agent Skills...",
                            id="skills-loading-state",
                        )
                        attach_label = "Attach local Skills to Console"
                        attach_disabled = True
                        attach_tooltip = "Stage local skill context after Skills finishes loading."
                    elif self._skills_lookup_error:
                        recovery_state = self._skills_lookup_recovery_state
                        yield Static(
                            self._skills_lookup_error,
                            id=(
                                recovery_state.stable_selector
                                if recovery_state is not None
                                else "skills-service-error"
                            ),
                        )
                        attach_label = "Attach local Skills to Console"
                        attach_disabled = True
                        attach_tooltip = (
                            recovery_state.disabled_tooltip
                            if recovery_state is not None
                            else "Skills service is unavailable; retry Skills later."
                        )
                    elif not self._local_skill_records:
                        yield Static(
                            "No local Agent Skills are installed yet.",
                            id="skills-empty-state",
                        )
                        attach_label = "Attach local Skills to Console"
                        attach_disabled = True
                        attach_tooltip = "Stage local skill context after installing a local Agent Skill."
                    else:
                        yield Static(
                            f"Installed local skills: {len(self._local_skill_records)}",
                            id="skills-local-summary",
                        )
                        for index, record in enumerate(self._local_skill_records):
                            name = self._skill_name(record)
                            description = self._skill_field(record, "description", "No description provided.")
                            metadata_valid = self._skill_validation_status(record) == "valid"
                            validation_copy = (
                                "Ready: valid SKILL.md"
                                if metadata_valid
                                else "Blocked: invalid SKILL.md"
                            )
                            validation_errors = "; ".join(self._skill_validation_errors(record))
                            yield Static(
                                Text.from_markup(
                                    f"{escape_markup(name)} - {escape_markup(description)}"
                                ),
                                id=f"skills-local-skill-{index}",
                            )
                            yield Static(
                                validation_copy,
                                id=f"skills-validation-status-{index}",
                            )
                            if validation_errors:
                                yield Static(
                                    self._plain_text(validation_errors),
                                    id=f"skills-validation-errors-{index}",
                                )
                            yield Static(
                                self._plain_text(self._skill_trust_copy(record)),
                                id=f"skills-trust-status-{index}",
                            )
                            trust_reason = self._skill_trust_reason(record)
                            if trust_reason:
                                yield Static(
                                    self._plain_text(f"Reason code: {trust_reason}"),
                                    id=f"skills-trust-reason-{index}",
                                )
                            changed_files = ", ".join(self._skill_trust_changed_files(record))
                            if changed_files:
                                yield Static(
                                    self._plain_text(f"Changed files: {changed_files}"),
                                    id=f"skills-trust-files-{index}",
                                )
                            yield Button(
                                "Use",
                                id=f"skills-select-local-{index}",
                                classes="skills-select-local",
                                tooltip=f"Use {name} as the Console skill target.",
                            )
                        attach_label = "Attach local Skills to Console"
                        attach_disabled = not (
                            selected_metadata and selected_metadata.get("stageable")
                        )
                        attach_tooltip = (
                            "Stage selected valid Agent Skill in Console."
                            if not attach_disabled
                            else "Resolve SKILL.md validation or local trust blocks before staging this skill in Console."
                        )
                yield self._column_divider("skills-detail-inspector-divider")
                with Vertical(id="skills-inspector-pane", classes="destination-workbench-pane ds-inspector"):
                    yield Static("Skill Inspector", classes="destination-section skills-column-title")
                    if selected_metadata:
                        yield Static(
                            "Selected Console target",
                            id="skills-selected-target-title",
                            classes="destination-section",
                        )
                        yield Static(
                            self._plain_text(f"Selected: {selected_metadata['selected_skill_name']}"),
                            id="skills-selected-context",
                        )
                        yield Static(
                            self._plain_text(f"Runtime target: {selected_metadata['selected_target_id']}"),
                            id="skills-selected-runtime-target",
                        )
                        is_selected_stageable = bool(selected_metadata["stageable"])
                        selected_blocked_reason = (
                            "; ".join(selected_metadata["validation_errors"])
                            or "Selected skill is not valid."
                        )
                        if selected_metadata["validation_status"] == "valid" and selected_metadata["trust_blocked"]:
                            selected_blocked_reason = (
                                self._skill_trust_blocked_copy(self._selected_skill_record() or {})
                            )
                        yield Static(
                            "Execution: ready to stage in Console"
                            if is_selected_stageable
                            else "Execution: blocked",
                            id="skills-execution-readiness",
                        )
                        yield Static(
                            self._plain_text(
                                "" if is_selected_stageable else f"Reason: {selected_blocked_reason}"
                            ),
                            id="skills-execution-blocked-reason",
                        )
                        yield Static(
                            self._plain_text(
                                f"Reason code: {selected_metadata['trust_reason_code']}"
                                if selected_metadata["trust_blocked"] and selected_metadata["trust_reason_code"]
                                else ""
                            ),
                            id="skills-selected-trust-reason-code",
                        )
                        if self._active_review_matches_selected(selected_metadata):
                            review_files = ", ".join(self._active_review_changed_files()) or "selected files"
                            yield Static(
                                "Trust Review",
                                id="skills-trust-review-title",
                                classes="destination-section",
                            )
                            yield Static(
                                self._plain_text(
                                    f"Reviewed skill: {selected_metadata['selected_skill_name']}"
                                ),
                                id="skills-trust-review-skill",
                            )
                            yield Static(
                                self._plain_text(
                                    f"Review captured: {review_files}. Confirm these current files should become the trusted baseline."
                                ),
                                id="skills-trust-review-summary",
                            )
                    yield Static("Actions", classes="destination-section")
                    yield Static(
                        "Skill import is not wired in this shell yet.",
                        id="skills-import-unavailable",
                    )
                    yield Button(
                        "Import Skill",
                        id="skills-import-skill",
                        disabled=True,
                        tooltip="Unavailable until skill import is wired in this shell.",
                    )
                    yield Button(
                        "Bootstrap Trust",
                        id="skills-bootstrap-trust",
                        disabled=not self._has_trust_status("trust_uninitialized"),
                        tooltip=(
                            "Create the first trusted baseline from current local skill files."
                            if self._has_trust_status("trust_uninitialized")
                            else "Bootstrap is available when local skill trust has not been initialized."
                        ),
                    )
                    yield Button(
                        "Unlock Trust",
                        id="skills-unlock-trust",
                        disabled=not self._has_trust_status("trust_locked"),
                        tooltip=(
                            "Unlock local skill trust with the trust passphrase for this session."
                            if self._has_trust_status("trust_locked")
                            else "Unlock is available when local skill trust is locked."
                        ),
                    )
                    review_enabled = bool(selected_metadata and selected_metadata.get("trust_reviewable"))
                    yield Button(
                        "Review Diff",
                        id="skills-review-diff",
                        disabled=not review_enabled,
                        tooltip=(
                            "Capture the selected skill files for review against the trusted baseline."
                            if review_enabled
                            else "Review is available only for metadata-valid local skills blocked by changed files."
                        ),
                    )
                    review_active = self._active_review_matches_selected(selected_metadata)
                    yield Button(
                        "Trust Reviewed Version",
                        id="skills-trust-reviewed-version",
                        disabled=not review_active,
                        tooltip=(
                            "Trust the captured reviewed skill snapshot."
                            if review_active
                            else "Capture a trust review before approving this skill version."
                        ),
                    )
                    yield Button(
                        attach_label,
                        id="skills-attach-to-console",
                        disabled=attach_disabled,
                        tooltip=attach_tooltip,
                    )

    @on(Button.Pressed, ".skills-select-local")
    def select_local_skill(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = str(event.button.id or "")
        prefix = "skills-select-local-"
        if not button_id.startswith(prefix):
            return
        try:
            index = int(button_id.removeprefix(prefix))
        except ValueError:
            return
        if not (0 <= index < len(self._local_skill_records)):
            return
        if self._selected_skill_index != index:
            self._active_trust_review = None
        self._selected_skill_index = index
        self._apply_selected_skill_widgets()

    def _notify_skill_trust_warning(self, message: str) -> None:
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(message, severity="warning")

    async def _reload_local_skills_context(self) -> None:
        records, lookup_error, recovery_state = await self._list_local_skills()
        self._apply_local_skills_context(records, lookup_error, recovery_state)

    async def _request_skill_trust_passphrase(self, confirm_bootstrap: bool) -> str | None:
        push_screen_wait = getattr(self.app, "push_screen_wait", None)
        if not callable(push_screen_wait):
            self._notify_skill_trust_warning("Local skill trust passphrase prompt is unavailable.")
            return None
        result = await push_screen_wait(
            SkillTrustPassphraseModal(confirm_bootstrap=confirm_bootstrap)
        )
        if isinstance(result, str) and result:
            return result
        return None

    async def _call_skill_trust_service(
        self,
        method_name: str,
        *args: Any,
    ) -> tuple[Any, bool]:
        trust_service = getattr(self.app_instance, "local_skill_trust_service", None)
        method = getattr(trust_service, method_name, None)
        if not callable(method):
            self._notify_skill_trust_warning("Local skill trust service is unavailable.")
            return None, False
        try:
            if inspect.iscoroutinefunction(method):
                result = await method(*args)
            else:
                result = await asyncio.to_thread(method, *args)
                if inspect.isawaitable(result):
                    result = await result
        except Exception as exc:
            logger.warning(
                "Local skill trust action failed.",
                action=method_name,
                error_type=type(exc).__name__,
            )
            self._notify_skill_trust_warning(
                "Local skill trust action could not be completed."
            )
            return None, False
        return result, True

    async def _handle_skill_trust_passphrase_action(
        self,
        *,
        method_name: str,
        confirm_bootstrap: bool,
    ) -> None:
        passphrase = await self._request_skill_trust_passphrase(
            confirm_bootstrap=confirm_bootstrap
        )
        if passphrase is None:
            self._notify_skill_trust_warning("Local skill trust action cancelled.")
            return
        _, ok = await self._call_skill_trust_service(method_name, passphrase)
        if ok:
            self._active_trust_review = None
            await self._reload_local_skills_context()

    @on(Button.Pressed, "#skills-bootstrap-trust")
    async def bootstrap_skill_trust(self, event: Button.Pressed) -> None:
        event.stop()
        await self._handle_skill_trust_passphrase_action(
            method_name="bootstrap_trust",
            confirm_bootstrap=True,
        )

    @on(Button.Pressed, "#skills-unlock-trust")
    async def unlock_skill_trust(self, event: Button.Pressed) -> None:
        event.stop()
        await self._handle_skill_trust_passphrase_action(
            method_name="unlock_with_passphrase",
            confirm_bootstrap=False,
        )

    @on(Button.Pressed, "#skills-review-diff")
    async def review_skill_trust_diff(self, event: Button.Pressed) -> None:
        event.stop()
        selected_record = self._selected_skill_record()
        if not self._can_review_skill_trust(selected_record):
            self._notify_skill_trust_warning(
                "Select a metadata-valid trust-blocked skill with changed files before reviewing."
            )
            return
        skill_name = self._skill_name(selected_record)
        selected_metadata = self._selected_skill_metadata()
        result, ok = await self._call_skill_trust_service("capture_review", skill_name)
        if not ok:
            return
        if not isinstance(result, Mapping) or not result.get("review_id"):
            self._notify_skill_trust_warning(
                "Local skill trust review could not be captured."
            )
            return
        self._active_trust_review = {
            **dict(result),
            "selected_skill_name": selected_metadata.get("selected_skill_name"),
            "selected_record_id": selected_metadata.get("selected_record_id"),
            "selected_target_id": selected_metadata.get("selected_target_id"),
        }
        self._apply_selected_skill_widgets()
        await self._reload_local_skills_context()

    @on(Button.Pressed, "#skills-trust-reviewed-version")
    async def trust_reviewed_skill_version(self, event: Button.Pressed) -> None:
        event.stop()
        if self._active_trust_review is None:
            self._notify_skill_trust_warning(
                "Capture a local skill trust review before approving this version."
            )
            return
        if not self._active_review_matches_selected():
            self._active_trust_review = None
            self._notify_skill_trust_warning(
                "Capture a fresh local skill trust review before approving this version."
            )
            self._apply_selected_skill_widgets()
            if self.is_mounted:
                self.refresh(recompose=True)
            return
        review_id = self._safe_skill_text(
            self._active_trust_review.get("review_id"),
            max_length=128,
        )
        if not review_id:
            self._notify_skill_trust_warning(
                "Local skill trust review could not be approved."
            )
            return
        _, ok = await self._call_skill_trust_service(
            "trust_reviewed_snapshot",
            review_id,
        )
        if ok:
            self._active_trust_review = None
            await self._reload_local_skills_context()

    @on(Button.Pressed, "#skills-attach-to-console")
    def attach_to_console(self, event: Button.Pressed) -> None:
        event.stop()
        selected_record = self._selected_skill_record()
        if selected_record is None:
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(
                    "No local Agent Skills are available to stage in Console.",
                    severity="warning",
                )
            return
        if not self._is_skill_valid(selected_record):
            if (
                self._skill_validation_status(selected_record) == "valid"
                and self._skill_trust_blocked(selected_record)
            ):
                warning = "Resolve the local skill trust block before staging this skill in Console."
            else:
                warning = "Fix SKILL.md validation errors before staging this skill in Console."
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(
                    warning,
                    severity="warning",
                )
            return
        open_chat_with_handoff = getattr(self.app_instance, "open_chat_with_handoff", None)
        if not callable(open_chat_with_handoff):
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(
                    "Console handoff is unavailable for Skills in this runtime.",
                    severity="warning",
                )
            return
        selected_records = (selected_record,)
        skill_count = 1
        skill_names = self._skill_names(selected_records)
        selected_metadata = self._selected_skill_metadata()
        selected_name = selected_metadata["selected_skill_name"]
        open_chat_with_handoff(
            ChatHandoffPayload(
                source="skills",
                item_type="skills-context",
                title=f"Local Agent Skill: {selected_name}",
                body=self._skill_body(selected_records),
                display_summary=f"Local Agent Skill {selected_name} staged.",
                suggested_prompt=f"Use the {selected_name} Agent Skill for the next response.",
                runtime_backend="local",
                source_owner="local",
                source_selector_state="local",
                metadata={
                    "skill_count": skill_count,
                    "skill_names": skill_names,
                    **selected_metadata,
                    "backend": "local",
                    "skills_dir": str(
                        getattr(
                            getattr(self.app_instance, "local_skills_service", None),
                            "skills_dir",
                            "",
                        )
                        or ""
                    ),
                },
            )
        )
