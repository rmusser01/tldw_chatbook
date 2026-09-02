"""Recurring-question automation-definition create modal (task-5).

Mirrors `ReminderForm`'s structure (ADR-099 idiom parity: the modal box is
its own scroll container, a docked footer keeps the live preview/errors/
actions visible, Escape triggers a discard guard). The recurring-cron
schedule sub-form reuses `reminder_form.py`'s task-23102 pure helpers
(`preset_to_cron`, `cron_to_preset`, `parse_time_of_day`,
`parse_forgiving_datetime`, timezone helpers) directly -- only the
Textual wiring (widget ids, compose layout) is necessarily duplicated,
since it is bound to this form's own field ids.

v1 is create-only (spec sec 8 / plan task 4 handoff): the Automations tab
has no local-definition listing or edit entry point yet (that is later
program work, spec sec 9), so this form never receives an existing
local row and always authors `mode="create"`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from croniter import croniter
from loguru import logger
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, Select, Static, TextArea

from tldw_chatbook.Scheduling.models import PreviewStatus, ScheduleKind

from .reminder_form import (
    _CURATED_TIMEZONES,
    _DEFAULT_TIMEZONE,
    _TIME_OF_DAY_PRESETS,
    _is_valid_zone,
    detect_system_timezone,
    parse_forgiving_datetime,
    preset_to_cron,
    system_timezone_name,
)

if TYPE_CHECKING:
    from tldw_chatbook.Scheduling.models import AutomationPreview
    from tldw_chatbook.Scheduling.services.scheduling_service import (
        SaveDefinitionOutcome,
        SchedulingService,
    )

#: recurring_question family per v1 scope guard (plan task 4).
_FAMILY = "recurring_question"

#: config.scope.sources vocabulary (recurring_question_scope.py); the v1
#: form offers exactly these three -- collections/tags/saved-searches are
#: deferred (spec sec 8).
_SCOPE_SOURCE_CHECKBOXES: tuple[tuple[str, str], ...] = (
    ("Media", "media_db"),
    ("Notes", "notes"),
    ("Chats", "chats"),
)

_GENERATION_MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Only when something new is found", "optional"),
    ("Always generate a draft", "required"),
    ("Never generate a draft", "disabled"),
)

_FINDING_POLICY_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Balanced findings", "balanced_findings"),
    ("High confidence only", "high_confidence_only"),
)

#: Validation-error `field` values this form can highlight in place, mapped
#: to the id of the Static that renders under the offending widget. Any
#: error whose field is not here (e.g. a payload-shape error this form
#: never produces, or a server field this local port does not know) falls
#: into the form-level error area instead of being silently dropped.
_FIELD_ERROR_WIDGET_IDS: dict[str, str] = {
    "name": "automation-name-error",
    "input.question": "automation-question-error",
    "config.scope.mode": "automation-scope-error",
    "config.scope": "automation-scope-error",
    "config.generation_mode": "automation-generation-mode-error",
    "config.finding_policy.preset": "automation-finding-policy-error",
    "schedule": "automation-schedule-error",
    "schedule.kind": "automation-schedule-error",
}


def _format_occurrence(raw: str) -> str:
    """Render one ISO-8601 `next_occurrences` entry as compact local text."""
    from datetime import datetime

    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return raw
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M")


class AutomationDefinitionForm(ModalScreen):
    """Create modal for a `recurring_question` automation definition."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=False),
    ]

    DEFAULT_CSS = """
    AutomationDefinitionForm {
        align: center middle;
    }

    AutomationDefinitionForm > VerticalScroll {
        width: 84;
        max-width: 100%;
        height: auto;
        max-height: 100%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    #automation-form-footer {
        dock: bottom;
        height: auto;
        background: $surface;
    }

    #automation-scope-sources-group,
    #automation-run-at-group,
    #automation-cron-group,
    #automation-timezone-group,
    #automation-preset-time-group,
    #automation-cron-custom-group,
    #automation-provider-group {
        height: auto;
    }

    #automation-form-errors {
        display: none;
    }

    #automation-question {
        height: 3;
        max-height: 5;
    }

    .form-title {
        text-style: bold;
        text-align: center;
        padding: 0;
    }

    .form-label {
        color: $text-muted;
        padding: 1 0 0 0;
    }

    .form-helper {
        color: $text-muted;
        height: auto;
        padding: 0;
    }

    .form-preview {
        color: $text-muted;
        height: auto;
        min-height: 1;
        padding: 0;
    }

    .error-text {
        color: $error;
        text-style: bold;
        height: auto;
        padding: 0;
    }

    .button-container {
        align: center middle;
        height: auto;
        padding: 0;
        margin-top: 1;
    }

    .button-container Button {
        margin: 0 1;
    }
    """

    def __init__(
        self,
        service: "SchedulingService",
        *,
        available_owners: Sequence[tuple[str, str]] = (("This device", "local"),),
        default_owner: str = "local",
    ) -> None:
        """Initialize the create form.

        Args:
            service: The active `SchedulingService` -- this form calls its
                `preview_definition`/`save_definition` facade directly
                (Task 4), the same way the Preview button and Save need
                live, in-modal feedback that a push_screen callback alone
                cannot provide.
            available_owners: `(label, owner_id)` options for "Runs on".
            default_owner: The owner preselected on open (the current
                screen owner, per spec sec 8).
        """
        super().__init__()
        self._service = service
        self._available_owners = list(available_owners) or [("This device", "local")]
        self._default_owner = default_owner
        self._dirty = False
        self._ready = False

    # -- discard guard (mirrors ReminderForm) --------------------------------

    def action_dismiss(self) -> None:
        self._maybe_discard()

    def _maybe_discard(self) -> None:
        if not self._dirty:
            self.dismiss(None)
            return
        from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

        async def _discard() -> None:
            self.dismiss(None)

        self.app.push_screen(
            ConfirmationDialog(
                title="Discard changes?",
                message="You have unsaved changes in this form.",
                confirm_label="Discard",
                cancel_label="Keep editing",
                confirm_callback=_discard,
            )
        )

    # -- compose --------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="automation-form-box"):
            yield Label("New Recurring Question", classes="form-title")

            yield Label("Runs on:", classes="form-label")
            yield Select(
                self._available_owners,
                allow_blank=False,
                value=self._default_owner,
                id="automation-runs-on",
            )
            yield Static(
                "Create it locally, or on the connected server if one is "
                "available.",
                classes="form-helper",
            )

            yield Label("Name:", classes="form-label")
            yield Input(placeholder="Name this automation…", id="automation-name")
            yield Static("", id="automation-name-error", classes="error-text")

            yield Label("Question:", classes="form-label")
            yield TextArea(id="automation-question")
            yield Static("", id="automation-question-error", classes="error-text")

            yield Label("Scope:", classes="form-label")
            yield Select(
                [
                    ("All readable library sources", "all_searchable_library"),
                    ("Choose specific sources…", "sources"),
                ],
                allow_blank=False,
                value="all_searchable_library",
                id="automation-scope-mode",
            )
            with Vertical(id="automation-scope-sources-group"):
                for label, value in _SCOPE_SOURCE_CHECKBOXES:
                    yield Checkbox(label, value=True, id=f"automation-scope-{value.split('_')[0]}")
            yield Static("", id="automation-scope-error", classes="error-text")

            yield Label("Schedule Kind:", classes="form-label")
            yield Select(
                [(kind.value.replace("_", " ").title(), kind.value) for kind in ScheduleKind],
                allow_blank=False,
                value=ScheduleKind.ONE_TIME.value,
                id="automation-schedule-kind",
            )
            yield Static("", id="automation-schedule-error", classes="error-text")

            with Vertical(id="automation-run-at-group"):
                yield Label("Run at:", classes="form-label")
                yield Input(placeholder="2026-08-28 09:00", id="automation-run-at")
                yield Static(
                    "A local time like 2026-08-28 09:00, or full ISO-8601 with offset.",
                    classes="form-helper",
                )

            with Vertical(id="automation-cron-group"):
                yield Label("Frequency:", classes="form-label")
                yield Select(
                    [
                        ("Every day at…", "daily"),
                        ("Every weekday at…", "weekday"),
                        ("Every Monday at…", "monday"),
                        ("Every hour", "hourly"),
                        ("Custom cron…", "custom"),
                    ],
                    allow_blank=False,
                    value="daily",
                    id="automation-cron-preset",
                )
                with Vertical(id="automation-preset-time-group"):
                    yield Label("Time of day (24-hour):", classes="form-label")
                    yield Input(placeholder="09:00", id="automation-preset-time")
                with Vertical(id="automation-cron-custom-group"):
                    yield Label("Cron Expression:", classes="form-label")
                    yield Input(placeholder="0 9 * * 1", id="automation-cron")

            with Vertical(id="automation-timezone-group"):
                yield Label("Timezone:", classes="form-label")
                yield Select(
                    self._timezone_options(),
                    allow_blank=False,
                    value=self._initial_timezone(),
                    id="automation-timezone",
                )

            yield Label("Generation mode:", classes="form-label")
            yield Select(
                list(_GENERATION_MODE_OPTIONS),
                allow_blank=False,
                value="optional",
                id="automation-generation-mode",
            )
            yield Static("", id="automation-generation-mode-error", classes="error-text")

            yield Label("Finding policy:", classes="form-label")
            yield Select(
                list(_FINDING_POLICY_OPTIONS),
                allow_blank=False,
                value="balanced_findings",
                id="automation-finding-policy",
            )
            yield Static("", id="automation-finding-policy-error", classes="error-text")

            yield Checkbox(
                "Notify me about results", value=True, id="automation-notify"
            )

            with Vertical(id="automation-provider-group"):
                yield Label(
                    "Provider/model pin (optional — leave blank for auto):",
                    classes="form-label",
                )
                with Horizontal():
                    yield Input(placeholder="provider", id="automation-provider")
                    yield Input(placeholder="model", id="automation-model")

            with Vertical(id="automation-form-footer"):
                yield Static("", id="automation-preview-text", classes="form-preview")
                yield Static("", id="automation-form-errors", classes="error-text")
                with Horizontal(classes="button-container"):
                    yield Button("Preview", id="automation-preview-btn")
                    yield Button("Save", variant="success", id="automation-save")
                    yield Button("Cancel", id="automation-cancel")

    def on_mount(self) -> None:
        self.query_one("#automation-preset-time", Input).value = "09:00"
        self.query_one("#automation-cron", Input).value = "0 9 * * *"
        self._update_schedule_field_visibility(ScheduleKind.ONE_TIME.value)
        self._update_preset_field_visibility("daily")
        self._update_scope_field_visibility("all_searchable_library")
        self.call_after_refresh(self._mark_ready)

    def _mark_ready(self) -> None:
        self._dirty = False
        self._ready = True

    # -- timezone helpers (reuse reminder_form's pure functions) -------------

    def _timezone_options(self) -> list[tuple[str, str]]:
        detected = detect_system_timezone()
        zones = [detected or _DEFAULT_TIMEZONE]
        for zone in _CURATED_TIMEZONES:
            if zone not in zones and _is_valid_zone(zone):
                zones.append(zone)
        return [(zone, zone) for zone in zones]

    def _initial_timezone(self) -> str:
        return system_timezone_name()

    # -- field visibility ------------------------------------------------------

    def _update_schedule_field_visibility(self, kind: str) -> None:
        run_at_group = self.query_one("#automation-run-at-group", Vertical)
        cron_group = self.query_one("#automation-cron-group", Vertical)
        tz_group = self.query_one("#automation-timezone-group", Vertical)
        one_time = kind == ScheduleKind.ONE_TIME.value
        run_at_group.display = one_time
        cron_group.display = not one_time
        tz_group.display = not one_time
        shown_group = run_at_group if one_time else cron_group
        self.call_after_refresh(shown_group.scroll_visible)

    def _update_preset_field_visibility(self, preset: str) -> None:
        time_group = self.query_one("#automation-preset-time-group", Vertical)
        custom_group = self.query_one("#automation-cron-custom-group", Vertical)
        time_group.display = preset in _TIME_OF_DAY_PRESETS
        custom_group.display = preset == "custom"

    def _update_scope_field_visibility(self, mode: str) -> None:
        group = self.query_one("#automation-scope-sources-group", Vertical)
        group.display = mode == "sources"

    def _selected_preset(self) -> str:
        return str(self.query_one("#automation-cron-preset", Select).value)

    def _regenerate_preset_cron(self) -> None:
        preset = self._selected_preset()
        if preset == "custom":
            return
        time_text = self.query_one("#automation-preset-time", Input).value
        cron = preset_to_cron(preset, time_text)
        if cron is not None:
            self.query_one("#automation-cron", Input).value = cron

    # -- change handlers --------------------------------------------------------

    def on_select_changed(self, event: Select.Changed) -> None:
        if not self._ready:
            return
        self._dirty = True
        if event.select.id == "automation-schedule-kind":
            self._update_schedule_field_visibility(str(event.value))
        elif event.select.id == "automation-cron-preset":
            preset = str(event.value)
            self._update_preset_field_visibility(preset)
            self._regenerate_preset_cron()
        elif event.select.id == "automation-scope-mode":
            self._update_scope_field_visibility(str(event.value))

    def on_input_changed(self, event: Input.Changed) -> None:
        if not self._ready:
            return
        self._dirty = True
        if event.input.id == "automation-preset-time":
            self._regenerate_preset_cron()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if self._ready:
            self._dirty = True

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if self._ready:
            self._dirty = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "automation-cancel":
            self._maybe_discard()
        elif event.button.id == "automation-preview-btn":
            self._run_preview()
        elif event.button.id == "automation-save":
            self._run_save()

    # -- payload building -------------------------------------------------------

    def _selected_owner(self) -> str:
        return str(self.query_one("#automation-runs-on", Select).value)

    def _scope_payload(self) -> dict[str, Any]:
        mode = str(self.query_one("#automation-scope-mode", Select).value)
        if mode != "sources":
            return {"mode": "all_searchable_library"}
        sources = [
            value
            for label, value in _SCOPE_SOURCE_CHECKBOXES
            if self.query_one(f"#automation-scope-{value.split('_')[0]}", Checkbox).value
        ]
        return {"mode": "sources", "sources": sources}

    def _schedule_payload(self) -> dict[str, Any]:
        kind = str(self.query_one("#automation-schedule-kind", Select).value)
        if kind == ScheduleKind.ONE_TIME.value:
            raw = self.query_one("#automation-run-at", Input).value.strip()
            parsed, _assumed_local = parse_forgiving_datetime(raw)
            return {"kind": "one_time", "run_at": parsed.isoformat() if parsed else raw}

        preset = self._selected_preset()
        if preset == "custom":
            cron = self.query_one("#automation-cron", Input).value.strip()
        else:
            time_text = self.query_one("#automation-preset-time", Input).value
            cron = preset_to_cron(preset, time_text) or ""
        timezone = str(self.query_one("#automation-timezone", Select).value)
        return {"kind": "cron", "cron": cron, "timezone": timezone}

    def _build_payload(self) -> dict[str, Any]:
        """Build a `ScheduledTaskPreviewCreateRequest`-shaped payload.

        v1 is create-only (module docstring): `mode` is always
        `"create"`, and `definition_id`/`definition_version` are never
        sent (the not-allowed-for-create validators would reject them).
        `visibility_policy`/`approval_policy`/`retention_policy` are not
        exposed as v1 fields (spec sec 8) -- omitting them lets the
        family default (`findings_only`) and the ported normalizers'
        own defaults apply, matching a payload that never mentioned them.
        """
        name = self.query_one("#automation-name", Input).value.strip()
        question = self.query_one("#automation-question", TextArea).text.strip()
        notify = bool(self.query_one("#automation-notify", Checkbox).value)
        provider = self.query_one("#automation-provider", Input).value.strip()
        model = self.query_one("#automation-model", Input).value.strip()

        input_fields: dict[str, Any] = {"question": question}
        if provider:
            input_fields["provider"] = provider
        if model:
            input_fields["model"] = model

        return {
            "family": _FAMILY,
            "mode": "create",
            "name": name,
            "input": input_fields,
            "schedule": self._schedule_payload(),
            "config": {
                "scope": self._scope_payload(),
                "generation_mode": str(
                    self.query_one("#automation-generation-mode", Select).value
                ),
                "finding_policy": {
                    "preset": str(
                        self.query_one("#automation-finding-policy", Select).value
                    )
                },
            },
            "notification_policy": {"on_success": notify, "on_failure": notify},
        }

    # -- validation-error rendering -----------------------------------------------

    def _set_validation_errors(self, errors: Sequence[dict[str, Any]]) -> None:
        """Map `{field, code, message}` errors onto their field's error line.

        An error whose `field` this form does not recognize renders in the
        form-level error area instead of being silently dropped -- this
        also covers a server preview's own error codes (Task 4's report:
        not guaranteed byte-identical to the local port), which still
        carry a `field` this form can match on even when the `code`/
        `message` text differs.
        """
        per_field: dict[str, list[str]] = {}
        unmatched: list[str] = []
        for error in errors:
            field = str(error.get("field") or "")
            message = str(error.get("message") or error.get("code") or "Invalid value.")
            widget_id = _FIELD_ERROR_WIDGET_IDS.get(field)
            if widget_id is None:
                unmatched.append(f"{field}: {message}" if field else message)
            else:
                per_field.setdefault(widget_id, []).append(message)

        for widget_id in set(_FIELD_ERROR_WIDGET_IDS.values()):
            error_widget = self.query_one(f"#{widget_id}", Static)
            messages = per_field.get(widget_id, [])
            error_widget.update("\n".join(messages))
            error_widget.display = bool(messages)

        form_errors = self.query_one("#automation-form-errors", Static)
        form_errors.update("\n".join(unmatched))
        form_errors.display = bool(unmatched)

    def _set_form_error(self, message: str) -> None:
        self._set_validation_errors([{"field": "", "message": message}] if message else [])

    def _client_side_schedule_error(self) -> str | None:
        """Same pre-flight schedule checks as `ReminderForm._save` runs.

        `validate_schedule` (the ported server validator) only checks
        `kind`, not kind-specific fields like `run_at`/`cron` -- a real
        server-parity gap, not something to invent extra strictness for.
        This mirrors ReminderForm's own client-side guard instead, so an
        obviously bad run-at/cron is caught before a preview/save round
        trip rather than silently producing a definition with no next run.
        """
        kind = str(self.query_one("#automation-schedule-kind", Select).value)
        if kind == ScheduleKind.ONE_TIME.value:
            raw = self.query_one("#automation-run-at", Input).value.strip()
            if not raw:
                return "Run at is required for one-time automations."
            parsed, _assumed_local = parse_forgiving_datetime(raw)
            if parsed is None:
                return "Run at must be a date and time like 2026-08-28 09:00."
            return None

        preset = self._selected_preset()
        if preset == "custom":
            cron = self.query_one("#automation-cron", Input).value.strip()
        else:
            time_text = self.query_one("#automation-preset-time", Input).value
            cron = preset_to_cron(preset, time_text) or ""
        if not cron:
            return "A frequency or cron expression is required."
        if not croniter.is_valid(cron):
            return "Cron expression is invalid."
        return None

    # -- preview ------------------------------------------------------------------

    def _run_preview(self) -> None:
        schedule_error = self._client_side_schedule_error()
        if schedule_error:
            self._set_validation_errors([{"field": "schedule", "message": schedule_error}])
            return
        payload = self._build_payload()
        owner = self._selected_owner()

        async def _do() -> None:
            await self._preview_async(payload, owner)

        self.run_worker(_do, exclusive=True, group="automation-form-preview")

    async def _preview_async(self, payload: dict[str, Any], owner: str) -> None:
        try:
            preview: "AutomationPreview" = await self._service.preview_definition(
                payload, owner
            )
        except Exception:  # noqa: BLE001 - never let a preview crash the modal
            logger.exception("Automation preview failed")
            self._set_form_error("Preview failed — check the log and try again.")
            return
        self._render_preview(preview)

    def _render_preview(self, preview: "AutomationPreview") -> None:
        self._set_validation_errors(preview.validation_errors or [])
        preview_text = self.query_one("#automation-preview-text", Static)
        if preview.status != PreviewStatus.VALID:
            preview_text.update("")
            return
        occurrences = (preview.schedule_preview or {}).get("next_occurrences") or []
        warnings = [str(w.get("message", "")) for w in (preview.warnings or []) if w.get("message")]
        lines = []
        if occurrences:
            lines.append(
                "Next runs: " + ", ".join(_format_occurrence(o) for o in occurrences[:3])
            )
        else:
            lines.append("Valid.")
        lines.extend(warnings)
        preview_text.update("\n".join(lines))

    # -- save ------------------------------------------------------------------------

    def _run_save(self) -> None:
        schedule_error = self._client_side_schedule_error()
        if schedule_error:
            self._set_validation_errors([{"field": "schedule", "message": schedule_error}])
            return
        payload = self._build_payload()
        owner = self._selected_owner()

        async def _do() -> None:
            await self._save_async(payload, owner)

        self.run_worker(_do, exclusive=True, group="automation-form-save")

    async def _save_async(self, payload: dict[str, Any], owner: str) -> None:
        try:
            outcome: "SaveDefinitionOutcome" = await self._service.save_definition(
                payload, owner
            )
        except Exception:  # noqa: BLE001 - never let a save crash the modal
            logger.exception("Automation save failed")
            self._set_form_error("Save failed — check the log and try again.")
            return
        if outcome.status in ("saved", "queued"):
            self._set_validation_errors([])
            self.dismiss(outcome)
            return
        self._set_validation_errors(outcome.errors or [])
