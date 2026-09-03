"""Console-native settings summary widget."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSettingsReadiness,
    ConsoleSettingsSummaryState,
)
from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard
from tldw_chatbook.Widgets.Console.console_bounded_section import (
    ConsoleBoundedSection,
)


CONSOLE_SETTINGS_BUTTON_HORIZONTAL_PADDING = 2
CONSOLE_SETTINGS_BUTTON_MIN_WIDTH = 9
CONSOLE_SETTINGS_BUTTON_MAX_WIDTH = 14
CONSOLE_SETTINGS_ROW_HEIGHT = 1


@dataclass(frozen=True, slots=True)
class ConsoleReadinessPresentation:
    """Safe fixed copy projected only from typed Console readiness fields."""

    primary_label: str
    detail: str
    action_label: str
    action_target: str
    action_tooltip: str
    provider_row: str
    credential_row: str
    endpoint_row: str
    model_row: str
    generation_row: str


_BLOCKER_COPY = {
    "provider_missing": "choose a provider",
    "provider_unsupported": "provider is not supported",
    "provider_configuration_invalid": "review provider settings",
    "endpoint_invalid": "invalid base URL",
    "endpoint_not_saved": "save the endpoint in Settings",
    "credential_missing": "missing API key",
    "credential_rejected": "credential was rejected",
    "model_missing": "choose a model",
    "endpoint_unreachable": "endpoint unreachable",
    "active_run": "current run is active",
    "readiness_unknown": "review provider settings",
}
_RECOVERY_COPY = {
    "select_provider": ("Choose provider", "console", "Choose a provider for this Console session"),
    "select_supported_provider": ("Choose provider", "console", "Choose a supported provider for this Console session"),
    "review_provider_settings": ("Review settings", "console", "Review this Console session's settings"),
    "configure_endpoint": ("Configure endpoint", "console", "Configure the provider endpoint before sending"),
    "save_endpoint": ("Configure endpoint", "settings", "Save the provider endpoint in Settings"),
    "configure_credential": ("Configure API key", "settings", "Configure the provider API and API key in Settings"),
    "select_model": ("Choose model", "console", "Choose a model for this Console session"),
    "retry_connection": ("Retry connection", "console", "Retry the provider connection"),
    "wait_for_active_run": ("Run active", "hidden", "Wait for the current run to finish"),
}
_ENDPOINT_COPY = {
    "not_tested": "Not tested",
    "testing": "Testing…",
    "reachable": "Reachable",
    "model_listing_unavailable": "Reachable — model listing unavailable",
    "changed_since_test": "Changed since test",
}
_ENDPOINT_FAILURE_COPY = {
    "timeout": "timed out",
    "connection_refused": "connection refused",
    "unauthorized": "authentication required",
    "forbidden": "access forbidden",
    "http_status": "provider returned an error",
    "invalid_payload": "invalid response",
    "connection_error": "connection error",
}
_GENERATION_COPY = {
    "not_tested": "Not tested",
    "testing": "Testing…",
    "succeeded": "Succeeded",
    "changed_since_test": "Changed since test",
}
_GENERATION_FAILURE_COPY = {
    "authentication": "authentication failed",
    "rate_limit": "rate limited",
    "bad_request": "request rejected",
    "timeout": "timed out",
    "connection_error": "connection error",
    "provider_error": "provider error",
}


def build_console_readiness_presentation(
    readiness: ConsoleSettingsReadiness,
) -> ConsoleReadinessPresentation:
    """Map validated readiness codes to safe UI-owned copy."""
    provider = readiness.provider_display_name or "Provider"
    if readiness.operability == "ready_to_send":
        primary = "Ready to send"
        if readiness.credential == "present_unverified":
            primary += " — credential not verified"
        detail = "A send attempt is permitted with these settings."
        action = ("Configure", "hidden", "Configure Console settings")
    else:
        blocker_copy = _BLOCKER_COPY.get(readiness.blocker, "review provider settings")
        if readiness.blocker in {"credential_missing", "credential_rejected"}:
            blocker_copy = f"{provider} {blocker_copy}"
        primary = f"Not ready — {blocker_copy}"
        detail = f"Provider setup needed: {blocker_copy}"
        action = _RECOVERY_COPY.get(
            readiness.recovery_action,
            ("Review settings", "console", "Review this Console session's settings"),
        )
        if readiness.recovery_action == "configure_credential":
            action = (
                "Configure API key",
                "settings",
                f"Configure {provider} API and API key in Settings",
            )
        elif readiness.recovery_action == "save_endpoint":
            action = (
                "Configure endpoint",
                "settings",
                f"Save the {provider} endpoint in Settings",
            )

    credential_value = {
        "missing": "Missing",
        "not_required": "Not required",
        "authenticated": "Authenticated",
        "present_unverified": "Present — not verified",
    }[readiness.credential]
    if readiness.credential == "present_unverified":
        source = {
            "stored": "local config",
            "environment": "environment variable",
            "draft": "unsaved draft",
        }.get(readiness.credential_source)
        if source:
            credential_value += f" ({source})"

    if readiness.endpoint == "unreachable":
        category = _ENDPOINT_FAILURE_COPY.get(
            readiness.endpoint_category, "connection failed"
        )
        endpoint_value = f"Unreachable — {category}"
    else:
        endpoint_value = _ENDPOINT_COPY[readiness.endpoint]

    if readiness.endpoint == "changed_since_test":
        model_value = "Changed since test"
    elif readiness.endpoint == "model_listing_unavailable":
        model_value = "Listing unavailable"
    else:
        model_value = {
            "missing": "Missing",
            "confirmed": "Confirmed",
            "unconfirmed": "Selected — not verified at this endpoint",
        }[readiness.model]

    if readiness.generation == "failed":
        category = _GENERATION_FAILURE_COPY.get(
            readiness.generation_category, "provider error"
        )
        generation_value = f"Failed — {category}"
    else:
        generation_value = _GENERATION_COPY[readiness.generation]

    return ConsoleReadinessPresentation(
        primary_label=primary,
        detail=detail,
        action_label=action[0],
        action_target=action[1],
        action_tooltip=action[2],
        provider_row=f"Provider: {provider}",
        credential_row=f"Credential · {credential_value}",
        endpoint_row=f"Endpoint · {endpoint_value}",
        model_row=f"Model · {model_value}",
        generation_row=f"Generation · {generation_value}",
    )


class ConsoleSettingsSummary(RecomposeCaptureGuard, Vertical):
    """Render compact Console session settings rows."""

    _BODY_ROWS = (
        ("console-settings-provider-row", "provider_row"),
        ("console-settings-model-row", "model_row"),
        ("console-settings-context-row", "context_row"),
        ("console-settings-endpoint-row", "endpoint_row"),
        ("console-settings-credential-row", "credential_row"),
        ("console-settings-transport-row", "transport_row"),
        ("console-settings-sampling-row", "sampling_row"),
        ("console-settings-identity-row", "identity_row"),
    )

    def __init__(
        self,
        state: ConsoleSettingsSummaryState,
        *,
        on_reconcile: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.state = state
        self._on_reconcile = on_reconcile
        self.add_class("console-settings-summary")
        self.styles.height = "auto"
        self.styles.min_height = 0

    def sync_state(self, state: ConsoleSettingsSummaryState) -> None:
        """Refresh the summary from a new state snapshot.

        No-ops when ``state`` equals the currently-applied snapshot
        (task-280: the Console 0.2s tick calls this unconditionally).

        Args:
            state: Latest settings-summary display state.

        Returns:
            None.
        """
        if state == self.state:
            return
        self.state = state
        try:
            for widget_id, value in self._body_row_values():
                row = self.query_one(f"#{widget_id}", Static)
                row.update(value)
                row.display = bool(value)
            button = self.query_one("#console-settings-open", Button)
        except NoMatches:
            self.refresh(recompose=True)
            self.call_after_refresh(self._request_section_reconcile)
            return

        button.label = self._action_label()
        button.tooltip = self._action_tooltip()
        self._apply_button_sizing(button)
        self.call_after_refresh(self._request_section_reconcile)

    def _apply_button_sizing(self, button: Button) -> None:
        button_width = min(
            max(
                len(self._action_label())
                + CONSOLE_SETTINGS_BUTTON_HORIZONTAL_PADDING,
                CONSOLE_SETTINGS_BUTTON_MIN_WIDTH,
            ),
            CONSOLE_SETTINGS_BUTTON_MAX_WIDTH,
        )
        button.styles.width = button_width
        button.styles.min_width = button_width
        button.styles.max_width = button_width
        button.styles.height = CONSOLE_SETTINGS_ROW_HEIGHT
        button.styles.min_height = CONSOLE_SETTINGS_ROW_HEIGHT
        button.styles.max_height = CONSOLE_SETTINGS_ROW_HEIGHT
        button.styles.margin = 0

    @staticmethod
    def _row_text(value: str | None) -> str:
        """Return a Textual-safe settings row label."""
        return value or ""

    @staticmethod
    def _context_row_text(value: str | None) -> str:
        """Name an absent context estimate without implying a failed estimate."""
        text = str(value or "").strip()
        if text.casefold() in {
            "context: unavailable",
            "context: unknown",
            "unavailable",
            "unknown",
            "",
        }:
            return "Context: Not estimated"
        return text

    @staticmethod
    def _endpoint_row_text(value: str | None) -> str:
        """Capitalize the explicit provider-inheritance label consistently."""
        text = str(value or "").strip()
        if text.casefold() == "endpoint: provider default":
            return "Endpoint: Provider default"
        return text

    def _presentation(self) -> ConsoleReadinessPresentation | None:
        readiness = self.state.readiness
        if readiness is None:
            return None
        return build_console_readiness_presentation(readiness)

    def _body_row_values(self) -> tuple[tuple[str, str], ...]:
        """Return ordered body rows for the current bounded presentation."""

        presentation = self._presentation()
        values: list[tuple[str, str]] = []
        if presentation is not None:
            values.append(
                ("console-settings-readiness-row", presentation.primary_label)
            )
        values.extend(
            (
                (
                    "console-settings-provider-row",
                    presentation.provider_row
                    if presentation is not None
                    else self._row_text(self.state.provider_row),
                ),
                ("console-settings-model-row", self._row_text(self.state.model_row)),
                (
                    "console-settings-context-row",
                    self._context_row_text(self.state.context_row),
                ),
                (
                    "console-settings-endpoint-row",
                    presentation.endpoint_row
                    if presentation is not None
                    else self._endpoint_row_text(self.state.endpoint_row),
                ),
                (
                    "console-settings-credential-row",
                    presentation.credential_row
                    if presentation is not None
                    else self._row_text(self.state.credential_row),
                ),
            )
        )
        if presentation is not None:
            values.extend(
                (
                    ("console-settings-model-evidence-row", presentation.model_row),
                    ("console-settings-generation-row", presentation.generation_row),
                )
            )
        values.extend(
            (
                (
                    "console-settings-transport-row",
                    self._row_text(self.state.transport_row),
                ),
                (
                    "console-settings-sampling-row",
                    self._row_text(self.state.sampling_row),
                ),
                (
                    "console-settings-identity-row",
                    self._row_text(self.state.identity_row),
                ),
            )
        )
        return tuple(values)

    def _action_label(self) -> str:
        presentation = self._presentation()
        return presentation.action_label if presentation else self.state.action_label

    def _action_tooltip(self) -> str:
        presentation = self._presentation()
        return presentation.action_tooltip if presentation else self.state.action_tooltip

    def compose(self) -> ComposeResult:
        header = Horizontal(
            id="console-settings-header", classes="console-settings-header"
        )
        header.styles.height = CONSOLE_SETTINGS_ROW_HEIGHT
        header.styles.min_height = CONSOLE_SETTINGS_ROW_HEIGHT
        header.styles.max_height = CONSOLE_SETTINGS_ROW_HEIGHT
        with header:
            title = Static(
                "Conversation settings",
                id="console-settings-title",
                classes="destination-section console-settings-title",
            )
            title.styles.width = "1fr"
            title.styles.min_width = 0
            title.styles.height = CONSOLE_SETTINGS_ROW_HEIGHT
            title.styles.min_height = CONSOLE_SETTINGS_ROW_HEIGHT
            title.styles.max_height = CONSOLE_SETTINGS_ROW_HEIGHT
            yield title

            button = Button(
                self._action_label(),
                id="console-settings-open",
                tooltip=self._action_tooltip(),
                compact=True,
            )
            self._apply_button_sizing(button)
            yield button
        rows = []
        for widget_id, value in self._body_row_values():
            row = Static(
                value,
                id=widget_id,
                classes="console-settings-row",
                markup=False,
            )
            row.display = bool(value)
            rows.append(row)
        yield ConsoleBoundedSection(*rows, section_id="session-settings")

    def _request_section_reconcile(self) -> None:
        """Settle local demand before invalidating the Inspector owner."""

        sections = list(self.query(ConsoleBoundedSection))
        if sections:
            sections[0].request_reconcile()
        if self._on_reconcile is not None:
            self._on_reconcile()
