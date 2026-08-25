"""Captured-context view for Console prompt improvement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Static, TextArea

from .console_composer_bar import ComposerDraftSnapshot, ComposerModelProjection


SYSTEM_ANALYSIS_LABEL = "Let the improver read the current System prompt"
SYSTEM_ANALYSIS_DISCLOSURE = (
    "Used only to improve the draft. It does not change this session."
)
SYSTEM_ANALYSIS_ABSENT_DISCLOSURE = (
    "There is no current System prompt to read. Improvement will use only the "
    "unsent message and will not change this session."
)
SYSTEM_ANALYSIS_ABSENT_TOOLTIP = (
    "Unavailable — this session has no current System prompt."
)


@dataclass(frozen=True)
class ConsolePromptImprovementContext:
    """Immutable user-visible context captured when the workbench opens."""

    session_id: str
    composer_snapshot: ComposerDraftSnapshot = field(repr=False)
    current_user_projection: ComposerModelProjection | None = field(repr=False)
    current_system_prompt: str = field(repr=False)
    current_system_fingerprint: str | None = field(repr=False)
    provider_label: str
    model_label: str
    endpoint_label: str = ""
    model_unavailable_reason: str = ""
    unavailable_recovery: Literal["provider", "draft", "reopen"] = "provider"
    pinned_resolution: object | None = field(default=None, repr=False)


def improvement_provider_summary(context: object) -> str:
    """Return the exact pinned provider target shown beside model actions."""

    provider = str(getattr(context, "provider_label", "") or "Not configured")
    model = str(getattr(context, "model_label", "") or "Not configured")
    endpoint = str(getattr(context, "endpoint_label", "") or "Provider default")
    return f"Provider: {provider} · Model: {model} · Endpoint: {endpoint}"


class ConsolePromptImproveView(Widget):
    """Render the three explicit improvement paths without starting a model."""

    DEFAULT_CSS = """
    ConsolePromptImproveView { width: 100%; height: 1fr; }
    #console-prompts-improve-scroll { width: 100%; height: 1fr; padding: 0 1; }
    .console-prompts-context-heading {
        width: 100%; height: 1; margin-top: 1; text-style: bold;
    }
    .console-prompts-context-preview { width: 100%; height: 5; }
    #console-prompts-improve-options {
        layout: vertical; width: 100%; height: auto; min-height: 9;
    }
    #console-prompts-improve-options Button { width: 100%; min-width: 18; }
    #console-prompts-provider-summary,
    #console-prompts-analysis-context-disclosure,
    #console-prompts-improvement-status { width: 100%; height: auto; min-height: 1; }
    #console-prompts-configure-provider { width: auto; min-width: 24; }
    #console-prompts-improvement-actions { width: 100%; height: 3; }
    #console-prompts-improvement-cancel,
    #console-prompts-improvement-retry,
    #console-prompts-persistence-retry { display: none; }
    """

    def __init__(
        self,
        context: ConsolePromptImprovementContext,
        *,
        model_unavailable_reason: str = "",
        show_configure_provider: bool = False,
        include_system_context: bool | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.context = context
        self.model_unavailable_reason = model_unavailable_reason.strip()
        self.show_configure_provider = show_configure_provider
        self.include_system_context = include_system_context

    def compose(self) -> ComposeResult:
        projection = self.context.current_user_projection
        preview = (
            projection.text
            if projection is not None
            else "Preview unavailable. Remove reserved protected-placeholder text from the draft to use model improvement."
        )
        with VerticalScroll(id="console-prompts-improve-scroll"):
            yield Static(
                "Current System prompt", classes="console-prompts-context-heading"
            )
            yield TextArea(
                self.context.current_system_prompt,
                read_only=True,
                id="console-prompts-current-system",
                classes="console-prompts-context-preview",
            )
            yield Static(
                "Current unsent message", classes="console-prompts-context-heading"
            )
            yield TextArea(
                preview,
                read_only=True,
                id="console-prompts-current-user",
                classes="console-prompts-context-preview",
            )
            yield Static(
                improvement_provider_summary(self.context),
                id="console-prompts-provider-summary",
                markup=False,
            )
            if self.model_unavailable_reason and self.show_configure_provider:
                yield Button(
                    "Configure provider / model",
                    id="console-prompts-configure-provider",
                )
            include_system = self.include_system_context
            if include_system is None:
                include_system = bool(self.context.current_system_prompt)
            has_system = bool(self.context.current_system_prompt)
            analysis_context = Checkbox(
                SYSTEM_ANALYSIS_LABEL,
                value=include_system,
                id="console-prompts-include-system",
                disabled=not has_system,
            )
            if not has_system:
                analysis_context.tooltip = SYSTEM_ANALYSIS_ABSENT_TOOLTIP
            yield analysis_context
            yield Static(
                SYSTEM_ANALYSIS_DISCLOSURE
                if has_system
                else SYSTEM_ANALYSIS_ABSENT_DISCLOSURE,
                id="console-prompts-analysis-context-disclosure",
                markup=False,
            )
            with Horizontal(id="console-prompts-improve-options"):
                for label, button_id in (
                    ("Replace draft automatically", "console-prompts-auto-improve"),
                    (
                        "Analyze and user review (Recommended)",
                        "console-prompts-review-improve",
                    ),
                    (
                        "Build a reusable prompt",
                        "console-prompts-structured-recipe",
                    ),
                ):
                    model_action = button_id != "console-prompts-structured-recipe"
                    disabled = model_action and bool(self.model_unavailable_reason)
                    button = Button(label, id=button_id, disabled=disabled)
                    if disabled:
                        button.tooltip = self.model_unavailable_reason
                    yield button
            yield Static("", id="console-prompts-improvement-status", markup=False)
            with Horizontal(id="console-prompts-improvement-actions"):
                yield Button("Cancel", id="console-prompts-improvement-cancel")
                yield Button("Retry", id="console-prompts-improvement-retry")
                yield Button("Retry save", id="console-prompts-persistence-retry")


__all__ = [
    "ConsolePromptImprovementContext",
    "ConsolePromptImproveView",
    "SYSTEM_ANALYSIS_ABSENT_DISCLOSURE",
    "SYSTEM_ANALYSIS_ABSENT_TOOLTIP",
    "SYSTEM_ANALYSIS_DISCLOSURE",
    "SYSTEM_ANALYSIS_LABEL",
    "improvement_provider_summary",
]
