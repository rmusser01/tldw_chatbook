"""Captured-context view for Console prompt improvement."""

from __future__ import annotations

from dataclasses import dataclass, field

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Static, TextArea

from .console_composer_bar import ComposerDraftSnapshot, ComposerModelProjection


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
        width: 100%; height: auto; min-height: 3; margin-top: 1;
    }
    #console-prompts-improve-options Button { width: 1fr; min-width: 18; }
    #console-prompts-provider-summary,
    #console-prompts-improvement-status { width: 100%; height: auto; min-height: 1; }
    #console-prompts-improvement-actions { width: 100%; height: 3; }
    #console-prompts-improvement-cancel,
    #console-prompts-improvement-retry,
    #console-prompts-persistence-retry { display: none; }
    ConsolePromptsModal.-narrow #console-prompts-improve-options {
        layout: vertical; min-height: 9;
    }
    ConsolePromptsModal.-narrow #console-prompts-improve-options Button {
        width: 100%;
    }
    """

    def __init__(
        self,
        context: ConsolePromptImprovementContext,
        *,
        model_unavailable_reason: str = "",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.context = context
        self.model_unavailable_reason = model_unavailable_reason.strip()

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
            yield Checkbox(
                "Include system prompt as analysis context",
                value=bool(self.context.current_system_prompt),
                id="console-prompts-include-system",
                disabled=not bool(self.context.current_system_prompt),
            )
            with Horizontal(id="console-prompts-improve-options"):
                for label, button_id in (
                    ("Analyze and auto-improve", "console-prompts-auto-improve"),
                    ("Analyze and user review", "console-prompts-review-improve"),
                    (
                        "Create or follow a structured recipe",
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
    "improvement_provider_summary",
]
