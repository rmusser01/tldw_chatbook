"""One-time consent modal for the startup model catalog refresh (ADR-020)."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import HorizontalGroup, VerticalGroup
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class ModelCatalogConsentModal(ModalScreen[bool]):
    """Ask, once, before the app contacts cloud providers for model lists.

    Dismisses with True (user allows the online check) or False (stay
    offline). The answer is persisted by the caller; this modal only
    collects it.
    """

    BINDINGS = [("escape", "dismiss(False)", "Don't check")]
    BUNDLED_CSS = """
    ModelCatalogConsentModal {
        align: center middle;
        background: $background 75%;
    }

    #model-catalog-consent-dialog {
        width: 72;
        height: auto;
        background: $panel;
        border: round $accent;
        padding: 1 2;
    }

    #model-catalog-consent-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #model-catalog-consent-copy {
        height: auto;
        margin-bottom: 1;
    }

    #model-catalog-consent-actions {
        height: 3;
        align-horizontal: right;
    }

    #model-catalog-consent-actions Button {
        min-width: 16;
        height: 3;
        border: none;
        margin-left: 1;
    }
    """

    def compose(self) -> ComposeResult:
        """Build the dialog: title, explanatory copy, and action buttons."""
        with VerticalGroup(id="model-catalog-consent-dialog"):
            yield Static(
                "Check model lists online?",
                id="model-catalog-consent-title",
            )
            yield Static(
                "tldw can fetch up-to-date model lists from your configured "
                "cloud providers (OpenAI, Anthropic, MistralAI, Moonshot, "
                "OpenRouter, QwenCloud, ZAI) at startup. This contacts those "
                "endpoints over the network using your configured API keys.\n\n"
                "Allow this once and it stays on until you turn it off in "
                "Settings. Nothing is checked until you say yes.",
                id="model-catalog-consent-copy",
            )
            with HorizontalGroup(id="model-catalog-consent-actions"):
                yield Button(
                    "Don't check",
                    id="model-catalog-consent-deny",
                    variant="default",
                )
                yield Button(
                    "Check online",
                    id="model-catalog-consent-allow",
                    variant="primary",
                )

    @on(Button.Pressed, "#model-catalog-consent-allow")
    def _allow(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#model-catalog-consent-deny")
    def _deny(self) -> None:
        self.dismiss(False)
