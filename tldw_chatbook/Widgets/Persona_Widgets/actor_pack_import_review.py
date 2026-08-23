"""Path-free review and consent UI for one staged Actor Pack."""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any

from PIL import Image as PILImage
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from tldw_chatbook.Actor_Packs.importer import (
    ActorPackImportReview,
    ActorPackPortraitPreview,
)
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

try:
    from textual_image.widget import Image as TerminalImage
except ImportError:  # pragma: no cover - optional renderer fallback
    TerminalImage = None  # type: ignore[assignment,misc]


class ActorPackImportReviewDialog(SafeModalDismissMixin, ModalScreen[str | None]):
    """Review an immutable lease and return one explicitly selected action."""

    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]
    SAFE_MODAL_CONTENT = "#actor-pack-import-review"

    DEFAULT_CSS = """
    ActorPackImportReviewDialog {
        align: center middle;
    }

    ActorPackImportReviewDialog #actor-pack-import-review {
        width: 88%;
        max-width: 92;
        height: 90%;
        max-height: 40;
        background: $surface;
        border: tall $accent;
        padding: 1 2;
    }

    ActorPackImportReviewDialog #actor-pack-import-title {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
    }

    ActorPackImportReviewDialog #actor-pack-import-scroll {
        height: 1fr;
        scrollbar-size: 1 1;
    }

    ActorPackImportReviewDialog .actor-pack-import-heading {
        height: 1;
        text-style: bold;
        margin-top: 1;
    }

    ActorPackImportReviewDialog .actor-pack-import-copy {
        height: auto;
        text-wrap: wrap;
    }

    ActorPackImportReviewDialog .actor-pack-import-portrait {
        width: 18;
        height: 7;
        margin-top: 1;
    }

    ActorPackImportReviewDialog #actor-pack-import-actions {
        height: auto;
        min-height: 3;
        margin-top: 1;
        align: right middle;
    }

    ActorPackImportReviewDialog #actor-pack-import-actions Button {
        width: auto;
        min-width: 11;
        margin-left: 1;
    }
    """

    def __init__(
        self,
        review: ActorPackImportReview,
        portrait: ActorPackPortraitPreview,
    ) -> None:
        if (
            type(review) is not ActorPackImportReview
            or type(portrait) is not ActorPackPortraitPreview
        ):
            raise ValueError("actor_pack_import_review_invalid")
        super().__init__()
        self.review = review
        self.portrait = portrait

    def compose(self) -> ComposeResult:
        review = self.review
        with Container(id="actor-pack-import-review"):
            yield Static("Review Actor Pack", id="actor-pack-import-title")
            with VerticalScroll(id="actor-pack-import-scroll"):
                yield Static("Identity", classes="actor-pack-import-heading")
                yield Static(
                    _identity_copy(review),
                    id="actor-pack-import-identity",
                    classes="actor-pack-import-copy",
                    markup=False,
                )
                yield Static("Actor details", classes="actor-pack-import-heading")
                yield Static(
                    _field_copy(review.actor_fields),
                    id="actor-pack-import-actor-fields",
                    classes="actor-pack-import-copy",
                    markup=False,
                )
                yield Static("Portrait", classes="actor-pack-import-heading")
                yield Static(
                    f"{review.portrait.mime_type} · "
                    f"{review.portrait.width}×{review.portrait.height} · "
                    f"{review.portrait.byte_count:,} bytes",
                    id="actor-pack-import-portrait-meta",
                    classes="actor-pack-import-copy",
                    markup=False,
                )
                preview = _portrait_widget(self.portrait)
                if preview is not None:
                    yield preview
                yield Static("Visual effects", classes="actor-pack-import-heading")
                yield Static(
                    "\n".join(
                        f"{_section_label(kind)}: {effect}"
                        for kind, effect in review.section_effects
                    ),
                    id="actor-pack-import-visuals",
                    classes="actor-pack-import-copy",
                    markup=False,
                )
                yield Static(
                    "License and provenance", classes="actor-pack-import-heading"
                )
                yield Static(
                    _metadata_copy(review),
                    id="actor-pack-import-provenance",
                    classes="actor-pack-import-copy",
                    markup=False,
                )
                yield Static("Warnings", classes="actor-pack-import-heading")
                yield Static(
                    _warning_copy(review),
                    id="actor-pack-import-warnings",
                    classes="actor-pack-import-copy",
                    markup=False,
                )
                yield Static("Changes", classes="actor-pack-import-heading")
                yield Static(
                    _difference_copy(review),
                    id="actor-pack-import-differences",
                    classes="actor-pack-import-copy",
                    markup=False,
                )
            with Horizontal(id="actor-pack-import-actions"):
                yield Button(
                    "Cancel",
                    id="actor-pack-import-cancel",
                    classes="console-action-secondary",
                )
                yield Button(
                    "Create new",
                    id="actor-pack-import-create-new",
                    variant="primary",
                )
                yield Button(
                    "Create copy",
                    id="actor-pack-import-create-copy",
                    classes="console-action-secondary",
                )
                yield Button(
                    "Review update",
                    id="actor-pack-import-update-existing",
                    variant="warning",
                )

    def on_mount(self) -> None:
        super().on_mount()
        actions = set(self.review.allowed_actions)
        action_buttons = {
            "create_new": self.query_one("#actor-pack-import-create-new", Button),
            "create_copy": self.query_one("#actor-pack-import-create-copy", Button),
            "update_existing": self.query_one(
                "#actor-pack-import-update-existing", Button
            ),
        }
        for action, button in action_buttons.items():
            button.display = action in actions
        for action in ("create_new", "create_copy", "update_existing"):
            if action in actions:
                action_buttons[action].focus()
                break

    @on(Button.Pressed, "#actor-pack-import-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    @on(Button.Pressed, "#actor-pack-import-create-new")
    def _create_new(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss_safe_once("create_new")

    @on(Button.Pressed, "#actor-pack-import-create-copy")
    def _create_copy(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss_safe_once("create_copy")

    @on(Button.Pressed, "#actor-pack-import-update-existing")
    def _update_existing(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss_safe_once("update_existing")


def _portrait_widget(preview: ActorPackPortraitPreview) -> Any | None:
    if TerminalImage is None:
        return None
    try:
        with PILImage.open(BytesIO(preview.data)) as source:
            image = source.convert("RGBA")
            image.load()
        widget = TerminalImage(image)
        widget.id = "actor-pack-import-portrait-preview"
        widget.add_class("actor-pack-import-portrait")
        return widget
    except Exception:
        return None


def _plain_value(value: object) -> str:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except (RecursionError, TypeError, UnicodeError, ValueError):
        rendered = "Complex value"
    return rendered if len(rendered) <= 512 else f"{rendered[:509]}…"


def _identity_copy(review: ActorPackImportReview) -> str:
    match = {
        "none": "No local UUID match",
        "same_kind": "Matches an existing local actor",
    }.get(review.uuid_match, "Identity unavailable")
    return (
        f"{review.actor_kind.title()} · {match}\nPortable UUID: {review.portable_uuid}"
    )


def _field_copy(fields: tuple[tuple[str, object], ...]) -> str:
    return "\n".join(f"{key}: {_plain_value(value)}" for key, value in fields)


def _section_label(kind: str) -> str:
    return {
        "shared-visual-identity": "Shared Visual Identity",
        "persona-runtime": "Persona Visual",
    }.get(kind, "Visual section")


def _metadata_copy(review: ActorPackImportReview) -> str:
    lines = [f"License {key}: {value}" for key, value in review.license]
    lines.extend(f"Provenance {key}: {value}" for key, value in review.provenance)
    return "\n".join(lines) if lines else "Not provided."


def _warning_copy(review: ActorPackImportReview) -> str:
    return "\n".join(review.warnings) if review.warnings else "No warnings."


def _difference_copy(review: ActorPackImportReview) -> str:
    if not review.differences:
        return (
            "No existing actor to compare."
            if review.uuid_match == "none"
            else "No portable field changes detected."
        )
    return "\n".join(
        f"{item.field_name}: {item.current_value} → {item.incoming_value}"
        for item in review.differences
    )


__all__ = ["ActorPackImportReviewDialog"]
