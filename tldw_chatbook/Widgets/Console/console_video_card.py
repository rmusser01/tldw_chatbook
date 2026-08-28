"""Transcript card for one video-generation message (task-3401.5).

Mirrors ``console_generation_card.py``'s spec/signature/widget layering but
for the ephemeral, name-referenced video model (ADR-044): the card renders
from ``VideoGenerationMetadata`` plus a VideoStore resolution result --
NEVER from bytes in the database. When the file is gone (restart/expiry/LRU
eviction) the SAME card renders the named tombstone with a regenerate
affordance; no exception, no empty row.

Video-specific Play and Save a copy actions live on this card; general
message actions remain in the selected message row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rich.table import Table
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_message_actions import ConsoleMessageAction
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata
from tldw_chatbook.Widgets.Console.console_video_preview import ConsoleVideoPreview

#: Same inline frame color as the rest of native Console's bordered panels
#: (duplicated locally, same rationale as console_generation_card.py).
CARD_BORDER_COLOR = "#6f7782"
CARD_TITLE = "Video Generation"

VideoCardStatus = Literal["ready", "expired"]


@dataclass(frozen=True)
class ConsoleVideoCardSpec:
    """Prebuilt payload for one transcript video-generation card row.

    Attributes:
        message_id: Native Console message ID this card renders.
        meta: The persisted video facts (from ``metadata_json``) -- present
            in BOTH states; it is the tombstone's entire payload.
        status: ``"ready"`` when the VideoStore resolved the file,
            ``"expired"`` otherwise (restart/TTL/LRU eviction).
        file_path: Resolved live path as a string, or ``None``. Rendered
            nowhere (paths are not durable facts, ADR-044) -- carried so the
            action row's play/save actions can act without re-resolving.
    """

    message_id: str
    meta: VideoGenerationMetadata
    status: VideoCardStatus
    file_path: str | None = None


def video_card_signature(spec: ConsoleVideoCardSpec) -> tuple:
    """Return the transcript reconcile-signature tuple for one card spec.

    Covers every render-affecting input: the status flip (the file coming
    into existence or expiring) and the metadata identity. ``file_path``
    itself is excluded -- a same-status path change does not alter the
    render (and keeping the signature path-free keeps it stable across
    store-root moves).
    """
    meta = spec.meta
    return (
        "video-card",
        spec.message_id,
        spec.status,
        (
            meta.name,
            meta.prompt,
            meta.negative_prompt,
            meta.backend,
            meta.model,
            meta.seed,
            meta.duration_seconds,
            meta.ratio,
        ),
    )


def _format_seed(seed: int | None) -> str:
    """Return the display string for a seed ("random" for None/-1)."""
    if seed is None or seed < 0:
        return "random"
    return str(seed)


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:g}s"


def _format_resolution(spec: ConsoleVideoCardSpec) -> str:
    meta = spec.meta
    if meta.width and meta.height:
        return f"{meta.width}x{meta.height}"
    if meta.ratio:
        return meta.ratio
    return "unknown"


def _detail_rows(spec: ConsoleVideoCardSpec) -> list[tuple[str, str]]:
    """Return the ordered (label, value) rows shared by the table and text renders."""
    meta = spec.meta
    rows = [
        ("Name", meta.name),
        ("Source", meta.backend),
        ("Seed", _format_seed(meta.seed)),
        ("Duration", _format_seconds(meta.duration_seconds)),
        ("Resolution", _format_resolution(spec)),
    ]
    if meta.fps is not None:
        rows.append(("FPS", f"{meta.fps:g}"))
    if meta.model:
        # Same rule as the image card: only rendered when known with certainty.
        rows.append(("Model", meta.model))
    rows.append(("Prompt", meta.prompt))
    if meta.negative_prompt:
        rows.append(("Negative", meta.negative_prompt))
    if meta.source_image_message_id:
        rows.append(("Animated from", "a kept image"))
    return rows


def video_card_details_table(spec: ConsoleVideoCardSpec) -> Table:
    """Build the Rich table rendering a card's details block."""
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", style="dim")
    table.add_column()
    for label, value in _detail_rows(spec):
        table.add_row(label, value)
    return table


def video_card_details_text(spec: ConsoleVideoCardSpec) -> str:
    """Return the details block as plain "Label: value" lines (tests/exports)."""
    return "\n".join(f"{label}: {value}" for label, value in _detail_rows(spec))


def video_card_status_line(spec: ConsoleVideoCardSpec) -> str:
    """Return the one-line status header for the card's current state."""
    if spec.status == "ready":
        return "▶ Ready — select for Play / Save a copy / ♻ Regenerate"
    return "⏳ Expired — the ephemeral file is gone; ♻ Regenerate recreates it"


class ConsoleVideoCard(Vertical):
    """Mounted transcript row rendering one video-generation message.

    A bordered panel titled "Video Generation" holding a status line above
    the details block. Every reconcile-signature change (see
    ``video_card_signature``) rebuilds the whole card, matching
    ``ConsoleGenerationCard``'s always-rebuild contract.
    """

    def __init__(
        self,
        spec: ConsoleVideoCardSpec,
        *,
        actions: tuple[ConsoleMessageAction, ...] | None = None,
    ) -> None:
        super().__init__(
            id=f"console-video-card-{spec.message_id}",
            classes="console-video-card",
        )
        self.spec = spec
        available = spec.status == "ready"
        reason = (
            ""
            if available
            else "The ephemeral video file is gone — regenerate to recreate it."
        )
        self.actions = (
            actions
            if actions is not None
            else (
                ConsoleMessageAction("video-play", "Play", available, reason),
                ConsoleMessageAction(
                    "video-save-copy", "Save a copy", available, reason
                ),
            )
        )
        self.border_title = CARD_TITLE
        self.styles.border = ("round", CARD_BORDER_COLOR)

    def compose(self) -> ComposeResult:
        if self.spec.status == "ready" and self.spec.file_path:
            # Silent in-card preview (task-3401.9): paused by default, never
            # decoded until an explicit click. Eligibility/cap failures and a
            # missing av extra render guidance inside the preview area
            # instead of a player -- never an exception.
            from tldw_chatbook.Media_Playback.preview_policy import (
                check_preview_eligibility,
            )

            eligibility = check_preview_eligibility(
                duration_seconds=self.spec.meta.duration_seconds,
                width=self.spec.meta.width,
                height=self.spec.meta.height,
            )
            yield ConsoleVideoPreview(
                self.spec.file_path,
                duration_seconds=self.spec.meta.duration_seconds,
                eligible=eligibility.eligible,
                ineligible_reason=eligibility.reason,
            )
        yield Static(
            video_card_status_line(self.spec),
            id=f"console-video-card-status-{self.spec.message_id}",
            classes=(
                "console-video-card-status"
                if self.spec.status == "ready"
                else "console-video-card-status-expired"
            ),
        )
        yield Static(
            video_card_details_table(self.spec),
            id=f"console-video-card-details-{self.spec.message_id}",
            classes="console-video-card-details",
        )
        buttons = []
        for action in self.actions:
            button = Button(
                action.label,
                id=f"console-message-action-{action.action_id}-{self.spec.message_id}",
                classes="console-media-card-action",
                disabled=not action.enabled,
            )
            button.console_action_id = action.action_id
            button.console_message_id = self.spec.message_id
            if action.disabled_reason:
                button.tooltip = action.disabled_reason
            buttons.append(button)
        yield Horizontal(*buttons, classes="console-media-card-actions")
