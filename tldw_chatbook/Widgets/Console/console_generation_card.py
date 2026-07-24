"""Console image-generation transcript card: spec, signature, and widget.

Mirrors ``console_image_view.py``'s ``ConsoleImageRowSpec``/render-cache
integration for the plain inline-image row, but for the richer
"Image Generation" card: a browsed variant's image plus a details block
(style/source/seed/prompt/negative) and an ``n/N`` variant indicator.

The screen (``chat_screen.py``) builds ``ConsoleGenerationCardSpec`` values
from store messages carrying non-empty ``generation_metadata`` and its own
ephemeral browse state, then hands them to
``ConsoleTranscript.set_generation_card_specs``. The transcript emits a
``"generation-card"`` row (this module's ``ConsoleGenerationCard`` widget)
INSTEAD of the plain ``"image"`` row for any message present in that
mapping -- see ``console_transcript.py``'s ``_transcript_rows``.

Browsing/keep controls (the action row that lets a reader step between
variants or promote one to canonical) are a later task; this module renders
state only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from loguru import logger
from PIL import Image as PILImage
from rich.table import Table
from rich_pixels import Pixels
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Static

from tldw_chatbook.Chat.console_chat_models import GenerationVariantMeta
from tldw_chatbook.Chat.console_image_view import (
    PIXELS_MAX_COLS,
    PIXELS_MAX_LINES,
    fit_image_cell_size,
)

#: Same inline frame color the rest of native Console uses for bordered
#: panels (``CONSOLE_FRAME_COLOR`` in ``chat_screen.py``) -- duplicated
#: locally rather than imported to avoid a screen<->widget circular import.
CARD_BORDER_COLOR = "#6f7782"
CARD_TITLE = "Image Generation"


@dataclass(frozen=True)
class ConsoleGenerationCardSpec:
    """Prebuilt payload for one transcript image-generation card row.

    The render fields (``mode``/``pixels``/``pil``) mirror
    ``ConsoleImageRowSpec`` exactly, but describe the BROWSED variant only
    -- the decoded image for that variant rides the same
    ``ConsoleImageRenderCache`` under the composite cache key
    ``f"{message_id}:{browsed_index}"`` (a generation message can browse
    between several decoded variants sharing one render cache instance).

    Attributes:
        message_id: Native Console message ID this card renders.
        browsed_index: Currently browsed variant index (0-based), clamped
            to ``[0, variant_count)`` by the builder.
        variant_count: Total number of generated variants for this message.
        meta: Generation metadata for the browsed variant
            (``message.generation_metadata[browsed_index]``).
        mode: Inline image render mode for the browsed variant
            ("pixels" or "graphics" -- a card is never "hidden"; the
            screen simply omits hidden-mode messages from the spec map).
        pixels: Prebuilt ``rich_pixels.Pixels`` for "pixels" mode, or
            ``None`` when not yet decoded or mode is "graphics".
        pil: Decoded PIL image for "graphics" mode, or ``None`` when not
            yet decoded or mode is "pixels".
    """

    message_id: str
    browsed_index: int
    variant_count: int
    meta: GenerationVariantMeta
    mode: Literal["pixels", "graphics"]
    pixels: "Pixels | None" = None
    pil: "PILImage.Image | None" = None


def generation_card_signature(spec: ConsoleGenerationCardSpec) -> tuple:
    """Return the transcript reconcile-signature tuple for one card spec.

    Covers every render-affecting input: a browse/keep change
    (``browsed_index``, ``variant_count``), a view-mode toggle (``mode``),
    the browsed variant's generation-metadata identity, and whether the
    browsed variant's image has finished decoding yet. That last bit is
    load-bearing: unlike the plain image row (which the screen omits from
    its spec map entirely until decoded, so its mere appearance changes the
    transcript's row set), a card spec is present -- and its row exists --
    from the moment the message appears, rendering a placeholder until the
    browsed variant decodes. Without a decoded/undecoded bit in the
    signature, that transition would never flip the signature and the
    placeholder would never rebuild into the real image. ``meta.params`` is
    deliberately excluded -- it is never rendered in the details block and
    (being a plain ``dict``) is not guaranteed hashable/orderable, so
    including it would risk an unstable or crashing signature.

    Args:
        spec: The card spec to fingerprint.

    Returns:
        A hashable tuple; any render-affecting change alters it.
    """
    meta = spec.meta
    meta_identity = (
        meta.prompt,
        meta.negative_prompt,
        meta.backend,
        meta.model,
        meta.seed,
        meta.style,
    )
    decoded = spec.pixels is not None or spec.pil is not None
    return (
        "generation-card",
        spec.message_id,
        spec.browsed_index,
        spec.variant_count,
        spec.mode,
        meta_identity,
        decoded,
    )


def _format_seed(seed: int | None) -> str:
    """Return the display string for a variant's seed ("random" for None/-1)."""
    if seed is None or seed < 0:
        return "random"
    return str(seed)


def _detail_rows(spec: ConsoleGenerationCardSpec) -> list[tuple[str, str]]:
    """Return the ordered (label, value) rows shared by the table and text renders."""
    meta = spec.meta
    rows = [
        ("Style", meta.style or "Custom"),
        ("Source", meta.backend),
        ("Seed", _format_seed(meta.seed)),
        ("Prompt", meta.prompt),
    ]
    if meta.negative_prompt:
        rows.append(("Negative", meta.negative_prompt))
    return rows


def generation_card_details_table(spec: ConsoleGenerationCardSpec) -> Table:
    """Build the Rich table rendering a card's details block.

    Args:
        spec: The card spec to render.

    Returns:
        A borderless grid table with one row per detail field, plus a
        trailing ``Variant n/N`` row when ``variant_count > 1``.
    """
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", style="dim")
    table.add_column()
    for label, value in _detail_rows(spec):
        table.add_row(label, value)
    if spec.variant_count > 1:
        table.add_row("Variant", f"{spec.browsed_index + 1}/{spec.variant_count}")
    return table


def generation_card_details_text(spec: ConsoleGenerationCardSpec) -> str:
    """Return the details block as plain "Label: value" lines (tests/exports).

    Args:
        spec: The card spec to render.

    Returns:
        One ``"Label: value"`` line per detail field, plus a trailing
        ``"n/N"`` line when ``variant_count > 1``.
    """
    lines = [f"{label}: {value}" for label, value in _detail_rows(spec)]
    if spec.variant_count > 1:
        lines.append(f"{spec.browsed_index + 1}/{spec.variant_count}")
    return "\n".join(lines)


def _generation_card_image_widget(spec: ConsoleGenerationCardSpec) -> Widget:
    """Build the browsed-variant image widget for one card.

    Mirrors ``ConsoleTranscript._image_row_widget``'s graphics/pixels
    fallback logic exactly, with one addition: when neither a decoded
    ``pil`` nor prebuilt ``pixels`` is available (the browsed variant's
    bytes have not been rehydrated/prepared yet), this renders a plain
    placeholder line instead of crashing on ``Pixels.from_image(None)``.

    Args:
        spec: The card spec describing the browsed variant's render state.

    Returns:
        A mounted-ready widget for the image area of the card.
    """
    widget: Widget | None = None
    if spec.mode == "graphics" and spec.pil is not None:
        try:
            from textual_image.widget import Image as _GraphicsImage

            widget = _GraphicsImage(
                spec.pil, id=f"console-generation-card-image-{spec.message_id}"
            )
            w_cells, h_cells = fit_image_cell_size(
                spec.pil.width, spec.pil.height, PIXELS_MAX_COLS, PIXELS_MAX_LINES
            )
            widget.styles.width = w_cells
            widget.styles.height = h_cells
        except Exception:
            logger.opt(exception=True).warning(
                "textual-image unavailable; falling back to pixels for generation card."
            )
            widget = None
    if widget is None:
        pixels = spec.pixels
        if pixels is None and spec.pil is not None:
            scaled = spec.pil.copy()
            scaled.thumbnail(
                (PIXELS_MAX_COLS, PIXELS_MAX_LINES * 2), PILImage.Resampling.LANCZOS
            )
            pixels = Pixels.from_image(scaled)
        if pixels is None:
            # Byte-less variant (not yet rehydrated/decoded): render a
            # placeholder line rather than crashing.
            widget = Static(
                "(image not loaded)",
                id=f"console-generation-card-image-{spec.message_id}",
                classes="console-generation-card-image-placeholder",
            )
        else:
            widget = Static(
                pixels, id=f"console-generation-card-image-{spec.message_id}"
            )
            widget.styles.max_width = PIXELS_MAX_COLS
            widget.styles.max_height = PIXELS_MAX_LINES
    widget.add_class("console-generation-card-image")
    return widget


class ConsoleGenerationCard(Vertical):
    """Mounted transcript row rendering one image-generation message.

    A bordered panel titled "Image Generation" holding the browsed
    variant's image above a details block (Style/Source/Seed/Prompt/
    Negative, plus an ``n/N`` indicator when the message has more than one
    variant). Every reconcile-signature change (see
    ``generation_card_signature``) rebuilds the whole card -- there is no
    partial in-place update, matching the existing plain-image row's
    always-rebuild-on-signature-change behavior in
    ``ConsoleTranscript._build_row_widget``/``_update_row_widget``.
    """

    def __init__(self, spec: ConsoleGenerationCardSpec) -> None:
        super().__init__(
            id=f"console-generation-card-{spec.message_id}",
            classes="console-generation-card",
        )
        self.spec = spec
        self.border_title = CARD_TITLE
        self.styles.border = ("round", CARD_BORDER_COLOR)

    def compose(self) -> ComposeResult:
        yield _generation_card_image_widget(self.spec)
        yield Static(
            generation_card_details_table(self.spec),
            id=f"console-generation-card-details-{self.spec.message_id}",
            classes="console-generation-card-details",
        )
