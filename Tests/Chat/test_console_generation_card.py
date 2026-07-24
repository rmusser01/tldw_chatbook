"""Image-generation Console card: transcript row + widget + spec build (P2a Task 7).

Covers the three mandatory behaviors named in the task brief plus a details-
render check:

1. A "generation-card" row replaces the plain "image" row for a message
   present in the card-spec map, even when an image spec is ALSO registered
   for the same message (proves precedence, not merely spec omission).
2. The reconcile signature differs when the browsed variant index changes.
3. ``ChatScreen._build_console_image_specs``/``_recent_console_image_messages``
   skip generation messages entirely (no double render, no LRU slot burn).
4. The details block renderable carries every Style/Source/Seed/Prompt/
   Negative field for a sample spec.

Plus self-review coverage: a byte-less browsed variant renders a placeholder
instead of crashing, and suppression never touches a NON-generation image
message.
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image as PILImage
from rich_pixels import Pixels

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    GenerationVariantMeta,
    MessageAttachment,
)
from tldw_chatbook.Chat.console_image_view import ConsoleImageRowSpec
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_generation_card import (
    ConsoleGenerationCard,
    ConsoleGenerationCardSpec,
    generation_card_details_text,
    generation_card_signature,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


def _bare_screen() -> ChatScreen:
    """Build a screen shell for direct helper calls, bypassing ``__init__``.

    Mirrors ``Tests/UI/test_console_character_avatar.py``'s
    ``_bare_console_screen`` pattern -- ``_ensure_console_image_view`` reads
    ``app_instance``/``app_config`` defensively via ``getattr`` specifically
    so tests can call it on an unmounted screen shell like this one.
    """
    return ChatScreen.__new__(ChatScreen)


def _meta(
    *,
    prompt: str = "a red dragon",
    negative_prompt: str = "blurry",
    backend: str = "swarmui",
    model: str | None = "sdxl",
    seed: int | None = 42,
    style: str | None = "cinematic",
) -> GenerationVariantMeta:
    return GenerationVariantMeta(
        prompt=prompt,
        negative_prompt=negative_prompt,
        backend=backend,
        model=model,
        seed=seed,
        style=style,
        params={},
    )


def _pil() -> PILImage.Image:
    return PILImage.new("RGB", (16, 16), (10, 120, 40))


def _png_bytes() -> bytes:
    """Return real (decodable) PNG bytes for cache-decode-path tests."""
    buffer = BytesIO()
    _pil().save(buffer, format="PNG")
    return buffer.getvalue()


def _card_spec(
    message_id: str,
    *,
    browsed_index: int = 0,
    variant_count: int = 1,
    meta: GenerationVariantMeta | None = None,
    mode: str = "pixels",
    decoded: bool = True,
) -> ConsoleGenerationCardSpec:
    pil = _pil() if decoded else None
    return ConsoleGenerationCardSpec(
        message_id=message_id,
        browsed_index=browsed_index,
        variant_count=variant_count,
        meta=meta or _meta(),
        mode=mode,
        pixels=Pixels.from_image(pil) if (decoded and mode == "pixels") else None,
        pil=pil if (decoded and mode == "graphics") else None,
    )


def _generation_message(
    *, variant_count: int = 1, message_id: str | None = None
) -> ConsoleChatMessage:
    """Build a message shaped like ``ConsoleChatStore.append_generation_message``'s output."""
    attachments = tuple(
        MessageAttachment(
            data=_png_bytes(),
            mime_type="image/png",
            display_name="",
            position=index,
        )
        for index in range(variant_count)
    )
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="[image] a red dragon",
        id=message_id or "gen-1",
    )
    message.attachments = attachments
    message.generation_metadata = tuple(_meta() for _ in range(variant_count))
    # `ConsoleChatStore._set_message_attachments` mirrors attachment #0 into
    # the scalar fields; real generation messages always carry this mirror.
    message.image_data = attachments[0].data
    message.image_mime_type = attachments[0].mime_type
    return message


def _plain_image_message(message_id: str = "img-1") -> ConsoleChatMessage:
    return ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="look",
        id=message_id,
        image_data=_png_bytes(),
        image_mime_type="image/png",
    )


def _image_row_spec(message_id: str) -> ConsoleImageRowSpec:
    return ConsoleImageRowSpec(
        message_id=message_id,
        mode="pixels",
        pixels=Pixels.from_image(_pil()),
    )


# --- 1. Card row replaces the image row -------------------------------------


def test_generation_card_row_replaces_image_row():
    transcript = ConsoleTranscript()
    message = _generation_message()
    transcript.set_messages([message])
    # Register BOTH an image spec and a card spec for the same message id --
    # the card must win even when an image spec would otherwise apply,
    # proving true precedence rather than merely "no image spec was set".
    transcript.set_image_specs({message.id: _image_row_spec(message.id)})
    transcript.set_generation_card_specs({message.id: _card_spec(message.id)})

    rows = transcript._transcript_rows()
    kinds = [row.kind for row in rows]

    assert "generation-card" in kinds
    assert "image" not in kinds
    card_row = next(row for row in rows if row.kind == "generation-card")
    assert card_row.key == f"generation-card:{message.id}"
    assert card_row.generation_card_spec is not None


def test_non_generation_image_message_still_renders_image_row():
    """Self-review: suppression must never hide a NON-generation image message."""
    transcript = ConsoleTranscript()
    message = _plain_image_message()
    transcript.set_messages([message])
    transcript.set_image_specs({message.id: _image_row_spec(message.id)})
    # No card spec registered at all for this message.

    rows = transcript._transcript_rows()
    kinds = [row.kind for row in rows]

    assert "image" in kinds
    assert "generation-card" not in kinds


# --- 2. Reconcile signature changes on browse --------------------------------


def test_card_signature_changes_on_browse():
    spec_at_0 = _card_spec("gen-1", browsed_index=0, variant_count=2)
    spec_at_1 = _card_spec("gen-1", browsed_index=1, variant_count=2)

    assert generation_card_signature(spec_at_0) != generation_card_signature(spec_at_1)


def test_transcript_row_signature_changes_on_browse():
    transcript = ConsoleTranscript()
    message = _generation_message(variant_count=2)
    transcript.set_messages([message])

    transcript.set_generation_card_specs(
        {message.id: _card_spec(message.id, browsed_index=0, variant_count=2)}
    )
    first = next(
        row for row in transcript._transcript_rows() if row.kind == "generation-card"
    )

    transcript.set_generation_card_specs(
        {message.id: _card_spec(message.id, browsed_index=1, variant_count=2)}
    )
    second = next(
        row for row in transcript._transcript_rows() if row.kind == "generation-card"
    )

    assert first.signature != second.signature


def test_card_signature_changes_when_browsed_variant_decodes():
    """A placeholder->real-image transition must also flip the signature.

    Unlike the plain image row (omitted from the spec map entirely until
    decoded), a card spec is present from the moment the message appears --
    without a decoded/undecoded bit in the signature the placeholder would
    never rebuild into the real image once decode completes.
    """
    undecoded = _card_spec("gen-1", decoded=False)
    decoded = _card_spec("gen-1", decoded=True)

    assert generation_card_signature(undecoded) != generation_card_signature(decoded)


# --- 3. Image-spec building excludes card messages ---------------------------


def test_image_specs_exclude_card_messages():
    screen = _bare_screen()
    generation_message = _generation_message()
    plain_message = _plain_image_message()
    messages = [generation_message, plain_message]

    recent = screen._recent_console_image_messages(messages)
    assert generation_message not in recent
    assert plain_message in recent

    _state, cache = screen._ensure_console_image_view()
    cache.prepare(generation_message.id, generation_message.image_data)
    cache.prepare(plain_message.id, plain_message.image_data)

    specs = screen._build_console_image_specs(messages)
    assert generation_message.id not in specs
    assert plain_message.id in specs


# --- 4. Details block renders every field -------------------------------------


def test_generation_card_details_text_contains_all_fields():
    meta = _meta(
        prompt="a lighthouse at dusk",
        negative_prompt="oversaturated",
        backend="swarmui",
        seed=7,
        style="watercolor",
    )
    spec = _card_spec("gen-1", browsed_index=1, variant_count=3, meta=meta)

    text = generation_card_details_text(spec)

    assert "Style: watercolor" in text
    assert "Source: swarmui" in text
    assert "Seed: 7" in text
    assert "Prompt: a lighthouse at dusk" in text
    assert "Negative: oversaturated" in text
    assert "2/3" in text  # n/N indicator, 1-based


def test_generation_card_details_text_style_and_seed_fallbacks():
    meta = _meta(style=None, seed=None)
    spec = _card_spec("gen-1", meta=meta)

    text = generation_card_details_text(spec)

    assert "Style: Custom" in text
    assert "Seed: random" in text


def test_generation_card_details_text_negative_seed_is_random():
    meta = _meta(seed=-1)
    spec = _card_spec("gen-1", meta=meta)

    assert "Seed: random" in generation_card_details_text(spec)


def test_generation_card_details_text_omits_indicator_for_single_variant():
    spec = _card_spec("gen-1", browsed_index=0, variant_count=1)

    text = generation_card_details_text(spec)

    assert "1/1" not in text


# --- Widget build: both modes + byte-less placeholder -------------------------


def test_generation_card_widget_builds_for_pixels_and_graphics_modes():
    # Children are only mounted once composed into a running app; call
    # `compose()` directly to inspect them unmounted, mirroring
    # `test_console_native_transcript.py`'s `list(transcript.compose())`.
    pixels_card = ConsoleGenerationCard(_card_spec("gen-1", mode="pixels"))
    assert pixels_card.id == "console-generation-card-gen-1"
    image_widget, details_widget = list(pixels_card.compose())
    assert image_widget.id == "console-generation-card-image-gen-1"
    assert details_widget.id == "console-generation-card-details-gen-1"

    graphics_card = ConsoleGenerationCard(_card_spec("gen-1", mode="graphics"))
    graphics_image_widget = next(iter(graphics_card.compose()))
    assert graphics_image_widget.id == "console-generation-card-image-gen-1"


def test_generation_card_widget_placeholder_for_byteless_variant():
    """Self-review: a not-yet-decoded browsed variant must render a
    placeholder, never raise (e.g. from ``Pixels.from_image(None)``)."""
    spec = _card_spec("gen-1", decoded=False)

    card = ConsoleGenerationCard(spec)

    image_widget = next(iter(card.compose()))
    assert image_widget.has_class("console-generation-card-image-placeholder")
    assert str(image_widget.renderable) == "(image not loaded)"


def test_generation_card_has_bordered_title():
    card = ConsoleGenerationCard(_card_spec("gen-1"))
    assert card.border_title == "Image Generation"
