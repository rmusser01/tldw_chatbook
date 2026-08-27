"""Image-generation Console card: transcript row + widget + spec build (P2a Task 7).

Covers the three mandatory behaviors named in the task brief plus a details-
render check:

1. A "generation-card" row replaces the plain "image" row for a message
   present in the card-spec map, even when an image spec is ALSO registered
   for the same message (proves precedence, not merely spec omission).
2. The reconcile signature differs when the browsed variant index changes.
3. ``ConsoleImageController._build_console_image_specs`` and the screen's
   ``_recent_console_image_messages`` skip generation messages entirely (no
   double render, no LRU slot burn).
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
from Tests.UI.console_controller_stubs import (
    NO_APP,
    stub_image_controller,
    stub_message_controller,
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

    ``_recent_console_image_messages`` moved to ``ConsoleMessageController``
    (wave-3 console decomposition, task 1) and is reached here through
    ``ChatScreen``'s delegation, so this shell needs a ``_message`` that
    ``__init__`` would otherwise have built. That one method is pure (it
    filters a list against ``IMAGE_CACHE_MAX_ENTRIES`` and touches no
    injected seam), so every constructor callable is stubbed to raise --
    a fail-loud guard if a future test in this file starts exercising a
    branch that needs one for real, rather than a silently-wrong no-op.
    """
    screen = ChatScreen.__new__(ChatScreen)
    stub_message_controller(
        screen,
        context="test_console_generation_card._bare_screen",
        # No harness app -- see the sibling note in
        # test_console_rag_settings_modal; declared, not inferred.
        app_instance=NO_APP,
    )
    stub_image_controller(
        screen,
        context="test_console_generation_card._bare_screen",
        app_instance=NO_APP,
        ensure_console_image_view=lambda: screen._ensure_console_image_view(),
        recent_console_image_messages=(
            lambda messages: screen._recent_console_image_messages(messages)
        ),
        console_image_default_mode=lambda: screen._console_image_default_mode,
        console_generation_browse=lambda: screen._console_generation_browse(),
    )
    return screen


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

    rows = transcript._flat_transcript_rows()
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

    rows = transcript._flat_transcript_rows()
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
        row
        for row in transcript._flat_transcript_rows()
        if row.kind == "generation-card"
    )

    transcript.set_generation_card_specs(
        {message.id: _card_spec(message.id, browsed_index=1, variant_count=2)}
    )
    second = next(
        row
        for row in transcript._flat_transcript_rows()
        if row.kind == "generation-card"
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

    specs = screen._image._build_console_image_specs(messages)
    assert generation_message.id not in specs
    assert plain_message.id in specs


def test_hidden_generation_message_keeps_card_spec_for_view_control():
    screen = _bare_screen()
    message = _generation_message(variant_count=3)
    state, _cache = screen._ensure_console_image_view()
    default_mode = screen._console_image_default_mode
    state.set_mode(message.id, "hidden", default=default_mode)

    specs = screen._image._build_generation_card_specs([message])

    assert specs[message.id].mode == "hidden"


# --- 4. Details block renders every field -------------------------------------


def test_generation_card_details_text_contains_all_fields():
    meta = _meta(
        prompt="a lighthouse at dusk",
        negative_prompt="oversaturated",
        backend="swarmui",
        seed=7,
        style="watercolor",
        model="sdxl_base_1.0",
    )
    spec = _card_spec("gen-1", browsed_index=1, variant_count=3, meta=meta)

    text = generation_card_details_text(spec)

    assert "Style: watercolor" in text
    assert "Source: swarmui" in text
    assert "Seed: 7" in text
    assert "Prompt: a lighthouse at dusk" in text
    assert "Negative: oversaturated" in text
    assert "Model: sdxl_base_1.0" in text
    assert "2/3" in text  # n/N indicator, 1-based


def test_generation_card_owns_image_actions():
    card = ConsoleGenerationCard(_card_spec("gen-1", browsed_index=1, variant_count=3))

    ids = [action.action_id for action in card.actions]

    assert ids == [
        "variant-previous",
        "variant-next",
        "keep",
        "toggle-image-view",
        "save-image",
    ]


def test_hidden_generation_card_exposes_only_view_control():
    card = ConsoleGenerationCard(
        _card_spec(
            "gen-hidden",
            browsed_index=1,
            variant_count=3,
            mode="hidden",
            decoded=False,
        )
    )

    children = list(card.compose())

    assert [action.action_id for action in card.actions] == ["toggle-image-view"]
    assert len(children) == 1
    assert children[0].has_class("console-media-card-actions")


def test_generation_card_details_text_omits_model_row_when_absent():
    """task-558 fix round 1: `resolved_model` is only known for some
    backends/requests (e.g. no configured SwarmUI default AND no explicit
    ``:model`` override) -- the Model row must not render "Model: None" or
    an empty value in that case, mirroring the Negative row's own
    conditional-omission idiom."""
    meta = _meta(model=None)
    spec = _card_spec("gen-1", meta=meta)

    text = generation_card_details_text(spec)

    assert "Model" not in text
    # Sanity: omission is targeted -- every other row still renders.
    assert "Style: cinematic" in text


def test_generation_card_details_text_uses_named_style_not_custom():
    """P2b pin: a spec whose meta carries a named style (e.g. a resolved
    ``@style`` template's display name) renders that name, never the
    ``meta.style is None`` fallback of "Custom"."""
    meta = _meta(style="Anime Style")
    spec = _card_spec("gen-1", meta=meta)

    text = generation_card_details_text(spec)

    assert "Style: Anime Style" in text
    assert "Style: Custom" not in text


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


def test_generation_card_details_text_omits_negative_row_when_empty():
    """task-558: `_detail_rows` only appends the Negative row `if
    meta.negative_prompt`; every other test in this file uses the
    default truthy "blurry" negative prompt, so the empty-string branch
    (an unstyled/raw-prompt generation with no negative prompt at all) was
    never exercised."""
    meta = _meta(negative_prompt="")
    spec = _card_spec("gen-1", meta=meta)

    text = generation_card_details_text(spec)

    assert "Negative" not in text
    # Sanity: the omission is targeted -- every other row still renders.
    assert "Prompt: a red dragon" in text
    assert "Style: cinematic" in text


# --- Widget build: both modes + byte-less placeholder -------------------------


def test_generation_card_widget_builds_for_pixels_and_graphics_modes():
    # Children are only mounted once composed into a running app; call
    # `compose()` directly to inspect them unmounted, mirroring
    # `test_console_native_transcript.py`'s `list(transcript.compose())`.
    pixels_card = ConsoleGenerationCard(_card_spec("gen-1", mode="pixels"))
    assert pixels_card.id == "console-generation-card-gen-1"
    image_widget, details_widget, actions_widget = list(pixels_card.compose())
    assert image_widget.id == "console-generation-card-image-gen-1"
    assert details_widget.id == "console-generation-card-details-gen-1"
    assert actions_widget.has_class("console-media-card-actions")

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
