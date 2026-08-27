"""No-mount contracts for the Console image/H3 controller."""

from __future__ import annotations

import ast
from dataclasses import replace
import inspect
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image as PILImage

from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    GenerationVariantMeta,
    MessageAttachment,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


def _image_bytes(color: tuple[int, int, int] = (0, 0, 0)) -> bytes:
    buffer = BytesIO()
    PILImage.new("RGB", (2, 2), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _generation_meta(prompt: str) -> GenerationVariantMeta:
    return GenerationVariantMeta(
        prompt=prompt,
        negative_prompt="",
        backend="openai",
        model="image-test",
        seed=7,
        style=None,
        params={"size": "small"},
    )


def _image_controller(
    *,
    store: ConsoleChatStore,
    browse: dict[str, int],
):
    from tldw_chatbook.UI.Console_Modules.image import ConsoleImageController

    return ConsoleImageController(
        object(),
        app_instance=object(),
        ensure_console_image_view=lambda: (
            object(),
            SimpleNamespace(evict_session=Mock()),
        ),
        recent_console_image_messages=lambda messages: list(messages),
        console_image_default_mode=lambda: "pixels",
        console_generation_browse=lambda: browse,
        sync_native_console_chat_ui=lambda: None,
        ensure_console_chat_store=lambda: store,
        build_console_provider_selection=lambda: None,
        ensure_console_provider_gateway=lambda: None,
        console_image_preparing=lambda: None,
        current_console_chat_store=lambda: store,
        console_composer_or_none=lambda: None,
        console_visible_draft_session_id=lambda: None,
        append_native_console_system_message=lambda *args, **kwargs: None,
        request_console_control_bar_sync=lambda: None,
        default_console_session_settings=lambda: None,
        clear_console_composer_draft=lambda: None,
    )


def _generation_message(
    store: ConsoleChatStore,
    *,
    session_id: str,
    message_id: str,
):
    message = store.append_message(
        session_id,
        role=ConsoleMessageRole.ASSISTANT,
        content=f"[image] {message_id}",
        message_id=message_id,
        attachments=(
            MessageAttachment(_image_bytes(), "image/png", "first.png", 0),
            MessageAttachment(
                _image_bytes((0, 255, 0)),
                "image/png",
                "second.png",
                1,
            ),
        ),
    )
    live = store._nodes_by_session[session_id][message.id]
    live.generation_metadata = (
        _generation_meta("first prompt"),
        _generation_meta("second prompt"),
    )
    return live


@pytest.mark.unit
def test_console_image_controller_exists_without_dom_access() -> None:
    """The extracted owner is importable and contains no DOM query seam."""
    from tldw_chatbook.UI.Console_Modules.image import ConsoleImageController

    source_path = Path(inspect.getsourcefile(ConsoleImageController) or "")
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    controller = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "ConsoleImageController"
    )
    queried = {
        node.attr
        for node in ast.walk(controller)
        if isinstance(node, ast.Attribute) and node.attr in {"query", "query_one"}
    }

    assert queried == set()


@pytest.mark.unit
def test_image_state_is_read_write_compatible_after_controller_wiring() -> None:
    """Screen compatibility names proxy one controller-owned state object."""
    from tldw_chatbook.UI.Console_Modules.image import ConsoleImageController

    screen = ChatScreen.__new__(ChatScreen)
    state_names = (
        "_imagegen_inflight_sessions",
        "_imagegen_inflight_message_ids",
        "_console_h3_ui_generations",
    )
    for name in state_names:
        with pytest.raises(RuntimeError, match="controller not wired"):
            getattr(screen, name)
        with pytest.raises(RuntimeError, match="controller not wired"):
            setattr(screen, name, object())

    screen._image = ConsoleImageController(
        screen,
        app_instance=object(),
        ensure_console_image_view=lambda: (object(), object()),
        recent_console_image_messages=lambda messages: list(messages),
        console_image_default_mode=lambda: "pixels",
        console_generation_browse=lambda: {},
        sync_native_console_chat_ui=lambda: None,
        ensure_console_chat_store=lambda: None,
        build_console_provider_selection=lambda: None,
        ensure_console_provider_gateway=lambda: None,
        console_image_preparing=lambda: None,
        current_console_chat_store=lambda: None,
        console_composer_or_none=lambda: None,
        console_visible_draft_session_id=lambda: None,
        append_native_console_system_message=lambda *args, **kwargs: None,
        request_console_control_bar_sync=lambda: None,
        default_console_session_settings=lambda: None,
        clear_console_composer_draft=lambda: None,
    )
    assert (
        screen._imagegen_inflight_sessions is screen._image._imagegen_inflight_sessions
    )
    assert (
        screen._imagegen_inflight_message_ids
        is screen._image._imagegen_inflight_message_ids
    )
    assert screen._console_h3_ui_generations is screen._image._console_h3_ui_generations

    replacement: set[str] = {"session-a"}
    screen._imagegen_inflight_sessions = replacement
    assert screen._image._imagegen_inflight_sessions is replacement
    assert "_imagegen_inflight_sessions" not in screen.__dict__


@pytest.mark.unit
def test_image_controller_captures_exact_generated_selection_fence() -> None:
    store = ConsoleChatStore()
    session = store.ensure_session()
    generated = _generation_message(
        store,
        session_id=session.id,
        message_id="generated-in-prefix",
    )
    ordinary = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="ordinary attachment",
        attachments=(MessageAttachment(b"sent", "text/plain", "sent.txt", 0),),
    )
    browse = {generated.id: 1}
    controller = _image_controller(store=store, browse=browse)

    captured = controller.capture_console_fork_image_selections((ordinary, generated))

    assert len(captured) == 1
    assert captured[0].native_message_id == generated.id
    assert captured[0].selected_position == 1
    assert captured[0].browse_revision == 0
    assert len(captured[0].attachment_meta_fingerprint) == 64
    assert controller.validate_console_fork_image_selections(
        (ordinary, generated), captured
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutation",
    ("browse", "keep", "attachment", "generation", "delete-variant", "cleanup"),
)
def test_image_selection_fence_stales_for_each_relevant_media_change(mutation) -> None:
    store = ConsoleChatStore()
    session = store.ensure_session()
    generated = _generation_message(
        store,
        session_id=session.id,
        message_id="generated",
    )
    browse = {generated.id: 1}
    controller = _image_controller(store=store, browse=browse)
    captured = controller.capture_console_fork_image_selections((generated,))

    if mutation == "browse":
        controller._select_console_generation_variant(
            generated, direction="variant-previous"
        )
    elif mutation == "keep":
        # Equal selected facts isolate the monotonic Keep revision from the
        # positional/fingerprint members of the fence.
        generated.attachments = (
            generated.attachments[0],
            replace(generated.attachments[0], position=1),
        )
        generated.generation_metadata = (
            generated.generation_metadata[0],
            generated.generation_metadata[0],
        )
        captured = controller.capture_console_fork_image_selections((generated,))
        controller._keep_console_generation_variant(generated)
        browse[generated.id] = 1
    elif mutation == "attachment":
        generated.attachments = (
            generated.attachments[0],
            replace(
                generated.attachments[1],
                data=_image_bytes((0, 0, 255)),
            ),
        )
    elif mutation == "generation":
        generated.generation_metadata = (
            generated.generation_metadata[0],
            replace(generated.generation_metadata[1], prompt="replacement prompt"),
        )
    elif mutation == "delete-variant":
        generated.attachments = generated.attachments[:1]
        generated.generation_metadata = generated.generation_metadata[:1]
    else:
        controller.invalidate_console_fork_image_selections((generated.id,))

    assert not controller.validate_console_fork_image_selections((generated,), captured)


@pytest.mark.unit
def test_unrelated_generated_image_outside_prefix_does_not_stale_fence() -> None:
    store = ConsoleChatStore()
    session = store.ensure_session()
    included = _generation_message(
        store,
        session_id=session.id,
        message_id="included",
    )
    excluded = _generation_message(
        store,
        session_id=session.id,
        message_id="excluded",
    )
    browse = {included.id: 1, excluded.id: 1}
    controller = _image_controller(store=store, browse=browse)
    captured = controller.capture_console_fork_image_selections((included,))

    controller._select_console_generation_variant(
        excluded, direction="variant-previous"
    )
    excluded.attachments = (
        replace(
            excluded.attachments[0],
            data=_image_bytes((255, 0, 0)),
        ),
        excluded.attachments[1],
    )

    assert controller.validate_console_fork_image_selections((included,), captured)
