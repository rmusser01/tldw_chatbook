"""No-mount contracts for the Console image/H3 controller."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


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
