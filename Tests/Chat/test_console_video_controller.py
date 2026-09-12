"""No-mount contracts for the Console generated-video controller."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from Tests.UI.console_controller_stubs import NO_APP, stub_message_controller
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


def _controller_type():
    try:
        module = importlib.import_module("tldw_chatbook.UI.Console_Modules.video")
    except ModuleNotFoundError:
        pytest.fail("ConsoleVideoController module has not been extracted")
    return module.ConsoleVideoController


def _controller(*, app_instance: object | None = None):
    controller_type = _controller_type()
    return controller_type(
        app_instance=app_instance or object(),
        sync_native_console_chat_ui=lambda: None,
        ensure_console_chat_store=lambda: None,
        wait_for_console_screen_result=lambda _screen: None,
        open_video_with_os=lambda _path: None,
        append_native_console_system_message=lambda *_args, **_kwargs: None,
        default_console_session_settings=lambda: None,
        console_composer_or_none=lambda: None,
        clear_console_composer_draft=lambda: None,
    )


@pytest.mark.unit
def test_console_video_controller_exists_without_dom_access() -> None:
    """The extracted owner is importable and contains no DOM query seam."""
    controller_type = _controller_type()
    source_path = Path(inspect.getsourcefile(controller_type) or "")
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    controller = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "ConsoleVideoController"
    )
    queried = {
        node.attr
        for node in ast.walk(controller)
        if isinstance(node, ast.Attribute) and node.attr in {"query", "query_one"}
    }

    assert queried == set()


@pytest.mark.unit
def test_video_state_is_read_write_compatible_after_controller_wiring() -> None:
    """All eight screen names proxy one controller-owned state object."""
    screen = ChatScreen.__new__(ChatScreen)
    state_names = (
        "_console_videogen_inflight",
        "_console_videogen_cancels",
        "_console_video_store",
        "_pending_video_artifacts",
        "_pending_video_artifacts_closed",
        "_pending_video_operation_cancels",
        "_pending_video_active_operations",
        "_pending_video_deferred_closes",
    )
    for name in state_names:
        with pytest.raises(RuntimeError, match="controller not wired"):
            getattr(screen, name)
        with pytest.raises(RuntimeError, match="controller not wired"):
            setattr(screen, name, object())

    screen._video = _controller()
    for name in state_names:
        assert getattr(screen, name) is getattr(screen._video, name)

    replacement: dict[str, object] = {"message": object()}
    screen._pending_video_artifacts = replacement
    assert screen._video._pending_video_artifacts is replacement
    assert "_pending_video_artifacts" not in screen.__dict__


@pytest.mark.unit
def test_video_publication_gate_identity_is_controller_owned() -> None:
    """The exact registered gate is reused by the pending operation."""
    controller = _controller()
    artifact = SimpleNamespace(message_id="message")
    gate = controller._register_console_video_publication_gate("message")
    controller._pending_video_artifacts["message"] = artifact

    assert controller._pending_video_operation_cancels["message"] is gate
    assert controller._begin_pending_console_video_operation(artifact) is gate


@pytest.mark.asyncio
async def test_registry_video_commands_delegate_to_controller() -> None:
    """Framework-bound screen methods are executable one-hop delegates."""
    screen = ChatScreen.__new__(ChatScreen)
    screen._video = SimpleNamespace(
        _console_command_generate_video=AsyncMock(),
        _console_command_stream_video=AsyncMock(),
    )
    parse = SimpleNamespace(args="prompt")

    await screen._video._console_command_generate_video(parse)
    await screen._video._console_command_stream_video(parse)

    screen._video._console_command_generate_video.assert_awaited_once_with(parse)
    screen._video._console_command_stream_video.assert_awaited_once_with(parse)


@pytest.mark.unit
def test_message_video_actions_are_wired_directly_to_video_controller() -> None:
    """Message actions do not detour through private ChatScreen methods."""
    wiring = Path("tldw_chatbook/UI/Console_Modules/wiring.py").read_text(
        encoding="utf-8"
    )
    message = Path("tldw_chatbook/UI/Console_Modules/message.py").read_text(
        encoding="utf-8"
    )

    assert "screen._video._play_console_video" in wiring
    assert "screen._video._save_console_video_copy" in wiring
    assert "screen._video._regenerate_console_video_message" in wiring
    assert "regenerate_console_video_message:" in message


@pytest.mark.unit
def test_shared_message_stub_accepts_video_action_dependencies() -> None:
    """The shared bare-screen stub exposes every video-action seam."""
    screen = SimpleNamespace()
    play = AsyncMock()
    save = AsyncMock()
    regenerate = AsyncMock()

    controller = stub_message_controller(
        screen,
        app_instance=NO_APP,
        play_console_video=play,
        save_console_video_copy=save,
        regenerate_console_video_message=regenerate,
    )

    assert controller._play_console_video is play
    assert controller._save_console_video_copy is save
    assert controller._regenerate_console_video_message is regenerate
