"""Fresh-process first actions retain original modal identities and callbacks."""

import importlib
import sys

import pytest

from Tests.private_profile import private_profile_test

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

_ACTIONS = (
    ("style", "console_style_picker_modal", "ConsoleStylePickerModal"),
    ("system", "console_system_prompt_modal", "ConsoleSystemPromptModal"),
    ("rewind", "console_rewind_modal", "ConsoleRewindModal"),
    ("comparison", "console_prompt_comparison_modal", "ConsolePromptComparisonModal"),
    ("feedback", "console_feedback_comment_modal", "ConsoleFeedbackCommentModal"),
)


@pytest.mark.parametrize("action,module_name,class_name", _ACTIONS)
@private_profile_test
async def test_actual_first_action_loads_original_modal(
    request, action, module_name, class_name
):
    from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
    from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
    )
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_command_grammar import CommandParse
    from tldw_chatbook.UI.Console_Modules.wiring import _show_console_feedback_comment

    qualified_name = f"tldw_chatbook.Widgets.Console.{module_name}"
    assert qualified_name not in sys.modules
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console._console_composer_or_none()
        composer.load_draft("Original first-use draft")
        original_draft = composer.draft_text()
        assert qualified_name not in sys.modules
        worker = None
        if action == "style":
            console.action_open_console_style_insert()
        elif action == "system":
            await console._open_console_system_prompt_editor()
        elif action == "rewind":
            store = console._ensure_console_chat_store()
            session = store.ensure_session()
            store.append_message(
                session.id, role=ConsoleMessageRole.USER, content="Original prompt"
            )
            assert await console._console_command_rewind(
                CommandParse(kind="command", name="rewind", args="")
            )
        elif action == "comparison":
            snapshot = composer.capture_draft_snapshot()
            assert (
                composer.apply_improvement(snapshot, "Improved first-use draft")
                is snapshot
            )
            original_draft = composer.draft_text()
            console._open_console_prompt_comparison()
        else:
            worker = console.run_worker(
                _show_console_feedback_comment(console, "comment", "Original quote"),
                exclusive=False,
            )
        await pilot.pause()
        assert qualified_name in sys.modules
        original_class = getattr(importlib.import_module(qualified_name), class_name)
        assert type(host.screen) is original_class
        assert host.screen.is_mounted
        await pilot.press("escape")
        await pilot.pause()
        assert host.screen is console
        assert composer.draft_text() == original_draft
        if worker is not None:
            assert await worker.wait() is None


@private_profile_test
async def test_reaction_value_first_use_preserves_original_custom_inventory(
    request, monkeypatch
):
    from types import SimpleNamespace

    from tldw_chatbook.UI.Console_Modules import session

    name = "tldw_chatbook.Widgets.Console.console_reaction_picker_modal"
    assert name not in sys.modules
    database = object()
    calls = []
    graph = None

    class Repository:
        def __init__(self, db):
            assert db is database

        def get_active_actor_pack(self, kind, actor):
            calls.append((kind, actor))
            return graph

    monkeypatch.setattr(session, "VisualIdentityRepository", Repository)
    scope = ("session", "character", "17")
    assert session._visual_identity_options_for_db(database, scope) == ()
    assert name not in sys.modules
    graph = {
        "assets": [
            {
                "expression_key": "custom:relief",
                "display_label": "Relief",
                "content_type": "image/webp",
                "is_animated": False,
            }
        ]
    }
    options = session._visual_identity_options_for_db(database, scope)
    original = importlib.import_module(name)
    assert len(options) == 1 and type(options[0]) is original.ReactionOption
    assert options[0].expression_key == "custom:relief"
    assert calls == [("character", "17"), ("character", "17")]
    screen = SimpleNamespace(
        app=SimpleNamespace(push_screen=lambda modal: mounted.append(modal))
    )
    controller = object.__new__(session.ConsoleSessionController)
    controller._screen = screen
    controller.app_instance = SimpleNamespace(notify=lambda *args, **kwargs: None)
    controller._visual_identity_request_context = lambda: (scope, "idle", None)
    controller._visual_identity_db_accessor = lambda: database
    mounted = []
    await controller._open_console_reaction_picker()
    assert len(mounted) == 1 and type(mounted[0]) is original.ConsoleReactionPickerModal
    assert mounted[0]._options == options


@private_profile_test
async def test_video_capacity_first_use_retires_original_artifact(request):
    import asyncio
    import tempfile

    from textual.widgets import Button

    from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
    from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
    from tldw_chatbook.Chat.console_generate_video import PendingVideoArtifact
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata

    name = "tldw_chatbook.Widgets.Console.console_video_capacity_modal"
    assert name not in sys.modules
    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    host = app
    stream = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115 - artifact owns stream; fixture closes on failure
    stream.write(b"original staged video")
    stream.seek(0)
    artifact = PendingVideoArtifact(
        metadata=VideoGenerationMetadata(
            name="fixture", prompt="fixture", backend="comfyui"
        ),
        message_id="first-use-video",
        slug="fixture",
        extension="mp4",
        size_bytes=21,
        max_bytes=1,
        reason="over_capacity",
        stream=stream,
    )
    try:
        async with host.run_test(size=(180, 48)) as pilot:
            async with asyncio.timeout(5):
                while not isinstance(host.screen, ChatScreen):
                    await pilot.pause()
            console = host.screen
            await _wait_for_selector(console, pilot, "#console-native-composer")
            store = console._ensure_console_chat_store()
            assert name not in sys.modules
            worker = console.run_worker(
                console._video._resolve_generated_video_outcome(
                    artifact,
                    session_id=store.ensure_session().id,
                    message_id=artifact.message_id,
                ),
                exclusive=False,
            )
            async with asyncio.timeout(5):
                while host.screen is console or not host.screen.is_mounted:
                    await pilot.pause()
            original = importlib.import_module(name)
            assert type(host.screen) is original.ConsoleVideoCapacityModal
            assert console._video._pending_console_video_artifacts() == {
                artifact.message_id: artifact
            }
            assert not stream.closed
            host.screen.query_one("#video-capacity-discard", Button).press()
            await worker.wait()
            await pilot.pause()
            assert host.screen is console
            assert stream.closed
            assert console._video._pending_console_video_artifacts() == {}
            assert console._video._pending_video_operation_cancels == {}
    finally:
        stream.close()
