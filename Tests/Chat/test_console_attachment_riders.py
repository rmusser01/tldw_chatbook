"""Console attachment riders (TASK-223/224/225).

- 223: submit_draft's vision gate must make ONE capability decision — the
  controller's documented ``is_vision_capable`` seam is injected into
  ``vision_block_reason`` instead of being re-checked around it. Previously
  the two seams could disagree under test: the controller's pre-check said
  "not capable" while ``vision_block_reason``'s internal check said
  "capable", returning None and letting the send THROUGH the gate.
- 224: an image-only user turn whose images all fall outside the budget
  (over-cap, or a non-vision model) must appear in provider payloads as a
  text placeholder instead of silently vanishing.
- 225: the Save Image toasts escape the interpolated filesystem path.
"""

import asyncio
from types import SimpleNamespace


from Tests.Chat.test_console_chat_controller import (
    RecordingStreamingGateway,
    _pending_image,
)
from tldw_chatbook.Chat import attachment_core
from tldw_chatbook.Chat import console_chat_controller as controller_module
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    MessageAttachment,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from Tests.console_provider_doubles import persisted_console_store


def _controller(store, gateway, model="test-model"):
    return ConsoleChatController(store=store, provider_gateway=gateway, model=model)


class TestVisionGateSingleSeam:
    """TASK-223 — the controller seam alone decides the gate."""

    def test_diverging_seams_still_block(self, monkeypatch):
        """The exact hazard the dedup removes: controller seam says
        non-vision, attachment_core's internal seam says vision. The old
        pre-check-then-recheck let the send THROUGH; the gate must block."""
        monkeypatch.setattr(controller_module, "is_vision_capable", lambda p, m: False)
        monkeypatch.setattr(attachment_core, "is_vision_capable", lambda p, m: True)
        store = persisted_console_store()
        controller = _controller(store, RecordingStreamingGateway())
        session = store.ensure_session()
        store.set_pending_attachment(session.id, _pending_image())

        result = asyncio.run(controller.submit_draft("look"))

        assert not result.accepted
        assert "can't accept images" in result.visible_copy

    def test_controller_seam_capable_sends(self, monkeypatch):
        """Inverse divergence: controller seam capable must send even when
        attachment_core's internal seam claims otherwise."""
        monkeypatch.setattr(controller_module, "is_vision_capable", lambda p, m: True)
        monkeypatch.setattr(attachment_core, "is_vision_capable", lambda p, m: False)
        store = persisted_console_store()
        gateway = RecordingStreamingGateway()
        controller = _controller(store, gateway)
        session = store.ensure_session()
        store.set_pending_attachment(session.id, _pending_image())

        result = asyncio.run(controller.submit_draft("look"))

        assert result.accepted
        assert gateway.messages_seen is not None


class TestOmittedImagePlaceholder:
    """TASK-224 — image-only turns outside the budget must not vanish."""

    @staticmethod
    def _image_only_message(store, session_id, count=1):
        attachments = tuple(
            MessageAttachment(
                data=f"img-{index}".encode(),
                mime_type="image/png",
                display_name=f"img{index}.png",
                position=index,
            )
            for index in range(count)
        )
        return store.append_message(
            session_id,
            role=ConsoleMessageRole.USER,
            content="",
            attachments=attachments,
        )

    def test_non_vision_image_only_turn_becomes_placeholder(self, monkeypatch):
        monkeypatch.setattr(controller_module, "is_vision_capable", lambda p, m: False)
        store = persisted_console_store()
        gateway = RecordingStreamingGateway()
        controller = _controller(store, gateway)
        session = store.ensure_session()
        self._image_only_message(store, session.id)

        result = asyncio.run(controller.submit_draft("and now?"))

        assert result.accepted
        contents = [m["content"] for m in gateway.messages_seen]
        assert "[image omitted]" in contents
        assert contents[-1] == "and now?"

    def test_over_cap_image_only_turn_becomes_placeholder(self, monkeypatch):
        monkeypatch.setattr(controller_module, "is_vision_capable", lambda p, m: True)
        monkeypatch.setattr(controller_module, "max_history_images", lambda p, m: 1)
        store = persisted_console_store()
        gateway = RecordingStreamingGateway()
        controller = _controller(store, gateway)
        session = store.ensure_session()
        self._image_only_message(store, session.id)  # older: loses the budget
        self._image_only_message(store, session.id)  # newer: takes the budget

        result = asyncio.run(controller.submit_draft("compare them"))

        assert result.accepted
        contents = [m["content"] for m in gateway.messages_seen]
        assert "[image omitted]" in contents  # the older turn survives as text
        image_part_payloads = [c for c in contents if isinstance(c, list)]
        assert len(image_part_payloads) == 1  # the newer turn kept its image

    def test_multiple_omitted_images_pluralize(self, monkeypatch):
        monkeypatch.setattr(controller_module, "is_vision_capable", lambda p, m: False)
        store = persisted_console_store()
        gateway = RecordingStreamingGateway()
        controller = _controller(store, gateway)
        session = store.ensure_session()
        self._image_only_message(store, session.id, count=3)

        result = asyncio.run(controller.submit_draft("next"))

        assert result.accepted
        contents = [m["content"] for m in gateway.messages_seen]
        assert "[3 images omitted]" in contents

    def test_captioned_image_message_keeps_its_text(self, monkeypatch):
        monkeypatch.setattr(controller_module, "is_vision_capable", lambda p, m: False)
        store = persisted_console_store()
        gateway = RecordingStreamingGateway()
        controller = _controller(store, gateway)
        session = store.ensure_session()
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="the caption",
            image_data=b"img",
            image_mime_type="image/png",
        )

        result = asyncio.run(controller.submit_draft("next"))

        assert result.accepted
        contents = [m["content"] for m in gateway.messages_seen]
        assert "the caption" in contents
        assert not any("omitted" in str(c) for c in contents)


class TestSaveImageToastEscaping:
    """TASK-225 — the save-path toasts render markup-like paths literally."""

    @staticmethod
    def _screen_with_message(tmp_path, monkeypatch, attachment_count=1):
        from tldw_chatbook.UI.Console_Modules import message as message_module
        from tldw_chatbook.UI.Console_Modules.message import ConsoleMessageController
        from Tests.UI.console_controller_stubs import (
            NO_APP,
            stub_fleet_controller,
            stub_library_activity_controller,
        )
        from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

        markup_dir = tmp_path / "sav[e]dir"
        markup_dir.mkdir()
        # `_save_console_message_image` moved to `ConsoleMessageController`
        # (wave-3 console decomposition, task 1) -- it reads `get_cli_
        # setting` from ITS OWN module namespace now, not `chat_screen`'s.
        monkeypatch.setattr(
            message_module,
            "get_cli_setting",
            lambda section, key=None, default=None: str(markup_dir),
        )
        store = persisted_console_store()
        session = store.ensure_session()
        attachments = tuple(
            MessageAttachment(
                data=f"img-{index}".encode(),
                mime_type="image/png",
                display_name=f"img{index}.png",
                position=index,
            )
            for index in range(attachment_count)
        )
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="pic",
            attachments=attachments,
        )
        notices: list[str] = []
        screen = ChatScreen.__new__(ChatScreen)
        # The `_console_chat_store` setter reaches
        # `ConsoleRuntime.attach_view` -> `ChatScreen.console_view_hooks`,
        # which reads `self._fleet._console_wake_user_priority` (TASK-21381)
        # and `self._library_activity.build_provider` (TASK-23144) unguarded.
        stub_fleet_controller(screen, context="attachment riders bare screen")
        stub_library_activity_controller(
            screen,
            context="attachment riders bare screen",
            # `app_instance` is assigned just below; no scenario here
            # reaches a library-activity seam.
            app_instance=NO_APP,
        )
        screen._console_chat_store = store
        screen._ensure_console_chat_store = lambda: store
        screen.app_instance = SimpleNamespace(
            notify=lambda text, **kwargs: notices.append(str(text)),
            chachanotes_db=None,
        )

        def _unreached(*_args, **_kwargs):
            raise AssertionError(
                "_screen_with_message: this constructor callable is not "
                "wired for real -- only _save_console_message_image is "
                "exercised here."
            )

        screen._message = ConsoleMessageController(
            screen,
            app_instance=screen.app_instance,
            chat_store_accessor=lambda: store,
            current_chat_store_accessor=lambda: store,
            ensure_console_chat_controller=_unreached,
            current_chat_controller_accessor=lambda: None,
            sync_native_console_chat_ui=_unreached,
            active_session_is_ephemeral=_unreached,
            active_native_console_session=_unreached,
            current_console_conversation_id=_unreached,
            active_console_provider_model_display=_unreached,
            console_initial_session_title_for_workspace=_unreached,
            console_change_review_run_id=_unreached,
            open_change_review=_unreached,
            start_console_transcript_sync_timer=_unreached,
            clear_native_console_message_selection=_unreached,
            regenerate_console_generation_variant=_unreached,
            select_console_generation_variant=_unreached,
            keep_console_generation_variant=_unreached,
            handle_console_toggle_image_view=_unreached,
            invalidate_console_persisted_rows_cache=_unreached,
        )
        return screen, message, notices

    def test_single_save_toast_escapes_path(self, tmp_path, monkeypatch):
        screen, message, notices = self._screen_with_message(tmp_path, monkeypatch)
        asyncio.run(screen._save_console_message_image(message.id))
        assert notices, "no toast fired"
        # Rich's escape only needs to neutralize the opening bracket.
        assert "sav\\[e]dir" in notices[-1]
        assert "sav[e]dir" not in notices[-1]

    def test_multi_save_toast_escapes_path(self, tmp_path, monkeypatch):
        screen, message, notices = self._screen_with_message(
            tmp_path, monkeypatch, attachment_count=2
        )
        asyncio.run(screen._save_console_message_image(message.id))
        assert notices, "no toast fired"
        assert notices[-1].startswith("Saved 2 images to ")
        assert "sav\\[e]dir" in notices[-1]
