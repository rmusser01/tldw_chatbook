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
from pathlib import Path
from types import SimpleNamespace

import pytest

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
        store = ConsoleChatStore()
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
        store = ConsoleChatStore()
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
        store = ConsoleChatStore()
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
        store = ConsoleChatStore()
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
        store = ConsoleChatStore()
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
        store = ConsoleChatStore()
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
        from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module
        from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

        markup_dir = tmp_path / "sav[e]dir"
        markup_dir.mkdir()
        monkeypatch.setattr(
            chat_screen_module,
            "get_cli_setting",
            lambda section, key=None, default=None: str(markup_dir),
        )
        store = ConsoleChatStore()
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
        screen._console_chat_store = store
        screen._ensure_console_chat_store = lambda: store
        screen.app_instance = SimpleNamespace(
            notify=lambda text, **kwargs: notices.append(str(text)),
            chachanotes_db=None,
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
