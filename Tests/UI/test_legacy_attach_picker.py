"""Legacy attach handler's picker must construct (TASK-219).

Pre-existing dead code from before PR #621: the handler called
``FileOpen(..., context="chat_images")``, but the ``FileOpen`` re-exported
from Third_Party.textual_fspicker accepts no ``context`` kwarg — the call
raised ``TypeError`` the moment the picker branch was exercised (reachable
from Chat_Window_Enhanced's attach button whenever the legacy
``#image-file-path-input`` isn't present). Every other picker surface uses
``EnhancedFileOpen``, which supports ``context`` (recents/bookmarks).
"""

from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Chat_Modules.chat_attachment_handler import ChatAttachmentHandler
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen


@pytest.fixture(autouse=True)
def _isolate_picker_config(monkeypatch):
    """Keep picker recents/bookmarks I/O away from the real config file.

    Constructing EnhancedFileOpen initializes RecentLocations and
    BookmarksManager, which read and WRITE ``[filepicker]`` settings through
    the live config module — a test must never touch the developer's real
    config.toml.
    """
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.enhanced_file_picker.get_cli_setting",
        lambda section, key, default=None: default,
    )
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.enhanced_file_picker.save_setting_to_cli_config",
        lambda section, key, value: None,
    )


def _handler_with_recording_app():
    """Build the handler over stubs that force the picker branch."""
    pushed: list = []

    def _query_one(_selector):
        raise LookupError("no legacy file input in this layout")

    app_instance = SimpleNamespace(
        push_screen=lambda screen, callback=None: pushed.append(screen),
        call_later=lambda fn: None,
    )
    chat_window = SimpleNamespace(
        app_instance=app_instance,
        _file_path_input=None,
        is_attached=False,
        query_one=_query_one,
    )
    return ChatAttachmentHandler(chat_window), pushed


@pytest.mark.asyncio
async def test_attach_button_constructs_and_pushes_the_picker():
    """Exercise the picker branch end to end without raising.

    A TypeError here is the TASK-219 regression; the pushed screen must be
    the context-capable EnhancedFileOpen.
    """
    handler, pushed = _handler_with_recording_app()

    await handler.handle_attach_image_button(event=None)

    assert len(pushed) == 1
    assert isinstance(pushed[0], EnhancedFileOpen)


def test_picker_kwargs_match_the_console_surface():
    """Pin the handler's picker kwargs against signature drift.

    The original bug's mechanism was exactly this: a kwarg the constructed
    class did not accept.
    """
    picker = EnhancedFileOpen(
        location=".",
        title="Select File to Attach",
        filters=None,
        context="chat_images",
    )
    assert picker is not None
