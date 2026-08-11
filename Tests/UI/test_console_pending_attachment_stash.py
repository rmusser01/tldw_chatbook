"""Pending Console attachments survive screen navigation (TASK-218).

Phase 1 (#621) deliberately dropped staged-but-unsent attachments on
navigation: screen-state serialization is metadata-only by spec (raw bytes
never serialize), and the staging list lived only on the screen-owned store.
The preservation strategy (user-approved): a bounded app-level in-memory
stash — full ``PendingAttachment`` objects (bytes included, clipboard grabs
too) snapshot onto the app object at save time and re-adopt into the store at
restore time. The stash is process-memory only: it never serializes, and it
dies with the app (restart drops pendings, which is the accepted trade).
"""

import pytest

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from Tests.UI.test_console_native_chat_flow import (
    RestoredConsoleHarness,
    _select_llamacpp_console,
)
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_image_edit_operations import (
    ImageEditCompletion,
    ImageEditOperationRegistry,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


def _image_pending(name: str, *, path: str) -> PendingAttachment:
    data = f"png-bytes-{name}".encode()
    return PendingAttachment(
        file_path=path,
        display_name=name,
        file_type="image",
        insert_mode="attachment",
        data=data,
        mime_type="image/png",
        text_content=None,
        original_size=len(data),
        processed_size=len(data),
    )


def _assert_no_bytes(node, crumb="state"):
    """AC #2: raw attachment bytes never enter screen-state serialization."""
    if isinstance(node, (bytes, bytearray)):
        raise AssertionError(f"raw bytes leaked into screen state at {crumb}")
    if isinstance(node, dict):
        for key, value in node.items():
            _assert_no_bytes(value, f"{crumb}.{key}")
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _assert_no_bytes(value, f"{crumb}[{index}]")


@pytest.mark.asyncio
async def test_pending_attachments_survive_screen_recreation():
    """Stage a path-backed pending AND a clipboard-style one (no file path),
    recreate the screen from saved state, and find both staged again with
    the composer indicator rebuilt."""
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"

    saved_state: dict | None = None
    session_id: str | None = None
    disk_pending = _image_pending("photo.png", path="/tmp/photo.png")
    clipboard_pending = _image_pending("clipboard-grab.png", path="")

    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        store = console._ensure_console_chat_store()
        session_id = store.ensure_session().id
        assert store.add_pending_attachment(session_id, disk_pending)
        assert store.add_pending_attachment(session_id, clipboard_pending)
        await console._sync_native_console_chat_ui()
        saved_state = console.save_state()

    assert saved_state is not None and session_id is not None
    _assert_no_bytes(saved_state)

    restored_host = RestoredConsoleHarness(app, saved_state)
    async with restored_host.run_test(size=(160, 48)) as pilot:
        console = restored_host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        pendings = store.pending_attachments(session_id)
        assert [p.display_name for p in pendings] == [
            "photo.png",
            "clipboard-grab.png",
        ]
        # Bytes preserved verbatim — including the path-less clipboard grab.
        assert pendings[0].data == disk_pending.data
        assert pendings[1].data == clipboard_pending.data
        await console._sync_native_console_chat_ui()
        composer = console.query_one("#console-native-composer")
        assert "2 files" in (composer._pending_attachment_label or "")


@pytest.mark.asyncio
async def test_stash_prunes_dead_sessions_and_empties_after_adoption():
    """A stash entry for a session that no longer exists must not crash the
    restore or linger in the stash."""
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"

    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        store = console._ensure_console_chat_store()
        session_id = store.ensure_session().id
        assert store.add_pending_attachment(
            session_id, _image_pending("keep.png", path="")
        )
        saved_state = console.save_state()

    stash = getattr(app, "_console_pending_attachment_stash", None)
    assert isinstance(stash, dict) and session_id in stash
    stash["dead-session-id"] = stash[session_id]  # simulate a stale entry

    restored_host = RestoredConsoleHarness(app, saved_state)
    async with restored_host.run_test(size=(160, 48)) as pilot:
        console = restored_host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        assert [p.display_name for p in store.pending_attachments(session_id)] == [
            "keep.png"
        ]
        assert "dead-session-id" not in store._sessions

    assert getattr(app, "_console_pending_attachment_stash", None) == {}


def test_adopt_resets_malformed_stash_without_crashing():
    """A corrupted stash value must be replaced with an empty dict on the
    first adopt attempt, releasing whatever it referenced (self-healing)."""
    from types import SimpleNamespace as _NS

    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen = ChatScreen.__new__(ChatScreen)
    screen.app_instance = _NS(_console_pending_attachment_stash="not-a-dict")
    store = ConsoleChatStore()
    screen._adopt_console_pending_attachments(store)
    assert screen.app_instance._console_pending_attachment_stash == {}

    # Malformed VALUES inside a well-formed dict are skipped, dict still reset.
    session_id = store.ensure_session().id
    screen.app_instance._console_pending_attachment_stash = {session_id: "junk"}
    screen._adopt_console_pending_attachments(store)
    assert screen.app_instance._console_pending_attachment_stash == {}
    assert store.pending_attachments(session_id) == []


def test_h3_completion_reconciliation_filters_exact_attachment_and_is_idempotent():
    store = ConsoleChatStore()
    session = store.ensure_session()
    source = _image_pending("source.png", path="/private/sentinel/source.png")
    other = _image_pending("other.png", path="/private/sentinel/other.png")
    store.add_pending_attachment(session.id, source)
    store.add_pending_attachment(session.id, other)
    store.set_session_draft(session.id, "captured edit draft")

    registry = ImageEditOperationRegistry()
    completion = ImageEditCompletion(
        session_id=session.id,
        generation="generation-1",
        message_id="persisted-message-1",
        attachment_id=source.attachment_id,
        captured_draft="captured edit draft",
    )
    registry.publish_completion(completion)
    app = type("App", (), {})()
    app.console_image_edit_operations = registry
    app._console_pending_attachment_stash = {
        session.id: (source, other),
        "unrelated": (_image_pending("unrelated.png", path=""),),
    }
    screen = ChatScreen.__new__(ChatScreen)
    screen.app_instance = app

    hydrated: list[tuple[str, str]] = []

    def _merge(session_id: str, message_id: str):
        hydrated.append((session_id, message_id))
        return type("Message", (), {"persisted_message_id": message_id})()

    store.merge_persisted_generation_message = _merge
    screen._reconcile_h3_image_edit_completions(store)
    screen._reconcile_h3_image_edit_completions(store)

    assert hydrated == [(session.id, "persisted-message-1")]
    assert store.pending_attachments(session.id) == [other]
    assert store.session_draft(session.id) == ""
    assert app._console_pending_attachment_stash[session.id] == (other,)
    assert "unrelated" in app._console_pending_attachment_stash
    assert registry.completion(session.id) is None


def test_h3_completion_preserves_replacement_draft_after_message_presence():
    store = ConsoleChatStore()
    session = store.ensure_session()
    source = _image_pending("source.png", path="")
    store.add_pending_attachment(session.id, source)
    store.set_session_draft(session.id, "replacement draft")
    registry = ImageEditOperationRegistry()
    registry.publish_completion(
        ImageEditCompletion(
            session_id=session.id,
            generation="generation-2",
            message_id="persisted-message-2",
            attachment_id=source.attachment_id,
            captured_draft="captured edit draft",
        )
    )
    screen = ChatScreen.__new__(ChatScreen)
    screen.app_instance = type(
        "App",
        (),
        {
            "console_image_edit_operations": registry,
            "_console_pending_attachment_stash": {},
        },
    )()
    store.merge_persisted_generation_message = lambda *_args: type(
        "Message", (), {"persisted_message_id": "persisted-message-2"}
    )()

    screen._reconcile_h3_image_edit_completions(store)

    assert registry.completion(session.id) is None
    assert store.pending_attachments(session.id) == []
    assert store.session_draft(session.id) == "replacement draft"


def test_h3_completion_waits_for_durable_message_presence_before_cleanup():
    store = ConsoleChatStore()
    session = store.ensure_session()
    source = _image_pending("source.png", path="")
    store.add_pending_attachment(session.id, source)
    store.set_session_draft(session.id, "captured edit draft")
    registry = ImageEditOperationRegistry()
    registry.publish_completion(
        ImageEditCompletion(
            session_id=session.id,
            generation="generation-3",
            message_id="persisted-message-3",
            attachment_id=source.attachment_id,
            captured_draft="captured edit draft",
        )
    )
    screen = ChatScreen.__new__(ChatScreen)
    screen.app_instance = type(
        "App",
        (),
        {
            "console_image_edit_operations": registry,
            "_console_pending_attachment_stash": {},
        },
    )()
    store.merge_persisted_generation_message = lambda *_args: None

    screen._reconcile_h3_image_edit_completions(store)

    assert registry.completion(session.id) is not None
    assert store.pending_attachments(session.id) == [source]
    assert store.session_draft(session.id) == "captured edit draft"
