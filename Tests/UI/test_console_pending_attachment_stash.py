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

import asyncio
from io import BytesIO
from types import SimpleNamespace
import threading

import pytest
from PIL import Image as PILImage
from textual.widgets import Button

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from Tests.UI.test_console_native_chat_flow import (
    RestoredConsoleHarness,
    _select_llamacpp_console,
)
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    GenerationVariantMeta,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_command_grammar import CommandParse
from tldw_chatbook.Chat.console_generate_image import BatchResult
from tldw_chatbook.Chat.console_image_edit_operations import (
    ImageEditCompletion,
    ImageEditOperationRegistry,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Image_Generation.exceptions import (
    ComfyUIImageEditError,
    ImageGenerationCancelled,
)
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module
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


def _png_bytes(size: tuple[int, int] = (13, 9)) -> bytes:
    buffer = BytesIO()
    PILImage.new("RGB", size, (25, 50, 75)).save(buffer, format="PNG")
    return buffer.getvalue()


def _real_image_pending(name: str, *, attachment_id: str) -> PendingAttachment:
    data = _png_bytes()
    return PendingAttachment(
        file_path=f"/private/sentinel/{name}",
        display_name=name,
        file_type="image",
        insert_mode="attachment",
        data=data,
        mime_type="image/png",
        original_size=len(data),
        processed_size=len(data),
        attachment_id=attachment_id,
    )


def _h3_config() -> SimpleNamespace:
    return SimpleNamespace(
        default_backend="comfyui",
        default_batch=7,
        max_variants_per_message=7,
        comfyui_image_default_seed=-1,
        comfyui_image_default_steps=17,
        comfyui_image_default_sampler="res_multistep",
    )


def _patch_h3_enabled(monkeypatch) -> None:
    monkeypatch.setattr(chat_screen_module, "get_image_generation_config", _h3_config)
    monkeypatch.setattr(
        chat_screen_module,
        "list_image_models_for_catalog",
        lambda: [{"name": "comfyui", "is_configured": True}],
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


@pytest.mark.asyncio
async def test_h3_start_immediately_paints_enabled_stop_on_live_screen(
    monkeypatch,
):
    _patch_h3_enabled(monkeypatch)
    app = _build_test_app()
    host = ConsoleHarness(app)
    started = threading.Event()
    release = threading.Event()

    def _blocked_batch(**kwargs):
        started.set()
        assert release.wait(2)
        raise ImageGenerationCancelled()

    monkeypatch.setattr(chat_screen_module, "run_generation_batch", _blocked_batch)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.pause()
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.add_pending_attachment(
            session.id,
            _real_image_pending("source.png", attachment_id="live-source"),
        )
        composer = console.query_one("#console-native-composer")
        composer.load_draft("/generate-image :comfyui change it")
        caller = asyncio.create_task(
            console._console_command_generate_image(
                CommandParse(
                    kind="command", name="generate-image", args=":comfyui change it"
                )
            )
        )
        assert await asyncio.to_thread(started.wait, 2)
        try:
            await pilot.pause()
            await pilot.pause()
            stop = console.query_one("#console-stop-generation", Button)
            assert stop.styles.display != "none"
            assert not stop.disabled
        finally:
            release.set()
            await caller


@pytest.mark.asyncio
async def test_actual_unmount_is_nonblocking_and_fresh_screen_shows_stopping(
    monkeypatch,
):
    _patch_h3_enabled(monkeypatch)
    app = _build_test_app()
    host = ConsoleHarness(app)
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def _blocked_batch(**kwargs):
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(2)
        assert kwargs["cancel_event"].is_set()
        raise ImageGenerationCancelled()

    monkeypatch.setattr(chat_screen_module, "run_generation_batch", _blocked_batch)
    async with host.run_test(size=(160, 48)) as pilot:
        old = host.screen_stack[-1]
        await _wait_for_selector(old, pilot, "#console-native-composer")
        store = old._ensure_console_chat_store()
        session = store.ensure_session()
        source = _real_image_pending("source.png", attachment_id="remount-source")
        store.add_pending_attachment(session.id, source)
        old.query_one("#console-native-composer").load_draft(
            "/generate-image :comfyui change it"
        )
        caller = asyncio.create_task(
            old._console_command_generate_image(
                CommandParse(
                    kind="command", name="generate-image", args=":comfyui change it"
                )
            )
        )
        assert await asyncio.to_thread(started.wait, 2)
        saved_state = old.save_state()

        await asyncio.wait_for(host.pop_screen(), timeout=0.5)
        active = app.console_image_edit_operations.active(session.id)
        assert active is not None and active.cancel_event.is_set()

        fresh = ChatScreen(app)
        fresh.restore_state(saved_state)
        await host.push_screen(fresh)
        await _wait_for_selector(fresh, pilot, "#console-native-composer")
        await fresh._sync_native_console_chat_ui()
        assert fresh._native_run_status_copy() == "Stopping image edit…"
        stop = fresh.query_one("#console-stop-generation", Button)
        assert stop.styles.display != "none"
        assert not stop.disabled

        await fresh._console_command_generate_image(
            CommandParse(
                kind="command", name="generate-image", args=":comfyui change it"
            )
        )
        assert calls == 1
        assert any(
            message.role is ConsoleMessageRole.SYSTEM
            and message.content == "An image edit is already running for this session."
            for message in fresh._ensure_console_chat_store().messages_for_session(
                session.id
            )
        )

        release.set()
        await caller


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_kind", "expected_copy"),
    (
        (
            "network",
            "The source image could not be uploaded. Please try again.",
        ),
        (
            "batch",
            "The image-edit operation did not complete. Please try again.",
        ),
        (
            "persistence",
            "The edited image could not be saved locally. The source remains staged.",
        ),
    ),
)
async def test_batch_failure_after_actual_unmount_never_syncs_stale_screen(
    monkeypatch,
    failure_kind,
    expected_copy,
):
    _patch_h3_enabled(monkeypatch)
    app = _build_test_app()
    host = ConsoleHarness(app)
    started = threading.Event()
    release = threading.Event()

    def _failed_batch(**kwargs):
        started.set()
        assert release.wait(2)
        if failure_kind == "network":
            raise ComfyUIImageEditError("source_upload")
        if failure_kind == "persistence":
            return BatchResult(
                successes=[
                    (
                        _png_bytes(),
                        "image/png",
                        GenerationVariantMeta(
                            prompt=kwargs["prompt"],
                            negative_prompt="",
                            backend="comfyui",
                            model=None,
                            seed=9,
                            style=None,
                            params={"operation": "edit"},
                        ),
                    )
                ],
                errors=[],
            )
        raise RuntimeError("sentinel response body and private descriptor")

    monkeypatch.setattr(chat_screen_module, "run_generation_batch", _failed_batch)
    async with host.run_test(size=(160, 48)) as pilot:
        old = host.screen_stack[-1]
        await _wait_for_selector(old, pilot, "#console-native-composer")
        store = old._ensure_console_chat_store()
        session = store.ensure_session()
        source = _real_image_pending("source.png", attachment_id="failure-source")
        store.add_pending_attachment(session.id, source)
        store.set_session_draft(session.id, "captured failure draft")
        if failure_kind == "persistence":
            monkeypatch.setattr(
                store,
                "append_generation_message",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("sentinel persistence path")
                ),
            )
        old.query_one("#console-native-composer").load_draft(
            "captured failure draft"
        )
        caller = asyncio.create_task(
            old._console_command_generate_image(
                CommandParse(
                    kind="command", name="generate-image", args=":comfyui change it"
                )
            )
        )
        assert await asyncio.to_thread(started.wait, 2)
        await asyncio.wait_for(host.pop_screen(), timeout=0.5)

        async def _stale_sync_tripwire():
            raise AssertionError("unmounted screen UI sync")

        old._sync_native_console_chat_ui = _stale_sync_tripwire
        old._message._sync_native_console_chat_ui_fn = _stale_sync_tripwire
        release.set()
        await caller

        assert store.pending_attachments(session.id) == [source]
        assert store.session_draft(session.id) == "captured failure draft"
        system_copy = [
            message.content
            for message in store.messages_for_session(session.id)
            if message.role is ConsoleMessageRole.SYSTEM
        ]
        assert system_copy == [expected_copy]
        assert "sentinel" not in repr(system_copy)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_kind", "expected_copy"),
    (
        (
            "network",
            "The source image could not be uploaded. Please try again.",
        ),
        (
            "batch",
            "The image-edit operation did not complete. Please try again.",
        ),
    ),
)
async def test_unmounted_h3_failure_guidance_is_durable_once_across_real_reload(
    monkeypatch,
    tmp_path,
    failure_kind,
    expected_copy,
):
    _patch_h3_enabled(monkeypatch)
    db = CharactersRAGDB(tmp_path / f"h3-failure-{failure_kind}.sqlite", "test_client")
    try:
        app = _build_test_app()
        app.chachanotes_db = db
        host = ConsoleHarness(app)
        started = threading.Event()
        release = threading.Event()

        def _failed_batch(**_kwargs):
            started.set()
            assert release.wait(2)
            if failure_kind == "network":
                raise ComfyUIImageEditError("source_upload")
            raise RuntimeError("sentinel response body /private/source.png")

        monkeypatch.setattr(chat_screen_module, "run_generation_batch", _failed_batch)
        async with host.run_test(size=(160, 48)) as pilot:
            old = host.screen_stack[-1]
            await _wait_for_selector(old, pilot, "#console-native-composer")
            store = ConsoleChatStore(persistence=ChatPersistenceService(db))
            old._console_chat_store = store
            session = store.create_session(title="Durable H3 failure")
            store.append_message(
                session.id,
                role=ConsoleMessageRole.USER,
                content="existing durable turn",
                persist=True,
            )
            conversation_id = session.persisted_conversation_id
            assert conversation_id is not None
            source = _real_image_pending(
                "source.png", attachment_id="durable-failure-source"
            )
            store.add_pending_attachment(session.id, source)
            store.set_session_draft(session.id, "preserved failure draft")
            old.query_one("#console-native-composer").load_draft(
                "preserved failure draft"
            )
            caller = asyncio.create_task(
                old._console_command_generate_image(
                    CommandParse(
                        kind="command",
                        name="generate-image",
                        args=":comfyui change it",
                    )
                )
            )
            assert await asyncio.to_thread(started.wait, 2)
            await asyncio.wait_for(host.pop_screen(), timeout=0.5)

            async def _stale_sync_tripwire():
                raise AssertionError("unmounted screen UI sync")

            old._sync_native_console_chat_ui = _stale_sync_tripwire
            old._message._sync_native_console_chat_ui_fn = _stale_sync_tripwire
            release.set()
            await caller

            assert store.pending_attachments(session.id) == [source]
            assert store.session_draft(session.id) == "preserved failure draft"

            def _build_reloaded_screen():
                fresh = ChatScreen(app)
                tree = ChatConversationService(db).get_conversation_tree(
                    conversation_id, depth_cap=10_000, root_limit=10_000
                )
                all_nodes = fresh._console_messages_from_conversation_tree(tree)
                fresh_store = ConsoleChatStore(
                    persistence=ChatPersistenceService(db)
                )
                fresh._console_chat_store = fresh_store
                restored = fresh_store.restore_persisted_session(
                    title="Durable H3 failure",
                    workspace_id=None,
                    persisted_conversation_id=conversation_id,
                    all_nodes=all_nodes,
                    active_leaf_persisted_id=db.get_conversation_active_leaf(
                        conversation_id
                    ),
                )
                fresh._reconcile_h3_image_edit_completions(fresh_store)
                fresh._reconcile_h3_image_edit_completions(fresh_store)
                return fresh, fresh_store, restored.id

            def _guidance(fresh_store, restored_session_id) -> list[str]:
                return [
                    message.content
                    for message in fresh_store.messages_for_session(
                        restored_session_id
                    )
                    if message.role is ConsoleMessageRole.SYSTEM
                ]

            fresh, fresh_store, restored_id = _build_reloaded_screen()
            await host.push_screen(fresh)
            await _wait_for_selector(fresh, pilot, "#console-native-composer")
            assert _guidance(fresh_store, restored_id) == [expected_copy]
            await asyncio.wait_for(host.pop_screen(), timeout=0.5)

            remounted, remounted_store, remounted_id = _build_reloaded_screen()
            await host.push_screen(remounted)
            await _wait_for_selector(remounted, pilot, "#console-native-composer")
            assert _guidance(remounted_store, remounted_id) == [expected_copy]
            assert "sentinel" not in repr(
                _guidance(remounted_store, remounted_id)
            )
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    "success_timing",
    ("before_stash", "after_stash_before_adoption", "after_adoption"),
)
def test_two_fresh_screens_reconcile_real_persisted_h3_success_at_every_boundary(
    tmp_path,
    success_timing,
):
    db = CharactersRAGDB(tmp_path / f"h3-{success_timing}.sqlite", "test_client")
    try:
        app = _build_test_app()
        app.chachanotes_db = db
        registry = ImageEditOperationRegistry()
        app.console_image_edit_operations = registry
        app._console_pending_attachment_stash = {}
        persistence = ChatPersistenceService(db)
        old_store = ConsoleChatStore(persistence=persistence)
        session = old_store.create_session(title="H3 lifecycle")
        source = _real_image_pending("source.png", attachment_id="exact-source")
        other = _real_image_pending("other.png", attachment_id="keep-other")
        old_store.add_pending_attachment(session.id, source)
        old_store.add_pending_attachment(session.id, other)
        old_store.set_session_draft(session.id, "captured draft")
        old = ChatScreen(app)
        old._console_chat_store = old_store
        generation = f"generation-{success_timing}"
        completion: ImageEditCompletion | None = None

        def _commit_success() -> ImageEditCompletion:
            message = old_store.append_generation_message(
                session.id,
                content="[image] exact edit",
                variants=[
                    (
                        _png_bytes((17, 12)),
                        "image/png",
                        GenerationVariantMeta(
                            prompt="exact edit",
                            negative_prompt="",
                            backend="comfyui",
                            model=None,
                            seed=41,
                            style=None,
                            params={"operation": "edit", "format": "png"},
                        ),
                    )
                ],
                persist=True,
            )
            record = ImageEditCompletion(
                session_id=session.id,
                generation=generation,
                message_id=message.persisted_message_id or "",
                attachment_id=source.attachment_id,
                captured_draft="captured draft",
            )
            assert registry.publish_completion(record)
            old._filter_h3_attachment_from_app_stash(
                session.id, source.attachment_id
            )
            return record

        if success_timing == "before_stash":
            completion = _commit_success()
            assert old._cleanup_h3_completion_in_store(
                old_store, completion, clear_visible_composer=False
            )
            assert registry.ack_completion(session.id, generation)

        payload = old._serialize_native_console_state()
        assert payload is not None

        if success_timing == "after_stash_before_adoption":
            completion = _commit_success()
            assert old._cleanup_h3_completion_in_store(
                old_store, completion, clear_visible_composer=False
            )

        fresh_store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        fresh = ChatScreen(app)
        fresh._console_chat_store = fresh_store
        fresh._restore_native_console_state(payload)

        restored_session = next(
            item for item in fresh_store.sessions() if item.id == session.id
        )
        assert (
            restored_session.persisted_conversation_id
            == session.persisted_conversation_id
        )

        if success_timing == "after_adoption":
            fresh_store.set_session_draft(session.id, "replacement draft")
            completion = _commit_success()
            fresh._reconcile_h3_image_edit_completions(fresh_store)

        assert registry.completion(session.id) is None

        messages = fresh_store.messages_for_session(session.id)
        generated = [
            message
            for message in messages
            if message.persisted_message_id
            == (completion.message_id if completion is not None else None)
        ]
        assert len(generated) == 1, [
            (message.id, message.persisted_message_id, message.content)
            for message in messages
        ]
        assert generated[0].image_mime_type == "image/png"
        assert generated[0].image_data == _png_bytes((17, 12))
        assert len(generated[0].generation_metadata) == 1
        assert generated[0].generation_metadata[0].params == {
            "operation": "edit",
            "format": "png",
        }
        assert fresh_store.pending_attachments(session.id) == [other]
        expected_draft = (
            "replacement draft" if success_timing == "after_adoption" else ""
        )
        assert fresh_store.session_draft(session.id) == expected_draft
        assert app._console_pending_attachment_stash == {}
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_confirmed_session_delete_drops_active_and_completion():
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        keeper = store.ensure_session(title="Keep")
        doomed = store.create_session(title="Delete")
        store.append_message(
            doomed.id,
            role=ConsoleMessageRole.USER,
            content="confirmation required",
        )
        store.switch_session(keeper.id)
        release = asyncio.Event()

        async def _runner(_generation: str) -> None:
            await release.wait()

        registry = console._h3_image_edit_registry()
        operation = registry.start(
            session_id=doomed.id,
            attachment_id="doomed-source",
            captured_draft="doomed draft",
            cancel_event=threading.Event(),
            runner=_runner,
        )
        assert operation is not None
        assert registry.publish_completion(
            ImageEditCompletion(
                session_id=doomed.id,
                generation=operation.generation,
                message_id="doomed-message",
                attachment_id="doomed-source",
                captured_draft="doomed draft",
            )
        )
        await console._sync_native_console_chat_ui()
        close_selector = f"#console-close-session-tab-{doomed.id}"
        await _wait_for_selector(console, pilot, close_selector)
        await pilot.click(close_selector)
        await pilot.pause()
        await pilot.click("#confirm-button")
        await pilot.pause()
        await pilot.pause()

        assert doomed.id not in {item.id for item in store.sessions()}
        assert operation.cancel_event.is_set()
        assert registry.active(doomed.id) is None
        assert registry.completion(doomed.id) is None
        release.set()
        await operation.task
