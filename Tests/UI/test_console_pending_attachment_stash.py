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
import time

import pytest
from PIL import Image as PILImage
from textual.widgets import Button

from Tests.console_resource_fixtures import (
    close_owned_console_resources as close_owned_console_resources,
    close_owned_console_test_apps as close_owned_console_test_apps,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.console_controller_stubs import stub_image_controller
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from Tests.UI.test_console_native_chat_flow import (
    RestoredConsoleHarness,
    _message_row_plain_text,
    _select_llamacpp_console,
)
from tldw_chatbook.Chat.attachment_core import PendingAttachment
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
    ImageEditFailureNotice,
    ImageEditOperationRegistry,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Image_Generation.exceptions import (
    ComfyUIImageEditError,
    ImageGenerationCancelled,
)
from tldw_chatbook.UI.Console_Modules import image as image_module
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


def _attach_image_controller(screen: ChatScreen) -> None:
    stub_image_controller(
        screen,
        context="test_console_pending_attachment_stash",
    )


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
    monkeypatch.setattr(image_module, "get_image_generation_config", _h3_config)
    monkeypatch.setattr(
        image_module,
        "list_image_models_for_catalog",
        lambda: [{"name": "comfyui", "is_configured": True}],
    )


async def _wait_for_h3_state(pilot, predicate, *, detail: str) -> None:
    """Bounded event-loop polling for asynchronous Textual settlement."""
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.1)
    raise AssertionError(f"Timed out waiting for {detail}")


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
    _attach_image_controller(screen)
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
    _attach_image_controller(screen)

    hydrated: list[tuple[str, str]] = []

    def _merge(session_id: str, message_id: str):
        hydrated.append((session_id, message_id))
        return type("Message", (), {"persisted_message_id": message_id})()

    store.merge_persisted_generation_message = _merge
    screen._image._reconcile_h3_image_edit_completions(store)
    screen._image._reconcile_h3_image_edit_completions(store)

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
    _attach_image_controller(screen)
    store.merge_persisted_generation_message = lambda *_args: type(
        "Message", (), {"persisted_message_id": "persisted-message-2"}
    )()

    screen._image._reconcile_h3_image_edit_completions(store)

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
    _attach_image_controller(screen)
    store.merge_persisted_generation_message = lambda *_args: None

    screen._image._reconcile_h3_image_edit_completions(store)

    assert registry.completion(session.id) is not None
    assert store.pending_attachments(session.id) == [source]
    assert store.session_draft(session.id) == "captured edit draft"


@pytest.mark.asyncio
async def test_h3_completion_for_session_a_never_clears_identical_visible_draft_b(
    monkeypatch, tmp_path
):
    db = CharactersRAGDB(tmp_path / "h3-two-session-draft.sqlite", "test_client")
    try:
        app = _build_test_app()
        app.chachanotes_db = db
        host = ConsoleHarness(app)
        async with host.run_test(size=(160, 48)) as pilot:
            screen = host.screen_stack[-1]
            await _wait_for_selector(screen, pilot, "#console-native-composer")
            store = ConsoleChatStore(persistence=ChatPersistenceService(db))
            screen._console_chat_store = store
            source = _real_image_pending("source.png", attachment_id="source-a")
            session_a = store.create_session(title="Session A")
            store.add_pending_attachment(session_a.id, source)
            store.set_session_draft(session_a.id, "identical draft")
            persisted = store.append_generation_message(
                session_a.id,
                content="[image] edited A",
                variants=[
                    (
                        _png_bytes((15, 9)),
                        "image/png",
                        GenerationVariantMeta(
                            prompt="edited A",
                            negative_prompt="",
                            backend="comfyui",
                            model=None,
                            seed=12,
                            style=None,
                            params={"operation": "edit"},
                        ),
                    )
                ],
                persist=True,
            )
            session_b = store.create_session(title="Session B")
            store.set_session_draft(session_b.id, "identical draft")
            screen._console_visible_draft_session_id = session_b.id
            composer = screen.query_one("#console-native-composer")
            composer.load_draft("identical draft")
            draft_writes: list[tuple[str, str]] = []
            real_set_draft = store.set_session_draft

            def _record_set_draft(session_id: str, draft: str) -> None:
                draft_writes.append((session_id, draft))
                real_set_draft(session_id, draft)

            monkeypatch.setattr(store, "set_session_draft", _record_set_draft)
            generation = "two-session-generation"
            assert app.console_image_edit_operations.publish_completion(
                ImageEditCompletion(
                    session_id=session_a.id,
                    generation=generation,
                    message_id=persisted.persisted_message_id or "",
                    attachment_id=source.attachment_id,
                    captured_draft="identical draft",
                )
            )

            screen._image._reconcile_h3_image_edit_completions(store)

            assert store.pending_attachments(session_a.id) == []
            assert store.session_draft(session_a.id) == ""
            assert store.session_draft(session_b.id) == "identical draft"
            assert composer.draft_text() == "identical draft"
            assert draft_writes == [(session_a.id, "")]
            assert app.console_image_edit_operations.completion(session_a.id) is None
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_h3_start_immediately_paints_enabled_stop_on_live_screen(
    monkeypatch,
):
    _patch_h3_enabled(monkeypatch)
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "test"}
    app.app_config["api_settings"] = {"openai": {"api_key": "test-key"}}
    host = ConsoleHarness(app)
    started = threading.Event()
    release = threading.Event()
    batch_calls = 0

    from PIL.PngImagePlugin import PngImageFile

    from tldw_chatbook.Image_Generation.request_validation import (
        validate_image_generation_request,
    )

    real_load = PngImageFile.load

    def _blocked_canonical_load(image, *args, **kwargs):
        started.set()
        assert release.wait(2)
        return real_load(image, *args, **kwargs)

    monkeypatch.setattr(PngImageFile, "load", _blocked_canonical_load)

    def _canonical_then_dispatch(**kwargs):
        nonlocal batch_calls
        issues = validate_image_generation_request(
            {
                "backend": kwargs["backend"],
                "prompt": kwargs["prompt"],
                "width": kwargs["width"],
                "height": kwargs["height"],
                "steps": kwargs["steps"],
                "cfg_scale": kwargs["cfg_scale"],
                "reference_image": kwargs["reference_image"],
            },
            config=_h3_config(),
        )
        assert issues == []
        if kwargs["cancel_event"].is_set():
            raise ImageGenerationCancelled()
        batch_calls += 1
        raise AssertionError("cancelled canonical decode reached dispatch")

    monkeypatch.setattr(image_module, "run_generation_batch", _canonical_then_dispatch)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        await _wait_for_h3_state(
            pilot,
            lambda: not console._console_setup_modal_blocking(),
            detail="dismissed Console setup modal",
        )
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.add_pending_attachment(
            session.id,
            _real_image_pending("source.png", attachment_id="live-source"),
        )
        composer = console.query_one("#console-native-composer")
        composer.load_draft("/generate-image :comfyui change it")
        console._request_console_control_bar_sync()
        await _wait_for_h3_state(
            pilot,
            lambda: (
                console.query_one("#console-send-message", Button).display
                and not console.query_one("#console-send-message", Button).disabled
            ),
            detail="enabled H3 Send button",
        )
        # Synthetic Send must begin from the ordinary typing focus so an
        # earlier navigation control cannot scroll the composer out of view.
        composer.focus()
        await pilot.pause(0.1)
        send = console.query_one("#console-send-message", Button)
        send.post_message(Button.Pressed(send))
        await asyncio.wait_for(pilot.pause(), timeout=0.5)
        assert await asyncio.to_thread(started.wait, 2)
        try:
            operation = app.console_image_edit_operations.active(session.id)
            assert operation is not None
            stop = console.query_one("#console-stop-generation", Button)
            assert stop.styles.display != "none"
            assert not stop.disabled
            actions = composer.query_one("#console-composer-actions")
            assert actions.content_region.contains_region(stop.region), {
                "actions": actions.content_region,
                "stop": stop.region,
            }
            assert console.region.contains_region(stop.region)
            assert (
                console.get_widget_at(
                    stop.region.x + stop.region.width // 2, stop.region.y
                )[0]
                is stop
            )
            assert await asyncio.wait_for(
                pilot.click("#console-stop-generation"), timeout=0.5
            )
            assert operation.cancel_event.is_set()
        finally:
            release.set()
        await operation.task
        assert batch_calls == 0


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
        assert release.wait(10)
        assert kwargs["cancel_event"].is_set()
        raise ImageGenerationCancelled()

    monkeypatch.setattr(image_module, "run_generation_batch", _blocked_batch)
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
            old._image._console_command_generate_image(
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

        await fresh._image._console_command_generate_image(
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
        await active.task


@pytest.mark.asyncio
async def test_real_app_shutdown_cancels_and_drains_owned_h3_operation():
    app = _build_test_app()
    app.app_config["_first_run"] = False
    app.app_config.setdefault("first_run", {})["setup_completed"] = True
    cancel_event = threading.Event()
    thread_started = threading.Event()
    thread_settled = threading.Event()
    durable_count = 0

    async def _runner(_generation: str) -> None:
        nonlocal durable_count

        def _blocking_success() -> None:
            try:
                thread_started.set()
                assert cancel_event.wait(2)
            finally:
                thread_settled.set()

        await asyncio.to_thread(_blocking_success)
        durable_count += 1

    async with app.run_test(size=(160, 48)):
        operation = app.console_image_edit_operations.start(
            session_id="shutdown-session",
            attachment_id="shutdown-attachment",
            captured_draft="shutdown draft",
            cancel_event=cancel_event,
            runner=_runner,
        )
        assert operation is not None
        assert await asyncio.to_thread(thread_started.wait, 2)

    try:
        assert cancel_event.is_set()
        assert thread_settled.is_set()
        assert durable_count == 1
        assert app.console_image_edit_operations.active("shutdown-session") is None
    finally:
        cancel_event.set()
        if not operation.task.done():
            await operation.task


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_kind", ("success", "failure", "cancel"))
async def test_fresh_mounted_screen_settles_late_h3_outcome_in_dom_and_controls(
    monkeypatch,
    tmp_path,
    terminal_kind,
):
    """Every terminal H3 outcome settles the adopted screen, never the old one."""
    _patch_h3_enabled(monkeypatch)
    db = CharactersRAGDB(
        tmp_path / f"h3-live-settlement-{terminal_kind}.sqlite", "test_client"
    )
    try:
        app = _build_test_app()
        app.chachanotes_db = db
        host = ConsoleHarness(app)
        started = threading.Event()
        release = threading.Event()

        def _terminal_batch(**kwargs):
            started.set()
            assert release.wait(10)
            if terminal_kind == "failure":
                raise RuntimeError("private late failure /private/source.png")
            if terminal_kind == "cancel":
                raise ImageGenerationCancelled()
            return BatchResult(
                successes=[
                    (
                        _png_bytes((19, 11)),
                        "image/png",
                        GenerationVariantMeta(
                            prompt=kwargs["prompt"],
                            negative_prompt="",
                            backend="comfyui",
                            model=None,
                            seed=73,
                            style=None,
                            params={"operation": "edit"},
                        ),
                    )
                ],
                errors=[],
            )

        monkeypatch.setattr(image_module, "run_generation_batch", _terminal_batch)
        async with host.run_test(size=(160, 48)) as pilot:
            old = host.screen_stack[-1]
            await _wait_for_selector(old, pilot, "#console-native-composer")
            old_store = ConsoleChatStore(persistence=ChatPersistenceService(db))
            old._console_chat_store = old_store
            session = old_store.create_session(title="Live H3 settlement")
            source = _real_image_pending(
                "source.png", attachment_id="live-settlement-source"
            )
            old_store.add_pending_attachment(session.id, source)
            old_store.set_session_draft(session.id, "settlement draft")
            old.query_one("#console-native-composer").load_draft("settlement draft")
            caller = asyncio.create_task(
                old._image._console_command_generate_image(
                    CommandParse(
                        kind="command",
                        name="generate-image",
                        args=":comfyui settle terminal",
                    )
                )
            )
            assert await asyncio.to_thread(started.wait, 2)
            operation = app.console_image_edit_operations.active(session.id)
            assert operation is not None
            old_generation = operation.generation
            saved_state = old.save_state()

            await asyncio.wait_for(host.pop_screen(), timeout=0.5)

            async def _stale_sync_tripwire():
                raise AssertionError("unmounted screen UI sync")

            old._sync_native_console_chat_ui = _stale_sync_tripwire
            old._message._sync_native_console_chat_ui_fn = _stale_sync_tripwire

            fresh = ChatScreen(app)
            fresh.restore_state(saved_state)
            await host.push_screen(fresh)
            await _wait_for_selector(fresh, pilot, "#console-native-composer")
            await fresh._sync_native_console_chat_ui()
            assert fresh._native_run_status_copy() == "Stopping image edit…"
            stop = fresh.query_one("#console-stop-generation", Button)
            assert stop.styles.display != "none"
            assert not stop.disabled

            settlement_events: list[str] = []
            original_reconcile = fresh._image._reconcile_h3_image_edit_completions
            original_transcript_sync = fresh._sync_native_console_chat_ui
            original_control_sync = fresh._request_console_control_bar_sync

            def _record_reconcile(store=None):
                settlement_events.append("reconcile")
                original_reconcile(store)

            async def _record_transcript_sync():
                settlement_events.append("transcript")
                await original_transcript_sync()

            def _record_control_sync():
                settlement_events.append("controls")
                original_control_sync()

            fresh._image._reconcile_h3_image_edit_completions = _record_reconcile
            fresh._sync_native_console_chat_ui = _record_transcript_sync
            fresh._request_console_control_bar_sync = _record_control_sync

            release.set()
            await caller
            await operation.task
            expected_content = {
                "success": "[image] settle terminal",
                "failure": (
                    "The image-edit operation did not complete. Please try again."
                ),
            }.get(terminal_kind)

            def _settled() -> bool:
                if "controls" not in settlement_events:
                    return False
                if stop.styles.display != "none":
                    return False
                if expected_content is None:
                    return True
                fresh_store = fresh._ensure_console_chat_store()
                messages = fresh_store.messages_for_session(session.id)
                matching = [
                    message
                    for message in messages
                    if message.content == expected_content
                ]
                if len(matching) != 1:
                    return False
                return bool(fresh.query(f"#console-message-{matching[0].id}"))

            await _wait_for_h3_state(
                pilot,
                _settled,
                detail=f"late H3 {terminal_kind} transcript/control settlement",
            )
            assert fresh._native_run_status_copy() == ""
            assert stop.styles.display == "none"
            assert settlement_events.index("reconcile") < settlement_events.index(
                "transcript"
            )
            assert settlement_events.index("transcript") < settlement_events.index(
                "controls"
            )
            if expected_content is None:
                assert (
                    fresh._ensure_console_chat_store().messages_for_session(session.id)
                    == []
                )
            else:
                matching = [
                    message
                    for message in fresh._ensure_console_chat_store().messages_for_session(
                        session.id
                    )
                    if message.content == expected_content
                ]
                assert len(matching) == 1
                assert expected_content in _message_row_plain_text(
                    fresh, matching[0].id
                )

            if terminal_kind == "cancel":
                fresh._image._reconcile_h3_image_edit_completions = original_reconcile
                fresh._sync_native_console_chat_ui = original_transcript_sync
                fresh._request_console_control_bar_sync = original_control_sync
                newer_release = asyncio.Event()

                async def _newer_runner(_generation: str) -> None:
                    await newer_release.wait()

                newer = app.console_image_edit_operations.start(
                    session_id=session.id,
                    attachment_id=source.attachment_id,
                    captured_draft="settlement draft",
                    cancel_event=threading.Event(),
                    runner=_newer_runner,
                )
                assert newer is not None
                fresh._request_console_control_bar_sync()
                await _wait_for_h3_state(
                    pilot,
                    lambda: stop.styles.display != "none",
                    detail="newer H3 controls",
                )
                transcript_syncs = 0
                control_syncs = 0
                newer_transcript_sync = fresh._sync_native_console_chat_ui
                newer_control_sync = fresh._request_console_control_bar_sync

                async def _count_transcript_sync():
                    nonlocal transcript_syncs
                    transcript_syncs += 1
                    await newer_transcript_sync()

                def _count_control_sync():
                    nonlocal control_syncs
                    control_syncs += 1
                    newer_control_sync()

                fresh._sync_native_console_chat_ui = _count_transcript_sync
                fresh._request_console_control_bar_sync = _count_control_sync
                await fresh._image._settle_current_h3_outcome(
                    session.id, old_generation
                )
                assert transcript_syncs == 0
                assert control_syncs == 0
                assert stop.styles.display != "none"
                newer_release.set()
                await newer.task
                fresh._console_h3_ui_generations = {session.id: newer.generation}
                await fresh._image._settle_current_h3_outcome(
                    session.id, old_generation
                )
                assert transcript_syncs == 0
                assert control_syncs == 0
                assert stop.styles.display != "none"
    finally:
        db.close_connection()


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

    monkeypatch.setattr(image_module, "run_generation_batch", _failed_batch)
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
        old.query_one("#console-native-composer").load_draft("captured failure draft")
        caller = asyncio.create_task(
            old._image._console_command_generate_image(
                CommandParse(
                    kind="command", name="generate-image", args=":comfyui change it"
                )
            )
        )
        assert await asyncio.to_thread(started.wait, 2)
        operation = app.console_image_edit_operations.active(session.id)
        assert operation is not None
        await asyncio.wait_for(host.pop_screen(), timeout=0.5)

        async def _stale_sync_tripwire():
            raise AssertionError("unmounted screen UI sync")

        old._sync_native_console_chat_ui = _stale_sync_tripwire
        old._message._sync_native_console_chat_ui_fn = _stale_sync_tripwire
        release.set()
        await caller
        await operation.task

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
async def test_late_first_persisted_h3_failure_reconciles_through_normal_restore(
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
            assert release.wait(10)
            if failure_kind == "network":
                raise ComfyUIImageEditError("source_upload")
            raise RuntimeError("sentinel response body /private/source.png")

        monkeypatch.setattr(image_module, "run_generation_batch", _failed_batch)
        async with host.run_test(size=(160, 48)) as pilot:
            old = host.screen_stack[-1]
            await _wait_for_selector(old, pilot, "#console-native-composer")
            old_store = ConsoleChatStore(persistence=ChatPersistenceService(db))
            old._console_chat_store = old_store
            session = old_store.create_session(title="Durable H3 failure")
            assert session.persisted_conversation_id is None
            source = _real_image_pending(
                "source.png", attachment_id="durable-failure-source"
            )
            other = PendingAttachment(
                file_path="/private/sentinel/context.txt",
                display_name="context.txt",
                file_type="text",
                insert_mode="attachment",
                data=b"preserved context",
                mime_type="text/plain",
                original_size=len(b"preserved context"),
                processed_size=len(b"preserved context"),
                attachment_id="preserved-other-source",
            )
            old_store.add_pending_attachment(session.id, source)
            old_store.set_session_draft(session.id, "preserved failure draft")
            old.query_one("#console-native-composer").load_draft(
                "preserved failure draft"
            )
            caller = asyncio.create_task(
                old._image._console_command_generate_image(
                    CommandParse(
                        kind="command",
                        name="generate-image",
                        args=":comfyui change it",
                    )
                )
            )
            assert await asyncio.to_thread(started.wait, 2)
            assert old_store.add_pending_attachment(session.id, other)
            operation = app.console_image_edit_operations.active(session.id)
            assert operation is not None
            saved_state = old.save_state()
            assert session.persisted_conversation_id is None
            await asyncio.wait_for(host.pop_screen(), timeout=0.5)

            async def _stale_sync_tripwire():
                raise AssertionError("unmounted screen UI sync")

            old._sync_native_console_chat_ui = _stale_sync_tripwire
            old._message._sync_native_console_chat_ui_fn = _stale_sync_tripwire

            fresh = ChatScreen(app)
            fresh.restore_state(saved_state)
            await host.push_screen(fresh)
            await _wait_for_selector(fresh, pilot, "#console-native-composer")
            fresh_store = fresh._ensure_console_chat_store()
            restored = next(
                item for item in fresh_store.sessions() if item.id == session.id
            )
            assert restored.persisted_conversation_id is None
            assert fresh_store.pending_attachments(session.id) == [source, other]
            assert fresh_store.session_draft(session.id) == "preserved failure draft"

            release.set()
            await caller
            await operation.task
            for _ in range(10):
                await pilot.pause()
                if app.console_image_edit_operations.failure_notice(session.id) is None:
                    break

            messages = fresh_store.messages_for_session(session.id)
            guidance = [
                message
                for message in messages
                if message.role is ConsoleMessageRole.SYSTEM
            ]
            assert len(guidance) == 1
            assert guidance[0].content == expected_copy
            assert guidance[0].persisted_message_id
            persisted_guidance_id = guidance[0].persisted_message_id
            assert restored.persisted_conversation_id is not None
            assert fresh_store.pending_attachments(session.id) == [source, other]
            assert fresh_store.session_draft(session.id) == "preserved failure draft"
            assert app.console_image_edit_operations.failure_notice(session.id) is None

            second_state = fresh.save_state()
            await asyncio.wait_for(host.pop_screen(), timeout=0.5)
            remounted = ChatScreen(app)
            remounted.restore_state(second_state)
            await host.push_screen(remounted)
            await _wait_for_selector(remounted, pilot, "#console-native-composer")
            remounted_store = remounted._ensure_console_chat_store()
            remounted_guidance = [
                message
                for message in remounted_store.messages_for_session(session.id)
                if message.role is ConsoleMessageRole.SYSTEM
            ]
            assert len(remounted_guidance) == 1
            assert remounted_guidance[0].content == expected_copy
            assert remounted_guidance[0].persisted_message_id == persisted_guidance_id
            assert remounted_store.pending_attachments(session.id) == [source, other]
            assert remounted_store.session_draft(session.id) == (
                "preserved failure draft"
            )
            assert "sentinel" not in repr(remounted_guidance)
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
            old._image._filter_h3_attachment_from_app_stash(
                session.id, source.attachment_id
            )
            return record

        if success_timing == "before_stash":
            completion = _commit_success()
            assert old._image._cleanup_h3_completion_in_store(
                old_store, completion, clear_visible_composer=False
            )
            assert registry.ack_completion(session.id, generation)

        payload = old._serialize_native_console_state()
        assert payload is not None

        if success_timing == "after_stash_before_adoption":
            completion = _commit_success()
            assert old._image._cleanup_h3_completion_in_store(
                old_store, completion, clear_visible_composer=False
            )

        # task-15860 Task 3: the second screen does NOT get a fresh store.
        # Console history is owned by the app-owned `ConsoleRuntime` and
        # survives the screen, so `_restore_native_console_state` restores
        # view state only -- a second screen on the same app reads the same
        # `ConsoleChatStore` the first one left behind. Building a fresh
        # store here would test a mechanism production no longer has (and
        # did: the restore left it empty). The H3 boundaries below are
        # unchanged -- they are about reconciling a completed edit exactly
        # once, at each timing.
        fresh = ChatScreen(app)
        fresh_store = fresh._ensure_console_chat_store()
        assert fresh_store is old_store, (
            "the app-owned runtime must hand the second screen the surviving store"
        )
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
            fresh._image._reconcile_h3_image_edit_completions(fresh_store)

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

        registry = console._image._h3_image_edit_registry()
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
        assert registry.publish_failure_notice(
            ImageEditFailureNotice(
                session_id=doomed.id,
                generation=operation.generation,
                message_id="doomed-system-message",
            )
        )
        await console._sync_native_console_chat_ui()
        close_selector = f"#console-close-session-tab-{doomed.id}"
        await _wait_for_selector(console, pilot, close_selector)
        console.query_one(close_selector, Button).press()
        await pilot.pause()
        host.screen_stack[-1].query_one("#confirm-button", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert doomed.id not in {item.id for item in store.sessions()}
        assert operation.cancel_event.is_set()
        assert registry.active(doomed.id) is None
        assert registry.completion(doomed.id) is None
        assert registry.failure_notice(doomed.id) is None
        release.set()
        await operation.task
