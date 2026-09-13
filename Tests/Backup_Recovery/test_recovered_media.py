"""Real-file lifecycle checks for the recovered-media owner (ADR-126)."""

import pytest

from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414
)


def test_deleted_asset_is_not_an_unexpected_missing_file(tmp_path, local_root):
    from tldw_chatbook.Backup_Recovery.recovered_media import RecoveredMedia

    source = tmp_path / "synthetic.webm"
    source.write_bytes(b"not decoded during registration")
    store = RecoveredMedia(tmp_path / "recovered")
    asset_id = store.retain(
        source, profile="p", message="m", slug="clip", media_type="video/webm"
    )
    store.delete(asset_id)
    assert store.resolve(asset_id) == ("deleted", None)


@pytest.fixture
def media(tmp_path, local_root):
    from tldw_chatbook.Backup_Recovery.recovered_media import RecoveredMedia

    source = tmp_path / "source"
    source.write_bytes(b"opaque media, never decoded")
    return RecoveredMedia(tmp_path / "recovered"), source


def retain(store, source, **overrides):
    identity = {
        "profile": "p",
        "message": "m",
        "slug": "same",
        "media_type": "image/png",
    }
    identity.update(overrides)
    return store.retain(source, **identity)


def test_identity_collisions_and_profile_isolation(media):
    store, source = media
    first = retain(store, source)
    assert retain(store, source) == first
    assert retain(store, source, profile="other") != first
    source.write_bytes(b"different")
    with pytest.raises(ValueError, match="collision"):
        retain(store, source)


@pytest.mark.parametrize("corrupt", [False, True])
def test_known_missing_never_becomes_deleted(media, corrupt):
    store, source = media
    asset = retain(store, source)
    path = store.resolve(asset)[1]
    if corrupt:
        path.write_bytes(b"bad")
    else:
        path.unlink()
    assert store.resolve(asset) == ("missing", None)


def test_shared_references_and_tombstone_restart(media):
    from tldw_chatbook.Backup_Recovery.recovered_media import RecoveredMedia

    store, source = media
    asset = retain(store, source)
    store.add_reference(
        asset, profile="p", message="other", slug="same", media_type="image/png"
    )
    store.delete(asset)
    restarted = RecoveredMedia(store.root)
    assert restarted.resolve_reference(
        profile="p", message="other", slug="same", media_type="image/png"
    ) == ("deleted", None)


@pytest.mark.parametrize("operation", ["retain", "delete"])
def test_interrupted_publication_or_retirement_recovers(media, monkeypatch, operation):
    from tldw_chatbook.Backup_Recovery.recovered_media import RecoveredMedia

    store, source = media
    asset = retain(store, source) if operation == "delete" else None
    method = "_finish_delete" if operation == "delete" else "_finish_retain"

    def stop(*args):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(store, method, stop)
    with pytest.raises(RuntimeError, match="interrupted"):
        store.delete(asset) if asset else retain(store, source)
    restarted = RecoveredMedia(store.root)
    state, path = restarted.resolve_reference(
        profile="p", message="m", slug="same", media_type="image/png"
    )
    assert state == ("deleted" if operation == "delete" else "ready")
    if path:
        assert path.read_bytes() == source.read_bytes()


def test_orphan_cleanup_rechecks_references_and_holds(media):
    store, source = media
    asset = retain(store, source)
    store.hold(asset, "recovery")
    store.release_reference(
        profile="p", message="m", slug="same", media_type="image/png"
    )
    assert not store.cleanup_orphan(asset)
    store.release_hold(asset, "recovery")
    assert store.cleanup_orphan(asset)
    assert store.resolve(asset) == ("deleted", None)


def test_recovered_owner_cannot_open_after_admission_closes(media):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.participants import _repository_participant

    store, source = media
    participant = _repository_participant(store)
    participant.close_admission()
    with pytest.raises(RecoveryRequired, match="paused"):
        retain(store, source)
    participant.resume()
    assert retain(store, source)


def test_adapter_catalog_and_payloads_are_one_baseline_group(media):
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
    )

    store, source = media
    asset = retain(store, source)
    adapter = store.recovery_adapter()
    config = {DISCOVERY_CONTEXT_KEY: DiscoveryContext(source, "p")}
    entries = adapter.discover(config)
    assert any(item.path == store.resolve(asset)[1] for item in entries)
    assert adapter.validate(store.db_path) == ()
    store.delete(asset)
    assert adapter.validate(store.db_path) == ()
    assert not any(item.path == store._path(asset) for item in adapter.discover(config))


@pytest.mark.parametrize("state", ["ready", "missing", "deleted"])
def test_video_resolver_never_falls_back_for_recovered_reference(media, state):
    from tldw_chatbook.Video_Generation.video_store import VideoStore

    store, source = media
    asset = retain(store, source, media_type="video/webm")
    if state == "missing":
        store.resolve(asset)[1].unlink()
    elif state == "deleted":
        store.delete(asset)
    videos = VideoStore(
        store.root.parent / "generated_videos",
        recovered_root=store.root,
        recovered_profile="p",
    )
    temporary = videos.root / "m" / "same.webm"
    temporary.parent.mkdir(parents=True)
    temporary.write_bytes(b"unrelated temporary")
    status, path = videos.resolve_state("m", "same", extension="webm")
    assert status == ("ready" if state == "ready" else "recovered_" + state)
    assert path != temporary


def test_enhanced_image_widget_renders_known_deletion(media):
    from tldw_chatbook.Widgets.chat_message_enhanced import ChatMessageEnhanced

    store, source = media
    asset = retain(store, source)
    store.delete(asset)
    widget = ChatMessageEnhanced(
        "caption",
        "assistant",
        message_id="m",
        image_data=b"unrelated",
        image_mime_type="image/png",
        recovered_root=store.root,
        recovered_profile="p",
    )
    assert widget.image_data is None
    assert widget.recovered_image_status == "deleted"


def test_generated_id_collision_never_overwrites_existing_payload(media, monkeypatch):
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery import recovered_media

    store, source = media
    collision = "a" * 32
    existing = store.root / (collision + ".payload")
    existing.write_bytes(b"owned by another operation")
    monkeypatch.setattr(
        recovered_media.uuid, "uuid4", lambda: SimpleNamespace(hex=collision)
    )
    with pytest.raises((ValueError, FileExistsError)):
        retain(store, source)
    assert existing.read_bytes() == b"owned by another operation"


def test_delete_refuses_unregistered_replacement_bytes(media):
    store, source = media
    asset = retain(store, source)
    path = store.resolve(asset)[1]
    path.write_bytes(b"unregistered replacement")
    with pytest.raises(ValueError, match="replacement"):
        store.delete(asset)
    assert path.read_bytes() == b"unregistered replacement"


def test_adapter_rejects_reference_type_mismatch(media):
    store, source = media
    retain(store, source)
    with store._connection() as connection, connection:
        connection.execute("UPDATE refs SET media_type='video/webm'")
    assert store.recovery_adapter().validate(store.db_path)


def test_catalog_payload_dependency_validation_rejects_missing_candidate(media):
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
    )

    store, source = media
    retain(store, source)
    adapter = store.recovery_adapter()
    entries = adapter.discover({DISCOVERY_CONTEXT_KEY: DiscoveryContext(source, "p")})
    catalog = next(item for item in entries if item.path == store.db_path)
    candidates = {item.logical_id: item.path for item in entries}
    assert adapter.validate_dependencies(catalog, store.db_path, candidates) == ()
    payload = next(item for item in entries if item.path.suffix == ".payload")
    del candidates[payload.logical_id]
    assert adapter.validate_dependencies(catalog, store.db_path, candidates) == (
        "recovered_payload_missing",
    )


def test_temporary_startup_cleanup_leaves_recovered_payload(media):
    from tldw_chatbook.Video_Generation.video_store import VideoStore

    store, source = media
    asset = retain(store, source, media_type="video/webm")
    videos = VideoStore(
        store.root.parent / "generated_videos",
        recovered_root=store.root,
        recovered_profile="p",
    )
    videos.clear_all()
    assert store.resolve(asset)[0] == "ready"


def test_catalog_round_trip_preserves_deleted_references(media, tmp_path):
    import shutil

    from tldw_chatbook.Backup_Recovery.recovered_media import RecoveredMedia

    store, source = media
    asset = retain(store, source)
    store.delete(asset)
    copied = tmp_path / "relocated"
    shutil.copytree(store.root, copied)
    restored = RecoveredMedia(copied)
    assert restored.resolve(asset) == ("deleted", None)
    assert restored.recovery_adapter().validate(restored.db_path) == ()


@pytest.mark.parametrize("state", ["missing", "deleted"])
def test_video_card_names_the_recovered_state(state):
    from Tests.Widgets.test_console_video_card import _spec
    from tldw_chatbook.Widgets.Console.console_video_card import video_card_status_line

    assert (
        video_card_status_line(_spec("recovered_" + state))
        == state.capitalize() + " recovered media"
    )


def test_pending_retain_retry_preserves_unrelated_destination(media, monkeypatch):
    from tldw_chatbook.Backup_Recovery import recovered_media

    store, source = media
    original = recovered_media.atomic_private_write_bytes

    def stop(*args, **kwargs):
        raise RuntimeError("before publication")

    monkeypatch.setattr(recovered_media, "atomic_private_write_bytes", stop)
    with pytest.raises(RuntimeError):
        retain(store, source)
    with store._connection() as connection:
        asset = connection.execute("SELECT asset_id FROM assets").fetchone()[0]
    occupied = store._path(asset)
    occupied.write_bytes(b"unrelated")
    monkeypatch.setattr(recovered_media, "atomic_private_write_bytes", original)
    restarted = recovered_media.RecoveredMedia(store.root)
    with pytest.raises(ValueError, match="collision"):
        retain(restarted, source)
    assert occupied.read_bytes() == b"unrelated"


def test_inconsistent_delete_journal_never_retires_ready_payload(media):
    import json

    from tldw_chatbook.Backup_Recovery.recovered_media import RecoveredMedia

    store, source = media
    asset = retain(store, source)
    payload = store._path(asset)
    with store._connection() as connection, connection:
        refs = list(
            connection.execute("SELECT profile,message,slug,media_type FROM refs")
        )
        connection.execute(
            "INSERT INTO tombstones VALUES (?,1,?)", (asset, json.dumps(refs))
        )
        connection.execute("INSERT INTO operations VALUES (?,'delete')", (asset,))
    with pytest.raises(ValueError, match="tombstone"):
        RecoveredMedia(store.root)
    assert payload.read_bytes() == source.read_bytes()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["ready", "missing", "deleted", "failed", "corrupt"])
@pytest.mark.parametrize("transient", [None, b"unrelated transient image"])
async def test_actual_console_image_specs_override_stale_cache(media, state, transient, monkeypatch):
    import inspect
    from io import BytesIO
    from types import SimpleNamespace

    from PIL import Image

    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage,
        ConsoleMessageRole,
    )
    from tldw_chatbook.Chat.console_image_view import (
        ConsoleImageRenderCache,
        ConsoleImageViewState,
    )
    from tldw_chatbook.UI.Console_Modules.image import ConsoleImageController
    from tldw_chatbook.UI.Console_Modules.message import ConsoleMessageController
    from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

    store, source = media
    data = BytesIO()
    Image.new("RGB", (2, 2), "red").save(data, format="PNG")
    source.write_bytes(b"invalid image" if state == "failed" else data.getvalue())
    asset = retain(store, source)
    if state == "missing":
        store.resolve(asset)[1].unlink()
    elif state == "deleted":
        store.delete(asset)
    elif state == "corrupt":
        store.db_path.write_bytes(b"invalid sqlite catalog")
    cache, view = ConsoleImageRenderCache(), ConsoleImageViewState()
    screen = SimpleNamespace(
        app_instance=SimpleNamespace(
            app_config={"chat": {"images": {"render_remote_images": True}}}
        )
    )

    def unused(*args, **kwargs):
        raise AssertionError("unused controller dependency")

    callbacks = {
        name: unused
        for name in inspect.signature(ConsoleImageController).parameters
        if name not in {"screen", "app_instance"}
    }
    callbacks.update(
        ensure_console_image_view=lambda: (view, cache),
        recent_console_image_messages=lambda messages: (
            ConsoleMessageController._recent_console_image_messages(None, messages)
        ),
        console_image_default_mode=lambda: "pixels",
    )
    screen._image = ConsoleImageController(
        screen, app_instance=screen.app_instance, **callbacks
    )
    screen.app_instance.recovered_media_root = store.root
    screen.app_instance.recovered_media_profile = "p"
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="![unrelated](https://example.invalid/transient.png)",
        status="complete",
    )
    message.persisted_message_id = "m"
    message.image_data = transient
    stale = BytesIO()
    Image.new("RGB", (2, 2), "blue").save(stale, format="PNG")
    cache.prepare(message.id, stale.getvalue())
    workers = []
    screen.run_worker = lambda coroutine, **kwargs: workers.append(coroutine)

    async def synced():
        pass

    screen._image._sync_native_console_chat_ui_fn = synced
    import threading

    from tldw_chatbook.Backup_Recovery import recovered_media

    original_read = recovered_media._read
    reads = []
    entered, release = threading.Event(), threading.Event()

    def worker_read(path):
        assert threading.current_thread() is not threading.main_thread()
        reads.append(path)
        entered.set()
        assert release.wait(3)
        return original_read(path)

    monkeypatch.setattr(recovered_media, "_read", worker_read)
    specs = screen._image._build_console_image_specs([message])
    import asyncio

    if state in {"ready", "missing", "failed"}:
        async with asyncio.timeout(2):
            while not entered.is_set():
                await asyncio.sleep(0.01)
        # The UI event loop remains available while the actual file read waits.
        assert screen._image._recovered_image_tasks
    release.set()
    await asyncio.gather(*screen._image._recovered_image_tasks)
    specs = screen._image._build_console_image_specs([message])
    await asyncio.gather(*screen._image._recovered_image_tasks)
    spec = specs[message.id]
    expected = "missing" if state == "corrupt" else state
    assert spec.recovered_status == expected
    if state == "ready":
        assert cache.get_pil(spec.recovered_key).getpixel((0, 0)) == (255, 0, 0)
        assert len(reads) == 1
    else:
        assert spec.pil is None and spec.pixels is None
        widget = ConsoleTranscript()._image_row_widget(spec)
        assert expected.capitalize() + " recovered media" in str(widget.render())
    if state == "ready":
        monkeypatch.setattr(recovered_media, "_read", original_read)
        store.delete(asset)
        screen._image._build_console_image_specs([message])
        await asyncio.gather(*screen._image._recovered_image_tasks)
        screen._image._recovered_images_close_admission()
        assert screen._image._build_console_image_specs([message])[message.id].recovered_status == "deleted"


@pytest.mark.asyncio
async def test_recovered_image_cancel_keeps_worker_until_native_read_settles(media, monkeypatch):
    import asyncio
    import threading
    import time

    from tldw_chatbook.Backup_Recovery import recovered_media
    from tldw_chatbook.Chat.console_image_view import ConsoleImageRenderCache
    from tldw_chatbook.UI.Console_Modules.image import ConsoleImageController

    store, source = media
    retain(store, source)
    entered, release = threading.Event(), threading.Event()
    read = recovered_media._read

    def blocked(path):
        entered.set()
        assert release.wait(3)
        return read(path)

    monkeypatch.setattr(recovered_media, "_read", blocked)
    controller = object.__new__(ConsoleImageController)
    controller._recovered_image_tasks = set()
    controller._recovered_image_paused = False
    task = asyncio.create_task(controller._load_recovered_images(
        (store.root, "p", (("ui-id", "m", "pixels"),)), ConsoleImageRenderCache()
    ))
    controller._recovered_image_tasks.add(task)
    task.add_done_callback(controller._recovered_image_tasks.discard)
    try:
        async with asyncio.timeout(2):
            while not entered.is_set():
                await asyncio.sleep(0.01)
        controller._recovered_images_close_admission()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        assert not await controller._recovered_images_drain(time.monotonic() + 0.03)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    assert await controller._recovered_images_drain(time.monotonic() + 1)


def test_recovered_image_metadata_never_reads_historical_payloads(media, monkeypatch):
    from tldw_chatbook.Backup_Recovery import recovered_media

    store, source = media
    retain(store, source)

    def unexpected(path):
        raise AssertionError("metadata lookup loaded a payload")

    monkeypatch.setattr(recovered_media, "_read", unexpected)
    metadata = recovered_media.message_image_metadata(
        store.root, "p", ["m", *[f"absent-{i}" for i in range(402)]]
    )
    assert list(metadata) == ["m"]
