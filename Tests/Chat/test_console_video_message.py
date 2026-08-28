"""Store-level tests for video generation messages (task-3401.4).

Follows ``test_console_generation_store.py``'s fixture/fake-persistence
style, plus one real-DB reload test mirroring its integration round trip.
Covers: the ``[video]`` marker as the only durable reference, the
namespaced video payload in ``metadata_json`` (never the v25 sidecar,
never provenance keys), clobber guards on every persistence seam, and
image-reader isolation (ADR-044).
"""

import json
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_chatbook.UI.Console_Modules.message import ConsoleMessageController
from tldw_chatbook.UI.Console_Modules.video import ConsoleVideoController
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata
from tldw_chatbook.Video_Generation.video_store import VideoStore, parse_video_marker
from tldw_chatbook.Widgets.Console.console_video_card import (
    ConsoleVideoCard,
    ConsoleVideoCardSpec,
)

from Tests.UI.app_factory import _build_test_app


def _video_meta(**overrides):
    base = {
        "name": "dusk-over-neon-tokyo",
        "prompt": "dusk over neon tokyo, cinematic",
        "backend": "minimax",
        "model": "MiniMax-H3",
        "seed": 7,
        "duration_seconds": 6.0,
    }
    base.update(overrides)
    return VideoGenerationMetadata(**base)


def _ttl_config():
    return SimpleNamespace(retention="ttl", retention_ttl_hours=1, max_store_mb=2048)


def _video_controller(video_store: VideoStore) -> ConsoleVideoController:
    return ConsoleVideoController(
        app_instance=SimpleNamespace(generated_video_store=video_store),
        sync_native_console_chat_ui=lambda: None,
        ensure_console_chat_store=lambda: None,
        wait_for_console_screen_result=lambda _screen: None,
        open_video_with_os=lambda _path: None,
        append_native_console_system_message=lambda *_args, **_kwargs: None,
        default_console_session_settings=lambda: None,
        console_composer_or_none=lambda: None,
        clear_console_composer_draft=lambda: None,
    )


def test_video_metadata_tombstone_discriminator_is_strict_and_backward_compatible():
    assert "is_unavailable_tombstone" in VideoGenerationMetadata.__dataclass_fields__
    live = _video_meta()
    tombstone = VideoGenerationMetadata(
        name=live.name,
        prompt=live.prompt,
        backend=live.backend,
        is_unavailable_tombstone=True,
    )
    assert live.is_unavailable_tombstone is False
    assert VideoGenerationMetadata.from_json(tombstone.to_json()) == tombstone

    legacy = json.loads(live.to_json())
    legacy["video_generation"].pop("is_unavailable_tombstone", None)
    restored_legacy = VideoGenerationMetadata.from_json(json.dumps(legacy))
    assert restored_legacy is not None
    assert restored_legacy.is_unavailable_tombstone is False

    invalid = json.loads(live.to_json())
    invalid["video_generation"]["is_unavailable_tombstone"] = "yes"
    assert VideoGenerationMetadata.from_json(json.dumps(invalid)) is None


def _reload_video_messages(db, conversation_id):
    conversation_service = ChatConversationService(db)
    tree = conversation_service.get_conversation_tree(
        conversation_id, depth_cap=10_000, root_limit=10_000
    )
    screen = ChatScreen(_build_test_app())
    screen.app_instance.chachanotes_db = db
    all_nodes = screen._console_messages_from_conversation_tree(tree)
    active_leaf_id = db.get_conversation_active_leaf(conversation_id)

    fresh_store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    fresh_session = fresh_store.restore_persisted_session(
        title="Video reload",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=all_nodes,
        active_leaf_persisted_id=active_leaf_id,
    )
    return screen, fresh_store.messages_for_session(fresh_session.id)


class FakeVideoPersistence:
    """Records every call; declares the metadata_json seams via **kwargs."""

    def __init__(self):
        self.created_messages = []
        self.content_updates = []
        self.metadata_updates = []

    def create_conversation(self, **kwargs):
        return "conv-1"

    def create_message(self, **kwargs):
        self.created_messages.append(kwargs)
        message_id = kwargs.get("message_id") or f"msg-{len(self.created_messages)}"
        return message_id

    def update_message_content(self, **kwargs):
        self.content_updates.append(kwargs)
        return True

    def update_message_metadata(self, **kwargs):
        self.metadata_updates.append(kwargs)
        return True


@pytest.fixture
def store_with_session():
    persistence = FakeVideoPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(title="t")
    store.active_session_id = session.id
    return store, session.id, persistence


def test_append_video_message_marker_and_shape(store_with_session):
    store, sid, _ = store_with_session
    msg = store.append_video_message(sid, video_metadata=_video_meta())
    assert msg.content == "[video] dusk-over-neon-tokyo"
    assert parse_video_marker(msg.content) == "dusk-over-neon-tokyo"
    # NOT an image generation message: no attachments, no sidecar metadata.
    assert msg.attachments == ()
    assert msg.generation_metadata == ()
    assert msg.image_data is None
    assert msg.video_metadata == _video_meta()
    # The marker must not collide with the image generation prefix.
    assert not msg.content.startswith("[image] ")


def test_video_card_owns_play_and_save_actions():
    spec = ConsoleVideoCardSpec(
        message_id="video-1",
        meta=_video_meta(),
        status="expired",
    )

    ids = [action.action_id for action in ConsoleVideoCard(spec).actions]

    assert ids == [
        "video-play",
        "video-save-copy",
    ]


def test_append_video_message_persists_namespaced_payload(store_with_session):
    store, sid, persistence = store_with_session
    preallocated_id = "video-message-fixed-id"
    msg = store.append_video_message(
        sid,
        video_metadata=_video_meta(),
        persist=True,
        message_id=preallocated_id,
    )
    created = persistence.created_messages[0]
    assert msg.id == preallocated_id
    assert created["message_id"] == preallocated_id
    assert msg.persisted_message_id == preallocated_id
    payload = created["metadata_json"]
    assert '"video_generation"' in payload
    # Never provenance keys, and never the v25 sidecar kwarg.
    assert '"interrupted"' not in payload
    assert "generation_metadata" not in created
    # The persisted payload round-trips back to the original facts.
    assert VideoGenerationMetadata.from_json(payload) == _video_meta()


@pytest.mark.integration
def test_append_video_message_explicit_id_conflict_propagates_without_overwrite(
    tmp_path,
):
    db = CharactersRAGDB(tmp_path / "video_id_conflict.sqlite", "test_client")
    try:
        persistence = ChatPersistenceService(db)
        original_store = ConsoleChatStore(persistence=persistence)
        original_session = original_store.create_session(title="Original")
        message_id = "video-message-conflict-id"
        original_store.append_video_message(
            original_session.id,
            video_metadata=_video_meta(),
            persist=True,
            message_id=message_id,
        )
        original_row = db.get_message_by_id(message_id)
        assert original_row is not None

        conflicting_store = ConsoleChatStore(persistence=persistence)
        conflicting_session = conflicting_store.create_session(title="Conflict")
        with pytest.raises(ConflictError):
            conflicting_store.append_video_message(
                conflicting_session.id,
                video_metadata=_video_meta(name="different-video"),
                persist=True,
                message_id=message_id,
            )

        assert db.get_message_by_id(message_id) == original_row
    finally:
        db.close_connection()


def test_content_update_rewrites_video_payload_not_provenance(store_with_session):
    """A later content edit threads the SAME video payload (idempotent) --
    it can never clobber the row with an all-defaults provenance payload."""
    store, sid, persistence = store_with_session
    msg = store.append_video_message(sid, video_metadata=_video_meta(), persist=True)
    store._persist_existing_message(msg)
    update = persistence.content_updates[-1]
    assert update["metadata_json"] == _video_meta().to_json()


def test_metadata_flush_prefers_video_payload(store_with_session):
    store, sid, persistence = store_with_session
    msg = store.append_video_message(sid, video_metadata=_video_meta(), persist=True)
    store._persist_metadata_only(msg)
    flush = persistence.metadata_updates[-1]
    assert flush["metadata_json"] == _video_meta().to_json()


def test_screen_state_round_trip_preserves_video_metadata(store_with_session):
    store, sid, _ = store_with_session
    msg = store.append_video_message(sid, video_metadata=_video_meta())
    payload = ConsoleMessageController._serialize_console_message(msg)
    assert payload["metadata_json"] == _video_meta().to_json()
    restored = ConsoleMessageController._restore_console_message(payload)
    assert restored is not None
    assert restored.video_metadata == _video_meta()
    # The provenance slot stays empty on a video row.
    assert restored.metadata is None
    assert parse_video_marker(restored.content) == "dusk-over-neon-tokyo"


def test_webm_video_card_survives_real_screen_state_round_trip(tmp_path):
    metadata = _video_meta(container="webm")
    message = ConsoleChatMessage(
        id="screen-state-webm-message",
        role=ConsoleMessageRole.ASSISTANT,
        content="[video] dusk-over-neon-tokyo",
        video_metadata=metadata,
    )
    video_store = VideoStore(root=tmp_path / "generated_videos", config=_ttl_config())
    stored_path = video_store.save(
        message.id,
        metadata.name,
        b"webm-video-bytes",
        extension="webm",
    )

    payload = ConsoleMessageController._serialize_console_message(message)
    restored = ChatScreen._restore_console_message(payload)
    assert restored is not None

    screen = ChatScreen.__new__(ChatScreen)
    screen._video = _video_controller(video_store)
    spec = screen._video._build_video_card_specs([restored])[restored.id]

    assert restored.video_metadata == metadata
    assert spec.status == "ready"
    assert spec.file_path == str(stored_path)


def test_console_video_store_prefers_explicit_test_override(tmp_path):
    app_store = VideoStore(root=tmp_path / "app")
    override = VideoStore(root=tmp_path / "override")
    screen = ChatScreen(_build_test_app())
    screen.app_instance.generated_video_store = app_store
    screen._console_video_store = override

    assert screen._video._ensure_console_video_store() is override


def test_console_video_store_borrows_app_owner_without_cleanup(
    monkeypatch,
    tmp_path,
):
    store = VideoStore(root=tmp_path / "app")
    screen = ChatScreen(_build_test_app())
    screen.app_instance.generated_video_store = store
    monkeypatch.setattr(
        store,
        "enforce_retention",
        lambda: pytest.fail("screen must not run retention"),
    )

    assert screen._video._ensure_console_video_store() is store
    assert screen._video._ensure_console_video_store() is store


def test_console_video_store_fails_loudly_without_app_owner():
    app = _build_test_app()
    del app.generated_video_store
    screen = ChatScreen(app)

    with pytest.raises(RuntimeError, match="app-owned generated video store"):
        screen._video._ensure_console_video_store()


def test_video_card_uses_persisted_id_for_storage_resolution(tmp_path):
    video_store = VideoStore(root=tmp_path / "generated_videos", config=_ttl_config())
    persisted_id = "persisted-video-message"
    stored_path = video_store.save(
        persisted_id,
        _video_meta().name,
        b"video-bytes",
        extension="mp4",
    )
    message = ConsoleChatMessage(
        id="fresh-native-message",
        persisted_message_id=persisted_id,
        role=ConsoleMessageRole.ASSISTANT,
        content="[video] dusk-over-neon-tokyo",
        video_metadata=_video_meta(),
    )
    screen = ChatScreen.__new__(ChatScreen)
    screen._video = _video_controller(video_store)

    specs = screen._video._build_video_card_specs([message])

    assert set(specs) == {message.id}
    assert specs[message.id].message_id == message.id
    assert specs[message.id].status == "ready"
    assert specs[message.id].file_path == str(stored_path)


def test_video_card_uses_native_id_when_message_is_not_persisted(tmp_path):
    video_store = VideoStore(root=tmp_path / "generated_videos", config=_ttl_config())
    message = ConsoleChatMessage(
        id="native-video-message",
        role=ConsoleMessageRole.ASSISTANT,
        content="[video] dusk-over-neon-tokyo",
        video_metadata=_video_meta(),
    )
    assert message.persisted_message_id is None
    stored_path = video_store.save(
        message.id, _video_meta().name, b"video-bytes", extension="mp4"
    )
    screen = ChatScreen.__new__(ChatScreen)
    screen._video = _video_controller(video_store)

    specs = screen._video._build_video_card_specs([message])

    assert set(specs) == {message.id}
    assert specs[message.id].message_id == message.id
    assert specs[message.id].status == "ready"
    assert specs[message.id].file_path == str(stored_path)


@pytest.mark.integration
def test_webm_video_message_reload_round_trip_and_image_reader_isolation(tmp_path):
    """A fresh Console resolves WebM from metadata persisted in SQLite."""
    db = CharactersRAGDB(tmp_path / "video_reload.sqlite", "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.create_session(title="Video reload")
        store.active_session_id = session.id

        preallocated_id = "persisted-video-reload-id"
        video_root = tmp_path / "generated_videos"
        initial_video_store = VideoStore(root=video_root, config=_ttl_config())
        stored_path = initial_video_store.save(
            preallocated_id,
            _video_meta().name,
            b"video-bytes",
            extension="webm",
        )
        msg = store.append_video_message(
            session.id,
            video_metadata=_video_meta(container="webm"),
            persist=True,
            message_id=preallocated_id,
        )
        conversation_id = session.persisted_conversation_id
        assert conversation_id is not None

        # ---- Simulate reload: persist -> DROP the store -> fresh store ----
        screen, reloaded = _reload_video_messages(db, conversation_id)
        video_msg = next(m for m in reloaded if m.video_metadata is not None)

        assert video_msg.id != preallocated_id
        assert video_msg.persisted_message_id == preallocated_id

        fresh_video_store = VideoStore(root=video_root, config=_ttl_config())
        retention = fresh_video_store.enforce_retention(
            now=stored_path.stat().st_mtime + 30 * 60
        )
        assert retention.removed_files == 0
        screen._console_video_store = fresh_video_store
        specs = screen._video._build_video_card_specs(reloaded)
        assert specs[video_msg.id].status == "ready"
        assert specs[video_msg.id].file_path == str(stored_path)

        # The named card's facts and still-within-TTL bytes survive reload.
        assert video_msg.video_metadata == _video_meta(container="webm")
        assert video_msg.metadata is None
        assert parse_video_marker(video_msg.content) == "dusk-over-neon-tokyo"

        # Image-reader isolation: the v25 sidecar holds nothing for this
        # message, so generation-metadata readers never consume it.
        sidecar = db.get_generation_metadata_for_messages([msg.persisted_message_id])
        assert sidecar.get(msg.persisted_message_id) in (None, [])

        # Export/copy grace: the screen-state serializer (the copy substrate)
        # renders the marker+metadata with no byte access and no error.
        payload = ConsoleMessageController._serialize_console_message(video_msg)
        assert payload["content"] == "[video] dusk-over-neon-tokyo"
        assert VideoGenerationMetadata.from_json(
            payload["metadata_json"]
        ) == _video_meta(container="webm")
    finally:
        db.close_connection()


@pytest.mark.integration
def test_historical_video_message_without_container_reloads_as_mp4(tmp_path):
    db = CharactersRAGDB(tmp_path / "historical_video_reload.sqlite", "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.create_session(title="Historical video reload")
        store.active_session_id = session.id
        message_id = "historical-video-reload-id"
        video_root = tmp_path / "generated_videos"
        video_store = VideoStore(root=video_root, config=_ttl_config())
        stored_path = video_store.save(
            message_id,
            "historical-clip",
            b"historical-mp4-bytes",
            extension="mp4",
        )
        store.append_video_message(
            session.id,
            video_metadata=_video_meta(name="historical-clip"),
            persist=True,
            message_id=message_id,
        )
        historical_payload = json.dumps(
            {
                "video_generation": {
                    "name": "historical-clip",
                    "prompt": "historical prompt",
                    "backend": "minimax",
                }
            }
        )
        assert db.update_message_metadata_local(message_id, historical_payload)
        conversation_id = session.persisted_conversation_id
        assert conversation_id is not None

        screen, reloaded = _reload_video_messages(db, conversation_id)
        video_msg = next(message for message in reloaded if message.video_metadata)
        fresh_video_store = VideoStore(root=video_root, config=_ttl_config())
        screen._console_video_store = fresh_video_store
        spec = screen._video._build_video_card_specs(reloaded)[video_msg.id]

        assert video_msg.video_metadata.container == "mp4"
        assert spec.status == "ready"
        assert spec.file_path == str(stored_path)
    finally:
        db.close_connection()
