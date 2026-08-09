"""Store-level tests for video generation messages (task-3401.4).

Follows ``test_console_generation_store.py``'s fixture/fake-persistence
style, plus one real-DB reload test mirroring its integration round trip.
Covers: the ``[video]`` marker as the only durable reference, the
namespaced video payload in ``metadata_json`` (never the v25 sidecar,
never provenance keys), clobber guards on every persistence seam, and
image-reader isolation (ADR-044).
"""

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
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata
from tldw_chatbook.Video_Generation.video_store import VideoStore, parse_video_marker

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
    return SimpleNamespace(
        retention="ttl", retention_ttl_hours=1, max_store_mb=2048
    )


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


def test_video_card_uses_persisted_id_for_storage_resolution(tmp_path):
    video_store = VideoStore(root=tmp_path / "generated_videos", config=_ttl_config())
    persisted_id = "persisted-video-message"
    stored_path = video_store.save(
        persisted_id, _video_meta().name, b"video-bytes"
    )
    message = ConsoleChatMessage(
        id="fresh-native-message",
        persisted_message_id=persisted_id,
        role=ConsoleMessageRole.ASSISTANT,
        content="[video] dusk-over-neon-tokyo",
        video_metadata=_video_meta(),
    )
    screen = ChatScreen.__new__(ChatScreen)
    screen._console_video_store = video_store

    specs = screen._build_video_card_specs([message])

    assert set(specs) == {message.id}
    assert specs[message.id].message_id == message.id
    assert specs[message.id].status == "ready"
    assert specs[message.id].file_path == str(stored_path)


@pytest.mark.integration
def test_video_message_reload_round_trip_and_image_reader_isolation(tmp_path):
    """Real-DB reload: persist a video message, drop the store, resume via
    the real converter + restore path. The video facts survive in
    metadata_json; image-generation readers see nothing (ADR-044)."""
    db = CharactersRAGDB(tmp_path / "video_reload.sqlite", "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.create_session(title="Video reload")
        store.active_session_id = session.id

        preallocated_id = "persisted-video-reload-id"
        video_root = tmp_path / "generated_videos"
        initial_video_store = VideoStore(root=video_root, config=_ttl_config())
        stored_path = initial_video_store.save(
            preallocated_id, _video_meta().name, b"video-bytes"
        )
        msg = store.append_video_message(
            session.id,
            video_metadata=_video_meta(),
            persist=True,
            message_id=preallocated_id,
        )
        conversation_id = session.persisted_conversation_id
        assert conversation_id is not None

        # ---- Simulate reload: persist -> DROP the store -> fresh store ----
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

        reloaded = fresh_store.messages_for_session(fresh_session.id)
        video_msg = next(m for m in reloaded if m.video_metadata is not None)

        assert video_msg.id != preallocated_id
        assert video_msg.persisted_message_id == preallocated_id

        fresh_video_store = VideoStore(root=video_root, config=_ttl_config())
        retention = fresh_video_store.enforce_retention(
            now=stored_path.stat().st_mtime + 30 * 60
        )
        assert retention.removed_files == 0
        screen._console_video_store = fresh_video_store
        specs = screen._build_video_card_specs(reloaded)
        assert specs[video_msg.id].status == "ready"
        assert specs[video_msg.id].file_path == str(stored_path)

        # The named card's facts and still-within-TTL bytes survive reload.
        assert video_msg.video_metadata == _video_meta()
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
        assert VideoGenerationMetadata.from_json(payload["metadata_json"]) == _video_meta()
    finally:
        db.close_connection()
