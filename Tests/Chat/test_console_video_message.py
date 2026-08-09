"""Store-level tests for video generation messages (task-3401.4).

Follows ``test_console_generation_store.py``'s fixture/fake-persistence
style, plus one real-DB reload test mirroring its integration round trip.
Covers: the ``[video]`` marker as the only durable reference, the
namespaced video payload in ``metadata_json`` (never the v25 sidecar,
never provenance keys), clobber guards on every persistence seam, and
image-reader isolation (ADR-044).
"""

import pytest

from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules.message import ConsoleMessageController
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata
from tldw_chatbook.Video_Generation.video_store import parse_video_marker

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
    msg = store.append_video_message(sid, video_metadata=_video_meta(), persist=True)
    assert msg.persisted_message_id is not None
    created = persistence.created_messages[0]
    payload = created["metadata_json"]
    assert '"video_generation"' in payload
    # Never provenance keys, and never the v25 sidecar kwarg.
    assert '"interrupted"' not in payload
    assert "generation_metadata" not in created
    # The persisted payload round-trips back to the original facts.
    assert VideoGenerationMetadata.from_json(payload) == _video_meta()


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

        msg = store.append_video_message(
            session.id, video_metadata=_video_meta(), persist=True
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

        # The named tombstone's facts survived the reload (the bytes' absence
        # is expected -- the VideoStore is what reports missing, not the DB).
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
