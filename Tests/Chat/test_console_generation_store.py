"""Store-level tests for the P2a generation-variant model (Task 5).

Follows ``Tests/Chat/test_console_chat_store.py``'s fixture/fake-persistence
style: a plain in-memory ``ConsoleChatStore`` plus a **kwargs-recording fake
persistence adapter, extended here with the narrow generation-variant ops
(``append_message_attachment`` / ``keep_message_attachment`` /
``get_generation_metadata_for_messages``) so the store's probe pattern
(``getattr(persistence, "<op>", None)``) has something to find.
"""

from dataclasses import replace

import pytest

from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import GenerationVariantMeta
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

from Tests.UI.test_screen_navigation import _build_test_app


def _meta(seed):
    return GenerationVariantMeta(
        prompt="p",
        negative_prompt="",
        backend="swarmui",
        model=None,
        seed=seed,
        style=None,
        params={},
    )


class FakeGenerationPersistence:
    """Records every call; never performs a full-list attachments rewrite.

    ``create_message`` echoes back an explicitly given ``message_id`` (as
    the real DB does when given an explicit id) so the store's
    ``persisted_message_id`` stays addressable by the narrow ops without a
    separate id-translation step -- mirroring how ``_persist_new_message``
    pins generation messages to their native id.
    """

    def __init__(self):
        self.created_messages = []
        self.appended = []
        self.kept = []
        self.full_rewrites = []

    def create_conversation(self, **kwargs):
        return "conv-1"

    def create_message(self, **kwargs):
        self.created_messages.append(kwargs)
        message_id = kwargs.get("message_id") or f"msg-{len(self.created_messages)}"
        return message_id

    def update_message_content(self, **kwargs):
        # Any full-list rewrite attempt (attachments explicitly supplied)
        # is recorded here -- keep/append must NEVER trigger this.
        if kwargs.get("attachments") is not None:
            self.full_rewrites.append(kwargs)
        return True

    def append_message_attachment(
        self, message_id, *, data, mime_type, display_name="", generation_metadata=None
    ):
        self.appended.append(
            (message_id, data, mime_type, display_name, generation_metadata)
        )
        return len(self.appended)

    def keep_message_attachment(self, message_id, position):
        self.kept.append((message_id, position))

    def get_generation_metadata_for_messages(self, message_ids):
        return {}


@pytest.fixture
def fake_persistence():
    return FakeGenerationPersistence()


@pytest.fixture
def store_with_session(fake_persistence):
    store = ConsoleChatStore(persistence=fake_persistence)
    session = store.create_session(title="t")
    store.active_session_id = session.id
    return store, session.id


def test_append_generation_message_sets_metadata_and_mirror(store_with_session):
    store, sid = store_with_session
    msg = store.append_generation_message(
        sid,
        content="[image] p",
        variants=[(b"a", "image/png", _meta(1)), (b"b", "image/png", _meta(2))],
    )
    assert msg.image_data == b"a"  # position-0 mirror
    assert [m.seed for m in msg.generation_metadata] == [1, 2]
    assert [a.position for a in msg.attachments] == [0, 1]


def test_keep_swaps_in_memory_and_calls_persistence(store_with_session, fake_persistence):
    store, sid = store_with_session
    msg = store.append_generation_message(
        sid,
        content="c",
        variants=[(b"a", "image/png", _meta(1)), (b"b", "image/png", _meta(2))],
        persist=True,
    )
    store.keep_generation_variant(sid, msg.id, position=1)
    assert msg.image_data == b"b" and [m.seed for m in msg.generation_metadata] == [2, 1]
    assert fake_persistence.kept == [(msg.id, 1)]  # persistence op invoked


def test_keep_with_byteless_memory_does_not_null(store_with_session, fake_persistence):
    store, sid = store_with_session
    msg = store.append_generation_message(
        sid,
        content="c",
        variants=[(b"a", "image/png", _meta(1)), (b"b", "image/png", _meta(2))],
        persist=True,
    )
    # simulate a rehydrated-without-bytes message (bytes|None contract)
    store._set_message_attachments(
        msg, tuple(replace(a, data=None) for a in msg.attachments)
    )
    store.keep_generation_variant(sid, msg.id, position=1)
    assert fake_persistence.kept == [(msg.id, 1)]  # narrow op used; NO full-list rewrite call
    assert not getattr(fake_persistence, "full_rewrites", [])


def test_append_variant_respects_positions(store_with_session, fake_persistence):
    store, sid = store_with_session
    msg = store.append_generation_message(
        sid, content="c", variants=[(b"a", "image/png", _meta(1))], persist=True
    )
    pos = store.append_generation_variant(
        sid, msg.id, data=b"b", mime_type="image/png", meta=_meta(2)
    )
    assert pos == 1 and msg.attachments[1].data == b"b"
    assert [m.seed for m in msg.generation_metadata] == [1, 2]


def test_keep_out_of_range_position_raises(store_with_session):
    store, sid = store_with_session
    msg = store.append_generation_message(
        sid, content="c", variants=[(b"a", "image/png", _meta(1))]
    )
    with pytest.raises(ValueError):
        store.keep_generation_variant(sid, msg.id, position=1)
    with pytest.raises(ValueError):
        store.keep_generation_variant(sid, msg.id, position=0)


def test_hydrate_generation_metadata_populates_from_persisted_id(store_with_session):
    store, sid = store_with_session
    msg = store.append_generation_message(
        sid,
        content="c",
        variants=[(b"a", "image/png", _meta(1)), (b"b", "image/png", _meta(2))],
        persist=True,
    )
    # Simulate a reload where the in-memory node lost its metadata.
    msg.generation_metadata = ()
    store.hydrate_generation_metadata(
        sid,
        {
            msg.persisted_message_id: [
                {
                    "position": 0,
                    "prompt": "p",
                    "negative_prompt": "",
                    "backend": "swarmui",
                    "model": None,
                    "seed": 7,
                    "style": None,
                    "params_json": "{}",
                },
                {
                    "position": 1,
                    "prompt": "p",
                    "negative_prompt": "",
                    "backend": "swarmui",
                    "model": None,
                    "seed": 8,
                    "style": None,
                    "params_json": "{}",
                },
            ]
        },
    )
    assert [m.seed for m in store.get_message(msg.id).generation_metadata] == [7, 8]


def test_hydrate_generation_metadata_ignores_unknown_and_empty_rows(store_with_session):
    store, sid = store_with_session
    msg = store.append_generation_message(
        sid,
        content="c",
        variants=[(b"a", "image/png", _meta(1))],
        persist=True,
    )
    store.hydrate_generation_metadata(sid, {"some-other-message": [{"position": 0}]})
    assert [m.seed for m in store.get_message(msg.id).generation_metadata] == [1]


def test_generation_variant_meta_row_round_trip():
    meta = GenerationVariantMeta(
        prompt="a red dragon",
        negative_prompt="blurry",
        backend="swarmui",
        model="sdxl",
        seed=42,
        style="fantasy",
        params={"width": 512, "height": 512},
    )
    row = meta.to_row(3)
    assert row["position"] == 3
    assert row["params_json"] == '{"width": 512, "height": 512}'
    rebuilt = GenerationVariantMeta.from_row(row)
    assert rebuilt == replace(meta)


def test_generation_variant_meta_from_row_degrades_bad_params_json():
    row = {
        "prompt": "p",
        "negative_prompt": "",
        "backend": "swarmui",
        "model": None,
        "seed": None,
        "style": None,
        "params_json": "not-json",
    }
    rebuilt = GenerationVariantMeta.from_row(row)
    assert rebuilt.params == {}


def test_append_variant_rejects_non_generation_message(store_with_session):
    """Precondition guard: append_generation_variant must reject plain messages."""
    store, sid = store_with_session
    # Create a plain message (no generation_metadata)
    msg = store.append_message(
        sid, role="assistant", content="plain text"
    )

    with pytest.raises(ValueError, match="requires a generation message"):
        store.append_generation_variant(
            sid, msg.id, data=b"variant", mime_type="image/png", meta=_meta(1)
        )


def test_append_variant_position_drift_raises_error(store_with_session):
    """Position reconciliation guard: assert persistence position matches computed position."""
    store, sid = store_with_session

    # Create a generation message with persist=True
    msg = store.append_generation_message(
        sid,
        content="[image]",
        variants=[(b"a", "image/png", _meta(1))],
        persist=True,
    )

    # Inject a faulty persistence that returns wrong position
    class FaultyPersistence(FakeGenerationPersistence):
        def append_message_attachment(self, *args, **kwargs):
            # Return wrong position (99 instead of expected 1)
            self.appended.append((args[0], kwargs.get("data"), kwargs.get("mime_type"),
                                 kwargs.get("display_name", ""), kwargs.get("generation_metadata")))
            return 99  # Intentionally wrong

    store.persistence = FaultyPersistence()

    with pytest.raises(RuntimeError, match="generation variant position drift"):
        store.append_generation_variant(
            sid, msg.id, data=b"b", mime_type="image/png", meta=_meta(2)
        )


def test_persist_generation_message_rejects_null_bytes(store_with_session):
    """Byte-filter symmetry guard: reject None-data in generation attachments."""
    store, sid = store_with_session

    # Create a message with generation metadata
    msg = store.append_generation_message(
        sid,
        content="[image]",
        variants=[(b"a", "image/png", _meta(1))],
        persist=False,  # Don't persist yet
    )

    # Manually corrupt the attachment data to None (simulating a caller bug)
    from dataclasses import replace as dc_replace
    msg.attachments = (
        dc_replace(msg.attachments[0], data=None),
    )

    # Now try to persist it; should fail with ValueError
    with pytest.raises(ValueError, match="has no bytes"):
        store._persist_new_message(session_id=sid, message=msg)


@pytest.mark.integration
def test_generation_message_reload_round_trip_keeps_variant_and_hydrates_metadata(
    tmp_path,
):
    """Reload round-trip against REAL persistence (Task 9).

    Persists a 2-variant generation message, keeps position 1 (promoting
    variant B to canonical), then simulates a genuine reload -- persist ->
    DROP the store -> a fresh ``ConsoleChatStore`` fed through the real
    ``ChatScreen._console_messages_from_conversation_tree`` flatten +
    ``restore_persisted_session`` load/ingest path (mirrors
    ``Tests/integration/test_console_branching_e2e.py``'s
    ``_resume_into_fresh_store``) -- and completes the round trip with the
    store's own hydration seam: a batch ``get_generation_metadata_for_messages``
    fetch feeding ``hydrate_generation_metadata`` (per both methods'
    docstrings and the design spec's Task 9 Step 1).

    Asserts the reloaded message's position-0/canonical bytes are the KEPT
    variant's, ``generation_metadata`` order matches the DB (kept first),
    and the message stays card-eligible (non-empty ``generation_metadata``).
    """
    db = CharactersRAGDB(tmp_path / "reload_round_trip.sqlite", "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.create_session(title="Generation reload")
        store.active_session_id = session.id

        msg = store.append_generation_message(
            session.id,
            content="[image] a red dragon",
            variants=[
                (b"variant-a-bytes", "image/png", _meta(1)),
                (b"variant-b-bytes", "image/png", _meta(2)),
            ],
            persist=True,
        )
        store.keep_generation_variant(session.id, msg.id, position=1)
        # Sanity: the live in-memory node already reflects the keep.
        assert msg.image_data == b"variant-b-bytes"
        assert [m.seed for m in msg.generation_metadata] == [2, 1]

        conversation_id = session.persisted_conversation_id
        assert conversation_id is not None  # real persistence engaged

        # ---- Simulate reload: persist -> DROP the store -> fresh store ----
        conversation_service = ChatConversationService(db)
        tree = conversation_service.get_conversation_tree(
            conversation_id, depth_cap=10_000, root_limit=10_000
        )
        screen = ChatScreen(_build_test_app())
        screen.app_instance.chachanotes_db = db
        all_nodes = screen._console_messages_from_conversation_tree(tree)
        active_leaf_id = db.get_conversation_active_leaf(conversation_id)

        fresh_persistence = ChatPersistenceService(db)
        fresh_store = ConsoleChatStore(persistence=fresh_persistence)
        fresh_session = fresh_store.restore_persisted_session(
            title="Generation reload",
            workspace_id=None,
            persisted_conversation_id=conversation_id,
            all_nodes=all_nodes,
            active_leaf_persisted_id=active_leaf_id,
        )

        # The load/ingest path (``restore_persisted_session`` fed by the real
        # tree flatten) restores the tree, ids, and position-0 bytes, but --
        # like the real ``ChatScreen`` resume flow -- does not itself fetch
        # the generation-metadata sidecar; the store exposes the seam that
        # completes it (``get_generation_metadata_for_messages`` batch fetch
        # + ``hydrate_generation_metadata``), which the caller (here, this
        # test, standing in for ``ChatScreen``'s own resume path) drives
        # once, covering every restored message.
        persisted_ids = [
            m.persisted_message_id
            for m in fresh_store.messages_for_session(fresh_session.id)
            if m.persisted_message_id
        ]
        rows_by_message = fresh_persistence.get_generation_metadata_for_messages(
            persisted_ids
        )
        fresh_store.hydrate_generation_metadata(fresh_session.id, rows_by_message)

        reloaded = fresh_store.messages_for_session(fresh_session.id)
        reloaded_generation_msg = next(m for m in reloaded if m.generation_metadata)

        # Card-eligible: non-empty generation_metadata survived the reload.
        assert len(reloaded_generation_msg.generation_metadata) == 2
        # Position-0/canonical bytes are the KEPT variant's (byte B), both on
        # the scalar mirror and the attachments[0] entry.
        assert reloaded_generation_msg.image_data == b"variant-b-bytes"
        assert reloaded_generation_msg.attachments[0].data == b"variant-b-bytes"
        # generation_metadata order matches the DB (kept variant -- seed 2 --
        # first, original position-0 variant -- seed 1 -- second).
        assert [m.seed for m in reloaded_generation_msg.generation_metadata] == [2, 1]
    finally:
        db.close_connection()
