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

from tldw_chatbook.Chat.console_chat_models import GenerationVariantMeta
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


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
