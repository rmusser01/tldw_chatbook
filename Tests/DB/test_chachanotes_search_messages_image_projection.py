"""task-34399 item 2: ``search_messages_by_content`` image-BLOB projection.

Both live callers (Character_Chat_Lib's ``search_messages_in_conversation``
and the MCP search tool's message preview) consume text only, yet the FTS
search used to transfer ``m.image_data`` for every hit. It now carries the
task-260 seam its sibling loaders already have: ``include_image_data=False``
returns the key as None so snippet builders skip the BLOB I/O, while the
default preserves the historical full-blob behavior.
"""

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

_IMAGE_BYTES = b"\x89PNG-fake-bytes-for-projection-test"
_NEEDLE = "xylophone-needle"


@pytest.fixture
def db(tmp_path):
    return CharactersRAGDB(tmp_path / "search-images.db", client_id="search-test")


@pytest.fixture
def conversation_id(db):
    return db.add_conversation({"character_id": 1, "title": "Projection Chat"})


def _add_image_message(db, conversation_id):
    return db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": f"about {_NEEDLE} with an image",
            "image_data": _IMAGE_BYTES,
            "image_mime_type": "image/png",
        }
    )


def test_include_image_data_false_returns_null_blob(db, conversation_id):
    _add_image_message(db, conversation_id)

    results = db.search_messages_by_content(
        _NEEDLE, conversation_id=conversation_id, include_image_data=False
    )

    assert len(results) == 1
    message = results[0]
    # Key still present (so callers can tell a message shape), BLOB skipped.
    assert "image_data" in message
    assert message["image_data"] is None
    # The mime type is always returned -- callers can still tell an image exists.
    assert message["image_mime_type"] == "image/png"
    assert message["content"] == f"about {_NEEDLE} with an image"


def test_default_call_still_returns_the_blob(db, conversation_id):
    _add_image_message(db, conversation_id)

    results = db.search_messages_by_content(_NEEDLE, conversation_id=conversation_id)

    assert len(results) == 1
    assert results[0]["image_data"] == _IMAGE_BYTES
    assert results[0]["image_mime_type"] == "image/png"


def test_projection_does_not_change_matches_or_ordering(db, conversation_id):
    first = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": f"first {_NEEDLE} hit",
        }
    )
    second = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": f"second {_NEEDLE} hit",
            "image_data": _IMAGE_BYTES,
            "image_mime_type": "image/png",
        }
    )

    projected = db.search_messages_by_content(
        _NEEDLE, conversation_id=conversation_id, include_image_data=False
    )
    full = db.search_messages_by_content(_NEEDLE, conversation_id=conversation_id)

    assert {row["id"] for row in projected} == {first, second}
    assert [row["id"] for row in projected] == [row["id"] for row in full]
