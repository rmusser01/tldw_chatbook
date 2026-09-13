"""TASK-22226: message-create readbacks must not re-hydrate the image BLOB.

``ChatPersistenceService.create_message`` re-reads the just-written row
(feedback + citation paths) only to learn its DB-normalized ``version``;
before TASK-22226 each readback went through ``get_message_by_id`` and
copied the full ``image_data`` BLOB out of SQLite — megabytes per
image-message persist.

The probe here counts actual BLOB bytes materialized through the live
connection's row factory during ``create_message``, so it reds on ANY
regression that routes a create-path readback back through a
BLOB-hydrating select — regardless of which method name does it.
"""

import sqlite3

import pytest

from Tests.Chat.test_citation_trace_repository import (
    _repository as citation_repository,
    _sealed_write,
)
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

ONE_MB_IMAGE = b"\x89PNG-fake-payload" + b"\x00" * (1024 * 1024)


@pytest.fixture
def client_id():
    return "test_create_readbacks_client"


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_create_readbacks.sqlite"


@pytest.fixture
def db_instance(db_path, client_id):
    db = CharactersRAGDB(db_path, client_id)
    yield db
    db.close_connection()


class _ImageBlobReadCounter:
    """Row factory that counts materialized ``image_data`` BLOB bytes.

    Installed on the live thread connection, it sees every row any SELECT
    fetches while active; a row whose cursor description contains an
    ``image_data`` column with a bytes value counts its length. Rows are
    still returned as ``sqlite3.Row``, so counted reads behave identically
    to uncounted ones.
    """

    def __init__(self) -> None:
        self.bytes_read = 0
        self.rows_with_blob = 0

    def __call__(self, cursor, row):
        for column, value in zip(cursor.description, row):
            if column[0] == "image_data" and isinstance(value, (bytes, bytearray)):
                self.bytes_read += len(value)
                self.rows_with_blob += 1
        return sqlite3.Row(cursor, row)


class _CountingConnection:
    """Context manager installing/removing the counter on the live connection."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._connection = db.get_connection()
        self.counter = _ImageBlobReadCounter()

    def __enter__(self) -> _ImageBlobReadCounter:
        self._connection.row_factory = self.counter
        return self.counter

    def __exit__(self, *exc_info) -> None:
        self._connection.row_factory = sqlite3.Row


@pytest.mark.integration
class TestCreateReadbacksDoNotHydrateImageBlob:
    def test_legacy_feedback_create_hydrates_no_image_blob(
        self, db_instance: CharactersRAGDB
    ):
        """The legacy-image + feedback create path reads back zero BLOB bytes."""
        conversation_id = db_instance.add_conversation(
            {"title": "Legacy feedback image", "character_id": None}
        )
        service = ChatPersistenceService(db_instance)

        with _CountingConnection(db_instance) as counter:
            message_id = service.create_message(
                conversation_id=conversation_id,
                sender="user",
                content="Look at this image.",
                image_data=ONE_MB_IMAGE,
                image_mime_type="image/png",
                feedback="1;",
            )

        assert counter.bytes_read == 0, (
            f"create-path readbacks hydrated {counter.bytes_read} BLOB bytes "
            f"across {counter.rows_with_blob} rows"
        )

        # Consumer contract: the readback existed to feed the feedback
        # update's optimistic lock -- feedback must land and version must
        # reflect insert (v1) + feedback update (v2).
        message = db_instance.get_message_by_id(message_id)
        assert message["feedback"] == "1;"
        assert message["version"] == 2
        assert message["image_data"] == ONE_MB_IMAGE
        assert message["image_mime_type"] == "image/png"

    def test_citation_feedback_create_hydrates_no_image_blob(
        self, db_instance: CharactersRAGDB
    ):
        """The citation + feedback create path reads back zero BLOB bytes."""
        conversation_id = db_instance.add_conversation(
            {"title": "Citation feedback image", "character_id": None}
        )
        service = ChatPersistenceService(
            db_instance,
            citation_repository=citation_repository(db_instance),
        )

        with _CountingConnection(db_instance) as counter:
            message_id = service.create_message(
                conversation_id=conversation_id,
                sender="assistant",
                content="Answer [S1].",
                message_id="citation-image-message",
                feedback="1;grounded",
                attachments=[
                    {
                        "position": 0,
                        "data": ONE_MB_IMAGE,
                        "mime_type": "image/png",
                        "display_name": "answer.png",
                    }
                ],
                citation_write=_sealed_write(),
            )

        assert counter.bytes_read == 0, (
            f"create-path readbacks hydrated {counter.bytes_read} BLOB bytes "
            f"across {counter.rows_with_blob} rows"
        )

        # Consumer contracts of the two citation-path readbacks:
        # the feedback update's optimistic lock (version 1 -> 2) and the
        # citation owner row pinned to the POST-feedback revision. The
        # repository independently re-validates message_revision against
        # the DB inside write_prepared, so a wrong version here could not
        # have committed at all.
        message = db_instance.get_message_by_id(message_id)
        assert message["feedback"] == "1;grounded"
        assert message["version"] == 2
        assert message["image_data"] == ONE_MB_IMAGE
        owner_revision = (
            db_instance.get_connection()
            .execute(
                "SELECT message_revision FROM rag_message_trace_owners"
                " WHERE message_id = ?",
                (message_id,),
            )
            .fetchone()[0]
        )
        assert owner_revision == 2

    def test_citation_create_without_feedback_hydrates_no_image_blob(
        self, db_instance: CharactersRAGDB
    ):
        """The citation-only create path (single readback site) is BLOB-free."""
        conversation_id = db_instance.add_conversation(
            {"title": "Citation image", "character_id": None}
        )
        service = ChatPersistenceService(
            db_instance,
            citation_repository=citation_repository(db_instance),
        )

        with _CountingConnection(db_instance) as counter:
            message_id = service.create_message(
                conversation_id=conversation_id,
                sender="assistant",
                content="Answer [S1].",
                message_id="citation-image-no-feedback",
                attachments=[
                    {
                        "position": 0,
                        "data": ONE_MB_IMAGE,
                        "mime_type": "image/png",
                        "display_name": "answer.png",
                    }
                ],
                citation_write=_sealed_write(),
            )

        assert counter.bytes_read == 0
        message = db_instance.get_message_by_id(message_id)
        assert message["version"] == 1
        assert message["image_data"] == ONE_MB_IMAGE

    def test_counter_probe_can_go_red(self, db_instance: CharactersRAGDB):
        """The probe itself discriminates: a BLOB-hydrating read is counted."""
        conversation_id = db_instance.add_conversation(
            {"title": "Probe self-check", "character_id": None}
        )
        service = ChatPersistenceService(db_instance)
        message_id = service.create_message(
            conversation_id=conversation_id,
            sender="user",
            content="img",
            image_data=ONE_MB_IMAGE,
            image_mime_type="image/png",
        )
        with _CountingConnection(db_instance) as counter:
            db_instance.get_message_by_id(message_id)
        assert counter.bytes_read == len(ONE_MB_IMAGE)
        assert counter.rows_with_blob == 1


@pytest.mark.integration
class TestGetMessageByIdWithoutBlobContract:
    def test_matches_full_reader_except_blob(self, db_instance: CharactersRAGDB):
        conversation_id = db_instance.add_conversation(
            {"title": "Narrow reader", "character_id": None}
        )
        service = ChatPersistenceService(db_instance)
        message_id = service.create_message(
            conversation_id=conversation_id,
            sender="user",
            content="img",
            image_data=ONE_MB_IMAGE,
            image_mime_type="image/png",
            feedback="1;",
        )

        full = db_instance.get_message_by_id(message_id)
        narrow = db_instance.get_message_by_id_without_blob(message_id)

        assert set(narrow) == (set(full) - {"image_data"}) | {"has_image"}
        assert narrow["has_image"] == 1
        for key in set(full) - {"image_data"}:
            assert narrow[key] == full[key], key

    def test_has_image_zero_for_text_only_message(
        self, db_instance: CharactersRAGDB
    ):
        conversation_id = db_instance.add_conversation(
            {"title": "Narrow reader text", "character_id": None}
        )
        message_id = db_instance.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "text only",
            }
        )
        narrow = db_instance.get_message_by_id_without_blob(message_id)
        assert narrow["has_image"] == 0
        assert "image_data" not in narrow

    def test_none_for_missing_and_deleted_rows(self, db_instance: CharactersRAGDB):
        assert db_instance.get_message_by_id_without_blob("missing-id") is None

        conversation_id = db_instance.add_conversation(
            {"title": "Narrow reader deleted", "character_id": None}
        )
        message_id = db_instance.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "to delete",
            }
        )
        row = db_instance.get_message_by_id_without_blob(message_id)
        db_instance.soft_delete_message(message_id, expected_version=row["version"])
        assert db_instance.get_message_by_id_without_blob(message_id) is None
