"""TASK-32954 (spec §3.1a): the local service used to drop image_base64."""

import base64

import pytest

from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


@pytest.fixture()
def service(tmp_path):
    db = CharactersRAGDB(":memory:", client_id="test")
    svc = LocalCharacterPersonaService(db=db)
    return svc, db


def test_create_persists_image_bytes(service):
    svc, db = service
    record = svc.create_character(
        {"name": "Aria", "image_base64": base64.b64encode(PNG).decode()}
    )
    assert db.get_character_card_by_id(record["id"])["image"] == PNG


def test_update_sets_and_clears_image(service):
    svc, db = service
    record = svc.create_character({"name": "Aria"})
    svc.update_character(
        record["id"],
        {"image_base64": base64.b64encode(PNG).decode()},
        expected_version=record["version"],
    )
    card = db.get_character_card_by_id(record["id"])
    assert card["image"] == PNG
    svc.update_character(
        record["id"], {}, expected_version=card["version"], clear_image=True
    )
    assert db.get_character_card_by_id(record["id"])["image"] is None


def test_invalid_base64_is_rejected(service):
    svc, _ = service
    with pytest.raises(ValueError, match="not valid base64"):
        svc.create_character({"name": "Aria", "image_base64": "%%%not-base64%%%"})
