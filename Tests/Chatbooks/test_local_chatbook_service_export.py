import asyncio
from unittest.mock import patch

import pytest

from tldw_chatbook.Chatbooks.local_chatbook_service import LocalChatbookService


def _service(tmp_path):
    db_paths = {
        "ChaChaNotes": str(tmp_path / "c.db"),
        "Media": str(tmp_path / "m.db"),
        "Prompts": str(tmp_path / "p.db"),
    }
    return LocalChatbookService(db_paths)


def test_export_forwards_hooks_and_maps_cancelled(tmp_path):
    svc = _service(tmp_path)
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return (
            False,
            "Export cancelled",
            {"cancelled": True, "missing_dependencies": [], "auto_included": []},
        )

    def cb(evt):
        return None

    def cc():
        return True

    with patch("tldw_chatbook.Chatbooks.local_chatbook_service.ChatbookCreator") as CC:
        CC.return_value.create_chatbook.side_effect = fake_create
        result = asyncio.run(
            svc.export_chatbook(
                {
                    "name": "N",
                    "content_selections": {},
                    "output_path": str(tmp_path / "o.zip"),
                },
                progress_callback=cb,
                cancel_check=cc,
            )
        )
    assert captured["progress_callback"] is cb
    assert captured["cancel_check"] is cc
    assert result["cancelled"] is True
    assert result["success"] is False


def test_export_success_reports_not_cancelled(tmp_path):
    svc = _service(tmp_path)
    with patch("tldw_chatbook.Chatbooks.local_chatbook_service.ChatbookCreator") as CC:
        CC.return_value.create_chatbook.return_value = (
            True,
            "ok",
            {
                "missing_dependencies": [],
                "auto_included": [],
                "archive_version": "3.0",
                "canvas_included": True,
            },
        )
        result = asyncio.run(
            svc.export_chatbook(
                {
                    "name": "N",
                    "content_selections": {},
                    "output_path": str(tmp_path / "o.zip"),
                }
            )
        )
    assert result["cancelled"] is False
    assert result["success"] is True
    assert result["archive_version"] == "3.0"
    assert result["canvas_included"] is True


@pytest.mark.bootstrap_profile
def test_character_chatbook_omits_profile_local_performance(tmp_path):
    import json
    import zipfile

    from Tests.Character_Chat.test_portrait_performance import png
    from tldw_chatbook.Character_Chat.portrait_performance import (
        FRAME_ROLES,
        capture_performance_source,
        normalize_performance_frames,
    )
    from tldw_chatbook.Chatbooks.chatbook_models import ContentType
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    database = CharactersRAGDB(tmp_path / "characters.db", "radio-export")
    try:
        cid = database.add_character_card({"name": "Local radio", "image": png()})
        source, _ = capture_performance_source(
            database, character_id=cid, requested_state="idle"
        )
        frames = normalize_performance_frames(
            {role: png("blue") for role in FRAME_ROLES}
        )
        database.apply_character_portrait_performance(
            source, frames, expected_revision=0
        )
        service = LocalChatbookService({"ChaChaNotes": str(database.db_path)})
        output = tmp_path / "radio.zip"
        result = asyncio.run(
            service.export_chatbook(
                {
                    "name": "Radio",
                    "content_selections": {ContentType.CHARACTER: [str(cid)]},
                    "output_path": str(output),
                    "auto_include_dependencies": False,
                }
            )
        )
        assert result["success"], result
        with zipfile.ZipFile(output) as archive:
            assert not any("performance" in name for name in archive.namelist())
            payload = json.loads(
                archive.read(f"content/characters/character_{cid}.json")
            )
            assert payload["name"] == "Local radio"
            assert "performance" not in json.dumps(payload)
            assert all(frame.sha256 not in json.dumps(payload) for frame in frames)
        assert database.get_character_card_by_id(cid)["image"] == png()
    finally:
        database.close()
