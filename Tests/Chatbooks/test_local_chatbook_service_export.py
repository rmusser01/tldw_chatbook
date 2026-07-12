import asyncio
from unittest.mock import patch, MagicMock

from tldw_chatbook.Chatbooks.local_chatbook_service import LocalChatbookService


def _service(tmp_path):
    db_paths = {"ChaChaNotes": str(tmp_path / "c.db"), "Media": str(tmp_path / "m.db"),
                "Prompts": str(tmp_path / "p.db")}
    return LocalChatbookService(db_paths)


def test_export_forwards_hooks_and_maps_cancelled(tmp_path):
    svc = _service(tmp_path)
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return False, "Export cancelled", {"cancelled": True, "missing_dependencies": [], "auto_included": []}

    cb = lambda evt: None
    cc = lambda: True
    with patch("tldw_chatbook.Chatbooks.local_chatbook_service.ChatbookCreator") as CC:
        CC.return_value.create_chatbook.side_effect = fake_create
        result = asyncio.run(svc.export_chatbook(
            {"name": "N", "content_selections": {}, "output_path": str(tmp_path / "o.zip")},
            progress_callback=cb, cancel_check=cc,
        ))
    assert captured["progress_callback"] is cb
    assert captured["cancel_check"] is cc
    assert result["cancelled"] is True
    assert result["success"] is False


def test_export_success_reports_not_cancelled(tmp_path):
    svc = _service(tmp_path)
    with patch("tldw_chatbook.Chatbooks.local_chatbook_service.ChatbookCreator") as CC:
        CC.return_value.create_chatbook.return_value = (True, "ok", {"missing_dependencies": [], "auto_included": []})
        result = asyncio.run(svc.export_chatbook(
            {"name": "N", "content_selections": {}, "output_path": str(tmp_path / "o.zip")}))
    assert result["cancelled"] is False
    assert result["success"] is True
