"""Cross-process receipt publication must preserve the previous complete value."""

from pathlib import Path

from Tests.Canvas.browser.canvas_live_chatbook_child import _publish_owner_receipt


def test_owner_receipt_is_complete_during_replacement(tmp_path, monkeypatch):
    target = tmp_path / "owner.json"
    target.write_text('{"served": false}', encoding="ascii")
    observations = []

    def observed_write(path, data, *, encoding):
        with path.open("w", encoding=encoding) as handle:
            observations.append(target.read_text(encoding="ascii"))
            return handle.write(data)

    monkeypatch.setattr(Path, "write_text", observed_write)
    _publish_owner_receipt(target, {"served": True})
    assert observations == ['{"served": false}']
    assert target.read_text(encoding="ascii") == '{"served": true}'
