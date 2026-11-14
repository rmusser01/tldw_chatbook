"""Cross-process receipt publication must preserve the previous complete value."""

from pathlib import Path
from types import SimpleNamespace

from Tests.Canvas.browser.canvas_live_chatbook_child import (
    _publish_owner_receipt,
    _ScriptedCanvasGateway,
)
from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
    cached_context_window_for_display,
)


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


def test_scripted_canvas_provider_supports_offline_console_capacity(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    gateway = _ScriptedCanvasGateway()

    capacity = cached_context_window_for_display(
        SimpleNamespace(),
        gateway,
        SimpleNamespace(
            provider="unlisted-canvas-provider", model="unlisted-canvas-model"
        ),
    )

    assert capacity.tokens == 32000
    assert capacity.source == "application fallback"
    assert capacity.verified is False
    assert gateway.calls == 0
