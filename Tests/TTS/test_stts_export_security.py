from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import STTSEventHandler


class NotifyCaptureApp:
    def __init__(self) -> None:
        self.notifications: list[SimpleNamespace] = []

    def notify(self, message: str, *, severity: str = "information") -> None:
        self.notifications.append(SimpleNamespace(message=message, severity=severity))


@pytest.mark.asyncio
async def test_stts_export_current_audio_validates_destination_path(tmp_path):
    app = NotifyCaptureApp()
    handler = STTSEventHandler(app=app)
    handler._current_audio_file = tmp_path / "source.wav"
    handler._current_audio_file.write_bytes(b"audio")

    target_path = tmp_path / "export.wav"

    await handler.export_current_audio(target_path)

    assert target_path.read_bytes() == b"audio"
    assert app.notifications[-1].severity == "information"


@pytest.mark.asyncio
async def test_stts_export_current_audio_rejects_dangerous_destination_path(tmp_path):
    app = NotifyCaptureApp()
    handler = STTSEventHandler(app=app)
    handler._current_audio_file = tmp_path / "source.wav"
    handler._current_audio_file.write_bytes(b"audio")

    target_path = tmp_path / "bad;name.wav"

    await handler.export_current_audio(target_path)

    assert not target_path.exists()
    assert app.notifications[-1].severity == "error"
    assert "dangerous pattern" in app.notifications[-1].message
