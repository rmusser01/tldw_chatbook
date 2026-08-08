"""Playback optional-dependency probing (task-3401.9)."""

from tldw_chatbook.Media_Playback import availability


def test_av_probe_matches_environment():
    # av is installed in the dev venv; the probe must agree with a real import.
    try:
        import av  # noqa: F401

        assert availability.av_available() is True
    except ImportError:
        assert availability.av_available() is False


def test_missing_reason_only_when_av_absent(monkeypatch):
    monkeypatch.setattr(availability, "av_available", lambda: True)
    assert availability.playback_missing_reason() is None
    monkeypatch.setattr(availability, "av_available", lambda: False)
    reason = availability.playback_missing_reason()
    assert reason == availability.VIDEO_PLAYBACK_INSTALL_GUIDANCE
    assert "video_playback" in reason and "pip install" in reason


def test_guidance_names_both_packages():
    assert "av" in availability.VIDEO_PLAYBACK_INSTALL_GUIDANCE
    assert "textual-canvas" in availability.VIDEO_PLAYBACK_INSTALL_GUIDANCE


def test_textual_canvas_probe_is_boolean():
    # textual-canvas may or may not be installed; the probe just must not raise.
    assert isinstance(availability.textual_canvas_available(), bool)
