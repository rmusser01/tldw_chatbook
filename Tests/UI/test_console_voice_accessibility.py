"""Accessibility policy for rapidly changing speculative voice previews."""

from __future__ import annotations

from tldw_chatbook.Widgets.Console.console_voice_preview import (
    VoiceStatusAnnouncementThrottle,
)


def test_status_announcements_are_transition_only_and_cadence_bounded():
    now = [10.0]
    announced: list[str] = []
    throttle = VoiceStatusAnnouncementThrottle(
        announced.append,
        clock=lambda: now[0],
        minimum_interval_seconds=0.75,
    )

    assert throttle.observe("listening") is True
    assert throttle.observe("listening") is False
    now[0] += 0.1
    assert throttle.observe("responding") is False
    now[0] += 0.7
    assert throttle.observe("responding") is True

    assert announced == ["Listening", "Responding"]


def test_text_revisions_do_not_enter_the_status_announcement_contract():
    announced: list[str] = []
    throttle = VoiceStatusAnnouncementThrottle(
        announced.append,
        clock=lambda: 20.0,
    )

    assert throttle.observe("speaking") is True
    # The API accepts status only: rolling user/assistant bodies never cross
    # this boundary, and repeating a state is silent regardless of revisions.
    for _revision in range(100):
        assert throttle.observe("speaking") is False

    assert announced == ["Speaking"]


def test_all_required_voice_states_have_user_readable_announcements():
    now = [0.0]
    announced: list[str] = []
    throttle = VoiceStatusAnnouncementThrottle(
        announced.append,
        clock=lambda: now[0],
        minimum_interval_seconds=0.01,
    )

    for status in (
        "listening",
        "transcribing",
        "responding",
        "speaking",
        "updating response",
        "aec warming",
        "half duplex",
        "cleanup quarantine",
    ):
        now[0] += 0.02
        assert throttle.observe(status) is True

    assert announced == [
        "Listening",
        "Transcribing",
        "Responding",
        "Speaking",
        "Updating response",
        "AEC warming",
        "Half duplex · echo cancellation unavailable",
        "Voice cleanup quarantine",
    ]
