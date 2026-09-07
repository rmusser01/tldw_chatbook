"""Decision-table tests for the pure Console auto-speak policy."""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_auto_speak import (
    AutoSpeakContext,
    AutoSpeakDisposition,
    decide_auto_speak,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_speech_preferences import ConsoleSpeechPreferences

DESTINATION = "sha256:" + "a" * 64
OTHER_DESTINATION = "sha256:" + "b" * 64
SESSION_ID = "active-session"


def _message(
    *,
    role: object = ConsoleMessageRole.ASSISTANT,
    content: object = "Ready.",
    status: object = "complete",
) -> ConsoleChatMessage:
    return ConsoleChatMessage(role=role, content=content, status=status)  # type: ignore[arg-type]


def _context(
    *,
    preferences: object | None = None,
    destination_fingerprint: object = DESTINATION,
    active_session_id: object = SESSION_ID,
    hands_free_active: object = False,
) -> AutoSpeakContext:
    if preferences is None:
        preferences = ConsoleSpeechPreferences(
            auto_speak=True,
            consent_destination=DESTINATION,
        )
    return AutoSpeakContext(
        preferences=preferences,  # type: ignore[arg-type]
        destination_fingerprint=destination_fingerprint,  # type: ignore[arg-type]
        active_session_id=active_session_id,  # type: ignore[arg-type]
        hands_free_active=hands_free_active,  # type: ignore[arg-type]
    )


def test_auto_speak_disposition_values_are_stable() -> None:
    assert {item.value for item in AutoSpeakDisposition} == {
        "speak",
        "disabled",
        "paused",
        "needs_consent",
        "handsfree_owns",
        "background",
        "ineligible",
    }


@pytest.mark.parametrize(
    (
        "preferences",
        "destination",
        "active_session",
        "hands_free",
        "status",
        "role",
        "expected",
    ),
    [
        (
            ConsoleSpeechPreferences(auto_speak=False),
            DESTINATION,
            SESSION_ID,
            False,
            "complete",
            ConsoleMessageRole.ASSISTANT,
            AutoSpeakDisposition.DISABLED,
        ),
        (
            ConsoleSpeechPreferences(
                auto_speak=True,
                paused=True,
                consent_destination=DESTINATION,
            ),
            DESTINATION,
            SESSION_ID,
            False,
            "complete",
            ConsoleMessageRole.ASSISTANT,
            AutoSpeakDisposition.PAUSED,
        ),
        (
            ConsoleSpeechPreferences(auto_speak=True),
            DESTINATION,
            SESSION_ID,
            False,
            "complete",
            ConsoleMessageRole.ASSISTANT,
            AutoSpeakDisposition.NEEDS_CONSENT,
        ),
        (
            ConsoleSpeechPreferences(
                auto_speak=True,
                consent_destination=OTHER_DESTINATION,
            ),
            DESTINATION,
            SESSION_ID,
            False,
            "complete",
            ConsoleMessageRole.ASSISTANT,
            AutoSpeakDisposition.NEEDS_CONSENT,
        ),
        (
            ConsoleSpeechPreferences(
                auto_speak=True,
                consent_destination=DESTINATION,
            ),
            DESTINATION,
            "other-session",
            False,
            "complete",
            ConsoleMessageRole.ASSISTANT,
            AutoSpeakDisposition.BACKGROUND,
        ),
        (
            ConsoleSpeechPreferences(
                auto_speak=True,
                consent_destination=DESTINATION,
            ),
            DESTINATION,
            SESSION_ID,
            True,
            "complete",
            ConsoleMessageRole.ASSISTANT,
            AutoSpeakDisposition.HANDSFREE_OWNS,
        ),
        (
            ConsoleSpeechPreferences(
                auto_speak=True,
                consent_destination=DESTINATION,
            ),
            DESTINATION,
            SESSION_ID,
            False,
            "streaming",
            ConsoleMessageRole.ASSISTANT,
            AutoSpeakDisposition.INELIGIBLE,
        ),
        (
            ConsoleSpeechPreferences(
                auto_speak=True,
                consent_destination=DESTINATION,
            ),
            DESTINATION,
            SESSION_ID,
            False,
            "complete",
            ConsoleMessageRole.ASSISTANT,
            AutoSpeakDisposition.SPEAK,
        ),
    ],
)
def test_auto_speak_decision_table(
    preferences: ConsoleSpeechPreferences,
    destination: str,
    active_session: str,
    hands_free: bool,
    status: str,
    role: ConsoleMessageRole,
    expected: AutoSpeakDisposition,
) -> None:
    context = AutoSpeakContext(
        preferences=preferences,
        destination_fingerprint=destination,
        active_session_id=active_session,
        hands_free_active=hands_free,
    )

    assert (
        decide_auto_speak(
            _message(role=role, status=status),
            session_id=SESSION_ID,
            context=context,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("role", "content", "status"),
    [
        (ConsoleMessageRole.USER, "Hello", "complete"),
        (ConsoleMessageRole.SYSTEM, "Instructions", "complete"),
        (ConsoleMessageRole.TOOL, "Result", "complete"),
        # Trusted character replies project into the Console as ASSISTANT rows.
        ("character", "Unknown role", "complete"),
        (ConsoleMessageRole.ASSISTANT, "Ready", "error"),
        (ConsoleMessageRole.ASSISTANT, "Ready", "failed"),
        (ConsoleMessageRole.ASSISTANT, "Ready", "cancelled"),
        (ConsoleMessageRole.ASSISTANT, "Ready", "stopped"),
        (ConsoleMessageRole.ASSISTANT, "Ready", "streaming"),
        (ConsoleMessageRole.ASSISTANT, "Ready", "pending"),
        (ConsoleMessageRole.ASSISTANT, "Ready", "partial"),
        (ConsoleMessageRole.ASSISTANT, "", "complete"),
        (ConsoleMessageRole.ASSISTANT, " \t\n", "complete"),
        (ConsoleMessageRole.ASSISTANT, None, "complete"),
        (ConsoleMessageRole.ASSISTANT, b"Ready", "complete"),
        (ConsoleMessageRole.ASSISTANT, 42, "complete"),
    ],
)
def test_non_reply_messages_are_ineligible(
    role: object,
    content: object,
    status: object,
) -> None:
    assert (
        decide_auto_speak(
            _message(role=role, content=content, status=status),
            session_id=SESSION_ID,
            context=_context(),
        )
        is AutoSpeakDisposition.INELIGIBLE
    )


def test_message_ineligibility_precedes_every_context_state() -> None:
    assert (
        decide_auto_speak(
            _message(role=ConsoleMessageRole.USER, content="", status="streaming"),
            session_id=SESSION_ID,
            context=_context(
                preferences=object(),
                destination_fingerprint=None,
                active_session_id=None,
                hands_free_active="yes",
            ),
        )
        is AutoSpeakDisposition.INELIGIBLE
    )


@pytest.mark.parametrize(
    ("preferences", "destination", "active_session", "hands_free", "expected"),
    [
        (
            ConsoleSpeechPreferences(auto_speak=False, paused=True),
            None,
            "other-session",
            True,
            AutoSpeakDisposition.DISABLED,
        ),
        (
            ConsoleSpeechPreferences(auto_speak=True, paused=True),
            None,
            "other-session",
            True,
            AutoSpeakDisposition.PAUSED,
        ),
        (
            ConsoleSpeechPreferences(auto_speak=True),
            None,
            "other-session",
            True,
            AutoSpeakDisposition.BACKGROUND,
        ),
        (
            ConsoleSpeechPreferences(
                auto_speak=True,
                consent_destination=DESTINATION,
            ),
            DESTINATION,
            "other-session",
            True,
            AutoSpeakDisposition.BACKGROUND,
        ),
        (
            ConsoleSpeechPreferences(
                auto_speak=True,
                consent_destination=DESTINATION,
            ),
            DESTINATION,
            SESSION_ID,
            True,
            AutoSpeakDisposition.HANDSFREE_OWNS,
        ),
    ],
)
def test_eligible_reply_decision_precedence_is_explicit(
    preferences: ConsoleSpeechPreferences,
    destination: object,
    active_session: object,
    hands_free: object,
    expected: AutoSpeakDisposition,
) -> None:
    assert (
        decide_auto_speak(
            _message(),
            session_id=SESSION_ID,
            context=_context(
                preferences=preferences,
                destination_fingerprint=destination,
                active_session_id=active_session,
                hands_free_active=hands_free,
            ),
        )
        is expected
    )


def test_malformed_preferences_fail_closed_without_raising() -> None:
    assert (
        decide_auto_speak(
            _message(),
            session_id=SESSION_ID,
            context=_context(preferences={"auto_speak": True}),
        )
        is AutoSpeakDisposition.DISABLED
    )

    forged = ConsoleSpeechPreferences(
        auto_speak=True,
        consent_destination=DESTINATION,
    )
    object.__setattr__(forged, "auto_speak", 1)
    assert (
        decide_auto_speak(
            _message(),
            session_id=SESSION_ID,
            context=_context(preferences=forged),
        )
        is AutoSpeakDisposition.DISABLED
    )


@pytest.mark.parametrize(
    "destination",
    [None, "", "sha256:" + "A" * 64, "sha256:" + "a" * 63, 42],
)
def test_malformed_current_destination_requires_consent(destination: object) -> None:
    assert (
        decide_auto_speak(
            _message(),
            session_id=SESSION_ID,
            context=_context(destination_fingerprint=destination),
        )
        is AutoSpeakDisposition.NEEDS_CONSENT
    )


@pytest.mark.parametrize(
    "destination",
    [None, OTHER_DESTINATION, "", "sha256:" + "A" * 64, 42],
)
def test_background_reply_precedes_destination_consent(
    destination: object,
) -> None:
    assert (
        decide_auto_speak(
            _message(),
            session_id=SESSION_ID,
            context=_context(
                destination_fingerprint=destination,
                active_session_id="other-session",
            ),
        )
        is AutoSpeakDisposition.BACKGROUND
    )


@pytest.mark.parametrize(
    "destination",
    [None, OTHER_DESTINATION, "", "sha256:" + "A" * 64, 42],
)
def test_active_hands_free_precedes_destination_consent(
    destination: object,
) -> None:
    assert (
        decide_auto_speak(
            _message(),
            session_id=SESSION_ID,
            context=_context(
                destination_fingerprint=destination,
                hands_free_active=True,
            ),
        )
        is AutoSpeakDisposition.HANDSFREE_OWNS
    )


def test_malformed_hands_free_precedes_destination_consent_fail_closed() -> None:
    assert (
        decide_auto_speak(
            _message(),
            session_id=SESSION_ID,
            context=_context(
                destination_fingerprint=None,
                hands_free_active="unknown",
            ),
        )
        is AutoSpeakDisposition.HANDSFREE_OWNS
    )


@pytest.mark.parametrize(
    ("session_id", "active_session_id"),
    [(None, SESSION_ID), (SESSION_ID, None), ("", ""), (1, 1)],
)
def test_malformed_or_nonmatching_session_identity_is_background(
    session_id: object,
    active_session_id: object,
) -> None:
    assert (
        decide_auto_speak(
            _message(),
            session_id=session_id,  # type: ignore[arg-type]
            context=_context(active_session_id=active_session_id),
        )
        is AutoSpeakDisposition.BACKGROUND
    )


@pytest.mark.parametrize("hands_free", [1, "false", None, object()])
def test_malformed_hands_free_state_reserves_speech_ownership(
    hands_free: object,
) -> None:
    assert (
        decide_auto_speak(
            _message(),
            session_id=SESSION_ID,
            context=_context(hands_free_active=hands_free),
        )
        is AutoSpeakDisposition.HANDSFREE_OWNS
    )


def test_malformed_message_or_context_fails_closed_without_raising() -> None:
    assert (
        decide_auto_speak(
            None,  # type: ignore[arg-type]
            session_id=SESSION_ID,
            context=_context(),
        )
        is AutoSpeakDisposition.INELIGIBLE
    )
    assert (
        decide_auto_speak(
            _message(),
            session_id=SESSION_ID,
            context=None,  # type: ignore[arg-type]
        )
        is AutoSpeakDisposition.DISABLED
    )


def test_decision_is_repeatable_and_does_not_mutate_inputs() -> None:
    message = _message()
    context = _context()
    before = (message.__dict__.copy(), context, context.preferences)

    first = decide_auto_speak(message, session_id=SESSION_ID, context=context)
    second = decide_auto_speak(message, session_id=SESSION_ID, context=context)

    assert first is second is AutoSpeakDisposition.SPEAK
    assert (message.__dict__, context, context.preferences) == before
