"""Answer the Hands-free consent in tests that turn Hands-free on by gesture.

TASK-33626.1: the header Hands-free control, Ctrl+Shift+H and the spoken
"hands free" command now show the Speak replies consent before any of the
speech pipeline starts. Tests that drive those gestures to reach the loop
answer it here, through the modal's real Enable button.
"""

from __future__ import annotations

import time

from textual.widgets import Button

CONSENT_SCREEN = "AutoSpeakConsentModal"


async def wait_for_hands_free_consent(host, pilot, *, timeout: float = 5.0):
    """Wait until the Hands-free consent is the top screen on ``host``.

    Args:
        host: The App whose screen stack receives the modal.
        pilot: The running test pilot.
        timeout: Seconds to wait for the modal to open.

    Returns:
        The open consent modal.

    Raises:
        AssertionError: The consent never opened, or a different speech
            consent opened instead.
    """
    deadline = time.monotonic() + timeout
    while type(host.screen_stack[-1]).__name__ != CONSENT_SCREEN:
        assert time.monotonic() < deadline, "the Hands-free consent never opened"
        await pilot.pause(0.02)
    modal = host.screen_stack[-1]
    assert modal.hands_free is True, "a different speech consent is open"
    return modal


async def accept_hands_free_consent(host, pilot, *, timeout: float = 5.0) -> None:
    """Wait for the Hands-free consent on ``host`` and press Enable.

    Args:
        host: The App whose screen stack receives the modal.
        pilot: The running test pilot.
        timeout: Seconds to wait for the modal to open and then close.

    Raises:
        AssertionError: The consent never opened, or did not close.
    """
    deadline = time.monotonic() + timeout
    modal = await wait_for_hands_free_consent(host, pilot, timeout=timeout)
    modal.query_one("#console-auto-speak-consent-confirm", Button).press()
    while modal in host.screen_stack:
        assert time.monotonic() < deadline, "the Hands-free consent did not close"
        await pilot.pause(0.02)
    await pilot.pause()
