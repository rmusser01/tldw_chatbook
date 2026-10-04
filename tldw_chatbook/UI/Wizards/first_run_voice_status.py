"""What the first-run Voice step says about a service and about a failed test.

TASK-34100.8 (voice-speech-03 / voice-speech-06). PocketTTS was preselected
although it is a separate server that usually isn't running, and every failed
sample read "Not tested yet — the sample failed." The step now shows, on a
line under the Service radio, whether the chosen service will work (a single
sub-second TCP connect run in a worker, or the OpenAI key's presence), and a
failed test starts "Test failed —" and names the cause.
"""

from __future__ import annotations

import socket
from urllib.parse import urlsplit

from tldw_chatbook.TTS.pocket_tts_native import is_pocket_tts_native_url
from tldw_chatbook.UI.Wizards import first_run_voice_prefill as prefill
from tldw_chatbook.UI.Wizards import first_run_voice_step_state as vs

#: One connect, well under a second (solution review: "a single connect with
#: a timeout under 1 s, run in a worker, never blocking the step").
PROBE_TIMEOUT_SECONDS = 0.5

NO_VOICE_COPY = "Nothing is saved. Set up a voice any time in Settings ▸ Speech & TTS."
DEFAULT_STATUS_COPY = "Optional — press Test and Hear to play a short sample."
PLAYED_COPY = (
    "Played the sample — sounds right? Continue with Next, or press Test and "
    "Hear to replay."
)
PLAYBACK_FAILED_COPY = (
    "The service answered, but this computer couldn't play the audio. Check "
    "your sound output, then replay."
)
TESTING_COPY = "Testing voice…"
CANCELLED_COPY = "Test cancelled. Press Test and Hear to retry."
KEY_NEEDED_COPY = (
    "OpenAI voice needs an OpenAI API key. Paste one below, pick another "
    'service, or choose "No voice for now".'
)
DEFAULT_HELP_COPY = "Turn on Speak replies in Console to hear answers automatically."
#: The auth option names the key; this line says where it comes from (the
#: long label was cut to "…from the Provider st…" even at 160 columns).
AUTH_HELP_COPY = (
    "API key uses your OpenAI key: the one from the Provider step or pasted "
    "here, the one saved in Settings, or OPENAI_API_KEY."
)
LEAVE_TITLE = "Leave setup?"
LEAVE_MESSAGE = (
    "Settings ▸ Speech & TTS opens so you can add the OpenAI key. Your progress "
    "is saved. Setup will pick up at Voice next time."
)


def probe_endpoint_reachable(url: str) -> bool:
    """Return whether anything accepts a TCP connection at ``url``'s host.

    Blocking by design (one ``create_connection`` with a sub-second timeout):
    call it from a worker thread.

    Args:
        url: A speech endpoint URL.

    Returns:
        True when the connect succeeded.
    """
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        return False
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=PROBE_TIMEOUT_SECONDS):
            return True
    except OSError:
        return False


def service_status_copy(
    preset: str,
    *,
    endpoint: str,
    reachable: bool | None = None,
    key_found: bool | None = None,
) -> str:
    """The line under the Service radio: will this service work?

    Args:
        preset: The selected Service radio.
        endpoint: The endpoint the controls hold.
        reachable: The probe's answer, or None while it runs.
        key_found: Whether an OpenAI key is available (OpenAI only).

    Returns:
        One short sentence.
    """
    if preset == vs.VOICE_PRESET_NONE:
        return NO_VOICE_COPY
    if preset == vs.VOICE_PRESET_OMNIVOICE:
        return "OmniVoice — runs on this computer after a one-time 1.1 GB download."
    if preset == vs.VOICE_PRESET_OFFICIAL_OPENAI:
        if key_found:
            return "OpenAI — uses your OpenAI key (key found)."
        return "OpenAI — no OpenAI API key found yet."
    host = vs.endpoint_host(endpoint)
    if preset == vs.VOICE_PRESET_POCKET_TTS:
        if reachable is None:
            return f"PocketTTS — checking {host}…"
        if reachable:
            # One TCP connect proves only that something listens there (live,
            # 127.0.0.1:8000 was a tldw_server), so never claim "running".
            return (
                f"PocketTTS — a server is listening at {host}. Test and Hear "
                "checks that it is PocketTTS."
            )
        return (
            f"PocketTTS — not running at {host}. It is a separate local server: "
            "start it with pocket-tts serve, or pick another service."
        )
    if reachable is False:
        return f"Custom — nothing answers at {host}. Check Endpoint under Advanced."
    return (
        f"Custom — {host or 'set Endpoint under Advanced'}: any OpenAI-compatible "
        "speech endpoint, or a PocketTTS /tts address."
    )


def no_voice_copy(saved: prefill.SavedVoice | None) -> str:
    """The line under the radio while "No voice for now" is chosen.

    Review round 1 (F6 / G8-V1-F2): with a voice already saved -- on a re-run,
    or earlier in this run before going Back -- "No voice for now" keeps it,
    and the line used to say "Nothing is saved" anyway.

    Args:
        saved: The voice saved now (None when nothing is).
    """
    if saved is None:
        return NO_VOICE_COPY
    if saved.other_provider or saved.legacy:
        return prefill.current_voice_copy(saved)
    return (
        f"Keeps your current voice ({prefill.voice_label(saved)}); Next changes "
        "nothing. Replies are read aloud only while Speak replies is on in Console."
    )


def default_help_copy(
    preset: str,
    *,
    locked: bool,
    ticked: bool,
    reply_voice: str = "",
    replaces: str = "",
) -> str:
    """The help line under "Use this voice when Chatbook reads replies aloud".

    Review round 1 (F1 / G8-V1-F1): it says which voice replies will use.

    Args:
        preset: The selected Service radio.
        locked: The OpenAI slot reads replies, so the box is forced on.
        ticked: The box's value.
        reply_voice: The provider reading replies when the box is free.
        replaces: The saved voice this pick replaces, if it differs.
    """
    if preset not in prefill.OPENAI_SLOT_PRESETS:
        return DEFAULT_HELP_COPY
    if locked:
        lead = (
            f"This becomes the voice replies use — it replaces {replaces}."
            if replaces
            else "Replies will use this voice — no other voice is set up."
        )
    elif ticked:
        lead = f"Replies will use this voice instead of {reply_voice}."
    else:
        lead = f"Saved for later; replies keep using {reply_voice}."
    return f"{lead} {DEFAULT_HELP_COPY}"


def voice_test_failure_copy(
    error: BaseException, *, preset: str, endpoint: str = ""
) -> str:
    """'Test failed — <cause>' for a failed sample.

    Args:
        error: What ``run_voice_sample`` raised.
        preset: The selected Service radio (PocketTTS gets its own advice).
        endpoint: The tested endpoint. A pocket-tts ``/tts`` address on
            another port is "Custom", but still gets the PocketTTS advice.

    Returns:
        The status line.
    """
    if not isinstance(error, vs.VoiceSampleError):
        # Never echo an unclassified error: it may carry server text.
        return (
            "Test failed — the sample could not be sent. Check the service, then retry."
        )
    host = error.host
    code = f" (HTTP {error.status_code})" if error.status_code else ""
    if error.kind == "not_running":
        if preset == vs.VOICE_PRESET_POCKET_TTS or is_pocket_tts_native_url(endpoint):
            return (
                f"Test failed — PocketTTS isn't running at {host}. It is a "
                "separate local server: start it with pocket-tts serve, or pick "
                "another service."
            )
        return (
            f"Test failed — the speech server isn't running at {host}. Start it, "
            "or check Endpoint under Advanced."
        )
    copy = {
        "unreachable": f"couldn't reach {host}. Check your connection, or Endpoint "
        "under Advanced.",
        "key_rejected": f"the speech server rejected the API key{code}. Check the "
        "key, or pick another service.",
        "no_endpoint": f"there is no speech endpoint at {host}{code}. Check Endpoint "
        "under Advanced.",
        "bad_request": f"the speech server refused the request{code}. Check Model "
        "and Voice under Advanced.",
        "timeout": "the speech server didn't answer within "
        f"{error.timeout_seconds or 20:g} seconds.",
        "not_audio": "the server answered, but not with audio. Check Endpoint and "
        "Format under Advanced.",
        "too_large": "the sample was larger than 8 MB.",
    }.get(
        error.kind,
        f"the speech server answered HTTP {error.status_code}. Retry, "
        "or check the server.",
    )
    return f"Test failed — {copy}"


__all__ = [
    "AUTH_HELP_COPY",
    "CANCELLED_COPY",
    "DEFAULT_HELP_COPY",
    "DEFAULT_STATUS_COPY",
    "KEY_NEEDED_COPY",
    "LEAVE_MESSAGE",
    "LEAVE_TITLE",
    "NO_VOICE_COPY",
    "PLAYBACK_FAILED_COPY",
    "PLAYED_COPY",
    "PROBE_TIMEOUT_SECONDS",
    "TESTING_COPY",
    "default_help_copy",
    "no_voice_copy",
    "probe_endpoint_reachable",
    "service_status_copy",
    "voice_test_failure_copy",
]
