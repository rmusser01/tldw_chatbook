"""Live probe for the OpenAI Realtime API — event-name ground truth.

NOT a pytest test (deliberately not named `test_*` so pytest never collects
it): a standalone script run manually against the real OpenAI Realtime
endpoint to observe actual wire event names/shapes. See
`.superpowers/sdd/2026-08-04-realtime-voice-engine/task-2-brief.md` Step 1
for the original (text-turn) motivation.

Usage:
    ./.venv/bin/python Tests/LLM_Calls/openai_realtime_probe.py           # text turn
    ./.venv/bin/python Tests/LLM_Calls/openai_realtime_probe.py --audio   # audio turn

Two modes, both sending the SAME nested GA `session.update` shape
(`session.type`, `session.audio.input`/`.output`, single-modality
`output_modalities`) that `openai_session.py`'s ground-truth header
documents:

  Text mode (default): a `conversation.item.create` text turn +
  `response.create`. Cannot ever observe `input_audio_buffer.*` events or
  `conversation.item.input_audio_transcription.*` events -- there is no
  input audio for the server to buffer, VAD, or transcribe. This is what
  the module's ORIGINAL docstring claimed to send ("audio+text modalities")
  without actually doing so (M9 (d) in the V4 final review: a stale
  description of a request shape the live GA endpoint outright rejects --
  see `openai_session.py`'s header, "audio+text together" discrepancy) --
  fixed here to describe reality.

  Audio mode (`--audio`): appends a short synthetic tone (`input_audio_
  buffer.append`) and commits it manually (`input_audio_buffer.commit`)
  rather than relying on server VAD to auto-commit. This is a DELIBERATE
  probe-only simplification, not a claim about production's request shape:
  `openai_session.py` always sends `turn_detection: {"type": "server_vad"}`
  or `{"type": "semantic_vad"}` (never `null`), and that shape is pinned by
  the fake-server suite's dedicated turn-detection tests, not by this
  script. Manual commit exists here only so a probe run is deterministic
  (no dependency on a synthetic tone's energy actually tripping the
  server's VAD threshold) while still reliably producing
  `input_audio_buffer.committed`, `conversation.item.input_audio_
  transcription.completed` (with its own `usage: {"type": "duration", ...}`
  field -- distinct from `response.done`'s token usage), and a full
  `response.done` usage payload with the `input_token_details`/
  `output_token_details` audio/text split.

Reads the API key from `API_KEY_PATH` below — never print or log the key
itself, never commit the key file, never write it into any output file.
Never prints a raw event payload either (`json.dumps(event)`): only the
event `type` and a handful of specific, credential-free fields (mirroring
`_safe_error_detail` below) -- OpenAI's error payloads echo the submitted
credential (PR #1350 review, Q3), and a probe's stdout lands in terminal
scrollback and agent transcripts either way.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import struct
import sys
from pathlib import Path

import websockets

# Deliberately a module-level constant (not a CLI arg) so this exact path is
# stable across re-runs at the live gate later in the program, per the task
# brief's instruction to "read the key file path from the constant in the
# brief".
API_KEY_PATH = Path(
    "/Users/macbook-dev/Documents/GitHub/tldw_chatbook/openai-api-key.txt"
)
MODEL = "gpt-realtime"
REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={MODEL}"
SAMPLE_RATE = 24000
#: Bounds the whole probe run -- an interactive script that never returns is
#: as good as a hang to whoever ran it.
PROBE_TIMEOUT_SECONDS = 30.0


def _safe_error_detail(event: dict) -> str:
    """Summarize an error event WITHOUT printing the payload.

    OpenAI's error payloads echo the submitted credential -- its
    invalid-key message is literally `Incorrect API key provided:
    sk-proj-…`, which is why production sanitizes before logging
    (`ChatScreen._sanitize_console_realtime_failure`). A probe's stdout
    lands in terminal scrollback and agent transcripts, so dumping
    `json.dumps(event)` here published the key to both (PR #1350 review,
    Q3).

    Same shape as production: the type/code/param, plus only the leading
    clause of the message -- providers put the human summary before the
    first colon and the offending value after it.

    Args:
        event: The decoded `error` event.

    Returns:
        A single-line, credential-free summary.
    """
    err = event.get("error") or {}
    message = str(err.get("message", ""))
    lead = message.splitlines()[0].split(":", 1)[0].strip() if message else ""
    return (
        f"type={err.get('type')!r} code={err.get('code')!r} "
        f"param={err.get('param')!r} message_lead={lead!r}"
    )


def _read_api_key() -> str:
    """Read the OpenAI API key from `API_KEY_PATH`.

    Returns:
        The API key string, stripped of surrounding whitespace.

    Raises:
        SystemExit: If the key file is missing or empty.
    """
    if not API_KEY_PATH.exists():
        sys.exit(f"API key file not found: {API_KEY_PATH}")
    key = API_KEY_PATH.read_text(encoding="utf-8").strip()
    if not key:
        sys.exit(f"API key file is empty: {API_KEY_PATH}")
    return key


def _make_tone_then_silence() -> bytes:
    """Build ~1.2s of a 440Hz tone followed by ~0.6s of near-silence, s16le
    mono PCM at `SAMPLE_RATE`.

    The trailing silence is what lets a real (non-null) `turn_detection`
    config's VAD observe an energy drop within the buffered audio -- not
    used by this script's own manual-commit path, but kept so this helper
    stays reusable if a future probe variant re-enables VAD.

    Returns:
        Raw PCM16 bytes, no WAV header.
    """
    tone_samples = int(SAMPLE_RATE * 1.2)
    silence_samples = int(SAMPLE_RATE * 0.6)
    frames = bytearray()
    amplitude = 12000
    for i in range(tone_samples):
        value = int(amplitude * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE))
        frames += struct.pack("<h", value)
    frames += bytes(2 * silence_samples)
    return bytes(frames)


def _build_session_update(*, transcription_on: bool) -> dict:
    """Build the nested GA `session.update` payload both modes send.

    Args:
        transcription_on: Whether to request input transcription -- the
            audio-turn mode needs it (to observe the transcription-usage
            event); the text-turn mode has no input audio to transcribe.

    Returns:
        The `session.update` event dict.
    """
    audio_input: dict = {
        "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
        # Manual-commit probe mode: no server VAD, the client decides when
        # a turn ends. See the module docstring's "Audio mode" section for
        # why this diverges from production's own request shape.
        "turn_detection": None,
    }
    if transcription_on:
        audio_input["transcription"] = {"model": "whisper-1"}
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": "You are a terse test assistant.",
            "output_modalities": ["audio"],
            "audio": {
                "input": audio_input,
                "output": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    "voice": "marin",
                },
            },
        },
    }


async def _run_text_turn(ws) -> None:
    """Drive one text turn: `conversation.item.create` + `response.create`.

    Args:
        ws: The connected WebSocket.

    Returns:
        None.
    """
    await ws.send(json.dumps(_build_session_update(transcription_on=False)))
    sent_turn = False
    async for raw in ws:
        event = json.loads(raw)
        event_type = event.get("type")
        print(f"<< {event_type}")
        if event.get("error") or event_type == "error":
            print(f"   error detail: {_safe_error_detail(event)}")

        if event_type == "session.updated" and not sent_turn:
            sent_turn = True
            print("Sending conversation.item.create + response.create ...")
            await ws.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "Say the word 'probe' and nothing else.",
                                }
                            ],
                        },
                    }
                )
            )
            await ws.send(json.dumps({"type": "response.create"}))

        if event_type == "response.done":
            usage = (event.get("response") or {}).get("usage")
            print(f"   usage={usage!r}")
        if event_type in ("response.done", "error"):
            print("Terminal event received; closing.")
            break


#: How long to keep listening after `response.done` for a still-in-flight
#: `conversation.item.input_audio_transcription.completed` -- live-observed
#: (this module's own probe runs) to sometimes arrive AFTER `response.done`
#: rather than before it, matching the SAME raciness `openai_session.py`'s
#: ground-truth header already documents for a different ad hoc audio-turn
#: script ("raced a second spurious input_audio_buffer.speech_started that
#: interrupted the response first"). Breaking on `response.done` alone
#: would silently miss the transcription-usage event on an unlucky run.
_AUDIO_TURN_GRACE_SECONDS = 5.0


async def _run_audio_turn(ws) -> None:
    """Drive one audio turn: append a synthetic tone, commit manually, and
    observe `input_audio_buffer.*`, the input-transcription usage event,
    and `response.done`'s full (audio/text-split) usage payload.

    Args:
        ws: The connected WebSocket.

    Returns:
        None.
    """
    await ws.send(json.dumps(_build_session_update(transcription_on=True)))
    audio = _make_tone_then_silence()
    chunk_size = 4800  # 100ms at 24kHz s16le
    sent_audio = False
    response_done_seen = False
    transcription_seen = False

    while True:
        try:
            raw = await asyncio.wait_for(
                ws.recv(), timeout=_AUDIO_TURN_GRACE_SECONDS
            )
        except (TimeoutError, asyncio.TimeoutError):
            if response_done_seen:
                print("(grace period elapsed; closing)")
                return
            raise
        event = json.loads(raw)
        event_type = event.get("type")
        print(f"<< {event_type}")
        if event.get("error") or event_type == "error":
            print(f"   error detail: {_safe_error_detail(event)}")
            return

        if event_type == "session.updated" and not sent_audio:
            sent_audio = True
            print("Sending synthetic audio, then committing manually ...")
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size]
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("ascii"),
                        }
                    )
                )
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            await ws.send(json.dumps({"type": "response.create"}))

        if event_type == "conversation.item.input_audio_transcription.completed":
            # `usage` here is `{"type": "duration", "seconds": N}` -- a
            # duration, not a token count, and independent of response.
            # done's usage below. Printing the transcript itself is safe
            # (self-generated tone, no PII/credential risk), unlike a raw
            # payload dump.
            print(f"   transcript={event.get('transcript')!r}")
            print(f"   usage={event.get('usage')!r}")
            transcription_seen = True

        if event_type == "response.done":
            usage = (event.get("response") or {}).get("usage")
            print(f"   usage={usage!r}")
            response_done_seen = True

        if response_done_seen and transcription_seen:
            print("Both terminal events observed; closing.")
            return


async def _run_probe(*, audio_mode: bool) -> None:
    """Connect to the live OpenAI Realtime endpoint and run one turn.

    Args:
        audio_mode: True to run `_run_audio_turn`, False for
            `_run_text_turn`.

    Returns:
        None. Findings are printed to stdout for manual transcription into
        `openai_session.py`'s ground-truth comment block.
    """
    api_key = _read_api_key()
    headers = {"Authorization": f"Bearer {api_key}"}

    print(f"Connecting to {REALTIME_URL} ({'audio' if audio_mode else 'text'} mode) ...")
    async with websockets.connect(REALTIME_URL, additional_headers=headers) as ws:
        print("Connected. Sending session.update ...")
        if audio_mode:
            await _run_audio_turn(ws)
        else:
            await _run_text_turn(ws)


def main() -> None:
    """Entry point: run the probe and surface connection-level errors.

    Returns:
        None.
    """
    audio_mode = "--audio" in sys.argv[1:]
    try:
        asyncio.run(asyncio.wait_for(_run_probe(audio_mode=audio_mode), timeout=PROBE_TIMEOUT_SECONDS))
    except Exception as exc:  # noqa: BLE001 - probe script, want to see everything
        print(f"Probe failed: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
