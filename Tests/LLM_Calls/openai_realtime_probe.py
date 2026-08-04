"""Live probe for the OpenAI Realtime API — event-name ground truth.

NOT a pytest test (deliberately not named `test_*` so pytest never collects
it): a standalone script run manually against the real OpenAI Realtime
endpoint to observe actual wire event names before `openai_session.py` is
implemented against them. See
`.superpowers/sdd/2026-08-04-realtime-voice-engine/task-2-brief.md` Step 1.

Usage:
    ./.venv/bin/python Tests/LLM_Calls/openai_realtime_probe.py

Reads the API key from `API_KEY_PATH` below — never print or log the key
itself, never commit the key file, never write it into any output file.
"""

from __future__ import annotations

import asyncio
import json
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


async def _run_probe() -> None:
    """Connect to the live OpenAI Realtime endpoint and print observed event
    types until `response.done` (or `error`) arrives.

    Sends a `session.update` (audio+text modalities, pcm16, input
    transcription on, server VAD on), then a text `conversation.item.create`
    + `response.create`, and prints every received event's `type` field.

    Returns:
        None. Findings are printed to stdout for manual transcription into
        `openai_session.py`'s ground-truth comment block.
    """
    api_key = _read_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    print(f"Connecting to {REALTIME_URL} ...")
    async with websockets.connect(REALTIME_URL, additional_headers=headers) as ws:
        print("Connected. Sending session.update ...")
        session_update = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": "You are a terse test assistant.",
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "transcription": {"model": "whisper-1"},
                        "turn_detection": {"type": "server_vad"},
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "voice": "marin",
                    },
                },
            },
        }
        await ws.send(json.dumps(session_update))

        sent_text_turn = False
        async for raw in ws:
            event = json.loads(raw)
            event_type = event.get("type")
            print(f"<< {event_type}")
            if event.get("error") or event_type == "error":
                print(f"   error detail: {json.dumps(event, indent=2)}")

            if event_type == "session.updated" and not sent_text_turn:
                sent_text_turn = True
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

            if event_type in ("response.done", "error"):
                print("Terminal event received; closing.")
                break


def main() -> None:
    """Entry point: run the probe and surface connection-level errors.

    Returns:
        None.
    """
    try:
        asyncio.run(_run_probe())
    except Exception as exc:  # noqa: BLE001 - probe script, want to see everything
        print(f"Probe failed: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
