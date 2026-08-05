"""Live probe for the OpenAI Realtime API — turn-detection ground truth.

NOT a pytest test (deliberately not named `test_*` so pytest never collects
it): a standalone script run manually against the real OpenAI Realtime
endpoint, sibling to `openai_realtime_probe.py`, to establish which
`turn_detection` shapes the GA endpoint actually accepts BEFORE any of them
are shipped as config.

Why: gate round 5 reported input transcription "picking up random words
instead of what I'm asking". The capture path was exonerated end to end
(tap-recorded 24 kHz audio transcribes perfectly through a local model), so
the leading hypothesis is the DEFAULT `server_vad` committing on non-speech
noise -- keyboard clatter (barge-in is a keypress next to a hot mic) and
room sound -- with `whisper-1` then hallucinating words out of the
fragments. The fix is turn-detection tuning, and it must be probe-first:
a rejected `session.update` is not a config option, it is an outage.

Usage:
    ./.venv/bin/python Tests/LLM_Calls/openai_realtime_turn_detection_probe.py

Reads the API key from `API_KEY_PATH` below — never print or log the key
itself, never commit the key file, never write it into any output file.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import websockets

API_KEY_PATH = Path(
    "/Users/macbook-dev/Documents/GitHub/tldw_chatbook/openai-api-key.txt"
)
MODEL = "gpt-realtime"
REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={MODEL}"
SAMPLE_RATE = 24000

#: Each candidate is sent as the `turn_detection` block of an otherwise
#: identical (and known-accepted) `session.update`, so the ONLY thing under
#: test is the block itself.
CANDIDATES: list[tuple[str, dict | None]] = [
    ("server_vad (baseline, currently shipped)", {"type": "server_vad"}),
    (
        "server_vad + tuning knobs",
        {
            "type": "server_vad",
            "threshold": 0.6,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 700,
        },
    ),
    ("semantic_vad (bare)", {"type": "semantic_vad"}),
    ("semantic_vad + eagerness=low", {"type": "semantic_vad", "eagerness": "low"}),
    ("semantic_vad + eagerness=auto", {"type": "semantic_vad", "eagerness": "auto"}),
    ("semantic_vad + eagerness=high", {"type": "semantic_vad", "eagerness": "high"}),
    (
        "semantic_vad + server_vad-only knobs (expected reject)",
        {"type": "semantic_vad", "threshold": 0.6},
    ),
]


def _read_api_key() -> str:
    if not API_KEY_PATH.exists():
        sys.exit(f"API key file not found: {API_KEY_PATH}")
    key = API_KEY_PATH.read_text(encoding="utf-8").strip()
    if not key:
        sys.exit(f"API key file is empty: {API_KEY_PATH}")
    return key


def _session_update(turn_detection: dict | None) -> dict:
    """Build the session.update this app really sends, varying only the
    turn-detection block."""
    audio_input: dict = {
        "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
        "transcription": {"model": "whisper-1"},
    }
    if turn_detection is not None:
        audio_input["turn_detection"] = turn_detection
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "output_modalities": ["audio"],
            "audio": {
                "input": audio_input,
                "output": {"format": {"type": "audio/pcm", "rate": SAMPLE_RATE}},
            },
        },
    }


async def _probe_one(ws, label: str, turn_detection: dict | None) -> None:
    """Send one candidate and print the server's verdict verbatim."""
    await ws.send(json.dumps(_session_update(turn_detection)))
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=15)
        event = json.loads(raw)
        kind = event.get("type")
        if kind == "session.updated":
            accepted = (
                event.get("session", {})
                .get("audio", {})
                .get("input", {})
                .get("turn_detection")
            )
            print(f"  ACCEPTED  {label}")
            print(f"      echoed back: {json.dumps(accepted, sort_keys=True)}")
            return
        if kind == "error":
            err = event.get("error", {})
            print(f"  REJECTED  {label}")
            print(
                f"      code={err.get('code')!r} param={err.get('param')!r} "
                f"message={err.get('message')!r}"
            )
            return


async def _run_probe() -> None:
    key = _read_api_key()
    async with websockets.connect(
        REALTIME_URL, additional_headers={"Authorization": f"Bearer {key}"}
    ) as ws:
        # Drain session.created before probing.
        first = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
        print(f"connected: {first.get('type')}")
        default_td = (
            first.get("session", {})
            .get("audio", {})
            .get("input", {})
            .get("turn_detection")
        )
        print(f"server default turn_detection: {json.dumps(default_td, sort_keys=True)}")
        print()
        for label, candidate in CANDIDATES:
            await _probe_one(ws, label, candidate)


if __name__ == "__main__":
    asyncio.run(_run_probe())
