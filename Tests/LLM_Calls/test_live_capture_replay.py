"""Live captures replay through the real engine under their preset's record (TASK-33640).

``Tests/fixtures/cloud_live/<key>.json`` holds a RAW capture of one engine
preset's real API, written by ``Tests/fixtures/cloud_live/capture.py``. Each
capture that exists replays here under the preset's REAL record -- strict
parser plus the record's allowances -- through the engine's own handler path
(factory transport construction and both response wrappers). A missing
capture skips; a capture that does not parse fails with the response key
names its record does not allow. That list is the evidence for amending the
record (cite the fixture; never widen by guess).

Rounds the provider refused (non-200) are recorded in the fixture but not
replayed: an error body is not a response shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from Tests.fixtures.cloud_live.capture import uncovered_keys
from tldw_chatbook.Chat.Chat_Deps import ChatProviderError
from tldw_chatbook.LLM_Calls import hosted_provider_engine
from tldw_chatbook.LLM_Calls.hosted_chat import HostedChatProtocolError
from tldw_chatbook.LLM_Calls.hosted_chat_streaming import SSERecord
from tldw_chatbook.LLM_Calls.hosted_provider_engine import (
    HostedProviderResolution,
    HostedProviderStream,
)
from tldw_chatbook.provider_registry import RECORDS_BY_KEY

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "cloud_live"
CAPTURES = sorted(path.stem for path in FIXTURE_DIR.glob("*.json"))


def _replay(
    monkeypatch: pytest.MonkeyPatch,
    record: Any,
    *,
    body: dict[str, Any] | None = None,
    stream_events: list[str] | None = None,
) -> Any:
    """Run one captured envelope through the engine's real handler path.

    Args:
        monkeypatch: Replaces resolution and transport with the capture.
        record: The preset's registry record.
        body: A captured non-streaming response body.
        stream_events: Captured SSE ``data`` payloads, ``[DONE]`` included.

    Returns:
        The handler's response, or its stream for a streamed replay.
    """
    streaming = stream_events is not None
    resolution = HostedProviderResolution(
        provider=record.key,
        model="captured-model",
        api_key="secret",
        base_url=record.default_base_url or "https://engine.invalid/v1",
        timeout=10.0,
        retries=0,
        retry_delay=0.0,
        streaming=streaming,
    )
    monkeypatch.setattr(hosted_provider_engine, "resolve_hosted_request", lambda _r, **_k: resolution)
    if streaming:
        records = iter([SSERecord(event=None, data=payload) for payload in stream_events])
        monkeypatch.setattr(hosted_provider_engine, "owned_json_post", lambda **_k: records)
    else:
        monkeypatch.setattr(hosted_provider_engine, "owned_json_post", lambda **_k: body)
    return hosted_provider_engine.build_hosted_chat_handler(record)(
        input_data=[{"role": "user", "content": "Say ok."}],
        api_key="secret",
        streaming=streaming,
    )


def _evidence(record: Any, fixture: dict[str, Any]) -> str:
    gaps = {level: keys for level, keys in uncovered_keys(record, fixture).items() if keys}
    return f"{record.key}: keys outside the record's allowances: {gaps or 'none'}"


@pytest.fixture(params=CAPTURES, ids=lambda name: name)
def capture(request: pytest.FixtureRequest) -> dict[str, Any]:
    """One captured fixture, loaded from disk."""
    return json.loads((FIXTURE_DIR / f"{request.param}.json").read_text(encoding="utf-8"))


def test_every_capture_belongs_to_an_engine_preset(capture: dict[str, Any]) -> None:
    """A capture file is named after, and records, a real engine preset."""
    record = RECORDS_BY_KEY.get(capture["server"])
    assert record is not None and record.engine_driven, capture["server"]


@pytest.mark.parametrize("round_name, field", [("plain", "chat_response"), ("tool", "tool_call_response")])
def test_captured_bodies_parse_under_their_record(
    capture: dict[str, Any], round_name: str, field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each successful non-streaming round parses into a turn with its finish reason.

    Args:
        capture: One captured fixture.
        round_name: The round's status key in the fixture.
        field: The fixture field holding the body.
        monkeypatch: Replaces resolution and transport with the capture.
    """
    body = capture.get(field)
    if capture["statuses"].get(round_name) != 200 or not isinstance(body, dict):
        pytest.skip(f"{round_name} round was not a successful response")
    record = RECORDS_BY_KEY[capture["server"]]
    try:
        result = _replay(monkeypatch, record, body=body)
    except (HostedChatProtocolError, ChatProviderError) as error:
        pytest.fail(f"{type(error).__name__}: {error} -- {_evidence(record, capture)}")
    assert result.terminal_turn.finish_reason == body["choices"][0]["finish_reason"]


def test_captured_stream_parses_under_its_record(
    capture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful, complete stream replays to a terminal turn.

    Args:
        capture: One captured fixture.
        monkeypatch: Replaces resolution and transport with the capture.
    """
    events = capture.get("stream_events") or []
    if capture["statuses"].get("stream") != 200 or not events or events[-1] != "[DONE]":
        pytest.skip("stream round was not a complete successful stream")
    record = RECORDS_BY_KEY[capture["server"]]
    stream = _replay(monkeypatch, record, stream_events=list(events))
    assert isinstance(stream, HostedProviderStream)
    try:
        list(stream)
    except (HostedChatProtocolError, ChatProviderError) as error:
        pytest.fail(f"{type(error).__name__}: {error} -- {_evidence(record, capture)}")
    assert stream.terminal_turn.finish_reason is not None
