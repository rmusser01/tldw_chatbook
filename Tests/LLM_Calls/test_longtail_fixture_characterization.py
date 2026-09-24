"""Per-server inventory of unknown keys at each closed allowlist level,
computed by set difference from captured fixtures. Empty sets are valid.
Task 4's flip: every long-tail inventory parses under the custom profile
(tolerant + the logprobs/stop_reason choice allowances) through the
engine's own path — BOTH parsers (non-streaming body AND the ordered
stream events). Cloud fixtures flip in Task 5, under preset allowances."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import pytest

from tldw_chatbook.LLM_Calls import hosted_provider_engine
from tldw_chatbook.LLM_Calls.hosted_chat_streaming import SSERecord
from tldw_chatbook.LLM_Calls.hosted_provider_engine import (
    HostedProviderResolution,
    HostedProviderStream,
)
from tldw_chatbook.provider_registry import DATABRICKS

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "longtail"

# The parser's closed allowlists (hosted_chat.py). The inventory is the
# set difference of real-server keys against these — never parsed from
# exception text, which does not name keys.
KNOWN_TOP = frozenset(
    {"id", "object", "created", "model", "system_fingerprint", "choices", "usage"}
)
KNOWN_CHOICE = frozenset({"index", "message", "finish_reason"})
KNOWN_MESSAGE = frozenset({"role", "content", "reasoning_content", "tool_calls"})
KNOWN_EVENT = KNOWN_TOP  # stream events close the same top-level shape
KNOWN_STREAM_CHOICE = frozenset({"index", "delta", "finish_reason", "usage"})
KNOWN_DELTA = KNOWN_MESSAGE  # deltas close the same message shape

# The pinned evidence: per server, the unknown keys each closed level
# actually carries in the captured fixtures. These literals ARE the
# committed evidence — regenerate by re-running the capture scripts and
# updating from the (temporary) enumeration printout.
EXPECTED_INVENTORIES: dict[str, dict[str, dict[str, frozenset[str]]]] = {
    "ollama": {
        "body": {"top": frozenset(), "choice": frozenset(), "message": frozenset()},
        "stream": {"event": frozenset(), "choice": frozenset(), "delta": frozenset()},
    },
    "llama-server": {
        "body": {"top": frozenset({"timings"}), "choice": frozenset(), "message": frozenset()},
        "stream": {"event": frozenset({"timings"}), "choice": frozenset(), "delta": frozenset()},
    },
}


def load_fixtures() -> Iterator[Path]:
    """Every captured fixture that exists — absent servers skip cleanly."""
    if not FIXTURE_DIR.is_dir():
        return
    yield from sorted(FIXTURE_DIR.glob("*.json"))


def _require_complete_stream(fixture: dict[str, Any]) -> None:
    """Loader-side completeness guard mirroring the capture scripts' guard.

    An empty or unterminated stream_events list is a degraded capture, not
    evidence: `stream_inventory([])` is all-empty sets, which would silently
    match a clean-pinned server (ollama) and stay green while the stream
    evidence rots. Fail loudly instead.
    """
    events = fixture.get("stream_events")
    if not isinstance(events, list) or not events:
        raise ValueError(
            f"fixture {fixture.get('server', '?')}: stream_events is empty -- degraded capture"
        )
    if events[-1] != "[DONE]":
        raise ValueError(
            f"fixture {fixture.get('server', '?')}: stream_events does not terminate in [DONE]"
            " -- degraded capture"
        )


def _fixture(path: Path) -> dict[str, Any]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    _require_complete_stream(fixture)
    return fixture


def body_inventory(fixture: dict[str, Any]) -> dict[str, frozenset[str]]:
    """Unknown keys at top/choice/message level across both bodies."""
    top: set[str] = set()
    choice_keys: set[str] = set()
    message_keys: set[str] = set()
    for field in ("chat_response", "tool_call_response"):
        body = fixture.get(field)
        if not isinstance(body, dict):
            continue
        top |= set(body) - KNOWN_TOP
        for choice in body.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            choice_keys |= set(choice) - KNOWN_CHOICE
            message = choice.get("message")
            if isinstance(message, dict):
                message_keys |= set(message) - KNOWN_MESSAGE
    return {"top": frozenset(top), "choice": frozenset(choice_keys), "message": frozenset(message_keys)}


def stream_inventory(fixture: dict[str, Any]) -> dict[str, frozenset[str]]:
    """Unknown keys at event/choice/delta level across ordered stream events."""
    event_keys: set[str] = set()
    choice_keys: set[str] = set()
    delta_keys: set[str] = set()
    for payload in fixture.get("stream_events") or []:
        if payload == "[DONE]":
            continue
        event = json.loads(payload)
        if not isinstance(event, dict):
            continue
        event_keys |= set(event) - KNOWN_EVENT
        for choice in event.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            choice_keys |= set(choice) - KNOWN_STREAM_CHOICE
            delta = choice.get("delta")
            if isinstance(delta, dict):
                delta_keys |= set(delta) - KNOWN_DELTA
    return {
        "event": frozenset(event_keys),
        "choice": frozenset(choice_keys),
        "delta": frozenset(delta_keys),
    }


# ---------------------------------------------------------------------------
# Task 4 flip: every captured long-tail body AND stream must parse clean
# under the custom profile (tolerant + the logprobs/stop_reason choice
# allowances) through the ENGINE's own path — factory transport
# construction, HostedPresetFinishPolicy, and both response wrappers.
# ---------------------------------------------------------------------------

# The synthetic custom profile: exactly the record shape Task 6's real
# CUSTOM_HOSTED will ship (tolerant + the two fixture-known non-null choice
# allowances). The future record's shape is pinned via this synthetic only.
_CUSTOM_PROFILE = replace(
    DATABRICKS,
    tolerant_response_extras=True,
    choice_allowances=frozenset({"logprobs", "stop_reason"}),
)


def _custom_profile_resolution(streaming: bool) -> HostedProviderResolution:
    return HostedProviderResolution(
        provider="databricks",  # the synthetic keeps Databricks' key
        model="longtail",
        api_key="secret",
        base_url="http://127.0.0.1/v1",
        timeout=10.0,
        retries=0,
        retry_delay=0.0,
        streaming=streaming,
    )


def replay_under_custom_profile(
    monkeypatch: pytest.MonkeyPatch,
    *,
    body: dict[str, Any] | None = None,
    stream_events: list[str] | None = None,
) -> Any:
    """Replay one captured envelope through the engine under the custom profile.

    Monkeypatches the resolver/transport the handler-test way, so the replay
    exercises the engine's own factory transport construction and response
    wrappers — not a hand-built HostedChatStream.
    """
    streaming = stream_events is not None
    monkeypatch.setattr(
        hosted_provider_engine,
        "resolve_hosted_request",
        lambda _record, **_kwargs: _custom_profile_resolution(streaming),
    )
    if streaming:
        records = iter(
            [SSERecord(event=None, data=payload) for payload in stream_events]
        )
        monkeypatch.setattr(
            hosted_provider_engine, "owned_json_post", lambda **_kwargs: records
        )
    else:
        monkeypatch.setattr(
            hosted_provider_engine, "owned_json_post", lambda **_kwargs: body
        )
    handler = hosted_provider_engine.build_hosted_chat_handler(_CUSTOM_PROFILE)
    return handler(
        input_data=[{"role": "user", "content": "Say ok."}],
        api_key="secret",
        streaming=streaming,
    )


@pytest.fixture(params=[p.stem for p in load_fixtures()], ids=lambda s: s)
def server_fixture(request: pytest.FixtureRequest) -> dict[str, Any]:
    path = FIXTURE_DIR / f"{request.param}.json"
    return _fixture(path)


def test_body_inventory_matches_pinned_evidence(server_fixture: dict[str, Any]) -> None:
    server = server_fixture["server"]
    computed = body_inventory(server_fixture)
    assert computed == EXPECTED_INVENTORIES[server]["body"]


def test_stream_inventory_matches_pinned_evidence(server_fixture: dict[str, Any]) -> None:
    server = server_fixture["server"]
    computed = stream_inventory(server_fixture)
    assert computed == EXPECTED_INVENTORIES[server]["stream"]


def test_fixture_loader_rejects_degraded_stream_evidence(tmp_path: Path) -> None:
    """A streamless/unterminated fixture must fail loudly, never load.

    Proves the loader-side completeness guard against a synthetic degraded
    fixture (the capture scripts carry the same guard at write time).
    """
    base: dict[str, Any] = {
        "server": "synthetic",
        "base_url": "http://example",
        "chat_response": {},
        "tool_call_response": {},
        "models_response": None,
        "captured_at": "1970-01-01T00:00:00+00:00",
        "capture_cmd": "curl <key>",
    }
    empty = tmp_path / "empty_stream.json"
    empty.write_text(json.dumps({**base, "stream_events": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        _fixture(empty)

    unterminated = tmp_path / "unterminated_stream.json"
    unterminated.write_text(json.dumps({**base, "stream_events": ["{}"]}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"\[DONE\]"):
        _fixture(unterminated)


def test_longtail_bodies_parse_under_the_custom_profile(
    server_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every captured body (plain + tool) parses through the engine under
    the custom profile: tolerant extras dropped (llama-server ``timings``,
    ollama tool-call ``index``), allowlisted choice extras dropped, and the
    tool-call turns keep their calls with extras stripped."""
    for field in ("chat_response", "tool_call_response"):
        body = server_fixture.get(field)
        if not isinstance(body, dict):
            continue
        result = replay_under_custom_profile(monkeypatch, body=body)
        expected_reason = body["choices"][0]["finish_reason"]
        assert result["choices"][0]["finish_reason"] == expected_reason
        assert result.terminal_turn.finish_reason == expected_reason
        message = result["choices"][0]["message"]
        if expected_reason == "tool_calls":
            # Controller ruling (a): ollama's call objects carry an extra
            # ``index``; the normalized calls keep id/type/function only.
            assert message["tool_calls"]
            for call in message["tool_calls"]:
                assert set(call) == {"id", "type", "function"}


def test_longtail_streams_parse_under_the_custom_profile(
    server_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every captured ordered stream replays to a clean terminal turn.

    Both long-tail servers terminate without a usage frame (controller
    ruling b): the tolerant profile turns that into a usage-None turn, not
    the "terminated before required metadata" failure. llama-server's
    terminal-event ``timings`` rides the tolerant event-level drop.
    """
    stream = replay_under_custom_profile(
        monkeypatch, stream_events=list(server_fixture["stream_events"])
    )
    assert isinstance(stream, HostedProviderStream)
    frames = list(stream)
    assert frames, "every captured event must replay as a visible frame"
    terminal = stream.terminal_turn
    assert terminal.finish_reason == "stop"
    assert terminal.text, "the captured plain-turn streams carry content"
    assert terminal.usage is None
