"""Per-server inventory of unknown keys at each closed allowlist level,
computed by set difference from captured fixtures. Empty sets are valid.
Tasks 4/6 must drive every long-tail inventory to acceptance under the
custom profile, and cloud inventories into preset allowances — the SAME
fixtures replayed through BOTH parsers (non-streaming body AND the
ordered stream events) once the widening lands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pytest

from tldw_chatbook.LLM_Calls.hosted_chat import (
    HostedChatProtocolError,
    HostedChatStream,
    HostedChatTurn,
    normalize_hosted_chat_response,
)
from tldw_chatbook.LLM_Calls.hosted_chat_streaming import SSERecord

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


def _fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
# Replay helpers — Task 4 flips these from "either outcome" scaffolding to
# acceptance assertions (every long-tail body AND stream must parse under
# the widened custom profile).
# ---------------------------------------------------------------------------


class _StrictFinishPolicy:
    """Mirror of the strict finish/reasoning policy used by hosted_chat tests."""

    reasoning_disposition = "ignored"

    def validate_finish(
        self, *, finish_reason: object, has_text: bool, has_calls: bool
    ) -> str:
        if finish_reason not in {"stop", "tool_calls", "length"}:
            raise HostedChatProtocolError("finish state is malformed")
        if (finish_reason == "tool_calls") != has_calls:
            raise HostedChatProtocolError("finish state conflicts with calls")
        if finish_reason == "stop" and not has_text:
            raise HostedChatProtocolError("finish state has no text")
        return str(finish_reason)

    def validate_reasoning_content(self, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise HostedChatProtocolError("reasoning is malformed")
        return value


_REPLAY_POLICY = _StrictFinishPolicy()


def replay_body(body: dict[str, Any]) -> HostedChatTurn:
    """Replay one captured body through the current strict parser.

    Raises HostedChatProtocolError while any unknown key remains outside
    the closed allowlists; Task 4 asserts acceptance instead.
    """
    return normalize_hosted_chat_response(body, finish_policy=_REPLAY_POLICY)


def replay_stream(events: list[str]) -> HostedChatTurn:
    """Replay the captured ordered SSE payloads through the strict stream parser."""
    records = iter([SSERecord(event=None, data=payload) for payload in events])
    stream = HostedChatStream(records, finish_policy=_REPLAY_POLICY)
    for _ in stream:
        pass
    return stream.terminal_turn


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


def test_replay_scaffolding_exercises_current_parser(server_fixture: dict[str, Any]) -> None:
    """Replay both parsers today: outcome is either a turn or a protocol error.

    Task 4 replaces the tolerate-both branch with acceptance assertions.
    """
    body_outcomes: list[str] = []
    for field in ("chat_response", "tool_call_response"):
        body = server_fixture.get(field)
        if not isinstance(body, dict):
            continue
        try:
            replay_body(body)
            body_outcomes.append("turn")
        except HostedChatProtocolError:
            body_outcomes.append("protocol-error")
    assert body_outcomes, "fixture must carry at least one replayable body"

    try:
        replay_stream(list(server_fixture.get("stream_events") or []))
        stream_outcome = "turn"
    except HostedChatProtocolError:
        stream_outcome = "protocol-error"
    assert stream_outcome in {"turn", "protocol-error"}
