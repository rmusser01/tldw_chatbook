"""Actual argv Stop handler and native rules share one durable follow-up."""

import json

import pytest

from Tests.Agents.test_hooks_v2_execution import command
from Tests.Chat.test_response_rules_runtime import activate, request
from Tests.Chat.test_response_rules_runtime import native as _native_fixture

native = _native_fixture
pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]


@pytest.mark.asyncio
async def test_real_external_stop_and_native_violation_commit_one_followup(
    native, tmp_path
):
    runtime, rules, chats, session, gateway, _controller = native
    activate(native)
    marker = tmp_path / "stop-invocations.txt"
    proposal = {
        "version": 2,
        "decision": "pass",
        "continuation": {
            "message": "Keep the original request and clarify the final answer."
        },
    }
    settled = {"version": 2, "decision": "pass"}
    code = (
        "from pathlib import Path; import json; "
        f"p=Path({str(marker)!r}); previous=p.read_text() if p.exists() else ''; "
        "p.write_text(previous+'stop\\n'); "
        f"print(json.dumps({proposal!r} if not previous else {settled!r}))"
    )
    handler = command(code, name="Stop", effects=["continuation"])
    runtime.ensure_hooks_v2(session.id, (handler,), lambda *_: True)
    gateway.reply = "Missing proof"

    def next_reply():
        if len(gateway.payloads) > 1:
            gateway.reply = "Evidence: no new work was performed."

    gateway.on_stream = next_reply
    turn = runtime.accept_turn(request(native))
    result = await runtime.wait_for_turn(turn)
    assert result.accepted
    assert marker.read_text().splitlines() == ["stop", "stop"]
    assert len(gateway.payloads) == 2
    assert rules.state(session.id).assessment.outcome == "pass"
    assert "Keep the original request" in str(gateway.payloads[1])
    assert "Include the evidence" in str(gateway.payloads[1])
    with chats.persistence.db.transaction() as cursor:
        rows = cursor.execute(
            "SELECT * FROM console_machine_followup_receipts"
        ).fetchall()
        assert len(rows) == 1
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM console_hook_continuation_receipts"
            ).fetchone()[0]
            == 1
        )
        columns = [
            entry[1]
            for entry in cursor.execute(
                "PRAGMA table_info(console_machine_followup_receipts)"
            )
        ]
        row = dict(zip(columns, rows[0], strict=True))
        assert json.loads(row["contributors_json"]) == ["native", "hook"]
        assert row["native_turns"] == row["admitted_turns"] == 1
    assert rules.helpers.unsettled_count == 0
