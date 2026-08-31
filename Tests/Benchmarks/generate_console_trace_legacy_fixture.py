"""Regenerate the pinned 200-turn v53 legacy Console trace fixture."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3

from tldw_chatbook.Chat.console_exchange_capture import (
    CaptureDetail,
    ExchangeCapture,
    capture_to_blob,
)
from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version


FIXTURE_DIR = Path(__file__).with_name("fixtures")
FIXTURE_PATH = FIXTURE_DIR / "console_trace_legacy_200_turn_v53.sqlite3"
CHECKSUM_PATH = FIXTURE_DIR / "console_trace_legacy_200_turn_v53.sha256"


def _body(turn: int, role: str) -> str:
    digest = hashlib.sha256(f"legacy-fixture:{turn}:{role}".encode()).hexdigest()
    return f"{role} turn {turn:03d} {digest}"


def generate() -> None:
    """Build, compact, and checksum the genuine historical database."""

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.unlink(missing_ok=True)
    history: list[dict[str, str]] = []
    with chachanotes_db_at_version(
        FIXTURE_PATH,
        53,
        client_id="legacy-trace-fixture-v53",
    ) as db:
        conversation_id = db.add_conversation({"title": "200-turn legacy trace"})
        assert conversation_id is not None
        for turn in range(200):
            user_text = _body(turn, "user")
            user_id = db.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "user",
                    "role": "user",
                    "content": user_text,
                }
            )
            assert user_id is not None
            history.append({"role": "user", "content": user_text})
            assistant_text = _body(turn, "assistant")
            capture = ExchangeCapture(
                run_tag=f"legacy-run-{turn:03d}",
                seq=0,
                created_at=f"2026-08-28T12:{turn // 60:02d}:{turn % 60:02d}+00:00",
                provider="openai",
                model="gpt-test",
                endpoint="https://example.test/v1",
                request={
                    "messages_payload": list(history),
                    "system_message": "Pinned legacy benchmark framing",
                    "tools": [],
                },
                response={"content": assistant_text, "tool_calls": []},
                status="complete",
                usage_json=None,
                omitted_keys=(),
                capture_detail=CaptureDetail.FULL,
            )
            assistant_id = db.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "assistant",
                    "role": "assistant",
                    "content": assistant_text,
                }
            )
            assert assistant_id is not None
            db.append_message_exchanges_local(
                assistant_id,
                [
                    {
                        "run_tag": capture.run_tag,
                        "seq": capture.seq,
                        "status": capture.status,
                        "abandoned": False,
                        "capture_detail": capture.capture_detail.value,
                        "capture_blob": capture_to_blob(capture),
                        "created_at": capture.created_at,
                    }
                ],
            )
            history.append({"role": "assistant", "content": assistant_text})
    connection = sqlite3.connect(FIXTURE_PATH)
    try:
        connection.execute("VACUUM")
    finally:
        connection.close()
    digest = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()
    CHECKSUM_PATH.write_text(f"{digest}  {FIXTURE_PATH.name}\n", encoding="ascii")


if __name__ == "__main__":
    generate()
