"""Chat handoff for completed research runs (task-16481).

Port of the server's ``chat_handoff`` behavior onto the local engine: when a
run launched with a chat target completes, insert an assistant message into
the originating conversation carrying the report and a
``deep_research_completion`` metadata block. Insertion failures return None
(the existing terminal-run notification remains the fallback path) rather
than raising -- the run is already completed and must not be retroactively
failed by a delivery problem.
"""

from __future__ import annotations

from typing import Any, Mapping

from loguru import logger

__all__ = ["insert_research_completion_message"]

_METADATA_BLOCK_KEY = "deep_research_completion"


def _handoff_message_content(payload: Mapping[str, Any]) -> str:
    question = str(payload.get("question") or "")
    report = str(payload.get("report_markdown") or "").strip()
    parts = [f"Deep research completed for: {question}", ""]
    if report:
        parts.append(report)
    else:
        parts.append("(no report was produced)")
    return "\n".join(parts)


def insert_research_completion_message(db: Any, payload: Mapping[str, Any]) -> str | None:
    """Insert the completion handoff message into the target conversation.

    Args:
        db: A ChaChaNotes-style DB exposing ``add_message(msg_data)``.
        payload: The engine's completion-handoff payload (``run_id``,
            ``question``, ``chat_handoff.conversation_id``,
            ``report_markdown``, ``bundle``, ``verification_summary``).

    Returns:
        The new message id, or None when the handoff could not be delivered
        (missing target, DB failure) -- never raises.
    """
    chat_handoff = payload.get("chat_handoff") or {}
    conversation_id = chat_handoff.get("conversation_id")
    if not conversation_id:
        logger.warning("Research handoff has no conversation_id; skipping insertion")
        return None
    verification = payload.get("verification_summary") or {}
    metadata = {
        _METADATA_BLOCK_KEY: {
            "run_id": payload.get("run_id"),
            "question": payload.get("question"),
            "source_count": (payload.get("bundle") or {}).get("source_count"),
            "confidence": verification.get("confidence"),
            "gate": verification.get("gate"),
        }
    }
    try:
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "assistant",
                "content": _handoff_message_content(payload),
                "metadata_json": metadata,
            }
        )
    except Exception as exc:  # noqa: BLE001 - delivery degrades to the notification fallback
        logger.warning(
            f"Research handoff message insertion failed for conversation "
            f"{conversation_id}: {exc}"
        )
        return None
    return message_id
