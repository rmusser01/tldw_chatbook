"""Test-only helper for the controller's explicit two-phase close contract."""

from __future__ import annotations

from typing import Any


def close_controller_session(controller: Any, session_id: str) -> Any | None:
    """Begin and finalize an already-quiesced controller-unit session close."""

    impact = controller.lifecycle_impact(session_id=session_id)
    ticket = controller.begin_session_close(
        session_id,
        expected_revision=impact.revision,
    )
    return controller.finalize_session_close(ticket)
