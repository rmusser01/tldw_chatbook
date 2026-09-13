"""Shared presentation for the disclosure profiles, off the heavy export stack.

The per-profile copy, radio labels, and the every-disclosure Full-export
confirmation are shared by two dialogs:

* ``trace_export_dialog.py`` (Trace v2, deferred with the trajectory family
  -- TASK-22213), which re-exports these names for its own consumers; and
* ``console_exchange_export_dialog.py``, which is on the Chat first-paint
  leg at module scope (via ``console_conversation_inspector`` ->
  ``UI/Screens/chat_screen.py``).

TASK-23020: the exchange dialog used to import these three names FROM
``trace_export_dialog``, whose module scope imports the whole
``Chat/trajectory_export.py`` engine -- one of the edges that put 1,463 LOC
back on the first-paint window ~24 hours after TASK-22213 removed it. They
live here so the Chat leg can present the profiles without resolving any
export engine. Keep this module light: the enum leaf
(``Chat/trace_export_profiles.py``) and ``ConfirmationDialog`` only --
never ``trajectory_export``, ``trajectory``, or ``trace_export_dialog``.
Guard: ``Tests/Packaging/test_exchange_export_trajectory_deferral.py``.
"""

from __future__ import annotations

from tldw_chatbook.Chat.trace_export_profiles import TraceExportProfile
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

__all__ = [
    "TRACE_EXPORT_PROFILE_COPY",
    "TRACE_EXPORT_PROFILE_LABELS",
    "full_trace_confirmation",
]


TRACE_EXPORT_PROFILE_COPY = {
    TraceExportProfile.SAFE_SUMMARY: (
        "Safe summary — causal structure, status, and coarse timing; payload bodies omitted."
    ),
    TraceExportProfile.REDACTED_DIAGNOSTIC: (
        "Redacted diagnostic — useful debugging context with paths, identifiers, "
        "and sensitive values governed. Recommended for collaboration."
    ),
    TraceExportProfile.FULL_TRACE: (
        "Full trace — includes ordinary captured detail after an additional warning. "
        "Credentials remain forbidden."
    ),
}

TRACE_EXPORT_PROFILE_LABELS = {
    TraceExportProfile.SAFE_SUMMARY: "Safe summary",
    TraceExportProfile.REDACTED_DIAGNOSTIC: "Redacted diagnostic (recommended)",
    TraceExportProfile.FULL_TRACE: "Full trace",
}


def full_trace_confirmation(*, noun: str) -> ConfirmationDialog:
    """Build the shared every-disclosure Full export warning."""
    return ConfirmationDialog(
        title=f"Export full {noun}?",
        message=(
            "Full trace may include prompts, injected instructions, tool arguments, "
            "outputs, and local paths. Credentials remain structurally blocked."
        ),
        confirm_label=f"Export full {noun.lower()}",
        cancel_label="Go back",
    )
