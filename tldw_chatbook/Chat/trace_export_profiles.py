"""The disclosure-profile vocabulary shared by every governed export surface.

``TraceExportProfile`` is the three-member contract two otherwise unrelated
export stacks agree on:

* the deferred trajectory family (``Chat/trajectory_export.py`` and its
  consumers ``trajectory_import``/``trajectory_screen``/
  ``Widgets/Console/trace_export_dialog.py``), which must stay OFF the Chat
  first-paint leg entirely (TASK-22213); and
* the Console exchange-export surface (``Chat/console_exchange_export.py``,
  ``Widgets/Console/console_exchange_export_dialog.py``), which sits ON that
  leg at module scope via ``console_conversation_inspector`` ->
  ``UI/Screens/chat_screen.py``.

TASK-23020: #2126 imported this enum from ``Chat/trajectory_export.py`` on
the exchange side, and that single-name import dragged the whole 1,463-LOC
exporter (plus ``Chat.trajectory``) onto the Chat first-paint window within
~24 hours of TASK-22213 shipping the deferral. This leaf exists so both
sides can share ONE enum object without the light side ever touching the
heavy module:

* Chat-leg modules import ``TraceExportProfile`` from HERE, never from
  ``Chat.trajectory_export`` and never through
  ``Widgets/Console/trace_export_dialog.py``.
* ``Chat/trajectory_export.py`` re-imports it from here (the
  ``RAG_Search/search_modes.py`` pattern), so no second copy can drift and
  the deferred family's existing import sites keep working.

Guards: ``Tests/Packaging/test_exchange_export_trajectory_deferral.py``
(names the offending file when an edge re-eagers),
``Tests/Packaging/test_rag_boot_import_closure.py`` and
``Tests/Performance/test_ui_ready_module_census.py`` (the leg-wide nets).
This module must stay import-cheap: stdlib only.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["TraceExportProfile", "TraceViewerProfile"]


class TraceViewerProfile(str, Enum):
    """Local disclosure choice over one stored semantic trace."""

    SAFE = "safe"
    FULL = "full"


class TraceExportProfile(str, Enum):
    """Privacy policy applied to a governed export bundle."""

    SAFE_SUMMARY = "safe_summary"
    REDACTED_DIAGNOSTIC = "redacted_diagnostic"
    FULL_TRACE = "full_trace"
