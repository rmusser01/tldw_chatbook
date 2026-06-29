"""Shared Textual workbench primitives for Chatbook destinations.

Package-level names are loaded lazily so importing
``tldw_chatbook.UI.Workbench`` does not eagerly import Textual widgets or
navigation modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = (
    "Density",
    "DestinationHeader",
    "CommandStrip",
    "ModeStrip",
    "RecoveryCallout",
    "RecoveryState",
    "StateBlock",
    "WorkbenchFocusRegistry",
    "WorkbenchAction",
    "WorkbenchActionRequested",
    "WorkbenchFrame",
    "WorkbenchHeaderState",
    "WorkbenchHelpPanel",
    "WorkbenchHelpState",
    "WorkbenchMode",
    "WorkbenchPane",
    "WorkbenchPaneState",
    "WorkbenchRouteCoverage",
    "WorkbenchState",
    "WorkbenchStatus",
    "WORKBENCH_ROUTE_OWNERS",
    "build_workbench_route_coverage",
)


_EXPORT_MODULES = {
    "Density": "tldw_chatbook.UI.Workbench.workbench_state",
    "RecoveryState": "tldw_chatbook.UI.Workbench.workbench_state",
    "WorkbenchAction": "tldw_chatbook.UI.Workbench.workbench_state",
    "WorkbenchHeaderState": "tldw_chatbook.UI.Workbench.workbench_state",
    "WorkbenchMode": "tldw_chatbook.UI.Workbench.workbench_state",
    "WorkbenchPaneState": "tldw_chatbook.UI.Workbench.workbench_state",
    "WorkbenchState": "tldw_chatbook.UI.Workbench.workbench_state",
    "WorkbenchStatus": "tldw_chatbook.UI.Workbench.workbench_state",
    "WorkbenchFocusRegistry": "tldw_chatbook.UI.Workbench.focus",
    "WorkbenchHelpPanel": "tldw_chatbook.UI.Workbench.help",
    "WorkbenchHelpState": "tldw_chatbook.UI.Workbench.help",
    "DestinationHeader": "tldw_chatbook.UI.Workbench.workbench_widgets",
    "CommandStrip": "tldw_chatbook.UI.Workbench.workbench_widgets",
    "ModeStrip": "tldw_chatbook.UI.Workbench.workbench_widgets",
    "RecoveryCallout": "tldw_chatbook.UI.Workbench.workbench_widgets",
    "StateBlock": "tldw_chatbook.UI.Workbench.workbench_widgets",
    "WorkbenchActionRequested": "tldw_chatbook.UI.Workbench.workbench_widgets",
    "WorkbenchFrame": "tldw_chatbook.UI.Workbench.workbench_widgets",
    "WorkbenchPane": "tldw_chatbook.UI.Workbench.workbench_widgets",
    "WorkbenchRouteCoverage": "tldw_chatbook.UI.Workbench.route_inventory",
    "WORKBENCH_ROUTE_OWNERS": "tldw_chatbook.UI.Workbench.route_inventory",
    "build_workbench_route_coverage": "tldw_chatbook.UI.Workbench.route_inventory",
}


def __getattr__(name: str) -> Any:
    """Load package exports on first access."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
