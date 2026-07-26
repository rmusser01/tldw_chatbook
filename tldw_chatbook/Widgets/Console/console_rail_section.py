"""Console's rail section header.

The implementation moved to ``Widgets/destination_rail.py`` when a fourth
destination needed it. These names are retained so existing imports and
selectors keep resolving.
"""

from __future__ import annotations

from tldw_chatbook.Widgets.destination_rail import (
    RAIL_SECTION_TOGGLE_PREFIX,
    DestinationRailSectionHeader,
)

CONSOLE_RAIL_SECTION_TOGGLE_PREFIX = RAIL_SECTION_TOGGLE_PREFIX
ConsoleRailSectionHeader = DestinationRailSectionHeader

__all__ = ["CONSOLE_RAIL_SECTION_TOGGLE_PREFIX", "ConsoleRailSectionHeader"]
