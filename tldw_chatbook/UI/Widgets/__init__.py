# __init__.py
# Description: UI Widgets module
#
"""UI Widgets.

Reusable widget components for the application.

The re-exports below are resolved LAZILY (PEP 562). This package's ``__init__``
used to import ``SmartContentTree`` and ``config_search_widget`` eagerly, which
meant that importing any unrelated member -- e.g. the four MCP modes, each of
which wants only the small ``table_click_select`` mixin -- pulled 653 LOC of
widgets nobody on that path uses. Those modules are reached by the screen
pre-importer, so the cost landed in the pre-import payload
(``Tests/Performance/test_screen_preimport_payload_budget.py``).

Same trap as ``Chunking/__init__.py`` in this repo's history (finding 21102): a
package ``__init__`` that eagerly imports submodules makes every consumer pay
for every sibling. ``from tldw_chatbook.UI.Widgets import SmartContentTree``
still works exactly as before -- it is resolved on first attribute access.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from .config_search_widget import ConfigSearchResult, UIElementSearchEngine
    from .SmartContentTree import (
        ContentNodeData,
        ContentSelectionChanged,
        SmartContentTree,
    )

#: Exported name -> the submodule that defines it.
_LAZY_EXPORTS = {
    "SmartContentTree": ".SmartContentTree",
    "ContentNodeData": ".SmartContentTree",
    "ContentSelectionChanged": ".SmartContentTree",
    "ConfigSearchResult": ".config_search_widget",
    "UIElementSearchEngine": ".config_search_widget",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    """Resolve a re-exported widget on first access (PEP 562)."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value  # cache: later lookups skip __getattr__ entirely
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})
