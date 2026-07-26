# Tools module initialization
"""
Tool execution framework for LLM function calling.

Lazy re-exports (PEP 562). The optional tool classes below resolve on
first attribute access rather than at package-import time, via module
``__getattr__`` -- the same pattern used by ``Local_Ingestion/__init__.py``
and ``tldw_api/__init__.py`` (see those modules' docstrings for the
general rationale). This matters here specifically because
``WebSearchTool`` used to be imported eagerly here, which pulls in
``Article_Extractor_Lib.py``'s module-scope playwright + trafilatura
imports (~197ms, see task-257) even though ``web_search_enabled`` defaults
to ``False`` and no user session had touched web search yet. Some of
these optional tools (file ops, note management) are wired into the
agent runtime through ``Agents/tool_catalog.py``'s gateable builtins
(see ``build_gateable_tool``), which already imports each one directly
from its own submodule, gated by a per-tool config flag -- it never
goes through this package's names. The remaining optional tool classes
(``WebSearchTool``, ``RAGSearchTool``, ``SearchNotesTool``) are made
lazy too, for consistency and because there is no reason to pay any
submodule's import cost for a tool nobody asked for.

``Tool``/``DateTimeTool``/``CalculatorTool`` stay eager: they are
lightweight (defined directly in ``tool_executor.py``, no heavy
transitive deps) and are the framework's core, always-needed surface.
A now-deleted execution-framework class used to live here too, along
with its two module-level accessor functions; that framework had no
production callers and was retired in TASK-545 P3.
"""

from typing import Any

from loguru import logger

from .tool_executor import (
    Tool,
    DateTimeTool,
    CalculatorTool,
)

__all__ = [
    "Tool",
    "DateTimeTool",
    "CalculatorTool",
    "WebSearchTool",
    "ReadFileTool",
    "ListDirectoryTool",
    "WriteFileTool",
    "RAGSearchTool",
    "CreateNoteTool",
    "SearchNotesTool",
    "UpdateNoteTool",
]

# Name -> submodule providing it. Kept as a flat mapping (rather than a
# `from .x import *`-style block per submodule) so `__getattr__` only ever
# imports the one submodule that actually owns the requested name.
_SUBMODULE_BY_NAME = {
    "WebSearchTool": "web_search_tool",
    "ReadFileTool": "file_operation_tools",
    "ListDirectoryTool": "file_operation_tools",
    "WriteFileTool": "file_operation_tools",
    "RAGSearchTool": "rag_search_tool",
    "CreateNoteTool": "note_management_tools",
    "SearchNotesTool": "note_management_tools",
    "UpdateNoteTool": "note_management_tools",
}


def __getattr__(name: str) -> Any:
    submodule_name = _SUBMODULE_BY_NAME.get(name)
    if submodule_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        submodule = importlib.import_module(f".{submodule_name}", __name__)
        value = getattr(submodule, name)
    except ImportError:
        # Preserve the previous eager-import fallback semantics (the name
        # binds to None when its optional dependency isn't installed) — but
        # keep the original traceback visible so a genuine import-time BUG
        # in the tool module can't hide behind a later NoneType error
        # (PR #672 review). Non-ImportError exceptions propagate.
        logger.debug(
            f"Optional tool {name!r} unavailable; binding to None.",
            exc_info=True,
        )
        value = None
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value
