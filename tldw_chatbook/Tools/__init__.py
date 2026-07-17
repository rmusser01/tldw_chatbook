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
to ``False`` and no user session had touched web search yet. The actual
tool-registration path (``tool_executor.get_tool_executor()``) already
imports each optional tool class directly from its own submodule, gated
by per-tool config flags -- it never goes through this package's names.
The remaining optional tool classes (file ops, RAG search, note
management) are made lazy too, for consistency and because there is no
reason to pay any submodule's import cost for a tool nobody asked for.

``Tool``/``ToolExecutor``/``DateTimeTool``/``CalculatorTool``/
``get_tool_executor``/``reload_tool_executor`` stay eager: they are
lightweight (defined directly in ``tool_executor.py``, no heavy
transitive deps) and are the framework's core, always-needed surface.
"""

from typing import Any

from .tool_executor import (
    Tool,
    ToolExecutor,
    DateTimeTool,
    CalculatorTool,
    get_tool_executor,
    reload_tool_executor
)

__all__ = [
    'Tool',
    'ToolExecutor',
    'DateTimeTool',
    'CalculatorTool',
    'get_tool_executor',
    'reload_tool_executor',
    'WebSearchTool',
    'ReadFileTool',
    'ListDirectoryTool',
    'WriteFileTool',
    'RAGSearchTool',
    'CreateNoteTool',
    'SearchNotesTool',
    'UpdateNoteTool',
]

# Name -> submodule providing it. Kept as a flat mapping (rather than a
# `from .x import *`-style block per submodule) so `__getattr__` only ever
# imports the one submodule that actually owns the requested name.
_SUBMODULE_BY_NAME = {
    'WebSearchTool': 'web_search_tool',
    'ReadFileTool': 'file_operation_tools',
    'ListDirectoryTool': 'file_operation_tools',
    'WriteFileTool': 'file_operation_tools',
    'RAGSearchTool': 'rag_search_tool',
    'CreateNoteTool': 'note_management_tools',
    'SearchNotesTool': 'note_management_tools',
    'UpdateNoteTool': 'note_management_tools',
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
        # Preserve the previous eager-import fallback semantics: the
        # optional dependency backing this tool isn't installed.
        value = None
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value