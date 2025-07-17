# Tools module initialization
"""
Tool execution framework for LLM function calling.
"""

from .tool_executor import (
    Tool,
    ToolExecutor,
    DateTimeTool,
    CalculatorTool,
    get_tool_executor,
    reload_tool_executor
)

# Import optional tools
try:
    from .web_search_tool import WebSearchTool
except ImportError:
    WebSearchTool = None

try:
    from .file_operation_tools import ReadFileTool, ListDirectoryTool, WriteFileTool
except ImportError:
    ReadFileTool = None
    ListDirectoryTool = None
    WriteFileTool = None

try:
    from .rag_search_tool import RAGSearchTool
except ImportError:
    RAGSearchTool = None

try:
    from .note_management_tools import CreateNoteTool, SearchNotesTool, UpdateNoteTool
except ImportError:
    CreateNoteTool = None
    SearchNotesTool = None
    UpdateNoteTool = None

__all__ = [
    'Tool',
    'ToolExecutor',
    'DateTimeTool',
    'CalculatorTool',
    'get_tool_executor',
    'reload_tool_executor'
]

# Add optional tools to __all__ if they're available
if WebSearchTool is not None:
    __all__.append('WebSearchTool')

if ReadFileTool is not None:
    __all__.append('ReadFileTool')

if ListDirectoryTool is not None:
    __all__.append('ListDirectoryTool')

if WriteFileTool is not None:
    __all__.append('WriteFileTool')

if RAGSearchTool is not None:
    __all__.append('RAGSearchTool')

if CreateNoteTool is not None:
    __all__.append('CreateNoteTool')

if SearchNotesTool is not None:
    __all__.append('SearchNotesTool')

if UpdateNoteTool is not None:
    __all__.append('UpdateNoteTool')