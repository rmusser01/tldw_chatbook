"""TASK-334: MCP server chat_with_llm repointed to chat_api_call (was ImportError)."""

import importlib
from pathlib import Path


def test_server_module_imports_without_error():
    # The dead `from ..LLM_Calls.LLM_API_Calls import chat_with_provider` used to
    # crash TldwMCPServer.__init__ -> _register_tools at construction time.
    mod = importlib.import_module("tldw_chatbook.MCP.server")
    assert mod is not None


def test_no_dead_chat_with_provider_import_in_server():
    import tldw_chatbook.MCP.server as srv
    src = Path(srv.__file__).read_text(encoding="utf-8")
    assert "import chat_with_provider" not in src
    assert "chat_api_call" in src
