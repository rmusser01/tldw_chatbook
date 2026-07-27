"""TASK-968: MCP server no longer references a nonexistent CharacterInteropService.

``TldwMCPServer._init_databases()`` used to construct a
``CharacterInteropService`` -- a class that does not exist anywhere in the
codebase (``Character_Chat_Lib.py`` is a free-function module, not a
service class). Because the import ran unconditionally inside
``_init_databases()``, constructing a ``TldwMCPServer`` always raised
``ImportError`` before it could open a single database connection. The
resulting ``self.character_service`` attribute was never read anywhere
else in the MCP package either: the character-related tools
(``MCPTools.chat_with_character`` / ``list_available_characters``) already
read character rows directly off ``self.chachanotes_db``
(``get_character_card_by_id`` / ``list_character_cards``). The dead
reference has been removed rather than resolved to a real service, since
there was no unfinished feature behind it to wire up.

Also covers the adjacent finding from this task's "look at the module more
broadly" mandate: ``chat_with_llm``'s API key lookup used to call
``get_cli_setting("API", f"{provider}_api_key", "")`` directly instead of
the declared accessor ``config.get_api_key()``, silently missing a key
configured via the newer ``api_settings.<provider>`` structure or a bare
environment variable (both of which ``get_api_key()`` checks first/last).
"""

from __future__ import annotations

from pathlib import Path

import pytest


class _PermissiveFakeService:
    """Stand-in for NotesInteropService, whose own call-signature mismatch
    is a separate, pre-existing defect out of this task's scope (see
    test_server_media_db_path.py and this task's Implementation Notes)."""

    def __init__(self, *args, **kwargs) -> None:
        pass


def _server_source() -> str:
    import tldw_chatbook.MCP.server as srv

    return Path(srv.__file__).read_text(encoding="utf-8")


def test_no_character_interop_service_reference_remains():
    """No reference to the nonexistent service remains in the module.

    Walks the AST rather than grepping raw text: an explanatory comment on
    the removal legitimately names the class it removed (see the source),
    which a text search would false-flag.
    """
    import ast

    tree = ast.parse(_server_source())

    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "CharacterInteropService" not in imported_names

    referenced_names = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    } | {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    assert "character_service" not in referenced_names
    assert "CharacterInteropService" not in referenced_names


def test_init_databases_no_longer_needs_a_character_service_stub(monkeypatch):
    """Regression guard for the exact bug: before the fix, driving
    ``_init_databases()`` far enough to construct required stubbing a
    ``CharacterInteropService`` into ``Character_Chat_Lib`` (a class that
    doesn't exist there) just so the import wouldn't raise. That stub
    should no longer be necessary at all -- and the constructed instance
    should have no ``character_service`` attribute.
    """
    import tldw_chatbook.Notes.Notes_Library as notes_library_module
    from tldw_chatbook.MCP import server as mcp_server_module

    monkeypatch.setattr(
        notes_library_module, "NotesInteropService", _PermissiveFakeService
    )

    instance = mcp_server_module.TldwMCPServer.__new__(mcp_server_module.TldwMCPServer)
    instance._init_databases()

    assert not hasattr(instance, "character_service")


def test_character_chat_lib_still_has_no_interop_service_class():
    """Sanity check on the decision itself: confirms the removal (not a
    rename-and-rewire) was correct by asserting the source module this bug
    imported from still has no such class -- if one is ever added, this
    test should be revisited alongside the removal in server.py.
    """
    import tldw_chatbook.Character_Chat.Character_Chat_Lib as character_chat_lib_module

    assert not hasattr(character_chat_lib_module, "CharacterInteropService")


def test_chat_with_llm_uses_the_declared_api_key_accessor():
    """The broader-look finding: the API key lookup must go through
    ``config.get_api_key()`` (checks api_settings.<provider>, the legacy
    [API] section, and a bare env var) rather than a direct
    ``get_cli_setting("API", ...)`` call that only covers the middle tier.

    Walks the AST rather than grepping raw text: explanatory comments in
    the module legitimately mention ``get_cli_setting`` by name (this is
    exactly what used to be called), so a text search would false-flag.
    """
    import ast

    tree = ast.parse(_server_source())
    called_names = {
        getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert "get_api_key" in called_names
    assert "get_cli_setting" not in called_names

    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "get_api_key" in imported_names
    assert "get_cli_setting" not in imported_names
