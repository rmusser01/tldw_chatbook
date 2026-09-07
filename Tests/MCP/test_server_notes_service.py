"""TASK-983: MCP notes tools must call a real ``NotesInteropService`` API.

``TldwMCPServer._init_databases()`` used to construct
``NotesInteropService(self.chachanotes_db)`` -- one positional argument, a
``CharactersRAGDB`` instance, bound to the real class's first parameter
(``base_db_directory: Union[str, Path]``). Every construction raised (either
immediately, or once ``verify_trusted_directory`` rejected the bogus "path"),
so the MCP server could never even finish initializing, let alone run a
notes tool.

The ``create_note`` and ``search_notes`` tool bodies then called a method
that doesn't exist at all (``notes_service.create_note(tags=, template=)``)
and one that exists but with the wrong keyword names and no ``user_id``
(``notes_service.search_notes(query=, limit=)`` instead of
``search_notes(user_id=, search_term=, limit=)``), then read the results as
attribute access (``note.id``, ``note.updated_at``) when
``NotesInteropService.search_notes`` returns plain dicts keyed by the real
``notes`` table columns (which has ``last_modified``, not ``updated_at``).

This test drives the real, unmodified ``_init_databases()`` and the real
tool closures ``_register_tools()`` defines, against a temp on-disk
database -- proving the fix works end to end rather than only importing the
module. ``HOME``/``TLDW_CONFIG_PATH`` are redirected to a temp profile so
this never touches a real user config or database.
"""

from __future__ import annotations

import pytest

gateway = pytest.importorskip(
    "mcp_unified.gateway", reason="mcp-unified extra not installed"
)
GatewayRequestContext = gateway.GatewayRequestContext


@pytest.fixture
def _isolated_profile(tmp_path, monkeypatch):
    """Redirect HOME and TLDW_CONFIG_PATH to a temp profile.

    Never touches the real user config/databases -- a fresh, nonexistent
    config path under ``tmp_path`` bootstraps its own defaults on first
    access, and every ``*_db_path`` accessor server.py calls resolves
    underneath the redirected HOME.
    """
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "profile_config.toml"))
    monkeypatch.delenv("USERS_NAME", raising=False)
    return tmp_path


def _build_server_with_real_notes_tools():
    """Construct a bare ``TldwMCPServer`` with real notes tools registered.

    Bypasses the legacy ``__init__`` via ``__new__``, runs the real,
    unmodified ``_init_databases()``, then registers and finalizes the real
    closures against the standalone gateway adapter.
    """
    from tldw_chatbook.MCP.gateway_runtime import ChatbookGatewayRuntime
    from tldw_chatbook.MCP import server as mcp_server_module

    instance = mcp_server_module.TldwMCPServer.__new__(mcp_server_module.TldwMCPServer)
    instance._init_databases()

    runtime = ChatbookGatewayRuntime(
        name="tldw_chatbook",
        version="0.1.0",
        tool_descriptors=mcp_server_module._describe_local_tools(),
    )
    instance.mcp = runtime
    instance._register_tools()
    runtime.finalize()

    return instance, runtime


@pytest.mark.asyncio
async def test_create_note_and_search_notes_work_end_to_end(_isolated_profile):
    """AC: create_note and search_notes work end to end against a temp DB."""
    _instance, runtime = _build_server_with_real_notes_tools()
    context = GatewayRequestContext(request_id="notes-end-to-end")

    created = await runtime.call_tool(
        "create_note",
        {"title": "Grocery List", "content": "Buy oat milk and coffee beans"},
        context,
    )
    assert "error" not in created
    assert created["title"] == "Grocery List"
    assert created["id"]

    results = await runtime.call_tool(
        "search_notes", {"query": "oat milk", "limit": 10}, context
    )
    assert results, f"expected a match, got: {results}"
    assert not any("error" in r for r in results)

    match = next(r for r in results if r["id"] == created["id"])
    assert match["title"] == "Grocery List"
    assert "Buy oat milk" in match["preview"]
    assert match["created"] is not None
    assert match["modified"] is not None


@pytest.mark.asyncio
async def test_create_note_tool_no_longer_accepts_tags_or_template(_isolated_profile):
    """Regression guard: ``tags``/``template`` are gone from the tool schema.

    Neither parameter maps to anything ``NotesInteropService.add_note`` (or
    the ``notes`` table) actually supports -- passing them used to be
    silently accepted syntax that still failed at the (nonexistent)
    ``create_note`` method call.
    """
    _instance, runtime = _build_server_with_real_notes_tools()
    descriptors = await runtime.list_tools(
        GatewayRequestContext(request_id="notes-schema")
    )
    create_note = next(item for item in descriptors if item["name"] == "create_note")
    parameters = set(create_note["inputSchema"]["properties"])

    assert parameters == {"title", "content"}


def test_notes_service_is_constructed_with_the_real_signature(_isolated_profile):
    """Regression guard for the exact bug: the service must construct
    without needing a permissive fake (unlike the pre-fix tests in
    ``test_server_media_db_path.py`` / ``test_server_character_service.py``,
    which had to stub ``NotesInteropService`` out entirely because the real
    class rejected the old call shape).
    """
    from tldw_chatbook.Notes.Notes_Library import NotesInteropService

    instance, _runtime = _build_server_with_real_notes_tools()

    assert isinstance(instance.notes_service, NotesInteropService)
    assert instance.notes_service.unified_db_template is instance.chachanotes_db
