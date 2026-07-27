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

from typing import Any, Callable, Dict

import pytest


class _RecordingFastMCP:
    """Stand-in for ``mcp.server.fastmcp.FastMCP``.

    Captures every ``@self.mcp.tool()``-decorated function by name so the
    test can call the exact closures ``_register_tools()`` defines, without
    requiring the optional ``mcp`` package to be installed (it is not,
    in this environment).
    """

    def __init__(self) -> None:
        self.tools: Dict[str, Callable[..., Any]] = {}

    def tool(self):
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.tools[func.__name__] = func
            return func

        return decorator


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


def _build_server_with_real_notes_tools(monkeypatch):
    """Construct a bare ``TldwMCPServer`` with real notes tools registered.

    Bypasses ``__init__`` (which requires the optional ``mcp`` package) via
    ``__new__``, runs the real, unmodified ``_init_databases()``, then
    registers tools using ``_RecordingFastMCP`` in place of ``FastMCP`` so
    the real ``create_note``/``search_notes`` closures can be captured and
    called directly.
    """
    from tldw_chatbook.MCP import server as mcp_server_module

    instance = mcp_server_module.TldwMCPServer.__new__(mcp_server_module.TldwMCPServer)
    instance._init_databases()

    fake_mcp = _RecordingFastMCP()
    instance.mcp = fake_mcp
    instance._register_tools()

    return instance, fake_mcp


@pytest.mark.asyncio
async def test_create_note_and_search_notes_work_end_to_end(
    _isolated_profile, monkeypatch
):
    """AC: create_note and search_notes work end to end against a temp DB."""
    instance, fake_mcp = _build_server_with_real_notes_tools(monkeypatch)

    create_note = fake_mcp.tools["create_note"]
    search_notes = fake_mcp.tools["search_notes"]

    created = await create_note(
        title="Grocery List", content="Buy oat milk and coffee beans"
    )
    assert "error" not in created
    assert created["title"] == "Grocery List"
    assert created["id"]

    results = await search_notes(query="oat milk", limit=10)
    assert results, f"expected a match, got: {results}"
    assert not any("error" in r for r in results)

    match = next(r for r in results if r["id"] == created["id"])
    assert match["title"] == "Grocery List"
    assert "Buy oat milk" in match["preview"]
    assert match["created"] is not None
    assert match["modified"] is not None


@pytest.mark.asyncio
async def test_create_note_tool_no_longer_accepts_tags_or_template(
    _isolated_profile, monkeypatch
):
    """Regression guard: ``tags``/``template`` are gone from the tool schema.

    Neither parameter maps to anything ``NotesInteropService.add_note`` (or
    the ``notes`` table) actually supports -- passing them used to be
    silently accepted syntax that still failed at the (nonexistent)
    ``create_note`` method call.
    """
    import inspect

    instance, fake_mcp = _build_server_with_real_notes_tools(monkeypatch)

    create_note = fake_mcp.tools["create_note"]
    parameters = set(inspect.signature(create_note).parameters)

    assert parameters == {"title", "content"}


def test_notes_service_is_constructed_with_the_real_signature(
    _isolated_profile, monkeypatch
):
    """Regression guard for the exact bug: the service must construct
    without needing a permissive fake (unlike the pre-fix tests in
    ``test_server_media_db_path.py`` / ``test_server_character_service.py``,
    which had to stub ``NotesInteropService`` out entirely because the real
    class rejected the old call shape).
    """
    from tldw_chatbook.Notes.Notes_Library import NotesInteropService

    instance, _fake_mcp = _build_server_with_real_notes_tools(monkeypatch)

    assert isinstance(instance.notes_service, NotesInteropService)
    assert instance.notes_service.unified_db_template is instance.chachanotes_db
