"""TASK-854: MCP/server.py's media DB must resolve via the real accessor.

``TldwMCPServer._init_databases()`` used to read a config key ("media_db")
that is declared nowhere in ``config.py`` -- the real key is
"media_db_path", whose accessor is ``config.get_media_db_path()`` (already
used, two lines above, for the chachanotes DB via
``get_chachanotes_db_path()``). Because the key never matched anything, the
lookup silently fell through to a CWD-relative literal
("media_library.db"), so the MCP server opened a database outside the
per-profile data directory -- and outside ``Utils.sensitive_paths``'
denylist coverage, since that denylists ``get_media_db_path()``'s result,
not a CWD-relative file.

``_init_databases()`` also used to construct a ``CharacterInteropService``
that does not exist anywhere in the codebase at all -- that dead reference
was removed by TASK-968 (it was never consumed by anything else in the MCP
package either; character-related tools already read ``chachanotes_db``
directly). ``NotesInteropService`` is still constructed with a call
signature that doesn't match the real class -- an unrelated, pre-existing
defect out of this task's scope (see TASK-968's Implementation Notes) --
so the tests below still stub it (a permissive fake swapped in at its
*source* module, so ``_init_databases``'s own local import picks it up)
to drive the real, unmodified method far enough to construct
``self.media_db`` -- proving the fix through the actual code path rather
than by re-implementing its logic in the test.
"""

from __future__ import annotations

from pathlib import Path

import pytest


class _PermissiveFakeService:
    """Stand-in for a collaborator whose real constructor signature the
    module under test does not actually match (see module docstring)."""

    def __init__(self, *args, **kwargs) -> None:
        pass


def _build_server_with_databases(monkeypatch):
    """Construct a bare ``TldwMCPServer`` and run its real ``_init_databases``.

    Bypasses ``__init__`` (which requires the optional ``mcp`` package) via
    ``__new__``, then calls the real, unmodified ``_init_databases()``
    directly -- exercising the actual fixed code path.
    """
    import tldw_chatbook.Notes.Notes_Library as notes_library_module
    from tldw_chatbook.MCP import server as mcp_server_module

    monkeypatch.setattr(
        notes_library_module, "NotesInteropService", _PermissiveFakeService
    )

    instance = mcp_server_module.TldwMCPServer.__new__(mcp_server_module.TldwMCPServer)
    instance._init_databases()
    return instance


def test_media_db_path_matches_get_media_db_path(monkeypatch):
    """The constructed media DB handle's path must equal
    ``config.get_media_db_path()`` -- not a hardcoded literal filename."""
    from tldw_chatbook.config import get_media_db_path

    instance = _build_server_with_databases(monkeypatch)

    assert Path(instance.media_db.db_path) == get_media_db_path()


def test_media_db_path_is_not_the_old_cwd_relative_literal(monkeypatch):
    """Regression guard for the exact bug: the resolved path must not be the
    bare, CWD-relative literal the old ``get_cli_setting("database",
    "media_db", "media_library.db")`` call silently fell back to."""
    instance = _build_server_with_databases(monkeypatch)

    old_buggy_path = (Path.cwd() / "media_library.db").resolve()
    assert Path(instance.media_db.db_path) != old_buggy_path


def test_resolved_media_db_path_is_sensitive(monkeypatch):
    """AC #4: once the fix lands, the path the MCP server actually opens must
    be covered by the agent-tool denylist (verified by test, not by
    inspection -- this is the exact gap TASK-854 closes)."""
    from tldw_chatbook.Utils.sensitive_paths import is_sensitive_path

    instance = _build_server_with_databases(monkeypatch)

    assert is_sensitive_path(Path(instance.media_db.db_path))


def test_no_other_undeclared_database_config_keys():
    """AC #2: grep-based check that every ``get_cli_setting("database", ...)``
    call site in the codebase uses one of the declared ``*_db_path`` (or
    known non-path) keys -- not a key that resolves to nothing, the way
    "media_db" did.

    See this task's Implementation Notes for the full list of keys this
    sweep found and how each was classified.
    """
    import ast
    import re
    from pathlib import Path as _Path

    package_root = _Path(__file__).resolve().parents[2] / "tldw_chatbook"

    # Keys read via get_cli_setting("database", <key>, ...) that are NOT a
    # ``*_db_path`` accessor key, but are legitimate, declared [database]
    # settings (not database *paths*) -- see config.py's [database] section.
    known_non_path_database_keys = {
        "check_integrity_on_startup",
        "integrity_check_timeout",
        "USER_DB_BASE_DIR",
    }

    offenders: list[str] = []

    for py_file in package_root.rglob("*.py"):
        try:
            source = py_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "get_cli_setting" not in source or '"database"' not in source and "'database'" not in source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            func_name = getattr(func, "attr", None) or getattr(func, "id", None)
            if func_name != "get_cli_setting":
                continue
            if len(node.args) < 2:
                continue
            section_arg, key_arg = node.args[0], node.args[1]
            section_value = getattr(section_arg, "value", None)
            if section_value != "database":
                continue
            key_value = getattr(key_arg, "value", None)
            if key_value is None:
                # Dynamic key (e.g. an f-string or variable) -- can't
                # statically classify; skip rather than false-flag.
                continue
            is_declared_db_path_key = bool(re.fullmatch(r"[a-z0-9_]+_db_path", key_value))
            if is_declared_db_path_key or key_value in known_non_path_database_keys:
                continue
            offenders.append(f"{py_file.relative_to(package_root)}: {key_value!r}")

    assert offenders == [], (
        "get_cli_setting('database', ...) call(s) using a key that is not a "
        f"declared '*_db_path' name: {offenders}"
    )
