# Tests/Utils/test_tldw_api_lazy.py
"""
Regression tests for the lazy PEP 562 re-export layer in
``tldw_chatbook/tldw_api/__init__.py`` (task-285).

The package used to eagerly import 54 schema submodules (~1,300 Pydantic
models) at package-import time. It now resolves names on first attribute
access via module ``__getattr__``, mirroring the pattern already used by
``Local_Ingestion/__init__.py``. These tests are the regression net that
guarantees the rewrite dropped nothing: every name the old file exported
must still resolve, from the correct defining submodule, with no import
errors -- while a bare package import stays cheap.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

import tldw_chatbook.tldw_api as tldw_api


# A sample of heavy schema submodules that used to be imported eagerly by
# tldw_api/__init__.py. None of these should be loaded by a bare
# `import tldw_chatbook.tldw_api` in a fresh process.
_HEAVY_SUBMODULES = [
    "tldw_chatbook.tldw_api.client",
    "tldw_chatbook.tldw_api.audiobook_schemas",
    "tldw_chatbook.tldw_api.kanban_schemas",
]


def test_bare_import_does_not_eagerly_load_heavy_submodules():
    """Fresh-subprocess check: importing just the package must not pull in
    the heavy schema submodules that used to be imported eagerly."""
    code = (
        "import tldw_chatbook.tldw_api\n"
        "import sys\n"
        f"heavy = {_HEAVY_SUBMODULES!r}\n"
        "loaded = [m for m in heavy if m in sys.modules]\n"
        "assert not loaded, f'eagerly imported: {loaded}'\n"
        "print('LAZY_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"subprocess failed (rc={result.returncode})\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "LAZY_OK" in result.stdout


def test_names_from_different_submodules_resolve_to_defining_module():
    """`from tldw_chatbook.tldw_api import X, Y, Z` where X/Y/Z come from
    different submodules must still work, and each resolved object's
    __module__ must be its actual defining submodule (not the package)."""
    from tldw_chatbook.tldw_api import (
        NoteResponse,  # notes_workspace_schemas
        FlashcardResponse,  # flashcards_schemas
        TLDWAPIClient,  # client
        MCPUnifiedClient,  # mcp_unified_client
        ChatLoopStartRequest,  # chat_loop_schemas
        KanbanCardResponse,  # kanban_schemas
    )

    assert NoteResponse.__module__ == "tldw_chatbook.tldw_api.notes_workspace_schemas"
    assert FlashcardResponse.__module__ == "tldw_chatbook.tldw_api.flashcards_schemas"
    assert TLDWAPIClient.__module__ == "tldw_chatbook.tldw_api.client"
    assert MCPUnifiedClient.__module__ == "tldw_chatbook.tldw_api.mcp_unified_client"
    assert ChatLoopStartRequest.__module__ == "tldw_chatbook.tldw_api.chat_loop_schemas"
    assert KanbanCardResponse.__module__ == "tldw_chatbook.tldw_api.kanban_schemas"


def test_module_level_alias_preserves_identity():
    """The old file defined `WebProcessResponse = IngestWebContentResponse`
    directly in __init__.py. The lazy layer must resolve both names to the
    exact same object (not two separately-constructed equivalents)."""
    from tldw_chatbook.tldw_api import WebProcessResponse, IngestWebContentResponse

    assert WebProcessResponse is IngestWebContentResponse
    assert WebProcessResponse.__module__ == "tldw_chatbook.tldw_api.media_reading_schemas"


def test_unknown_attribute_raises_attribute_error_naming_the_package():
    with pytest.raises(AttributeError, match="tldw_chatbook.tldw_api"):
        getattr(tldw_api, "NoSuchSchemaNameXYZ")


def test_dir_includes_lazy_names():
    d = dir(tldw_api)
    assert "NoteResponse" in d
    assert "TLDWAPIClient" in d
    assert "WebProcessResponse" in d


def test_all_dunder_all_names_resolve_without_error():
    """Big regression net: every name in __all__ (the package's public,
    star-import-visible surface) must resolve via __getattr__ with no
    exceptions. May take a few seconds -- that's expected and fine."""
    failed = []
    for name in tldw_api.__all__:
        try:
            getattr(tldw_api, name)
        except Exception as exc:  # pragma: no cover - failure path
            failed.append((name, repr(exc)))
    assert not failed, f"{len(failed)} of {len(tldw_api.__all__)} names in __all__ failed to resolve: {failed[:20]}"


def test_app_import_does_not_load_tldw_api_client():
    """Regression net for the eager-import chain that used to defeat the lazy
    package layer: runtime_policy/bootstrap.py (reached via
    Chat/server_chat_conversation_service.py early in app.py's import chain)
    did a module-scope `from tldw_chatbook.tldw_api import TLDWAPIClient`,
    which pulled in tldw_api/client.py -- a 15k-line module that re-imports
    virtually the entire schema surface. After task-285's TYPE_CHECKING
    conversion of every on-app-path importer, a bare `import
    tldw_chatbook.app` must NOT load tldw_api.client; it loads on demand the
    first time a server client is actually constructed
    (runtime_policy/bootstrap.py::build_runtime_api_client)."""
    code = (
        "import tldw_chatbook.app\n"
        "import sys\n"
        "assert 'tldw_chatbook.tldw_api.client' not in sys.modules, "
        "'tldw_api.client was eagerly loaded by app import'\n"
        "print('APP_LAZY_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"subprocess failed (rc={result.returncode})\n"
        f"stdout={result.stdout}\nstderr={result.stderr[-3000:]}"
    )
    assert "APP_LAZY_OK" in result.stdout


def test_full_submodule_mapping_resolves_without_error():
    """Even broader net than __all__: `_SUBMODULE_BY_NAME` is a superset of
    __all__ (some names were importable via `from tldw_chatbook.tldw_api
    import X` in the old eager file even though X was never listed in
    __all__). Every one of those must still resolve too."""
    names = tldw_api._SUBMODULE_BY_NAME
    failed = []
    for name in names:
        try:
            getattr(tldw_api, name)
        except Exception as exc:  # pragma: no cover - failure path
            failed.append((name, repr(exc)))
    assert not failed, f"{len(failed)} of {len(names)} mapped names failed to resolve: {failed[:20]}"
    # Sanity: the mapping is genuinely a superset of __all__.
    assert set(tldw_api.__all__) <= set(names)
