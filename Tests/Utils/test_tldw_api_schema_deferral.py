# Tests/Utils/test_tldw_api_schema_deferral.py
"""
Regression net for task-285 phase 2: deferring ``tldw_api`` *schema submodule*
imports (not just the package itself, and not just ``TLDWAPIClient`` -- see
``test_tldw_api_lazy.py`` for those) off the ``import tldw_chatbook.app`` path.

Phase 1 made ``tldw_api/__init__.py`` a lazy PEP 562 re-export layer and
deferred every on-path ``TLDWAPIClient`` import, but ~47 schema submodules
still loaded eagerly because ~44 ``Server*Service``/``Local*Service`` modules
named their request/response schema types at *module scope* (annotations and,
far more often, runtime `SomeRequest(...)` construction inside methods), and
the lazy package layer faithfully served those module-scope lookups.

Phase 2 converted those sites: annotation-only/unused names moved under
``if TYPE_CHECKING:``, and runtime-constructed names moved to function-local
imports (or, for the one module that keeps a genuine module-scope dispatch
table mapping action names to real schema classes --
``Kanban_Interop/server_kanban_service.py`` -- left alone and allowlisted,
per the "don't force a module/class-scope site" instruction).

These tests are the regression net: a fresh-process ``import
tldw_chatbook.app`` must not pull in schema submodules beyond the tiny
allowlist below, and a representative sample of the converted services must
still behave correctly (construct with ``client=None`` via their
``from_config``-style fallback, exercise a method that builds a schema
object, and either succeed end-to-end or fail exactly the way they always
did when no client/client_provider is configured).
"""
from __future__ import annotations

import subprocess
import sys

import pytest


# The only tldw_api submodule an `import tldw_chatbook.app` may still load.
# `kanban_schemas` is a deliberate, documented exception: server_kanban_service.py
# keeps a module-scope `KANBAN_OPERATION_SPECS` dict mapping action names to
# real schema *classes* (not strings) for runtime dispatch -- a genuine
# module-scope need, not an oversight (task-285 phase 2 Implementation Notes).
ALLOWED_SCHEMA_SUBMODULES = {
    "tldw_chatbook.tldw_api",
    "tldw_chatbook.tldw_api.kanban_schemas",
}


def test_app_import_schema_submodule_set_is_within_allowlist():
    """Fresh-subprocess check: after `import tldw_chatbook.app`, every loaded
    tldw_api.* submodule must be in ALLOWED_SCHEMA_SUBMODULES. This is the
    headline regression net for phase 2 -- before phase 2, ~47 schema
    submodules loaded this way; after, only the allowlisted one does."""
    code = (
        "import tldw_chatbook.app\n"
        "import sys, json\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('tldw_chatbook.tldw_api'))\n"
        "print('LOADED_JSON:' + json.dumps(loaded))\n"
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
    import json

    line = next(l for l in result.stdout.splitlines() if l.startswith("LOADED_JSON:"))
    loaded = set(json.loads(line[len("LOADED_JSON:"):]))
    extra = loaded - ALLOWED_SCHEMA_SUBMODULES
    assert not extra, (
        f"tldw_api submodules loaded by `import tldw_chatbook.app` beyond the "
        f"phase-2 allowlist: {sorted(extra)}"
    )


def test_server_chat_grammars_service_defers_schema_import_then_fails_gracefully():
    """A converted Server*Service constructed with client=None (the
    from_config fallback shape) must still be able to build its request
    schema (proving the function-local import resolves lazily and works),
    and must fail exactly the way it always did -- a plain ValueError -- once
    it discovers there's no client/client_provider to actually send the
    request to."""
    from tldw_chatbook.Chat_Grammars_Interop.server_chat_grammars_service import (
        ServerChatGrammarsService,
    )

    service = ServerChatGrammarsService(client=None)

    import asyncio

    with pytest.raises(ValueError, match="TLDW API client is required"):
        asyncio.run(
            service.create_grammar(name="test-grammar", grammar_text="root ::= 'x'")
        )


def test_local_chat_grammars_service_defers_schema_import_and_succeeds(tmp_path):
    """The fully-local counterpart needs no client at all -- constructing the
    deferred schema object and completing the whole operation must work
    end-to-end, proving the function-local import isn't just resolving but
    is usable for real construction/validation."""
    from tldw_chatbook.Chat_Grammars_Interop.local_chat_grammars_service import (
        LocalChatGrammarsService,
    )

    service = LocalChatGrammarsService(store_path=tmp_path / "grammars.json")

    import asyncio

    result = asyncio.run(
        service.create_grammar(name="test-grammar", grammar_text="root ::= 'x'")
    )
    assert result["name"] == "test-grammar"
    assert result["grammar_text"] == "root ::= 'x'"


def test_server_kanban_service_allowlisted_module_still_works_end_to_end():
    """The one module left deliberately eager (module-scope operation-spec
    registry mapping action names to real schema classes) must still behave
    correctly: constructing with client=None and invoking an operation still
    builds the request object via the registry and fails the same
    pre-existing way once there's no client to call."""
    from tldw_chatbook.Kanban_Interop.server_kanban_service import ServerKanbanService

    service = ServerKanbanService(client=None)

    import asyncio

    with pytest.raises(ValueError, match="TLDW API client is required"):
        asyncio.run(
            service.create_board(request_data={"name": "Board", "client_id": "c1"})
        )
