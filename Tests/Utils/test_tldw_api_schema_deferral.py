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

import os
import subprocess
import sys
from pathlib import Path

import pytest


# The only tldw_api submodule an `import tldw_chatbook.app` may still load.
# TASK-21107 closed the last exception: `kanban_schemas` used to ride the
# app import via `Kanban_Interop/server_kanban_service.py`'s module-scope
# 31-name import from `..tldw_api` (a genuine module-scope need inside that
# package -- `KANBAN_OPERATION_SPECS` maps action names to real schema
# classes), reached because `app_service_wiring.py` imported Kanban_Interop
# at module scope to construct the services at boot. TASK-21239 made the
# trio lazy (first-use properties), which took the whole package --
# including `kanban_schemas`' 76 pydantic models -- off the boot path
# without touching the spec table. The allowlist is now empty beyond the
# package facade itself.
ALLOWED_SCHEMA_SUBMODULES = {
    "tldw_chatbook.tldw_api",
    # `exceptions` is a deliberate exception (pre-existing, re-verified
    # 2026-10-11): `Audio_Services_Interop/server_audio_services_service.py`
    # imports two of its exception types at module scope for runtime
    # `except` clauses across a dozen methods. The module is a handful of
    # stdlib-only exception classes -- no pydantic, no schemas -- so the
    # deferral win of moving every except site to function-local imports is
    # negligible next to the churn. Named here so it is a reviewed entry,
    # not a silent pass.
    "tldw_chatbook.tldw_api.exceptions",
}


def test_app_import_schema_submodule_set_is_within_allowlist(tmp_path):
    """Fresh-subprocess check: the app import stays within the allowlist.

    After ``import tldw_chatbook.app``, every loaded ``tldw_api.*`` submodule
    must be in ``ALLOWED_SCHEMA_SUBMODULES`` -- the headline regression net
    for phase 2 (before it, ~47 schema submodules loaded this way). The
    subprocess is hermetic (PR #674 review): cwd pinned to the repo root so
    the editable install resolves THIS tree, and HOME/XDG isolated to
    ``tmp_path`` so the import surface can't vary with the runner's
    environment or real user config.

    Args:
        tmp_path: pytest-provided scratch directory used as the isolated
            HOME/XDG root for the subprocess.
    """
    code = (
        "import tldw_chatbook.app\n"
        "import sys, json\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('tldw_chatbook.tldw_api'))\n"
        "print('LOADED_JSON:' + json.dumps(loaded))\n"
    )
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env.update(
        HOME=str(tmp_path),
        XDG_CONFIG_HOME=str(tmp_path / ".config"),
        XDG_DATA_HOME=str(tmp_path / ".local" / "share"),
        PYTHONPATH=str(repo_root),
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=repo_root,
        env=env,
    )
    assert result.returncode == 0, (
        f"subprocess failed (rc={result.returncode})\n"
        f"stdout={result.stdout}\nstderr={result.stderr[-3000:]}"
    )
    import json

    line = next(
        line for line in result.stdout.splitlines() if line.startswith("LOADED_JSON:")
    )
    loaded = set(json.loads(line[len("LOADED_JSON:") :]))
    extra = loaded - ALLOWED_SCHEMA_SUBMODULES
    assert not extra, (
        f"tldw_api submodules loaded by `import tldw_chatbook.app` beyond the "
        f"phase-2 allowlist: {sorted(extra)}"
    )
    # task-21106: `Actor_Packs/creation.py` (on app.py's own import chain)
    # regressed to a module-scope import of this 79-model schema module,
    # defeating the lazy facade. Pin the exact module by name so the next
    # regression reads as itself, not just an allowlist diff.
    assert "tldw_chatbook.tldw_api.character_persona_schemas" not in loaded, (
        "`import tldw_chatbook.app` loaded character_persona_schemas eagerly "
        "(task-21106 regression — check Actor_Packs/creation.py first)"
    )


def test_server_chat_grammars_service_defers_schema_import_then_fails_gracefully():
    """Converted service still builds schemas lazily and fails as before.

    A Server*Service constructed with ``client=None`` (the from_config
    fallback shape) must still build its request schema — proving the
    function-local import resolves lazily — and must fail exactly the way
    it always did (a plain ValueError) once it discovers there is no
    client/client_provider to send the request to.
    """
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
    is usable for real construction/validation.
    Args:
        tmp_path: pytest-provided scratch directory for the store file.
    """
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


def test_kanban_schemas_absent_from_app_import_closure():
    """TASK-21107 AC: the 76-model kanban_schemas module stays off app import.

    A fresh-process import of the app module must not load
    ``tldw_chatbook.tldw_api.kanban_schemas`` -- pinned as its own test (not
    just allowlist membership) so the win is named and a re-eagering of the
    Kanban chain fails with the offender in the message.
    """
    import importlib
    import subprocess

    code = (
        "import tldw_chatbook.app\n"
        "import sys\n"
        "assert 'tldw_chatbook.tldw_api.kanban_schemas' not in sys.modules, "
        "'kanban_schemas re-eagered on the app import path'\n"
        "assert not any('Kanban_Interop' in m for m in sys.modules), "
        "'Kanban_Interop re-eagered on the app import path'\n"
        "print('CLEAN')\n"
    )
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(repo_root),
        },
        check=False,
        timeout=120,
    )
    assert result.returncode == 0 and "CLEAN" in result.stdout, (
        f"kanban import leak: {result.stderr[-2000:]}"
    )
    # importlib import kept for parity with the module's other helpers
    del importlib
