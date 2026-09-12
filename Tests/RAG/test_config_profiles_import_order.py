"""Import-order regression guard for the config_profiles <-> simplified cycle.

TASK-21160: ``RAG_Search/config_profiles.py`` imports ``.simplified.config``,
which executes ``simplified/__init__``, which (eagerly) executed
``enhanced_rag_service_v2`` / ``rag_factory`` / ``active_config`` -- each of
which imported ``..config_profiles`` back at module level. A
``config_profiles``-FIRST import order therefore hit the partially-initialized
module and raised ImportError. The cycle edges were latent for as long as the
eager ``RAG_Search/__init__`` front-loaded ``simplified`` in the safe order;
TASK-21102's lazy facade unmasked them (standalone collection of
``Tests/UI/test_console_runtime_ownership.py`` failed on dev from d60ebe1d0
until this fix).

Each test runs in a fresh subprocess so this file's own import order cannot
mask a regression.
"""

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _fresh_import(statement: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=120,
    )


def test_config_profiles_first_import_succeeds():
    """The order that reproduced the dev regression: config_profiles first."""
    result = _fresh_import(
        "import tldw_chatbook.RAG_Search.config_profiles as cp; "
        "assert callable(cp.get_profile_manager)"
    )
    assert result.returncode == 0, (
        "config_profiles-first import failed -- the "
        f"config_profiles<->simplified cycle is back:\n{result.stderr[-2000:]}"
    )


def test_simplified_first_import_still_succeeds():
    """The historically-safe order must keep working after the deferral."""
    result = _fresh_import(
        "import tldw_chatbook.RAG_Search.simplified as s; "
        "import tldw_chatbook.RAG_Search.config_profiles as cp; "
        "assert callable(cp.get_profile_manager); "
        "assert s.EnhancedRAGServiceV2 is not None"
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_simplified_modules_do_not_import_config_profiles_at_module_level():
    """Static edge census: no module executed by ``simplified/__init__`` may
    re-import config_profiles at module level (function-local and
    TYPE_CHECKING imports are the sanctioned shapes)."""
    import ast

    pkg = _REPO_ROOT / "tldw_chatbook" / "RAG_Search" / "simplified"
    offenders = []
    for path in sorted(pkg.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module and "config_profiles" in node.module:
                # Module-level = the import statement is a direct child of the
                # module body or of a plain `if TYPE_CHECKING:` block; only
                # the former is an offense.
                for stmt in tree.body:
                    if stmt is node:
                        offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "module-level config_profiles imports re-create the circular-import "
        f"edge (defer to use-site or TYPE_CHECKING): {offenders}"
    )
