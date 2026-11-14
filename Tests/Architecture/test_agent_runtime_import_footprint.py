"""Shrink-only ratchet on what ``import tldw_chatbook.Agents.agent_runtime`` loads.

The module documents itself as pure, but today it imports chains that
end at textual, httpx, sqlite3 and the DB layer. TASK-36944.10 makes it pure;
until then this pins the footprint so it cannot grow (TASK-36942.2).

The provider types the runtime imports live in ``tldw_chatbook.Provider_Contract``
(TASK-36944.1). The tests at the end keep that package free of the layers above
it, and keep its old ``Chat`` import paths working.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

PIN = Path(__file__).with_name("agent_runtime_import_footprint.txt")
PROBE = (
    "import sys, json\n"
    "import tldw_chatbook.Agents.agent_runtime\n"
    "mods = [m for m in sys.modules if m.startswith('tldw_chatbook')]\n"
    "mods += [m for m in ('textual', 'httpx', 'sqlite3') if m in sys.modules]\n"
    "print('FOOTPRINT=' + json.dumps(sorted(mods)))\n"
)


def _pinned() -> set[str]:
    return {
        line.strip()
        for line in PIN.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }


def test_agent_runtime_import_footprint_only_shrinks():
    # Inherit os.environ: the scratch HOME/XDG/TLDW_CONFIG_PATH must reach the child.
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True, text=True, env=dict(os.environ), timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    line = next(l for l in result.stdout.splitlines() if l.startswith("FOOTPRINT="))
    loaded = set(json.loads(line[len("FOOTPRINT="):]))
    pinned = _pinned()

    grew = sorted(loaded - pinned)
    assert not grew, (
        "Importing tldw_chatbook.Agents.agent_runtime now loads modules that are "
        f"not pinned (the runtime must not gain imports): {grew}"
    )
    stale = sorted(pinned - loaded)
    assert not stale, (
        "These pinned entries are no longer loaded; good. Remove them from "
        f"{PIN.name} so the ratchet tightens: {stale}"
    )


RUNTIME_FORBIDDEN = (
    "tldw_chatbook.DB", "tldw_chatbook.Chat", "tldw_chatbook.UI", "tldw_chatbook.config",
    "textual", "httpx", "sqlite3",
)


def test_agent_runtime_import_loads_no_ui_db_http_or_config():
    loaded = _probe(
        "import json, sys\n"
        "import tldw_chatbook.Agents.agent_runtime\n"
        "print('LOADED=' + json.dumps(sorted(sys.modules)))\n",
        "LOADED=",
    )
    forbidden = sorted(
        m for m in loaded
        if any(m == p or m.startswith(p + ".") for p in RUNTIME_FORBIDDEN)
    )
    assert not forbidden, f"agent_runtime import loads (TASK-36944.10): {forbidden[:20]}"


def test_old_ephemeral_origin_and_tool_catalog_paths_still_resolve():
    same = _probe(
        "import json\n"
        "from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY as a\n"
        "from tldw_chatbook.Provider_Contract.provider_continuation import EPHEMERAL_ORIGIN_KEY as b\n"
        "from tldw_chatbook.Agents.tool_catalog import PathAwareToolProvider as c, ToolPathTarget as d\n"
        "from tldw_chatbook.Agents import agent_models as m\n"
        "print('SAME=' + json.dumps(a == b and c is m.PathAwareToolProvider and d is m.ToolPathTarget))\n",
        "SAME=",
    )
    assert same is True


# The package's modules, and the old Chat module each one used to be (TASK-36944.1).
PROVIDER_CONTRACT_MOVES = {
    "tldw_chatbook.Provider_Contract": "tldw_chatbook.Chat.Chat_Deps",
    "tldw_chatbook.Provider_Contract.provider_continuation": (
        "tldw_chatbook.Chat.provider_continuation"
    ),
    "tldw_chatbook.Provider_Contract.provider_endpoint_contract": (
        "tldw_chatbook.Chat.provider_endpoint_contract"
    ),
    "tldw_chatbook.Provider_Contract.provider_setup_facts": (
        "tldw_chatbook.Chat.provider_setup_facts"
    ),
    "tldw_chatbook.Provider_Contract.sampling_params": (
        "tldw_chatbook.Chat.sampling_params"
    ),
    "tldw_chatbook.Provider_Contract.thinking_blocks": (
        "tldw_chatbook.Chat.thinking_blocks"
    ),
}
# What importing the package must not load: tldw_chatbook's higher packages and config,
# plus the UI, HTTP and database libraries.
PROVIDER_CONTRACT_FORBIDDEN = (
    *(
        f"tldw_chatbook.{package}"
        for package in (
            "Chat", "DB", "Agents", "LLM_Calls", "UI", "Widgets", "Event_Handlers"
        )
    ),
    "tldw_chatbook.config",
    "textual",
    "httpx",
    "sqlite3",
)


def _probe(code: str, marker: str) -> object:
    """Run ``code`` in a fresh interpreter and decode the JSON it prints after ``marker``."""
    # Inherit os.environ: the scratch HOME/XDG/TLDW_CONFIG_PATH must reach the child.
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=dict(os.environ), timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    line = next(l for l in result.stdout.splitlines() if l.startswith(marker))
    return json.loads(line[len(marker):])


def test_provider_contract_loads_no_higher_layer():
    loaded = _probe(
        "import importlib, json, sys\n"
        f"for name in {sorted(PROVIDER_CONTRACT_MOVES)!r}:\n"
        "    importlib.import_module(name)\n"
        "print('LOADED=' + json.dumps(sorted(sys.modules)))\n",
        "LOADED=",
    )
    forbidden = sorted(
        module
        for module in loaded
        if any(
            module == prefix or module.startswith(prefix + ".")
            for prefix in PROVIDER_CONTRACT_FORBIDDEN
        )
    )
    assert not forbidden, (
        "Importing tldw_chatbook.Provider_Contract loads modules it must not depend "
        f"on (TASK-36944.1): {forbidden}. Pass what it needs in, or import it inside "
        "the function that uses it."
    )


def test_old_provider_paths_are_the_moved_modules():
    # Old path first, in a fresh interpreter: the alias must hold in that order too.
    # The same module object means imports, isinstance checks and test patches
    # through the old path all reach the moved code.
    same = _probe(
        "import importlib, json\n"
        f"moves = {PROVIDER_CONTRACT_MOVES!r}\n"
        "old = {new: importlib.import_module(path) for new, path in moves.items()}\n"
        "print('SAME=' + json.dumps({new: old[new] is importlib.import_module(new)"
        " for new in moves}))\n",
        "SAME=",
    )
    assert same == dict.fromkeys(PROVIDER_CONTRACT_MOVES, True), same


def test_production_code_imports_provider_types_from_the_package():
    # The old Chat paths stay as aliases for outside callers; nothing in the app
    # itself may import through them, so Agents and LLM_Calls never run Chat/__init__.
    import ast
    import warnings

    old = {old_path for old_path in PROVIDER_CONTRACT_MOVES.values()}
    root = Path(__file__).resolve().parents[2] / "tldw_chatbook"
    alias_files = {root / "Chat" / (path.rsplit(".", 1)[1] + ".py") for path in old}
    offenders = []
    for file in sorted(root.rglob("*.py")):
        if file in alias_files:
            continue
        package = ".".join(("tldw_chatbook", *file.relative_to(root).parent.parts))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = package.split(".")[: len(package.split(".")) - node.level + 1]
                    base = ".".join([*parts, *([base] if base else [])])
                targets = [base, *(f"{base}.{alias.name}" for alias in node.names)]
            offenders += [
                f"{file.relative_to(root)}:{node.lineno} {target}"
                for target in targets
                if target in old
            ]
    assert not offenders, f"Import these from tldw_chatbook.Provider_Contract: {offenders}"


# Every import that leaves the package, top-level or inside a function (a call-time
# import still couples the layers; the fresh-interpreter test above sees import time
# only). Pinned exactly: a new edge fails, and a removed one must be deleted here.
# Provider facts come from the registry records (TASK-36946.1), not Chat.provider_readiness.
# Plugins.continuation stays lazy so V1 restores skip the plugin stack.
PROVIDER_CONTRACT_OUTBOUND_IMPORTS = {
    ("provider_continuation", "tldw_chatbook.Plugins.continuation"),
    ("provider_continuation", "tldw_chatbook.Utils.input_validation"),
    ("provider_continuation", "tldw_chatbook.model_capabilities"),
    ("provider_continuation", "tldw_chatbook.provider_registry"),
    ("provider_endpoint_contract", "tldw_chatbook.provider_registry"),
    ("provider_setup_facts", "tldw_chatbook.provider_registry"),
    ("thinking_blocks", "tldw_chatbook.Utils.input_validation"),
}


def test_provider_contract_outbound_imports_are_pinned():
    import ast

    package_dir = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "Provider_Contract"
    found = set()
    for file in sorted(package_dir.glob("*.py")):
        tree = ast.parse(file.read_text(encoding="utf-8"))
        type_only = {
            id(inner)
            for node in ast.walk(tree)
            if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test)
            for inner in ast.walk(node)
        }
        for node in ast.walk(tree):
            if id(node) in type_only:
                continue
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                names = [node.module]
            else:
                # Relative imports stay inside the package.
                continue
            found |= {
                (file.stem, name)
                for name in names
                if name.startswith("tldw_chatbook.")
                and not name.startswith("tldw_chatbook.Provider_Contract")
            }
    assert found == PROVIDER_CONTRACT_OUTBOUND_IMPORTS, (
        f"new: {sorted(found - PROVIDER_CONTRACT_OUTBOUND_IMPORTS)}; "
        f"gone (delete the pin row): {sorted(PROVIDER_CONTRACT_OUTBOUND_IMPORTS - found)}. "
        "Provider_Contract must stay neutral: pass what it needs in instead of importing it."
    )
