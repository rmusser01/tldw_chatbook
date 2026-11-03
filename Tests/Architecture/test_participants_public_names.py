"""TASK-36948.2: outside Backup_Recovery, participant helpers are used by public name.

Covers every Backup_Recovery module whose name contains "participant".  Three forms
are flagged in production code: ``from ...x_participants import _name``,
``module._name`` attribute reads and ``getattr(module, "_name")`` /
``vars(module).get("_name")``.  Module STATE that is rebound or shared
(not a def/class) stays private and is listed in ``STATE_ALLOWLIST`` until it has
an accessor API.
"""

import ast
import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "tldw_chatbook"
PARTICIPANT = re.compile(r"participants?$")

# (module, attribute) -> why it is still read privately from outside Backup_Recovery.
STATE_ALLOWLIST = {
    ("raw_participants", "_states"): "module state table; tests rebind it and callers need an accessor API",
    ("raw_participants", "_local"): "thread-local state; needs an accessor API",
    ("raw_participants", "_check"): "patched by tests via setattr, so an alias would split the patch",
    ("raw_participants", "_close_descriptor"): "patched by tests via setattr",
    ("raw_participants", "_participant_identity"): "patched by tests via setattr",
    ("raw_participants", "_unlink"): "patched by tests via setattr",
    ("raw_participants", "_CONSOLE_PERMISSION_LOAD_GUARDS"): "provenance data tuple, not a helper",
    ("raw_participants", "_PARENT_WALK_SOURCE"): "provenance data tuple, not a helper",
    ("raw_participants", "_parent_walk_operation"): "assigned from a source-bound factory, not a def",
    ("raw_participants", "_check_directory_allocation"): "assigned from a source-bound factory, not a def",
    ("mcp_source_participants", "_BINDINGS"): "module state registry",
    ("participant", "_condition"): "lock on the imported rag_definition participant singleton, not a helper",
    ("participant", "_lock"): "lock on the imported participant singleton, not a helper",
    ("participant", "_borrowers"): "borrower table of the imported participant singleton, not a helper",
    ("persona_visual_participants", "_repositories"): "module state registry",
}


def _bound_participant_modules(tree: ast.AST) -> dict[str, str]:
    bound = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if PARTICIPANT.search(alias.name):
                    bound[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                leaf = alias.name.split(".")[-1]
                if PARTICIPANT.search(leaf):
                    bound[alias.asname or leaf] = leaf
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            text = ast.unparse(node.value)
            match = re.search(r"Backup_Recovery\.(\w*participants?)[\"']", text)
            if match and ("modules" in text or "import_module" in text):
                bound[node.targets[0].id] = match.group(1)
    return bound


def _private(name: object) -> bool:
    return isinstance(name, str) and name.startswith("_") and not name.startswith("__")


def private_participant_uses(root: Path = PACKAGE) -> list[str]:
    found = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if "Backup_Recovery" in rel.parts:
            continue
        start = len(found)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        bound = _bound_participant_modules(tree)

        def flag(node, module, name, form):
            if (module, name) not in STATE_ALLOWLIST:
                found.append(f"{rel}:{node.lineno} {module}.{name} ({form})")

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and PARTICIPANT.search((node.module or "").split(".")[-1])
            ):
                for alias in node.names:
                    if _private(alias.name):
                        flag(node, node.module.split(".")[-1], alias.name, "import")
            elif (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in bound
                and _private(node.attr)
            ):
                flag(node, bound[node.value.id], node.attr, "attribute")
            elif isinstance(node, ast.Call) and len(node.args) >= 2:
                target = node.args[0]
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and isinstance(target, ast.Name)
                    and target.id in bound
                    and isinstance(node.args[1], ast.Constant)
                    and _private(node.args[1].value)
                ):
                    flag(node, bound[target.id], node.args[1].value, "getattr")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Name)
                and node.func.value.func.id == "vars"
                and node.func.value.args
                and isinstance(node.func.value.args[0], ast.Name)
                and node.func.value.args[0].id in bound
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and _private(node.args[0].value)
            ):
                flag(node, bound[node.func.value.args[0].id], node.args[0].value, "vars().get")
        found[start:] = sorted(found[start:], key=lambda row: int(row.split(":")[1].split()[0]))
    return found


def test_no_private_participant_use_outside_backup_recovery():
    assert private_participant_uses() == []


def test_allowlist_entries_are_still_needed():
    stale = []
    tree_text = "\n".join(
        p.read_text(encoding="utf-8")
        for p in PACKAGE.rglob("*.py")
        if "Backup_Recovery" not in p.relative_to(PACKAGE).parts
    )
    for (_, name), _reason in STATE_ALLOWLIST.items():
        if name not in tree_text:
            stale.append(name)
    assert stale == []


def test_guard_detects_injected_private_uses(tmp_path):
    (tmp_path / "store.py").write_text(
        "from tldw_chatbook.Backup_Recovery.participants import _core_access\n"
        "from ..Backup_Recovery.raw_participants import _scope\n"
        "from ..Backup_Recovery.rag_definition_participant import _x\n"
        "from ..Backup_Recovery.participants import core_access\n"
        "from tldw_chatbook.Backup_Recovery import raw_participants as raw\n"
        "raw._live_state(1)\n"
        "raw._states\n"
        "getattr(raw, '_check_temporary_identity')\n"
        "vars(raw).get('_scope')\n"
    )
    assert private_participant_uses(tmp_path) == [
        "store.py:1 participants._core_access (import)",
        "store.py:2 raw_participants._scope (import)",
        "store.py:3 rag_definition_participant._x (import)",
        "store.py:6 raw_participants._live_state (attribute)",
        "store.py:8 raw_participants._check_temporary_identity (getattr)",
        "store.py:9 raw_participants._scope (vars().get)",
    ]
