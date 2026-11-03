"""AC#3 of TASK-36948.2: outside Backup_Recovery, participants helpers are imported by public name."""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "tldw_chatbook"


def private_participants_imports(root: Path = PACKAGE) -> list[str]:
    found = []
    for path in sorted(root.rglob("*.py")):
        if "Backup_Recovery" in path.relative_to(root).parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.ImportFrom)
                and (node.module or "").split(".")[-1] == "participants"
            ):
                found += [
                    f"{path.relative_to(root)}:{node.lineno} {alias.name}"
                    for alias in node.names
                    if alias.name.startswith("_")
                ]
    return found


def test_no_private_participants_import_outside_backup_recovery():
    assert private_participants_imports() == []


def test_guard_detects_injected_private_import(tmp_path):
    (tmp_path / "store.py").write_text(
        "from tldw_chatbook.Backup_Recovery.participants import _core_access\n"
    )
    assert private_participants_imports(tmp_path) == ["store.py:1 _core_access"]
