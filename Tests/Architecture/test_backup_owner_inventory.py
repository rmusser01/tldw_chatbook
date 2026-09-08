"""Exact source-symbol census: new persistence calls require ownership review."""

import ast
from collections import Counter
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "backlog/docs/backup-recovery-owner-inventory.md"

# Broad discovery signal, not proof of durability. Every hit receives an explicit
# reviewed row, including memory/external/process/generic helper classifications.
PRODUCER_CALLS = frozenset(
    {
        "open",
        "write",
        "write_text",
        "write_bytes",
        "writelines",
        "dump",
        "mkdir",
        "makedirs",
        "copy2",
        "copyfile",
        "copytree",
        "atomic_private_write_text",
        "atomic_private_write_bytes",
        "create_private_text",
        "create_private_file",
        "create_private_binary",
        "open_private_binary",
        "open_private_text_append_stream",
        "open_private_text_append",
        "secure_private_directory",
        "connect",
        "connect_private_sqlite",
        "connect_private_sqlite_descriptor",
        "PersistentClient",
        "backup_connection_to_private",
        "backup_open_connections_to_private",
        "copy_private_sqlite",
        "save_pretrained",
        "to_json",
        "to_csv",
        "to_parquet",
        "to_sql",
        "ZipFile",
        "set_password",
        "delete_password",
    }
)


def census(source: str) -> Counter:
    result = Counter()

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.scope = []
            self.imports = {}

        def visit_ImportFrom(self, node):
            for alias in node.names:
                self.imports[alias.asname or alias.name] = alias.name

        def visit_ClassDef(self, node):
            self.scope.append(node.name)
            for base in node.bases:
                if getattr(base, "id", "") == "BaseDB":
                    result[(".".join(self.scope), "inherits:BaseDB")] += 1
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node):
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", "")
            )
            name = self.imports.get(name, name)
            if name == "open":
                # Builtin/Path read-only opens are not producers. os.open flags
                # and dynamic modes remain conservatively inventoried.
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                ):
                    pass
                else:
                    positional = (
                        node.args[1:] if isinstance(node.func, ast.Name) else node.args
                    )
                    mode = next(
                        (kw.value for kw in node.keywords if kw.arg == "mode"),
                        positional[0] if positional else ast.Constant("r"),
                    )
                    if (
                        isinstance(mode, ast.Constant)
                        and isinstance(mode.value, str)
                        and not any(flag in mode.value for flag in "wax+")
                    ):
                        return self.generic_visit(node)
            if name in PRODUCER_CALLS:
                result[(".".join(self.scope) or "<module>", name)] += 1
            self.generic_visit(node)

    Visitor().visit(ast.parse(source))
    return result


def production_census():
    return {
        (path.relative_to(ROOT).as_posix(), symbol, call): count
        for path in sorted((ROOT / "tldw_chatbook").rglob("*.py"))
        for (symbol, call), count in census(path.read_text(encoding="utf-8")).items()
    }


def test_every_persistence_source_symbol_has_reviewed_owner_row():
    rows = {}
    for line in DOC.read_text().splitlines():
        if line.startswith("| tldw_chatbook/"):
            fields = [field.strip() for field in line.strip("|").split("|")]
            module, symbol, call, count, classification, cohort = fields
            assert classification in {
                "unsupported",
                "external_input",
                "server",
                "memory",
                "process_artifact",
                "disposable",
                "generic_boundary",
                "derived",
                "diagnostics",
                "old_backups",
            }
            assert cohort
            key = module, symbol, call
            assert key not in rows
            rows[key] = int(count)
    assert rows == production_census(), (
        "Persistence producer census changed; review owner, resolver, cohort and exclusion."
    )


def test_census_detects_new_producer_and_new_call_in_existing_symbol():
    before = census('def save():\n    target.write_text("one")\n')
    after = census(
        'def save():\n    target.write_text("one")\n    target.write_text("two")\ndef new():\n    os.open(path, flags)\n'
    )
    assert before != after
    assert after[("save", "write_text")] == 2
    assert after[("new", "open")] == 1


def test_sqlite_registry_members_have_explicit_census_rows():
    tree = ast.parse((ROOT / "tldw_chatbook/DB/private_sqlite.py").read_text())
    registry = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", "") == "_SQLITE_OWNER_POLICIES"
    )
    names = {key.value for key in registry.keys}
    documented = set(re.findall(r"^\| sqlite:([^ |]+) \|", DOC.read_text(), re.M))
    assert names == documented
