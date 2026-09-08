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

        def visit_Import(self, node):
            for alias in node.names:
                bound_name = alias.asname or alias.name.split(".")[0]
                self.imports[bound_name] = alias.name if alias.asname else bound_name

        def visit_ImportFrom(self, node):
            for alias in node.names:
                module = "." * node.level + (node.module or "")
                self.imports[alias.asname or alias.name] = f"{module}.{alias.name}"

        def call_identity(self, node):
            if isinstance(node, ast.Name):
                return self.imports.get(
                    node.id, "builtins.open" if node.id == "open" else None
                )
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Call):
                    factory = self.call_identity(node.value.func)
                    if factory in {
                        "pathlib.Path",
                        "pathlib.PosixPath",
                        "pathlib.WindowsPath",
                    }:
                        return f"{factory}().{node.attr}"
                parent = self.call_identity(node.value)
                if parent is not None:
                    return f"{parent}.{node.attr}"
            return None

        def is_known_read_only_open(self, node, identity):
            # Interpret positional modes only after establishing the signature.
            # Unknown receivers may take a filename first; os.open takes flags.
            if identity in {"builtins.open", "io.open", "_io.open"}:
                mode_index = 1
            elif identity in {
                "pathlib.Path().open",
                "pathlib.PosixPath().open",
                "pathlib.WindowsPath().open",
            }:
                mode_index = 0
            else:
                return False
            if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
                keyword.arg is None for keyword in node.keywords
            ):
                return False
            mode = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "mode"),
                node.args[mode_index]
                if len(node.args) > mode_index
                else ast.Constant("r"),
            )
            return isinstance(mode, ast.Constant) and mode.value in {
                "r",
                "rb",
                "rt",
                "br",
                "tr",
            }

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
            identity = self.call_identity(node.func)
            canonical_name = identity.rsplit(".", 1)[-1] if identity else name
            name = "open" if "open" in {name, canonical_name} else canonical_name
            if name == "open" and self.is_known_read_only_open(node, identity):
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


def test_census_detects_module_qualified_and_aliased_writable_opens():
    examples = (
        'open("store.bin", "wb").close()',
        'import io\nio.open("store.bin", "wb").close()',
        'import io as streams\nstreams.open("store.bin", "wb").close()',
        'import os as native\nnative.open("store.bin", flags)',
        'from os import open as raw_open\nraw_open("store.bin", flags)',
        'from io import open as stream_open\nstream_open("store.bin", "wb")',
        'import builtins as native\nnative.open("store.bin", "wb")',
        'from pathlib import Path as P\nP("store.bin").open("wb")',
    )
    for source in examples:
        assert census(source)[("<module>", "open")] == 1, source


def test_census_keeps_ambiguous_open_signatures_and_dynamic_arguments():
    examples = (
        'unknown.open("store.bin", "wb")',
        'unknown.open("store.bin")',
        'unknown.open("r")',
        'from custom_store import open as custom_open\ncustom_open("r")',
        "open(*arguments)",
        'open("store.bin", **options)',
        'import io\nio.open("store.bin", mode=selected_mode)',
    )
    for source in examples:
        assert census(source)[("<module>", "open")] == 1, source


def test_census_excludes_only_known_read_only_open_signatures():
    examples = (
        'open("store.bin")',
        'open("store.bin", "rb")',
        'import io\nio.open("store.bin", "rb")',
        'import io as streams\nstreams.open("store.bin", mode="r")',
        'from builtins import open as reader\nreader("store.bin", "rb")',
        'from pathlib import Path as P\nP("store.bin").open("rb")',
        'import pathlib as paths\npaths.Path("store.bin").open()',
    )
    for source in examples:
        assert census(source)[("<module>", "open")] == 0, source
