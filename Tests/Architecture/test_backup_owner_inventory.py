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

    tree = ast.parse(source)
    imported_candidates = {}
    # This is a conservative call census, not Python name/signature inference.
    # Gather every imported function alias before walking calls; an import in a
    # different scope can add candidates but can never overwrite/remove one.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_candidates.setdefault(alias.asname or alias.name, set()).add(
                    alias.name
                )

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.scope = []

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
            candidates = {name}
            if isinstance(node.func, ast.Name):
                candidates.update(imported_candidates.get(node.func.id, ()))
            # Even literal read-only opens remain candidates: parameters, local
            # assignments and other ordinary bindings can change their meaning.
            for candidate in candidates & PRODUCER_CALLS:
                result[(".".join(self.scope) or "<module>", candidate)] += 1
            self.generic_visit(node)

    Visitor().visit(tree)
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


def test_census_retains_unshadowed_read_only_opens_as_conservative_candidates():
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
        assert census(source)[("<module>", "open")] == 1, source


def test_census_retains_bare_opens_under_normal_python_bindings():
    examples = (
        'def save(open):\n    open("r")',
        'open = writer\nopen("r")',
        'def open(mode): pass\nopen("r")',
        '(lambda open: open("r"))(writer)',
        'for open in writers:\n    open("r")',
        'with writer_context() as open:\n    open("r")',
        'try: pass\nexcept Writer as open:\n    open("r")',
        '[open("r") for open in writers]',
        '(open := writer)\nopen("r")',
        'match writer:\n    case {"writer": open}:\n        open("r")',
    )
    for source in examples:
        assert (
            sum(count for (_, call), count in census(source).items() if call == "open")
            == 1
        ), source


def test_function_local_import_cannot_hide_module_open_or_imported_alias():
    examples = (
        'import custom as helper\ndef unrelated():\n    import io as helper\nhelper.open("store.bin", "r")',
        'from custom import open as writer\ndef unrelated():\n    from json import load as writer\nwriter("r")',
        'from io import open as writer\ndef unrelated():\n    from custom import action as writer\nwriter("store.bin", "wb")',
    )
    for source in examples:
        assert census(source)[("<module>", "open")] == 1, source
