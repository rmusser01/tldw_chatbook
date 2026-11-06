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
        # yaml serializers (TASK-39053 review: EvalConfigLoader.save).
        "safe_dump",
        "safe_dump_all",
        "dump_all",
        "mkdir",
        "makedirs",
        "copy2",
        "copyfile",
        "copytree",
        "atomic_private_write_text",
        "atomic_private_write_bytes",
        # Utils/atomic_file_ops writers (TASK-39053: callers that moved to them
        # had silently left the census).
        "atomic_write_text",
        "atomic_write_bytes",
        "atomic_write_json",
        "atomic_copy",
        "_publish_activation_record",
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
        "renameatx_np",
        "os.replace",
        "os.unlink",
        "os.rename",
        "os.link",
    }
)


# os-module publication seams counted as `os.<name>` (rename/link: TASK-39053 review).
NATIVE_OS_CALLS = frozenset({"replace", "unlink", "rename", "link"})


def census(source: str) -> Counter:
    result = Counter()

    tree = ast.parse(source)
    imported_candidates = {}
    # Assignment aliases of native seams are lexical: keyed by the enclosing
    # function node (None = module), visible to that function and its nested ones.
    assigned_candidates = {}
    # `import os` and any `from <shim> import os` (Utils/platform_files) bind os.
    os_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
        if alias.name == "os"
    }

    def native_candidate(value):
        if isinstance(value, ast.Attribute):
            if value.attr == "renameatx_np":
                return "renameatx_np"
            if (
                value.attr in NATIVE_OS_CALLS
                and getattr(value.value, "id", None) in os_names
            ):
                return f"os.{value.attr}"
        return None

    # This is a conservative call census, not Python name/signature inference.
    # Gather every imported function alias before walking calls; an import in a
    # different scope can add candidates but can never overwrite/remove one.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_candidates.setdefault(alias.asname or alias.name, set()).add(
                    f"os.{alias.name}"
                    if node.module == "os" and alias.name in NATIVE_OS_CALLS
                    else alias.name
                )

    def collect_assignments(node, owner):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                collect_assignments(child, child)
                continue
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    # `fn, flags = libc.renameatx_np, 4` pairs element-wise.
                    pairs = (
                        zip(target.elts, child.value.elts)
                        if isinstance(target, ast.Tuple)
                        and isinstance(child.value, ast.Tuple)
                        and len(target.elts) == len(child.value.elts)
                        else ((target, child.value),)
                    )
                    for name, value in pairs:
                        if isinstance(name, ast.Name) and (
                            candidate := native_candidate(value)
                        ):
                            assigned_candidates.setdefault(owner, {}).setdefault(
                                name.id, set()
                            ).add(candidate)
            collect_assignments(child, owner)

    collect_assignments(tree, None)

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.scope = []
            self.functions = [None]

        def candidates_for(self, value, argument=False):
            """Producer names an expression can denote, as a callee or a callable argument.

            A bare name passed as an argument counts only through an imported or
            assigned binding: a parameter called ``open`` or ``write`` is a value.
            """
            name = (
                value.attr
                if isinstance(value, ast.Attribute)
                else getattr(value, "id", "")
            )
            found = set() if argument and isinstance(value, ast.Name) else {name}
            if candidate := native_candidate(value):
                found.add(candidate)
            if isinstance(value, ast.Name):
                found.update(imported_candidates.get(value.id, ()))
                for owner in self.functions:
                    found.update(assigned_candidates.get(owner, {}).get(value.id, ()))
            return found & PRODUCER_CALLS

        def visit_ClassDef(self, node):
            self.scope.append(node.name)
            for base in node.bases:
                if getattr(base, "id", "") == "BaseDB":
                    result[(".".join(self.scope), "inherits:BaseDB")] += 1
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node):
            self.scope.append(node.name)
            self.functions.append(node)
            self.generic_visit(node)
            self.functions.pop()
            self.scope.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, node):
            self.functions.append(node)
            self.generic_visit(node)
            self.functions.pop()

        def visit_Call(self, node):
            symbol = ".".join(self.scope) or "<module>"
            # Even literal read-only opens remain candidates: parameters, local
            # assignments and other ordinary bindings can change their meaning.
            for candidate in self.candidates_for(node.func):
                result[(symbol, candidate)] += 1
            # A producer passed by reference runs elsewhere, for example
            # `asyncio.to_thread(path.write_bytes, data)` (TASK-39053 review).
            # `x.open` / `x.connect` values are not counted here: in this tree they
            # are widget state (`header.open`) or `webbrowser.open`, and a call
            # through them is still counted where it happens.
            for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
                if isinstance(argument, (ast.Attribute, ast.Name)):
                    for candidate in self.candidates_for(argument, argument=True):
                        if candidate not in {"open", "connect"}:
                            result[(symbol, candidate)] += 1
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
                "qualified",
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
    production = production_census()
    added = sorted(production.keys() - rows.keys())
    gone = sorted(rows.keys() - production.keys())
    changed = sorted(k for k in production.keys() & rows.keys() if rows[k] != production[k])
    if added or gone or changed:
        # Raised, not asserted: pytest truncates assertion explanations, and the
        # reviewer needs every row (TASK-39053: a truncated diff hid 850 rows).
        listing = [
            f"Persistence producer census changed: {len(added)} added, {len(gone)} gone, "
            f"{len(changed)} changed. Review owner, resolver, cohort and exclusion.",
            *(f"added   | {m} | {s} | {c} | {production[m, s, c]} |" for m, s, c in added),
            *(f"gone    | {m} | {s} | {c} | {rows[m, s, c]} |" for m, s, c in gone),
            *(
                f"changed | {m} | {s} | {c} | {rows[m, s, c]} -> {production[m, s, c]} |"
                for m, s, c in changed
            ),
        ]
        raise AssertionError("\n".join(listing))


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
    if names != documented:
        raise AssertionError(
            f"SQLite registry members without a row: {sorted(names - documented)}\n"
            f"rows without a registry member: {sorted(documented - names)}"
        )


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


def test_census_tracks_native_publication_callable_alias_and_registry_replace():
    source = """
import ctypes
import os

def publish():
    libc = ctypes.CDLL(None)
    native = libc.renameatx_np
    native(source_fd, source_name, target_fd, target_name, 4)

def register():
    os.replace(stage, registry, src_dir_fd=parent, dst_dir_fd=parent)
"""
    assert census(source)[("publish", "renameatx_np")] == 1
    assert census(source)[("register", "os.replace")] == 1


def test_census_tracks_native_callable_bound_by_tuple_assignment():
    # TASK-39053: Backup_Recovery/native_platform.rename_noreplace binds the
    # native no-replace publication as `fn, flags = libc.renameatx_np, 0x4`.
    source = """
import ctypes

def publish():
    libc = ctypes.CDLL(None)
    fn, flags = libc.renameatx_np, 0x4
    fn(source_fd, source_name, target_fd, target_name, flags)
"""
    assert census(source)[("publish", "renameatx_np")] == 1


def test_census_native_alias_assignment_is_lexically_scoped():
    # TASK-39053 review: `fn` bound to renameatx_np in one function must not turn
    # an unrelated `fn = libc.fstatfs; fn(...)` elsewhere into a producer.
    source = """
import ctypes

native = ctypes.CDLL(None).renameatx_np

def publish():
    fn, flags = libc.renameatx_np, 0x4
    def retry():
        fn(source_fd, source_name, target_fd, target_name, flags)
    fn(source_fd, source_name, target_fd, target_name, flags)

def identity():
    fn = libc.fstatfs
    fn(fd, data)

def module_alias():
    native(source_fd, source_name, target_fd, target_name, 4)
"""
    result = census(source)
    assert result[("publish", "renameatx_np")] == 1
    assert result[("publish.retry", "renameatx_np")] == 1
    assert result[("identity", "renameatx_np")] == 0
    assert result[("module_alias", "renameatx_np")] == 1


def test_census_counts_producer_callables_passed_by_reference():
    # TASK-39053 review: the Personas exports write through asyncio.to_thread.
    source = """
import asyncio
from tldw_chatbook.Utils.atomic_file_ops import atomic_write_text

async def export(path, data, archive):
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(path.write_bytes, data)
    drain_thread(archive.write_bytes)
    await loop.run_in_executor(None, atomic_write_text, path, "x")
    submit(callback=path.write_text)
"""
    result = census(source)
    assert result[("export", "mkdir")] == 1
    assert result[("export", "write_bytes")] == 2
    assert result[("export", "atomic_write_text")] == 1
    assert result[("export", "write_text")] == 1


def test_census_counts_yaml_safe_dump_serializers():
    # TASK-39053 review: EvalConfigLoader.save writes through yaml.safe_dump into
    # a stream from a non-vocabulary opener.
    source = """
import yaml
yaml.safe_dump(snapshot, stream)
yaml.safe_dump_all(documents, stream)
yaml.dump_all(documents, stream)
"""
    result = census(source)
    assert result[("<module>", "safe_dump")] == 1
    assert result[("<module>", "safe_dump_all")] == 1
    assert result[("<module>", "dump_all")] == 1


def test_census_native_replace_is_distinct_from_string_replace():
    assert census('value.replace("a", "b")') == Counter()
    source = """
import os as filesystem
from os import replace as commit_record
filesystem.replace(stage, destination)
commit_record(stage, destination)
"""
    assert census(source)[("<module>", "os.replace")] == 2


def test_census_tracks_os_bound_by_from_import_shim():
    # TASK-39053: 60+ modules switched to `from ...platform_files import os`;
    # their os.replace/os.unlink producers silently left the census.
    source = """
from tldw_chatbook.Utils.platform_files import os
from tldw_chatbook.Utils.platform_files import os as native
os.replace(stage, destination)
native.unlink(stage)
"""
    assert census(source)[("<module>", "os.replace")] == 1
    assert census(source)[("<module>", "os.unlink")] == 1


def test_census_tracks_native_rename_and_link_publication():
    # TASK-39053 review: os.rename/os.link publish like os.replace (snapshot
    # store moves, Model_Artifacts promotion, backup crypto transform).
    source = """
import os as filesystem
from tldw_chatbook.Utils.platform_files import os
from os import rename as move_record
filesystem.rename(stage, destination)
os.link(stage, destination)
move_record(stage, destination)
profile.rename(name)
"""
    result = census(source)
    assert result[("<module>", "os.rename")] == 2
    assert result[("<module>", "os.link")] == 1


def test_census_records_native_intent_removal_boundary():
    source = """
import os as filesystem
from os import unlink as remove_intent
filesystem.unlink("registry.pending.json", dir_fd=parent)
remove_intent("registry.pending.json", dir_fd=parent)
"""
    assert census(source)[("<module>", "os.unlink")] == 2
