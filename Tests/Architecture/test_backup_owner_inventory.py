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
        "renameat2",
        # tempfile creation (TASK-39111): 22 producing symbols had no other row.
        "NamedTemporaryFile",
        "TemporaryFile",
        "SpooledTemporaryFile",
        "TemporaryDirectory",
        "mkstemp",
        "mkdtemp",
        "os.replace",
        "os.replace_open_target",
        "os.unlink",
        "os.rename",
        "os.link",
    }
)


# Calls whose arguments are inspected, never run (TASK-39053 review r11b-rev2).
INTROSPECTION_CALLS = frozenset({"signature", "isinstance", "issubclass", "callable"})

# os-module publication seams counted as `os.<name>` (rename/link: TASK-39053 review;
# replace_open_target is the Windows `platform_files` shim's, TASK-39112).
NATIVE_OS_CALLS = frozenset(
    {"replace", "unlink", "rename", "link", "replace_open_target"}
)

# libc no-replace renames: macOS and Linux (renameat2: TASK-39115).
NATIVE_LIBC_CALLS = frozenset({"renameatx_np", "renameat2"})


def census(source: str) -> Counter:
    result = Counter()

    tree = ast.parse(source)
    imported_candidates = {}
    # Assignment aliases are lexical: keyed by the enclosing function node
    # (None = module), visible to that function and its nested ones. They resolve
    # in source order (ponytail: an alias a nested function assigns from an outer
    # alias bound later in the source is missed; iterate to a fixed point if that
    # form appears).
    assigned_candidates = {}
    os_aliases = {}  # names bound to `os` by a helper call (TASK-39112)
    string_aliases = {}  # names bound to string constants, for getattr (TASK-39115)
    # `import os` and any `from <shim> import os` (Utils/platform_files) bind os.
    os_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
        if alias.name == "os"
    }
    # A function that returns an os binding, as Tools' `_write_file_os()` does.
    # The text check skips the per-function walk in modules with no such return.
    returns_os = os_names and re.search(
        r"\breturn\s+\(?\s*(?:%s)\b" % "|".join(map(re.escape, os_names)), source
    )
    os_helpers = {
        node.name
        for node in (ast.walk(tree) if returns_os else ())
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(child, ast.Return)
            and getattr(child.value, "id", None) in os_names
            for child in ast.walk(node)
        )
    }

    # pydub writers (TASK-39111 AC#2): `.export` counts as `pydub.export` only in
    # a module that imports pydub, unless its target is a BytesIO (bound in scope
    # or built inline), which writes no file.
    imports_pydub = "pydub" in source and any(
        name.split(".")[0] == "pydub"
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for name in (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
        )
    )
    buffers = {}  # names bound to BytesIO(...), per lexical scope

    def is_bytes_io(value):
        return isinstance(value, ast.Call) and "BytesIO" in (
            getattr(value.func, "id", None),
            getattr(value.func, "attr", None),
        )

    def scoped(table, owners, name):
        return {
            value for owner in owners for value in table.get(owner, {}).get(name, ())
        }

    def strings_for(value, owners):
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return {value.value}
        if isinstance(value, ast.IfExp):
            return strings_for(value.body, owners) | strings_for(value.orelse, owners)
        if isinstance(value, ast.Name):
            return scoped(string_aliases, owners, value.id)
        return set()

    def native_candidates(value, owners):
        """Native publication names an expression denotes in a lexical scope."""
        if isinstance(value, ast.IfExp):
            return native_candidates(value.body, owners) | native_candidates(
                value.orelse, owners
            )
        # `getattr(libc, name)` is `libc.<name>` for each string `name` can hold.
        if (
            isinstance(value, ast.Call)
            and getattr(value.func, "id", None) == "getattr"
            and len(value.args) >= 2
        ):
            return {
                found
                for attr in strings_for(value.args[1], owners)
                for found in native_candidates(
                    ast.Attribute(value=value.args[0], attr=attr, ctx=ast.Load()),
                    owners,
                )
            }
        if not isinstance(value, ast.Attribute):
            return set()
        if value.attr in NATIVE_LIBC_CALLS:
            return {value.attr}
        base = getattr(value.value, "id", None)
        if value.attr in NATIVE_OS_CALLS and (
            base in os_names
            or base in {name for owner in owners for name in os_aliases.get(owner, ())}
        ):
            return {f"os.{value.attr}"}
        return set()

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

    def collect_assignments(node, owners):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                collect_assignments(child, (*owners, child))
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
                        if not isinstance(name, ast.Name):
                            continue
                        if candidates := native_candidates(value, owners):
                            assigned_candidates.setdefault(owners[-1], {}).setdefault(
                                name.id, set()
                            ).update(candidates)
                        if strings := strings_for(value, owners):
                            string_aliases.setdefault(owners[-1], {}).setdefault(
                                name.id, set()
                            ).update(strings)
                        if isinstance(value, ast.Call) and (
                            getattr(value.func, "id", None) in os_helpers
                        ):
                            os_aliases.setdefault(owners[-1], set()).add(name.id)
                        if imports_pydub and is_bytes_io(value):
                            buffers.setdefault(owners[-1], {})[name.id] = {"BytesIO"}
            collect_assignments(child, owners)

    collect_assignments(tree, (None,))

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
            found |= native_candidates(value, self.functions)
            if isinstance(value, ast.Name):
                found.update(imported_candidates.get(value.id, ()))
                found |= scoped(assigned_candidates, self.functions, value.id)
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
            if imports_pydub and getattr(node.func, "attr", None) == "export":
                target = next(
                    iter(
                        node.args[:1]
                        or [k.value for k in node.keywords if k.arg == "out_f"]
                    ),
                    None,
                )
                if not (
                    is_bytes_io(target)
                    or isinstance(target, ast.Name)
                    and scoped(buffers, self.functions, target.id)
                ):
                    result[(symbol, "pydub.export")] += 1
            # A producer passed by reference runs elsewhere, for example
            # `asyncio.to_thread(path.write_bytes, data)` (TASK-39053 review).
            # `x.open` / `x.connect` values are not counted here: in this tree they
            # are widget state (`header.open`) or `webbrowser.open`, and a call
            # through them is still counted where it happens.
            # Introspection runs nothing: inspect.signature(os.replace),
            # isinstance(x, zipfile.ZipFile), callable(stream.write).
            callee = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", "")
            )
            arguments = (
                ()
                if callee in INTROSPECTION_CALLS
                else (*node.args, *(keyword.value for keyword in node.keywords))
            )
            for argument in arguments:
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
    # TASK-39113: each row's module and target-kind columns, not just its name.
    policies = {
        key.value: (value.args[0].value, value.args[1].id)
        for key, value in zip(registry.keys, registry.values)
    }
    rows = {
        owner: (module, kinds)
        for owner, module, kinds in re.findall(
            r"^\| sqlite:([^ |]+) \| ([^ |]+) \| ([^ |]+) \|",
            DOC.read_text(),
            re.MULTILINE,
        )
    }
    assert {
        owner: (rows.get(owner), policy)
        for owner, policy in policies.items()
        if rows.get(owner) != policy
    } == {}


def test_personal_context_stores_preview_under_their_installed_owners(tmp_path):
    """TASK-39110: both Personal Context databases previewed as owner
    ``unknown``. The interview drafts (and sidecars) are intentionally
    excluded and refuse capture; the repository stays an explicit
    ``unsupported`` row until its capture lands (owner decision 2026-11-06)."""
    from threading import Event

    import pytest

    from tldw_chatbook.Backup_Recovery.inventory import discover
    from tldw_chatbook.Backup_Recovery.owner_registry import (
        install_adapters,
        registered,
    )

    config = tmp_path / "selected.toml"
    config.write_text(
        f'[general]\nusers_name = "Ada"\n[paths]\ndata_dir = "{tmp_path / "data"}"\n'
    )
    root = tmp_path / "data" / "Ada"
    root.mkdir(parents=True)
    suffixes = ("", "-wal", "-shm", "-journal")
    drafts = root / "tldw_chatbook_personal_context_interviews.db"
    repository = root / "tldw_chatbook_personal_context.db"
    for database in (drafts, repository):
        for suffix in suffixes:
            Path(str(database) + suffix).write_bytes(b"private")
    install_adapters()
    items = {
        item.path: item for item in discover((config,)).items if item.path is not None
    }

    def rows(database):
        return [
            (items[path].owner, items[path].status)
            for path in (Path(str(database) + suffix) for suffix in suffixes)
        ]

    assert (rows(drafts), rows(repository)) == (
        [("personal_context.interview_drafts", "intentionally_excluded")] * 4,
        [("personal_context.repository", "unsupported")] * 4,
    )
    owners = {adapter.owner_id: adapter for adapter in registered()}
    for owner, database in (
        ("personal_context.interview_drafts", drafts),
        ("personal_context.repository", repository),
    ):
        with pytest.raises(ValueError, match="personal_context"):
            owners[owner].capture(items[database], tmp_path / "archive", Event())
    assert not (tmp_path / "archive").exists()


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


def test_census_ignores_arguments_to_introspection_calls():
    # TASK-39053 review r11b-rev2: a signature probe or type check runs nothing.
    source = """
import inspect, os, zipfile
inspect.signature(os.replace)
inspect.signature(os.link)
isinstance(archive, zipfile.ZipFile)
issubclass(kind, zipfile.ZipFile)
callable(stream.write)
"""
    assert census(source) == Counter()


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


def test_census_counts_tempfile_creation():
    # TASK-39111: tempfile creation produced files the review could not see.
    source = """
import tempfile
from tempfile import mkdtemp as make_scratch
tempfile.NamedTemporaryFile(delete=False)
tempfile.TemporaryFile()
tempfile.SpooledTemporaryFile()
tempfile.TemporaryDirectory()
tempfile.mkstemp()
make_scratch()
"""
    result = census(source)
    for name in (
        "NamedTemporaryFile",
        "TemporaryFile",
        "SpooledTemporaryFile",
        "TemporaryDirectory",
        "mkstemp",
        "mkdtemp",
    ):
        assert result[("<module>", name)] == 1, name


def test_census_tracks_os_returned_by_a_helper():
    # TASK-39112: Tools' `_atomic_write_target` publishes through
    # `file_os = _write_file_os()`, which returns `os` or the Windows shim.
    source = """
import os

def helper():
    if os.name == "nt":
        from tldw_chatbook.Utils.platform_files import os as native
        return native
    return os

def publish(a, b):
    f = helper()
    f.replace(a, b)
    f.link(a, b)
    f.unlink(a)
    commit = f.replace_open_target if windows else f.replace
    commit(a, b)

def unrelated(a):
    f = text()
    f.replace("a", "b")
"""
    result = census(source)
    assert result[("publish", "os.replace")] == 2
    assert result[("publish", "os.link")] == 1
    assert result[("publish", "os.unlink")] == 1
    assert result[("publish", "os.replace_open_target")] == 1
    assert not any(symbol == "unrelated" for symbol, _ in result)


def test_census_counts_linux_renameat2_like_renameatx_np():
    # TASK-39115: direct, tuple-aliased and getattr-bound native no-replace renames.
    source = """
import ctypes

def direct():
    _LIBC.renameat2(a, b, c, d, 1)

def tuple_alias():
    fn, flags = libc.renameat2, 1
    fn(a, b, c, d, flags)

def by_name(darwin):
    libc = ctypes.CDLL(None)
    function_name = "renameatx_np" if darwin else "renameat2"
    function = getattr(libc, function_name, None)
    function(a, b, c, d, 0)

def literal():
    getattr(libc, "renameat2")(a, b, c, d, 1)

def probe():
    return getattr(ctypes.CDLL(None), "renameat2", None) is not None
"""
    result = census(source)
    assert result[("direct", "renameat2")] == 1
    assert result[("tuple_alias", "renameat2")] == 1
    assert result[("by_name", "renameat2")] == 1
    assert result[("by_name", "renameatx_np")] == 1
    assert result[("literal", "renameat2")] == 1
    assert not any(symbol == "probe" for symbol, _ in result)


def test_census_counts_pydub_export_only_where_pydub_is_imported():
    # TASK-39111 AC#2 (owner decision 2026-11-10): pydub `export` is counted in
    # modules that import pydub; an export into a BytesIO writes no file.
    importing = """
import io
from pydub import AudioSegment

def save(segment, path):
    segment.export(path, format="mp4")
    segment.export(out_f=path)

def encode(segment):
    buffer = io.BytesIO()
    segment.export(buffer, format="wav")
    segment.export(io.BytesIO())
"""
    result = census(importing)
    assert result[("save", "pydub.export")] == 2
    assert not any(symbol == "encode" for symbol, _ in result)
    assert census("def save(exporter, path):\n    exporter.export(path)\n") == Counter()


def test_actor_pack_persona_images_preview_and_validate_with_persona_artwork(
    tmp_path,
):
    """TASK-39116: activating an Actor Pack publishes its persona images under
    ``<user data>/actor_packs/<digest>/persona-runtime/`` and points ChaChaNotes
    ``persona_visual_assets`` rows at them. They previewed as owner ``unknown``,
    and ``persona.assets`` refused those rows. Owner decision 2026-11-10: capture
    them with the persona artwork group."""
    from Tests.Persona_Visual.test_persona_visual_publication import _snapshot
    from tldw_chatbook.Backup_Recovery.data_groups import group_for_owner
    from tldw_chatbook.Backup_Recovery.inventory import discover
    from tldw_chatbook.Backup_Recovery.owner_registry import (
        install_adapters,
        registered,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Persona_Visual.publication import publish_persona_visual
    from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository

    config = tmp_path / "selected.toml"
    config.write_text(
        f'[general]\nusers_name = "Ada"\n[paths]\ndata_dir = "{tmp_path / "data"}"\n'
    )
    root = tmp_path / "data" / "Ada"
    root.mkdir(parents=True)
    source = tmp_path / "source"
    source.mkdir()
    runtime = root / "actor_packs" / ("a" * 64) / "persona-runtime"
    install_adapters()

    def core_edges():
        core = next(
            item
            for item in discover((config,)).items
            if item.logical_id.endswith(":db.chachanotes.primary")
        )
        return {
            owner
            for owner in ("persona.assets", "persona.actor_packs")
            if core.logical_id.replace("db.chachanotes.primary", owner)
            in core.dependencies
        }

    db = CharactersRAGDB(root / "tldw_chatbook_ChaChaNotes.db", "recovery")
    try:
        publish_persona_visual(
            PersonaVisualRepository(db),
            _snapshot(source),
            source_root=source,
            profile_root=root,
            authority_guard=lambda: True,
        )
        # The core depends only on the owner whose rows exist: an absent
        # actor_packs/ folder would otherwise be a dependency_unavailable.
        assert core_edges() == {"persona.assets"}
        # The activation's stored shape: the asset rows name actor_packs/ files.
        runtime.mkdir(parents=True)
        connection = db.get_connection()
        for row_id, relpath in connection.execute(
            "SELECT id, storage_relpath FROM persona_visual_assets"
        ).fetchall():
            moved = runtime.relative_to(root) / f"asset-{row_id:04d}.png"
            (root / relpath).rename(root / moved)
            connection.execute(
                "UPDATE persona_visual_assets SET storage_relpath = ? WHERE id = ?",
                (moved.as_posix(), row_id),
            )
        connection.commit()
    finally:
        db.close()
    (runtime / "manifest.json").write_text("{}")
    asset = next(runtime.glob("asset-*.png"))
    assert core_edges() == {"persona.actor_packs"}

    def preview():
        items = discover((config,)).items
        return items, {
            item.path: item
            for item in items
            if item.path is not None and item.path.is_relative_to(root / "actor_packs")
        }

    items, under = preview()
    assert {item.owner for item in under.values()} == {"persona.actor_packs"}
    assert (under[asset].status, under[runtime / "manifest.json"].status) == (
        "included",
        "included",
    )
    # persona.assets no longer refuses the actor_packs/ rows as unreadable.
    assert [
        item.logical_id
        for item in items
        if item.owner == "persona.assets" and item.path is None
    ] == []
    assert group_for_owner("persona.actor_packs") == group_for_owner("persona.assets")
    adapters = {adapter.owner_id: adapter for adapter in registered()}
    available = {item.logical_id: item.path for item in items if item.path is not None}
    tops = {
        owner: next(item for item in items if item.logical_id.endswith(":" + owner))
        for owner in ("persona.actor_packs", "persona.assets", "db.chachanotes.primary")
    }
    assert under[asset].logical_id in tops["persona.actor_packs"].dependencies
    assert tops["persona.actor_packs"].logical_id in (
        tops["db.chachanotes.primary"].dependencies
    )
    for owner, top in tops.items():
        candidates = {key: available[key] for key in top.dependencies}
        assert adapters[owner].validate_dependencies(top, top.path, candidates) == ()
    # A restore validates the archived tree topology the same way.
    topology = {
        item.logical_id: (
            item.metadata.root_id,
            item.metadata.parent_id,
            item.metadata.relative_path,
            item.metadata.kind,
        )
        for item in items
        if item.metadata is not None
    }
    for owner in ("persona.actor_packs", "persona.assets"):
        top = tops[owner]
        assert (
            adapters[owner].validate_restore_dependencies(
                top, top.path, available, topology=topology
            )
            == ()
        )
    # The row's recorded digest is checked against the actor_packs/ file.
    asset.write_bytes(b"changed")
    assert preview()[1][asset].status == "unsupported"
