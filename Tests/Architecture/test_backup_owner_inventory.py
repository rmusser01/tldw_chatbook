"""Exact source-symbol census: new persistence calls require ownership review."""

import ast
from collections import Counter
from pathlib import Path
import re

import pytest

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

# Statements and expressions that bind a name to a value (review r13b-rev1).
BINDINGS = (ast.Assign, ast.AnnAssign, ast.NamedExpr)


def census(source: str) -> Counter:
    result = Counter()

    tree = ast.parse(source)
    imported_candidates = {}
    # Assignment aliases are lexical: keyed by the enclosing function node
    # (None = module), visible to that function and its nested ones. They resolve
    # in source order. ponytail: not resolved, so these os helper forms are missed:
    # an alias a nested function assigns from an outer alias bound later in the
    # source, a returned local alias (`m = os; return m`), a helper imported from
    # another module, and `os` held in an attribute or container (`self.os`).
    # Iterate to a fixed point or add a binding pass if one of them appears.
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

    def callee(call):
        """A called function's or method's name: `h()` and `self.h()` are `h`."""
        return getattr(call.func, "id", None) or getattr(call.func, "attr", None)

    # A function that returns an os binding, as Tools' `_write_file_os()` does, or
    # returns another helper's call (`return base()`, `return self._os()`). The
    # text check skips the per-function walk in modules with no such return.
    returns_os = os_names and re.search(
        r"\breturn\W*(?:%s)\b" % "|".join(map(re.escape, os_names)), source
    )
    # Same-named defs share one entry: any of them returning os makes the name a
    # helper (review r13b-rev2: a later `def base(self): pass` erased it).
    returned = {}
    for node in ast.walk(tree) if returns_os else ():
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            returned.setdefault(node.name, []).extend(
                child.value
                for child in ast.walk(node)
                if isinstance(child, ast.Return) and child.value is not None
            )
    os_helpers = set()
    while found := {
        name
        for name, values in returned.items()
        if name not in os_helpers
        and any(
            getattr(value, "id", None) in os_names
            or isinstance(value, ast.Call)
            and callee(value) in os_helpers
            for value in values
        )
    }:
        os_helpers |= found

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
    # name -> True when every binding of it in that scope is `= BytesIO(...)`.
    # Read in the innermost scope only, so a parameter is a file target, and any
    # other binding (an assignment or walrus to anything else, a loop, with-as,
    # except-as, unpacking or comprehension target) makes it one too. ponytail: match-case captures, imports
    # and global/nonlocal rebinding are not seen; add them if a writer uses one.
    buffers = {}
    bytes_io_targets = set()  # the Name nodes of `x = BytesIO(...)`

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
            # `_write_file_os().replace(...)`, a helper's call used in place.
            or os_helpers
            and isinstance(value.value, ast.Call)
            and callee(value.value) in os_helpers
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
            if imports_pydub:
                stored = (
                    child.id
                    if isinstance(child, ast.Name)
                    and isinstance(child.ctx, ast.Store)
                    and child not in bytes_io_targets
                    else child.name
                    if isinstance(child, ast.ExceptHandler)
                    else None
                )
                if stored:
                    buffers.setdefault(owners[-1], {})[stored] = False
            # `x = v`, `x: T = v` and `(x := v)` bind alike.
            if isinstance(child, BINDINGS) and child.value:
                bound = child.value
                targets = (
                    child.targets if isinstance(child, ast.Assign) else (child.target,)
                )
            else:
                targets = ()
            for target in targets:
                # `fn, flags = libc.renameatx_np, 4` pairs element-wise.
                pairs = (
                    zip(target.elts, bound.elts, strict=True)
                    if isinstance(target, ast.Tuple)
                    and isinstance(bound, ast.Tuple)
                    and len(target.elts) == len(bound.elts)
                    else ((target, bound),)
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
                    if (
                        os_helpers
                        and isinstance(value, ast.Call)
                        and callee(value) in os_helpers
                    ):
                        os_aliases.setdefault(owners[-1], set()).add(name.id)
                    if imports_pydub and is_bytes_io(value):
                        bytes_io_targets.add(name)
                        buffers.setdefault(owners[-1], {}).setdefault(name.id, True)
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
                    and buffers.get(self.functions[-1], {}).get(target.id, False)
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
    changed = sorted(
        k for k in production.keys() & rows.keys() if rows[k] != production[k]
    )
    if added or gone or changed:
        # Raised, not asserted: pytest truncates assertion explanations, and the
        # reviewer needs every row (TASK-39053: a truncated diff hid 850 rows).
        listing = [
            f"Persistence producer census changed: {len(added)} added, {len(gone)} gone, "
            f"{len(changed)} changed. Review owner, resolver, cohort and exclusion.",
            *(
                f"added   | {m} | {s} | {c} | {production[m, s, c]} |"
                for m, s, c in added
            ),
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
    """TASK-39110 / TASK-40134: the interview drafts (and sidecars) are
    intentionally excluded and refuse capture. The repository, with its live
    ``-wal``, previews as captured in the selectable Personal Context group and
    no longer makes the preview incomplete."""
    from threading import Event

    import pytest

    from tldw_chatbook.Backup_Recovery.data_groups import (
        BACKUP_GROUPS,
        group_for_owner,
    )
    from tldw_chatbook.Backup_Recovery.inventory import BLOCKING, discover
    from tldw_chatbook.Backup_Recovery.owner_registry import (
        install_adapters,
        registered,
    )

    config, root = _personal_context_profile(tmp_path)
    suffixes = ("", "-wal", "-shm", "-journal")
    drafts = root / "tldw_chatbook_personal_context_interviews.db"
    for suffix in suffixes:
        Path(str(drafts) + suffix).write_bytes(b"private")
    repository = root / "tldw_chatbook_personal_context.db"
    with _wal_writer(repository):
        install_adapters()
        inventory = discover((config,))
    items = {item.path: item for item in inventory.items if item.path is not None}

    assert [
        (items[Path(str(drafts) + suffix)].owner, items[Path(str(drafts) + suffix)].status)
        for suffix in suffixes
    ] == [("personal_context.interview_drafts", "intentionally_excluded")] * 4
    assert (items[repository].owner, items[repository].status) == (
        "personal_context.repository",
        "included",
    )
    wal = items[Path(str(repository) + "-wal")]
    assert wal.status not in BLOCKING
    assert not [
        item
        for item in inventory.items
        if item.status in BLOCKING
        and (
            item.owner == "personal_context.repository"
            or item.path is not None
            and item.path.name.startswith(repository.name)
        )
    ]
    assert group_for_owner("personal_context.repository") == "personal_context"
    assert ("personal_context", "Personal Context") in {
        (group.group_id, group.label) for group in BACKUP_GROUPS
    }
    owners = {adapter.owner_id: adapter for adapter in registered()}
    with pytest.raises(ValueError, match="personal_context"):
        owners["personal_context.interview_drafts"].capture(
            items[drafts], tmp_path / "archive", Event()
        )
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
    # Review r13b-rev1: the text prefilter must not need a space after `return`.
    for spelling in ("return(os)", "return \\\n        os"):
        alone = (
            f"import os\ndef h():\n    {spelling}\n"
            "def use(a):\n    f = h()\n    f.unlink(a)\n"
        )
        assert census(alone)[("use", "os.unlink")] == 1, spelling
    assert result[("publish", "os.replace")] == 2
    assert result[("publish", "os.link")] == 1
    assert result[("publish", "os.unlink")] == 1
    assert result[("publish", "os.replace_open_target")] == 1
    assert not any(symbol == "unrelated" for symbol, _ in result)


def test_census_tracks_other_forms_of_an_os_helper():
    # TASK-39112 review r13b-rev1: a method helper, a helper that returns another
    # helper's call, a call used in place, and annotated or walrus bindings.
    source = """
import os

class Store:
    def _os(self):
        return os

    def publish(self, a, b):
        f = self._os()
        f.replace(a, b)

def base():
    return os

def chained():
    return base()

def in_place(a, b):
    base().replace(a, b)

def via_chain(a):
    f = chained()
    f.unlink(a)

def annotated(a, b):
    f: object = base()
    f.link(a, b)

def walrus(a, b):
    if (f := base()):
        f.rename(a, b)

class Later:
    def base(self):
        pass

class Other:
    def _os(self):
        pass
"""
    # Review r13b-rev2: a later def with the same name that returns nothing
    # (`Later.base`, `Other._os`) must not erase the helper.
    result = census(source)
    assert result[("Store.publish", "os.replace")] == 1
    assert result[("in_place", "os.replace")] == 1
    assert result[("via_chain", "os.unlink")] == 1
    assert result[("annotated", "os.link")] == 1
    assert result[("walrus", "os.rename")] == 1


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

shared = io.BytesIO()

def to_parameter(segment, shared):
    segment.export(shared)

def rebound(segment, path):
    out = io.BytesIO()
    out = path
    segment.export(out)

def looped(segment, paths):
    buf = io.BytesIO()
    for buf in paths:
        segment.export(buf)

def opened(segment, path):
    buf = io.BytesIO()
    with open(path, "wb") as buf:
        segment.export(buf)

def unpacked(segment, pair):
    buf = io.BytesIO()
    buf, other = pair
    segment.export(buf)

def excepted(segment):
    buf = io.BytesIO()
    try:
        pass
    except OSError as buf:
        segment.export(buf)
"""
    result = census(importing)
    assert result[("save", "pydub.export")] == 2
    # Review r13b-rev1: only the innermost scope's binding hides an export, and a
    # name rebound away from BytesIO is a file target again.
    assert result[("to_parameter", "pydub.export")] == 1
    assert result[("rebound", "pydub.export")] == 1
    # Review r13b-rev2: loop, with-as, unpacking and except-as bind it too.
    for symbol in ("looped", "opened", "unpacked", "excepted"):
        assert result[(symbol, "pydub.export")] == 1, symbol
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
    import sqlite3
    from contextlib import closing

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
    # Review r13b-rev1: a locator that is exactly "actor_packs" is not under
    # actor_packs/. persona.assets keeps it (and refuses it), as the core's
    # dependency query does, so preview and capture agree.
    with closing(sqlite3.connect(root / "tldw_chatbook_ChaChaNotes.db")) as raw:
        with raw:
            raw.execute(
                "UPDATE persona_visual_assets SET storage_relpath='actor_packs'"
            )
    assert core_edges() == {"persona.assets"}
    assert [
        item.logical_id.rsplit(":", 1)[1]
        for item in discover((config,)).items
        if item.owner == "persona.assets" and item.path is None
    ] == ["references_unavailable"]


def _actor_pack_activation_on_profile(
    tmp_path, monkeypatch, actor_kind="persona", *, bound=False
):
    """TASK-40304: an Actor Pack review on a configured file-backed profile.

    Activating a persona pack publishes into both captured trees:
    ``visual_identities/actor_packs/<digest>/`` (shared visual) and
    ``actor_packs/<digest>/persona-runtime/`` (persona runtime). A character
    pack publishes only the shared visual. ``bound`` builds the persona
    service the way the app does (``build_persona_service``).
    """
    from Tests.Actor_Packs.test_actor_pack_activation import (
        _persona_runtime_section,
        _shared_visual_section,
        _write_archive,
    )
    from Tests.Backup_Recovery.config_test_support import install_config_source
    from tldw_chatbook.Actor_Packs.activation import ActorPackActivationService
    from tldw_chatbook.Actor_Packs.importer import ActorPackImportService
    from tldw_chatbook.Actor_Packs.persona_coordinator import (
        PersonaActorPackCoordinator,
    )
    from tldw_chatbook.Actor_Packs.repository import ActorPackRepository
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Character_Chat.local_character_persona_service import (
        LocalCharacterPersonaService,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    root = tmp_path / "bootstrap"
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    (tmp_path / "data").mkdir(mode=0o700)
    selector = tmp_path / "config.toml"
    selector.write_text(
        f'[paths]\ndata_dir = "{tmp_path / "data"}"\n'
        f'[database]\nchachanotes_db_path = "{tmp_path / "chat.sqlite"}"\n'
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    config = install_config_source(monkeypatch)
    profile = config.get_user_data_dir()
    db = CharactersRAGDB(tmp_path / "chat.sqlite", "actor-pack-storage")
    repository = ActorPackRepository(db)
    if bound:
        from tldw_chatbook.Backup_Recovery.chat_source_participants import (
            build_persona_service,
        )

        local_service = build_persona_service(db)
    else:
        local_service = LocalCharacterPersonaService(
            db, persona_store_path=profile / "personas.json"
        )
    importer = ActorPackImportService(
        repository,
        staging_root=profile / "actor_pack_imports",
        profile_root=profile,
        local_service=local_service,
    )
    activation = ActorPackActivationService(
        db,
        local_service,
        repository,
        PersonaActorPackCoordinator(repository, local_service),
        importer,
    )
    sections = (_shared_visual_section(),)
    if actor_kind == "persona":
        sections += (_persona_runtime_section(),)
    review = importer.inspect_archive(
        _write_archive(
            tmp_path / f"{actor_kind}.tldw-actor-pack",
            actor_kind=actor_kind,
            portable_uuid="123e4567-e89b-42d3-a456-426614174000",
            sections=sections,
        )
    )
    trees = (
        profile / "visual_identities" / "actor_packs" / review.content_digest,
        profile / "actor_packs" / review.content_digest,
    )
    return db, activation, review, trees


def _published_files(trees):
    return sorted(
        str(path) for tree in trees if tree.exists() for path in tree.rglob("*")
        if path.is_file()
    )


def _activate_on_worker(activation, review, **kwargs):
    """Run activation as a finite worker: its own DB connection retires after."""
    import threading

    from tldw_chatbook.Backup_Recovery.participants import run_finite_local_worker

    outcome = []

    def run():
        try:
            outcome.append(
                run_finite_local_worker(
                    activation.activate, review, "create_new", **kwargs
                )
            )
        except BaseException as error:  # noqa: BLE001 - relayed to the test thread.
            outcome.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    return worker, outcome


def test_actor_pack_activation_refuses_while_a_capture_or_restore_holds_storage(
    tmp_path, monkeypatch
):
    """TASK-40304 AC1: the in-process capture/restore pause refuses a new
    activation with a named, path-free reason before it reads the store or
    writes anything."""
    from tldw_chatbook.Actor_Packs.activation import ActorPackActivationError
    from tldw_chatbook.Actor_Packs.import_controller import _error_category
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.UI.Screens.personas_screen import (
        _actor_pack_import_error_copy,
    )

    db, activation, review, trees = _actor_pack_activation_on_profile(
        tmp_path, monkeypatch
    )
    reads = []
    material = activation.importer._activation_material

    def observed_material(*args, **kwargs):
        reads.append(storage.local_pause_active())
        return material(*args, **kwargs)

    monkeypatch.setattr(activation.importer, "_activation_material", observed_material)
    try:
        pause = storage._begin_local_pause()
        try:
            with pytest.raises(ActorPackActivationError) as raised:
                activation.activate(review, "create_new")
            assert _published_files(trees) == []
        finally:
            pause.resume()
        # Refused by admission, not by a store read the pause turned away.
        assert reads == []
        assert raised.value.category == "actor_pack_import_storage_unavailable"
        assert _error_category(raised.value) == raised.value.category
        assert raised.value.cleanup_pending is False
        assert "Backup or restore" in _actor_pack_import_error_copy(
            raised.value.category
        )
        # AC2: the inventory's activation rows name the admission they run under.
        activation_rows = [
            line
            for line in DOC.read_text().splitlines()
            if line.startswith("| tldw_chatbook/Actor_Packs/activation.py |")
        ]
        assert len(activation_rows) == 6
        assert not [row for row in activation_rows if "no participant token" in row]
        # Nothing was committed either: the same review still activates after.
        result = activation.activate(review, "create_new")
        assert result.actor_kind == "persona"
        assert reads == [False]
        assert len(_published_files(trees)) == 3
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    ("actor_kind", "cancel", "category"),
    [
        ("persona", False, "actor_pack_import_storage_unavailable"),
        ("character", False, "actor_pack_import_storage_unavailable"),
        # A user's cancel keeps its own name while a pause is held.
        ("persona", True, "actor_pack_import_cancelled"),
    ],
)
def test_actor_pack_activation_in_flight_keeps_a_capture_drain_waiting(
    tmp_path, monkeypatch, actor_kind, cancel, category
):
    """TASK-40304 AC1: a pause that begins after the first asset is published
    cannot drain until the activation has retired, and the activation leaves no
    file in either tree for the capture to find."""
    import threading
    import time

    from tldw_chatbook.Actor_Packs import activation as activation_module
    from tldw_chatbook.Actor_Packs.activation import ActorPackActivationError
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import (
        retire_current_thread_caches,
    )

    db, activation, review, trees = _actor_pack_activation_on_profile(
        tmp_path, monkeypatch, actor_kind
    )
    first, release, cancelled = threading.Event(), threading.Event(), threading.Event()
    publish = activation_module._publish_immutable

    def publish_then_wait(*args, **kwargs):
        created = publish(*args, **kwargs)
        if not first.is_set():
            # Retire the worker's cached ChaChaNotes connection, so only the
            # activation's own admission can keep a drain waiting.
            db.close_connection()
            first.set()
            assert release.wait(30)
        return created

    monkeypatch.setattr(activation_module, "_publish_immutable", publish_then_wait)
    worker, outcome = _activate_on_worker(
        activation, review, cancel_requested=cancelled.is_set
    )
    try:
        assert first.wait(30)
        assert _published_files(trees)
        # Runtime maintenance's order: fence, retire this thread's own DB
        # cache, then require a drain.
        pause = storage._begin_local_pause()
        try:
            retire_current_thread_caches(pause)
            drained_early = pause.drain(time.monotonic() + 0.3)
            if cancel:
                cancelled.set()
            release.set()
            worker.join(30)
            assert not worker.is_alive()
            assert pause.drain(time.monotonic() + 10)
            # What a capture starting now would copy.
            leftover = _published_files(trees)
        finally:
            pause.resume()
        assert drained_early is False, "a capture drained over an in-flight activation"
        assert leftover == []
        assert type(outcome[0]) is ActorPackActivationError, outcome
        assert outcome[0].category == category
        assert outcome[0].cleanup_pending is False
    finally:
        release.set()
        worker.join(30)
        db.close_connection()


def test_actor_pack_pause_after_the_persona_commit_keeps_its_assets(
    tmp_path, monkeypatch
):
    """TASK-40304 review C1: a pause that begins after the persona commit
    refuses only the intent's cleanup. The activation reports the committed
    actor with cleanup pending and keeps every published file."""
    import threading
    import time

    from tldw_chatbook.Actor_Packs.activation import ActorPackActivationResult
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import (
        retire_current_thread_caches,
    )

    db, activation, review, trees = _actor_pack_activation_on_profile(
        tmp_path, monkeypatch
    )
    committed, release = threading.Event(), threading.Event()
    repository = activation.persona_coordinator.repository
    cleanup = repository.cleanup_persona_intent

    def cleanup_after_pause(*args, **kwargs):
        committed.set()
        assert release.wait(30)
        return cleanup(*args, **kwargs)

    monkeypatch.setattr(repository, "cleanup_persona_intent", cleanup_after_pause)
    worker, outcome = _activate_on_worker(activation, review)
    try:
        assert committed.wait(30)
        at_commit = _published_files(trees)
        pause = storage._begin_local_pause()
        try:
            retire_current_thread_caches(pause)
            release.set()
            worker.join(30)
            assert not worker.is_alive()
            assert pause.drain(time.monotonic() + 10)
            after = _published_files(trees)
        finally:
            pause.resume()
        assert len(at_commit) == 3
        assert after == at_commit, "committed Actor Pack assets were removed"
        assert type(outcome[0]) is ActorPackActivationResult, outcome
        assert outcome[0].cleanup_pending is True
        identity = activation.repository.get_identity_by_portable_uuid(
            review.portable_uuid
        )
        assert identity is not None and identity.local_actor_id == (
            outcome[0].local_actor_id
        )
    finally:
        release.set()
        worker.join(30)
        db.close_connection()


def test_actor_pack_store_error_before_commit_removes_published_files(
    tmp_path, monkeypatch
):
    """TASK-40304 review M2: a store error that is not a sqlite3.Error (here
    CharactersRAGDBError from the character insert) used to escape activation
    and leave the published shared-visual asset behind."""
    from tldw_chatbook.Actor_Packs.activation import ActorPackActivationError
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDBError

    db, activation, review, trees = _actor_pack_activation_on_profile(
        tmp_path, monkeypatch, "character"
    )

    def refused_insert(*args, **kwargs):
        assert _published_files(trees)
        raise CharactersRAGDBError("store refused")

    monkeypatch.setattr(db, "_insert_character_card_in_transaction", refused_insert)
    try:
        with pytest.raises(ActorPackActivationError) as raised:
            activation.activate(review, "create_new")
        assert raised.value.category == "actor_pack_import_activation_failed"
        assert raised.value.cleanup_pending is False
        assert _published_files(trees) == []
        assert activation.repository.get_identity_by_portable_uuid(
            review.portable_uuid
        ) is None
    finally:
        db.close_connection()


def test_actor_pack_pause_between_persona_apply_and_commit_leaves_no_half_persona(
    tmp_path, monkeypatch
):
    """TASK-41114 AC2: a capture's pause that begins after the persona plan
    is applied but before its SQLite commit must not leave a persona without
    its committed identity or artwork, neither in what the capture copies nor
    in the running app after the pause ends. The commit needs the shared
    visual tree's own admission, which the pause refuses, so the activation
    is refused and every store goes back to its state before it."""
    import json
    import threading
    import time

    from tldw_chatbook.Actor_Packs.activation import ActorPackActivationError
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import (
        retire_current_thread_caches,
    )

    db, activation, review, trees = _actor_pack_activation_on_profile(
        tmp_path, monkeypatch, bound=True
    )
    local = activation.local_service
    repository = activation.repository
    applied, release = threading.Event(), threading.Event()
    apply = type(local)._actor_pack_apply_persona_plan

    def apply_then_wait(plan):
        apply(local, plan)
        applied.set()
        assert release.wait(30)

    monkeypatch.setattr(local, "_actor_pack_apply_persona_plan", apply_then_wait)

    def stored_personas():
        store = local.persona_store_path
        if not store.exists():
            return []
        return [row["id"] for row in json.loads(store.read_bytes())["profiles"]]

    worker, outcome = _activate_on_worker(activation, review)
    try:
        assert applied.wait(30)
        assert len(stored_personas()) == 1
        pause = storage._begin_local_pause()
        try:
            release.set()
            worker.join(60)
            assert not worker.is_alive()
            # Runtime maintenance's drain loop: retire this thread's caches
            # until every other holder has finished.
            deadline = time.monotonic() + 10
            while not pause.drain(time.monotonic()):
                assert time.monotonic() < deadline, "the pause never drained"
                retire_current_thread_caches(pause)
                time.sleep(0.01)
            # What a capture starting now would copy.
            copied = (stored_personas(), _published_files(trees))
        finally:
            pause.resume()
        identity = repository.get_identity_by_portable_uuid(review.portable_uuid)
        intents = [row.state for row in repository.list_persona_intents()]
        live = [row["id"] for row in local.list_persona_profiles()]
    finally:
        release.set()
        worker.join(30)
        db.close_idle_connections()
        db.close_connection()
    assert type(outcome[0]) is ActorPackActivationError, outcome
    assert outcome[0].category == "actor_pack_import_storage_unavailable"
    assert outcome[0].cleanup_pending is False
    assert copied == ([], []), "a capture copied a half-created persona"
    assert (identity, intents, live) == (None, [], []), "a half-created persona"


def _stuck_persona_import(activation, intent_ids=("d" * 32,)):
    """TASK-41114: prepared persona intents whose store the user then changed.

    Startup recovery quarantines them (``authority_mismatch``), which blocks
    every later persona Actor Pack creation.
    """
    local, repository = activation.local_service, activation.repository
    for index, intent_id in enumerate(intent_ids):
        stuck = {"id": f"local-persona-stuck-{index}", "name": "Stuck", "version": 1}
        plan = local._actor_pack_plan_persona_profile(stuck, operation="create")
        repository.prepare_persona_intent(
            persona_id=stuck["id"],
            operation="create",
            old_profile=None,
            new_profile=stuck,
            old_store_sha256=plan.old_store_sha256,
            new_store_sha256=plan.new_store_sha256,
            old_registry_uuid=None,
            new_registry_uuid=f"{index + 2}23e4567-e89b-42d3-a456-426614174000",
            intent_id=intent_id,
        )
    local.create_persona_profile({"name": "Existing persona"})
    activation.persona_coordinator.ensure_recovered()
    assert activation.persona_coordinator.blocked_intent_ids == tuple(intent_ids)


def test_interrupted_persona_import_blocks_until_cleared_and_keeps_personas(
    tmp_path, monkeypatch
):
    """TASK-41114 AC1, owner decision 2026-11-16 ("Clear the interrupted
    import"): a stuck intent refuses a new persona import with copy that names
    it and the way out; clearing deletes only the intent rows, every persona
    stays as it was, and the same import then succeeds."""
    from tldw_chatbook.Actor_Packs.activation import ActorPackActivationError
    from tldw_chatbook.UI.Screens.personas_screen import (
        _actor_pack_import_error_copy,
    )

    db, activation, review, trees = _actor_pack_activation_on_profile(
        tmp_path, monkeypatch, bound=True
    )
    local, repository = activation.local_service, activation.repository
    try:
        _stuck_persona_import(activation)
        store = local.persona_store_path.read_bytes()
        personas = sorted(row["id"] for row in local.list_persona_profiles())
        with pytest.raises(ActorPackActivationError) as raised:
            activation.activate(review, "create_new")
        assert raised.value.category == "actor_pack_import_creation_blocked"
        copy = _actor_pack_import_error_copy(raised.value.category)
        assert "interrupted" in copy and "clear the interrupted import" in copy
        assert _published_files(trees) == []

        assert activation.persona_coordinator.clear_blocked_intents() == 1
        assert repository.list_persona_intents() == ()
        assert activation.persona_coordinator.blocked_intent_ids == ()
        assert local.persona_store_path.read_bytes() == store
        assert sorted(row["id"] for row in local.list_persona_profiles()) == personas

        result = activation.activate(review, "create_new")
        assert result.actor_kind == "persona"
        assert sorted(row["id"] for row in local.list_persona_profiles()) == sorted(
            [*personas, result.local_actor_id]
        )
    finally:
        db.close_idle_connections()
        db.close_connection()


def test_clearing_the_interrupted_import_refuses_during_a_storage_pause(
    tmp_path, monkeypatch
):
    """TASK-41114 AC1: the clear action takes ChaChaNotes admission, so a
    capture's or restore's pause refuses it cleanly and changes nothing."""
    from tldw_chatbook.Actor_Packs.persona_coordinator import (
        PersonaActorPackCoordinatorError,
    )
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    db, activation, review, trees = _actor_pack_activation_on_profile(
        tmp_path, monkeypatch, bound=True
    )
    coordinator, repository = activation.persona_coordinator, activation.repository
    try:
        _stuck_persona_import(activation)
        pause = storage._begin_local_pause()
        try:
            with pytest.raises(PersonaActorPackCoordinatorError) as raised:
                coordinator.clear_blocked_intents()
        finally:
            pause.resume()
        assert raised.value.category == "actor_pack_storage_unavailable"
        assert coordinator.blocked_intent_ids == ("d" * 32,)
        assert [row.state for row in repository.list_persona_intents()] == [
            "quarantined"
        ]
        assert coordinator.clear_blocked_intents() == 1
    finally:
        db.close_idle_connections()
        db.close_connection()


def test_a_pause_between_two_clears_keeps_the_remaining_intent_blocked(
    tmp_path, monkeypatch
):
    """TASK-41114, review r15ac-rev3: a pause that begins after the first of
    two deletes refuses the second. The live ids and the cached recovery
    summary the app reads next both name only the remaining intent, and the
    Personas notice still shows for it."""
    import asyncio
    from types import SimpleNamespace

    from textual.app import App
    from textual.containers import Vertical
    from textual.widgets import Static

    from tldw_chatbook.Actor_Packs.persona_coordinator import (
        PersonaActorPackCoordinatorError,
    )
    from tldw_chatbook.app_service_wiring import ServiceWiringMixin
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Widgets.Persona_Widgets.actor_pack_recovery_notice import (
        NOTICE_ID,
        show_actor_pack_recovery_notice,
    )

    db, activation, review, trees = _actor_pack_activation_on_profile(
        tmp_path, monkeypatch, bound=True
    )
    coordinator, repository = activation.persona_coordinator, activation.repository
    first, remaining = "d" * 32, "e" * 32
    pauses = []
    cleanup = repository.cleanup_persona_intent

    def cleanup_then_pause(intent_id, *, expected_state):
        cleanup(intent_id, expected_state=expected_state)  # The pause refuses the 2nd.
        pauses.append(storage._begin_local_pause())

    try:
        _stuck_persona_import(activation, (first, remaining))
        monkeypatch.setattr(repository, "cleanup_persona_intent", cleanup_then_pause)
        try:
            with pytest.raises(PersonaActorPackCoordinatorError) as raised:
                coordinator.clear_blocked_intents()
        finally:
            for pause in pauses:
                pause.resume()
        assert raised.value.category == "actor_pack_storage_unavailable"
        assert len(pauses) == 1
        assert coordinator.blocked_intent_ids == (remaining,)
        assert coordinator.ensure_recovered().blocked_intent_ids == (remaining,)
        assert [
            (row.intent_id, row.state) for row in repository.list_persona_intents()
        ] == [(remaining, "quarantined")]
        app = SimpleNamespace(
            persona_actor_pack_coordinator=coordinator,
            chachanotes_db=db,
            actor_pack_recovery_error=None,
            local_character_persona_service=activation.local_service,
            loguru_logger=SimpleNamespace(error=lambda *a: None, warning=lambda *a: None),
        )
        ServiceWiringMixin.ensure_actor_pack_recovery(app)
        assert app.actor_pack_recovery_error == "actor_pack_recovery_blocked"

        class Personas(App):
            def compose(self):
                with Vertical(id="personas-shell"):
                    yield Static("Characters", id="personas-purpose")

        async def notice_shows():
            shell = Personas()
            shell.persona_actor_pack_coordinator = coordinator
            async with shell.run_test(size=(140, 20)) as pilot:
                await show_actor_pack_recovery_notice(shell.screen)
                await pilot.pause()
                return bool(shell.screen.query(f"#{NOTICE_ID}"))

        assert asyncio.run(notice_shows())
    finally:
        db.close_idle_connections()
        db.close_connection()


def test_cleared_interrupted_import_stays_cleared_on_the_next_recovery_read(
    tmp_path, monkeypatch
):
    """TASK-41114, review r15ac-rev2: the next Personas visit reruns the app's
    recovery read, which returned the cached summary that still named the
    cleared intent. That put actor_pack_recovery_error back and skipped the
    Buddy setup until restart."""
    from types import SimpleNamespace

    import tldw_chatbook.app_service_wiring as wiring
    import tldw_chatbook.Persona_Buddy.library as buddy_library
    from tldw_chatbook.app_service_wiring import ServiceWiringMixin

    db, activation, review, trees = _actor_pack_activation_on_profile(
        tmp_path, monkeypatch, bound=True
    )
    buddy = []

    class Library:  # Records that the Buddy block ran, then stops it.
        def __init__(self, *args, **kwargs):
            buddy.append(True)
            raise RuntimeError("buddy block reached")

    monkeypatch.setattr(buddy_library, "BuddyLibrary", Library)
    # The wiring module bound its own config import; point it at this profile.
    monkeypatch.setattr(wiring, "get_user_data_dir", lambda: tmp_path / "data")
    app = SimpleNamespace(
        persona_actor_pack_coordinator=activation.persona_coordinator,
        chachanotes_db=db,
        actor_pack_recovery_error=None,
        local_character_persona_service=activation.local_service,
        loguru_logger=SimpleNamespace(error=lambda *a: None, warning=lambda *a: None),
    )
    try:
        _stuck_persona_import(activation)
        ServiceWiringMixin.ensure_actor_pack_recovery(app)
        assert app.actor_pack_recovery_error == "actor_pack_recovery_blocked"
        assert buddy == []
        assert activation.persona_coordinator.clear_blocked_intents() == 1
        app.actor_pack_recovery_error = None  # What the notice does on success.
        ServiceWiringMixin.ensure_actor_pack_recovery(app)
        assert app.actor_pack_recovery_error is None
        assert buddy == [True]
    finally:
        db.close_idle_connections()
        db.close_connection()


def test_persona_actor_pack_creation_blocked_names_the_stuck_import():
    """TASK-41114, review r15ac-rev2: creating a persona Actor Pack while an
    interrupted import is stuck names it and the way out, and asks for the
    notice (a block can also appear mid-session)."""
    from types import SimpleNamespace

    from tldw_chatbook.Actor_Packs.creation import ActorPackCreationError
    from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
    from tldw_chatbook.Widgets.Persona_Widgets.actor_pack_recovery_notice import (
        show_actor_pack_recovery_notice,
    )

    notices, scheduled = [], []
    screen = SimpleNamespace(
        _actor_pack_session_is_current=lambda session: True,
        _notify=lambda message, severity: notices.append((message, severity)),
        call_later=lambda callback, *args: scheduled.append((callback, args)),
    )
    PersonasScreen._show_actor_pack_creation_error(
        screen, ActorPackCreationError("actor_pack_creation_blocked"), object()
    )
    [(message, severity)] = notices
    assert severity == "error"
    assert "interrupted" in message
    assert "Keep my current personas and clear the interrupted import" in message
    assert scheduled == [(show_actor_pack_recovery_notice, (screen,))]
    # Any other failure keeps the generic copy and no notice.
    notices.clear(), scheduled.clear()
    PersonasScreen._show_actor_pack_creation_error(
        screen, ActorPackCreationError("actor_pack_creation_failed"), object()
    )
    assert notices == [("Actor Pack creation failed.", "error")] and scheduled == []


@pytest.mark.parametrize("refusal", [None, "actor_pack_storage_unavailable"])
def test_interrupted_import_notice_clears_from_the_keyboard_off_the_ui_thread(
    refusal,
):
    """TASK-41114 AC1: the Personas notice and its action are reachable by
    Tab and Enter, the clear runs off the UI thread, and a refused clear keeps
    the notice with a usable button."""
    import asyncio
    import threading

    from textual.containers import Vertical
    from textual.widgets import Button, Static

    from Tests.UI.consolidated_css import ConsolidatedCSSApp
    from tldw_chatbook.Actor_Packs.persona_coordinator import (
        PersonaActorPackCoordinatorError,
    )
    from tldw_chatbook.Widgets.Persona_Widgets.actor_pack_recovery_notice import (
        CLEAR_BUTTON_ID,
        NOTICE_ID,
        NOTICE_TEXT,
        show_actor_pack_recovery_notice,
    )

    class Coordinator:
        def __init__(self) -> None:
            self.blocked_intent_ids = ("d" * 32,)
            self.threads: list = []

        def clear_blocked_intents(self):
            self.threads.append(threading.current_thread())
            if refusal is not None:
                raise PersonaActorPackCoordinatorError(refusal)
            self.blocked_intent_ids = ()
            return 1

    class Personas(ConsolidatedCSSApp):
        def __init__(self) -> None:
            super().__init__()
            self.persona_actor_pack_coordinator = Coordinator()
            self.actor_pack_recovery_error = "actor_pack_recovery_blocked"
            self.notices: list = []

        def compose(self):
            with Vertical(id="personas-shell"):
                yield Static("Characters", id="personas-purpose")
                yield Button("Library", id="personas-first")

        def notify(self, message, **kwargs):
            self.notices.append((message, kwargs.get("markup")))

    async def run():
        app = Personas()
        async with app.run_test(size=(140, 30)) as pilot:
            await show_actor_pack_recovery_notice(app.screen)
            await show_actor_pack_recovery_notice(app.screen)  # Once only.
            await pilot.pause()
            shell = app.screen.query_one("#personas-shell")
            assert [child.id for child in shell.children] == [
                "personas-purpose",
                NOTICE_ID,
                "personas-first",
            ]
            text = app.screen.query_one(f"#{NOTICE_ID}").query_one(Static)
            assert str(text.render()) == NOTICE_TEXT
            for _ in range(4):
                if getattr(app.focused, "id", None) == CLEAR_BUTTON_ID:
                    break
                await pilot.press("tab")
            assert app.focused.id == CLEAR_BUTTON_ID
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            coordinator = app.persona_actor_pack_coordinator
            assert coordinator.threads
            assert threading.main_thread() not in coordinator.threads
            assert all(markup is False for _, markup in app.notices), app.notices
            if refusal is None:
                assert not app.screen.query(f"#{NOTICE_ID}")
                assert app.actor_pack_recovery_error is None
                assert "cleared" in app.notices[-1][0]
            else:
                assert "Backup or restore" in app.notices[-1][0]
                assert not app.screen.query_one(f"#{CLEAR_BUTTON_ID}").disabled
                assert app.actor_pack_recovery_error == "actor_pack_recovery_blocked"

    asyncio.run(run())


@pytest.mark.bootstrap_profile  # A real-app mount (TASK-32873 per-node opt-in).
def test_personas_shows_the_interrupted_import_notice_after_recovery():
    """TASK-41114 AC1: opening Personas with a stuck persona Actor Pack
    import shows the notice under the purpose line, after recovery ran."""
    import asyncio

    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
    from tldw_chatbook.Widgets.Persona_Widgets.actor_pack_recovery_notice import (
        NOTICE_ID,
    )

    class Coordinator:
        blocked_intent_ids = ("d" * 32,)
        recovery_attempted = True
        recovery_error = None

        def ensure_recovered(self):
            return None

    async def run():
        app = _build_test_app()
        app.persona_actor_pack_coordinator = Coordinator()
        app.actor_pack_recovery_error = "actor_pack_recovery_blocked"
        async with app.run_test(size=(235, 52)) as pilot:
            await pilot.pause()
            screen = PersonasScreen(app)
            await app.push_screen(screen)
            for _ in range(40):
                await pilot.pause()
                if screen.query(f"#{NOTICE_ID}"):
                    break
            shell = screen.query_one("#personas-shell")
            ids = [child.id for child in shell.children]
            assert NOTICE_ID in ids, ids
            assert ids[ids.index(NOTICE_ID) - 1] == "personas-purpose"

    asyncio.run(run())


def test_actor_pack_import_leaves_no_worker_connection_for_a_later_drain(
    tmp_path, monkeypatch
):
    """TASK-41110: the import controller inspects and activates on
    ``asyncio.to_thread``. A capture that begins after the import has
    finished must drain, so the executor thread must not keep the
    ChaChaNotes connection the import opened there."""
    import asyncio
    import time

    from tldw_chatbook.Actor_Packs.import_controller import (
        ActorPackImportController,
    )
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import (
        retire_current_thread_caches,
    )

    db, activation, review, trees = _actor_pack_activation_on_profile(
        tmp_path, monkeypatch
    )
    activation.importer.cleanup_review(review)  # The controller inspects anew.
    controller = ActorPackImportController(activation.importer, activation)

    async def import_pack():
        request = controller.create_request(tmp_path / "persona.tldw-actor-pack")
        inspected = await controller.wait(controller.start_inspection(request))
        assert inspected.review is not None, inspected
        activated = await controller.wait(
            controller.start_activation(inspected.review, "create_new")
        )
        await controller.shutdown()
        return activated

    try:
        outcome = asyncio.run(import_pack())
        assert outcome.error_category is None, outcome
        assert outcome.result.actor_kind == "persona"
        assert len(_published_files(trees)) == 3
        # A capture starting now: fence, retire this thread's own caches, drain.
        pause = storage._begin_local_pause()
        try:
            retire_current_thread_caches(pause)
            drained = pause.drain(time.monotonic() + 5)
            holders = sorted(
                getattr(getattr(lease, "resource_thread", None), "name", "?")
                for lease in storage._live_leases
                if getattr(lease, "resource_path", None) == db.db_path
            )
        finally:
            pause.resume()
        assert drained, f"ChaChaNotes leases still held by threads {holders}"
    finally:
        db.close_idle_connections()  # Every thread's handle, also when red.
        db.close_connection()


def test_actor_pack_and_reaction_trees_have_their_own_review_labels():
    """TASK-40305 AC1: restore, recovery-copy and rollback review name owners
    through ``owner_label``; these two trees read as "Other saved data"."""
    from tldw_chatbook.Backup_Recovery.recovery_service import owner_label

    persona = (
        "persona.actor_packs",
        "persona.visual_identity",
        "persona.assets",
        "persona.visual_identity_builtin",
    )
    labels = [owner_label(owner) for owner in persona]
    assert "Other saved data" not in labels, labels
    assert len(set(labels)) == len(persona), labels


def test_every_installed_backup_owner_has_its_own_review_label():
    """TASK-41111: every installed owner can reach a review (archive Contents,
    dependency groups, portable relocation), so each has its own name there;
    ``recovery.credentials`` is the credential material include-mode capture
    adds. A newly installed owner fails here until it is named."""
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
    from tldw_chatbook.Backup_Recovery.recovery_service import _OWNER_LABELS

    owners = [adapter.owner_id for adapter in install_adapters()]
    owners.append("recovery.credentials")
    assert [owner for owner in owners if owner not in _OWNER_LABELS] == []
    labels = [_OWNER_LABELS[owner] for owner in owners]
    assert len(set(labels)) == len(labels), sorted(labels)


@pytest.mark.parametrize(
    ("retained", "issues", "expected"),
    [
        (("personal_context",), ("personal_context_restore_readable:pc",), True),
        # A server credential the restore remaps is installed, not retained.
        (("personal_context", "server:create"), (), True),
        (("personal_context", "generation"), (), False),
        (("server", "personal_context"), (), False),
        (
            ("personal_context",),
            ("personal_context_restore_locked:personal_context_key_install_unavailable:pc",),
            False,
        ),
        ((), (), False),
    ],
)
def test_retention_review_names_only_keys_the_restore_installs(
    retained, issues, expected
):
    """TASK-41124: the retention review drops "manual recovery" only when every
    retained scope is a Personal Context key and no Personal Context line in the
    plan is locked (a locked key is not installed)."""
    import json

    from tldw_chatbook.Backup_Recovery.recovery_service import (
        retains_only_installed_keys,
    )

    records, scopes = [], {}
    for index, entry in enumerate(retained):
        kind, _, action = entry.partition(":")
        records.append({"id": f"r{index}", "kind": kind})
        scopes[f"r{index}"] = json.dumps({"action": action or "retain"})
    assert retains_only_installed_keys(scopes, records, issues) is expected


def test_retention_classification_reads_without_hashing_staged_files(
    tmp_path, monkeypatch
):
    """TASK-41124 review r15br-rev1: choosing the review's copy reads only the
    candidate's ``candidate.json`` and credential material. It never re-hashes
    staged payloads (``publication._descriptor`` hashes every one, twice), and a
    candidate made for another plan reads as manual."""
    import json
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery import archive_reader, journal, publication
    from tldw_chatbook.Backup_Recovery.recovery_service import (
        _retention_installs_keys,
    )

    calls = []
    for module, name in (
        (publication, "observe_artifact"),
        (journal, "observe_artifact"),
        (archive_reader, "_hash"),
    ):
        monkeypatch.setattr(module, name, lambda *a, _n=name, **k: calls.append(_n))
    candidate = tmp_path / "candidate"
    candidate.mkdir(mode=0o700)
    issues = ["personal_context_restore_readable:pc"]
    for name, document in (
        (
            "candidate.json",
            {
                "version": 1,
                "archive_digest": "digest",
                "issues": issues,
                "credential_scopes": {"r0": json.dumps({"action": "retain"})},
                "artifacts": [{"candidate": str(candidate / "payload")}],
            },
        ),
        (
            "credential-recovery.json",
            {"version": 1, "records": [{"id": "r0", "kind": "personal_context"}]},
        ),
    ):
        (candidate / name).write_text(json.dumps(document))
        (candidate / name).chmod(0o600)
    (candidate / "payload").write_bytes(b"x" * 1024)
    plan = SimpleNamespace(archive_digest="digest", issues=tuple(issues))
    assert _retention_installs_keys(candidate, plan) is True
    assert calls == []
    other = SimpleNamespace(archive_digest="other", issues=tuple(issues))
    assert _retention_installs_keys(candidate, other) is False


# TASK-40134: the Personal Context repository is a captured backup owner. These
# fixtures use the in-memory key protector; no test here reaches an OS keyring.


def _personal_context_profile(tmp_path, *, layout="fresh"):
    """A selected config whose data dir holds a repository with a device-only record."""
    import os
    import sqlite3
    from contextlib import closing

    from Tests.Personal_Context.test_repository import _historical_fresh_schema
    from tldw_chatbook.Personal_Context.key_protector import (
        InMemoryProfileKeyProtector,
    )
    from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
    from tldw_chatbook.Personal_Context.service import PersonalContextService

    config = tmp_path / "selected.toml"
    config.write_text(
        f'[general]\nusers_name = "Ada"\n[paths]\ndata_dir = "{tmp_path / "data"}"\n'
    )
    root = tmp_path / "data" / "Ada"
    root.mkdir(parents=True)
    root.chmod(0o700)
    path = root / "tldw_chatbook_personal_context.db"
    if layout != "fresh":
        with closing(sqlite3.connect(path)) as connection, connection:
            for statement in _historical_fresh_schema(layout):
                connection.execute(statement)
            connection.execute(
                "INSERT INTO personal_context_schema(singleton, version) VALUES (1, ?)",
                (layout,),
            )
        os.chmod(path, 0o600)
    protector = InMemoryProfileKeyProtector()
    repository = PersonalContextRepository(path, key_protector=protector)
    PersonalContextService(repository).create_profile()
    repository.commit_record_version(
        _device_only_record(repository), expected_version_id=None
    )
    _personal_context_profile.protector = protector
    return config, root


def _device_only_record(repository):
    from tldw_profile_core import (
        AgentVisibility,
        PreferencePayload,
        ProfileControls,
        ProfileProvenance,
        ProfileRecord,
        ProvenanceSource,
        RecordKind,
        RecordState,
        SemanticKey,
        SyncMode,
    )
    from tldw_profile_core.models import ActorType

    manifest = repository.get_manifest()
    scope = repository.list_scopes()[0]
    return ProfileRecord(
        profile_id=manifest.profile_id,
        record_id="record-device-only",
        scope_id=scope.scope_id,
        kind=RecordKind.PREFERENCE,
        payload=PreferencePayload(subject="response.detail", polarity="like", value="v"),
        semantic_key=SemanticKey(namespace="preference", subject="response.detail"),
        state=RecordState.ACTIVE,
        controls=ProfileControls(
            sync_mode=SyncMode.DEVICE_ONLY,
            agent_visibility=AgentVisibility.AGENT_VISIBLE,
        ),
        provenance=ProfileProvenance(
            source=ProvenanceSource.MANUAL,
            actor=ActorType.USER,
            reason_code="settings_edit",
        ),
        version_id="record-device-only-v1",
        parent_version_id=None,
        created_at=manifest.created_at,
        updated_at=manifest.created_at,
    )


class _wal_writer:
    """Keep committed frames in the repository's ``-wal`` while the preview runs."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        import sqlite3

        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA wal_autocheckpoint=0")
        self.connection.execute(
            "INSERT INTO quarantine VALUES "
            "('q-wal', 'record', 'r', NULL, 'fixture', '2026-11-12T00:00:00.000Z')"
        )
        self.connection.commit()
        assert Path(str(self.path) + "-wal").stat().st_size > 0
        return self.connection

    def __exit__(self, *_exc):
        self.connection.close()


def _capture_personal_context(tmp_path, monkeypatch, root):
    from threading import Event

    from Tests.Backup_Recovery.test_core_owners import application_authority
    from tldw_chatbook.Backup_Recovery.models import StorageItem
    from tldw_chatbook.Personal_Context.recovery import recovery_adapters

    (adapter,) = recovery_adapters()
    path = root / "tldw_chatbook_personal_context.db"
    item = StorageItem(
        adapter.owner_id, "profile:test:" + adapter.owner_id, path, "included", ()
    )
    authority = application_authority(tmp_path, root, monkeypatch)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700, exist_ok=True)
    destination = stage / path.name
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 1) as session,
        session.capture_scope((path,), stage),
    ):
        adapter.capture(item, destination, Event())
        # The installed validation, as capture and archive checks run it.
        return adapter.validate(destination), destination


@pytest.mark.parametrize("layout", ["fresh", 1, 2, 4, 5, 6, 7, 8])
def test_personal_context_capture_is_a_validated_copy_of_every_v9_layout(
    tmp_path, monkeypatch, layout
):
    """TASK-40134 AC#2: the archive copy (WAL frames folded in) passes the installed
    exact-schema validation for every v9 layout and keeps device-only records."""
    import sqlite3
    from contextlib import closing

    from tldw_chatbook.Personal_Context.recovery import REPOSITORY_SCHEMAS

    _config, root = _personal_context_profile(tmp_path, layout=layout)
    with _wal_writer(root / "tldw_chatbook_personal_context.db"):
        issues, captured = _capture_personal_context(tmp_path, monkeypatch, root)

    assert issues == ()
    with closing(sqlite3.connect(captured)) as connection:
        catalog = tuple(
            row[0]
            for row in connection.execute(
                "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
            )
        )
        assert connection.execute(
            "SELECT 1 FROM quarantine WHERE quarantine_id = 'q-wal'"
        ).fetchone()
        # Device-only rows travel too (decision 8); their bodies stay ciphertext.
        assert connection.execute(
            "SELECT object_id FROM object_heads WHERE object_type = 'record'"
        ).fetchall() == [("record-device-only",)]
    assert catalog in {schema for _, schema in REPOSITORY_SCHEMAS}


@pytest.mark.parametrize("marker", ["freeze", "rebaseline"])
def test_personal_context_capture_refuses_a_first_link_in_flight(
    tmp_path, monkeypatch, marker
):
    """TASK-40134 AC#7: a freeze or rebaseline row refuses capture by name, in the
    preview and in capture itself."""
    import sqlite3
    from contextlib import closing

    from tldw_chatbook.Backup_Recovery.inventory import discover
    from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
    from tldw_chatbook.Backup_Recovery.recovery_service import issue_message

    config, root = _personal_context_profile(tmp_path)
    path = root / "tldw_chatbook_personal_context.db"
    with closing(sqlite3.connect(path)) as connection, connection:
        if marker == "freeze":
            connection.execute(
                "INSERT INTO first_link_freeze VALUES "
                "(1, 'plan-1', 'token', 'profile', 0, 1, '2026-11-12T00:00:00.000Z')"
            )
        else:
            connection.execute(
                "INSERT INTO first_link_rebaseline_commit VALUES "
                "(1, 'plan-1', 'server', 'key-1', 'record-1', 0, 2, "
                "'2026-11-12T00:00:00.000Z')"
            )
    install_adapters()
    inventory = discover((config,))

    assert "personal_context_first_link_in_progress" in inventory.issues
    assert not inventory.complete
    rows = [item for item in inventory.items if item.owner == "personal_context.repository"]
    assert {(item.path, item.status) for item in rows} >= {(path, "unavailable")}
    assert any(item.logical_id.endswith(":first_link") for item in rows)
    assert "linking to your home server" in issue_message(
        "personal_context_first_link_in_progress"
    )
    with pytest.raises(ValueError, match="personal_context_first_link_in_progress"):
        _capture_personal_context(tmp_path, monkeypatch, root)
    assert not (tmp_path / "stage" / path.name).exists()


def test_no_reviewer_approves_the_shared_sync_owners():
    """TASK-40137: after a restore, nothing approves `config` or
    `runtime.sync_state`, and that is what keeps Notes and Chat sync and the
    Personal Context link from resuming as the restored device. Once both are
    approved, every Sync v2 route sends the restored `device_id` and cursors
    (probe in TASK-40137's notes). By the owner's ruling (2026-11-12), the
    only approvers allowed are TASK-40711's settings review (`config` alone,
    `RecoveryService.approve_config`) and TASK-40139's sync-identity review
    (both owners, with the rebind), which adds its own row here. Any
    approval, called or passed by reference, is counted here."""
    from tldw_chatbook.MCP.recovery_activation import _OWNERS as mcp_owners
    from tldw_chatbook.Personal_Context.recovery import OWNER as personal_context_owner
    from tldw_chatbook.Plugins.backup import OWNER as plugins_owner
    from tldw_chatbook.RAG_Search.activation import _REVIEW_OWNERS as rag_owners
    from tldw_chatbook.Sync_Interop.recovery import (
        SYNC_IDENTITY_OWNERS as sync_identity_owners,
    )

    calls = Counter()
    for path in sorted((ROOT / "tldw_chatbook").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "ActivationStore" not in text:
            continue
        tree = ast.parse(text)
        called = {
            id(node.func): node for node in ast.walk(tree) if isinstance(node, ast.Call)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "approve":
                call = called.get(id(node))
                owner = (
                    ", ".join(ast.unparse(arg) for arg in call.args[1:])
                    if call
                    else "<reference>"
                )
                calls[(path.relative_to(ROOT).as_posix(), owner)] += 1
    assert calls == Counter(
        {
            ("tldw_chatbook/MCP/recovery_activation.py", "row.owner"): 1,
            # review_pairing refuses any other owner than "notes.file_notes"
            # (Notes/recovery_review.py:253).
            ("tldw_chatbook/Notes/recovery_review.py", "owner"): 1,
            # TASK-40134's restore review approves only the repository.
            ("tldw_chatbook/Personal_Context/recovery.py", "OWNER"): 1,
            ("tldw_chatbook/Plugins/backup.py", "OWNER"): 1,
            ("tldw_chatbook/RAG_Search/activation.py", "owner"): 1,
            ("tldw_chatbook/Skills_Interop/recovery_activation.py", "'skills'"): 1,
            # TASK-40711: Backup & Restore's settings review, `config` only.
            ("tldw_chatbook/Backup_Recovery/recovery_service.py", "'config'"): 1,
            # TASK-40139: the sync-identity review, the only approver of
            # `runtime.sync_state`; it clears the restored identity first.
            ("tldw_chatbook/Sync_Interop/recovery.py", "owner"): 1,
            # TASK-41119: Backup & Restore's speech review, the only approver
            # of the two speech owners speech start-up admits through.
            ("tldw_chatbook/Backup_Recovery/recovery_service.py", "'tts.profile_store'"): 1,
            ("tldw_chatbook/Backup_Recovery/recovery_service.py", "'tts.voices'"): 1,
            # TASK-41301 (owner, 2026-11-16): Backup & Restore's model-files
            # review is the one approver of `models.artifacts`; RAG's local
            # model review now records only its exact-files receipt.
            ("tldw_chatbook/Backup_Recovery/recovery_service.py", "'models.artifacts'"): 1,
        }
    )
    assert {"config", "runtime.sync_state"}.isdisjoint(
        {*mcp_owners, personal_context_owner, plugins_owner, *rag_owners}
    )
    assert {"tts.profile_store", "tts.voices", "models.artifacts"}.isdisjoint(
        {
            *mcp_owners,
            personal_context_owner,
            plugins_owner,
            *rag_owners,
            *sync_identity_owners,
        }
    )
    assert sync_identity_owners == ("config", "runtime.sync_state")


_CONFIG_REVIEW = r"""
import asyncio, os, sys
from pathlib import Path
from types import SimpleNamespace
from Tests.network_guard import install, blocked_attempts
install()
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.Backup_Recovery.activation import ActivationStore, bind_activation
from tldw_chatbook.Backup_Recovery.control_records import admission_authority, register_pending
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
state = sys.argv[1]
selector = Path(os.environ['TLDW_CONFIG_PATH'])
data = selector.parent.parent / 'data'
selector.write_text('[general]\nusers_name="test"\n[paths]\ndata_dir="' + str(data) + '"\n')
selector.chmod(0o600)
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.Sync_Interop.server_sync_service import ServerSyncService
repo = SyncStateRepository(data / 'sync.db')
root = bootstrap.default_bootstrap_root()
startup = storage._startups.pop((os.getpid(), str(root)), None)
if startup: startup.close()
authority = admission_authority(root)
authority.register('profile', (selector.parent, data))
control = selector.parent.parent / 'operation'
control.mkdir(mode=0o700)
register_pending(root, 'restore', ('profile',), control, (selector,))
with authority.maintenance(('profile',), 2) as session:
    bind_activation(root, 'restore', selector, 'generation',
                    ('config', 'runtime.sync_state'), session=session)
(root / ('pending-' + bootstrap._key('restore') + '.json')).unlink()

# The review: the current profile names `config` as pending until approved.
service = RecoveryService(selector.parent.parent / 'jobs')
try:
    before = service.current_requirements()
    assert before['generation'] == 'generation', dict(before)
    assert before['pending_owners'] == ('config', 'runtime.sync_state'), dict(before)
    if state == 'approved':
        try:
            service.approve_config('another-generation')
        except ValueError as error:
            assert str(error) == 'config_review_changed', error
        else:
            raise AssertionError('a stale review approved config')
        service.approve_config('generation')
    after = service.current_requirements()
finally:
    service.close()
approved = state == 'approved'
assert after['pending_owners'] == (
    ('runtime.sync_state',) if approved else ('config', 'runtime.sync_state')
), dict(after)
store = ActivationStore(control / 'activation')
assert store.allowed('generation', 'config') == approved
assert not store.allowed('generation', 'runtime.sync_state')

effects = []
# Model-catalog auto-refresh: both the startup scheduling and the run itself.
from tldw_chatbook.app import TldwCli
import tldw_chatbook.app_feature_glue as glue_module
glue_module.load_settings = lambda: {'model_catalog': {
    'auto_refresh_enabled': True, 'refresh_consent_recorded': True}}
async def refresh(**kwargs):
    effects.append('catalog refresh')
    return SimpleNamespace(outcomes=(), disk_write_failed=False)
app = SimpleNamespace(
    app_config={}, model_catalog_disk_store=object(),
    local_llm_provider_catalog_service=SimpleNamespace(
        refresh_stale_configured_providers=refresh),
    _init_providers_models=lambda: None, post_message=lambda *a: None,
    notify=lambda *a, **k: None, _refresh_model_catalogs=None,
    run_worker=lambda *a, **k: effects.append('catalog scheduled'),
    call_after_refresh=lambda *a, **k: effects.append('consent'))
scheduled = TldwCli._schedule_startup_model_catalog_refresh(
    app, after_setup_completion=True)
assert scheduled == approved, scheduled
asyncio.run(TldwCli._refresh_model_catalogs(app))

# Native workspace change tracking: the capture's `config` gate.
from tldw_chatbook.Backup_Recovery import profile_paths
from tldw_chatbook.Workspaces.change_tracking import _NativeInitializationIssuer
class Boundary(Exception): pass
def data_dir(settings):
    effects.append('change tracking')
    raise Boundary()
profile_paths.user_data_dir = data_dir
issuer = object.__new__(_NativeInitializationIssuer)
issuer.check = lambda: None
issuer.runtime = SimpleNamespace(_disposed=False)
issuer._capacity = issuer._activity = object()
try:
    issuer.capture(SimpleNamespace(root=str(data)))
except Boundary:
    assert approved
except ValueError as error:
    assert not approved and str(error) == 'native initialization source unavailable', error
else:
    raise AssertionError('capture passed its boundary')

# Server sync stays refused: `runtime.sync_state` waits for the sync-identity review.
class Provider:
    def build_client(self):
        effects.append('sync client')
        raise AssertionError('sync reached its client')
server = ServerSyncService(None, client_provider=Provider(), state_repository=repo)
try:
    asyncio.run(server.get_changes(client_id='fixture'))
except PermissionError as error:
    assert str(error) == 'sync_activation_required', error
else:
    raise AssertionError('sync ran with runtime.sync_state unapproved')
repo.close()
assert effects == (
    ['catalog scheduled', 'catalog refresh', 'change tracking'] if approved else []
), effects
assert not blocked_attempts(), blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize("state", ["pending", "approved"])
def test_the_config_review_turns_config_consumers_back_on_and_never_sync(
    tmp_path, state
):
    """TASK-40711: until Backup & Restore's settings review approves `config`,
    model-catalog auto-refresh and native workspace change tracking stay off
    and the current profile names `config` as pending. The review approves
    only the reviewed generation's `config`: both consumers run again, and
    server sync is still refused with `sync_activation_required` because
    `runtime.sync_state` waits for the sync-identity review (TASK-40139)."""
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, state, "config_review", script=_CONFIG_REVIEW, timeout=90)


def _ungated_sync_client_routes(text):
    """ServerSyncService methods that reach `_require_client` without `_sync_call`."""
    tree = ast.parse(text)
    (service,) = (
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ServerSyncService"
    )
    gated, ungated = set(), set()
    for method in service.body:
        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(
            isinstance(node, ast.Attribute) and node.attr == "_require_client"
            for node in ast.walk(method)
        ):
            continue
        decorated = any(
            ast.unparse(getattr(decorator, "func", decorator)) == "_sync_call"
            for decorator in method.decorator_list
        )
        (gated if decorated else ungated).add(method.name)
    return gated, ungated


def test_every_server_sync_client_route_holds_the_activation_gate():
    """TASK-40711 review r14cfg-rev1: the settings review approves `config`
    alone, and the config-review test pins the sync refusal on one route
    (`get_changes`). Every other route to the sync client must hold the same
    `_sync_call` gate, which needs `runtime.sync_state` too. The two private
    helpers are reached only from gated methods (TASK-40137's notes)."""
    text = (ROOT / "tldw_chatbook/Sync_Interop/server_sync_service.py").read_text(
        encoding="utf-8"
    )
    gated, ungated = _ungated_sync_client_routes(text)
    assert {"get_changes", "send_changes", "push_v2_envelopes"} & gated, gated
    assert ungated <= {
        "__init__",
        "_require_client",
        "_push_v2_envelopes",
        "_pull_v2_envelopes",
    }, ungated


def test_only_the_settings_review_calls_approve_config():
    """TASK-40711 review r14cfg-rev1: `RecoveryService.approve_config`
    approves `config` for every profile its generation restored, so only
    Backup & Restore's reviewed **Approve settings** may call it."""
    uses = Counter()
    for path in sorted((ROOT / "tldw_chatbook").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "approve_config" not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Attribute) and node.attr == "approve_config":
                uses[path.relative_to(ROOT).as_posix()] += 1
    assert uses == Counter({"tldw_chatbook/UI/Screens/backup_restore_screen.py": 1})


_SPEECH_REVIEW = r"""
import asyncio, os, sys
from pathlib import Path
from types import SimpleNamespace
from Tests.network_guard import install, blocked_attempts
install()
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.Backup_Recovery.activation import ActivationStore, bind_activation
from tldw_chatbook.Backup_Recovery.control_records import admission_authority, register_pending
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
# Imported before binding, as the app is: config computes its client id on import.
from tldw_chatbook import app_speech
state = sys.argv[1]
selector = Path(os.environ['TLDW_CONFIG_PATH'])
data = selector.parent.parent / 'data'
selector.write_text('[general]\nusers_name="test"\n[paths]\ndata_dir="' + str(data) + '"\n')
selector.chmod(0o600)
root = bootstrap.default_bootstrap_root()
startup = storage._startups.pop((os.getpid(), str(root)), None)
if startup: startup.close()
authority = admission_authority(root)
authority.register('profile', (selector.parent, data))
control = selector.parent.parent / 'operation'
control.mkdir(mode=0o700)
register_pending(root, 'restore', ('profile',), control, (selector,))
# Exactly the owners speech start-up admits through (app_speech.py).
OWNERS = ('config', 'models.artifacts', 'tts.profile_store', 'tts.voices')
with authority.maintenance(('profile',), 2) as session:
    bind_activation(root, 'restore', selector, 'generation', OWNERS, session=session)
(root / ('pending-' + bootstrap._key('restore') + '.json')).unlink()
store = ActivationStore(control / 'activation')
SPEECH = ('tts.profile_store', 'tts.voices')
REVIEWS = {'speech': (SPEECH, 'approve_speech', 'speech_review_changed'),
           'models': (('models.artifacts',), 'approve_models', 'models_review_changed')}
done = {'pending': (), 'speech': ('speech',), 'models': ('models',),
        'both': ('speech', 'models')}[state]
service = RecoveryService(selector.parent.parent / 'jobs')
try:
    service.approve_config('generation')  # the settings review (TASK-40711)
    before = service.current_requirements()
    assert before['pending_owners'] == ('models.artifacts', *SPEECH), dict(before)
    for review in done:
        _, approve, changed = REVIEWS[review]
        try:
            getattr(service, approve)('another-generation')
        except ValueError as error:
            assert str(error) == changed, error
        else:
            raise AssertionError('a stale review approved ' + review)
        getattr(service, approve)('generation')
    after = service.current_requirements()
finally:
    service.close()
approved = {owner for review in done for owner in REVIEWS[review][0]}
assert after['pending_owners'] == tuple(
    owner for owner in ('models.artifacts', *SPEECH) if owner not in approved
), dict(after)
for owner in ('models.artifacts', *SPEECH):
    assert store.allowed('generation', owner) == (owner in approved), owner
approved = state == 'both'

# Speech start-up: the admitted initializer runs only once speech is approved.
effects = []
async def initialize():
    effects.append('speech started')
    return 'handler'
app = SimpleNamespace(_speech_initialization_allowed=lambda kind: True)
for kind in ('tts', 'stts'):
    result = asyncio.run(app_speech._run_speech_initialization(app, kind, initialize))
    assert result == ('handler' if approved else None), (kind, result)
assert effects == (['speech started'] * 2 if approved else []), effects
assert not blocked_attempts(), blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize("state", ["pending", "speech", "models", "both"])
def test_the_speech_review_turns_speech_start_up_back_on(tmp_path, state):
    """TASK-41119 / TASK-41301: speech start-up admits through `config`,
    `models.artifacts`, `tts.profile_store` and `tts.voices`. With the settings
    review done, speech stays off until both Backup & Restore's speech review
    (`approve_speech`, the two speech owners) and its model-files review
    (`approve_models`, `models.artifacts`, no local embedding model needed)
    approve the reviewed generation, then starts; each refuses a stale
    generation and approves only its own owners."""
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, state, "speech_review", script=_SPEECH_REVIEW, timeout=90)


@pytest.mark.parametrize("approval", ["approve_speech", "approve_models"])
def test_only_backup_and_restore_calls_its_owner_reviews(approval):
    """TASK-41119 / TASK-41301: like `approve_config`, `approve_speech` and
    `approve_models` approve their owners for every profile their generation
    restored, so only Backup & Restore's reviewed **Approve voices** and
    **Approve model files** may call them."""
    uses = Counter()
    for path in sorted((ROOT / "tldw_chatbook").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if approval not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Attribute) and node.attr == approval:
                uses[path.relative_to(ROOT).as_posix()] += 1
    assert uses == Counter({"tldw_chatbook/UI/Screens/backup_restore_screen.py": 1})


_ELSEWHERE_MODEL = r"""
import os, sys
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.Backup_Recovery.activation import ActivationStore, bind_activation
from tldw_chatbook.Backup_Recovery.control_records import admission_authority, register_pending
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
# Imported before binding, as the app is: config computes its client id on import.
from tldw_chatbook.Embeddings.Embeddings_Lib import HFModelCfg
from tldw_chatbook.RAG_Search import model_recovery as recovery
from tldw_chatbook.RAG_Search.activation import require_local_model_construction
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
state = sys.argv[1]
base = Path(os.environ['TLDW_CONFIG_PATH']).parent.parent
profiles = {}
for name in ('current', 'other'):
    folder = base / name
    (folder / 'config').mkdir(parents=True, mode=0o700)
    (folder / 'data').mkdir(mode=0o700)
    selector = folder / 'config' / 'config.toml'
    selector.write_text('[general]\nusers_name="test"\n[paths]\ndata_dir="'
                        + str(folder / 'data') + '"\n')
    selector.chmod(0o600)
    profiles[name] = selector
selector = Path(os.environ['TLDW_CONFIG_PATH'])
selector.write_text(profiles['current'].read_text())
selector.chmod(0o600)
profiles['current'] = selector
# A local model directory in the current profile's data that the OTHER
# restored profile's roots also cover (its recovery restored that folder).
shared = base / 'current' / 'data' / 'models'
shared.mkdir(mode=0o700)
model = shared / 'local-model'
model.mkdir(mode=0o700)
(model / 'config.json').write_text('{}')
(model / 'model.safetensors').write_bytes(b'weights')
root = bootstrap.default_bootstrap_root()
startup = storage._startups.pop((os.getpid(), str(root)), None)
if startup: startup.close()
authority = admission_authority(root)
authority.register('current', (selector.parent, base / 'current' / 'data'))
authority.register('other', (profiles['other'].parent, shared))
stores = {}
for name, generation in (('current', 'generation'), ('other', 'other-generation')):
    control = base / ('operation-' + name)
    control.mkdir(mode=0o700)
    register_pending(root, name, (name,), control, (profiles[name],))
    with authority.maintenance((name,), 2) as session:
        bind_activation(root, name, profiles[name], generation,
                        ('config', 'models.artifacts'), session=session)
    (root / ('pending-' + bootstrap._key(name) + '.json')).unlink()
    stores[generation] = ActivationStore(control / 'activation')
config = RAGConfig.from_dict({
    'embedding': {'model': str(model), 'device': 'cpu'},
    'vector_store': {'persist_directory': str(base / 'current' / 'data' / 'vectors')}})
service = RecoveryService(base / 'jobs')
try:
    service.approve_models('generation')  # the current profile's Approve model files
finally:
    service.close()
if state == 'approved':  # the other profile's own Approve model files
    stores['other-generation'].approve('other-generation', 'models.artifacts')
review = recovery.preview_local_embedding(config)
assert set(review.generations) == {'generation', 'other-generation'}, review
assert review.elsewhere_pending == (
    () if state == 'approved' else ('other-generation',)), review
recovery.approve_local_embedding(config, review.fingerprint)
try:
    require_local_model_construction(config)
except (OSError, ValueError, RuntimeError):
    allowed = False
else:
    allowed = True
assert allowed == (state == 'approved'), (state, allowed)
assert not blocked_attempts(), blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize("state", ["pending", "approved"])
def test_local_model_review_names_another_restored_generation(tmp_path, state):
    """TASK-41301 review r15br-rev2: a RAG model directory inside another
    restored profile carries that profile's generation, and RAG needs
    `models.artifacts` approved for every generation it touches. The current
    profile's Approve model files covers only its own, so the review names the
    other generation still pending (``elsewhere_pending``) instead of leaving
    RAG off silently; once that profile's model files are approved, nothing is
    named and the model is admitted."""
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, state, "elsewhere_model", script=_ELSEWHERE_MODEL, timeout=120)


@pytest.mark.parametrize("elsewhere", [(), ("other-generation",)])
def test_local_model_review_copy_records_files_and_names_the_model_files_review(
    elsewhere,
):
    """TASK-41301 review r15br-rev2: Settings' local model dialog no longer says
    it allows the model. Confirming records the review of these exact files,
    the copy names Backup & Restore's Approve model files, and any other
    restored generation still pending is named."""
    from tldw_chatbook.RAG_Search.model_recovery import (
        LocalEmbeddingReview,
        local_embedding_review_text,
    )

    text = local_embedding_review_text(
        LocalEmbeddingReview("f", "/models/m", 3, ("generation",), elsewhere)
    )
    assert "Record the review of these exact files?" in text
    assert "Approve model files in Backup & Restore > Restored profiles" in text
    assert "Allow" not in text
    assert ("another restored profile (generation other-generation)" in text) is bool(
        elsewhere
    )
