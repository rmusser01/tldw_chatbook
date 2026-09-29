"""Patches on ``tldw_chatbook.app`` must reach the code they were written for.

TASK-33011 moves ``TldwCli`` code out of ``app.py`` into sibling modules
(``EXTRACTED_MODULES``). A moved body resolves module-level names through its
NEW module's globals, so ``patch("tldw_chatbook.app.<name>")`` no longer
reaches it. Nothing fails: the patch applies, the moved code reads the real
object, and a test that redirected a database path or a config read into a
temp dir quietly talks to the real one instead.

Two rules close that hole:

1. **Stale target.** A test may not patch ``tldw_chatbook.app.<name>`` when
   app.py never reads ``<name>`` but an extracted module does. The patch can
   only have been meant for code that moved; the failure names the module to
   patch instead. Nor may it patch a name ``tldw_chatbook.app`` does not bind
   at all: ``patch()`` would raise, but ``app_module.<name> = ...`` silently
   creates an attribute nothing reads.
2. **Shared target.** When app.py AND a module in
   ``Tests/app_module_patches.APP_GLOBAL_MODULES`` both read ``<name>``, a
   bare app-module patch reaches only half the reads. Use
   ``patch_app_global`` / ``set_app_global``, which patch every module with
   one object.

Patch forms recognised: a ``"tldw_chatbook.app.<name>"`` string target
(``patch``, ``monkeypatch.setattr``, ``mocker.patch`` ...); ``setattr`` /
``delattr`` / ``patch.object`` on a name bound to the app module; and
``<alias>.<name> = ...``. A dotted target such as
``"tldw_chatbook.app.logger.error"`` patches an attribute of a shared object
and reaches every module, so it is not checked. Python source embedded in a
string (the subprocess scripts several tests run) is parsed and scanned the
same way: ``Tests/Backup_Recovery/test_skills_recovery_review.py`` assigned
two moved names on the app module inside such a script.
"""

from __future__ import annotations

import ast
import re
import symtable
import textwrap
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest

from Tests.app_module_patches import APP_GLOBAL_MODULES

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE = _REPO_ROOT / "tldw_chatbook"
_TESTS = _REPO_ROOT / "Tests"
_APP_MODULE = "tldw_chatbook.app"

#: Modules TASK-33011 moved out of app.py. Add each new one here.
EXTRACTED_MODULES = (
    "app_entry",
    "app_destinations",
    "app_ingest_queue",
    "app_service_wiring",
    "app_speech",
    "app_lifecycle",
    "app_navigation",
    "app_command_providers",
    "app_feature_glue",
)

#: Rule 2 exemptions: (test file, name) -> why a bare app-module patch is right.
SHARED_TARGET_EXEMPTIONS = {
    ("Tests/TTS/test_voice_bundle_maintenance.py", "get_user_data_dir"): (
        "the reader under test, TldwCli._ensure_tts_voice_bundle_service, "
        "stays in app.py"
    ),
}

_TARGET_RE = re.compile(r"^tldw_chatbook\.app\.([A-Za-z_]\w*)$")


@dataclass(frozen=True)
class PatchSite:
    path: str
    line: int
    name: str


@cache
def _module_reads(path: Path) -> frozenset[str]:
    """Names the module's code looks up in its OWN globals (what a patch hits).

    A name a function imports locally, or binds as a local, is not a global
    read: a module patch would not reach it either way.
    """
    reads: set[str] = set()

    def visit(table: symtable.SymbolTable) -> None:
        for symbol in table.get_symbols():
            if not symbol.is_referenced():
                continue
            if table.get_type() == "module" or symbol.is_global() or (
                table.get_type() == "class"
                and not (
                    symbol.is_assigned()
                    or symbol.is_imported()
                    or symbol.is_parameter()
                )
            ):
                reads.add(symbol.get_name())
        for child in table.get_children():
            visit(child)

    visit(symtable.symtable(path.read_text(encoding="utf-8"), str(path), "exec"))
    return frozenset(reads)


@cache
def _module_bindings(path: Path) -> frozenset[str]:
    """Names bound at module scope, plus PEP 562 lazy exports (``_APP_ENTRY_EXPORTS``)."""
    source = path.read_text(encoding="utf-8")
    table = symtable.symtable(source, str(path), "exec")
    bound = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_assigned() or symbol.is_imported()
    }
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_APP_ENTRY_EXPORTS"
            for target in node.targets
        ):
            bound.update(
                item.value
                for item in ast.walk(node.value)
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
    return frozenset(bound)


def _app_aliases(tree: ast.AST) -> set[str]:
    """Names this file binds to the app module object."""
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update(
                alias.asname
                for alias in node.names
                if alias.name == _APP_MODULE and alias.asname
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "tldw_chatbook":
            aliases.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "app"
            )
    module_lookups = {
        f"importlib.import_module('{_APP_MODULE}')",
        f"sys.modules['{_APP_MODULE}']",
    }
    aliases.update(
        node.targets[0].id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and ast.unparse(node.value) in module_lookups
    )
    return aliases


def _is_app_module(node: ast.AST, aliases: set[str]) -> bool:
    return (isinstance(node, ast.Name) and node.id in aliases) or (
        ast.unparse(node) == _APP_MODULE
    )


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


def _embedded_code(tree: ast.AST) -> list[tuple[int, ast.Module]]:
    """(first line, parsed module) for string constants that are Python source."""
    found = []
    # An f-string script is a JoinedStr, not a Constant: rebuild its text with a
    # placeholder name for each interpolation (the literal parts are already
    # unescaped, so ``{{`` reads back as ``{``). Its literal parts are also
    # Constant nodes; skip them so a site is never reported twice.
    fragments = {
        id(value)
        for node in ast.walk(tree)
        if isinstance(node, ast.JoinedStr)
        for value in node.values
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in fragments:
                continue
            text = node.value
        elif isinstance(node, ast.JoinedStr):
            text = "".join(
                value.value if isinstance(value, ast.Constant) else "_FSTRING_VALUE"
                for value in node.values
            )
        else:
            continue
        if "\n" not in text or _APP_MODULE not in text:
            continue
        try:
            found.append((node.lineno, ast.parse(textwrap.dedent(text))))
        except SyntaxError:
            continue
    return found


def app_patch_sites(path: Path, root: Path = _REPO_ROOT) -> list[PatchSite]:
    """Every patch of a top-level ``tldw_chatbook.app`` name in one file.

    Args:
        path: The test file to scan.
        root: Repository root; each site's ``path`` is reported relative to it.

    Returns:
        The file's patch sites sorted by line: direct patch forms plus those
        found in Python source embedded in string constants (subprocess
        scripts). An empty list when the file never mentions the app module.
    """
    source = path.read_text(encoding="utf-8")
    if _APP_MODULE not in source and "import app" not in source:
        return []
    tree = ast.parse(source, filename=str(path))
    rel = path.relative_to(root).as_posix()
    sites = _tree_patch_sites(tree, rel, 0)
    for first_line, embedded in _embedded_code(tree):
        sites.extend(_tree_patch_sites(embedded, rel, first_line - 1))
    return sorted(sites, key=lambda site: site.line)


def _tree_patch_sites(tree: ast.AST, rel: str, line_offset: int) -> list[PatchSite]:
    aliases = _app_aliases(tree)
    sites: list[PatchSite] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if not (name.endswith("patch") or name in {"setattr", "delattr", "object"}):
                continue
            first = node.args[0] if node.args else None
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                match = _TARGET_RE.match(first.value)
                if match:
                    sites.append(
                        PatchSite(rel, node.lineno + line_offset, match.group(1))
                    )
            elif (
                first is not None
                and len(node.args) >= 2
                and _is_app_module(first, aliases)
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                sites.append(
                    PatchSite(rel, node.lineno + line_offset, node.args[1].value)
                )
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            sites.extend(
                PatchSite(rel, node.lineno + line_offset, target.attr)
                for target in targets
                if isinstance(target, ast.Attribute)
                and _is_app_module(target.value, aliases)
            )
    return sites


def _violations(
    test_root: Path, package: Path, root: Path
) -> tuple[list[str], list[str]]:
    app_reads = _module_reads(package / "app.py")
    app_bindings = _module_bindings(package / "app.py")
    extracted = {
        module: _module_reads(package / f"{module}.py")
        for module in EXTRACTED_MODULES
        if (package / f"{module}.py").exists()
    }
    shared_modules = [module.rsplit(".", 1)[1] for module in APP_GLOBAL_MODULES[1:]]
    stale: list[str] = []
    shared: list[str] = []
    for path in sorted(test_root.rglob("*.py")):
        if path.resolve() == Path(__file__).resolve():
            continue  # this file's synthetic fixtures are not real patches
        for site in app_patch_sites(path, root):
            readers = sorted(
                module for module, reads in extracted.items() if site.name in reads
            )
            where = f"{site.path}:{site.line}"
            if site.name not in app_reads and readers:
                stale.append(
                    f"{where} patches tldw_chatbook.app.{site.name}, which app.py "
                    f"never reads; patch tldw_chatbook.{readers[0]}.{site.name}"
                )
            elif site.name not in app_bindings:
                stale.append(
                    f"{where} patches tldw_chatbook.app.{site.name}, which the "
                    "module does not define; the patch reaches nothing"
                )
            elif (
                site.name in app_reads
                and any(module in readers for module in shared_modules)
                and (site.path, site.name) not in SHARED_TARGET_EXEMPTIONS
            ):
                shared.append(
                    f"{where} patches tldw_chatbook.app.{site.name} alone; "
                    f"{', '.join(readers)} also read it -- use "
                    "Tests.app_module_patches.patch_app_global/set_app_global"
                )
    return stale, shared


def test_no_app_module_patch_targets_only_moved_code() -> None:
    stale, _shared = _violations(_TESTS, _PACKAGE, _REPO_ROOT)
    assert stale == [], "\n".join(stale)


def test_shared_app_names_are_patched_on_every_reader() -> None:
    _stale, shared = _violations(_TESTS, _PACKAGE, _REPO_ROOT)
    assert shared == [], "\n".join(shared)


def test_shared_target_exemptions_are_still_needed() -> None:
    for rel_path, name in SHARED_TARGET_EXEMPTIONS:
        sites = app_patch_sites(_REPO_ROOT / rel_path)
        assert any(site.name == name for site in sites), (
            f"{rel_path} no longer patches tldw_chatbook.app.{name}; "
            "drop its SHARED_TARGET_EXEMPTIONS row"
        )


def test_scanner_flags_each_patch_form(tmp_path: Path) -> None:
    """Negative control: every stale form, an unbound name and a bare shared patch."""
    package = tmp_path / "tldw_chatbook"
    package.mkdir()
    (package / "app.py").write_text(
        "from x import get_cli_setting\nget_cli_setting()\n", encoding="utf-8"
    )
    (package / "app_service_wiring.py").write_text(
        "from x import get_cli_setting, get_user_data_dir\n"
        "get_cli_setting()\nget_user_data_dir()\n",
        encoding="utf-8",
    )
    tests = tmp_path / "Tests"
    tests.mkdir()
    (tests / "test_x.py").write_text(
        "from unittest.mock import patch\n"
        "import tldw_chatbook.app as app_module\n"
        "from tldw_chatbook import app\n"
        "def test(monkeypatch):\n"
        "    patch('tldw_chatbook.app.get_user_data_dir')\n"
        "    monkeypatch.setattr(app_module, 'get_user_data_dir', None)\n"
        "    patch.object(app, 'get_user_data_dir')\n"
        "    app_module.get_user_data_dir = None\n"
        "    patch('tldw_chatbook.app.get_cli_setting')\n"
        "    patch('tldw_chatbook.app.logger.error')\n"
        'SCRIPT = """\n'
        "import tldw_chatbook.app as app_module\n"
        "app_module.get_user_data_dir = None\n"
        '"""\n'
        "app_module.no_longer_defined = None\n"
        "value = None\n"
        'FSCRIPT = f"""\n'
        "import tldw_chatbook.app as app_module\n"
        'state = {{"k": 1}}\n'
        "app_module.get_user_data_dir = {value}\n"
        '"""\n',
        encoding="utf-8",
    )
    _module_reads.cache_clear()
    _module_bindings.cache_clear()
    try:
        stale, shared = _violations(tests, package, tmp_path)
    finally:
        _module_reads.cache_clear()
        _module_bindings.cache_clear()
    assert [line.split(" ", 1)[0] for line in stale] == [
        "Tests/test_x.py:5",
        "Tests/test_x.py:6",
        "Tests/test_x.py:7",
        "Tests/test_x.py:8",
        "Tests/test_x.py:13",
        "Tests/test_x.py:15",
        "Tests/test_x.py:20",
    ]
    unbound = [line for line in stale if line.startswith("Tests/test_x.py:15 ")]
    assert len(unbound) == 1 and "does not define" in unbound[0]
    assert all(
        "patch tldw_chatbook.app_service_wiring." in line
        for line in stale
        if not line.startswith("Tests/test_x.py:15 ")
    )
    assert [line.split(" ", 1)[0] for line in shared] == ["Tests/test_x.py:9"]


def test_shared_patch_helper_leaves_a_lazy_module_unloaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deciding where to patch reads source; it never imports a lazy module.

    ``_build_test_app`` patches ``get_subscriptions_db_path`` on every app
    module. ``app_speech`` does not bind it, so the patch must not load it --
    otherwise every factory-built test preloads speech instead of exercising
    its production first-use import.
    """
    import sys

    from Tests.app_module_patches import _binding_extracted_modules

    monkeypatch.delitem(sys.modules, "tldw_chatbook.app_speech", raising=False)
    bound_in = _binding_extracted_modules("get_subscriptions_db_path")
    assert "tldw_chatbook.app_speech" not in bound_in
    assert "tldw_chatbook.app_speech" not in sys.modules


def test_source_derived_bindings_match_runtime_globals() -> None:
    """``module_scope_names`` agrees with what each module really binds."""
    import importlib

    from Tests.app_module_patches import module_scope_names

    for module_name in APP_GLOBAL_MODULES[1:]:
        module = importlib.import_module(module_name)
        runtime = {name for name in vars(module) if not name.startswith("__")}
        assert module_scope_names(module_name) == runtime, module_name
