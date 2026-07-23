from __future__ import annotations

import ast
import warnings
from collections import Counter
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tldw_chatbook.DB.private_sqlite import (
    SQLITE_OWNER_REGISTRY,
    SQLiteOwnerPolicy,
    SQLiteTargetKind,
)


PROJECT_ROOT = Path(__file__).parents[2]
PRODUCTION_ROOT = PROJECT_ROOT / "tldw_chatbook"
INVENTORY_PATH = PROJECT_ROOT / "backlog/docs/sqlite-private-owner-inventory.md"

ALLOWED_CLASSIFICATIONS = {kind.value for kind in SQLiteTargetKind}
ALLOWED_PARENT_DISPOSITIONS = {
    "centralize_backup",
    "justified_exclusion",
    "remove_custom_creation",
    "remove_obsolete_creation",
    "secure_default",
}

EXPECTED_PARENT_CREATORS = {
    (
        "tldw_chatbook/config",
        "get_user_data_dir",
        "user_dir.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/config",
        "load_settings",
        "main_db_file_path_server.parent.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/config",
        "load_settings",
        "user_data_base_dir_server.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/Utils/paths",
        "get_project_databases_dir",
        "PROJECT_DATABASES_DIR.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/Utils/paths",
        "get_user_database_path",
        "USER_DB_DIR.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/DB/base_db",
        "BaseDB.__init__",
        "self.db_path.parent.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/DB/ChaChaNotes_DB",
        "CharactersRAGDB.__init__",
        "self.db_path.parent.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/DB/ChaChaNotes_DB",
        "CharactersRAGDB.backup_database",
        "backup_db_path_obj.parent.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/DB/Client_Media_DB_v2",
        "MediaDatabase.__init__",
        "self.db_path.parent.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/DB/Client_Media_DB_v2",
        "MediaDatabase.backup_database",
        "backup_db_path.parent.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/DB/Prompts_DB",
        "PromptsDatabase.__init__",
        "self.db_path.parent.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/DB/Prompts_DB",
        "PromptsDatabase.backup_database",
        "backup_db_path_obj.parent.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/DB/RAG_Indexing_DB",
        "RAGIndexingDB.__init__",
        "self.db_path.parent.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/DB/Evals_DB",
        "EvalsDB.__init__",
        "self.db_path.parent.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/DB/search_history_db",
        "SearchHistoryDB.__init__",
        "self.db_path.parent.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/Kanban_Interop/local_kanban_db",
        "open_connection",
        "Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/Research_Interop/local_research_service",
        "LocalResearchService.__init__",
        "self.db_path.parent.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/Writing_Interop/local_writing_service",
        "LocalWritingService.__init__",
        "self.db_path.parent.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage",
        "SQLiteStorage.__init__",
        "self.db_path.parent.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/UI/Tools_Settings_Window",
        "ToolsSettingsWindow._backup_worker",
        "backup_dir.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/UI/Tools_Settings_Window",
        "ToolsSettingsWindow._backup_single_worker",
        "backup_dir.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/UI/Tools_Settings_Window",
        "ToolsSettingsWindow._restore_single_database",
        "backup_dir.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/Evals/eval_orchestrator",
        "EvaluationOrchestrator._initialize_database",
        "Path(db_path).parent.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/Event_Handlers/eval_events",
        "get_orchestrator",
        "db_path.parent.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/app",
        "TldwCli._init_prompts_service",
        "prompts_db_path.parent.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/Notes/Notes_Library",
        "NotesInteropService.__init__",
        "self.base_db_directory.mkdir(parents=True, exist_ok=True)",
    ),
    (
        "tldw_chatbook/DB/Sync_Client",
        "<module>",
        "os.makedirs(os.path.dirname(DATABASE_PATH) or '.', exist_ok=True)",
    ),
    (
        "tldw_chatbook/runtime_policy/server_parity_state",
        "build_server_parity_state_repositories",
        "resolved_data_dir.mkdir(parents=True, exist_ok=True)",
    ),
}


def _inventory_rows(prefix: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in INVENTORY_PATH.read_text(encoding="utf-8").splitlines():
        if not line.startswith(f"| {prefix}"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if prefix == "C":
            columns = (
                "id",
                "module",
                "symbol",
                "owner_id",
                "classification",
                "intent",
                "disposition",
            )
        elif prefix == "B":
            columns = (
                "id",
                "module",
                "symbol",
                "owner_id",
                "classification",
                "operation",
                "disposition",
            )
        elif prefix == "P":
            columns = (
                "id",
                "module",
                "symbol",
                "creator_call",
                "state",
                "owner_id",
                "disposition",
                "rationale",
            )
        else:
            raise AssertionError(f"Unsupported inventory prefix: {prefix}")
        assert len(cells) == len(columns), line
        rows.append(dict(zip(columns, cells, strict=True)))
    return rows


def _is_sqlite3_connect(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "connect"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "sqlite3"
    )


class _QualifiedCallVisitor(ast.NodeVisitor):
    def __init__(self, predicate) -> None:
        self.predicate = predicate
        self.symbol_stack: list[str] = []
        self.calls: Counter[str] = Counter()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.symbol_stack.append(node.name)
        self.generic_visit(node)
        self.symbol_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.symbol_stack.append(node.name)
        self.generic_visit(node)
        self.symbol_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        if self.predicate(node):
            symbol = ".".join(self.symbol_stack) if self.symbol_stack else "<module>"
            self.calls[symbol] += 1
        self.generic_visit(node)


class _QualifiedCallNodeVisitor(ast.NodeVisitor):
    def __init__(self, predicate) -> None:
        self.predicate = predicate
        self.symbol_stack: list[str] = []
        self.calls: list[tuple[str, ast.Call]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.symbol_stack.append(node.name)
        self.generic_visit(node)
        self.symbol_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.symbol_stack.append(node.name)
        self.generic_visit(node)
        self.symbol_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        if self.predicate(node):
            symbol = ".".join(self.symbol_stack) if self.symbol_stack else "<module>"
            self.calls.append((symbol, node))
        self.generic_visit(node)


def _qualified_calls(source_path: Path, predicate) -> Counter[str]:
    visitor = _QualifiedCallVisitor(predicate)
    visitor.visit(_parse_source(source_path))
    return visitor.calls


def _qualified_sqlite_connect_calls(source_path: Path) -> Counter[str]:
    tree = _parse_source(source_path)
    module_aliases: set[str] = set()
    callable_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"sqlite3", "sqlite3.dbapi2"}:
                    module_aliases.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module in {
            "sqlite3",
            "sqlite3.dbapi2",
        }:
            for alias in node.names:
                if alias.name in {"connect", "Connection"}:
                    callable_aliases.add(alias.asname or alias.name)
                elif alias.name == "dbapi2":
                    module_aliases.add(alias.asname or alias.name)
                elif alias.name == "*":
                    callable_aliases.update({"connect", "Connection"})

    def is_raw_callable(expression: ast.expr) -> bool:
        if isinstance(expression, ast.Name):
            return expression.id in callable_aliases
        if not isinstance(expression, ast.Attribute) or expression.attr not in {
            "connect",
            "Connection",
        }:
            return False
        root = expression.value
        while isinstance(root, ast.Attribute):
            root = root.value
        return isinstance(root, ast.Name) and root.id in module_aliases

    def is_raw_module(expression: ast.expr) -> bool:
        if isinstance(expression, ast.Name):
            return expression.id in module_aliases
        if not isinstance(expression, ast.Attribute):
            return False
        root = expression
        while isinstance(root, ast.Attribute):
            if root.attr != "dbapi2":
                return False
            root = root.value
        return isinstance(root, ast.Name) and root.id in module_aliases

    rebound_aliases: list[tuple[str, ast.expr]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            rebound_aliases.extend(
                (target.id, node.value)
                for target in node.targets
                if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                rebound_aliases.append((node.target.id, node.value))

    changed = True
    while changed:
        changed = False
        for alias, expression in rebound_aliases:
            if alias not in module_aliases and is_raw_module(expression):
                module_aliases.add(alias)
                changed = True
            if alias not in callable_aliases and is_raw_callable(expression):
                callable_aliases.add(alias)
                changed = True

    def is_raw_connect(call: ast.Call) -> bool:
        return is_raw_callable(call.func)

    visitor = _QualifiedCallVisitor(is_raw_connect)
    visitor.visit(tree)
    return visitor.calls


def _dotted_name(expression: ast.expr) -> str | None:
    parts: list[str] = []
    current = expression
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


_PUBLIC_PRIVATE_SQLITE_SEAMS = {
    "backup_connection_to_private",
    "connect_private_sqlite",
    "copy_private_sqlite",
    "restore_private_sqlite",
}


def _qualified_private_sqlite_calls(
    source_path: Path,
) -> list[tuple[str, str, ast.Call]]:
    tree = _parse_source(source_path)
    module_aliases: set[str] = set()
    callable_aliases: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "tldw_chatbook.DB.private_sqlite":
                    module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            imported_module = node.module or ""
            if imported_module.endswith("private_sqlite"):
                for alias in node.names:
                    if alias.name in _PUBLIC_PRIVATE_SQLITE_SEAMS:
                        callable_aliases[alias.asname or alias.name] = alias.name
                    elif alias.name == "*":
                        callable_aliases.update(
                            {
                                seam_name: seam_name
                                for seam_name in _PUBLIC_PRIVATE_SQLITE_SEAMS
                            }
                        )
            elif imported_module.endswith("DB") or (
                node.level > 0 and not imported_module
            ):
                for alias in node.names:
                    if alias.name == "private_sqlite":
                        module_aliases.add(alias.asname or "private_sqlite")

    def seam_callable_name(expression: ast.expr) -> str | None:
        if isinstance(expression, ast.Name):
            return callable_aliases.get(expression.id)
        dotted = _dotted_name(expression)
        if dotted is None:
            return None
        module_name, separator, callable_name = dotted.rpartition(".")
        if (
            separator
            and module_name in module_aliases
            and callable_name in _PUBLIC_PRIVATE_SQLITE_SEAMS
        ):
            return callable_name
        return None

    def is_seam_module(expression: ast.expr) -> bool:
        dotted = _dotted_name(expression)
        return dotted in module_aliases if dotted is not None else False

    rebound_aliases: list[tuple[str, ast.expr]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            rebound_aliases.extend(
                (target.id, node.value)
                for target in node.targets
                if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                rebound_aliases.append((node.target.id, node.value))

    changed = True
    while changed:
        changed = False
        for alias, expression in rebound_aliases:
            if alias not in module_aliases and is_seam_module(expression):
                module_aliases.add(alias)
                changed = True
            rebound_name = seam_callable_name(expression)
            if alias not in callable_aliases and rebound_name is not None:
                callable_aliases[alias] = rebound_name
                changed = True

    visitor = _QualifiedCallNodeVisitor(
        lambda call: seam_callable_name(call.func) is not None,
    )
    visitor.visit(tree)
    return [
        (symbol, seam_callable_name(call.func) or "", call)
        for symbol, call in visitor.calls
    ]


def _literal_string_argument(
    call: ast.Call,
    position: int,
    keyword_name: str,
) -> str | None:
    owner_expression: ast.expr | None = (
        call.args[position] if len(call.args) > position else None
    )
    if owner_expression is None:
        owner_expression = next(
            (keyword.value for keyword in call.keywords if keyword.arg == keyword_name),
            None,
        )
    if isinstance(owner_expression, ast.Constant) and isinstance(
        owner_expression.value, str
    ):
        return owner_expression.value
    return None


def _private_sqlite_seam_violations(
    source_path: Path,
    production_module: str,
) -> tuple[list[tuple[str, str, ast.Call]], list[str]]:
    calls = _qualified_private_sqlite_calls(source_path)
    violations: list[str] = []
    for symbol, seam_name, call in calls:
        owner_arguments = [(0, "owner_id")]
        if seam_name == "restore_private_sqlite":
            owner_arguments.append((1, "pre_restore_owner_id"))
        for position, keyword_name in owner_arguments:
            owner_id = _literal_string_argument(
                call,
                position,
                keyword_name,
            )
            if owner_id is None:
                violations.append(
                    f"{production_module}:{symbol}: non-literal {keyword_name}"
                )
                continue
            policy = SQLITE_OWNER_REGISTRY.get(owner_id)
            if policy is None:
                violations.append(
                    f"{production_module}:{symbol}: unknown owner {owner_id!r}"
                )
                continue
            if policy.production_module != production_module:
                violations.append(
                    f"{production_module}:{symbol}: owner {owner_id!r} belongs to "
                    f"{policy.production_module}"
                )
    return calls, violations


def _current_direct_connect_sites() -> Counter[tuple[str, str]]:
    calls: Counter[tuple[str, str]] = Counter()
    for source_path in PRODUCTION_ROOT.rglob("*.py"):
        module = source_path.relative_to(PROJECT_ROOT).with_suffix("").as_posix()
        for symbol, count in _qualified_sqlite_connect_calls(source_path).items():
            calls[(module, symbol)] += count
    return calls


def _parse_source(source_path: Path) -> ast.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(source_path.read_text(encoding="utf-8"))


def _is_named_call(call: ast.Call, owner: str, method: str) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == method
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == owner
    )


def _classifications(row: dict[str, str]) -> set[str]:
    return {value.strip() for value in row["classification"].split(",")}


def _assert_raw_connection_census(
    documented_legacy: Counter[tuple[str, str]],
    current: Counter[tuple[str, str]],
    *,
    seam_exists: bool,
) -> None:
    seam_site = ("tldw_chatbook/DB/private_sqlite", "_connect_registered_sqlite")
    if not seam_exists:
        assert sum(current.values()) == 31
        assert current == documented_legacy
        return

    assert current[seam_site] == 1
    assert current == Counter({seam_site: 1})


def test_inventory_has_stable_unique_connection_and_backup_ids() -> None:
    connection_rows = _inventory_rows("C")
    backup_rows = _inventory_rows("B")

    assert [row["id"] for row in connection_rows] == [
        f"C{number:02d}" for number in range(1, 32)
    ]
    assert [row["id"] for row in backup_rows] == [
        f"B{number:02d}" for number in range(1, 10)
    ]


def test_raw_connection_census_is_qualified_and_transition_aware() -> None:
    documented_legacy = Counter(
        (row["module"], row["symbol"]) for row in _inventory_rows("C")
    )
    current = _current_direct_connect_sites()
    seam_module = "tldw_chatbook/DB/private_sqlite"
    seam_tree = _parse_source(PROJECT_ROOT / f"{seam_module}.py")
    seam_exists = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "connect_private_sqlite"
        for node in seam_tree.body
    )

    if not seam_exists:
        assert len({module for module, _symbol in current}) == 18
    _assert_raw_connection_census(
        documented_legacy,
        current,
        seam_exists=seam_exists,
    )


def test_transition_census_rejects_unapproved_or_duplicate_raw_calls() -> None:
    legacy_site = ("tldw_chatbook/DB/legacy", "Owner.connect")
    seam_site = ("tldw_chatbook/DB/private_sqlite", "_connect_registered_sqlite")
    documented = Counter({legacy_site: 31})

    _assert_raw_connection_census(
        documented,
        Counter({legacy_site: 31}),
        seam_exists=False,
    )
    with pytest.raises(AssertionError):
        _assert_raw_connection_census(
            documented,
            Counter({legacy_site: 30}),
            seam_exists=False,
        )

    _assert_raw_connection_census(
        documented,
        Counter({seam_site: 1}),
        seam_exists=True,
    )
    with pytest.raises(AssertionError):
        _assert_raw_connection_census(
            documented,
            Counter({legacy_site: 7, seam_site: 1}),
            seam_exists=True,
        )
    with pytest.raises(AssertionError):
        _assert_raw_connection_census(
            documented,
            Counter({legacy_site: 7, seam_site: 2}),
            seam_exists=True,
        )
    with pytest.raises(AssertionError):
        _assert_raw_connection_census(
            documented,
            Counter(
                {
                    legacy_site: 7,
                    seam_site: 1,
                    ("tldw_chatbook/new_owner", "open_database"): 1,
                }
            ),
            seam_exists=True,
        )


def test_private_sqlite_seam_calls_use_literal_module_owned_ids() -> None:
    total_calls = 0
    violations: list[str] = []
    for source_path in PRODUCTION_ROOT.rglob("*.py"):
        module = source_path.relative_to(PROJECT_ROOT).with_suffix("").as_posix()
        calls, source_violations = _private_sqlite_seam_violations(
            source_path,
            module,
        )
        total_calls += len(calls)
        violations.extend(source_violations)

    assert total_calls > 0
    assert violations == []


def test_private_sqlite_seam_guard_detects_aliases_and_owner_bypasses(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "seam_bypasses.py"
    source_path.write_text(
        "\n".join(
            (
                "import tldw_chatbook.DB.private_sqlite as private_db",
                "from tldw_chatbook.DB.private_sqlite import "
                "connect_private_sqlite as checked_connect",
                "from tldw_chatbook.DB.private_sqlite import "
                "copy_private_sqlite as checked_copy",
                "from tldw_chatbook.DB.private_sqlite import *",
                "from . import private_sqlite as relative_private_db",
                "",
                "rebound_connect = checked_connect",
                "rebound_copy = checked_copy",
                "rebound_module = private_db",
                "",
                "def valid_alias():",
                "    return checked_connect('writing.local', ':memory:')",
                "",
                "def valid_module_alias():",
                "    return private_db.connect_private_sqlite("
                "'writing.local', ':memory:')",
                "",
                "def star_import_owner():",
                "    return connect_private_sqlite('research.local', ':memory:')",
                "",
                "def dynamic_owner(owner_id):",
                "    return rebound_connect(owner_id, ':memory:')",
                "",
                "def rebound_module_owner(owner_id):",
                "    return rebound_module.connect_private_sqlite("
                "owner_id, ':memory:')",
                "",
                "def relative_module_owner(owner_id):",
                "    return relative_private_db.connect_private_sqlite("
                "owner_id, ':memory:')",
                "",
                "def dynamic_copy_owner(owner_id):",
                "    return rebound_copy(owner_id, 'a.db', 'b.db')",
                "",
                "def mismatched_copy_owner():",
                "    return private_db.copy_private_sqlite("
                "'settings.bulk_backup', 'a.db', 'b.db')",
                "",
                "def dynamic_restore_owners(owner_id, pre_owner_id):",
                "    return restore_private_sqlite("
                "owner_id, pre_owner_id, 'a.db', 'b.db', 'c.db')",
                "",
                "def unknown_owner():",
                "    return checked_connect('unknown.owner', ':memory:')",
                "",
                "def mismatched_owner():",
                "    return checked_connect('research.local', ':memory:')",
            )
        ),
        encoding="utf-8",
    )

    calls, violations = _private_sqlite_seam_violations(
        source_path,
        "tldw_chatbook/Writing_Interop/local_writing_service",
    )

    assert [symbol for symbol, _seam_name, _call in calls] == [
        "valid_alias",
        "valid_module_alias",
        "star_import_owner",
        "dynamic_owner",
        "rebound_module_owner",
        "relative_module_owner",
        "dynamic_copy_owner",
        "mismatched_copy_owner",
        "dynamic_restore_owners",
        "unknown_owner",
        "mismatched_owner",
    ]
    assert len(violations) == 10
    assert any("non-literal owner" in violation for violation in violations)
    assert any(
        "non-literal pre_restore_owner_id" in violation for violation in violations
    )
    assert any("unknown owner 'unknown.owner'" in violation for violation in violations)
    assert any(
        "owner 'research.local' belongs to" in violation for violation in violations
    )


def test_raw_connection_census_detects_sqlite_import_aliases(tmp_path: Path) -> None:
    source_path = tmp_path / "aliased_sqlite.py"
    source_path.write_text(
        "\n".join(
            (
                "import sqlite3 as sql",
                "import sqlite3.dbapi2 as dbapi",
                "from sqlite3 import connect as direct_connect",
                "",
                "def first():",
                "    return sql.connect(':memory:')",
                "",
                "class Owner:",
                "    def second(self):",
                "        return direct_connect(':memory:')",
                "",
                "def third():",
                "    return dbapi.connect(':memory:')",
                "",
                "def fourth():",
                "    from sqlite3 import connect as local_connect",
                "    return local_connect(':memory:')",
            )
        ),
        encoding="utf-8",
    )

    assert _qualified_sqlite_connect_calls(source_path) == Counter(
        {"first": 1, "Owner.second": 1, "third": 1, "fourth": 1}
    )


def test_raw_connection_census_detects_constructor_and_rebound_bypasses(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "sqlite_bypasses.py"
    source_path.write_text(
        "\n".join(
            (
                "import sqlite3",
                "import sqlite3 as sql",
                "import sqlite3.dbapi2 as dbapi",
                "from sqlite3 import Connection as ImportedConnection",
                "from sqlite3 import dbapi2 as imported_dbapi",
                "from sqlite3 import *",
                "",
                "raw_connect = sqlite3.connect",
                "raw_connection = sql.Connection",
                "",
                "def module_constructor():",
                "    return sqlite3.Connection(':memory:')",
                "",
                "def nested_dbapi_constructor():",
                "    return sqlite3.dbapi2.Connection(':memory:')",
                "",
                "def nested_dbapi_connect():",
                "    return sqlite3.dbapi2.connect(':memory:')",
                "",
                "def aliased_module_constructor():",
                "    return dbapi.Connection(':memory:')",
                "",
                "def imported_constructor():",
                "    return ImportedConnection(':memory:')",
                "",
                "def imported_dbapi_constructor():",
                "    return imported_dbapi.Connection(':memory:')",
                "",
                "def rebound_connect():",
                "    return raw_connect(':memory:')",
                "",
                "def rebound_constructor():",
                "    return raw_connection(':memory:')",
                "",
                "def star_connect():",
                "    return connect(':memory:')",
                "",
                "def star_constructor():",
                "    return Connection(':memory:')",
            )
        ),
        encoding="utf-8",
    )

    assert _qualified_sqlite_connect_calls(source_path) == Counter(
        {
            "module_constructor": 1,
            "nested_dbapi_constructor": 1,
            "nested_dbapi_connect": 1,
            "aliased_module_constructor": 1,
            "imported_constructor": 1,
            "imported_dbapi_constructor": 1,
            "rebound_connect": 1,
            "rebound_constructor": 1,
            "star_connect": 1,
            "star_constructor": 1,
        }
    )


def test_raw_connection_census_detects_rebound_module_aliases(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "rebound_sqlite_modules.py"
    source_path.write_text(
        "\n".join(
            (
                "import sqlite3 as sql",
                "from sqlite3 import dbapi2 as dbapi",
                "",
                "raw_module = sql",
                "second_module = raw_module",
                "rebound_dbapi = dbapi",
                "",
                "def direct_rebound():",
                "    return raw_module.connect(':memory:')",
                "",
                "def chained_rebound():",
                "    return second_module.Connection(':memory:')",
                "",
                "def dbapi_rebound():",
                "    return rebound_dbapi.connect(':memory:')",
            )
        ),
        encoding="utf-8",
    )

    assert _qualified_sqlite_connect_calls(source_path) == Counter(
        {
            "direct_rebound": 1,
            "chained_rebound": 1,
            "dbapi_rebound": 1,
        }
    )


def test_every_connection_and_backup_row_links_to_a_matching_policy() -> None:
    for row in [*_inventory_rows("C"), *_inventory_rows("B")]:
        policy = SQLITE_OWNER_REGISTRY[row["owner_id"]]
        assert policy.production_module == row["module"]
        assert _classifications(row) <= {
            kind.value for kind in policy.allowed_target_kinds
        }
        assert _classifications(row) <= ALLOWED_CLASSIFICATIONS
        assert row["disposition"].strip()


def test_registry_is_immutable_complete_and_points_to_production_modules() -> None:
    rows = [
        *_inventory_rows("C"),
        *_inventory_rows("B"),
        *_inventory_rows("P"),
    ]
    documented_owner_ids = {row["owner_id"] for row in rows}

    assert set(SQLITE_OWNER_REGISTRY) == documented_owner_ids
    assert SQLiteOwnerPolicy.__dataclass_params__.frozen
    for owner_id, policy in SQLITE_OWNER_REGISTRY.items():
        assert owner_id.strip() == owner_id
        assert owner_id
        assert policy.reason.strip()
        assert policy.allowed_target_kinds
        assert policy.allowed_target_kinds <= set(SQLiteTargetKind)
        assert (PROJECT_ROOT / f"{policy.production_module}.py").is_file()
        with pytest.raises(FrozenInstanceError):
            policy.reason = "mutated"  # type: ignore[misc]


def test_backup_and_restore_rows_explicitly_opt_into_centralized_backup() -> None:
    backup_rows = _inventory_rows("B")

    assert Counter(row["operation"] for row in backup_rows) == Counter(
        {
            "backup_connection_to_private": 3,
            "copy_private_sqlite": 4,
            "restore_private_sqlite": 2,
        }
    )
    assert all(
        SQLITE_OWNER_REGISTRY[row["owner_id"]].centralized_backup_allowed
        for row in backup_rows
    )


def test_backup_inventory_matches_current_sqlite_and_settings_operations() -> None:
    direct_backup_modules: Counter[str] = Counter()
    for source_path in PRODUCTION_ROOT.rglob("*.py"):
        backup_count = sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "backup"
            for node in ast.walk(_parse_source(source_path))
        )
        if backup_count:
            module = source_path.relative_to(PROJECT_ROOT).with_suffix("").as_posix()
            direct_backup_modules[module] = backup_count

    assert direct_backup_modules == Counter({"tldw_chatbook/DB/private_sqlite": 1})

    settings_path = PROJECT_ROOT / "tldw_chatbook/UI/Tools_Settings_Window.py"
    settings_copy_count = sum(
        isinstance(node, ast.Call) and _is_named_call(node, "shutil", "copy2")
        for node in ast.walk(_parse_source(settings_path))
    )
    assert settings_copy_count == 0

    expected_calls = Counter(
        {
            (
                "tldw_chatbook/DB/ChaChaNotes_DB",
                "CharactersRAGDB.backup_database",
                "backup_connection_to_private",
            ): 1,
            (
                "tldw_chatbook/DB/Client_Media_DB_v2",
                "MediaDatabase.backup_database",
                "backup_connection_to_private",
            ): 1,
            (
                "tldw_chatbook/DB/Prompts_DB",
                "PromptsDatabase.backup_database",
                "backup_connection_to_private",
            ): 1,
            (
                "tldw_chatbook/UI/Tools_Settings_Window",
                "ToolsSettingsWindow._backup_worker",
                "copy_private_sqlite",
            ): 3,
            (
                "tldw_chatbook/UI/Tools_Settings_Window",
                "ToolsSettingsWindow._backup_single_worker",
                "copy_private_sqlite",
            ): 1,
            (
                "tldw_chatbook/UI/Tools_Settings_Window",
                "ToolsSettingsWindow._restore_single_worker",
                "restore_private_sqlite",
            ): 1,
        }
    )
    actual_calls: Counter[tuple[str, str, str]] = Counter()
    for source_path in PRODUCTION_ROOT.rglob("*.py"):
        module = source_path.relative_to(PROJECT_ROOT).with_suffix("").as_posix()
        for symbol, seam_name, _call in _qualified_private_sqlite_calls(source_path):
            if seam_name in {
                "backup_connection_to_private",
                "copy_private_sqlite",
                "restore_private_sqlite",
            }:
                actual_calls[(module, symbol, seam_name)] += 1

    assert actual_calls == expected_calls


def test_parent_creator_inventory_is_checked_and_has_a_disposition() -> None:
    parent_rows = _inventory_rows("P")

    assert {
        (row["module"], row["symbol"], row["creator_call"]) for row in parent_rows
    } == (EXPECTED_PARENT_CREATORS)
    assert len({row["id"] for row in parent_rows}) == len(parent_rows)
    assert all(row["disposition"] in ALLOWED_PARENT_DISPOSITIONS for row in parent_rows)
    assert all(row["state"] in {"current", "migrated"} for row in parent_rows)
    assert all(row["rationale"].strip() for row in parent_rows)
    for row in parent_rows:
        policy = SQLITE_OWNER_REGISTRY[row["owner_id"]]
        assert policy.production_module == row["module"]

    for module, creator_call in {
        (row["module"], row["creator_call"]) for row in parent_rows
    }:
        source_path = PROJECT_ROOT / f"{module}.py"
        creator_calls = _qualified_calls(
            source_path,
            lambda call: ast.unparse(call) == creator_call,
        )
        expected_calls = Counter(
            row["symbol"]
            for row in parent_rows
            if row["module"] == module
            and row["creator_call"] == creator_call
            and row["state"] == "current"
        )
        assert creator_calls == expected_calls


def test_task_three_parent_creators_are_recorded_as_migrated() -> None:
    parent_rows = {row["id"]: row for row in _inventory_rows("P")}

    assert {
        parent_id: parent_rows[parent_id]["state"]
        for parent_id in (
            "P01",
            "P02",
            "P03",
            "P23",
            "P24",
            "P25",
            "P26",
            "P27",
            "P28",
        )
    } == {
        "P01": "migrated",
        "P02": "migrated",
        "P03": "migrated",
        "P23": "migrated",
        "P24": "migrated",
        "P25": "migrated",
        "P26": "migrated",
        "P27": "migrated",
        "P28": "migrated",
    }


def test_task_four_parent_creators_are_recorded_as_migrated() -> None:
    parent_rows = {row["id"]: row for row in _inventory_rows("P")}

    assert {
        parent_id: parent_rows[parent_id]["state"]
        for parent_id in (
            "P06",
            "P07",
            "P08",
            "P09",
            "P10",
            "P11",
            "P12",
            "P13",
            "P14",
            "P15",
        )
    } == {
        "P06": "migrated",
        "P07": "migrated",
        "P08": "migrated",
        "P09": "migrated",
        "P10": "migrated",
        "P11": "migrated",
        "P12": "migrated",
        "P13": "migrated",
        "P14": "migrated",
        "P15": "migrated",
    }


def test_task_five_parent_creators_are_recorded_as_migrated() -> None:
    parent_rows = {row["id"]: row for row in _inventory_rows("P")}

    assert {
        parent_id: parent_rows[parent_id]["state"]
        for parent_id in ("P16", "P17", "P18", "P19")
    } == {
        "P16": "migrated",
        "P17": "migrated",
        "P18": "migrated",
        "P19": "migrated",
    }


def test_parent_creator_discovery_boundary_is_explicit() -> None:
    inventory = INVENTORY_PATH.read_text(encoding="utf-8")
    normalized_inventory = " ".join(inventory.split())

    assert (
        "arbitrary `mkdir` calls do not reveal whether a directory will own SQLite data"
        in normalized_inventory
    )
    assert (
        "A new non-direct database-parent owner must add a checked `P` row"
        in normalized_inventory
    )


def test_legacy_memory_and_parent_semantics_are_preserved_explicitly() -> None:
    connection_rows = {row["id"]: row for row in _inventory_rows("C")}
    parent_rows = {row["id"]: row for row in _inventory_rows("P")}

    for connection_id in ("C01", "C02", "C24", "C25", "C26", "C27", "C28", "C29"):
        assert "memory" in _classifications(connection_rows[connection_id])
        owner_id = connection_rows[connection_id]["owner_id"]
        assert SQLiteTargetKind.MEMORY in (
            SQLITE_OWNER_REGISTRY[owner_id].allowed_target_kinds
        )

    assert parent_rows["P02"]["disposition"] == "remove_obsolete_creation"
    assert parent_rows["P03"]["disposition"] == "remove_obsolete_creation"
    assert parent_rows["P27"]["disposition"] == "secure_default"
    assert parent_rows["P28"]["disposition"] == "secure_default"
    assert SQLITE_OWNER_REGISTRY[
        "runtime.server_parity_parent"
    ].allowed_target_kinds == frozenset({SQLiteTargetKind.PRIVATE_FILE})


def test_explicit_exclusions_and_absence_of_async_owner_are_documented() -> None:
    inventory = INVENTORY_PATH.read_text(encoding="utf-8")

    assert "JSONStorage._create_backup" in inventory
    assert "create_incremental_backup" in inventory
    assert "create_automated_backup" in inventory
    assert "No production `aiosqlite.connect` owner exists." in inventory
    assert not any(
        "aiosqlite.connect" in source_path.read_text(encoding="utf-8")
        for source_path in PRODUCTION_ROOT.rglob("*.py")
    )

    media_tree = _parse_source(PROJECT_ROOT / "tldw_chatbook/DB/Client_Media_DB_v2.py")
    media_placeholders = {
        node.name: node
        for node in media_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"create_incremental_backup", "create_automated_backup"}
    }
    assert set(media_placeholders) == {
        "create_incremental_backup",
        "create_automated_backup",
    }
    for function in media_placeholders.values():
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
        assert len(calls) == 1
        assert isinstance(calls[0].func, ast.Attribute)
        assert calls[0].func.attr == "warning"
        assert isinstance(function.body[-1], ast.Pass)

    tamagotchi_tree = _parse_source(
        PROJECT_ROOT / "tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage.py"
    )
    json_storage = next(
        node
        for node in tamagotchi_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "JSONStorage"
    )
    json_backup = next(
        node
        for node in json_storage.body
        if isinstance(node, ast.FunctionDef) and node.name == "_create_backup"
    )
    assert any(
        isinstance(node, ast.Call) and _is_named_call(node, "shutil", "copy2")
        for node in ast.walk(json_backup)
    )
    assert not any(
        isinstance(node, ast.Call) and _is_sqlite3_connect(node)
        for node in ast.walk(json_backup)
    )
