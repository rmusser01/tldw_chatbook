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


def _qualified_calls(source_path: Path, predicate) -> Counter[str]:
    visitor = _QualifiedCallVisitor(predicate)
    visitor.visit(_parse_source(source_path))
    return visitor.calls


def _qualified_sqlite_connect_calls(source_path: Path) -> Counter[str]:
    tree = _parse_source(source_path)
    module_aliases: set[str] = set()
    connect_aliases: set[str] = set()
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
                if alias.name == "connect":
                    connect_aliases.add(alias.asname or alias.name)
                elif alias.name == "dbapi2":
                    module_aliases.add(alias.asname or alias.name)

    def is_raw_connect(call: ast.Call) -> bool:
        if isinstance(call.func, ast.Name):
            return call.func.id in connect_aliases
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "connect":
            return False
        root = call.func.value
        while isinstance(root, ast.Attribute):
            root = root.value
        return isinstance(root, ast.Name) and root.id in module_aliases

    visitor = _QualifiedCallVisitor(is_raw_connect)
    visitor.visit(tree)
    return visitor.calls


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
    seam_site = ("tldw_chatbook/DB/private_sqlite", "connect_private_sqlite")
    if not seam_exists:
        assert sum(current.values()) == 31
        assert current == documented_legacy
        return

    assert current[seam_site] == 1
    unexpected = set(current) - set(documented_legacy) - {seam_site}
    assert not unexpected
    for site, count in current.items():
        if site != seam_site:
            assert count <= documented_legacy[site]


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
    seam_site = ("tldw_chatbook/DB/private_sqlite", "connect_private_sqlite")
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

    assert sum(row["operation"] == "Connection.backup" for row in backup_rows) == 3
    assert sum(row["operation"] == "shutil.copy2" for row in backup_rows) == 6
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

    backup_rows = _inventory_rows("B")
    documented_direct = Counter(
        row["module"] for row in backup_rows if row["operation"] == "Connection.backup"
    )
    assert direct_backup_modules == documented_direct

    settings_path = PROJECT_ROOT / "tldw_chatbook/UI/Tools_Settings_Window.py"
    settings_copy_count = sum(
        isinstance(node, ast.Call) and _is_named_call(node, "shutil", "copy2")
        for node in ast.walk(_parse_source(settings_path))
    )
    assert settings_copy_count == 6


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
