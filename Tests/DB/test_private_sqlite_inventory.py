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
    ("tldw_chatbook/config", "get_user_data_dir"),
    ("tldw_chatbook/config", "load_settings DATABASE_URL setup"),
    ("tldw_chatbook/config", "load_settings USER_DB_BASE_DIR setup"),
    ("tldw_chatbook/Utils/paths", "get_project_databases_dir"),
    ("tldw_chatbook/Utils/paths", "get_user_database_path"),
    ("tldw_chatbook/DB/base_db", "BaseDB.__init__"),
    ("tldw_chatbook/DB/ChaChaNotes_DB", "CharactersRAGDB.__init__"),
    ("tldw_chatbook/DB/ChaChaNotes_DB", "CharactersRAGDB.backup_database"),
    ("tldw_chatbook/DB/Client_Media_DB_v2", "MediaDatabase.__init__"),
    ("tldw_chatbook/DB/Client_Media_DB_v2", "MediaDatabase.backup_database"),
    ("tldw_chatbook/DB/Prompts_DB", "PromptsDatabase.__init__"),
    ("tldw_chatbook/DB/Prompts_DB", "PromptsDatabase.backup_database"),
    ("tldw_chatbook/DB/RAG_Indexing_DB", "RAGIndexingDB.__init__"),
    ("tldw_chatbook/DB/Evals_DB", "EvalsDB.__init__"),
    ("tldw_chatbook/DB/search_history_db", "SearchHistoryDB.__init__"),
    ("tldw_chatbook/Kanban_Interop/local_kanban_db", "open_connection"),
    (
        "tldw_chatbook/Research_Interop/local_research_service",
        "LocalResearchService.__init__",
    ),
    (
        "tldw_chatbook/Writing_Interop/local_writing_service",
        "LocalWritingService.__init__",
    ),
    (
        "tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage",
        "SQLiteStorage.__init__",
    ),
    (
        "tldw_chatbook/UI/Tools_Settings_Window",
        "ToolsSettingsWindow._backup_worker",
    ),
    (
        "tldw_chatbook/UI/Tools_Settings_Window",
        "ToolsSettingsWindow._backup_single_worker",
    ),
    (
        "tldw_chatbook/UI/Tools_Settings_Window",
        "ToolsSettingsWindow._restore_single_database",
    ),
    (
        "tldw_chatbook/Evals/eval_orchestrator",
        "EvaluationOrchestrator._initialize_database",
    ),
    ("tldw_chatbook/Event_Handlers/eval_events", "get_orchestrator"),
    ("tldw_chatbook/app", "TldwCli._init_prompts_service"),
    (
        "tldw_chatbook/Notes/Notes_Library",
        "NotesInteropService.__init__",
    ),
    ("tldw_chatbook/DB/Sync_Client", "executable example setup"),
    (
        "tldw_chatbook/runtime_policy/server_parity_state",
        "build_server_parity_state_repositories",
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


def _current_direct_connect_modules() -> Counter[str]:
    calls: Counter[str] = Counter()
    for source_path in PRODUCTION_ROOT.rglob("*.py"):
        tree = _parse_source(source_path)
        count = sum(
            isinstance(node, ast.Call) and _is_sqlite3_connect(node)
            for node in ast.walk(tree)
        )
        if count:
            module = source_path.relative_to(PROJECT_ROOT).with_suffix("").as_posix()
            calls[module] = count
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


def test_inventory_has_stable_unique_connection_and_backup_ids() -> None:
    connection_rows = _inventory_rows("C")
    backup_rows = _inventory_rows("B")

    assert [row["id"] for row in connection_rows] == [
        f"C{number:02d}" for number in range(1, 32)
    ]
    assert [row["id"] for row in backup_rows] == [
        f"B{number:02d}" for number in range(1, 10)
    ]


def test_inventory_matches_every_current_raw_connection_module_and_count() -> None:
    documented = Counter(row["module"] for row in _inventory_rows("C"))
    current = _current_direct_connect_modules()

    assert sum(current.values()) == 31
    assert len(current) == 18
    assert documented == current


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

    assert {(row["module"], row["symbol"]) for row in parent_rows} == (
        EXPECTED_PARENT_CREATORS
    )
    assert len({row["id"] for row in parent_rows}) == len(parent_rows)
    assert all(row["disposition"] in ALLOWED_PARENT_DISPOSITIONS for row in parent_rows)
    assert all(row["rationale"].strip() for row in parent_rows)
    for row in parent_rows:
        policy = SQLITE_OWNER_REGISTRY[row["owner_id"]]
        assert policy.production_module == row["module"]


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
