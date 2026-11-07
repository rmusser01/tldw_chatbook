"""TASK-36948.2: outside Backup_Recovery, participant helpers are used by public name.

Covers every Backup_Recovery module whose name contains "participant".  Three forms
are flagged in production code: ``from ...x_participants import _name``,
``module._name`` attribute reads and ``getattr(module, "_name")`` /
``vars(module).get("_name")``.  Module STATE that is rebound or shared
(not a def/class) stays private and is listed in ``STATE_ALLOWLIST`` until it has
an accessor API.

TASK-36948.1: stores declare their own registration (``@declare_store``);
``participants.py`` keeps no store table, and the declared set is frozen here.
"""

import ast
import re
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "tldw_chatbook"
PARTICIPANT = re.compile(r"participants?$")

# (module, attribute) -> why it is still read privately from outside Backup_Recovery.
STATE_ALLOWLIST = {
    ("raw_participants", "_states"): "module state table; mutated in place by callers and tests, which need an accessor API",
    ("raw_participants", "_local"): "thread-local state; needs an accessor API",
    ("raw_participants", "_unlink"): "patched by tests via setattr",
    ("raw_participants", "_CONSOLE_PERMISSION_LOAD_GUARDS"): "provenance data tuple, not a helper",
    ("mcp_source_participants", "_BINDINGS"): "module state registry",
    ("participant", "_condition"): "lock on the imported rag_definition participant singleton, not a helper",
    ("persona_visual_participants", "_repositories"): "module state registry",
}


def _bound_participant_modules(tree: ast.AST) -> dict[str, str]:
    bound = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if PARTICIPANT.search(alias.name):
                    bound[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                leaf = alias.name.split(".")[-1]
                if PARTICIPANT.search(leaf):
                    bound[alias.asname or leaf] = leaf
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            text = ast.unparse(node.value)
            match = re.search(r"Backup_Recovery\.(\w*participants?)[\"']", text)
            if match and ("modules" in text or "import_module" in text):
                bound[node.targets[0].id] = match.group(1)
    return bound


def _private(name: object) -> bool:
    return isinstance(name, str) and name.startswith("_") and not name.startswith("__")


def private_participant_uses(root: Path = PACKAGE, allowlist=None) -> list[str]:
    allowlist = STATE_ALLOWLIST if allowlist is None else allowlist
    found = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if "Backup_Recovery" in rel.parts:
            continue
        start = len(found)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        bound = _bound_participant_modules(tree)

        def flag(node, module, name, form):
            if (module, name) not in allowlist:
                found.append(f"{rel}:{node.lineno} {module}.{name} ({form})")

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and PARTICIPANT.search((node.module or "").split(".")[-1])
            ):
                for alias in node.names:
                    if _private(alias.name):
                        flag(node, node.module.split(".")[-1], alias.name, "import")
            elif (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in bound
                and _private(node.attr)
            ):
                flag(node, bound[node.value.id], node.attr, "attribute")
            elif isinstance(node, ast.Call) and len(node.args) >= 2:
                target = node.args[0]
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and isinstance(target, ast.Name)
                    and target.id in bound
                    and isinstance(node.args[1], ast.Constant)
                    and _private(node.args[1].value)
                ):
                    flag(node, bound[target.id], node.args[1].value, "getattr")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Name)
                and node.func.value.func.id == "vars"
                and node.func.value.args
                and isinstance(node.func.value.args[0], ast.Name)
                and node.func.value.args[0].id in bound
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and _private(node.args[0].value)
            ):
                flag(node, bound[node.func.value.args[0].id], node.args[0].value, "vars().get")
        found[start:] = sorted(found[start:], key=lambda row: int(row.split(":")[1].split()[0]))
    return found


def test_no_private_participant_use_outside_backup_recovery():
    assert private_participant_uses() == []


def test_allowlist_entries_are_still_needed():
    raw_hits = {
        tuple(row.split(" ")[1].split(".", 1)) for row in private_participant_uses(allowlist={})
    }
    assert set(STATE_ALLOWLIST) <= raw_hits, sorted(set(STATE_ALLOWLIST) - raw_hits)


def test_guard_detects_injected_private_uses(tmp_path):
    (tmp_path / "store.py").write_text(
        "from tldw_chatbook.Backup_Recovery.participants import _core_access\n"
        "from ..Backup_Recovery.raw_participants import _scope\n"
        "from ..Backup_Recovery.rag_definition_participant import _x\n"
        "from ..Backup_Recovery.participants import core_access\n"
        "from tldw_chatbook.Backup_Recovery import raw_participants as raw\n"
        "raw._live_state(1)\n"
        "raw._states\n"
        "getattr(raw, '_check_temporary_identity')\n"
        "vars(raw).get('_scope')\n"
    )
    assert private_participant_uses(tmp_path) == [
        "store.py:1 participants._core_access (import)",
        "store.py:2 raw_participants._scope (import)",
        "store.py:3 rag_definition_participant._x (import)",
        "store.py:6 raw_participants._live_state (attribute)",
        "store.py:8 raw_participants._check_temporary_identity (getattr)",
        "store.py:9 raw_participants._scope (vars().get)",
    ]


# TASK-36948.1: each store declares its backup-participant registration with
# @declare_store. owner id -> (class under tldw_chatbook, the decorator's keywords).
# A new store, or a changed closer or policy, updates this table in the same commit.
DECLARED_STORES = {
    "db.agent_runs": ("DB.AgentRuns_DB.AgentRunsDB", "closer='close', policy='db.base', finite_worker=False"),
    "db.chachanotes.primary": ("DB.ChaChaNotes_DB.CharactersRAGDB", "closer='close_connection'"),
    "db.dreams": ("DB.Dreams_DB.DreamsDB", "closer='close', policy='db.base'"),
    "db.evals": ("DB.Evals_DB.EvalsDB", "closer='close'"),
    "db.guardian": ("DB.Guardian_DB.GuardianDB", "closer='close', policy='db.base'"),
    "db.library_collections": ("DB.Library_Collections_DB.LibraryCollectionsDB", "closer='close', policy='db.base'"),
    "db.library_ingest_jobs": ("DB.Library_Ingest_Jobs_DB.LibraryIngestJobsDB", "closer='close', finite_worker=False, shared_connection=True"),
    "db.media.primary": ("DB.Client_Media_DB_v2.MediaDatabase", "closer='close_connection'"),
    "db.prompts.primary": ("DB.Prompts_DB.PromptsDatabase", "closer='close_connection'"),
    "db.rag_indexing": ("DB.RAG_Indexing_DB.RAGIndexingDB", "closer='close', finite_worker=False"),
    "db.scheduled_tasks": ("Scheduling.db.scheduled_tasks_db.ScheduledTasksDB", "policy='db.base'"),
    "db.subscriptions": ("DB.Subscriptions_DB.SubscriptionsDB", "closer='close', policy='db.base', read_only_policy='db.subscriptions.agent_read'"),
    "db.subscriptions.site_configs": ("Subscriptions.site_config_manager.SiteConfigManager", ""),
    "db.workspace_project_index": ("DB.Workspace_Project_Index_DB.WorkspaceProjectIndexDB", "closer='close'"),
    "db.workspaces": ("DB.Workspace_DB.WorkspaceDB", "closer='close', policy='db.base'"),
    "kanban.local": ("Kanban_Interop.local_kanban_service.LocalKanbanService", ""),
    "notes.file_notes": ("Notes.file_notes_replica.FileNotesReplica", "closer='close', policy='notes.file_notes_replica', finite_worker=False, shared_connection=True"),
    "notes.sync_state": ("Notes.notes_device_state_store.NotesDeviceStateStore", "thread_cache=lambda store: getattr(store._thread_local, 'connection', None)"),
    "notifications.client": ("Notifications.client_notifications_db.ClientNotificationsDB", "closer='close', policy='db.base'"),
    "recovered.media": ("Backup_Recovery.recovered_media.RecoveredMedia", ""),
    "research.local": ("Research_Interop.local_research_service.LocalResearchService", "thread_cache=_thread_cache"),
    "runtime.event_state": ("Notifications.event_state_repository.EventStateRepository", "thread_cache=_thread_cache, policy='notifications.event_state'"),
    "runtime.sync_state": ("Sync_Interop.sync_state_repository.SyncStateRepository", "policy='db.base'"),
    "writing.local": ("Writing_Interop.local_writing_service.LocalWritingService", "thread_cache=_thread_cache"),
}


def declared_stores(root: Path = PACKAGE) -> dict[str, tuple[str, str]]:
    """Every ``@declare_store`` class in the source tree, read statically."""
    found = {}
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "declare_store(" not in source:
            continue
        module = ".".join(path.relative_to(root).with_suffix("").parts)
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ClassDef):
                continue
            for decorator in node.decorator_list:
                # Both @declare_store(...) and @module.declare_store(...).
                called = getattr(decorator, "func", None)
                if (getattr(called, "id", None) or getattr(called, "attr", None)) == "declare_store":
                    owner = decorator.args[0].value
                    assert owner not in found, f"{owner} is declared twice"
                    keywords = ", ".join(ast.unparse(keyword) for keyword in decorator.keywords)
                    found[owner] = (f"{module}.{node.name}", keywords)
    return found


def test_declared_store_set_is_frozen():
    assert declared_stores() == DECLARED_STORES

    import importlib

    from tldw_chatbook.Backup_Recovery.participants import repository_types

    for owner, (dotted, _) in DECLARED_STORES.items():
        module, name = dotted.rsplit(".", 1)
        store = getattr(importlib.import_module(f"tldw_chatbook.{module}"), name)
        assert repository_types()[store] == owner


def imported_modules(tree: ast.AST, package: str) -> set[str]:
    """Every module an import in ``tree`` can bind, relative imports resolved against ``package``."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package.rsplit(".", node.level - 1)[0] if node.level else ""
            module = ".".join(part for part in (base, node.module) if part)
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
    return found


def test_participants_keeps_no_store_table():
    tree = ast.parse((PACKAGE / "Backup_Recovery" / "participants.py").read_text(encoding="utf-8"))
    strings = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    policies = {"db.base", "notes.file_notes_replica", "notifications.event_state", "db.subscriptions.agent_read"}
    assert strings & (set(DECLARED_STORES) | policies) == set()
    store_modules = {f"tldw_chatbook.{dotted.rsplit('.', 1)[0]}" for dotted, _ in DECLARED_STORES.values()}
    assert imported_modules(tree, "tldw_chatbook.Backup_Recovery") & store_modules == set()


def test_declare_store_refuses_a_missing_closer_and_a_duplicate_owner():
    from tldw_chatbook.Backup_Recovery.participants import repository_types
    from tldw_chatbook.DB import declare_store, store_declarations
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

    class Store:
        def close(self):
            pass

    with pytest.raises(TypeError, match="clsoe"):
        declare_store("tests.closer_typo", closer="clsoe")(Store)
    with pytest.raises(ValueError, match="db.workspaces"):
        declare_store("db.workspaces", closer="close")(Store)
    assert "tests.closer_typo" not in store_declarations
    assert Store not in repository_types()
    assert repository_types()[WorkspaceDB] == "db.workspaces"


def test_registration_table_is_built_once_not_per_connection_access(tmp_path):
    from tldw_chatbook.Backup_Recovery.participants import repository_types
    from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB

    table = repository_types()
    db = LibraryCollectionsDB(tmp_path / "collections.sqlite")
    try:
        with db.transaction() as connection:
            connection.execute("SELECT 1")
        assert repository_types() is table
        assert table[LibraryCollectionsDB] == "db.library_collections"
    finally:
        db.close()
