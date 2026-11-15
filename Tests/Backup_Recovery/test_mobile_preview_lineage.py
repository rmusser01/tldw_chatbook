"""Pre-constructor preview collisions reject through the actual native read seam."""

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.DB.mobile_preview_lineage import (
    IncompatiblePreviewError,
    reject_incompatible_console_stores,
)


def _preview(path, family, version, marker, *, wal=False):
    connection = sqlite3.connect(path)
    if wal:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        connection.execute("PRAGMA wal_autocheckpoint=0")
    if family == "chat":
        connection.execute(
            "CREATE TABLE db_schema_version(schema_name TEXT,version INTEGER)"
        )
        connection.execute(
            "INSERT INTO db_schema_version VALUES ('rag_char_chat_schema',?)",
            (version,),
        )
    else:
        connection.execute("CREATE TABLE schema_version(version INTEGER)")
        connection.execute("INSERT INTO schema_version VALUES (?)", (version,))
    if marker:
        connection.execute("CREATE TABLE " + marker + "(id INTEGER)")
    connection.commit()
    path.chmod(0o600)
    return connection


def _main_wal(path):
    return {
        suffix: (
            p.stat().st_ino,
            p.stat().st_mode,
            hashlib.sha256(p.read_bytes()).hexdigest(),
        )
        for suffix in ("", "-wal")
        if (p := Path(str(path) + suffix)).exists()
    }


@pytest.mark.parametrize(
    "family,version,marker",
    [
        ("chat", 82, "mobile_consolidation_batches"),
        ("workspaces", 9, "mobile_import_authority_holds"),
        ("runs", 25, "console_command_receipts"),
        ("runs", 26, "mobile_consolidation_batches"),
    ],
)
@pytest.mark.parametrize("wal", [False, True])
def test_actual_constructor_refuses_preview_before_writer_setup(
    tmp_path, monkeypatch, family, version, marker, wal
):
    from tldw_chatbook.Chat.console_execution_owner import ConsoleExecutionOwner
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

    path = tmp_path / "selected.db"
    with closing(_preview(path, family, version, marker, wal=wal)):
        before = _main_wal(path)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("writer setup reached")

        if family == "chat":
            monkeypatch.setattr(CharactersRAGDB, "_initialize_schema", forbidden)
            with pytest.raises(CharactersRAGDBError, match="matching build") as refused:
                CharactersRAGDB(path, "fixture")
            assert isinstance(refused.value.__cause__, IncompatiblePreviewError)
        elif family == "workspaces":
            monkeypatch.setattr(WorkspaceDB, "_open_connection", forbidden)
            with pytest.raises(IncompatiblePreviewError, match="matching build"):
                WorkspaceDB(path)
        else:
            monkeypatch.setattr(AgentRunsDB, "_initialize_schema", forbidden)
            with (
                ConsoleExecutionOwner.acquire(path) as owner,
                pytest.raises(IncompatiblePreviewError, match="matching build"),
            ):
                AgentRunsDB(path, execution_owner=owner)
        assert _main_wal(path) == before
        # Native mode=ro may coordinate SHM; it never explicitly removes it.
        if wal:
            assert Path(str(path) + "-shm").exists()


def test_all_selected_stores_checked_before_first_owner_upgrade(tmp_path):
    paths = tuple(tmp_path / name for name in ("chat.db", "workspaces.db", "runs.db"))
    for path, family, version, marker in zip(
        paths,
        ("chat", "workspaces", "runs"),
        (81, 9, 25),
        (None, None, "console_command_receipts"),
        strict=True,
    ):
        _preview(path, family, version, marker).close()
    before = tuple(_main_wal(path) for path in paths)
    with pytest.raises(IncompatiblePreviewError):
        reject_incompatible_console_stores(*paths)
    assert tuple(_main_wal(path) for path in paths) == before


def test_authoritative_snapshot_and_explicit_owner_paths(tmp_path, monkeypatch):
    from tldw_chatbook import config

    selected = tmp_path / "custom"
    selected.mkdir(mode=0o700)
    paths = selected / "chat.db", selected / "workspace.db", selected / "agent_runs.db"
    _preview(paths[1], "workspaces", 9, "mobile_consolidation_batches").close()
    monkeypatch.setattr(
        config,
        "get_atomic_config_snapshot",
        lambda: SimpleNamespace(
            values={
                "database": {
                    "chachanotes_db_path": str(paths[0]),
                    "workspaces_db_path": str(paths[1]),
                },
                "Paths": {"data_dir": str(tmp_path / "wrong-default")},
            }
        ),
    )
    monkeypatch.setattr(
        config, "load_settings", lambda: pytest.fail("lossy settings projection used")
    )
    with pytest.raises(IncompatiblePreviewError):
        config.preflight_selected_console_stores()
    # A caller-selected Workspace path supersedes the unrelated config sibling.
    config.preflight_selected_console_stores(
        workspace_path=tmp_path / "explicit-absent.db"
    )


def test_repeated_guard_reads_new_wal_marker(tmp_path):
    from tldw_chatbook.DB.Workspace_DB import reject_incompatible_preview

    path = tmp_path / "workspaces.db"
    with closing(_preview(path, "workspaces", 9, None, wal=True)) as writer:
        reject_incompatible_preview(path)
        writer.execute("CREATE TABLE mobile_consolidation_batches(id INTEGER)")
        writer.commit()
        before = _main_wal(path)
        with pytest.raises(IncompatiblePreviewError):
            reject_incompatible_preview(path)
        assert _main_wal(path) == before


def test_corrupt_existing_file_is_not_absence(tmp_path):
    from tldw_chatbook.DB.Workspace_DB import reject_incompatible_preview

    path = tmp_path / "bad.db"
    path.write_bytes(b"retained non-database source")
    path.chmod(0o600)
    before = path.read_bytes()
    with pytest.raises(sqlite3.DatabaseError):
        reject_incompatible_preview(path)
    assert path.read_bytes() == before


def test_checkpointed_wal_preview_allows_only_native_empty_wal_coordination(tmp_path):
    from tldw_chatbook.DB.Workspace_DB import reject_incompatible_preview

    path = tmp_path / "workspaces.db"
    _preview(path, "workspaces", 9, "mobile_consolidation_batches", wal=True).close()
    assert not Path(str(path) + "-wal").exists()
    assert not Path(str(path) + "-shm").exists()
    before = _main_wal(path)
    with pytest.raises(IncompatiblePreviewError):
        reject_incompatible_preview(path)
    import json

    after = _main_wal(path)
    (tmp_path / "wal-preflight-refusal-evidence.json").write_text(
        json.dumps(
            {
                "path": str(path),
                "before": before,
                "after": after,
                "before_wal_exists": False,
                "before_shm_exists": False,
                "after_wal_size": Path(str(path) + "-wal").stat().st_size
                if Path(str(path) + "-wal").exists()
                else None,
                "after_shm_size": Path(str(path) + "-shm").stat().st_size
                if Path(str(path) + "-shm").exists()
                else None,
                "main_wal_changed": after != before,
                "outcome": "IncompatiblePreviewError before writer setup",
            },
            indent=2,
            sort_keys=True,
        )
    )
    assert after[""] == before[""]
    # R173: native nonimmutable SQLite may allocate coordination sidecars.
    # Retain them, and accept only a newly allocated empty WAL; an existing
    # WAL remains subject to the exact identity/mode/content checks above.
    wal_path = Path(str(path) + "-wal")
    if wal_path.exists():
        assert wal_path.stat().st_size == 0
        assert wal_path.stat().st_mode & 0o777 == 0o600
        assert after["-wal"][2] == hashlib.sha256(b"").hexdigest()


def test_only_literal_string_memory_skips_file_preflight(tmp_path, monkeypatch):
    from tldw_chatbook.DB.Workspace_DB import reject_incompatible_preview

    monkeypatch.chdir(tmp_path)
    _preview(
        tmp_path / ":memory:", "workspaces", 9, "mobile_consolidation_batches"
    ).close()
    reject_incompatible_preview(":memory:")
    with pytest.raises(IncompatiblePreviewError):
        reject_incompatible_preview(Path(":memory:"))


def test_ordinary_preflight_preserves_existing_nonprivate_mode(tmp_path):
    from tldw_chatbook.DB.Workspace_DB import reject_incompatible_preview

    path = tmp_path / "workspaces.db"
    _preview(path, "workspaces", 9, "mobile_consolidation_batches").close()
    path.chmod(0o640)
    before = _main_wal(path)
    with pytest.raises(IncompatiblePreviewError):
        reject_incompatible_preview(path)
    assert _main_wal(path) == before


def test_coordinator_path_memory_name_keeps_native_sibling_selection(
    tmp_path, monkeypatch
):
    from tldw_chatbook import config

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        config, "get_atomic_config_snapshot", lambda: SimpleNamespace(values={})
    )
    _preview(tmp_path / "agent_runs.db", "runs", 25, "console_command_receipts").close()
    before = _main_wal(tmp_path / "agent_runs.db")
    with pytest.raises(IncompatiblePreviewError):
        config.preflight_selected_console_stores(
            chat_path=Path(":memory:"), workspace_path=tmp_path / "absent.db"
        )
    assert _main_wal(tmp_path / "agent_runs.db") == before
    config.preflight_selected_console_stores(
        chat_path=":memory:", workspace_path=tmp_path / "absent.db"
    )


def test_preview_lineage_compatibility_exports_have_one_complete_profile_owner():
    from tldw_chatbook.Backup_Recovery import profile_paths
    from tldw_chatbook.DB import mobile_preview_lineage

    for name in (
        "IncompatiblePreviewError",
        "_GUIDANCE",
        "_existing_preview_target",
        "_reject_preview_connection",
        "reject_incompatible_console_stores",
    ):
        assert getattr(mobile_preview_lineage, name) is getattr(profile_paths, name)
    assert profile_paths._existing_preview_target.__globals__ is vars(profile_paths)
    assert profile_paths._reject_preview_connection.__globals__ is vars(profile_paths)


@pytest.mark.parametrize("raw_kind", ["ordinary", "nul"])
def test_profile_preflight_preserves_budget_before_path_expansion_and_stat(
    tmp_path, monkeypatch, raw_kind
):
    from tldw_chatbook.Backup_Recovery import profile_paths, storage_admission
    from tldw_chatbook.Chat.console_trace_read_budget import (
        _TraceReadBudget,
        _TraceReadLimit,
    )
    from tldw_chatbook.DB.mobile_preview_lineage import _existing_preview_target

    path = tmp_path / "never-inspected.db"
    raw = str(path) + ("\x00" if raw_kind == "nul" else "")
    budget = _TraceReadBudget()
    stat_calls = []

    def unexpected_stat(*args, **kwargs):
        stat_calls.append(args)
        pytest.fail("preflight inspected a path before accounting")

    with storage_admission._preview_reads(_read_budget=budget):
        budget.reserve(budget._remaining)
        with monkeypatch.context() as patch:
            patch.setattr(
                profile_paths,
                "os",
                SimpleNamespace(path=profile_paths.os.path, stat=unexpected_stat),
            )
            # Keep the existing absolute-Path identity/no-charge fast path.
            assert profile_paths.lexical_path(path) is path
            assert budget._remaining == 0
            with pytest.raises(_TraceReadLimit, match="mobile_trace_input_limit"):
                _existing_preview_target(raw)
        assert not stat_calls and budget._remaining == 0
