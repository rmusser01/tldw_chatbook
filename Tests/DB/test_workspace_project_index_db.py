"""Atomic derived cache, durable scope and writer ownership contracts."""

from dataclasses import replace
import sqlite3

import pytest

from Tests.Workspaces.project_index_support import (
    make_index_key,
    open_index_store,
    page,
    search_authority,
)
from tldw_chatbook.Workspaces.project_index.models import ProjectIndexPolicy


def enabled_store(tmp_path):
    store = open_index_store(tmp_path)
    key = make_index_key(tmp_path)
    scope = store.configure(
        key,
        ProjectIndexPolicy((), (".py",)),
        enabled=True,
        expected_scope=store.load_status(key).scope,
    )
    claim = store.claim_job(
        scope,
        owner_token="owner-1",
        authority_revision=1,
        expected_job_revision=store.load_status(key).job_revision,
    )
    return store, key, scope, claim


def test_fresh_off_and_clear_never_repeats_scope(tmp_path):
    store = open_index_store(tmp_path)
    key = make_index_key(tmp_path)
    old = store.load_status(key)
    assert old.state == "Off"
    new = store.revoke(key, clear=True)
    assert new.epoch > old.scope.epoch
    with pytest.raises(ValueError, match="stale_scope"):
        store.configure(
            key, ProjectIndexPolicy((), ()), enabled=True, expected_scope=old.scope
        )
    store.close()
    reopened = open_index_store(tmp_path)
    assert reopened.load_status(key).scope == new
    reopened.close()


def test_page_fts_duplicate_and_revoke(tmp_path):
    store, key, scope, claim = enabled_store(tmp_path)
    scanned = page()
    assert store.commit_scan_page(
        scope,
        claim.job_id,
        scanned,
        expected_job_revision=claim.job_revision,
        authority_revision=1,
    )
    status = store.load_status(key)
    assert status.indexed == 1 and status.manifest_revision > 0
    candidates = store.lexical_candidates(
        search_authority(tmp_path, scope), "uniqueword", limit=10
    )
    assert [c.relative_path for c in candidates] == ["src/example.py"]
    assert store.commit_scan_page(
        scope,
        claim.job_id,
        scanned,
        expected_job_revision=claim.job_revision,
        authority_revision=1,
    )
    assert store.load_status(key).manifest_revision == status.manifest_revision
    store.revoke(key, clear=True)
    assert not store.commit_scan_page(
        scope,
        claim.job_id,
        scanned,
        expected_job_revision=claim.job_revision,
        authority_revision=1,
    )
    assert not store.lexical_candidates(
        search_authority(tmp_path, scope), "uniqueword", limit=10
    )
    store.close()


def test_page_rollback_keeps_manifest_text_fts_frontier_and_revision(tmp_path):
    store, key, scope, claim = enabled_store(tmp_path)
    before = store.load_status(key)
    with store.transaction() as conn:
        conn.execute(
            "CREATE TRIGGER fail_frontier BEFORE INSERT ON directories BEGIN SELECT RAISE(ABORT, 'frontier failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="frontier failure"):
        store.commit_scan_page(
            scope,
            claim.job_id,
            page(),
            expected_job_revision=claim.job_revision,
            authority_revision=1,
        )
    after = store.load_status(key)
    assert (after.indexed, after.discovered, after.manifest_revision) == (
        before.indexed,
        before.discovered,
        before.manifest_revision,
    )
    with store.connection() as conn:
        for table in ("files", "chunks", "chunk_fts", "directories", "scan_pages"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == (
                1 if table == "directories" else 0
            )
    store.close()


def test_claim_excludes_second_connection_and_requires_explicit_retirement(tmp_path):
    store, key, scope, claim = enabled_store(tmp_path)
    second = open_index_store(tmp_path)
    with pytest.raises(ValueError, match="job_active"):
        second.claim_job(
            scope,
            owner_token="other",
            authority_revision=1,
            expected_job_revision=claim.job_revision,
        )
    assert not second.commit_scan_page(
        scope,
        claim.job_id,
        page(),
        expected_job_revision=claim.job_revision,
        authority_revision=1,
    )
    assert not second.retire_job(
        scope,
        claim.job_id,
        owner_token="other",
        expected_job_revision=claim.job_revision,
        state="Paused",
    )
    assert store.retire_job(
        scope,
        claim.job_id,
        owner_token="owner-1",
        expected_job_revision=claim.job_revision,
        state="Paused",
    )
    next_claim = second.claim_job(
        scope,
        owner_token="other",
        authority_revision=1,
        expected_job_revision=second.load_status(key).job_revision,
    )
    assert (
        next_claim.job_id != claim.job_id
        and next_claim.job_revision > claim.job_revision
    )
    assert not store.commit_scan_page(
        scope,
        claim.job_id,
        page(),
        expected_job_revision=claim.job_revision,
        authority_revision=1,
    )
    second.close()
    store.close()


def test_wrong_key_authority_and_hostile_query_fail_closed(tmp_path):
    store, key, scope, claim = enabled_store(tmp_path)
    assert not store.commit_scan_page(
        scope,
        claim.job_id,
        page(),
        expected_job_revision=claim.job_revision,
        authority_revision=2,
    )
    store.commit_scan_page(
        scope,
        claim.job_id,
        page(),
        expected_job_revision=claim.job_revision,
        authority_revision=1,
    )
    for query in ('" OR 1=1; DROP TABLE files; --', '"', "NEAR(", "x:*"):
        store.lexical_candidates(search_authority(tmp_path, scope), query, limit=10)
    with pytest.raises(ValueError, match="key_mismatch"):
        store.load_status(replace(key, data_authority_id="restored-authority"))
    assert store.load_status(key).indexed == 1
    store.close()


def test_corrupt_schema_is_not_repaired(tmp_path):
    path = tmp_path / "workspace_project_index.db"
    store = open_index_store(tmp_path)
    with store.transaction() as conn:
        conn.execute("DROP TABLE directories")
    store.close()
    before = path.read_bytes()
    with pytest.raises(ValueError, match="schema"):
        open_index_store(tmp_path)
    assert path.read_bytes() == before


def test_orphan_takeover_requires_positive_exact_retirement_proof(tmp_path):
    store, key, scope, old = enabled_store(tmp_path)
    second = open_index_store(tmp_path)
    with pytest.raises(ValueError, match="retirement_unproven"):
        second.replace_retired_job(
            old,
            owner_token="new",
            authority_revision=1,
            retirement_verifier=lambda claim: False,
        )
    assert store.load_status(key).job_id == old.job_id
    assert second.read_claim(scope) == old
    # Test's observation establishes actual native owner-thread retirement.
    store.close()
    replacement = second.replace_retired_job(
        old,
        owner_token="new",
        authority_revision=1,
        retirement_verifier=lambda claim: claim == old
        and store._thread_local.conn is None,
    )
    assert (
        replacement.job_revision > old.job_revision and replacement.job_id != old.job_id
    )
    with pytest.raises(ValueError, match="stale_job"):
        second.replace_retired_job(
            old,
            owner_token="again",
            authority_revision=1,
            retirement_verifier=lambda claim: True,
        )
    second.close()


def test_prepare_required_and_exclusion_floor_before_top_k(tmp_path):
    from tldw_chatbook.DB.Workspace_Project_Index_DB import prepare_scan_page
    from tldw_chatbook.Utils.sensitive_paths import SensitiveExclusion

    store, key, scope, claim = enabled_store(tmp_path)
    prepared = page(path="secret/first.py")
    with pytest.raises(TypeError, match="prepared_scan_page_required"):
        store.commit_scan_page(
            scope,
            claim.job_id,
            prepared.page,
            expected_job_revision=claim.job_revision,
            authority_revision=1,
        )
    for prepared in (prepared, page(path="src/second.py")):
        store.commit_scan_page(
            scope,
            claim.job_id,
            prepared,
            expected_job_revision=claim.job_revision,
            authority_revision=1,
        )
    authority = search_authority(
        tmp_path, scope, (SensitiveExclusion("subtree", "SECRET"),)
    )
    assert [
        c.relative_path
        for c in store.lexical_candidates(authority, "uniqueword", limit=1)
    ] == ["src/second.py"]
    assert all(c.line_start == 1 and c.line_end == 1 for c in prepared.chunks)
    assert prepare_scan_page(prepared.page) == prepared
    store.close()


def test_models_normalize_relative_paths_and_reject_escape(tmp_path):
    from tldw_chatbook.Workspaces.project_index.models import (
        ProjectIndexPolicy,
        ScanEntry,
    )

    assert ScanEntry("src\\test.py", "file", "id", 0, 0).relative_path == "src/test.py"
    assert ProjectIndexPolicy(("src/./",), (".PY",)).text_suffixes == (".py",)
    for path in ("../secret", "/secret", "C:/secret", "src/../../secret"):
        with pytest.raises(ValueError, match="relative_path"):
            ScanEntry(path, "file", "id", 0, 0)


def test_second_process_cannot_claim_active_writer(tmp_path):
    import subprocess
    import sys

    store, key, scope, claim = enabled_store(tmp_path)
    script = """
import sys
from pathlib import Path
from Tests.Workspaces.project_index_support import make_index_key,open_index_store
root=Path(sys.argv[1]); store=open_index_store(root); key=make_index_key(root)
status=store.load_status(key)
try:
    store.claim_job(status.scope,owner_token='other-process',authority_revision=1,expected_job_revision=status.job_revision)
except ValueError as error:
    assert str(error)=='job_active'
else:
    raise AssertionError('competing writer admitted')
finally:
    store.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert store.read_claim(scope) == claim
    store.close()


def test_delayed_clear_cannot_revoke_new_scope_and_frontier_is_bounded(tmp_path):
    store, key, scope, claim = enabled_store(tmp_path)
    assert store.pending_directories(scope, limit=1) == ("",)
    store.commit_scan_page(
        scope,
        claim.job_id,
        page(),
        expected_job_revision=claim.job_revision,
        authority_revision=1,
    )
    newer = store.configure(
        key, ProjectIndexPolicy(("src",), (".py",)), enabled=True, expected_scope=scope
    )
    with pytest.raises(ValueError, match="stale_scope"):
        store.revoke(key, clear=True, expected_scope=scope)
    assert store.load_status(key).scope == newer
    store.close()


def test_native_capacity_limit_is_partial_and_rolls_back_page(tmp_path):
    store, key, scope, claim = enabled_store(tmp_path)
    with store.connection() as conn:
        pages = conn.execute("PRAGMA page_count").fetchone()[0]
        conn.execute(f"PRAGMA max_page_count={pages + 1}")
    assert not store.commit_scan_page(
        scope,
        claim.job_id,
        page(text="largeword " * 20000),
        expected_job_revision=claim.job_revision,
        authority_revision=1,
    )
    status = store.load_status(key)
    assert status.state == "Partial" and status.indexed == 0
    store.close()


def test_revoke_is_visible_before_cleanup_and_old_cleanup_cannot_erase_enable(
    tmp_path, monkeypatch
):
    store, key, scope, claim = enabled_store(tmp_path)
    store.commit_scan_page(
        scope,
        claim.job_id,
        page(),
        expected_job_revision=claim.job_revision,
        authority_revision=1,
    )
    observer = open_index_store(tmp_path)
    original = store.clear_revoked
    observed = []

    def clear(revoked_scope):
        status = observer.load_status(key)
        assert (
            status.scope == revoked_scope
            and status.state == "Off"
            and status.indexed == 1
        )
        observed.append(status)
        return original(revoked_scope)

    monkeypatch.setattr(store, "clear_revoked", clear)
    revoked = store.revoke(key, clear=True)
    assert observed and observer.load_status(key).indexed == 0
    newer = store.configure(
        key, ProjectIndexPolicy((), (".py",)), enabled=True, expected_scope=revoked
    )
    assert not original(revoked)
    assert observer.load_status(key).scope == newer
    observer.close()
    store.close()


def test_narrowed_processing_policy_filters_old_candidates(tmp_path):
    store, key, scope, claim = enabled_store(tmp_path)
    store.commit_scan_page(
        scope,
        claim.job_id,
        page(),
        expected_job_revision=claim.job_revision,
        authority_revision=1,
    )
    scope = store.configure(
        key,
        ProjectIndexPolicy(("other",), (".txt",)),
        enabled=True,
        expected_scope=scope,
    )
    assert (
        store.lexical_candidates(
            search_authority(tmp_path, scope), "uniqueword", limit=10
        )
        == ()
    )
    store.close()


@pytest.mark.parametrize(
    "mismatch",
    [
        "scope",
        "job_id",
        "owner_token",
        "job_revision",
        "authority_revision",
        "paused",
        "disabled",
    ],
)
def test_scanner_admission_requires_exact_enabled_building_claim(tmp_path, mismatch):
    store, key, scope, claim = enabled_store(tmp_path)
    try:
        assert store.admit_job(claim)
        candidate = claim
        if mismatch == "scope":
            candidate = replace(claim, scope=replace(scope, epoch=scope.epoch + 1))
        elif mismatch in {"job_id", "owner_token"}:
            candidate = replace(claim, **{mismatch: "other"})
        elif mismatch in {"job_revision", "authority_revision"}:
            candidate = replace(claim, **{mismatch: getattr(claim, mismatch) + 1})
        elif mismatch == "paused":
            store.suspend_job(claim, state="Paused", advance=False)
        else:
            store.revoke(key, clear=False, expected_scope=scope)
        assert not store.admit_job(candidate)
    finally:
        store.close()


def test_lexical_policy_filter_matches_path_suffix_runtime_oracle(tmp_path):
    """Canonical metadata keeps Path.suffix semantics on every supported runtime."""
    from hashlib import sha256
    from pathlib import Path
    from tldw_chatbook.DB.Workspace_Project_Index_DB import prepare_scan_page
    from tldw_chatbook.Utils.sensitive_paths import SensitiveExclusion
    from tldw_chatbook.Workspaces.project_index.models import (
        FileCapture,
        ScanEntry,
        ScanPage,
    )

    names = (
        "plain",
        ".hidden",
        ".txt",
        "..txt",
        "...txt",
        "name.txt",
        "name.TXT",
        "name.tar.gz",
        "name.",
        "name..",
        "name..txt",
        "café.Λ",
        "name.ß",
        "file.PY",
        ".hidden.py",
        "name...",
        "...",
    )
    paths = tuple(
        prefix + name
        for prefix in ("", "src/", "src/nested/", "src-peer/")
        for name in names
    )
    text = "oracle marker"
    captures = tuple(
        FileCapture(
            path, "logical", 1, len(text), sha256(text.encode()).hexdigest(), text
        )
        for path in paths
    )
    entries = tuple(
        ScanEntry(c.relative_path, "file", c.identity, c.mtime_ns, c.size_bytes)
        for c in captures
    )
    store = open_index_store(tmp_path)
    key = make_index_key(tmp_path)
    try:
        scope = store.configure(
            key,
            ProjectIndexPolicy((), ()),
            enabled=True,
            expected_scope=store.load_status(key).scope,
        )
        claim = store.claim_job(
            scope,
            owner_token="oracle",
            authority_revision=1,
            expected_job_revision=store.load_status(key).job_revision,
        )
        assert store.commit_scan_page(
            scope,
            claim.job_id,
            prepare_scan_page(ScanPage("", "root", entries, captures, True, 1)),
            expected_job_revision=claim.job_revision,
            authority_revision=1,
        )
        floor = (
            SensitiveExclusion("subtree", "src/nested"),
            SensitiveExclusion("file", ".txt"),
        )
        for prefixes in ((), ("",), ("src",), ("src/nested", "src-peer")):
            for suffixes in (
                (),
                (".txt",),
                (".",),
                ("..txt",),
                (".tar.gz", ".gz"),
                (".λ", ".ss"),
                (".PY",),
            ):
                policy = ProjectIndexPolicy(prefixes, suffixes)
                scope = store.configure(key, policy, enabled=True, expected_scope=scope)
                result = store.lexical_candidates(
                    search_authority(tmp_path, scope, floor), "oracle", limit=100
                )
                expected = {
                    path
                    for path in paths
                    if not path.startswith("src/nested/")
                    and path != ".txt"
                    and (
                        not policy.include_prefixes
                        or any(
                            not prefix
                            or path == prefix
                            or path.startswith(prefix + "/")
                            for prefix in policy.include_prefixes
                        )
                    )
                    and (
                        not policy.text_suffixes
                        or Path(path).suffix.casefold() in policy.text_suffixes
                    )
                }
                assert {candidate.relative_path for candidate in result} == expected, (
                    policy,
                    expected,
                    result,
                )
    finally:
        store.close_native()


def test_scan_unit_atomic_receipts_replay_and_indexed_attempt_frontier(tmp_path):
    from tldw_chatbook.Workspaces.project_index.models import (
        ScanEntry,
        DirectoryReceipt,
        ScanUnit,
    )
    from tldw_chatbook.DB.Workspace_Project_Index_DB import prepare_scan_unit

    store, key, scope, claim = enabled_store(tmp_path)
    try:
        unit = prepare_scan_unit(
            ScanUnit(
                (
                    DirectoryReceipt(
                        "",
                        "root",
                        (
                            ScanEntry("a", "directory", "a", 0, 0),
                            ScanEntry("z", "directory", "z", 0, 0),
                        ),
                        True,
                        False,
                        (),
                    ),
                    DirectoryReceipt("a", "a", (), True, False, ()),
                    DirectoryReceipt("z", "z", (), False, False, ("directory_io",)),
                ),
                (),
                1,
                4,
            )
        )
        assert store.commit_scan_unit(
            scope,
            claim.job_id,
            unit,
            expected_job_revision=claim.job_revision,
            authority_revision=1,
        )
        status = store.load_status(key)
        assert store.pending_directories(scope) == ("z",)
        assert store.eligible_directories(claim) == ()
        assert store.commit_scan_unit(
            scope,
            claim.job_id,
            unit,
            expected_job_revision=claim.job_revision,
            authority_revision=1,
        )
        assert store.load_status(key).manifest_revision == status.manifest_revision
        with store.connection() as connection:
            plan = " ".join(
                str(row[3])
                for row in connection.execute(
                    "EXPLAIN QUERY PLAN SELECT path FROM directories WHERE complete=0 AND attempt_revision<? ORDER BY attempt_revision,path LIMIT 256",
                    (claim.job_revision,),
                )
            )
            assert "directories_eligible" in plan and "TEMP B-TREE" not in plan
            connection.execute(
                "INSERT INTO directories(path,identity,complete,attempt_revision) VALUES('0-new','new',0,0)"
            )
        assert store.eligible_directories(claim) == ("0-new",)
        store.suspend_job(claim, state="Paused")
        resumed = store.replace_retired_job(
            store.read_claim(scope),
            owner_token="owner-2",
            authority_revision=1,
            retirement_verifier=lambda _: True,
        )
        assert store.eligible_directories(resumed) == ("0-new", "z")
    finally:
        store.close_native()


@pytest.mark.parametrize("text", ["bodyonly", "", " \n\t "])
def test_filename_only_keyword_without_body_chunks(tmp_path, text):
    store, key, scope, claim = enabled_store(tmp_path)
    try:
        assert store.commit_scan_page(
            scope,
            claim.job_id,
            page(text, "src/uniquefilename.py"),
            expected_job_revision=claim.job_revision,
            authority_revision=1,
        )
        hits = store.lexical_candidates(
            search_authority(tmp_path, scope), "uniquefilename", limit=10
        )
        assert len(hits) == 1
        assert hits[0].relative_path == "src/uniquefilename.py"
        assert hits[0].line_start == 1
    finally:
        store.close_native()


def _legacy_v2_cache(tmp_path):
    store, key, scope, claim = enabled_store(tmp_path)
    assert store.commit_scan_page(
        scope,
        claim.job_id,
        page("", "src/legacyfilename.py"),
        expected_job_revision=claim.job_revision,
        authority_revision=1,
    )
    with store.transaction() as conn:
        conn.execute("DROP TABLE chunk_fts")
        conn.execute("DELETE FROM chunks")
        conn.execute("CREATE VIRTUAL TABLE chunk_fts USING fts5(text)")
        conn.execute("DROP INDEX directories_eligible")
        conn.execute("ALTER TABLE directories DROP COLUMN attempt_revision")
        conn.execute("DROP TABLE schema_version")
        conn.execute(
            "CREATE TABLE schema_version(version INTEGER PRIMARY KEY CHECK(version=2))"
        )
        conn.execute("INSERT INTO schema_version VALUES(2)")
    path = store.db_path
    store.close_native()
    return path, key, scope, claim


def test_v3_migration_preserves_state_backfills_paths_and_cleanup_skips_allocation(
    tmp_path,
):
    from tldw_chatbook.DB.Workspace_Project_Index_DB import WorkspaceProjectIndexDB

    path, key, scope, claim = _legacy_v2_cache(tmp_path)
    before = path.stat().st_size
    cleanup = WorkspaceProjectIndexDB(
        path,
        cleanup_only=True,
        migration_context=lambda _: pytest.fail("cleanup must not migrate"),
    )
    with cleanup.connection() as conn:
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 2
        assert not conn.execute(
            "SELECT 1 FROM sqlite_schema WHERE name='directories_eligible'"
        ).fetchone()
    cleanup.close_native()
    assert path.stat().st_size == before
    migrated = WorkspaceProjectIndexDB(path)
    try:
        assert migrated.load_status(key).scope == scope
        assert migrated.read_claim(scope) == claim
        assert migrated.pending_directories(scope) == ("",)
        assert migrated.lexical_candidates(
            search_authority(tmp_path, scope), "legacyfilename", limit=10
        )
        with migrated.connection() as conn:
            assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 3
    finally:
        migrated.close_native()


def test_migration_binding_capacity_and_native_serialization(tmp_path, monkeypatch):
    from tldw_chatbook.DB.Workspace_Project_Index_DB import WorkspaceProjectIndexDB
    from tldw_chatbook.Workspaces.project_index import runtime
    from tldw_chatbook.Workspaces.project_index.lifetime import (
        BindingLifetime,
        LifetimeBusy,
    )

    namespace = tmp_path / "binding" / "namespace"
    namespace.mkdir(parents=True)
    path, key, scope, claim = _legacy_v2_cache(namespace)
    factory = object.__new__(runtime._AdmittedStoreFactory)
    factory._unretired = {}
    other = namespace.parent / "old-namespace"
    other.mkdir()
    (other / "retained.bin").write_bytes(b"x" * 65536)
    size = path.stat().st_size
    monkeypatch.setattr(runtime, "MAX_CACHE_BYTES", 1024**2 + 2 * size + 32768)

    def context(store):
        return factory._migration_context(key, store)

    with pytest.raises(ValueError, match="index_cache_limit"):
        WorkspaceProjectIndexDB(path, migration_context=context)
    cleanup = WorkspaceProjectIndexDB(path, cleanup_only=True)
    with cleanup.connection() as conn:
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 2
    cleanup.close_native()
    monkeypatch.setattr(runtime, "MAX_CACHE_BYTES", 8 * 1024**2)
    lease = BindingLifetime.acquire(namespace.parent / key.lifetime_name)
    try:
        with pytest.raises(LifetimeBusy):
            WorkspaceProjectIndexDB(path, migration_context=context)
    finally:
        lease.close()
    migrated = WorkspaceProjectIndexDB(path, migration_context=context)
    migrated.close_native()
    lease = BindingLifetime.acquire(namespace.parent / key.lifetime_name)
    try:
        current = WorkspaceProjectIndexDB(path, migration_context=context)
        assert current.load_status(key).scope == scope
        current.close_native()
    finally:
        lease.close()


def test_v3_native_full_rolls_back_all_ddl_and_keeps_v2_clearable(
    tmp_path, monkeypatch
):
    from tldw_chatbook.DB.Workspace_Project_Index_DB import WorkspaceProjectIndexDB
    from tldw_chatbook.Workspaces.project_index import runtime

    namespace = tmp_path / "binding" / "namespace"
    namespace.mkdir(parents=True)
    path, key, scope, claim = _legacy_v2_cache(namespace)
    legacy = WorkspaceProjectIndexDB(path, cleanup_only=True)
    with legacy.transaction() as conn:
        conn.executemany(
            "INSERT INTO directories VALUES(?,?,0)",
            ((f"long-directory-{i:06}-" + "x" * 80, "id") for i in range(5000)),
        )
    legacy.close_native()
    factory = object.__new__(runtime._AdmittedStoreFactory)
    factory._unretired = {}
    monkeypatch.setattr(runtime, "MAX_CACHE_BYTES", 1024**2 + 2 * path.stat().st_size)
    with pytest.raises(sqlite3.OperationalError) as error:
        WorkspaceProjectIndexDB(
            path, migration_context=lambda store: factory._migration_context(key, store)
        )
    assert error.value.sqlite_errorcode == sqlite3.SQLITE_FULL
    legacy = WorkspaceProjectIndexDB(path, cleanup_only=True)
    try:
        with legacy.connection() as conn:
            assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 2
            assert not conn.execute(
                "SELECT 1 FROM sqlite_schema WHERE name='directories_eligible'"
            ).fetchone()
            assert (
                conn.execute("SELECT COUNT(*) FROM directories").fetchone()[0] >= 5000
            )
            assert (
                conn.execute(
                    "SELECT sql FROM sqlite_schema WHERE name='chunk_fts'"
                ).fetchone()[0]
                == "CREATE VIRTUAL TABLE chunk_fts USING fts5(text)"
            )
        assert legacy.read_claim(scope) == claim
    finally:
        legacy.close_native()


def test_failed_migration_native_close_retains_count_and_binding_capability(
    tmp_path, monkeypatch
):
    from tldw_chatbook.DB.Workspace_Project_Index_DB import WorkspaceProjectIndexDB
    from tldw_chatbook.Workspaces.project_index import runtime
    from tldw_chatbook.Workspaces.project_index.lifetime import (
        BindingLifetime,
        LifetimeBusy,
    )
    from tldw_chatbook.Workspaces.project_index.models import (
        ProjectIndexRetirementUnproven,
    )
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import _core_cached_connection

    namespace = tmp_path / "binding" / "namespace"
    namespace.mkdir(parents=True)
    path, key, _, _ = _legacy_v2_cache(namespace)
    factory = object.__new__(runtime._AdmittedStoreFactory)
    factory._unretired = {}
    factory.current_data_authority = lambda: key.data_authority_id
    held = []

    def fail_migration(conn, suffix):
        held.append(conn)
        original = type(conn).close

        def failed_close(connection):
            if connection is conn:
                raise OSError("injected native migration close failure")
            return original(connection)

        monkeypatch.setattr(type(conn), "close", failed_close)
        raise ValueError("injected DDL failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                WorkspaceProjectIndexDB, "_migrate", staticmethod(fail_migration)
            )
            with pytest.raises(ProjectIndexRetirementUnproven):
                WorkspaceProjectIndexDB(
                    path,
                    migration_context=lambda store: factory._migration_context(
                        key, store
                    ),
                )
        retained_store, lease = factory._unretired[key.binding_namespace]
        assert not lease.closed
        assert _core_cached_connection(retained_store, held[0]) is held[0]
        assert any(item.resource_path == path for item in storage._live_leases)
        with pytest.raises(LifetimeBusy):
            BindingLifetime.acquire(namespace.parent / key.lifetime_name)
        with pytest.raises(ProjectIndexRetirementUnproven):
            factory(key)
        assert len(factory._unretired) == 1
        with pytest.raises(ProjectIndexRetirementUnproven):
            factory.assert_retired()
    finally:
        monkeypatch.undo()
        if factory._unretired:
            retained_store, lease = factory._unretired[key.binding_namespace]
            # Same test owner, after evidence: no product retry or foreign close.
            retained_store.close_native()
            lease.close()


def test_lexical_exclusion_predicates_match_live_oracle_before_top_k(tmp_path):
    """Every fresh run floor filters real FTS candidates before ranking limits."""
    from hashlib import sha256
    from tldw_chatbook.DB.Workspace_Project_Index_DB import (
        _path_allowed,
        prepare_scan_page,
    )
    from tldw_chatbook.Utils.sensitive_paths import SensitiveExclusion
    from tldw_chatbook.Workspaces.project_index.models import (
        FileCapture,
        ScanEntry,
        ScanPage,
    )

    paths = (
        "root.py",
        "secret.py",
        "SECRET.PY",
        "src",
        "src/a.py",
        "src/nested/a.py",
        "src/nested/secret.py",
        "src-peer/a.py",
        "srcpeer/a.py",
        "other/a.py",
        "other/secret.py",
        "other/SECRET.PY",
        "Stra\u00dfe/a.py",
        "STRASSE/b.py",
        "caf\u00e9/\u039b.py",
        "CAF\u00c9/\u03bb.py",
        "other/Stra\u00dfe.py",
        "other/STRASSE.PY",
    )
    text = "equivalence marker"
    captures = tuple(
        FileCapture(
            path, "logical", 1, len(text), sha256(text.encode()).hexdigest(), text
        )
        for path in paths
    )
    entries = tuple(
        ScanEntry(c.relative_path, "file", c.identity, c.mtime_ns, c.size_bytes)
        for c in captures
    )
    floors = (
        (),
        (("subtree", "SRC"),),
        (("subtree", "/src\\nested/"),),
        (("subtree", "\\"),),
        (("file", "/SECRET.PY/"),),
        (("file", "\\src\\a.py"),),
        (("name", "/secret.PY/"),),
        (("direct_children", ""),),
        (("direct_children", "/SRC/"),),
        (("direct_children", "src\\nested"),),
        (("subtree", "STRASSE"),),
        (("file", "CAF\u00c9/\u03bb.PY"),),
        (("name", "strasse.py"),),
        (
            ("subtree", "src/nested"),
            ("file", "root.py"),
            ("name", "secret.py"),
            ("direct_children", "OTHER"),
        ),
        (),
    )
    store = open_index_store(tmp_path)
    key = make_index_key(tmp_path)
    try:
        scope = store.configure(
            key,
            ProjectIndexPolicy((), ()),
            enabled=True,
            expected_scope=store.load_status(key).scope,
        )
        claim = store.claim_job(
            scope,
            owner_token="exclusion-oracle",
            authority_revision=1,
            expected_job_revision=store.load_status(key).job_revision,
        )
        assert store.commit_scan_page(
            scope,
            claim.job_id,
            prepare_scan_page(ScanPage("", "root", entries, captures, True, 1)),
            expected_job_revision=claim.job_revision,
            authority_revision=1,
        )
        ordered = tuple(
            c.relative_path
            for c in store.lexical_candidates(
                search_authority(tmp_path, scope),
                "equivalence",
                limit=100,
            )
        )
        assert set(ordered) == set(paths)
        for raw_floor in floors:
            floor = tuple(SensitiveExclusion(kind, value) for kind, value in raw_floor)
            expected = tuple(path for path in ordered if _path_allowed(path, floor))
            for limit in (1, 100):
                found = store.lexical_candidates(
                    search_authority(tmp_path, scope, floor),
                    "equivalence",
                    limit=limit,
                )
                assert (
                    tuple(c.relative_path for c in found) == expected[:limit]
                ), raw_floor
    finally:
        store.close_native()
