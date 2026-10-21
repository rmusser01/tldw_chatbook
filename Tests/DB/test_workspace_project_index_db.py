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
