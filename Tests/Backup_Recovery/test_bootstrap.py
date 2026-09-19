"""Recovery fences use private local evidence before configuration bootstrap."""


def test_custom_root_pending_operation_blocks_startup(tmp_path):
    from tldw_chatbook.Backup_Recovery.control_records import register_pending
    from tldw_chatbook.Backup_Recovery.bootstrap import startup_permission

    bootstrap = tmp_path / "bootstrap"
    control = tmp_path / "control"
    config = tmp_path / "broken.toml"
    config.write_text("this is not TOML [")
    register_pending(bootstrap, "op1", ("profile1",), control, (config,))
    allowed, reason = startup_permission(config, bootstrap)
    assert allowed is False
    assert reason == "recovery_pending"


import os
from pathlib import Path
import pytest

from tldw_chatbook.Backup_Recovery.bootstrap import startup_permission
from tldw_chatbook.Backup_Recovery.control_records import register_pending


@pytest.fixture
def fence(tmp_path):
    root, control, config = (
        tmp_path / p for p in ("bootstrap", "control", "config.toml")
    )
    config.write_text("invalid [")
    register_pending(root, "pending", ("profile",), control, (config,))
    return root, control, config


@pytest.mark.parametrize(
    "damage", ["custom_missing", "config_replaced", "catalog_lost"]
)
def test_pending_survives_independent_locator_loss(fence, damage):
    root, control, config = fence
    if damage == "config_replaced":
        config.unlink()
        config.write_text("new invalid [")
    assert not control.exists()
    assert startup_permission(config, root) == (False, "recovery_pending")


@pytest.mark.parametrize("raw", [b"{", b'{"version":true}', b'{"version":99}', b"{}"])
def test_unknown_or_corrupt_records_are_preserved(tmp_path, raw):
    root = tmp_path / "bootstrap"
    root.mkdir(mode=0o700)
    evidence = root / "unknown.json"
    evidence.write_bytes(raw)
    evidence.chmod(0o600)
    assert startup_permission(tmp_path / "config", root) == (
        False,
        "recovery_scope_uncertain",
    )
    assert evidence.read_bytes() == raw


def test_interrupted_registration_leaves_fail_closed_evidence(tmp_path, monkeypatch):
    import tldw_chatbook.Backup_Recovery.control_records as records

    config = tmp_path / "config"
    config.write_text("broken [")
    root = tmp_path / "bootstrap"
    root.mkdir(mode=0o700)

    def failed_flush(fd):
        raise OSError("injected_metadata_barrier_failure")

    monkeypatch.setattr(records, "flush_directory", failed_flush)
    with pytest.raises(OSError):
        register_pending(root, "op", ("p",), tmp_path / "control", (config,))
    assert startup_permission(config, root) == (False, "recovery_pending")
    assert len(list(root.glob("pending-*"))) == 1


def test_disjoint_requires_an_intact_local_binding(tmp_path):
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
    )

    root = tmp_path / "bootstrap"
    authority = admission_authority(root)
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    ca, cb = tmp_path / "a.toml", tmp_path / "b.toml"
    ca.write_text("a")
    cb.write_text("b")
    authority.register("a", (a,))
    authority.register("b", (b, cb))
    bind_profile(root, cb, ("b",), root / "admission")
    register_pending(root, "op", ("a",), tmp_path / "control", (ca,))
    assert startup_permission(cb, root) == (True, "startup_allowed")
    cb.write_text("changed mapping")
    assert startup_permission(cb, root) == (False, "recovery_scope_uncertain")


def test_memory_and_foreign_read_only_exemptions_but_private_open_fenced(
    fence, monkeypatch, tmp_path
):
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
    import tldw_chatbook.Backup_Recovery.bootstrap as bootstrap

    root, control, config = fence
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config))
    conn = connect_private_sqlite("db.base", ":memory:")
    conn.close()
    destination = tmp_path / "must-not-exist.db"
    with pytest.raises(bootstrap.RecoveryRequired, match="recovery_pending"):
        connect_private_sqlite("db.base", destination)
    assert not destination.exists()


def test_owned_file_writer_fenced_before_create(fence, monkeypatch, tmp_path):
    from tldw_chatbook.Utils.private_paths import create_private_text
    import tldw_chatbook.Backup_Recovery.bootstrap as bootstrap

    root, _, config = fence
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config))
    destination = tmp_path / "must-not-exist.txt"
    with pytest.raises(bootstrap.RecoveryRequired, match="recovery_pending"):
        create_private_text(destination, "no mutation")
    assert not destination.exists()


def test_partial_authority_initialization_is_uncertain(tmp_path):
    root = tmp_path / "bootstrap"
    root.mkdir(mode=0o700)
    marker = root / "unbound-owner"
    marker.write_text("local enrollment owner\n")
    marker.chmod(0o600)
    assert startup_permission(tmp_path / "config", root) == (
        False,
        "recovery_scope_uncertain",
    )


@pytest.fixture
def local_scope(tmp_path, monkeypatch):
    import tldw_chatbook.Backup_Recovery.bootstrap as bootstrap
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority

    root = tmp_path / "bootstrap"
    config = tmp_path / "config"
    config.write_text("scope1")
    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    authority = admission_authority(root)
    authority.register("profile", (config, data))
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config))
    return root, config, data, authority


def test_unbound_sqlite_lifetime_blocks_enrollment_and_retires_on_worker(local_scope):
    from concurrent.futures import ThreadPoolExecutor
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.control_records import bind_profile
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    root, config, data, authority = local_scope
    first = connect_private_sqlite("db.base", data / "one.db", check_same_thread=False)
    second = connect_private_sqlite("db.base", data / "two.db")
    first.execute("CREATE TABLE entries(value)")
    with pytest.raises(RecoveryRequired, match="close_unenrolled_clients_and_restart"):
        bind_profile(root, config, ("profile",), root / "admission")
    with ThreadPoolExecutor() as executor:
        executor.submit(first.close).result()
    with pytest.raises(RecoveryRequired, match="close_unenrolled_clients_and_restart"):
        bind_profile(root, config, ("profile",), root / "admission")
    second.close()
    bind_profile(root, config, ("profile",), root / "admission")
    assert list(root.glob("profile-*"))


def test_open_stream_retains_enrollment_until_close(local_scope):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.control_records import bind_profile
    from tldw_chatbook.Utils.private_paths import open_private_text_append_stream

    root, config, data, _ = local_scope
    stream = open_private_text_append_stream(data / "notes.txt")
    stream.write("durable")
    with pytest.raises(RecoveryRequired, match="close_unenrolled_clients_and_restart"):
        bind_profile(root, config, ("profile",), root / "admission")
    stream.close()
    bind_profile(root, config, ("profile",), root / "admission")
    assert (data / "notes.txt").read_text() == "durable"


def test_scope_neutral_config_edit_does_not_strand_live_owners(local_scope):
    from tldw_chatbook.Backup_Recovery.control_records import bind_profile
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
    from tldw_chatbook.Utils.private_paths import atomic_private_write_text

    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    first = connect_private_sqlite("db.base", data / "one.db")
    try:
        atomic_private_write_text(config, "new preferences")
        second = connect_private_sqlite("db.base", data / "two.db")
        second.close()
    finally:
        first.close()


def test_maintenance_holder_refuses_ordinary_seam_without_deadlock(local_scope):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    root, config, data, authority = local_scope
    with authority.maintenance(("profile",), 1):
        with pytest.raises(
            RecoveryRequired, match="maintenance_requires_owner_capability"
        ):
            connect_private_sqlite("db.base", data / "must-not-exist.db")
    assert not (data / "must-not-exist.db").exists()


def test_registry_intent_never_becomes_plain_startup_permission(local_scope):
    root, config, data, _ = local_scope
    intent = root / "admission" / "registry.pending.json"
    intent.write_text('{"version":1}')
    intent.chmod(0o600)
    assert startup_permission(config, root) == (False, "recovery_scope_uncertain")
    assert intent.read_text() == '{"version":1}'


def test_separate_process_authority_for_aliased_storage_is_refused(
    local_scope, tmp_path
):
    import subprocess
    import sys
    from tldw_chatbook.Backup_Recovery.admission import Admission

    root, config, data, authority = local_scope
    alias = tmp_path / "alias"
    alias.symlink_to(data, target_is_directory=True)
    independent = Admission(tmp_path / "independent")
    independent.register("other", (alias, config))
    code = """
import sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.control_records import bind_profile
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
try:
    bind_profile(Path(sys.argv[1]), Path(sys.argv[2]), ("other",), Path(sys.argv[3]))
except RecoveryRequired as error:
    print(str(error))
else:
    Path(sys.argv[4]).write_text("unsafe mutation")
"""
    destination = data / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(root),
            str(config),
            str(independent.control_root),
            str(destination),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "conflicting_admission_authority"
    assert not destination.exists()
    assert not list(root.glob("profile-*"))


def test_enrolled_real_process_connection_blocks_maintenance_until_close(local_scope):
    from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
    import selectors
    import subprocess
    import sys
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.control_records import bind_profile

    root, config, data, authority = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    code = """
import sys
from pathlib import Path
import tldw_chatbook.Backup_Recovery.bootstrap as bootstrap
bootstrap.default_bootstrap_root = lambda: Path(sys.argv[1])
from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
conn = connect_private_sqlite("db.base", Path(sys.argv[2]))
conn.execute("CREATE TABLE records(value)")
conn.execute("INSERT INTO records VALUES ('held')")
print("ready", flush=True)
sys.stdin.readline()
conn.commit()
conn.close()
print("retired", flush=True)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(root), str(data / "held.db")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        with selectors.DefaultSelector() as ready:
            ready.register(child.stdout, selectors.EVENT_READ)
            assert ready.select(15), "child did not establish connection"
        assert child.stdout.readline().strip() == "ready"
        with pytest.raises(AdmissionTimeout):
            with authority.maintenance(("profile",), 0.1):
                pass
        child.stdin.write("close\n")
        child.stdin.flush()
        output, error = child.communicate(timeout=15)
        assert child.returncode == 0, error
        assert "retired" in output
        with authority.maintenance(("profile",), 1):
            import sqlite3

            connection = sqlite3.connect(data / "held.db")
            assert connection.execute("SELECT value FROM records").fetchone() == (
                "held",
            )
            connection.close()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_unqualified_native_does_not_bypass_fence_or_block_clean_startup(
    local_scope, monkeypatch
):
    import tldw_chatbook.Backup_Recovery.storage_admission as storage
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    root, config, data, authority = local_scope
    monkeypatch.setattr(
        storage, "qualified_for", lambda *args: (False, "operation_not_qualified")
    )
    with storage.acquire_storage(data / "plain.txt"):
        pass
    register_pending(root, "op", ("profile",), data.parent / "control", (config,))
    with pytest.raises(RecoveryRequired, match="recovery_pending"):
        storage.acquire_storage(data / "plain.txt")


def test_abandoned_connection_retires_native_lease(local_scope):
    import gc
    from tldw_chatbook.Backup_Recovery.control_records import bind_profile
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    root, config, data, _ = local_scope
    connection = connect_private_sqlite("db.base", data / "abandoned.db")
    connection.execute("CREATE TABLE entry(value)")
    del connection
    gc.collect()
    bind_profile(root, config, ("profile",), root / "admission")


def test_read_only_private_owner_is_fenced_but_classified_foreign_source_is_exempt(
    fence, monkeypatch, tmp_path
):
    import sqlite3
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
    import tldw_chatbook.Backup_Recovery.bootstrap as bootstrap

    root, _, config = fence
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config))
    source = tmp_path / "source.db"
    connection = sqlite3.connect(source)
    connection.execute("CREATE TABLE records(value)")
    connection.close()
    source.chmod(0o600)
    with pytest.raises(bootstrap.RecoveryRequired, match="recovery_pending"):
        connect_private_sqlite("rag.chachanotes_keyword_leg", source, read_only=True)
    foreign = connect_private_sqlite("cookies.chrome", source, read_only=True)
    foreign.close()


def test_binding_requires_config_in_declared_admission_scope(tmp_path):
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
    )

    root = tmp_path / "bootstrap"
    config = tmp_path / "config"
    config.write_text("private")
    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    authority = admission_authority(root)
    authority.register("incomplete", (data,))
    with pytest.raises(ValueError, match="selector_not_in_admission_scope"):
        bind_profile(root, config, ("incomplete",), root / "admission")
    assert not list(root.glob("profile-*"))


def test_missing_enrollment_marker_is_not_recreated(local_scope):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage

    root, config, data, _ = local_scope
    marker = root / "unbound-owner"
    marker.unlink()
    with pytest.raises(RecoveryRequired, match="recovery_scope_uncertain"):
        acquire_storage(data / "never-opened.db")
    assert not marker.exists()


def test_unqualified_host_with_intact_binding_and_no_recovery_can_start(
    local_scope, monkeypatch
):
    import tldw_chatbook.Backup_Recovery.storage_admission as storage
    from tldw_chatbook.Backup_Recovery.control_records import bind_profile

    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    monkeypatch.setattr(
        storage, "qualified_for", lambda *args: (False, "operation_not_qualified")
    )
    with storage.acquire_storage(data / "plain.txt"):
        pass


def test_forked_child_cannot_reuse_parent_coordinator(local_scope):
    import subprocess
    import sys

    root, _, data, _ = local_scope
    code = """
import os
import sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage
bootstrap.default_bootstrap_root = lambda: Path(sys.argv[1])
lease = acquire_storage(Path(sys.argv[2]))
child = os.fork()
if child == 0:
    try:
        acquire_storage(Path(sys.argv[2]))
    except bootstrap.RecoveryRequired as error:
        print(str(error), flush=True)
        os._exit(0)
    os._exit(3)
_, status = os.waitpid(child, 0)
lease.close()
raise SystemExit(os.waitstatus_to_exitcode(status))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(root), str(data / "never-written")],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "forked_owner_restart_required"
    assert not (data / "never-written").exists()


def test_enrolled_scope_refuses_ancestor_chmod_but_allows_roots_children_and_file_aliases(
    local_scope,
):
    import stat
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.control_records import bind_profile
    from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage
    from tldw_chatbook.Utils.private_paths import (
        secure_private_directory,
        create_private_text,
    )

    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    ancestor = data.parent
    ancestor.chmod(0o755)
    with pytest.raises(RecoveryRequired, match="storage_scope_not_enrolled"):
        secure_private_directory(ancestor, create=False, application_owned=True)
    assert stat.S_IMODE(ancestor.stat().st_mode) == 0o755
    secure_private_directory(data, create=False, application_owned=True)
    child = data / "child"
    secure_private_directory(child, create=True, application_owned=True)
    create_private_text(child / "owned.txt", "owned")
    assert (child / "owned.txt").read_text() == "owned"
    file_alias = ancestor / "config-alias"
    os.link(config, file_alias)
    with acquire_storage(file_alias):
        pass
    outside = ancestor / "outside"
    outside.mkdir(mode=0o755)
    escaped = data / "escape"
    escaped.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RecoveryRequired, match="storage_scope_not_enrolled"):
        create_private_text(escaped / "undeclared.txt", "no mutation")
    assert not (outside / "undeclared.txt").exists()


def test_deferred_factory_close_retires_native_handle_before_maintenance(local_scope):
    import sqlite3
    from tldw_chatbook.Backup_Recovery.control_records import bind_profile
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    root, config, data, authority = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    calls = []

    class DeferredClose(sqlite3.Connection):
        def close(self):
            calls.append("deferred")

    connection = connect_private_sqlite(
        "db.base", data / "deferred.db", factory=DeferredClose
    )
    connection.execute("CREATE TABLE entries(value)")
    connection.close()
    try:
        with authority.maintenance(("profile",), 0.2):
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                sqlite3.Connection.execute(
                    connection, "INSERT INTO entries VALUES ('unsafe')"
                )
        assert calls == ["deferred"]
    finally:
        sqlite3.Connection.close(connection)


def test_native_unavailable_preserves_disjoint_pending_profile_scope(
    local_scope, monkeypatch
):
    import tldw_chatbook.Backup_Recovery.storage_admission as storage
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.control_records import bind_profile
    from tldw_chatbook.Utils.private_paths import create_private_text

    root, config, data, authority = local_scope
    other_config = data.parent / "other-config"
    other_config.write_text("other profile")
    other_data = data.parent / "other-data"
    other_data.mkdir(mode=0o700)
    authority.register("other", (other_config, other_data))
    bind_profile(root, config, ("profile",), root / "admission")
    register_pending(
        root, "other-operation", ("other",), data.parent / "control", (other_config,)
    )
    assert startup_permission(config, root) == (True, "startup_allowed")
    monkeypatch.setattr(
        storage, "qualified_for", lambda *args: (False, "operation_not_qualified")
    )
    create_private_text(data / "still-usable.txt", "ordinary data")
    assert (data / "still-usable.txt").read_text() == "ordinary data"
    with pytest.raises(RecoveryRequired, match="storage_scope_not_enrolled"):
        create_private_text(other_data / "must-not-exist.txt", "outside scope")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(other_config))
    with pytest.raises(RecoveryRequired, match="recovery_pending"):
        storage.acquire_storage(other_data / "must-not-exist.txt")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config))
    config.write_text("changed scope")
    with pytest.raises(RecoveryRequired, match="recovery_scope_uncertain"):
        storage.acquire_storage(data / "must-not-exist.txt")


def test_failed_explicit_factory_close_retains_admission_without_gc_retry(local_scope):
    import gc
    import sqlite3
    from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
    from tldw_chatbook.Backup_Recovery.control_records import bind_profile
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    root, config, data, authority = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    calls = []

    class FailedClose(sqlite3.Connection):
        def close(self):
            calls.append("failed")
            raise sqlite3.OperationalError("injected_close_failure")

    connection = connect_private_sqlite(
        "db.base", data / "retained.db", factory=FailedClose
    )
    connection.execute("CREATE TABLE entries(value)")
    with pytest.raises(sqlite3.OperationalError, match="injected_close_failure"):
        connection.close()
    connection.execute("INSERT INTO entries VALUES ('still live')")
    connection.commit()
    with pytest.raises(AdmissionTimeout):
        with authority.maintenance(("profile",), 0.1):
            pass
    del connection
    gc.collect()
    assert calls == ["failed"]
    with pytest.raises(AdmissionTimeout):
        with authority.maintenance(("profile",), 0.1):
            pass
