"""Real closed-store proof, immutable validation, and original-cohort ownership."""

import errno
import importlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager

import pytest


def module(name):
    return importlib.import_module(f"tldw_chatbook.{name}")


def closed_store(tmp_path):
    path = tmp_path / "profiles.sqlite3"
    module("TTS.profile_schema").open_profile_store(path).close()
    return path


@pytest.mark.parametrize(
    "mutation,expected",
    [
        (None, "ok"),
        ("PRAGMA user_version = 99", "exact_not_current"),
        ("CREATE TABLE unexpected (value)", "schema_corrupt"),
        (
            (
                "INSERT INTO tts_generation_profiles VALUES "
                "('bad', 'name', 'name', 'audio_cpp', 'model', NULL, 'wav', "
                "1.0, '{}', 1, '2026-01-01T00:00:00.000000Z', "
                "'2026-01-01T00:00:00.000000Z')"
            ),
            "corrupt_data",
        ),
    ],
)
def test_fixed_child_proves_closed_store_and_preserves_validation_errors(
    tmp_path, mutation, expected
):
    path = closed_store(tmp_path)
    if mutation:
        connection = sqlite3.connect(path)
        try:
            connection.execute(mutation)
            connection.commit()
        finally:
            connection.close()
    process = module("DB.private_sqlite_process")
    protocol = module("DB.private_sqlite_protocol")
    deadline = process.OperationDeadline(time.monotonic() + 30)
    with process.HelperAdmission().reserve(
        transient=1, retained=1, deadline=deadline
    ) as reservation:
        lease = process.HelperLease.start(
            protocol.PrepareRequest(str(path), False, False, False),
            operation="tts_exact_current",
            reservation=reservation,
            deadline=deadline,
        )
        try:
            reply = lease.initial_response
            if expected == "ok":
                assert reply["status"] == "ok"
                reply = lease.request("tts_recheck", deadline=deadline)
                assert set(reply) == {"version", "operation", "status", "identity"}
                assert reply["identity"]["main"]["ino"] == path.stat().st_ino
                assert reply["identity"]["wal"] is None
                assert reply["identity"]["shm"] is None
            else:
                assert reply == {
                    "version": 1,
                    "operation": "tts_exact_current",
                    "status": "tts_error",
                    "reason": expected,
                }
                assert lease._failed
                with pytest.raises(process.HelperUnavailableError):
                    lease.request("tts_recheck", deadline=deadline)
                assert lease._child.wait(timeout=2) == 0
        finally:
            lease.close()


def test_schema_compatibility_exports_are_shared_validators(tmp_path):
    schema = module("TTS.profile_schema")
    validation = module("TTS.profile_validation")
    for name in (
        "_validate_schema",
        "_validate_schema_body",
        "validate_profile_store_rows",
        "_stream_exact_store_metadata_evidence",
        "decode_options",
    ):
        assert getattr(schema, name) is getattr(validation, name)
    connection = schema.open_profile_store(tmp_path / "profiles.sqlite3")
    try:
        validation._validate_schema(connection)
        validation.validate_profile_store_rows(connection)
        assert validation._stream_exact_store_metadata_evidence(connection)[1] == (
            0,
            0,
            0,
        )
    finally:
        connection.close()


def test_pin_sidecars_binds_once_and_export_requires_complete_cohort(tmp_path):
    proof_api = module("TTS.profile_sqlite_proof")
    path = closed_store(tmp_path)
    proof = proof_api.TTSProof(path)
    try:
        proof.initialize()
        assert proof.recheck()["wal"] is None
        with pytest.raises(proof_api.TTSProofError):
            proof.export_restore_authority()
        live = sqlite3.connect(path)
        try:
            live.execute("PRAGMA user_version").fetchone()
            proof.pin_sidecars()
            authority = proof.export_restore_authority()
            assert authority.wal.ino == path.with_name(path.name + "-wal").stat().st_ino
            assert authority.shm.ino == path.with_name(path.name + "-shm").stat().st_ino
            assert proof.pin_sidecars() == proof.recheck()
            wal = path.with_name(path.name + "-wal")
            retained = path.with_name("retained-wal")
            wal.rename(retained)
            wal.touch(mode=0o600)
            with pytest.raises(proof_api.TTSProofError):
                proof.pin_sidecars()
            wal.unlink()
            retained.rename(wal)
        finally:
            live.close()
    finally:
        proof.close()


@pytest.mark.parametrize(
    "change", ["mode", "main", "parent", "mixed", "journal", "oversize"]
)
def test_proof_refuses_unsafe_initial_namespace_without_mutating_it(tmp_path, change):
    proof_api = module("TTS.profile_sqlite_proof")
    path = closed_store(tmp_path)
    if change == "mode":
        path.chmod(0o644)
    elif change == "main":
        original = tmp_path / "original"
        path.rename(original)
        path.symlink_to(original)
    elif change == "parent":
        tmp_path.chmod(0o777)
    elif change in {"mixed", "journal"}:
        path.with_name(path.name + ("-wal" if change == "mixed" else "-journal")).touch(
            mode=0o600
        )
    else:
        with path.open("r+b") as stream:
            stream.truncate(576 * 1024 * 1024 + 1)
    before = path.lstat()
    proof = proof_api.TTSProof(path)
    try:
        with pytest.raises(
            (
                proof_api.TTSProofError,
                OSError,
                module("Utils.private_paths").PrivatePathError,
            )
        ):
            proof.initialize()
        assert path.lstat() == before
    finally:
        proof.close()
        tmp_path.chmod(0o700)


def test_proof_recheck_refuses_original_main_replacement(tmp_path):
    proof_api = module("TTS.profile_sqlite_proof")
    path = closed_store(tmp_path)
    proof = proof_api.TTSProof(path)
    try:
        proof.initialize()
        path.rename(tmp_path / "original")
        path.touch(mode=0o600)
        with pytest.raises(proof_api.TTSProofError):
            proof.recheck()
    finally:
        proof.close()


def test_proof_closes_sql_before_raw_pins_and_retains_pins_on_sql_close_failure(
    tmp_path, monkeypatch
):
    proof_api = module("TTS.profile_sqlite_proof")
    path = closed_store(tmp_path)
    real_connect = sqlite3.connect
    views = []

    class CloseFailure(sqlite3.Connection):
        refuse = True

        def close(self):
            if self.refuse:
                raise sqlite3.OperationalError("private close failure")
            super().close()

    def connect(*args, **kwargs):
        result = real_connect(*args, **kwargs, factory=CloseFailure)
        views.append(result)
        return result

    monkeypatch.setattr(proof_api.sqlite3, "connect", connect)
    proof = proof_api.TTSProof(path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            proof.initialize()
        assert os.fstat(proof.file_fd).st_ino == path.stat().st_ino
        views[0].refuse = False
    finally:
        proof.close()
    assert proof.file_fd == proof.parent_fd == -1


def test_deadline_remains_primary_when_evidence_close_also_fails(tmp_path, monkeypatch):
    proof_api = module("TTS.profile_sqlite_proof")
    path = closed_store(tmp_path)
    real_connect = sqlite3.connect
    views = []
    signal = proof_api.TTSProofTimeout()

    class RefuseClose(sqlite3.Connection):
        refuse = True

        def close(self):
            if self.refuse:
                raise sqlite3.OperationalError("private close failure")
            super().close()

    def connect(*args, **kwargs):
        view = real_connect(*args, **kwargs, factory=RefuseClose)
        views.append(view)
        return view

    def expired(*args, **kwargs):
        raise signal

    monkeypatch.setattr(proof_api.sqlite3, "connect", connect)
    monkeypatch.setattr(proof_api, "_validate_schema", expired)
    proof = proof_api.TTSProof(path)
    try:
        with pytest.raises(proof_api.TTSProofTimeout) as caught:
            proof.initialize()
        assert caught.value is signal
        assert os.fstat(proof.file_fd).st_ino == path.stat().st_ino
    finally:
        for view in views:
            view.refuse = False
        proof.close()


def test_large_reference_proof_never_selects_payload_or_serializes(
    tmp_path, monkeypatch
):
    proof_api = module("TTS.profile_sqlite_proof")
    path = closed_store(tmp_path)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "INSERT INTO tts_generation_profiles VALUES "
            "('00000000-0000-4000-8000-000000000001', 'name', 'name', 'audio_cpp', "
            "'model', NULL, 'wav', 1.0, '{}', 1, '2026-01-01T00:00:00.000000Z', "
            "'2026-01-01T00:00:00.000000Z')"
        )
        connection.execute(
            "INSERT INTO tts_profile_clone_references VALUES "
            "('00000000-0000-4000-8000-000000000001', '00000000-0000-4000-8000-000000000099', "
            "zeroblob(8388608), 'private transcript', ?, 8388608, 1000, 24000, 1, 'pcm_s16le', "
            "'2026-01-01T00:00:00.000000Z', '2026-01-01T00:00:00.000000Z', NULL, NULL)",
            ("0" * 64,),
        )
        connection.commit()
    finally:
        connection.close()
    real_connect = sqlite3.connect
    statements = []

    class MetadataOnly(sqlite3.Connection):
        def serialize(self, *args, **kwargs):
            raise AssertionError("proof must not serialize")

        def execute(self, sql, *args, **kwargs):
            normalized = " ".join(sql.lower().split())
            statements.append(normalized)
            if (
                normalized.startswith("select")
                and "from tts_profile_clone_references" in normalized
            ):
                assert "length(wav_bytes)" in normalized
                assert "length(cast(reference_text as blob))" in normalized
                projection = normalized.split("from")[0]
                for field in projection.removeprefix("select ").split(","):
                    assert field.strip() not in {"wav_bytes", "reference_text", "*"}
            return super().execute(sql, *args, **kwargs)

    def connect(*args, **kwargs):
        assert kwargs == {"uri": True, "isolation_level": None}
        assert args[0].startswith("file:/dev/fd/") and args[0].endswith(
            "?mode=ro&immutable=1"
        )
        return real_connect(*args, **kwargs, factory=MetadataOnly)

    monkeypatch.setattr(proof_api.sqlite3, "connect", connect)
    proof = proof_api.TTSProof(path)
    try:
        assert proof.initialize()["wal"] is None
        assert proof.evidence is None
        assert (
            len(
                [
                    sql
                    for sql in statements
                    if "from tts_profile_clone_references" in sql
                ]
            )
            == 1
        )
    finally:
        proof.close()


def test_stream_metadata_ceiling_refuses_first_row_above_bound(tmp_path, monkeypatch):
    validation = module("TTS.profile_validation")
    path = closed_store(tmp_path)
    connection = sqlite3.connect(path)
    try:
        for index in range(2):
            connection.execute(
                "INSERT INTO tts_generation_profiles VALUES (?, ?, ?, 'audio_cpp', 'model', NULL, "
                "'wav', 1.0, '{}', 1, '2026-01-01T00:00:00.000000Z', '2026-01-01T00:00:00.000000Z')",
                (str(index), str(index), str(index)),
            )
        monkeypatch.setattr(validation, "_MAX_EXACT_METADATA_ROWS", 1)
        with pytest.raises(
            module("TTS.profile_errors").ProfileRepositoryError, match="corrupt_data"
        ):
            validation._stream_exact_store_metadata_evidence(connection)
    finally:
        connection.close()


@pytest.mark.parametrize("unsafe", ["mode", "symlink", "hardlink", "directory"])
def test_initial_unsafe_sidecar_keeps_exact_not_current_classification(
    tmp_path, unsafe
):
    proof_api = module("TTS.profile_sqlite_proof")
    path = closed_store(tmp_path)
    wal = path.with_name(path.name + "-wal")
    shm = path.with_name(path.name + "-shm")
    shm.touch(mode=0o600)
    if unsafe == "mode":
        wal.touch(mode=0o644)
        wal.chmod(0o644)
    elif unsafe == "symlink":
        wal.symlink_to(shm)
    elif unsafe == "hardlink":
        os.link(shm, wal)
    else:
        wal.mkdir()
    proof = proof_api.TTSProof(path)
    try:
        with pytest.raises(proof_api.TTSProofError) as caught:
            proof.initialize()
        assert caught.value.reason == "exact_not_current"
    finally:
        proof.close()


def test_restore_export_projects_current_metadata_of_original_sidecar_pins(tmp_path):
    proof_api = module("TTS.profile_sqlite_proof")
    path = closed_store(tmp_path)
    wal = path.with_name(path.name + "-wal")
    shm = path.with_name(path.name + "-shm")
    wal.touch(mode=0o600)
    shm.touch(mode=0o600)
    proof = proof_api.TTSProof(path)
    try:
        proof.initialize()
        with wal.open("ab") as output:
            output.write(b"owned fixture metadata change")
        authority = proof.export_restore_authority()
        assert authority.wal.size == wal.stat().st_size
        assert authority.wal.mtime_ns == wal.stat().st_mtime_ns
    finally:
        proof.close()


def test_actual_child_pins_and_exports_without_parent_original_file_opens(
    tmp_path, monkeypatch
):
    path = closed_store(tmp_path)
    process = module("DB.private_sqlite_process")
    protocol = module("DB.private_sqlite_protocol")
    deadline = process.OperationDeadline(time.monotonic() + 30)
    real_open = os.open
    original_names = {path.name + suffix for suffix in ("", "-wal", "-shm")}

    def no_parent_original_open(selected, flags, *args, **kwargs):
        assert os.fspath(selected).split("/")[-1] not in original_names
        return real_open(selected, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", no_parent_original_open)
    with process.HelperAdmission().reserve(
        transient=1, retained=1, deadline=deadline
    ) as owner:
        lease = process.HelperLease.start(
            protocol.PrepareRequest(str(path), False, False, False),
            operation="tts_exact_current",
            reservation=owner,
            deadline=deadline,
        )
        assert lease.initial_response["status"] == "ok"
        live = sqlite3.connect(path)
        try:
            live.execute("PRAGMA user_version").fetchone()
            assert (
                lease.request("tts_pin_sidecars", deadline=deadline)["status"] == "ok"
            )
            reply = lease.request("tts_export_restore_authority", deadline=deadline)
            authority = protocol.TTSRestoreAuthority.from_payload(reply["identity"])
            assert authority.main.ino == path.stat().st_ino
            assert authority.wal.ino == path.with_name(path.name + "-wal").stat().st_ino
            assert authority.shm.ino == path.with_name(path.name + "-shm").stat().st_ino
            assert lease.request("tts_recheck", deadline=deadline)["status"] == "ok"
            lease.close()
            assert lease.cleanup_state == "reaped"
            assert live.execute("PRAGMA user_version").fetchone()[0] == 4
        finally:
            live.close()


@contextmanager
def substituted_proof_namespace(path, target):
    """Temporarily replace a closed owned fixture, then restore its exact inode."""
    if target == "parent_permissions":
        original_mode = path.parent.stat().st_mode & 0o7777
        path.parent.chmod(0o777)
        try:
            yield
        finally:
            path.parent.chmod(original_mode)
        return
    selected = (
        path.parent
        if target == "parent"
        else path.with_name(path.name + ("" if target == "main" else "-" + target))
    )
    retained = selected.with_name("retained-" + selected.name)
    selected.rename(retained)
    try:
        if target == "parent":
            selected.mkdir(mode=0o700)
            path.touch(mode=0o600)
        else:
            selected.touch(mode=0o600)
        yield
    finally:
        if target == "parent":
            path.unlink()
            selected.rmdir()
        else:
            selected.unlink()
        retained.rename(selected)


@pytest.mark.parametrize(
    "target", ["main", "wal", "shm", "parent", "parent_permissions"]
)
def test_actual_lease_retains_original_proof_across_authority_refusal(
    tmp_path, monkeypatch, target
):
    store = tmp_path / "store"
    store.mkdir(mode=0o700)
    path = closed_store(store)
    for suffix in ("-wal", "-shm"):
        path.with_name(path.name + suffix).touch(mode=0o600)
    process = module("DB.private_sqlite_process")
    protocol = module("DB.private_sqlite_protocol")
    admission = process.HelperAdmission()
    pin_observations = tmp_path / "pin-observations.jsonl"
    real_popen = subprocess.Popen

    def launch(args, **kwargs):
        # Observe the real proof's original descriptor identities on each
        # shared pin-check return (including refusals), without changing operations.
        script = f"""
import json,os,runpy,sys
def observe(frame,event,arg):
    if event == 'return' and frame.f_code.co_name == '_recheck_original_pins' and frame.f_globals.get('__name__') == 'tldw_chatbook.TTS.profile_sqlite_proof':
        proof = frame.f_locals['self']
        descriptors = dict(parent=proof.parent_fd,main=proof.file_fd,**proof.sidecars)
        observation = {{name:os.fstat(fd).st_ino for name,fd in descriptors.items()}}
        observation['pid'] = os.getpid()
        observation['sql_closed'] = proof.evidence is None
        with open({str(pin_observations)!r},'a') as output:
            output.write(json.dumps(observation)+'\\n')
sys.setprofile(observe)
sys.argv = [{args[-1]!r}]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
        return real_popen([args[0], "-I", "-S", "-c", script], **kwargs)

    monkeypatch.setattr(process.subprocess, "Popen", launch)
    deadline = process.OperationDeadline(time.monotonic() + 30)
    lease = None
    try:
        with admission.reserve(transient=1, retained=1, deadline=deadline) as owner:
            lease = process.HelperLease.start(
                protocol.PrepareRequest(str(path), False, False, False),
                operation="tts_exact_current",
                reservation=owner,
                deadline=deadline,
            )
            original = lease.initial_response["identity"]
            child = lease._child
            pid = child.pid
            owner.handoff_retained(lease)
        with admission.reserve(transient=4, retained=3, deadline=deadline):
            with substituted_proof_namespace(path, target):
                refused = lease.request("tts_recheck", deadline=deadline)
                assert refused == {
                    "version": 1,
                    "operation": "tts_recheck",
                    "status": "tts_error",
                    "reason": "operation_failed",
                }
                assert not lease._failed
                assert lease.cleanup_state == "still_owned"
                assert child.poll() is None and lease._child.pid == pid
                with pytest.raises(process.HelperTimeoutError):
                    admission.reserve(
                        transient=0,
                        retained=1,
                        deadline=process.OperationDeadline(time.monotonic()),
                    )
                for operation in (
                    "tts_recheck",
                    "tts_pin_sidecars",
                    "tts_export_restore_authority",
                ):
                    assert lease.request(operation, deadline=deadline) == {
                        **refused,
                        "operation": operation,
                    }
                observed = [
                    json.loads(line)
                    for line in pin_observations.read_text().splitlines()
                ]
                assert len(observed) >= 5
                assert all(
                    item
                    == {
                        **{
                            key: original[key]["ino"]
                            for key in ("parent", "main", "wal", "shm")
                        },
                        "pid": pid,
                        "sql_closed": True,
                    }
                    for item in observed
                )
            restored = lease.request("tts_recheck", deadline=deadline)
            assert restored["status"] == "ok"
            for key in ("parent", "main", "wal", "shm"):
                assert restored["identity"][key]["ino"] == original[key]["ino"]
                assert restored["identity"][key]["dev"] == original[key]["dev"]
            authority = protocol.TTSRestoreAuthority.from_payload(
                lease.request("tts_export_restore_authority", deadline=deadline)[
                    "identity"
                ]
            )
            assert authority.main.ino == original["main"]["ino"]
            assert authority.wal.ino == original["wal"]["ino"]
            assert authority.shm.ino == original["shm"]["ino"]
            assert child.poll() is None and lease._child.pid == pid
        lease.close()
        assert lease.cleanup_state == "reaped" and child.poll() == 0
        with admission.reserve(transient=4, retained=4, deadline=deadline):
            pass
    finally:
        if lease is not None:
            lease.close()


def test_actual_lease_can_close_healthy_child_while_namespace_is_refused(tmp_path):
    path = closed_store(tmp_path)
    process = module("DB.private_sqlite_process")
    protocol = module("DB.private_sqlite_protocol")
    deadline = process.OperationDeadline(time.monotonic() + 30)
    with process.HelperAdmission().reserve(
        transient=1, retained=1, deadline=deadline
    ) as owner:
        lease = process.HelperLease.start(
            protocol.PrepareRequest(str(path), False, False, False),
            operation="tts_exact_current",
            reservation=owner,
            deadline=deadline,
        )
        assert lease.initial_response["status"] == "ok"
        with substituted_proof_namespace(path, "main"):
            assert (
                lease.request("tts_recheck", deadline=deadline)["status"] == "tts_error"
            )
            assert not lease._failed
            lease.close()
            assert lease._child.returncode == 0
            assert lease.cleanup_state == "reaped"


def test_failed_initializer_exits_without_accepting_queued_control_frame(tmp_path):
    path = closed_store(tmp_path)
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE unexpected (value)")
        connection.commit()
    finally:
        connection.close()
    protocol = module("DB.private_sqlite_protocol")
    process = module("DB.private_sqlite_process")
    request = {
        "version": 1,
        "operation": "tts_exact_current",
        "path": str(path),
        "writable": False,
        "create_if_missing": False,
        "preserve_source_mode": False,
    }
    control = {"version": 1, "operation": "tts_recheck"}
    entry = os.path.join(
        os.path.dirname(process.__file__), "private_sqlite_helper_entry.py"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", entry],
        input=protocol.encode_frame(request) + protocol.encode_frame(control),
        capture_output=True,
        timeout=3,
        check=False,
        env={**os.environ, "_TLDW_PRIVATE_SQLITE_PARENT_PID": str(os.getpid())},
    )
    assert result.returncode == 0 and result.stderr == b""
    # Decoding exactly one frame also refuses any attempted control reply.
    assert protocol.decode_frame(result.stdout) == {
        "version": 1,
        "operation": "tts_exact_current",
        "status": "tts_error",
        "reason": "schema_corrupt",
    }


def test_actual_first_pin_nofollow_refusal_restores_absent_then_complete_cohort(
    tmp_path,
):
    path = closed_store(tmp_path)
    process = module("DB.private_sqlite_process")
    protocol = module("DB.private_sqlite_protocol")
    deadline = process.OperationDeadline(time.monotonic() + 30)
    wal = path.with_name(path.name + "-wal")
    shm = path.with_name(path.name + "-shm")
    with process.HelperAdmission().reserve(
        transient=1, retained=1, deadline=deadline
    ) as owner:
        lease = process.HelperLease.start(
            protocol.PrepareRequest(str(path), False, False, False),
            operation="tts_exact_current",
            reservation=owner,
            deadline=deadline,
        )
        original = lease.initial_response["identity"]
        child = lease._child
        assert original["wal"] is None and original["shm"] is None
        wal.symlink_to(path)
        try:
            assert lease.request("tts_pin_sidecars", deadline=deadline) == {
                "version": 1,
                "operation": "tts_pin_sidecars",
                "status": "tts_error",
                "reason": "operation_failed",
            }
            assert not lease._failed and child.poll() is None
        finally:
            wal.unlink()
        restored = lease.request("tts_recheck", deadline=deadline)
        assert restored["status"] == "ok"
        assert restored["identity"]["main"]["ino"] == original["main"]["ino"]
        assert (
            restored["identity"]["wal"] is None and restored["identity"]["shm"] is None
        )
        wal.touch(mode=0o600)
        shm.touch(mode=0o600)
        assert lease.request("tts_pin_sidecars", deadline=deadline)["status"] == "ok"
        authority = protocol.TTSRestoreAuthority.from_payload(
            lease.request("tts_export_restore_authority", deadline=deadline)["identity"]
        )
        assert authority.wal.ino == wal.stat().st_ino
        assert authority.shm.ino == shm.stat().st_ino
        assert lease._child is child and child.poll() is None
        lease.close()
        assert child.returncode == 0 and lease.cleanup_state == "reaped"


@pytest.mark.parametrize("target", ["wal", "main", "parent", "parent_permissions"])
def test_actual_partial_capture_retains_pins_until_exact_authority_restoration(
    tmp_path, monkeypatch, target
):
    store = tmp_path / "store"
    store.mkdir(mode=0o700)
    path = closed_store(store)
    wal = path.with_name(path.name + "-wal")
    shm = path.with_name(path.name + "-shm")
    process = module("DB.private_sqlite_process")
    protocol = module("DB.private_sqlite_protocol")
    admission = process.HelperAdmission()
    observations = tmp_path / "capture-pins.jsonl"
    real_popen = subprocess.Popen

    def launch(args, **kwargs):
        script = f"""
import json,os,runpy,sys
opens = {{'main':0,'wal':0}}
def observe(frame,event,arg):
    if event == 'return' and frame.f_code.co_name == '_open_artifact_fd' and isinstance(arg,int):
        leaf = frame.f_locals['leaf']
        if leaf == {path.name!r}:
            opens['main'] += 1
        elif leaf == {wal.name!r}:
            opens['wal'] += 1
    if event == 'return' and frame.f_code.co_name == 'pin_sidecars' and frame.f_globals.get('__name__') == 'tldw_chatbook.TTS.profile_sqlite_proof':
        proof = frame.f_locals['self']
        pins = dict(parent=proof.parent_fd,main=proof.file_fd,**proof.sidecars)
        observation = {{name:[fd,os.fstat(fd).st_ino] for name,fd in pins.items()}}
        observation['pid'] = os.getpid()
        observation['sql_closed'] = proof.evidence is None
        observation['opens'] = dict(opens)
        with open({str(observations)!r},'a') as output:
            output.write(json.dumps(observation)+'\\n')
sys.setprofile(observe)
sys.argv = [{args[-1]!r}]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
        return real_popen([args[0], "-I", "-S", "-c", script], **kwargs)

    monkeypatch.setattr(process.subprocess, "Popen", launch)
    deadline = process.OperationDeadline(time.monotonic() + 30)
    lease = None
    try:
        with admission.reserve(transient=1, retained=1, deadline=deadline) as owner:
            lease = process.HelperLease.start(
                protocol.PrepareRequest(str(path), False, False, False),
                operation="tts_exact_current",
                reservation=owner,
                deadline=deadline,
            )
            assert lease.initial_response["status"] == "ok"
            child = lease._child
            owner.handoff_retained(lease)
        wal.touch(mode=0o600)
        shm.symlink_to(path)
        refusal = {
            "version": 1,
            "operation": "tts_pin_sidecars",
            "status": "tts_error",
            "reason": "operation_failed",
        }
        with admission.reserve(transient=4, retained=3, deadline=deadline):
            assert lease.request("tts_pin_sidecars", deadline=deadline) == refusal
            first = json.loads(observations.read_text().splitlines()[0])
            assert set(first) == {"parent", "main", "wal", "pid", "sql_closed", "opens"}
            assert first["wal"][1] == wal.stat().st_ino
            assert first["opens"] == {"main": 1, "wal": 1}
            shm.unlink()
            for operation in ("tts_recheck", "tts_export_restore_authority"):
                assert lease.request(operation, deadline=deadline) == {
                    **refusal,
                    "operation": operation,
                }
            # Missing SHM still refuses without forgetting the acquired WAL.
            assert lease.request("tts_pin_sidecars", deadline=deadline) == refusal
            with substituted_proof_namespace(path, target):
                # Even a valid missing sidecar must not bind under changed authority.
                shm.touch(mode=0o600)
                try:
                    assert (
                        lease.request("tts_pin_sidecars", deadline=deadline) == refusal
                    )
                finally:
                    shm.unlink()
                assert not lease._failed and lease.cleanup_state == "still_owned"
                assert child.poll() is None
                with pytest.raises(process.HelperTimeoutError):
                    admission.reserve(
                        transient=0,
                        retained=1,
                        deadline=process.OperationDeadline(time.monotonic()),
                    )
            seen = [json.loads(line) for line in observations.read_text().splitlines()]
            assert len(seen) == 3
            assert all(item == first for item in seen)
            assert first["pid"] == child.pid and first["sql_closed"]
            shm.touch(mode=0o600)
            assert (
                lease.request("tts_pin_sidecars", deadline=deadline)["status"] == "ok"
            )
            complete = json.loads(observations.read_text().splitlines()[-1])
            assert {key: complete[key] for key in first} == first
            assert complete["shm"][1] == shm.stat().st_ino
            authority = protocol.TTSRestoreAuthority.from_payload(
                lease.request("tts_export_restore_authority", deadline=deadline)[
                    "identity"
                ]
            )
            assert authority.wal.ino == first["wal"][1]
            assert authority.shm.ino == complete["shm"][1]
            with substituted_proof_namespace(path, "shm"):
                assert lease.request("tts_pin_sidecars", deadline=deadline) == refusal
            assert lease.request("tts_recheck", deadline=deadline)["status"] == "ok"
        lease.close()
        assert child.returncode == 0 and lease.cleanup_state == "reaped"
        with admission.reserve(transient=4, retained=4, deadline=deadline):
            pass
    finally:
        if lease is not None:
            lease.close()


@pytest.mark.parametrize("boundary", ["open", "identity"])
@pytest.mark.parametrize(
    "number",
    [
        errno.ELOOP,
        errno.EACCES,
        errno.EPERM,
        errno.ENOENT,
        errno.ENOTDIR,
        errno.EBADF,
        errno.EIO,
        errno.EMFILE,
        None,
    ],
)
def test_actual_capture_errno_classification(tmp_path, monkeypatch, number, boundary):
    path = closed_store(tmp_path)
    process = module("DB.private_sqlite_process")
    protocol = module("DB.private_sqlite_protocol")
    real_popen = subprocess.Popen

    def launch(args, **kwargs):
        # Inject one filesystem-boundary failure after successful initialization.
        script = f"""
import runpy,sys
def inject(frame,event,arg):
    if event == 'call' and frame.f_code.co_name == 'pin_sidecars' and frame.f_globals.get('__name__') == 'tldw_chatbook.TTS.profile_sqlite_proof':
        proof = frame.f_locals['self']
        original = frame.f_globals['_open_artifact_fd'] if {boundary!r} == 'open' else proof._file_identity
        def refused(*args,**kwargs):
            if {boundary!r} == 'open':
                frame.f_globals['_open_artifact_fd'] = original
            else:
                proof._file_identity = original
            if {number!r} is None:
                raise RuntimeError('private-test-detail')
            raise OSError({number},'private-test-detail')
        if {boundary!r} == 'open':
            frame.f_globals['_open_artifact_fd'] = refused
        else:
            proof._file_identity = refused
        sys.setprofile(None)
sys.setprofile(inject)
sys.argv = [{args[-1]!r}]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
        return real_popen([args[0], "-I", "-S", "-c", script], **kwargs)

    monkeypatch.setattr(process.subprocess, "Popen", launch)
    deadline = process.OperationDeadline(time.monotonic() + 30)
    with process.HelperAdmission().reserve(
        transient=1, retained=1, deadline=deadline
    ) as owner:
        lease = process.HelperLease.start(
            protocol.PrepareRequest(str(path), False, False, False),
            operation="tts_exact_current",
            reservation=owner,
            deadline=deadline,
        )
        assert lease.initial_response["status"] == "ok"
        if number in {errno.EBADF, errno.EIO, errno.EMFILE, None}:
            with pytest.raises(process.HelperUnavailableError):
                lease.request("tts_pin_sidecars", deadline=deadline)
            assert lease._failed
            assert lease._child.wait(timeout=2) == 0
            with pytest.raises(process.HelperUnavailableError):
                lease.request("tts_recheck", deadline=deadline)
        else:
            assert lease.request("tts_pin_sidecars", deadline=deadline) == {
                "version": 1,
                "operation": "tts_pin_sidecars",
                "status": "tts_error",
                "reason": "operation_failed",
            }
            assert not lease._failed and lease._child.poll() is None
            assert lease.request("tts_recheck", deadline=deadline)["status"] == "ok"
        lease.close()
        assert lease.cleanup_state == "reaped"


def test_actual_capture_privacy_refusal_never_remints_acquired_shm(tmp_path):
    path = closed_store(tmp_path)
    process = module("DB.private_sqlite_process")
    protocol = module("DB.private_sqlite_protocol")
    deadline = process.OperationDeadline(time.monotonic() + 30)
    wal = path.with_name(path.name + "-wal")
    shm = path.with_name(path.name + "-shm")
    with process.HelperAdmission().reserve(
        transient=1, retained=1, deadline=deadline
    ) as owner:
        lease = process.HelperLease.start(
            protocol.PrepareRequest(str(path), False, False, False),
            operation="tts_exact_current",
            reservation=owner,
            deadline=deadline,
        )
        assert lease.initial_response["status"] == "ok"
        wal.touch(mode=0o600)
        shm.touch(mode=0o600)
        shm.chmod(0o644)
        original_shm = shm.stat().st_ino
        refusal = {
            "version": 1,
            "operation": "tts_pin_sidecars",
            "status": "tts_error",
            "reason": "operation_failed",
        }
        assert lease.request("tts_pin_sidecars", deadline=deadline) == refusal
        with substituted_proof_namespace(path, "shm"):
            assert lease.request("tts_pin_sidecars", deadline=deadline) == refusal
            assert lease.request("tts_export_restore_authority", deadline=deadline) == {
                **refusal,
                "operation": "tts_export_restore_authority",
            }
        shm.chmod(0o600)
        assert lease.request("tts_pin_sidecars", deadline=deadline)["status"] == "ok"
        authority = protocol.TTSRestoreAuthority.from_payload(
            lease.request("tts_export_restore_authority", deadline=deadline)["identity"]
        )
        assert authority.shm.ino == original_shm
        assert not lease._failed and lease._child.poll() is None
        lease.close()
        assert lease.cleanup_state == "reaped" and lease._child.returncode == 0


def test_actual_initial_unsafe_parent_keeps_private_path_classification(tmp_path):
    store = tmp_path / "store"
    store.mkdir(mode=0o700)
    path = closed_store(store)
    process = module("DB.private_sqlite_process")
    protocol = module("DB.private_sqlite_protocol")
    deadline = process.OperationDeadline(time.monotonic() + 30)
    with process.HelperAdmission().reserve(
        transient=1, retained=1, deadline=deadline
    ) as owner:
        with substituted_proof_namespace(path, "parent_permissions"):
            lease = process.HelperLease.start(
                protocol.PrepareRequest(str(path), False, False, False),
                operation="tts_exact_current",
                reservation=owner,
                deadline=deadline,
            )
            assert lease.initial_response == {
                "version": 1,
                "operation": "tts_exact_current",
                "status": "private_path_error",
                "privacy_status": "unsafe_parent",
                "reason": "shared_writable_parent",
            }
            assert lease._failed and lease._child.wait(timeout=2) == 0
        with pytest.raises(process.HelperUnavailableError):
            lease.request("tts_recheck", deadline=deadline)
        lease.close()
        assert lease.cleanup_state == "reaped"


@pytest.mark.parametrize("operation", ["tts_recheck", "tts_pin_sidecars"])
@pytest.mark.parametrize(
    "status,reason,recoverable",
    [
        ("unsafe_parent", "shared_writable_parent", True),
        ("unsafe_parent", "untrusted_directory_owner", True),
        ("unsafe_parent", "missing_parent", True),
        ("link_or_non_regular", "non_directory_parent", True),
        ("link_or_non_regular", "symlink_hop_limit_exceeded", True),
        ("link_or_non_regular", "OSError", True),
        ("link_or_non_regular", "NotADirectoryError", True),
        ("operation_failed", "PermissionError", True),
        ("operation_failed", "FileNotFoundError", True),
        ("operation_failed", "OSError", False),
        ("operation_failed", "invalid_absolute_path", False),
        ("unsafe_parent", "operation_failed", False),
        ("wrong_owner", "shared_writable_parent", False),
    ],
)
def test_actual_typed_parent_refusal_is_closed_and_operation_scoped(
    tmp_path, monkeypatch, operation, status, reason, recoverable
):
    path = closed_store(tmp_path)
    process = module("DB.private_sqlite_process")
    protocol = module("DB.private_sqlite_protocol")
    real_popen = subprocess.Popen

    def launch(args, **kwargs):
        script = f"""
import runpy,sys
def inject(frame,event,arg):
    if event == 'call' and frame.f_code.co_name == {operation.removeprefix("tts_")!r} and frame.f_globals.get('__name__') == 'tldw_chatbook.TTS.profile_sqlite_proof' and frame.f_back.f_code.co_name != 'initialize' and frame.f_locals['self'].initialized:
        paths = frame.f_globals['private_paths']
        original = paths._open_verified_parent
        def refused(*args,**kwargs):
            paths._open_verified_parent = original
            raise paths.PrivatePathError(paths.PrivatePathResult(args[0],paths.PrivatePathStatus({status!r}),reason={reason!r}))
        paths._open_verified_parent = refused
        sys.setprofile(None)
sys.setprofile(inject)
sys.argv = [{args[-1]!r}]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
        return real_popen([args[0], "-I", "-S", "-c", script], **kwargs)

    monkeypatch.setattr(process.subprocess, "Popen", launch)
    deadline = process.OperationDeadline(time.monotonic() + 30)
    with process.HelperAdmission().reserve(
        transient=1, retained=1, deadline=deadline
    ) as owner:
        lease = process.HelperLease.start(
            protocol.PrepareRequest(str(path), False, False, False),
            operation="tts_exact_current",
            reservation=owner,
            deadline=deadline,
        )
        assert lease.initial_response["status"] == "ok"
        response = lease.request(operation, deadline=deadline)
        if recoverable:
            assert response == {
                "version": 1,
                "operation": operation,
                "status": "tts_error",
                "reason": "operation_failed",
            }
            assert not lease._failed and lease._child.poll() is None
            assert lease.request("tts_recheck", deadline=deadline)["status"] == "ok"
        else:
            assert response == {
                "version": 1,
                "operation": operation,
                "status": "private_path_error",
                "privacy_status": status,
                "reason": reason,
            }
            assert lease._failed and lease._child.wait(timeout=2) == 0
            with pytest.raises(process.HelperUnavailableError):
                lease.request("tts_recheck", deadline=deadline)
        lease.close()
        assert lease.cleanup_state == "reaped"
