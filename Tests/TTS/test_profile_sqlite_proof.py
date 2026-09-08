"""Real closed-store proof, immutable validation, and original-cohort ownership."""

import importlib
import os
import sqlite3
import time

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
