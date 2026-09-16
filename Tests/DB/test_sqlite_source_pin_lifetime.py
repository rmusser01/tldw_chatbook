"""Real delegated pin/capture resources and exact retirement outcomes."""

import pytest

from Tests.TTS.test_profile_repository_maintenance import _run_private_child


def _probe(control, names):
    import json
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json, sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.admission import Admission, AdmissionTimeout
try:
    with Admission(Path(sys.argv[1])).maintenance(tuple(json.loads(sys.argv[2])), .15):
        print("entered")
except AdmissionTimeout:
    print("blocked")
""",
            str(control),
            json.dumps(names),
        ],
        capture_output=True,
        text=True,
        timeout=4,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _pin_child(root, mode, after):
    import os
    import sqlite3
    import time
    import types

    from tldw_chatbook.DB import private_sqlite as private
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from Tests.Backup_Recovery.test_core_owners import application_authority

    source = root / "source.sqlite"
    db = sqlite3.connect(source)
    db.execute("CREATE TABLE records(value)")
    db.execute("INSERT INTO records VALUES ('kept')")
    db.commit()
    db.close()
    source.chmod(0o600)
    native_os = private.os
    private.os = types.SimpleNamespace(**vars(native_os))
    actual_prepare = private._prepare_source_artifacts
    actual_open = private._open_artifact_fd
    actual_native_close = private.private_paths._native_close
    prepared = False
    in_preflight = False
    descriptors = []
    calls = []
    body = ValueError("original body")

    def prepare(*args, **kwargs):
        nonlocal prepared, in_preflight
        in_preflight = True
        try:
            result = actual_prepare(*args, **kwargs)
        finally:
            in_preflight = False
        prepared = True
        return result

    def open_file(parent, leaf, **kwargs):
        descriptor = actual_open(parent, leaf, **kwargs)
        if prepared:
            descriptors.append((parent, descriptor))
        return descriptor

    def close(descriptor):
        role = 0 if mode == "parent" else 1
        if descriptors and descriptor == descriptors[0][role] and mode != "success":
            calls.append(descriptor)
            if after:
                native_os.close(descriptor)
                with pytest.raises(OSError):
                    native_os.fstat(descriptor)
            else:
                assert native_os.fstat(descriptor)
            raise OSError("pin close uncertainty")
        return native_os.close(descriptor)

    def traversal_close(descriptor):
        if mode == "parent" and descriptors and descriptor == descriptors[0][0]:
            return close(descriptor)
        if mode == "traversal" and in_preflight and not calls:
            calls.append(descriptor)
            if after:
                actual_native_close(descriptor)
            raise OSError("traversal close uncertainty")
        return actual_native_close(descriptor)

    private._prepare_source_artifacts = prepare
    private._open_artifact_fd = open_file
    private.os.close = close
    private.private_paths._native_close = traversal_close
    if mode.startswith("capture"):
        patch = pytest.MonkeyPatch()
        authority = application_authority(root, source, patch)
        stage = root / "stage"
        stage.mkdir(mode=0o700)
        from tldw_chatbook.Backup_Recovery.admission import AdmissionError

        with pytest.raises(AdmissionError, match="capture_resources_not_retired"):
            with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
                with session.capture_scope((source,), stage):
                    with pytest.raises(OSError):
                        with private._pin_sqlite_source(
                            "recovery.files.tts", source, allow_memory=False
                        ):
                            pass
                    job = next(
                        j
                        for j in storage._raw_operations
                        if type(j) is private._SQLiteSourcePinJob
                    )
                    assert job.capture_lease in job.capture_lease.scope.resources
                    assert job in job.capture_lease.scope.resources
                    assert not job.leases
        assert storage._failed_capture_holds
        assert (
            _probe(authority.control_root, ("core", "bootstrap.unbound")) == "blocked"
        )
        with pytest.raises(Exception):
            job.retire()
        assert len(calls) == 1
        return

    if mode == "success":
        with private._pin_sqlite_source(
            "tts.profile_backup", source, allow_memory=False
        ) as pinned:
            actual = (pinned.parent_fd, pinned.file_fd)
            job = next(
                j
                for j in storage._raw_operations
                if type(j) is private._SQLiteSourcePinJob
            )
            hold = storage._holds[job.leases[0]._key]
            assert os.fstat(actual[1]).st_ino == source.stat().st_ino
        for descriptor in actual:
            with pytest.raises(OSError):
                os.fstat(descriptor)
        assert not any(
            type(j) is private._SQLiteSourcePinJob for j in storage._raw_operations
        )
    else:
        with pytest.raises((OSError, ValueError)):
            with private._pin_sqlite_source(
                "tts.profile_backup", source, allow_memory=False
            ):
                if mode == "body":
                    raise body
        job = next(
            j for j in storage._raw_operations if type(j) is private._SQLiteSourcePinJob
        )
        assert job.cleanup_errors and len(calls) == 1
        if mode == "body":
            assert job.body_error is body
        if mode == "traversal":
            assert job.traversal_failures and job.traversal_descriptors
        hold = storage._holds[job.leases[0]._key]
        assert _probe(hold.authority.control_root, hold.names) == "blocked"
        with pytest.raises(Exception):
            job.retire()
        assert len(calls) == 1
    pause = storage._begin_local_pause()
    try:
        assert pause.drain(time.monotonic() + 0.02) is (mode == "success")
        if mode == "success":
            assert (
                not storage._startups
            )  # This minimal helper child never admits app startup.
            assert _probe(hold.authority.control_root, hold.names) == "entered"
    finally:
        pause.resume()


@pytest.mark.parametrize("mode", ["file", "parent", "body", "traversal", "capture"])
@pytest.mark.parametrize("after", [False, True])
def test_native_pin_failure_keeps_original_resources_and_excludes_maintenance(
    tmp_path, mode, after
):
    _run_private_child(
        tmp_path,
        """
import sys
from pathlib import Path
from Tests.DB.test_sqlite_source_pin_lifetime import _pin_child
_pin_child(Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'True')
""",
        mode,
        str(after),
    )


def test_positive_native_pin_retirement_allows_later_maintenance(tmp_path):
    _run_private_child(
        tmp_path,
        """
import sys
from pathlib import Path
from Tests.DB.test_sqlite_source_pin_lifetime import _pin_child
_pin_child(Path(sys.argv[1]), "success", False)
""",
    )


def test_capture_mismatched_pin_refuses_before_preflight(tmp_path, monkeypatch):
    import sqlite3
    from Tests.Backup_Recovery.test_core_owners import application_authority
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.DB import private_sqlite as private

    source, foreign = tmp_path / "source.sqlite", tmp_path / "foreign.sqlite"
    for name in (source, foreign):
        db = sqlite3.connect(name)
        db.execute("CREATE TABLE records(value)")
        db.close()
        name.chmod(0o600)
    authority = application_authority(tmp_path, source, monkeypatch)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    effects = []
    original = private._prepare_source_artifacts

    def observe(*args, **kwargs):
        effects.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(private, "_prepare_source_artifacts", observe)
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        with session.capture_scope((source,), stage):
            with pytest.raises(RecoveryRequired, match="capture_path_outside_scope"):
                with private._pin_sqlite_source(
                    "recovery.files.tts", foreign, allow_memory=False
                ):
                    pytest.fail("foreign pin was admitted")
    assert effects == []


def test_missing_source_rejection_positively_retires_preflight(tmp_path):
    from tldw_chatbook.DB import private_sqlite as private
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    before = set(storage._raw_operations)
    with pytest.raises(OSError):
        with private._pin_sqlite_source(
            "tts.profile_backup", tmp_path / "missing.sqlite", allow_memory=False
        ):
            pytest.fail("missing source admitted")
    assert storage._raw_operations == before


def _partial_child(root, phase):
    import os
    import sqlite3
    import types
    from tldw_chatbook.DB import private_sqlite as private
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    source = root / "source.sqlite"
    db = sqlite3.connect(source)
    db.execute("CREATE TABLE records(value)")
    db.close()
    source.chmod(0o600)
    if phase == "symlink_unreturned":
        from pathlib import Path

        source = Path("/var") / source.relative_to("/private/var")
    pause = None
    actual = private._open_artifact_fd
    descriptors = []
    unknown = phase in (
        "directory_unknown",
        "unknown_then_success",
        "capture_unknown",
        "file_unknown",
        "final_file_unknown",
    )
    failure = (
        OSError("native allocation return failed")
        if unknown
        else ValueError("native allocation return failed")
    )
    final_ready = False
    if phase.startswith("final_file"):
        actual_prepare_final = private._prepare_source_artifacts

        def prepare_final(*args, **kwargs):
            nonlocal final_ready
            result = actual_prepare_final(*args, **kwargs)
            final_ready = True
            return result

        private._prepare_source_artifacts = prepare_final

    def open_file(*args, **kwargs):
        if phase == "file_unknown" or phase == "final_file_unknown" and final_ready:
            native_os = private.os

            def provider(*args, **kwargs):
                fd = native_os.open(*args, **kwargs)
                descriptors.append(fd)
                raise failure

            private.os = types.SimpleNamespace(**vars(native_os))
            private.os.open = provider
            try:
                return actual(*args, **kwargs)
            finally:
                private.os = native_os
        fd = actual(*args, **kwargs)
        if not phase.startswith("final_file") or final_ready:
            descriptors.append(fd)
        if phase == "unreturned" or phase == "final_file_wrapper" and final_ready:
            raise failure
        return fd

    private._open_artifact_fd = open_file
    if phase in (
        "directory_unreturned",
        "directory_unknown",
        "unknown_then_success",
        "capture_unknown",
        "component_unreturned",
        "symlink_unreturned",
        "pause",
    ):
        root_opens = 0
        actual_prepare = private._prepare_source_artifacts
        native_open = private.private_paths._native_open
        active = False

        def prepare(*args, **kwargs):
            nonlocal active
            active = True
            try:
                return actual_prepare(*args, **kwargs)
            finally:
                active = False

        def open_directory(*args, **kwargs):
            nonlocal pause, root_opens
            if active and phase in (
                "directory_unknown",
                "unknown_then_success",
                "capture_unknown",
            ):
                native_os = private.private_paths.os

                def provider(*args, **kwargs):
                    fd = native_os.open(*args, **kwargs)
                    descriptors.append(fd)
                    raise failure

                private.private_paths.os = types.SimpleNamespace(**vars(native_os))
                private.private_paths.os.open = provider
                try:
                    return native_open(*args, **kwargs)
                finally:
                    private.private_paths.os = native_os
            fd = native_open(*args, **kwargs)
            if active:
                if args[0] == os.sep:
                    root_opens += 1
                chosen = (
                    phase == "directory_unreturned"
                    or phase == "component_unreturned"
                    and args[0] != os.sep
                    or phase == "symlink_unreturned"
                    and args[0] == os.sep
                    and root_opens == 2
                    or phase == "pause"
                )
                if chosen:
                    descriptors.append(fd)
                    if phase == "pause":
                        pause = storage._begin_local_pause()
                    else:
                        raise failure
            return fd

        private._prepare_source_artifacts = prepare
        private.private_paths._native_open = open_directory
    if phase == "metadata":

        def check(*args, **kwargs):
            raise failure

        private._artifact_postcondition_holds = check
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    def run():
        with pytest.raises((ValueError, OSError, RecoveryRequired)):
            with private._pin_sqlite_source(
                "recovery.files.tts"
                if phase == "capture_unknown"
                else "tts.profile_backup",
                source,
                allow_memory=False,
            ):
                pytest.fail("failed source admitted")

    if phase == "capture_unknown":
        from Tests.Backup_Recovery.test_core_owners import application_authority
        from tldw_chatbook.Backup_Recovery.admission import AdmissionError

        authority = application_authority(root, source, pytest.MonkeyPatch())
        stage = root / "stage"
        stage.mkdir(mode=0o700)
        with pytest.raises(AdmissionError, match="capture_resources_not_retired"):
            with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
                with session.capture_scope((source,), stage):
                    run()
        assert storage._failed_capture_holds
        assert (
            _probe(authority.control_root, ("core", "bootstrap.unbound")) == "blocked"
        )
    else:
        run()
    jobs = [
        j for j in storage._raw_operations if type(j) is private._SQLiteSourcePinJob
    ]
    if phase in (
        "file_unknown",
        "final_file_unknown",
        "directory_unknown",
        "unknown_then_success",
        "capture_unknown",
    ):
        assert len(jobs) == 1 and jobs[0].allocation_pending
        assert os.fstat(descriptors[0])
        if phase == "unreturned":
            assert os.fstat(descriptors[0]).st_ino == source.stat().st_ino
        assert failure is jobs[0].body_error or failure in jobs[0].allocation_failures
        if phase == "unknown_then_success":
            job = jobs[0]
            job.open_traversal(os.sep, private.private_paths._DIRECTORY_OPEN_FLAGS)
            job.finish(None)
            assert not job.allocation_pending and job.allocation_failures
            assert job in storage._raw_operations
        if phase != "capture_unknown":
            hold = storage._holds[jobs[0].leases[0]._key]
            assert _probe(hold.authority.control_root, hold.names) == "blocked"
    else:
        assert not jobs
        with pytest.raises(OSError):
            os.fstat(descriptors[0])
        if pause is not None:
            import time

            assert pause.drain(time.monotonic() + 0.05)
            pause.resume()


@pytest.mark.parametrize(
    "phase",
    [
        "unreturned",
        "file_unknown",
        "final_file_unknown",
        "final_file_wrapper",
        "metadata",
        "directory_unreturned",
        "directory_unknown",
        "unknown_then_success",
        "capture_unknown",
        "component_unreturned",
        "symlink_unreturned",
        "pause",
    ],
)
def test_partial_native_allocation_distinguishes_unreturned_from_retired(
    tmp_path, phase
):
    _run_private_child(
        tmp_path,
        """
import sys
from pathlib import Path
from Tests.DB.test_sqlite_source_pin_lifetime import _partial_child
_partial_child(Path(sys.argv[1]), sys.argv[2])
""",
        phase,
    )


def _alias_child(root, mode):
    import sqlite3
    import time
    from pathlib import Path
    from threading import Event
    from Tests.Backup_Recovery.test_core_owners import application_authority
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.admission import AdmissionError
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
    )
    from tldw_chatbook.Backup_Recovery.profile_paths import database_path
    from tldw_chatbook.DB import private_sqlite as private
    from tldw_chatbook.TTS.recovery import recovery_adapters, _SCHEMA

    lexical_root = Path("/var") / root.relative_to("/private/var")
    config = {
        "paths": {"data_dir": str(lexical_root / "data")},
        "general": {"users_name": "Ada"},
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(root / "profile.toml", "p"),
    }
    source = database_path(config, "tts_profiles_db_path")
    assert str(source).startswith("/var/") and source != source.resolve()
    canonical = source.resolve()
    canonical.parent.mkdir(parents=True, mode=0o700)
    db = sqlite3.connect(canonical)
    for sql in sorted(
        _SCHEMA[0][1], key=lambda text: not text.startswith("CREATE TABLE")
    ):
        db.execute(sql)
    db.execute("PRAGMA user_version=4")
    db.close()
    canonical.chmod(0o600)
    custom = dict(config, database={"tts_profiles_db_path": str(source)})
    assert database_path(custom, "tts_profiles_db_path") == canonical
    adapter = next(a for a in recovery_adapters() if a.owner_id == "tts.profile_store")
    item = next(
        i
        for i in adapter.discover(config)
        if i.path == source and i.status == "included"
    )
    stage = root / "stage"
    stage.mkdir(mode=0o700)
    destination = stage / "tts.db"
    if mode == "ordinary":
        private.copy_private_sqlite("recovery.files.tts", source, destination)
        assert destination.is_file()
    else:
        patch = pytest.MonkeyPatch()
        authority = application_authority(root, source, patch)
        with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
            with session.capture_scope((source,), stage):
                adapter.capture(item, destination, Event())
                assert destination.is_file()
        assert not storage._failed_capture_holds
        assert (
            _probe(authority.control_root, ("core", "bootstrap.unbound")) == "entered"
        )
    jobs = [
        job
        for job in storage._raw_operations
        if type(job) is private._SQLiteSourcePinJob
    ]
    assert not jobs
    pause = storage._begin_local_pause()
    try:
        assert pause.drain(time.monotonic() + 0.02)
    finally:
        pause.resume()


@pytest.mark.parametrize("mode", ["ordinary", "capture"])
def test_trusted_alias_copy_positively_retires_native_resources(tmp_path, mode):
    _run_private_child(
        tmp_path,
        """
import sys
from pathlib import Path
from Tests.DB.test_sqlite_source_pin_lifetime import _alias_child
_alias_child(Path(sys.argv[1]), sys.argv[2])
""",
        mode,
    )


def _optional_sidecar_child(root, mode, suffix, race="disappear"):
    import os
    import sqlite3
    import time
    from tldw_chatbook.DB import private_sqlite as private
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    source = root / "source.sqlite"
    db = sqlite3.connect(source)
    db.execute("CREATE TABLE records(value)")
    db.execute("INSERT INTO records VALUES ('preserved')")
    db.commit()
    db.close()
    source.chmod(0o600)
    sidecar = source.with_name(source.name + suffix)
    sidecar.write_bytes(b"")
    sidecar.chmod(0o600)
    native_open = private._open_artifact_fd
    removed = []
    opened = []
    unknown_descriptors = []

    def disappear(parent, leaf, **kwargs):
        if leaf == sidecar.name:
            opened.append(leaf)
            if not removed:
                removed.append(leaf)
                if race == "unknown_retry":
                    import types

                    native_os = private.os

                    def provider(*args, **kwargs):
                        fd = native_os.open(*args, **kwargs)
                        unknown_descriptors.append(fd)
                        raise FileNotFoundError("unreturned native allocation")

                    private.os = types.SimpleNamespace(**vars(native_os))
                    private.os.open = provider
                    try:
                        return native_open(parent, leaf, **kwargs)
                    finally:
                        private.os = native_os
                if race == "replace":
                    fd = native_open(parent, leaf, **kwargs)
                    os.unlink(leaf, dir_fd=parent)
                    sidecar.write_bytes(b"")
                    sidecar.chmod(0o600)
                    return fd
                os.unlink(leaf, dir_fd=parent)
        return native_open(parent, leaf, **kwargs)

    private._open_artifact_fd = disappear
    stage = root / "stage"
    stage.mkdir(mode=0o700)
    destination = stage / "copy.sqlite"
    if mode == "capture":
        from Tests.Backup_Recovery.test_core_owners import application_authority
        from tldw_chatbook.Backup_Recovery.admission import AdmissionError
        from contextlib import nullcontext

        authority = application_authority(root, source, pytest.MonkeyPatch())
        expected = (
            pytest.raises(AdmissionError, match="capture_resources_not_retired")
            if race == "unknown_retry"
            else nullcontext()
        )
        with expected:
            with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
                with session.capture_scope((source,), stage):
                    private.copy_private_sqlite(
                        "recovery.files.tts", source, destination
                    )
        assert _probe(authority.control_root, ("core", "bootstrap.unbound")) == (
            "blocked" if race == "unknown_retry" else "entered"
        )
    else:
        private.copy_private_sqlite("recovery.files.tts", source, destination)
    assert removed
    if race == "disappear":
        assert not sidecar.exists()
    else:
        assert len(opened) >= 2  # The actual optional generation retry ran.
    if unknown_descriptors:
        assert os.fstat(unknown_descriptors[0])
        assert any(
            job.allocation_failures
            for job in storage._raw_operations
            if type(job) is private._SQLiteSourcePinJob
        )
    db = sqlite3.connect(destination)
    assert db.execute("SELECT value FROM records").fetchall() == [("preserved",)]
    db.close()
    pause = storage._begin_local_pause()
    try:
        assert pause.drain(time.monotonic() + 0.03) is (race != "unknown_retry")
    finally:
        pause.resume()


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
@pytest.mark.parametrize("mode", ["ordinary", "capture"])
def test_optional_sidecar_disappearing_before_open_preserves_copy_and_retirement(
    tmp_path, mode, suffix
):
    _run_private_child(
        tmp_path,
        """
import sys
from pathlib import Path
from Tests.DB.test_sqlite_source_pin_lifetime import _optional_sidecar_child
_optional_sidecar_child(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
""",
        mode,
        suffix,
    )


@pytest.mark.parametrize("race", ["replace", "unknown_retry"])
@pytest.mark.parametrize("mode", ["ordinary", "capture"])
def test_optional_generation_retry_preserves_known_and_unknown_outcomes(
    tmp_path, race, mode
):
    _run_private_child(
        tmp_path,
        """
import sys
from pathlib import Path
from Tests.DB.test_sqlite_source_pin_lifetime import _optional_sidecar_child
_optional_sidecar_child(Path(sys.argv[1]), sys.argv[2], "-journal", sys.argv[3])
""",
        mode,
        race,
    )
