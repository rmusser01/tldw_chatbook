"""Closed private SQLite frames and the real isolated file helper."""

import importlib
import io
import json
import os
import select
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest


def protocol():
    return importlib.import_module("tldw_chatbook.DB.private_sqlite_protocol")


def framed(body):
    return struct.pack("!I", len(body)) + body


def test_frame_round_trip_preserves_closed_request():
    codec = protocol()
    value = {"version": 1, "operation": "close"}
    assert codec.decode_frame(codec.encode_frame(value)) == value


@pytest.mark.parametrize(
    "body",
    [
        b'{"version":1,"version":2,"operation":"close"}',
        b'{"version":1,"operation":"close","unexpected":true}',
        b'{"version":1,"operation":"fetch"}',
        b'{"version":true,"operation":"close"}',
        b'{"version":2,"operation":"close"}',
        b'{"version":NaN,"operation":"close"}',
        b'{"version":Infinity,"operation":"close"}',
        b'{"version":-Infinity,"operation":"close"}',
        b'{"version":1,"operation":"close","x":{"a":1,"a":2}}',
        b"\xff",
        b"[]",
        b"{}",
        b"[" * 1000 + b"]" * 1000,
    ],
)
def test_closed_frame_refuses_ambiguous_or_unknown_requests(body):
    codec = protocol()
    with pytest.raises(codec.ProtocolError, match="^private_sqlite_protocol_error$"):
        codec.decode_frame(framed(body))


@pytest.mark.parametrize(
    "frame",
    [
        b"",
        b"\x00",
        struct.pack("!I", 65537),
        struct.pack("!I", 1),
        framed(b"{}") + b"x",
        framed(b"x" * 65537),
    ],
)
def test_invalid_frame_lengths_are_rejected(frame):
    codec = protocol()
    with pytest.raises(codec.ProtocolError):
        codec.decode_frame(frame)


def test_reader_refuses_oversized_header_before_reading_body():
    codec = protocol()

    class HeaderOnly:
        def read(self, count):
            assert count == 4
            return struct.pack("!I", 65537)

    with pytest.raises(codec.ProtocolError):
        codec.read_frame(HeaderOnly())


def test_reader_handles_short_reads_and_clean_eof():
    codec = protocol()
    value = {"version": 1, "operation": "close"}

    class ShortReads(io.BytesIO):
        def read(self, count):
            return super().read(min(count, 1))

    assert codec.read_frame(ShortReads(codec.encode_frame(value))) == value
    assert codec.read_frame(io.BytesIO()) is None
    with pytest.raises(codec.ProtocolError):
        codec.read_frame(io.BytesIO(b"\x00"))


def test_encode_rejects_oversized_path_and_nonfinite_values():
    codec = protocol()
    for value in [
        request("x" * 65536),
        {"version": float("nan"), "operation": "close"},
    ]:
        with pytest.raises(codec.ProtocolError):
            codec.encode_frame(value)


def request(path, operation="prepare", **flags):
    return {
        "version": 1,
        "operation": operation,
        "path": str(path),
        "writable": False,
        "create_if_missing": False,
        "preserve_source_mode": False,
        **flags,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"writable": 1},
        {"create_if_missing": 0},
        {"preserve_source_mode": "yes"},
        {"path": 1},
        {"path": "a\x00b"},
        {"suffix": "-other"},
        {"owner": "db.base"},
    ],
)
def test_prepare_request_fields_are_closed_and_typed(change):
    codec = protocol()
    with pytest.raises(codec.ProtocolError):
        codec.decode_frame(
            framed(json.dumps({**request("/private/example"), **change}).encode())
        )


def test_recheck_cannot_supply_new_path():
    codec = protocol()
    with pytest.raises(codec.ProtocolError):
        codec.encode_frame(
            {"version": 1, "operation": "recheck_source", "path": "/other"}
        )


def test_identity_projection_is_exact_private_and_bool_is_not_integer(tmp_path):
    codec = protocol()
    target = tmp_path / "identity"
    target.touch()
    identity = codec.FileIdentity.from_stat(target.stat())
    assert identity.ino == target.stat().st_ino
    assert identity.mtime_ns == target.stat().st_mtime_ns
    assert identity.same_inode(codec.FileIdentity.from_stat(target.stat()))
    assert "ino=" not in repr(identity)
    response = {
        "version": 1,
        "operation": "prepare",
        "status": "ok",
        "result": {
            "main_identity": identity.to_payload(),
            "artifacts": ["already_private", "absent", "absent", "absent"],
        },
    }
    assert codec.decode_frame(codec.encode_frame(response)) == response
    response["result"]["main_identity"]["ino"] = True
    with pytest.raises(codec.ProtocolError):
        codec.encode_frame(response)


@pytest.mark.skipif(os.name != "posix", reason="POSIX artifact policy")
@pytest.mark.parametrize("preserve", [False, True])
def test_leaf_batch_hardens_fixed_cohort_or_preserves_source_mode(tmp_path, preserve):
    files = importlib.import_module("tldw_chatbook.DB.private_sqlite_files")
    codec = protocol()
    target = tmp_path / "db"
    for suffix in ("", "-wal", "-shm", "-journal"):
        artifact = Path(f"{target}{suffix}")
        artifact.write_bytes(b"fixture")
        artifact.chmod(0o644)
    result = files.prepare_batch(
        codec.PrepareRequest(str(target), False, False, preserve)
    )
    assert result.main_identity.same_inode(codec.FileIdentity.from_stat(target.stat()))
    assert all(
        Path(f"{target}{suffix}").stat().st_mode & 0o777
        == (0o644 if preserve else 0o600)
        for suffix in ("", "-wal", "-shm", "-journal")
    )


def child_command():
    entry = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook/DB/private_sqlite_helper_entry.py"
    )
    return [sys.executable, "-I", "-S", str(entry)]


def child_environment():
    return {**os.environ, "_TLDW_PRIVATE_SQLITE_PARENT_PID": str(os.getpid())}


def run_child(payload, cwd):
    codec = protocol()
    result = subprocess.run(
        child_command(),
        env=child_environment(),
        input=codec.encode_frame(payload),
        capture_output=True,
        cwd=cwd,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0
    assert result.stderr == b""
    return codec.decode_frame(result.stdout)


@pytest.mark.parametrize(
    "metadata", [None, "", "0", "-1", "01", "unknown-parent", "9" * 100]
)
def test_fixed_entry_refuses_missing_or_malformed_original_parent_metadata(
    tmp_path, metadata
):
    target = tmp_path / "must-not-be-created"
    environment = os.environ.copy()
    environment.pop("_TLDW_PRIVATE_SQLITE_PARENT_PID", None)
    if metadata is not None:
        environment["_TLDW_PRIVATE_SQLITE_PARENT_PID"] = metadata
    result = subprocess.run(
        child_command(),
        env=environment,
        input=protocol().encode_frame(
            request(target, writable=True, create_if_missing=True)
        ),
        capture_output=True,
        timeout=3,
        check=False,
    )
    assert result.returncode == 1
    assert result.stdout == result.stderr == b""
    assert not target.exists()


def test_real_child_ignores_hostile_cwd_and_environment_import_paths(
    tmp_path, monkeypatch
):
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    package = hostile / "tldw_chatbook"
    package.mkdir()
    marker = tmp_path / "imported-hostile-code"
    for path in (
        package / "__init__.py",
        hostile / "sitecustomize.py",
        hostile / "json.py",
    ):
        path.write_text(
            f"open({str(marker)!r}, 'w').write('bad')\nraise RuntimeError('hostile')"
        )
    monkeypatch.setenv("PYTHONPATH", str(hostile))
    database = tmp_path / "db"
    result = run_child(
        request(database, writable=True, create_if_missing=True), hostile
    )
    assert result["status"] == "ok"
    assert database.stat().st_mode & 0o777 == 0o600
    assert not marker.exists()


def test_real_child_missing_path_returns_only_fixed_failure(tmp_path):
    result = run_child(request(tmp_path / "private-sentinel"), tmp_path)
    assert result["status"] == "private_path_error"
    assert result["reason"] == "missing_sqlite_artifact"
    assert "private-sentinel" not in json.dumps(result)


@pytest.mark.skipif(os.name != "posix", reason="POSIX filename length limit")
@pytest.mark.parametrize("artifact", ["main", "sidecar"])
def test_real_child_lstat_failure_remains_a_private_path_error(tmp_path, artifact):
    name_limit = os.pathconf(tmp_path, "PC_NAME_MAX")
    target = tmp_path / ("x" * (name_limit + (artifact == "main")))
    if artifact == "sidecar":
        # Main fits the filesystem limit; its fixed -wal sibling does not.
        target.touch(mode=0o600)
    result = run_child(request(target), tmp_path)
    assert result == {
        "version": 1,
        "operation": "prepare",
        "status": "private_path_error",
        "privacy_status": "operation_failed",
        "reason": "OSError",
    }


def test_batch_identity_is_from_validated_descriptor_not_later_path(
    tmp_path, monkeypatch
):
    files = importlib.import_module("tldw_chatbook.DB.private_sqlite_files")
    codec = protocol()
    target = tmp_path / "db"
    target.touch(mode=0o600)
    original = codec.FileIdentity.from_stat(target.stat())
    replacement = tmp_path / "replacement"
    replacement.touch(mode=0o600)
    real_prepare = files._prepare_artifact

    def replace_after_prepare(selected, **kwargs):
        result = real_prepare(selected, **kwargs)
        if selected == target:
            replacement.replace(target)
        return result

    monkeypatch.setattr(files, "_prepare_artifact", replace_after_prepare)
    result = files.prepare_batch(codec.PrepareRequest(str(target), False, False, False))
    assert result.main_identity.same_inode(original)


@pytest.mark.parametrize("unsafe", ["symlink", "hardlink", "directory", "owner"])
def test_leaf_refuses_unsafe_main_and_does_not_change_mode(
    tmp_path, monkeypatch, unsafe
):
    files = importlib.import_module("tldw_chatbook.DB.private_sqlite_files")
    codec = protocol()
    from tldw_chatbook.Utils.private_paths import PrivatePathError

    target = tmp_path / "db"
    other = tmp_path / "other"
    other.write_bytes(b"untouched")
    other.chmod(0o644)
    if unsafe == "symlink":
        target.symlink_to(other)
    elif unsafe == "hardlink":
        os.link(other, target)
    elif unsafe == "directory":
        target.mkdir()
    else:
        target.write_bytes(b"owner")
        real_classify = files.private_paths._classify_private_file_stat

        def wrong_owner(value, *, expected_uid):
            return real_classify(value, expected_uid=expected_uid + 1)

        monkeypatch.setattr(
            files.private_paths, "_classify_private_file_stat", wrong_owner
        )
    with pytest.raises(PrivatePathError):
        files.prepare_batch(codec.PrepareRequest(str(target), True, False, False))
    assert other.read_bytes() == b"untouched"
    assert other.stat().st_mode & 0o777 == 0o644


def test_leaf_optional_churn_exhausts_exactly_four_generations(tmp_path, monkeypatch):
    files = importlib.import_module("tldw_chatbook.DB.private_sqlite_files")
    codec = protocol()
    from tldw_chatbook.Utils.private_paths import PrivatePathError

    target = tmp_path / "db"
    target.touch(mode=0o600)
    sidecar = tmp_path / "db-wal"
    sidecar.touch(mode=0o600)
    real_open = files._open_artifact_fd
    generations = 0

    def replace_each_open(parent_fd, leaf, **kwargs):
        nonlocal generations
        if leaf == sidecar.name:
            replacement = tmp_path / "replacement"
            replacement.touch(mode=0o600)
            replacement.replace(sidecar)
            generations += 1
        return real_open(parent_fd, leaf, **kwargs)

    monkeypatch.setattr(files, "_open_artifact_fd", replace_each_open)
    with pytest.raises(PrivatePathError) as caught:
        files.prepare_batch(codec.PrepareRequest(str(target), False, False, False))
    assert caught.value.result.reason == "optional_sqlite_generation_churn"
    assert generations == 4


def exchange(child, payload):
    codec = protocol()
    child.stdin.write(codec.encode_frame(payload))
    child.stdin.flush()
    return codec.read_frame(child.stdout)


@pytest.mark.parametrize("replace", ["main", "parent", "mode", "none"])
def test_real_source_pin_rechecks_bound_authority_and_closes(tmp_path, replace):
    directory = tmp_path / "parent"
    directory.mkdir()
    target = directory / "db"
    target.write_bytes(b"initial")
    target.chmod(0o600)
    with subprocess.Popen(
        child_command(),
        env=child_environment(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=tmp_path,
    ) as child:
        try:
            first = exchange(child, request(target, operation="pin_source"))
            assert first["status"] == "ok"
            if replace == "main":
                replacement = directory / "replacement"
                replacement.touch(mode=0o600)
                replacement.replace(target)
            elif replace == "parent":
                directory.rename(tmp_path / "old-parent")
                directory.mkdir()
                target.touch(mode=0o600)
            elif replace == "mode":
                target.chmod(0o644)
            else:
                target.write_bytes(b"changed size is allowed for a live backup")
            result = exchange(child, {"version": 1, "operation": "recheck_source"})
            assert result["status"] == (
                "ok" if replace == "none" else "private_path_error"
            )
            if replace == "none":
                assert (
                    result["result"]["main_identity"]["size"]
                    != first["result"]["main_identity"]["size"]
                )
                assert exchange(child, {"version": 1, "operation": "close"}) == {
                    "version": 1,
                    "operation": "close",
                    "status": "ok",
                }
            assert child.wait(timeout=5) == 0
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)


def test_real_helper_rejects_oversized_frame_without_waiting_for_body(tmp_path):
    codec = protocol()
    with subprocess.Popen(
        child_command(),
        env=child_environment(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=tmp_path,
    ) as child:
        try:
            child.stdin.write(struct.pack("!I", 65537))
            child.stdin.flush()
            # Keep stdin open: refusal must precede both a body and EOF.
            assert select.select([child.stdout], [], [], 2)[0]
            assert codec.read_frame(child.stdout)["status"] == "protocol_error"
            assert child.wait(timeout=5) == 0
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)


def test_isolated_entry_skips_all_package_initializers(tmp_path):
    """Run the actual code from a package tree whose initializers must not run."""
    codec = protocol()
    source_package = Path(child_command()[-1]).parent.parent
    installed = tmp_path / "installed" / "tldw_chatbook"
    for directory in (installed, installed / "DB", installed / "Utils"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text(
            "raise RuntimeError('package startup forbidden')"
        )
    for name in (
        "private_sqlite_protocol",
        "private_sqlite_files",
        "private_sqlite_helper",
        "private_sqlite_helper_entry",
    ):
        shutil.copyfile(
            source_package / "DB" / f"{name}.py", installed / "DB" / f"{name}.py"
        )
    shutil.copyfile(
        source_package / "Utils/private_paths.py", installed / "Utils/private_paths.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(installed / "DB/private_sqlite_helper_entry.py"),
        ],
        env=child_environment(),
        input=codec.encode_frame(
            request(tmp_path / "db", writable=True, create_if_missing=True)
        ),
        capture_output=True,
        cwd=tmp_path,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0
    assert result.stderr == b""
    assert codec.decode_frame(result.stdout)["status"] == "ok"


@pytest.mark.parametrize("operation", ["prepare", "pin_source"])
def test_real_child_refuses_retargeting_after_initialization(tmp_path, operation):
    target = tmp_path / "db"
    target.touch(mode=0o600)
    other = tmp_path / "other"
    with subprocess.Popen(
        child_command(),
        env=child_environment(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=tmp_path,
    ) as child:
        try:
            assert (
                exchange(child, request(target, operation=operation))["status"] == "ok"
            )
            assert (
                exchange(child, request(other, writable=True, create_if_missing=True))[
                    "status"
                ]
                == "protocol_error"
            )
            assert child.wait(timeout=5) == 0
            assert not other.exists()
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)


def test_real_pinned_child_exits_on_eof(tmp_path):
    target = tmp_path / "db"
    target.touch(mode=0o600)
    with subprocess.Popen(
        child_command(),
        env=child_environment(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=tmp_path,
    ) as child:
        try:
            assert (
                exchange(child, request(target, operation="pin_source"))["status"]
                == "ok"
            )
            child.stdin.close()
            assert child.wait(timeout=5) == 0
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)


@pytest.mark.parametrize(
    "payload",
    [
        {"version": 1, "operation": "close", "status": "ok", "result": {}},
        {
            "version": 1,
            "operation": "close",
            "status": "helper_unavailable",
            "reason": "private text",
        },
        {"version": 1, "operation": "prepare", "status": "ok", "result": {}},
        {
            "version": 1,
            "operation": "prepare",
            "status": "private_path_error",
            "privacy_status": "operation_failed",
            "reason": "private text",
        },
    ],
)
def test_response_fields_and_reasons_are_closed(payload):
    codec = protocol()
    with pytest.raises(codec.ProtocolError):
        codec.decode_frame(framed(json.dumps(payload).encode()))


def test_real_child_wrong_mode_sidecars_are_private(tmp_path):
    target = tmp_path / "db"
    for suffix in ("", "-wal", "-shm", "-journal"):
        artifact = Path(f"{target}{suffix}")
        artifact.touch()
        artifact.chmod(0o644)
    result = run_child(request(target, writable=True), tmp_path)
    assert result["result"]["artifacts"] == ["hardened_private"] * 4
    assert all(
        Path(f"{target}{suffix}").stat().st_mode & 0o777 == 0o600
        for suffix in ("", "-wal", "-shm", "-journal")
    )


def test_leaf_missing_main_requires_creation_and_optional_absence_is_allowed(tmp_path):
    files = importlib.import_module("tldw_chatbook.DB.private_sqlite_files")
    codec = protocol()
    from tldw_chatbook.Utils.private_paths import PrivatePathError

    target = tmp_path / "db"
    with pytest.raises(PrivatePathError) as caught:
        files.prepare_batch(codec.PrepareRequest(str(target), False, False, False))
    assert caught.value.result.reason == "missing_sqlite_artifact"
    assert not target.exists()
    created = files.prepare_batch(codec.PrepareRequest(str(target), True, True, False))
    assert created.artifacts == ("created_private", "absent", "absent", "absent")
    already = files.prepare_batch(codec.PrepareRequest(str(target), True, False, False))
    assert already.artifacts == ("already_private", "absent", "absent", "absent")


def test_leaf_parent_replaced_by_symlink_is_refused(tmp_path, monkeypatch):
    files = importlib.import_module("tldw_chatbook.DB.private_sqlite_files")
    codec = protocol()
    from tldw_chatbook.Utils.private_paths import PrivatePathError

    directory = tmp_path / "parent"
    directory.mkdir()
    target = directory / "db"
    target.touch(mode=0o600)
    real_prepare = files._prepare_artifact

    def replace_parent(selected, **kwargs):
        result = real_prepare(selected, **kwargs)
        if selected == target:
            directory.rename(tmp_path / "old-parent")
            directory.symlink_to(tmp_path / "old-parent", target_is_directory=True)
        return result

    monkeypatch.setattr(files, "_prepare_artifact", replace_parent)
    with pytest.raises(PrivatePathError):
        files.prepare_batch(codec.PrepareRequest(str(target), False, False, False))


@pytest.mark.parametrize("change", ["missing", "symlink"])
@pytest.mark.parametrize("dispatch", [False, True], ids=["constructor", "dispatcher"])
def test_source_pin_initialization_path_failure_is_classified(
    tmp_path, monkeypatch, change, dispatch
):
    codec = protocol()
    files = importlib.import_module("tldw_chatbook.DB.private_sqlite_files")
    helper = importlib.import_module("tldw_chatbook.DB.private_sqlite_helper")
    from tldw_chatbook.Utils.private_paths import PrivatePathError

    target = tmp_path / "db"
    target.touch(mode=0o600)
    other = tmp_path / "other"
    other.write_bytes(b"must remain unchanged")
    other.chmod(0o644)
    real_prepare = files.prepare_batch
    real_open = files._open_artifact_fd
    pinned_parent_fds = []

    def observe_pin_open(parent_fd, *args, **kwargs):
        pinned_parent_fds.append(parent_fd)
        return real_open(parent_fd, *args, **kwargs)

    def substitute_after_preparation(selected):
        result = real_prepare(selected)
        monkeypatch.setattr(files, "_open_artifact_fd", observe_pin_open)
        target.unlink()
        if change == "symlink":
            target.symlink_to(other)
        return result

    expected_status = (
        "operation_failed" if change == "missing" else "link_or_non_regular"
    )
    expected_reason = "FileNotFoundError" if change == "missing" else "OSError"
    if dispatch:
        # Exercise the actual fixed dispatcher with private in-memory pipes;
        # only the deterministic prepare→pin mutation is injected.
        input_pipe = io.BytesIO(codec.encode_frame(request(target, "pin_source")))
        output_pipe = io.BytesIO()

        class PrivatePipe:
            deadline = None
            read = input_pipe.read
            write = output_pipe.write

            def __init__(self, parent_pid):
                pass

        monkeypatch.setattr(helper, "_PrivatePipe", PrivatePipe)
        monkeypatch.setattr(files, "prepare_batch", substitute_after_preparation)
        assert helper.run(os.getppid()) == 0
        assert codec.decode_frame(output_pipe.getvalue()) == {
            "version": 1,
            "operation": "pin_source",
            "status": "private_path_error",
            "privacy_status": expected_status,
            "reason": expected_reason,
        }
    else:
        selected = codec.PrepareRequest(str(target), False, False, False)
        result = substitute_after_preparation(selected)
        with pytest.raises(PrivatePathError) as caught:
            helper.SourcePin(selected, result)
        assert caught.value.result.status.value == expected_status
        assert caught.value.result.reason == expected_reason
    assert len(pinned_parent_fds) == 1
    with pytest.raises(OSError):
        os.fstat(pinned_parent_fds[0])
    assert other.read_bytes() == b"must remain unchanged"
    assert other.stat().st_mode & 0o777 == 0o644


def test_source_pin_preserves_existing_private_path_error(tmp_path, monkeypatch):
    codec = protocol()
    files = importlib.import_module("tldw_chatbook.DB.private_sqlite_files")
    helper = importlib.import_module("tldw_chatbook.DB.private_sqlite_helper")
    from tldw_chatbook.Utils.private_paths import PrivatePathError, PrivatePathStatus

    target = tmp_path / "db"
    target.touch(mode=0o600)
    selected = codec.PrepareRequest(str(target), False, False, False)
    prepared = files.prepare_batch(selected)
    original = files._failure(
        target, PrivatePathStatus.WRONG_OWNER, "unsafe_sqlite_artifact"
    )

    def refuse_open(*args, **kwargs):
        raise original

    monkeypatch.setattr(files, "_open_artifact_fd", refuse_open)
    with pytest.raises(PrivatePathError) as caught:
        helper.SourcePin(selected, prepared)
    assert caught.value is original
