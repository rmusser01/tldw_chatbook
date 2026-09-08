"""Real helper qualification and adversarial process/file boundary evidence."""

import hashlib
import json
import os
import shutil
import struct
import subprocess
from threading import Event, Thread
import time

import pytest


def _copy_helper_package(helper_resource_root, destination):
    package_root = destination / "package"
    root = package_root / "_age"
    shutil.copytree(helper_resource_root, root)
    shutil.copy2(
        helper_resource_root.parent / "helper_manifest.json",
        package_root / "helper_manifest.json",
    )
    return root


def test_encrypted_stream_round_trip(tmp_path, helper_resource_root, monkeypatch):
    from tldw_chatbook.Backup_Recovery.crypto import transform
    from tldw_chatbook.Backup_Recovery import crypto

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    source, encrypted, restored = (tmp_path / n for n in ("in", "sealed", "out"))
    source.write_bytes(b"synthetic recovery bytes" * 10000)
    transform(
        source,
        encrypted,
        password=b"test-only passphrase",
        decrypt=False,
        cancel=Event(),
    )
    transform(
        encrypted,
        restored,
        password=b"test-only passphrase",
        decrypt=True,
        cancel=Event(),
    )
    assert restored.read_bytes() == source.read_bytes()
    assert b"synthetic recovery bytes" not in encrypted.read_bytes()


@pytest.fixture
def crypto(helper_resource_root, monkeypatch):
    from tldw_chatbook.Backup_Recovery import crypto

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    return crypto


def test_unbuilt_capability_is_unavailable(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import crypto

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: tmp_path)
    assert crypto.helper_capability() == (False, "helper_unavailable")


@pytest.mark.parametrize("password", [b"", b"x" * 4097], ids=["empty", "oversized"])
def test_invalid_password_does_not_create_output(tmp_path, crypto, password):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"synthetic")
    with pytest.raises(crypto.CryptoError, match="invalid_password"):
        crypto.transform(
            source, target, password=password, decrypt=False, cancel=Event()
        )
    assert not target.exists()


@pytest.mark.parametrize(
    "corruption",
    [
        "wrong_password",
        "truncated",
        "work_factor",
        "multi_recipient",
        "huge_header",
        "malformed",
    ],
)
def test_rejects_untrusted_ciphertext(tmp_path, crypto, corruption):
    source, sealed, target = (tmp_path / n for n in ("source", "sealed", "target"))
    source.write_bytes(b"secret-content-sentinel" * 5000)
    password = b"secret-password-sentinel"
    crypto.transform(source, sealed, password=password, decrypt=False, cancel=Event())
    data = sealed.read_bytes()
    if corruption == "wrong_password":
        password = b"wrong-secret-sentinel"
    elif corruption == "truncated":
        data = data[:-1]
    elif corruption == "work_factor":
        data = data.replace(b" 18\n", b" 19\n", 1)
    elif corruption == "multi_recipient":
        data = data.replace(
            b"--- ",
            b"-> scrypt AAAAAAAAAAAAAAAAAAAAAA 18\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n--- ",
            1,
        )
    elif corruption == "huge_header":
        data = b"age-encryption.org/v1\n" + b"a" * 65537
    else:
        data = b"not-age-secret-sentinel"
    sealed.write_bytes(data)
    with pytest.raises(crypto.CryptoError) as caught:
        crypto.transform(
            sealed, target, password=password, decrypt=True, cancel=Event()
        )
    assert "sentinel" not in str(caught.value)
    assert not target.exists()
    assert not list(tmp_path.glob(".backup-age-*"))


def test_existing_destination_survives(tmp_path, crypto):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"input")
    target.write_bytes(b"preserve")
    with pytest.raises(crypto.CryptoError):
        crypto.transform(
            source, target, password=b"test", decrypt=False, cancel=Event()
        )
    assert target.read_bytes() == b"preserve"


def test_helper_info_requires_no_password(helper_resource_root):
    binary = helper_resource_root / "backup-age"
    result = subprocess.run(
        [str(binary), "info"], input=b"", capture_output=True, timeout=5
    )
    assert result.returncode == 0
    assert len(result.stdout) < 1024 and result.stderr == b""
    assert json.loads(result.stdout)["protocol"] == 1


def test_official_age_interoperability(tmp_path, crypto, official_age):
    import pty
    import select

    password = b"interop-synthetic-password"

    def official(mode, source, target):
        terminal, slave = pty.openpty()
        child = subprocess.Popen(
            [str(official_age), mode, "-o", str(target), str(source)],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            start_new_session=True,
        )
        os.close(slave)
        pending = b""
        prompts = (
            [b"Enter passphrase:"]
            if mode == "-d"
            else [
                b"Enter passphrase (leave empty to autogenerate a secure one):",
                b"Confirm passphrase:",
            ]
        )
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and child.poll() is None:
                ready, _, _ = select.select([terminal], [], [], 0.05)
                if ready:
                    try:
                        pending += os.read(terminal, 4096)
                    except OSError:
                        break
                    if prompts and prompts[0] in pending:
                        prompts.pop(0)
                        pending = b""
                        os.write(terminal, password + b"\n")
            assert child.wait(timeout=5) == 0
            assert not prompts
        finally:
            os.close(terminal)
            if child.poll() is None:
                child.kill()
            child.wait()

    source = tmp_path / "source"
    source.write_bytes(bytes(range(256)) * 2048)
    ours, official_output, official_sealed, restored = (
        tmp_path / n for n in ("ours", "official-out", "official-sealed", "restored")
    )
    crypto.transform(source, ours, password=password, decrypt=False, cancel=Event())
    official("-d", ours, official_output)
    assert official_output.read_bytes() == source.read_bytes()
    official("-p", source, official_sealed)
    crypto.transform(
        official_sealed, restored, password=password, decrypt=True, cancel=Event()
    )
    assert restored.read_bytes() == source.read_bytes()


@pytest.mark.parametrize(
    "defect",
    [
        "missing_binary",
        "digest",
        "platform",
        "protocol",
        "oversized_manifest",
        "unknown_field",
        "binary_symlink",
        "boolean_protocol",
    ],
)
def test_unqualified_resource_is_unavailable(
    tmp_path, helper_resource_root, monkeypatch, defect
):
    from tldw_chatbook.Backup_Recovery import crypto

    root = _copy_helper_package(helper_resource_root, tmp_path)
    manifest_path = root.parent / "helper_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    qualified = next(
        entry for entry in manifest["helpers"] if entry["status"] == "qualified"
    )
    if defect == "missing_binary":
        (root / "backup-age").unlink()
    elif defect == "digest":
        qualified["sha256"] = "0" * 64
    elif defect == "platform":
        qualified["arch"] = "unsupported"
    elif defect == "protocol":
        qualified["protocol"] = 2
    elif defect == "boolean_protocol":
        qualified["protocol"] = True
    elif defect == "unknown_field":
        qualified["path"] = "/untrusted"
    elif defect == "binary_symlink":
        (root / "backup-age").unlink()
        (root / "backup-age").symlink_to(helper_resource_root / "backup-age")
    manifest_path.write_text(
        json.dumps(manifest) + (" " * 17000 if defect == "oversized_manifest" else "")
    )
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: root)
    expected = (
        "helper_integrity_mismatch" if defect == "digest" else "helper_unavailable"
    )
    assert crypto.helper_capability() == (False, expected)


def test_byte_budgets_count_streamed_input_and_output(tmp_path, crypto, monkeypatch):
    monkeypatch.setattr(crypto, "_MAX_CONTAINER", 1024)
    source, target = tmp_path / "source", tmp_path / "target"
    for size in (1000, 2000):
        source.write_bytes(b"s" * size)
        with pytest.raises(crypto.CryptoError):
            crypto.transform(
                source, target, password=b"test", decrypt=False, cancel=Event()
            )
        assert not target.exists()


def test_pipe_backpressure_and_memory_are_bounded(tmp_path, crypto, monkeypatch):
    import psutil

    source, target, restored = (tmp_path / n for n in ("source", "target", "restored"))
    with source.open("wb") as stream:
        stream.truncate(128 * 1024**2)
    original = subprocess.Popen
    children = []
    monkeypatch.setenv("BACKUP_SECRET_SENTINEL", "ambient-secret-must-not-reach-child")

    def observe(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append((child, args, kwargs))
        return child

    monkeypatch.setattr(crypto.subprocess, "Popen", observe)
    outcome = []
    cancel = Event()

    def execute():
        try:
            crypto.transform(
                source,
                target,
                password=b"pipe-password-sentinel",
                decrypt=False,
                cancel=cancel,
            )
        except Exception as exc:
            outcome.append(exc)

    baseline = psutil.Process().memory_info().rss
    peak_parent, peak_child = baseline, 0
    thread = Thread(target=execute)
    thread.start()
    try:
        deadline = time.monotonic() + 30
        while thread.is_alive() and time.monotonic() < deadline:
            peak_parent = max(peak_parent, psutil.Process().memory_info().rss)
            for child, _, _ in children:
                try:
                    peak_child = max(
                        peak_child, psutil.Process(child.pid).memory_info().rss
                    )
                except psutil.NoSuchProcess:
                    pass
            thread.join(0.01)
        assert not thread.is_alive(), "pipe backpressure deadlocked"
    finally:
        _cleanup_stream_worker(thread, cancel, children)
    assert outcome == []
    assert peak_parent - baseline < 32 * 1024**2
    assert 200 * 1024**2 < peak_child < 384 * 1024**2
    for child, args, kwargs in children:
        assert child.returncode == 0
        assert args[0][1] in ("info", "encrypt") and len(args[0]) == 2
        assert "sentinel" not in repr(args) + repr(kwargs)
    crypto.transform(
        target,
        restored,
        password=b"pipe-password-sentinel",
        decrypt=True,
        cancel=Event(),
    )
    with source.open("rb") as a, restored.open("rb") as b:
        assert (
            hashlib.file_digest(a, "sha256").digest()
            == hashlib.file_digest(b, "sha256").digest()
        )
    assert target.stat().st_mode & 0o777 == 0o600


def test_cancel_and_crash_reap_real_child_and_preserve_other_files(
    tmp_path, crypto, monkeypatch
):
    source = tmp_path / "source"
    with source.open("wb") as stream:
        stream.truncate(1024**3)
    preserve = tmp_path / "unrelated"
    preserve.write_bytes(b"keep")
    original = subprocess.Popen
    for action in ("cancel", "crash"):
        started = Event()
        children = []
        cancel = Event()
        outcome = []

        def observe(*args, **kwargs):
            child = original(*args, **kwargs)
            if args[0][1] == "encrypt":
                children.append(child)
                started.set()
            return child

        monkeypatch.setattr(crypto.subprocess, "Popen", observe)

        def execute():
            try:
                crypto.transform(
                    source,
                    tmp_path / action,
                    password=b"cancel-secret-sentinel",
                    decrypt=False,
                    cancel=cancel,
                )
            except crypto.CryptoError as exc:
                outcome.append(str(exc))

        thread = Thread(target=execute)
        thread.start()
        assert started.wait(5)
        if action == "cancel":
            cancel.set()
        else:
            children[0].kill()
        thread.join(5)
        assert not thread.is_alive()
        assert outcome == (
            ["cancelled"] if action == "cancel" else ["transform_failed"]
        )
        assert all(child.returncode is not None for child in children)
        assert not (tmp_path / action).exists()
        assert not list(tmp_path.glob(".backup-age-*"))
        assert preserve.read_bytes() == b"keep"
    assert not any(
        t.name == "backup-age-pipe" for t in __import__("threading").enumerate()
    )


def test_queued_job_cancels_without_second_kdf(tmp_path, crypto, monkeypatch):
    source = tmp_path / "source"
    with source.open("wb") as stream:
        stream.truncate(1024**3)
    started, release = Event(), Event()
    original = subprocess.Popen
    transforms = []

    def observe(*args, **kwargs):
        child = original(*args, **kwargs)
        if args[0][1] == "encrypt":
            transforms.append(child)
            started.set()
        return child

    monkeypatch.setattr(crypto.subprocess, "Popen", observe)
    results = []

    def first():
        try:
            crypto.transform(
                source,
                tmp_path / "first",
                password=b"test",
                decrypt=False,
                cancel=release,
            )
        except crypto.CryptoError as exc:
            results.append(str(exc))

    thread = Thread(target=first)
    thread.start()
    try:
        assert started.wait(5)
        cancelled = Event()
        cancelled.set()
        with pytest.raises(crypto.CryptoError, match="cancelled"):
            crypto.transform(
                source,
                tmp_path / "second",
                password=b"test",
                decrypt=False,
                cancel=cancelled,
            )
        assert len(transforms) == 1
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and results == ["cancelled"]


def test_helper_protocol_errors_are_fixed_and_secret_free(helper_resource_root):
    binary = helper_resource_root / "backup-age"
    for data in [struct.pack(">I", n) for n in (0, 4097, 0xFFFFFFFF)] + [b"\0\0\0\4x"]:
        result = subprocess.run(
            [str(binary), "encrypt"], input=data, capture_output=True, timeout=5
        )
        assert (
            result.returncode != 0
            and result.stdout == b""
            and result.stderr == b"protocol_error\n"
        )
    result = subprocess.run(
        [str(binary), "decrypt"],
        input=struct.pack(">I", 15) + b"secret-sentinel" + b"malformed-secret-sentinel",
        capture_output=True,
        timeout=5,
    )
    assert (
        result.returncode != 0
        and result.stdout == b""
        and result.stderr == b"invalid_header\n"
    )


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory", "nul"])
def test_invalid_sources_are_rejected_without_output(tmp_path, crypto, kind):
    source, target = tmp_path / "source", tmp_path / "target"
    if kind == "symlink":
        real = tmp_path / "real"
        real.write_bytes(b"synthetic")
        source.symlink_to(real)
    elif kind == "fifo":
        os.mkfifo(source)
    elif kind == "directory":
        source.mkdir()
    else:
        source = type(source)(str(source) + "\x00")
    with pytest.raises(crypto.CryptoError):
        crypto.transform(
            source, target, password=b"test", decrypt=False, cancel=Event()
        )
    assert not target.exists()


def test_cleanup_error_does_not_strand_serial_job_lock(tmp_path, crypto, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"input")
    destination = tmp_path / "private"
    destination.mkdir(mode=0o700)
    original = subprocess.Popen

    def observe(*args, **kwargs):
        child = original(*args, **kwargs)
        if args[0][1] == "encrypt":
            wait = child.wait

            def deny_cleanup(*args, **kwargs):
                result = wait(*args, **kwargs)
                destination.chmod(0o500)
                return result

            child.wait = deny_cleanup
        return child

    monkeypatch.setattr(crypto.subprocess, "Popen", observe)
    try:
        with pytest.raises(crypto.CryptoError, match="cleanup_failed"):
            crypto.transform(
                source,
                destination / "target",
                password=b"test",
                decrypt=False,
                cancel=Event(),
            )
        assert crypto._JOBS.acquire(blocking=False)
        crypto._JOBS.release()
        assert not (destination / "target").exists()
    finally:
        destination.chmod(0o700)


@pytest.mark.parametrize("mode", ["info_flood", "stderr_flood"])
def test_misbehaving_child_output_is_bounded(
    tmp_path, helper_resource_root, monkeypatch, mode
):
    import sys
    from tldw_chatbook.Backup_Recovery import crypto

    root = _copy_helper_package(helper_resource_root, tmp_path)
    manifest_path = root.parent / "helper_manifest.json"
    metadata = json.loads(manifest_path.read_text())
    qualified = next(
        entry for entry in metadata["helpers"] if entry["status"] == "qualified"
    )
    info = {
        key: value
        for key, value in qualified.items()
        if key not in {"python_versions", "resource", "sha256", "status"}
    }
    binary = root / "backup-age"
    binary.write_text(
        f"#!{sys.executable}\nimport os, sys\n"
        + (
            f"if sys.argv[1] == 'info':\n print({json.dumps(info)!r})\n sys.exit(0)\n"
            if mode == "stderr_flood"
            else ""
        )
        + f"while True: os.write({2 if mode == 'stderr_flood' else 1}, b'secret-sentinel' * 1024)\n"
    )
    binary.chmod(0o700)
    qualified["sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(metadata))
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: root)
    start = time.monotonic()
    if mode == "info_flood":
        assert crypto.helper_capability() == (False, "helper_unavailable")
    else:
        source = tmp_path / "source"
        source.write_bytes(b"s" * 1024**2)
        with pytest.raises(crypto.CryptoError, match="transform_failed") as caught:
            crypto.transform(
                source,
                tmp_path / "target",
                password=b"test",
                decrypt=False,
                cancel=Event(),
            )
        assert "sentinel" not in str(caught.value)
        assert not (tmp_path / "target").exists()
    assert time.monotonic() - start < 5
    assert not any(
        t.name == "backup-age-pipe" for t in __import__("threading").enumerate()
    )


def test_empty_payload_and_maximum_binary_password(tmp_path, crypto):
    source, sealed, restored = (tmp_path / n for n in ("empty", "sealed", "restored"))
    source.write_bytes(b"")
    password = b"\xff\x00" * 2048
    crypto.transform(source, sealed, password=password, decrypt=False, cancel=Event())
    crypto.transform(sealed, restored, password=password, decrypt=True, cancel=Event())
    assert restored.read_bytes() == b""
    assert len(sealed.read_bytes()) > 16


@pytest.mark.parametrize("name", ["helper_manifest.json", "backup-age"])
def test_nonregular_resource_does_not_block_capability(
    tmp_path, helper_resource_root, name
):
    import sys

    root = _copy_helper_package(helper_resource_root, tmp_path)
    path = root.parent / name if name == "helper_manifest.json" else root / name
    path.unlink()
    os.mkfifo(path)
    # Inherit Tests/conftest.py's isolated config/home; bound the probe itself.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; from tldw_chatbook.Backup_Recovery import crypto; crypto._package_resource_root=lambda: Path(sys.argv[1]); print(crypto.helper_capability())",
            str(root),
        ],
        capture_output=True,
        timeout=2,
    )
    assert probe.returncode == 0
    assert probe.stdout.strip() == b"(False, 'helper_unavailable')"


def _cleanup_stream_worker(thread, cancel, children):
    """Bound teardown even when a pipe regression ignores cancellation."""
    cancel.set()
    thread.join(0.2)
    try:
        for child, _, _ in children:
            if child.poll() is None:
                child.terminate()
            try:
                child.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
    finally:
        thread.join(5)
    assert not thread.is_alive(), "stream worker survived child cleanup"


def test_backpressure_harness_reaps_stalled_work():
    import sys

    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('ready', flush=True); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
    )
    cancel = Event()
    thread = Thread(target=lambda: child.stdout.read())
    try:
        assert child.stdout.readline() == b"ready\n"
        thread.start()
        assert thread.is_alive() and child.poll() is None
        _cleanup_stream_worker(thread, cancel, [(child, (), {})])
        assert not thread.is_alive()
        assert child.returncode is not None
        assert cancel.is_set()
    finally:
        # Protect the regression test itself when the cleanup helper is broken.
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        if thread.ident is not None:
            thread.join(5)
        child.stdout.close()
